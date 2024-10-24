from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import logging
import multiprocessing
import random
import sqlite3
import time
import traceback
from multiprocessing.context import BaseContext
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple, Union

from blspy import AugSchemeMPL
from web3 import Web3

from bpx.consensus.block_creation import unfinished_block_to_full_block
from bpx.consensus.block_record import BlockRecord
from bpx.consensus.blockchain import Blockchain, ReceiveBlockResult, StateChangeSummary
from bpx.consensus.blockchain_interface import BlockchainInterface
from bpx.consensus.constants import ConsensusConstants
from bpx.consensus.difficulty_adjustment import get_next_sub_slot_iters_and_difficulty
from bpx.consensus.make_sub_epoch_summary import next_sub_epoch_summary
from bpx.consensus.multiprocess_validation import PreValidationResult
from bpx.consensus.pot_iterations import calculate_sp_iters
from bpx.beacon.block_store import BlockStore
from bpx.beacon.beacon_api import BeaconAPI
from bpx.beacon.beacon_store import BeaconStore, BeaconStorePeakResult
from bpx.beacon.lock_queue import LockClient, LockQueue
from bpx.beacon.signage_point import SignagePoint
from bpx.beacon.sync_store import SyncStore
from bpx.beacon.weight_proof import WeightProofHandler
from bpx.protocols import farmer_protocol, beacon_protocol, timelord_protocol
from bpx.protocols.beacon_protocol import RequestBlocks, RespondBlock, RespondBlocks, RespondSignagePoint
from bpx.protocols.protocol_message_types import ProtocolMessageTypes
from bpx.rpc.rpc_server import StateChangedProtocol
from bpx.server.node_discovery import BeaconPeers
from bpx.server.outbound_message import Message, NodeType, make_msg
from bpx.server.peer_store_resolver import PeerStoreResolver
from bpx.server.server import BpxServer
from bpx.server.ws_connection import WSBpxConnection
from bpx.types.blockchain_format.classgroup import ClassgroupElement
from bpx.types.blockchain_format.sized_bytes import bytes32
from bpx.types.blockchain_format.sub_epoch_summary import SubEpochSummary
from bpx.types.blockchain_format.vdf import CompressibleVDFField, VDFInfo, VDFProof
from bpx.types.end_of_slot_bundle import EndOfSubSlotBundle
from bpx.types.full_block import FullBlock
from bpx.types.header_block import HeaderBlock
from bpx.types.unfinished_block import UnfinishedBlock
from bpx.util.check_fork_next_block import check_fork_next_block
from bpx.util.config import process_config_start_method, lock_and_load_config, save_config
from bpx.util.db_synchronous import db_synchronous_on
from bpx.util.db_version import lookup_db_version
from bpx.util.db_wrapper import DbWrapper, manage_connection
from bpx.util.errors import ConsensusError, Err, ValidationError
from bpx.util.ints import uint8, uint32, uint64, uint128
from bpx.util.limited_semaphore import LimitedSemaphore
from bpx.util.path import path_from_root
from bpx.util.profiler import mem_profile_task, profile_task
from bpx.util.safe_cancel_task import cancel_task_safe
from bpx.beacon.execution_client import ExecutionClient


# This is the result of calling peak_post_processing, which is then fed into peak_post_processing_2
@dataclasses.dataclass
class PeakPostProcessingResult:
    fns_peak_result: BeaconStorePeakResult  # The result of calling BeaconStore.new_peak


class Beacon:
    _segment_task: Optional[asyncio.Task[None]]
    initialized: bool
    root_path: Path
    config: Dict[str, Any]
    _server: Optional[BpxServer]
    _shut_down: bool
    constants: ConsensusConstants
    pow_creation: Dict[bytes32, asyncio.Event]
    state_changed_callback: Optional[StateChangedProtocol] = None
    beacon_peers: Optional[BeaconPeers]
    sync_store: SyncStore
    signage_point_times: List[float]
    beacon_store: BeaconStore
    uncompact_task: Optional[asyncio.Task[None]]
    compact_vdf_requests: Set[bytes32]
    log: logging.Logger
    multiprocessing_context: Optional[BaseContext]
    _ui_tasks: Set[asyncio.Task[None]]
    db_path: Path
    _sync_task: Optional[asyncio.Task[None]]
    _compact_vdf_sem: Optional[LimitedSemaphore]
    _new_peak_sem: Optional[LimitedSemaphore]
    _db_wrapper: Optional[DbWrapper]
    _block_store: Optional[BlockStore]
    _init_weight_proof: Optional[asyncio.Task[None]]
    _blockchain: Optional[Blockchain]
    _timelord_lock: Optional[asyncio.Lock]
    weight_proof_handler: Optional[WeightProofHandler]
    _blockchain_lock_queue: Optional[LockQueue]
    _maybe_blockchain_lock_high_priority: Optional[LockClient]
    _maybe_blockchain_lock_low_priority: Optional[LockClient]
    execution_client: ExecutionClient

    @property
    def server(self) -> BpxServer:
        # This is a stop gap until the class usage is refactored such the values of
        # integral attributes are known at creation of the instance.
        if self._server is None:
            raise RuntimeError("server not assigned")

        return self._server

    def __init__(
        self,
        config: Dict[str, Any],
        root_path: Path,
        consensus_constants: ConsensusConstants,
        name: str = __name__,
    ) -> None:
        self._segment_task = None
        self.initialized = False
        self.root_path = root_path
        self.config = config
        self._server = None
        self._shut_down = False  # Set to true to close all infinite loops
        self.constants = consensus_constants
        self.pow_creation = {}
        self.state_changed_callback = None
        self.beacon_peers = None
        self.sync_store = SyncStore()
        self.signage_point_times = [time.time() for _ in range(self.constants.NUM_SPS_SUB_SLOT)]
        self.beacon_store = BeaconStore(self.constants)
        self.uncompact_task = None
        self.compact_vdf_requests = set()
        self.log = logging.getLogger(name)
        self.execution_client = None

        # TODO: Logging isn't setup yet so the log entries related to parsing the
        #       config would end up on stdout if handled here.
        self.multiprocessing_context = None

        self._ui_tasks = set()

        db_path_replaced: str = config["database_path"].replace("CHALLENGE", config["selected_network"])
        self.db_path = path_from_root(root_path, db_path_replaced)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._sync_task = None
        self._compact_vdf_sem = None
        self._new_peak_sem = None
        self._db_wrapper = None
        self._block_store = None
        self._init_weight_proof = None
        self._blockchain = None
        self._timelord_lock = None
        self.weight_proof_handler = None
        self._blockchain_lock_queue = None
        self._maybe_blockchain_lock_high_priority = None
        self._maybe_blockchain_lock_low_priority = None

    @property
    def block_store(self) -> BlockStore:
        assert self._block_store is not None
        return self._block_store

    @property
    def _blockchain_lock_high_priority(self) -> LockClient:
        assert self._maybe_blockchain_lock_high_priority is not None
        return self._maybe_blockchain_lock_high_priority

    @property
    def _blockchain_lock_low_priority(self) -> LockClient:
        assert self._maybe_blockchain_lock_low_priority is not None
        return self._maybe_blockchain_lock_low_priority

    @property
    def timelord_lock(self) -> asyncio.Lock:
        assert self._timelord_lock is not None
        return self._timelord_lock

    @property
    def blockchain(self) -> Blockchain:
        assert self._blockchain is not None
        return self._blockchain

    @property
    def db_wrapper(self) -> DbWrapper:
        assert self._db_wrapper is not None
        return self._db_wrapper

    @property
    def new_peak_sem(self) -> LimitedSemaphore:
        assert self._new_peak_sem is not None
        return self._new_peak_sem

    @property
    def compact_vdf_sem(self) -> LimitedSemaphore:
        assert self._compact_vdf_sem is not None
        return self._compact_vdf_sem

    def get_connections(self, request_node_type: Optional[NodeType]) -> List[Dict[str, Any]]:
        connections = self.server.get_connections(request_node_type)
        con_info: List[Dict[str, Any]] = []
        if self.sync_store is not None:
            peak_store = self.sync_store.peer_to_peak
        else:
            peak_store = None
        for con in connections:
            if peak_store is not None and con.peer_node_id in peak_store:
                peak = peak_store[con.peer_node_id]
                peak_height = peak.height
                peak_hash = peak.header_hash
                peak_weight = peak.weight
            else:
                peak_height = None
                peak_hash = None
                peak_weight = None
            con_dict: Dict[str, Any] = {
                "type": con.connection_type,
                "local_port": con.local_port,
                "peer_host": con.peer_host,
                "peer_port": con.peer_port,
                "peer_server_port": con.peer_server_port,
                "node_id": con.peer_node_id,
                "creation_time": con.creation_time,
                "bytes_read": con.bytes_read,
                "bytes_written": con.bytes_written,
                "last_message_time": con.last_message_time,
                "peak_height": peak_height,
                "peak_weight": peak_weight,
                "peak_hash": peak_hash,
            }
            con_info.append(con_dict)

        return con_info

    def _set_state_changed_callback(self, callback: StateChangedProtocol) -> None:
        self.state_changed_callback = callback

    async def _start(self) -> None:
        if not Web3.is_address(self.config["coinbase"]):
            raise ValueError("Coinbase is not a valid address, please check your config file")
        self.execution_client = ExecutionClient(self)
        
        self._timelord_lock = asyncio.Lock()
        self._compact_vdf_sem = LimitedSemaphore.create(active_limit=4, waiting_limit=20)

        # We don't want to run too many concurrent new_peak instances, because it would fetch the same block from
        # multiple peers and re-validate.
        self._new_peak_sem = LimitedSemaphore.create(active_limit=2, waiting_limit=20)

        sql_log_path: Optional[Path] = None
        if self.config.get("log_sqlite_cmds", False):
            sql_log_path = path_from_root(self.root_path, "log/sql.log")
            self.log.info(f"logging SQL commands to {sql_log_path}")

        # create the store (db) and beacon client instance
        # TODO: is this standardized and thus able to be handled by DbWrapper?
        async with manage_connection(self.db_path, log_path=sql_log_path, name="version_check") as db_connection:
            db_version = await lookup_db_version(db_connection)
        self.log.info(f"using blockchain database {self.db_path}, which is version {db_version}")

        db_sync = db_synchronous_on(self.config.get("db_sync", "auto"))
        self.log.info(f"opening blockchain DB: synchronous={db_sync}")

        self._db_wrapper = await DbWrapper.create(
            self.db_path,
            db_version=db_version,
            reader_count=4,
            log_path=sql_log_path,
            synchronous=db_sync,
        )

        self._block_store = await BlockStore.create(self.db_wrapper)
        self.log.info("Initializing blockchain from disk")
        start_time = time.time()
        reserved_cores = self.config.get("reserved_cores", 0)
        single_threaded = self.config.get("single_threaded", False)
        multiprocessing_start_method = process_config_start_method(config=self.config, log=self.log)
        self.multiprocessing_context = multiprocessing.get_context(method=multiprocessing_start_method)
        self._blockchain = await Blockchain.create(
            block_store=self.block_store,
            consensus_constants=self.constants,
            execution_client=self.execution_client,
            blockchain_dir=self.db_path.parent,
            reserved_cores=reserved_cores,
            multiprocessing_context=self.multiprocessing_context,
            single_threaded=single_threaded,
        )

        blockchain_lock_queue = LockQueue(self.blockchain.lock)
        self._blockchain_lock_queue = blockchain_lock_queue
        self._maybe_blockchain_lock_high_priority = LockClient(0, blockchain_lock_queue)
        self._maybe_blockchain_lock_low_priority = LockClient(1, blockchain_lock_queue)

        self._init_weight_proof = asyncio.create_task(self.initialize_weight_proof())

        if self.config.get("enable_profiler", False):
            asyncio.create_task(profile_task(self.root_path, "node", self.log))

        if self.config.get("enable_memory_profiler", False):
            asyncio.create_task(mem_profile_task(self.root_path, "node", self.log))
        
        time_taken = time.time() - start_time
        
        asyncio.create_task(self.execution_client.exchange_transition_configuration_task())
        
        peak: Optional[BlockRecord] = self.blockchain.get_peak()
        if peak is None:
            self.log.info(f"Initialized with empty blockchain time taken: {int(time_taken)}s")
        else:
            self.log.info(
                f"Blockchain initialized to peak {peak.header_hash} height"
                f" {peak.height}, "
                f"time taken: {int(time_taken)}s"
            )

            full_peak: Optional[FullBlock] = await self.blockchain.get_full_peak()
            assert full_peak is not None
            state_change_summary = StateChangeSummary(peak, uint32(max(peak.height - 1, 0)))
            ppp_result: PeakPostProcessingResult = await self.peak_post_processing(
                full_peak, state_change_summary, None
            )
            await self.peak_post_processing_2(full_peak, None, state_change_summary, ppp_result)
    
            tx_peak: BlockRecord = peak
            while not tx_peak.is_transaction_block:
                tx_peak = await self.blockchain.get_block_record_from_db(tx_peak.prev_hash)
            
            try:
                if tx_peak.height != 0:
                    full_tx_peak = await self.blockchain.get_full_block(tx_peak.header_hash)
                    assert full_tx_peak is not None
                    assert full_tx_peak.execution_payload is not None
                    
                    status = await self.execution_client.new_payload(full_tx_peak.execution_payload)
                    if status == "INVALID" or status == "INVALID_BLOCK_HASH":
                        raise RuntimeError(f"Payload status: {status}. Database is corrupted.")
                    elif status != "VALID" and status != "SYNCING" and status != "ACCEPTED":
                        raise RuntimeError("Unexpected payload status.")
                    
                status = await self.execution_client.forkchoice_update(tx_peak)
                if status == "INVALID" or status == "INVALID_BLOCK_HASH":
                    raise RuntimeError(f"Fork choice status: {status}. Database is corrupted.")
                elif status == "VALID":
                    self.log.info("Execution chain head has been updated.")
                elif status == "SYNCING" or status == "ACCEPTED":
                    self.log.info("Execution chain synchronization has been started.")
                else:
                    raise RuntimeError("Unexpected fork choice status.")
            except Exception as e:
                self.log.error(f"Exception in peak initialization: {e}")
            
        if self.config["send_uncompact_interval"] != 0:
            sanitize_weight_proof_only = False
            if "sanitize_weight_proof_only" in self.config:
                sanitize_weight_proof_only = self.config["sanitize_weight_proof_only"]
            assert self.config["target_uncompact_proofs"] != 0
            self.uncompact_task = asyncio.create_task(
                self.broadcast_uncompact_blocks(
                    self.config["send_uncompact_interval"],
                    self.config["target_uncompact_proofs"],
                    sanitize_weight_proof_only,
                )
            )
        
        self.initialized = True
        
        if self.beacon_peers is not None:
            asyncio.create_task(self.beacon_peers.start())
        

    async def initialize_weight_proof(self) -> None:
        self.weight_proof_handler = WeightProofHandler(
            constants=self.constants,
            blockchain=self.blockchain,
            multiprocessing_context=self.multiprocessing_context,
        )
        peak = self.blockchain.get_peak()
        if peak is not None:
            await self.weight_proof_handler.create_sub_epoch_segments()

    def set_server(self, server: BpxServer) -> None:
        self._server = server
        dns_servers: List[str] = []
        network_name = self.config["selected_network"]
        try:
            default_port = self.config["network_overrides"]["config"][network_name]["default_beacon_port"]
        except Exception:
            self.log.info("Default port field not found in config.")
            default_port = None
        if "dns_servers" in self.config:
            dns_servers = self.config["dns_servers"]
        elif network_name == "mainnet":
            # If `dns_servers` is missing from the `config`, hardcode it if we're running mainnet.
            dns_servers.append("dns-introducer.mainnet.bpxchain.cc")
        try:
            self.beacon_peers = BeaconPeers(
                self.server,
                self.config["target_outbound_peer_count"],
                PeerStoreResolver(
                    self.root_path,
                    self.config,
                    selected_network=network_name,
                    peers_file_path_key="peers_file_path",
                    default_peers_file_path="db/peers.dat",
                ),
                self.config["introducer_peer"],
                dns_servers,
                self.config["peer_connect_interval"],
                self.config["selected_network"],
                default_port,
                self.log,
            )
        except Exception as e:
            error_stack = traceback.format_exc()
            self.log.error(f"Exception: {e}")
            self.log.error(f"Exception in peer discovery: {e}")
            self.log.error(f"Exception Stack: {error_stack}")

    def _state_changed(self, change: str, change_data: Optional[Dict[str, Any]] = None) -> None:
        if self.state_changed_callback is not None:
            self.state_changed_callback(change, change_data)

    async def short_sync_batch(self, peer: WSBpxConnection, start_height: uint32, target_height: uint32) -> bool:
        """
        Tries to sync to a chain which is not too far in the future, by downloading batches of blocks. If the first
        block that we download is not connected to our chain, we return False and do an expensive long sync instead.
        Long sync is not preferred because it requires downloading and validating a weight proof.

        Args:
            peer: peer to sync from
            start_height: height that we should start downloading at. (Our peak is higher)
            target_height: target to sync to

        Returns:
            False if the fork point was not found, and we need to do a long sync. True otherwise.

        """
        # Don't trigger multiple batch syncs to the same peer

        if (
            peer.peer_node_id in self.sync_store.backtrack_syncing
            and self.sync_store.backtrack_syncing[peer.peer_node_id] > 0
        ):
            return True  # Don't batch sync, we are already in progress of a backtrack sync
        if peer.peer_node_id in self.sync_store.batch_syncing:
            return True  # Don't trigger a long sync
        self.sync_store.batch_syncing.add(peer.peer_node_id)

        self.log.info(f"Starting batch short sync from {start_height} to height {target_height}")
        if start_height > 0:
            first = await peer.call_api(
                BeaconAPI.request_block, beacon_protocol.RequestBlock(uint32(start_height))
            )
            if first is None or not isinstance(first, beacon_protocol.RespondBlock):
                self.sync_store.batch_syncing.remove(peer.peer_node_id)
                raise ValueError(f"Error short batch syncing, could not fetch block at height {start_height}")
            if not self.blockchain.contains_block(first.block.prev_header_hash):
                self.log.info("Batch syncing stopped, this is a deep chain")
                self.sync_store.batch_syncing.remove(peer.peer_node_id)
                # First sb not connected to our blockchain, do a long sync instead
                return False

        batch_size = self.constants.MAX_BLOCK_COUNT_PER_REQUESTS
        if self._segment_task is not None and (not self._segment_task.done()):
            try:
                self._segment_task.cancel()
            except Exception as e:
                self.log.warning(f"failed to cancel segment task {e}")
            self._segment_task = None

        try:
            for height in range(start_height, target_height, batch_size):
                end_height = min(target_height, height + batch_size)
                request = RequestBlocks(uint32(height), uint32(end_height))
                response = await peer.call_api(BeaconAPI.request_blocks, request)
                if not response:
                    raise ValueError(f"Error short batch syncing, invalid/no response for {height}-{end_height}")
                async with self._blockchain_lock_high_priority:
                    state_change_summary: Optional[StateChangeSummary]
                    success, state_change_summary = await self.add_block_batch(response.blocks, peer, None)
                    if not success:
                        raise ValueError(f"Error short batch syncing, failed to validate blocks {height}-{end_height}")
                    if state_change_summary is not None:
                        try:
                            peak_fb: Optional[FullBlock] = await self.blockchain.get_full_peak()
                            assert peak_fb is not None
                            ppp_result: PeakPostProcessingResult = await self.peak_post_processing(
                                peak_fb,
                                state_change_summary,
                                peer,
                            )
                            await self.peak_post_processing_2(peak_fb, peer, state_change_summary, ppp_result)
                        except Exception:
                            # Still do post processing after cancel (or exception)
                            peak_fb = await self.blockchain.get_full_peak()
                            assert peak_fb is not None
                            await self.peak_post_processing(peak_fb, state_change_summary, peer)
                            raise
                        finally:
                            self.log.info(f"Added blocks {height}-{end_height}")
        except (asyncio.CancelledError, Exception):
            self.sync_store.batch_syncing.remove(peer.peer_node_id)
            raise
        self.sync_store.batch_syncing.remove(peer.peer_node_id)
        return True

    async def short_sync_backtrack(
        self, peer: WSBpxConnection, peak_height: uint32, target_height: uint32, target_unf_hash: bytes32
    ) -> bool:
        """
        Performs a backtrack sync, where blocks are downloaded one at a time from newest to oldest. If we do not
        find the fork point 5 deeper than our peak, we return False and do a long sync instead.

        Args:
            peer: peer to sync from
            peak_height: height of our peak
            target_height: target height
            target_unf_hash: partial hash of the unfinished block of the target

        Returns:
            True iff we found the fork point, and we do not need to long sync.
        """
        try:
            if peer.peer_node_id not in self.sync_store.backtrack_syncing:
                self.sync_store.backtrack_syncing[peer.peer_node_id] = 0
            self.sync_store.backtrack_syncing[peer.peer_node_id] += 1

            unfinished_block: Optional[UnfinishedBlock] = self.beacon_store.get_unfinished_block(target_unf_hash)
            curr_height: int = target_height
            found_fork_point = False
            blocks = []
            while curr_height > peak_height - 5:
                curr = await peer.call_api(
                    BeaconAPI.request_block, beacon_protocol.RequestBlock(uint32(curr_height))
                )
                if curr is None:
                    raise ValueError(f"Failed to fetch block {curr_height} from {peer.get_peer_logging()}, timed out")
                if curr is None or not isinstance(curr, beacon_protocol.RespondBlock):
                    raise ValueError(
                        f"Failed to fetch block {curr_height} from {peer.get_peer_logging()}, wrong type {type(curr)}"
                    )
                blocks.append(curr.block)
                if self.blockchain.contains_block(curr.block.prev_header_hash) or curr_height == 0:
                    found_fork_point = True
                    break
                curr_height -= 1
            if found_fork_point:
                for block in reversed(blocks):
                    await self.add_block(block, peer)
        except (asyncio.CancelledError, Exception):
            self.sync_store.backtrack_syncing[peer.peer_node_id] -= 1
            raise

        self.sync_store.backtrack_syncing[peer.peer_node_id] -= 1
        return found_fork_point

    async def _refresh_ui_connections(self, sleep_before: float = 0) -> None:
        if sleep_before > 0:
            await asyncio.sleep(sleep_before)
        self._state_changed("peer_changed_peak")

    async def new_peak(self, request: beacon_protocol.NewPeak, peer: WSBpxConnection) -> None:
        """
        We have received a notification of a new peak from a peer. This happens either when we have just connected,
        or when the peer has updated their peak.

        Args:
            request: information about the new peak
            peer: peer that sent the message

        """

        try:
            seen_header_hash = self.sync_store.seen_header_hash(request.header_hash)
            # Updates heights in the UI. Sleeps 1.5s before, so other peers have time to update their peaks as well.
            # Limit to 3 refreshes.
            if not seen_header_hash and len(self._ui_tasks) < 3:
                self._ui_tasks.add(asyncio.create_task(self._refresh_ui_connections(1.5)))
            # Prune completed connect tasks
            self._ui_tasks = set(filter(lambda t: not t.done(), self._ui_tasks))
        except Exception as e:
            self.log.warning(f"Exception UI refresh task: {e}")

        # Store this peak/peer combination in case we want to sync to it, and to keep track of peers
        self.sync_store.peer_has_block(request.header_hash, peer.peer_node_id, request.weight, request.height, True)

        if self.blockchain.contains_block(request.header_hash):
            return None

        # Not interested in less heavy peaks
        peak: Optional[BlockRecord] = self.blockchain.get_peak()
        curr_peak_height = uint32(0) if peak is None else peak.height
        if peak is not None and peak.weight > request.weight:
            return None

        if self.sync_store.get_sync_mode():
            # If peer connects while we are syncing, check if they have the block we are syncing towards
            target_peak = self.sync_store.target_peak
            if target_peak is not None and request.header_hash != target_peak.header_hash:
                peak_peers: Set[bytes32] = self.sync_store.get_peers_that_have_peak([target_peak.header_hash])
                # Don't ask if we already know this peer has the peak
                if peer.peer_node_id not in peak_peers:
                    target_peak_response: Optional[RespondBlock] = await peer.call_api(
                        BeaconAPI.request_block,
                        beacon_protocol.RequestBlock(target_peak.height),
                        timeout=10,
                    )
                    if target_peak_response is not None and isinstance(target_peak_response, RespondBlock):
                        self.sync_store.peer_has_block(
                            target_peak.header_hash,
                            peer.peer_node_id,
                            target_peak_response.block.weight,
                            target_peak.height,
                            False,
                        )
        else:
            if (
                curr_peak_height <= request.height
                and request.height <= curr_peak_height + self.config["short_sync_blocks_behind_threshold"]
            ):
                # This is the normal case of receiving the next block
                if await self.short_sync_backtrack(
                    peer, curr_peak_height, request.height, request.unfinished_reward_block_hash
                ):
                    return None

            if request.height < self.constants.WEIGHT_PROOF_RECENT_BLOCKS:
                # This is the case of syncing up more than a few blocks, at the start of the chain
                self.log.debug("Doing batch sync, no backup")
                await self.short_sync_batch(peer, uint32(0), request.height)
                return None

            if (
                curr_peak_height <= request.height
                and request.height < curr_peak_height + self.config["sync_blocks_behind_threshold"]
            ):
                # This case of being behind but not by so much
                if await self.short_sync_batch(peer, uint32(max(curr_peak_height - 6, 0)), request.height):
                    return None

            # This is the either the case where we were not able to sync successfully (for example, due to the fork
            # point being in the past), or we are very far behind. Performs a long sync.
            self._sync_task = asyncio.create_task(self._sync())

    async def send_peak_to_timelords(
        self, peak_block: Optional[FullBlock] = None, peer: Optional[WSBpxConnection] = None
    ) -> None:
        """
        Sends current peak to timelords
        """
        if peak_block is None:
            peak_block = await self.blockchain.get_full_peak()
        if peak_block is not None:
            peak = self.blockchain.block_record(peak_block.header_hash)
            difficulty = self.blockchain.get_next_difficulty(peak.header_hash, False)
            ses: Optional[SubEpochSummary] = next_sub_epoch_summary(
                self.constants,
                self.blockchain,
                peak.required_iters,
                peak_block,
                True,
            )
            recent_rc = self.blockchain.get_recent_reward_challenges()

            curr = peak
            while not curr.is_challenge_block(self.constants) and not curr.first_in_sub_slot:
                curr = self.blockchain.block_record(curr.prev_hash)

            if curr.is_challenge_block(self.constants):
                last_csb_or_eos = curr.total_iters
            else:
                last_csb_or_eos = curr.ip_sub_slot_total_iters(self.constants)

            curr = peak
            passed_ses_height_but_not_yet_included = True
            while (curr.height % self.constants.SUB_EPOCH_BLOCKS) != 0:
                if curr.sub_epoch_summary_included:
                    passed_ses_height_but_not_yet_included = False
                curr = self.blockchain.block_record(curr.prev_hash)
            if curr.sub_epoch_summary_included or curr.height == 0:
                passed_ses_height_but_not_yet_included = False

            timelord_new_peak: timelord_protocol.NewPeakTimelord = timelord_protocol.NewPeakTimelord(
                peak_block.reward_chain_block,
                difficulty,
                peak.deficit,
                peak.sub_slot_iters,
                ses,
                recent_rc,
                last_csb_or_eos,
                passed_ses_height_but_not_yet_included,
            )

            msg = make_msg(ProtocolMessageTypes.new_peak_timelord, timelord_new_peak)
            if peer is None:
                await self.server.send_to_all([msg], NodeType.TIMELORD)
            else:
                await self.server.send_to_specific([msg], peer.peer_node_id)

    async def synced(self) -> bool:
        curr: Optional[BlockRecord] = self.blockchain.get_peak()
        if curr is None:
            return False
        
        while curr is not None and not curr.is_transaction_block:
            curr = self.blockchain.try_block_record(curr.prev_hash)

        now = time.time()
        if (
            curr is None
            or curr.timestamp is None
            or curr.timestamp < uint64(int(now - 60 * 7))
            or self.sync_store.get_sync_mode()
        ):
            return False
        else:
            return True

    async def on_connect(self, connection: WSBpxConnection) -> None:
        """
        Whenever we connect to another node, send them our current heads. Also send heads to farmers
        and challenges to timelords.
        """

        self._state_changed("add_connection")
        self._state_changed("sync_mode")
        if self.beacon_peers is not None:
            asyncio.create_task(self.beacon_peers.on_connect(connection))

        if self.initialized is False:
            return None

        peak_full: Optional[FullBlock] = await self.blockchain.get_full_peak()

        if peak_full is not None:
            peak: BlockRecord = self.blockchain.block_record(peak_full.header_hash)
            if connection.connection_type is NodeType.BEACON:
                request_node = beacon_protocol.NewPeak(
                    peak.header_hash,
                    peak.height,
                    peak.weight,
                    peak.height,
                    peak_full.reward_chain_block.get_unfinished().get_hash(),
                )
                await connection.send_message(make_msg(ProtocolMessageTypes.new_peak, request_node))

            elif connection.connection_type is NodeType.TIMELORD:
                await self.send_peak_to_timelords()

    def on_disconnect(self, connection: WSBpxConnection) -> None:
        self.log.info(f"peer disconnected {connection.get_peer_logging()}")
        self._state_changed("close_connection")
        self._state_changed("sync_mode")
        if self.sync_store is not None:
            self.sync_store.peer_disconnected(connection.peer_node_id)

    def _close(self) -> None:
        self._shut_down = True
        if self._init_weight_proof is not None:
            self._init_weight_proof.cancel()

        # blockchain is created in _start and in certain cases it may not exist here during _close
        if self._blockchain is not None:
            self.blockchain.shut_down()

        if self.beacon_peers is not None:
            asyncio.create_task(self.beacon_peers.close())
        if self.uncompact_task is not None:
            self.uncompact_task.cancel()
        if self._blockchain_lock_queue is not None:
            self._blockchain_lock_queue.close()
        cancel_task_safe(task=self._sync_task, log=self.log)

    async def _await_closed(self) -> None:
        await self.db_wrapper.close()
        if self._init_weight_proof is not None:
            await asyncio.wait([self._init_weight_proof])
        if self._blockchain_lock_queue is not None:
            await self._blockchain_lock_queue.await_closed()
        if self._sync_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._sync_task

    async def _sync(self) -> None:
        """
        Performs a full sync of the blockchain up to the peak.
            - Wait a few seconds for peers to send us their peaks
            - Select the heaviest peak, and request a weight proof from a peer with that peak
            - Validate the weight proof, and disconnect from the peer if invalid
            - Find the fork point to see where to start downloading blocks
            - Download blocks in batch (and in parallel) and verify them one at a time
            - Disconnect peers that provide invalid blocks or don't have the blocks
        """
        if self.weight_proof_handler is None:
            return None
        # Ensure we are only syncing once and not double calling this method
        if self.sync_store.get_sync_mode():
            return None

        if self.sync_store.get_long_sync():
            self.log.debug("already in long sync")
            return None

        self.sync_store.set_long_sync(True)
        self.log.debug("long sync started")
        try:
            self.log.info("Starting to perform sync.")
            self.log.info("Waiting to receive peaks from peers.")

            # Wait until we have 3 peaks or up to a max of 30 seconds
            peaks = []
            for i in range(300):
                peaks = [peak.header_hash for peak in self.sync_store.get_peak_of_each_peer().values()]
                if len(self.sync_store.get_peers_that_have_peak(peaks)) < 3:
                    if self._shut_down:
                        return None
                    await asyncio.sleep(0.1)
                    continue
                break

            self.log.info(f"Collected a total of {len(peaks)} peaks.")

            # Based on responses from peers about the current peaks, see which peak is the heaviest
            # (similar to longest chain rule).
            target_peak = self.sync_store.get_heaviest_peak()

            if target_peak is None:
                raise RuntimeError("Not performing sync, no peaks collected")

            self.sync_store.target_peak = target_peak

            self.log.info(f"Selected peak {target_peak}")
            # Check which peers are updated to this height

            peers = self.server.get_connections(NodeType.BEACON)
            coroutines = []
            for peer in peers:
                coroutines.append(
                    peer.call_api(
                        BeaconAPI.request_block,
                        beacon_protocol.RequestBlock(target_peak.height),
                        timeout=10,
                    )
                )
            for i, target_peak_response in enumerate(await asyncio.gather(*coroutines)):
                if target_peak_response is not None and isinstance(target_peak_response, RespondBlock):
                    self.sync_store.peer_has_block(
                        target_peak.header_hash, peers[i].peer_node_id, target_peak.weight, target_peak.height, False
                    )
            # TODO: disconnect from peer which gave us the heaviest_peak, if nobody has the peak

            peer_ids: Set[bytes32] = self.sync_store.get_peers_that_have_peak([target_peak.header_hash])
            peers_with_peak: List[WSBpxConnection] = [
                c for c in self.server.all_connections.values() if c.peer_node_id in peer_ids
            ]

            # Request weight proof from a random peer
            self.log.info(f"Total of {len(peers_with_peak)} peers with peak {target_peak.height}")
            weight_proof_peer: WSBpxConnection = random.choice(peers_with_peak)
            self.log.info(
                f"Requesting weight proof from peer {weight_proof_peer.peer_host} up to height {target_peak.height}"
            )
            cur_peak: Optional[BlockRecord] = self.blockchain.get_peak()
            if cur_peak is not None and target_peak.weight <= cur_peak.weight:
                raise ValueError("Not performing sync, already caught up.")

            wp_timeout = 360
            if "weight_proof_timeout" in self.config:
                wp_timeout = self.config["weight_proof_timeout"]
            self.log.debug(f"weight proof timeout is {wp_timeout} sec")
            request = beacon_protocol.RequestProofOfWeight(target_peak.height, target_peak.header_hash)
            response = await weight_proof_peer.call_api(
                BeaconAPI.request_proof_of_weight, request, timeout=wp_timeout
            )

            # Disconnect from this peer, because they have not behaved properly
            if response is None or not isinstance(response, beacon_protocol.RespondProofOfWeight):
                await weight_proof_peer.close(600)
                raise RuntimeError(f"Weight proof did not arrive in time from peer: {weight_proof_peer.peer_host}")
            if response.wp.recent_chain_data[-1].reward_chain_block.height != target_peak.height:
                await weight_proof_peer.close(600)
                raise RuntimeError(f"Weight proof had the wrong height: {weight_proof_peer.peer_host}")
            if response.wp.recent_chain_data[-1].reward_chain_block.weight != target_peak.weight:
                await weight_proof_peer.close(600)
                raise RuntimeError(f"Weight proof had the wrong weight: {weight_proof_peer.peer_host}")

            # dont sync to wp if local peak is heavier,
            # dont ban peer, we asked for this peak
            current_peak = self.blockchain.get_peak()
            if current_peak is not None:
                if response.wp.recent_chain_data[-1].reward_chain_block.weight <= current_peak.weight:
                    raise RuntimeError(f"current peak is heavier than Weight proof peek: {weight_proof_peer.peer_host}")

            try:
                validated, fork_point, summaries = await self.weight_proof_handler.validate_weight_proof(response.wp)
            except Exception as e:
                await weight_proof_peer.close(600)
                raise ValueError(f"Weight proof validation threw an error {e}")

            if not validated:
                await weight_proof_peer.close(600)
                raise ValueError("Weight proof validation failed")

            self.log.info(f"Re-checked peers: total of {len(peers_with_peak)} peers with peak {target_peak.height}")
            self.sync_store.set_sync_mode(True)
            self._state_changed("sync_mode")
            # Ensures that the fork point does not change
            async with self._blockchain_lock_high_priority:
                await self.blockchain.warmup(fork_point)
                await self.sync_from_fork_point(fork_point, target_peak.height, target_peak.header_hash, summaries)
        except asyncio.CancelledError:
            self.log.warning("Syncing failed, CancelledError")
        except Exception as e:
            tb = traceback.format_exc()
            self.log.error(f"Error with syncing: {type(e)}{tb}")
        finally:
            if self._shut_down:
                return None
            await self._finish_sync()

    async def sync_from_fork_point(
        self,
        fork_point_height: uint32,
        target_peak_sb_height: uint32,
        peak_hash: bytes32,
        summaries: List[SubEpochSummary],
    ) -> None:
        buffer_size = 4
        self.log.info(f"Start syncing from fork point at {fork_point_height} up to {target_peak_sb_height}")
        peers_with_peak: List[WSBpxConnection] = self.get_peers_with_peak(peak_hash)
        fork_point_height = await check_fork_next_block(
            self.blockchain, fork_point_height, peers_with_peak, node_next_block_check
        )
        batch_size = self.constants.MAX_BLOCK_COUNT_PER_REQUESTS

        async def fetch_block_batches(
            batch_queue: asyncio.Queue[Optional[Tuple[WSBpxConnection, List[FullBlock]]]]
        ) -> None:
            start_height, end_height = 0, 0
            new_peers_with_peak: List[WSBpxConnection] = peers_with_peak[:]
            try:
                for start_height in range(fork_point_height, target_peak_sb_height, batch_size):
                    end_height = min(target_peak_sb_height, start_height + batch_size)
                    request = RequestBlocks(uint32(start_height), uint32(end_height))
                    fetched = False
                    for peer in random.sample(new_peers_with_peak, len(new_peers_with_peak)):
                        if peer.closed:
                            peers_with_peak.remove(peer)
                            continue
                        response = await peer.call_api(BeaconAPI.request_blocks, request, timeout=30)
                        if response is None:
                            await peer.close()
                            peers_with_peak.remove(peer)
                        elif isinstance(response, RespondBlocks):
                            await batch_queue.put((peer, response.blocks))
                            fetched = True
                            break
                    if fetched is False:
                        self.log.error(f"failed fetching {start_height} to {end_height} from peers")
                        await batch_queue.put(None)
                        return
                    if self.sync_store.peers_changed.is_set():
                        new_peers_with_peak = self.get_peers_with_peak(peak_hash)
                        self.sync_store.peers_changed.clear()
            except Exception as e:
                self.log.error(f"Exception fetching {start_height} to {end_height} from peer {e}")
            finally:
                # finished signal with None
                await batch_queue.put(None)

        async def validate_block_batches(
            inner_batch_queue: asyncio.Queue[Optional[Tuple[WSBpxConnection, List[FullBlock]]]]
        ) -> None:
            advanced_peak: bool = False
            while True:
                res: Optional[Tuple[WSBpxConnection, List[FullBlock]]] = await inner_batch_queue.get()
                if res is None:
                    self.log.debug("done fetching blocks")
                    return None
                peer, blocks = res
                start_height = blocks[0].height
                end_height = blocks[-1].height
                success, state_change_summary = await self.add_block_batch(
                    blocks, peer, None if advanced_peak else uint32(fork_point_height), summaries
                )
                if success is False:
                    if peer in peers_with_peak:
                        peers_with_peak.remove(peer)
                    await peer.close(600)
                    raise ValueError(f"Failed to validate block batch {start_height} to {end_height}")
                self.log.info(f"Added blocks {start_height} to {end_height}")
                peak: Optional[BlockRecord] = self.blockchain.get_peak()
                if state_change_summary is not None:
                    advanced_peak = True
                    assert peak is not None
                self.blockchain.clean_block_record(end_height - self.constants.BLOCKS_CACHE_SIZE)

        batch_queue_input: asyncio.Queue[Optional[Tuple[WSBpxConnection, List[FullBlock]]]] = asyncio.Queue(
            maxsize=buffer_size
        )
        fetch_task = asyncio.Task(fetch_block_batches(batch_queue_input))
        validate_task = asyncio.Task(validate_block_batches(batch_queue_input))
        try:
            await asyncio.gather(fetch_task, validate_task)
        except Exception as e:
            assert validate_task.done()
            fetch_task.cancel()  # no need to cancel validate_task, if we end up here validate_task is already done
            self.log.error(f"sync from fork point failed err: {e}")

    def get_peers_with_peak(self, peak_hash: bytes32) -> List[WSBpxConnection]:
        peer_ids: Set[bytes32] = self.sync_store.get_peers_that_have_peak([peak_hash])
        if len(peer_ids) == 0:
            self.log.warning(f"Not syncing, no peers with header_hash {peak_hash} ")
            return []
        return [c for c in self.server.all_connections.values() if c.peer_node_id in peer_ids]

    async def add_block_batch(
        self,
        all_blocks: List[FullBlock],
        peer: WSBpxConnection,
        fork_point: Optional[uint32],
        wp_summaries: Optional[List[SubEpochSummary]] = None,
    ) -> Tuple[bool, Optional[StateChangeSummary]]:
        # Precondition: All blocks must be contiguous blocks, index i+1 must be the parent of index i
        # Returns a bool for success, as well as a StateChangeSummary if the peak was advanced

        blocks_to_validate: List[FullBlock] = []
        for i, block in enumerate(all_blocks):
            if not self.blockchain.contains_block(block.header_hash):
                blocks_to_validate = all_blocks[i:]
                break
        if len(blocks_to_validate) == 0:
            return True, None

        pre_validate_start = time.monotonic()
        pre_validation_results: List[PreValidationResult] = await self.blockchain.pre_validate_blocks_multiprocessing(
            blocks_to_validate, wp_summaries=wp_summaries
        )
        pre_validate_end = time.monotonic()
        pre_validate_time = pre_validate_end - pre_validate_start

        self.log.log(
            logging.WARNING if pre_validate_time > 10 else logging.DEBUG,
            f"Block pre-validation time: {pre_validate_end - pre_validate_start:0.2f} seconds "
            f"({len(blocks_to_validate)} blocks, start height: {blocks_to_validate[0].height})",
        )
        for i, block in enumerate(blocks_to_validate):
            if pre_validation_results[i].error is not None:
                self.log.error(
                    f"Invalid block from peer: {peer.get_peer_logging()} {Err(pre_validation_results[i].error)}"
                )
                return False, None

        agg_state_change_summary: Optional[StateChangeSummary] = None

        for i, block in enumerate(blocks_to_validate):
            assert pre_validation_results[i].required_iters is not None
            state_change_summary: Optional[StateChangeSummary]
            advanced_peak = agg_state_change_summary is not None
            result, error, state_change_summary = await self.blockchain.receive_block(
                block, pre_validation_results[i], None if advanced_peak else fork_point
            )

            if result == ReceiveBlockResult.NEW_PEAK:
                assert state_change_summary is not None
                # Since all blocks are contiguous, we can simply append the rollback changes and npc results
                if agg_state_change_summary is None:
                    agg_state_change_summary = state_change_summary
                else:
                    # Keeps the old, original fork_height, since the next blocks will have fork height h-1
                    # Groups up all state changes into one
                    agg_state_change_summary = StateChangeSummary(
                        state_change_summary.peak,
                        agg_state_change_summary.fork_height,
                    )
            elif result == ReceiveBlockResult.INVALID_BLOCK or result == ReceiveBlockResult.DISCONNECTED_BLOCK:
                if error is not None:
                    self.log.error(f"Error: {error}, Invalid block from peer: {peer.get_peer_logging()} ")
                return False, agg_state_change_summary
            block_record = self.blockchain.block_record(block.header_hash)
            if block_record.sub_epoch_summary_included is not None:
                if self.weight_proof_handler is not None:
                    await self.weight_proof_handler.create_prev_sub_epoch_segments()
        if agg_state_change_summary is not None:
            self._state_changed("new_peak")
            self.log.debug(
                f"Total time for {len(blocks_to_validate)} blocks: {time.time() - pre_validate_start}, "
                f"advanced: True"
            )
        return True, agg_state_change_summary

    async def _finish_sync(self) -> None:
        """
        Finalize sync by setting sync mode to False, clearing all sync information, and adding any final
        blocks that we have finalized recently.
        """
        self.log.info("long sync done")
        self.sync_store.set_long_sync(False)
        self.sync_store.set_sync_mode(False)
        self._state_changed("sync_mode")
        if self._server is None:
            return None

        async with self._blockchain_lock_high_priority:
            await self.sync_store.clear_sync_info()

            peak: Optional[BlockRecord] = self.blockchain.get_peak()
            peak_fb: Optional[FullBlock] = await self.blockchain.get_full_peak()
            if peak_fb is not None:
                assert peak is not None
                state_change_summary = StateChangeSummary(peak, uint32(max(peak.height - 1, 0)))
                ppp_result: PeakPostProcessingResult = await self.peak_post_processing(
                    peak_fb, state_change_summary, None
                )
                await self.peak_post_processing_2(peak_fb, None, state_change_summary, ppp_result)

        if peak is not None and self.weight_proof_handler is not None:
            await self.weight_proof_handler.get_proof_of_weight(peak.header_hash)
            self._state_changed("block")

    async def signage_point_post_processing(
        self,
        request: beacon_protocol.RespondSignagePoint,
        peer: WSBpxConnection,
        ip_sub_slot: Optional[EndOfSubSlotBundle],
    ) -> None:
        self.log.info(
            f"⏲️  Finished signage point {request.index_from_challenge}/"
            f"{self.constants.NUM_SPS_SUB_SLOT}: "
            f"CC: {request.challenge_chain_vdf.output.get_hash()} "
            f"RC: {request.reward_chain_vdf.output.get_hash()} "
        )
        self.signage_point_times[request.index_from_challenge] = time.time()
        sub_slot_tuple = self.beacon_store.get_sub_slot(request.challenge_chain_vdf.challenge)
        prev_challenge: Optional[bytes32]
        if sub_slot_tuple is not None:
            prev_challenge = sub_slot_tuple[0].challenge_chain.challenge_chain_end_of_slot_vdf.challenge
        else:
            prev_challenge = None

        # Notify nodes of the new signage point
        broadcast = beacon_protocol.NewSignagePointOrEndOfSubSlot(
            prev_challenge,
            request.challenge_chain_vdf.challenge,
            request.index_from_challenge,
            request.reward_chain_vdf.challenge,
        )
        msg = make_msg(ProtocolMessageTypes.new_signage_point_or_end_of_sub_slot, broadcast)
        await self.server.send_to_all([msg], NodeType.BEACON, peer.peer_node_id)

        peak = self.blockchain.get_peak()
        if peak is not None and peak.height > self.constants.MAX_SUB_SLOT_BLOCKS:
            sub_slot_iters = peak.sub_slot_iters
            difficulty = uint64(peak.weight - self.blockchain.block_record(peak.prev_hash).weight)
            # Makes sure to potentially update the difficulty if we are past the peak (into a new sub-slot)
            assert ip_sub_slot is not None
            if request.challenge_chain_vdf.challenge != ip_sub_slot.challenge_chain.get_hash():
                next_difficulty = self.blockchain.get_next_difficulty(peak.header_hash, True)
                next_sub_slot_iters = self.blockchain.get_next_slot_iters(peak.header_hash, True)
                difficulty = next_difficulty
                sub_slot_iters = next_sub_slot_iters
        else:
            difficulty = self.constants.DIFFICULTY_STARTING
            sub_slot_iters = self.constants.SUB_SLOT_ITERS_STARTING

        # Notify farmers of the new signage point
        broadcast_farmer = farmer_protocol.NewSignagePoint(
            request.challenge_chain_vdf.challenge,
            request.challenge_chain_vdf.output.get_hash(),
            request.reward_chain_vdf.output.get_hash(),
            difficulty,
            sub_slot_iters,
            request.index_from_challenge,
        )
        msg = make_msg(ProtocolMessageTypes.new_signage_point, broadcast_farmer)
        await self.server.send_to_all([msg], NodeType.FARMER)

        self._state_changed("signage_point", {"broadcast_farmer": broadcast_farmer})

    async def peak_post_processing(
        self,
        block: FullBlock,
        state_change_summary: StateChangeSummary,
        peer: Optional[WSBpxConnection],
    ) -> PeakPostProcessingResult:
        """
        Must be called under self.blockchain.lock. This updates the internal state of the beacon client with the
        latest peak information. It also notifies peers about the new peak.
        """

        record = state_change_summary.peak
        difficulty = self.blockchain.get_next_difficulty(record.header_hash, False)
        sub_slot_iters = self.blockchain.get_next_slot_iters(record.header_hash, False)

        self.log.info(
            f"🌱 Updated peak to height {record.height}, weight {record.weight}, "
            f"hh {record.header_hash}, "
            f"forked at {state_change_summary.fork_height}, rh: {record.reward_infusion_new_challenge}, "
            f"total iters: {record.total_iters}, "
            f"overflow: {record.overflow}, "
            f"deficit: {record.deficit}, "
            f"difficulty: {difficulty}, "
            f"sub slot iters: {sub_slot_iters}"
        )

        sub_slots = await self.blockchain.get_sp_and_ip_sub_slots(record.header_hash)
        assert sub_slots is not None

        if not self.sync_store.get_sync_mode():
            self.blockchain.clean_block_records()

        fork_block: Optional[BlockRecord] = None
        if state_change_summary.fork_height != block.height - 1 and block.height != 0:
            # This is a reorg
            fork_hash: Optional[bytes32] = self.blockchain.height_to_hash(state_change_summary.fork_height)
            assert fork_hash is not None
            fork_block = self.blockchain.block_record(fork_hash)

        fns_peak_result: BeaconStorePeakResult = self.beacon_store.new_peak(
            record,
            block,
            sub_slots[0],
            sub_slots[1],
            fork_block,
            self.blockchain,
        )

        if fns_peak_result.new_signage_points is not None and peer is not None:
            for index, sp in fns_peak_result.new_signage_points:
                assert (
                    sp.cc_vdf is not None
                    and sp.cc_proof is not None
                    and sp.rc_vdf is not None
                    and sp.rc_proof is not None
                )
                await self.signage_point_post_processing(
                    RespondSignagePoint(index, sp.cc_vdf, sp.cc_proof, sp.rc_vdf, sp.rc_proof), peer, sub_slots[1]
                )

        if sub_slots[1] is None:
            assert record.ip_sub_slot_total_iters(self.constants) == 0
        # Ensure the signage point is also in the store, for consistency
        self.beacon_store.new_signage_point(
            record.signage_point_index,
            self.blockchain,
            record,
            record.sub_slot_iters,
            SignagePoint(
                block.reward_chain_block.challenge_chain_sp_vdf,
                block.challenge_chain_sp_proof,
                block.reward_chain_block.reward_chain_sp_vdf,
                block.reward_chain_sp_proof,
            ),
            skip_vdf_validation=True,
        )

        return PeakPostProcessingResult(fns_peak_result)

    async def peak_post_processing_2(
        self,
        block: FullBlock,
        peer: Optional[WSBpxConnection],
        state_change_summary: StateChangeSummary,
        ppp_result: PeakPostProcessingResult,
    ) -> None:
        """
        Does NOT need to be called under the blockchain lock. Handle other parts of post processing like communicating
        with peers
        """
        record = state_change_summary.peak

        # If there were pending end of slots that happen after this peak, broadcast them if they are added
        if ppp_result.fns_peak_result.added_eos is not None:
            broadcast = beacon_protocol.NewSignagePointOrEndOfSubSlot(
                ppp_result.fns_peak_result.added_eos.challenge_chain.challenge_chain_end_of_slot_vdf.challenge,
                ppp_result.fns_peak_result.added_eos.challenge_chain.get_hash(),
                uint8(0),
                ppp_result.fns_peak_result.added_eos.reward_chain.end_of_slot_vdf.challenge,
            )
            msg = make_msg(ProtocolMessageTypes.new_signage_point_or_end_of_sub_slot, broadcast)
            await self.server.send_to_all([msg], NodeType.BEACON)

        # TODO: maybe add and broadcast new IPs as well

        if record.height % 1000 == 0:
            # Occasionally clear data in beacon client store to keep memory usage small
            self.beacon_store.clear_seen_unfinished_blocks()
            self.beacon_store.clear_old_cache_entries()
        
        if self.sync_store.get_sync_mode() is False:
            await self.send_peak_to_timelords(block)

            # Tell beacon clients about the new peak
            msg = make_msg(
                ProtocolMessageTypes.new_peak,
                beacon_protocol.NewPeak(
                    record.header_hash,
                    record.height,
                    record.weight,
                    state_change_summary.fork_height,
                    block.reward_chain_block.get_unfinished().get_hash(),
                ),
            )
            if peer is not None:
                await self.server.send_to_all([msg], NodeType.BEACON, peer.peer_node_id)
            else:
                await self.server.send_to_all([msg], NodeType.BEACON)

        self._state_changed("new_peak")

    async def add_block(
        self,
        block: FullBlock,
        peer: Optional[WSBpxConnection] = None,
        raise_on_disconnected: bool = False,
    ) -> Optional[Message]:
        """
        Add a full block from a peer beacon client (or ourselves).
        """
        if self.sync_store.get_sync_mode():
            return None

        # Adds the block to seen, and check if it's seen before (which means header is in memory)
        header_hash = block.header_hash
        if self.blockchain.contains_block(header_hash):
            return None

        pre_validation_result: Optional[PreValidationResult] = None
        state_change_summary: Optional[StateChangeSummary] = None
        ppp_result: Optional[PeakPostProcessingResult] = None
        async with self._blockchain_lock_high_priority:
            # After acquiring the lock, check again, because another asyncio thread might have added it
            if self.blockchain.contains_block(header_hash):
                return None
            validation_start = time.time()

            pre_validation_results = await self.blockchain.pre_validate_blocks_multiprocessing(
                [block]
            )
            added: Optional[ReceiveBlockResult] = None
            pre_validation_time = time.time() - validation_start
            try:
                if len(pre_validation_results) < 1:
                    raise ValueError(f"Failed to validate block {header_hash} height {block.height}")
                if pre_validation_results[0].error is not None:
                    if Err(pre_validation_results[0].error) == Err.INVALID_PREV_BLOCK_HASH:
                        added = ReceiveBlockResult.DISCONNECTED_BLOCK
                        error_code: Optional[Err] = Err.INVALID_PREV_BLOCK_HASH
                    else:
                        raise ValueError(
                            f"Failed to validate block {header_hash} height "
                            f"{block.height}: {Err(pre_validation_results[0].error).name}"
                        )
                else:
                    result_to_validate = (
                        pre_validation_results[0] if pre_validation_result is None else pre_validation_result
                    )
                    assert result_to_validate.required_iters == pre_validation_results[0].required_iters
                    (added, error_code, state_change_summary) = await self.blockchain.receive_block(
                        block, result_to_validate, None
                    )
                if added == ReceiveBlockResult.ALREADY_HAVE_BLOCK:
                    return None
                elif added == ReceiveBlockResult.INVALID_BLOCK:
                    assert error_code is not None
                    self.log.error(f"Block {header_hash} at height {block.height} is invalid with code {error_code}.")
                    raise ConsensusError(error_code, [header_hash])
                elif added == ReceiveBlockResult.DISCONNECTED_BLOCK:
                    self.log.info(f"Disconnected block {header_hash} at height {block.height}")
                    if raise_on_disconnected:
                        raise RuntimeError("Expected block to be added, received disconnected block.")
                    return None
                elif added == ReceiveBlockResult.NEW_PEAK:
                    # Only propagate blocks which extend the blockchain (becomes one of the heads)
                    assert state_change_summary is not None
                    ppp_result = await self.peak_post_processing(block, state_change_summary, peer)

                elif added == ReceiveBlockResult.ADDED_AS_ORPHAN:
                    self.log.info(
                        f"Received orphan block of height {block.height} rh {block.reward_chain_block.get_hash()}"
                    )
                else:
                    # Should never reach here, all the cases are covered
                    raise RuntimeError(f"Invalid result from receive_block {added}")
            except asyncio.CancelledError:
                # We need to make sure to always call this method even when we get a cancel exception, to make sure
                # the node stays in sync
                if added == ReceiveBlockResult.NEW_PEAK:
                    assert state_change_summary is not None
                    await self.peak_post_processing(block, state_change_summary, peer)
                raise

            validation_time = time.time() - validation_start

        if ppp_result is not None:
            assert state_change_summary is not None
            await self.peak_post_processing_2(block, peer, state_change_summary, ppp_result)

        self.log.log(
            logging.WARNING if validation_time > 2 else logging.DEBUG,
            f"Block validation time: {validation_time:0.2f} seconds, "
            f"pre_validation time: {pre_validation_time:0.2f} seconds, "
            f"header_hash: {header_hash} height: {block.height}",
        )

        # This code path is reached if added == ADDED_AS_ORPHAN or NEW_TIP
        peak = self.blockchain.get_peak()
        assert peak is not None

        # Removes all temporary data for old blocks
        clear_height = uint32(max(0, peak.height - 50))
        self.beacon_store.clear_candidate_blocks_below(clear_height)
        self.beacon_store.clear_unfinished_blocks_below(clear_height)
        if peak.height % 1000 == 0 and not self.sync_store.get_sync_mode():
            await self.sync_store.clear_sync_info()  # Occasionally clear sync peer info

        state_changed_data: Dict[str, Any] = {
            "transaction_block": False,
            "k_size": block.reward_chain_block.proof_of_space.size,
            "header_hash": block.header_hash,
            "height": block.height,
            "validation_time": validation_time,
            "pre_validation_time": pre_validation_time,
        }
        
        if block.foliage_transaction_block is not None:
            state_changed_data["timestamp"] = block.foliage_transaction_block.timestamp

        if added is not None:
            state_changed_data["receive_block_result"] = added.value

        self._state_changed("block", state_changed_data)

        record = self.blockchain.block_record(block.header_hash)
        if self.weight_proof_handler is not None and record.sub_epoch_summary_included is not None:
            if self._segment_task is None or self._segment_task.done():
                self._segment_task = asyncio.create_task(self.weight_proof_handler.create_prev_sub_epoch_segments())
        return None

    async def add_unfinished_block(
        self,
        block: UnfinishedBlock,
        peer: Optional[WSBpxConnection],
        farmed_block: bool = False,
        block_bytes: Optional[bytes] = None,
    ) -> None:
        """
        We have received an unfinished block, either created by us, or from another peer.
        We can validate and add it and if it's a good block, propagate it to other peers and
        timelords.
        """
        receive_time = time.time()

        if block.prev_header_hash != self.constants.GENESIS_CHALLENGE and not self.blockchain.contains_block(
            block.prev_header_hash
        ):
            # No need to request the parent, since the peer will send it to us anyway, via NewPeak
            self.log.debug("Received a disconnected unfinished block")
            return None

        # Adds the unfinished block to seen, and check if it's seen before, to prevent
        # processing it twice. This searches for the exact version of the unfinished block (there can be many different
        # foliages for the same trunk). This is intentional, to prevent DOS attacks.
        # Note that it does not require that this block was successfully processed
        if self.beacon_store.seen_unfinished_block(block.get_hash()):
            return None

        block_hash = block.reward_chain_block.get_hash()

        # This searched for the trunk hash (unfinished reward hash). If we have already added a block with the same
        # hash, return
        if self.beacon_store.get_unfinished_block(block_hash) is not None:
            return None

        peak: Optional[BlockRecord] = self.blockchain.get_peak()
        if peak is not None:
            if block.total_iters < peak.sp_total_iters(self.constants):
                # This means this unfinished block is pretty far behind, it will not add weight to our chain
                return None

        if block.prev_header_hash == self.constants.GENESIS_CHALLENGE:
            prev_b = None
        else:
            prev_b = self.blockchain.block_record(block.prev_header_hash)

        # Count the blocks in sub slot, and check if it's a new epoch
        if len(block.finished_sub_slots) > 0:
            num_blocks_in_ss = 1  # Curr
        else:
            curr = self.blockchain.try_block_record(block.prev_header_hash)
            num_blocks_in_ss = 2  # Curr and prev
            while (curr is not None) and not curr.first_in_sub_slot:
                curr = self.blockchain.try_block_record(curr.prev_hash)
                num_blocks_in_ss += 1

        if num_blocks_in_ss > self.constants.MAX_SUB_SLOT_BLOCKS:
            # TODO: potentially allow overflow blocks here, which count for the next slot
            self.log.warning("Too many blocks added, not adding block")
            return None

        pre_validation_time = None

        async with self._blockchain_lock_high_priority:
            start_header_time = time.time()
            _, header_error = await self.blockchain.validate_unfinished_block_header(block)
            if header_error is not None:
                raise ConsensusError(header_error)
            validate_time = time.time() - start_header_time
            self.log.log(
                logging.WARNING if validate_time > 2 else logging.DEBUG,
                f"Time for header validate: {validate_time:0.3f}s",
            )

        async with self._blockchain_lock_high_priority:
            # TODO: pre-validate VDFs outside of lock
            validation_start = time.time()
            validate_result = await self.blockchain.validate_unfinished_block(block)
            if validate_result.error is not None:
                raise ConsensusError(Err(validate_result.error))
            validation_time = time.time() - validation_start

        validate_result = dataclasses.replace(validate_result)

        assert validate_result.required_iters is not None

        # Perform another check, in case we have already concurrently added the same unfinished block
        if self.beacon_store.get_unfinished_block(block_hash) is not None:
            return None

        if block.prev_header_hash == self.constants.GENESIS_CHALLENGE:
            height = uint32(0)
        else:
            height = uint32(self.blockchain.block_record(block.prev_header_hash).height + 1)

        ses: Optional[SubEpochSummary] = next_sub_epoch_summary(
            self.constants,
            self.blockchain,
            validate_result.required_iters,
            block,
            True,
        )

        self.beacon_store.add_unfinished_block(height, block, validate_result)
        pre_validation_log = (
            f"pre_validation time {pre_validation_time:0.4f}, " if pre_validation_time is not None else ""
        )
        if farmed_block is True:
            self.log.info(
                f"🍀 ️Farmed unfinished_block {block_hash}, SP: {block.reward_chain_block.signage_point_index}, "
                f"validation time: {validation_time:0.4f} seconds, {pre_validation_log}"
            )
        else:
            self.log.info(
                f"Added unfinished_block {block_hash}, not farmed by us,"
                f" SP: {block.reward_chain_block.signage_point_index} farmer response time: "
                f"{receive_time - self.signage_point_times[block.reward_chain_block.signage_point_index]:0.4f}, "
                f"validation time: {validation_time:0.4f} seconds, {pre_validation_log}"
            )

        sub_slot_iters, difficulty = get_next_sub_slot_iters_and_difficulty(
            self.constants,
            len(block.finished_sub_slots) > 0,
            prev_b,
            self.blockchain,
        )

        if block.reward_chain_block.signage_point_index == 0:
            res = self.beacon_store.get_sub_slot(block.reward_chain_block.pos_ss_cc_challenge_hash)
            if res is None:
                if block.reward_chain_block.pos_ss_cc_challenge_hash == self.constants.GENESIS_CHALLENGE:
                    rc_prev = self.constants.GENESIS_CHALLENGE
                else:
                    self.log.warning(f"Do not have sub slot {block.reward_chain_block.pos_ss_cc_challenge_hash}")
                    return None
            else:
                rc_prev = res[0].reward_chain.get_hash()
        else:
            assert block.reward_chain_block.reward_chain_sp_vdf is not None
            rc_prev = block.reward_chain_block.reward_chain_sp_vdf.challenge

        timelord_request = timelord_protocol.NewUnfinishedBlockTimelord(
            block.reward_chain_block,
            difficulty,
            sub_slot_iters,
            block.foliage,
            ses,
            rc_prev,
        )

        timelord_msg = make_msg(ProtocolMessageTypes.new_unfinished_block_timelord, timelord_request)
        await self.server.send_to_all([timelord_msg], NodeType.TIMELORD)

        beacon_request = beacon_protocol.NewUnfinishedBlock(block.reward_chain_block.get_hash())
        msg = make_msg(ProtocolMessageTypes.new_unfinished_block, beacon_request)
        if peer is not None:
            await self.server.send_to_all([msg], NodeType.BEACON, peer.peer_node_id)
        else:
            await self.server.send_to_all([msg], NodeType.BEACON)

        self._state_changed("unfinished_block")

    async def new_infusion_point_vdf(
        self, request: timelord_protocol.NewInfusionPointVDF, timelord_peer: Optional[WSBpxConnection] = None
    ) -> Optional[Message]:
        # Lookup unfinished blocks
        unfinished_block: Optional[UnfinishedBlock] = self.beacon_store.get_unfinished_block(
            request.unfinished_reward_hash
        )

        if unfinished_block is None:
            self.log.warning(
                f"Do not have unfinished reward chain block {request.unfinished_reward_hash}, cannot finish."
            )
            return None

        prev_b: Optional[BlockRecord] = None

        target_rc_hash = request.reward_chain_ip_vdf.challenge
        last_slot_cc_hash = request.challenge_chain_ip_vdf.challenge

        # Backtracks through end of slot objects, should work for multiple empty sub slots
        for eos, _, _ in reversed(self.beacon_store.finished_sub_slots):
            if eos is not None and eos.reward_chain.get_hash() == target_rc_hash:
                target_rc_hash = eos.reward_chain.end_of_slot_vdf.challenge
        if target_rc_hash == self.constants.GENESIS_CHALLENGE:
            prev_b = None
        else:
            # Find the prev block, starts looking backwards from the peak. target_rc_hash must be the hash of a block
            # and not an end of slot (since we just looked through the slots and backtracked)
            curr: Optional[BlockRecord] = self.blockchain.get_peak()

            for _ in range(10):
                if curr is None:
                    break
                if curr.reward_infusion_new_challenge == target_rc_hash:
                    # Found our prev block
                    prev_b = curr
                    break
                curr = self.blockchain.try_block_record(curr.prev_hash)

            # If not found, cache keyed on prev block
            if prev_b is None:
                self.beacon_store.add_to_future_ip(request)
                self.log.warning(f"Previous block is None, infusion point {request.reward_chain_ip_vdf.challenge}")
                return None

        finished_sub_slots: Optional[List[EndOfSubSlotBundle]] = self.beacon_store.get_finished_sub_slots(
            self.blockchain,
            prev_b,
            last_slot_cc_hash,
        )
        if finished_sub_slots is None:
            return None

        sub_slot_iters, difficulty = get_next_sub_slot_iters_and_difficulty(
            self.constants,
            len(finished_sub_slots) > 0,
            prev_b,
            self.blockchain,
        )

        if unfinished_block.reward_chain_block.pos_ss_cc_challenge_hash == self.constants.GENESIS_CHALLENGE:
            sub_slot_start_iters = uint128(0)
        else:
            ss_res = self.beacon_store.get_sub_slot(unfinished_block.reward_chain_block.pos_ss_cc_challenge_hash)
            if ss_res is None:
                self.log.warning(f"Do not have sub slot {unfinished_block.reward_chain_block.pos_ss_cc_challenge_hash}")
                return None
            _, _, sub_slot_start_iters = ss_res
        sp_total_iters = uint128(
            sub_slot_start_iters
            + calculate_sp_iters(
                self.constants,
                sub_slot_iters,
                unfinished_block.reward_chain_block.signage_point_index,
            )
        )

        block: FullBlock = unfinished_block_to_full_block(
            unfinished_block,
            request.challenge_chain_ip_vdf,
            request.challenge_chain_ip_proof,
            request.reward_chain_ip_vdf,
            request.reward_chain_ip_proof,
            request.infused_challenge_chain_ip_vdf,
            request.infused_challenge_chain_ip_proof,
            finished_sub_slots,
            prev_b,
            self.blockchain,
            sp_total_iters,
            difficulty,
        )
        try:
            await self.add_block(block, raise_on_disconnected=True)
        except Exception as e:
            self.log.warning(f"Consensus error validating block: {e}")
            if timelord_peer is not None:
                # Only sends to the timelord who sent us this VDF, to reset them to the correct peak
                await self.send_peak_to_timelords(peer=timelord_peer)
        return None

    async def add_end_of_sub_slot(
        self, end_of_slot_bundle: EndOfSubSlotBundle, peer: WSBpxConnection
    ) -> Tuple[Optional[Message], bool]:
        fetched_ss = self.beacon_store.get_sub_slot(end_of_slot_bundle.challenge_chain.get_hash())

        # We are not interested in sub-slots which have the same challenge chain but different reward chain. If there
        # is a reorg, we will find out through the broadcast of blocks instead.
        if fetched_ss is not None:
            # Already have the sub-slot
            return None, True

        async with self.timelord_lock:
            fetched_ss = self.beacon_store.get_sub_slot(
                end_of_slot_bundle.challenge_chain.challenge_chain_end_of_slot_vdf.challenge
            )
            if (
                (fetched_ss is None)
                and end_of_slot_bundle.challenge_chain.challenge_chain_end_of_slot_vdf.challenge
                != self.constants.GENESIS_CHALLENGE
            ):
                # If we don't have the prev, request the prev instead
                beacon_request = beacon_protocol.RequestSignagePointOrEndOfSubSlot(
                    end_of_slot_bundle.challenge_chain.challenge_chain_end_of_slot_vdf.challenge,
                    uint8(0),
                    bytes32([0] * 32),
                )
                return (
                    make_msg(ProtocolMessageTypes.request_signage_point_or_end_of_sub_slot, beacon_request),
                    False,
                )

            peak = self.blockchain.get_peak()
            if peak is not None and peak.height > 2:
                next_sub_slot_iters = self.blockchain.get_next_slot_iters(peak.header_hash, True)
                next_difficulty = self.blockchain.get_next_difficulty(peak.header_hash, True)
            else:
                next_sub_slot_iters = self.constants.SUB_SLOT_ITERS_STARTING
                next_difficulty = self.constants.DIFFICULTY_STARTING

            # Adds the sub slot and potentially get new infusions
            new_infusions = self.beacon_store.new_finished_sub_slot(
                end_of_slot_bundle,
                self.blockchain,
                peak,
                await self.blockchain.get_full_peak(),
            )
            # It may be an empty list, even if it's not None. Not None means added successfully
            if new_infusions is not None:
                self.log.info(
                    f"⏲️  Finished sub slot, SP {self.constants.NUM_SPS_SUB_SLOT}/{self.constants.NUM_SPS_SUB_SLOT}, "
                    f"{end_of_slot_bundle.challenge_chain.get_hash()}, "
                    f"number of sub-slots: {len(self.beacon_store.finished_sub_slots)}, "
                    f"RC hash: {end_of_slot_bundle.reward_chain.get_hash()}, "
                    f"Deficit {end_of_slot_bundle.reward_chain.deficit}"
                )
                # Reset farmer response timer for sub slot (SP 0)
                self.signage_point_times[0] = time.time()
                # Notify beacon clients of the new sub-slot
                broadcast = beacon_protocol.NewSignagePointOrEndOfSubSlot(
                    end_of_slot_bundle.challenge_chain.challenge_chain_end_of_slot_vdf.challenge,
                    end_of_slot_bundle.challenge_chain.get_hash(),
                    uint8(0),
                    end_of_slot_bundle.reward_chain.end_of_slot_vdf.challenge,
                )
                msg = make_msg(ProtocolMessageTypes.new_signage_point_or_end_of_sub_slot, broadcast)
                await self.server.send_to_all([msg], NodeType.BEACON, peer.peer_node_id)

                for infusion in new_infusions:
                    await self.new_infusion_point_vdf(infusion)

                # Notify farmers of the new sub-slot
                broadcast_farmer = farmer_protocol.NewSignagePoint(
                    end_of_slot_bundle.challenge_chain.get_hash(),
                    end_of_slot_bundle.challenge_chain.get_hash(),
                    end_of_slot_bundle.reward_chain.get_hash(),
                    next_difficulty,
                    next_sub_slot_iters,
                    uint8(0),
                )
                msg = make_msg(ProtocolMessageTypes.new_signage_point, broadcast_farmer)
                await self.server.send_to_all([msg], NodeType.FARMER)
                return None, True
            else:
                self.log.info(
                    f"End of slot not added CC challenge "
                    f"{end_of_slot_bundle.challenge_chain.challenge_chain_end_of_slot_vdf.challenge}"
                )
        return None, False
    
    async def get_coinbase(self) -> str:
        return self.config["coinbase"]

    def set_coinbase(self, coinbase: str) -> None:
        if not Web3.is_address(coinbase):
            raise ValueError("Invalid address")
            
        self.config["coinbase"] = coinbase
        
        with lock_and_load_config(self.root_path, "config.yaml") as config:
            config["beacon"]["coinbase"] = coinbase
            save_config(self.root_path, "config.yaml", config)

    async def _needs_compact_proof(
        self, vdf_info: VDFInfo, header_block: HeaderBlock, field_vdf: CompressibleVDFField
    ) -> bool:
        if field_vdf == CompressibleVDFField.CC_EOS_VDF:
            for sub_slot in header_block.finished_sub_slots:
                if sub_slot.challenge_chain.challenge_chain_end_of_slot_vdf == vdf_info:
                    if (
                        sub_slot.proofs.challenge_chain_slot_proof.witness_type == 0
                        and sub_slot.proofs.challenge_chain_slot_proof.normalized_to_identity
                    ):
                        return False
                    return True
        if field_vdf == CompressibleVDFField.ICC_EOS_VDF:
            for sub_slot in header_block.finished_sub_slots:
                if (
                    sub_slot.infused_challenge_chain is not None
                    and sub_slot.infused_challenge_chain.infused_challenge_chain_end_of_slot_vdf == vdf_info
                ):
                    assert sub_slot.proofs.infused_challenge_chain_slot_proof is not None
                    if (
                        sub_slot.proofs.infused_challenge_chain_slot_proof.witness_type == 0
                        and sub_slot.proofs.infused_challenge_chain_slot_proof.normalized_to_identity
                    ):
                        return False
                    return True
        if field_vdf == CompressibleVDFField.CC_SP_VDF:
            if header_block.reward_chain_block.challenge_chain_sp_vdf is None:
                return False
            if vdf_info == header_block.reward_chain_block.challenge_chain_sp_vdf:
                assert header_block.challenge_chain_sp_proof is not None
                if (
                    header_block.challenge_chain_sp_proof.witness_type == 0
                    and header_block.challenge_chain_sp_proof.normalized_to_identity
                ):
                    return False
                return True
        if field_vdf == CompressibleVDFField.CC_IP_VDF:
            if vdf_info == header_block.reward_chain_block.challenge_chain_ip_vdf:
                if (
                    header_block.challenge_chain_ip_proof.witness_type == 0
                    and header_block.challenge_chain_ip_proof.normalized_to_identity
                ):
                    return False
                return True
        return False

    async def _can_accept_compact_proof(
        self,
        vdf_info: VDFInfo,
        vdf_proof: VDFProof,
        height: uint32,
        header_hash: bytes32,
        field_vdf: CompressibleVDFField,
    ) -> bool:
        """
        - Checks if the provided proof is indeed compact.
        - Checks if proof verifies given the vdf_info from the start of sub-slot.
        - Checks if the provided vdf_info is correct, assuming it refers to the start of sub-slot.
        - Checks if the existing proof was non-compact. Ignore this proof if we already have a compact proof.
        """
        is_fully_compactified = await self.block_store.is_fully_compactified(header_hash)
        if is_fully_compactified is None or is_fully_compactified:
            self.log.info(f"Already compactified block: {header_hash}. Ignoring.")
            return False
        peak = self.blockchain.get_peak()
        if peak is None or peak.height - height < 5:
            self.log.debug("Will not compactify recent block")
            return False
        if vdf_proof.witness_type > 0 or not vdf_proof.normalized_to_identity:
            self.log.error(f"Received vdf proof is not compact: {vdf_proof}.")
            return False
        if not vdf_proof.is_valid(self.constants, ClassgroupElement.get_default_element(), vdf_info):
            self.log.error(f"Received compact vdf proof is not valid: {vdf_proof}.")
            return False
        header_block = await self.blockchain.get_header_block_by_height(height, header_hash)
        if header_block is None:
            self.log.error(f"Can't find block for given compact vdf. Height: {height} Header hash: {header_hash}")
            return False
        is_new_proof = await self._needs_compact_proof(vdf_info, header_block, field_vdf)
        if not is_new_proof:
            self.log.info(f"Duplicate compact proof. Height: {height}. Header hash: {header_hash}.")
        return is_new_proof

    # returns True if we ended up replacing the proof, and False otherwise
    async def _replace_proof(
        self,
        vdf_info: VDFInfo,
        vdf_proof: VDFProof,
        header_hash: bytes32,
        field_vdf: CompressibleVDFField,
    ) -> bool:
        block = await self.block_store.get_full_block(header_hash)
        if block is None:
            return False

        new_block = None

        if field_vdf == CompressibleVDFField.CC_EOS_VDF:
            for index, sub_slot in enumerate(block.finished_sub_slots):
                if sub_slot.challenge_chain.challenge_chain_end_of_slot_vdf == vdf_info:
                    new_proofs = dataclasses.replace(sub_slot.proofs, challenge_chain_slot_proof=vdf_proof)
                    new_subslot = dataclasses.replace(sub_slot, proofs=new_proofs)
                    new_finished_subslots = block.finished_sub_slots
                    new_finished_subslots[index] = new_subslot
                    new_block = dataclasses.replace(block, finished_sub_slots=new_finished_subslots)
                    break
        if field_vdf == CompressibleVDFField.ICC_EOS_VDF:
            for index, sub_slot in enumerate(block.finished_sub_slots):
                if (
                    sub_slot.infused_challenge_chain is not None
                    and sub_slot.infused_challenge_chain.infused_challenge_chain_end_of_slot_vdf == vdf_info
                ):
                    new_proofs = dataclasses.replace(sub_slot.proofs, infused_challenge_chain_slot_proof=vdf_proof)
                    new_subslot = dataclasses.replace(sub_slot, proofs=new_proofs)
                    new_finished_subslots = block.finished_sub_slots
                    new_finished_subslots[index] = new_subslot
                    new_block = dataclasses.replace(block, finished_sub_slots=new_finished_subslots)
                    break
        if field_vdf == CompressibleVDFField.CC_SP_VDF:
            if block.reward_chain_block.challenge_chain_sp_vdf == vdf_info:
                assert block.challenge_chain_sp_proof is not None
                new_block = dataclasses.replace(block, challenge_chain_sp_proof=vdf_proof)
        if field_vdf == CompressibleVDFField.CC_IP_VDF:
            if block.reward_chain_block.challenge_chain_ip_vdf == vdf_info:
                new_block = dataclasses.replace(block, challenge_chain_ip_proof=vdf_proof)
        if new_block is None:
            return False
        async with self.db_wrapper.writer():
            try:
                await self.block_store.replace_proof(header_hash, new_block)
                return True
            except BaseException as e:
                self.log.error(
                    f"_replace_proof error while adding block {block.header_hash} height {block.height},"
                    f" rolling back: {e} {traceback.format_exc()}"
                )
                raise

    async def add_compact_proof_of_time(self, request: timelord_protocol.RespondCompactProofOfTime) -> None:
        field_vdf = CompressibleVDFField(int(request.field_vdf))
        if not await self._can_accept_compact_proof(
            request.vdf_info, request.vdf_proof, request.height, request.header_hash, field_vdf
        ):
            return None
        async with self.blockchain.compact_proof_lock:
            replaced = await self._replace_proof(request.vdf_info, request.vdf_proof, request.header_hash, field_vdf)
        if not replaced:
            self.log.error(f"Could not replace compact proof: {request.height}")
            return None
        self.log.info(f"Replaced compact proof at height {request.height}")
        msg = make_msg(
            ProtocolMessageTypes.new_compact_vdf,
            beacon_protocol.NewCompactVDF(request.height, request.header_hash, request.field_vdf, request.vdf_info),
        )
        if self._server is not None:
            await self.server.send_to_all([msg], NodeType.BEACON)

    async def new_compact_vdf(self, request: beacon_protocol.NewCompactVDF, peer: WSBpxConnection) -> None:
        is_fully_compactified = await self.block_store.is_fully_compactified(request.header_hash)
        if is_fully_compactified is None or is_fully_compactified:
            return None
        header_block = await self.blockchain.get_header_block_by_height(
            request.height, request.header_hash
        )
        if header_block is None:
            return None
        field_vdf = CompressibleVDFField(int(request.field_vdf))
        if await self._needs_compact_proof(request.vdf_info, header_block, field_vdf):
            peer_request = beacon_protocol.RequestCompactVDF(
                request.height, request.header_hash, request.field_vdf, request.vdf_info
            )
            response = await peer.call_api(BeaconAPI.request_compact_vdf, peer_request, timeout=10)
            if response is not None and isinstance(response, beacon_protocol.RespondCompactVDF):
                await self.add_compact_vdf(response, peer)

    async def request_compact_vdf(self, request: beacon_protocol.RequestCompactVDF, peer: WSBpxConnection) -> None:
        header_block = await self.blockchain.get_header_block_by_height(
            request.height, request.header_hash
        )
        if header_block is None:
            return None
        vdf_proof: Optional[VDFProof] = None
        field_vdf = CompressibleVDFField(int(request.field_vdf))
        if field_vdf == CompressibleVDFField.CC_EOS_VDF:
            for sub_slot in header_block.finished_sub_slots:
                if sub_slot.challenge_chain.challenge_chain_end_of_slot_vdf == request.vdf_info:
                    vdf_proof = sub_slot.proofs.challenge_chain_slot_proof
                    break
        if field_vdf == CompressibleVDFField.ICC_EOS_VDF:
            for sub_slot in header_block.finished_sub_slots:
                if (
                    sub_slot.infused_challenge_chain is not None
                    and sub_slot.infused_challenge_chain.infused_challenge_chain_end_of_slot_vdf == request.vdf_info
                ):
                    vdf_proof = sub_slot.proofs.infused_challenge_chain_slot_proof
                    break
        if (
            field_vdf == CompressibleVDFField.CC_SP_VDF
            and header_block.reward_chain_block.challenge_chain_sp_vdf == request.vdf_info
        ):
            vdf_proof = header_block.challenge_chain_sp_proof
        if (
            field_vdf == CompressibleVDFField.CC_IP_VDF
            and header_block.reward_chain_block.challenge_chain_ip_vdf == request.vdf_info
        ):
            vdf_proof = header_block.challenge_chain_ip_proof
        if vdf_proof is None or vdf_proof.witness_type > 0 or not vdf_proof.normalized_to_identity:
            self.log.error(f"{peer} requested compact vdf we don't have, height: {request.height}.")
            return None
        compact_vdf = beacon_protocol.RespondCompactVDF(
            request.height,
            request.header_hash,
            request.field_vdf,
            request.vdf_info,
            vdf_proof,
        )
        msg = make_msg(ProtocolMessageTypes.respond_compact_vdf, compact_vdf)
        await peer.send_message(msg)

    async def add_compact_vdf(self, request: beacon_protocol.RespondCompactVDF, peer: WSBpxConnection) -> None:
        field_vdf = CompressibleVDFField(int(request.field_vdf))
        if not await self._can_accept_compact_proof(
            request.vdf_info, request.vdf_proof, request.height, request.header_hash, field_vdf
        ):
            return None
        async with self.blockchain.compact_proof_lock:
            if self.blockchain.seen_compact_proofs(request.vdf_info, request.height):
                return None
            replaced = await self._replace_proof(request.vdf_info, request.vdf_proof, request.header_hash, field_vdf)
        if not replaced:
            self.log.error(f"Could not replace compact proof: {request.height}")
            return None
        msg = make_msg(
            ProtocolMessageTypes.new_compact_vdf,
            beacon_protocol.NewCompactVDF(request.height, request.header_hash, request.field_vdf, request.vdf_info),
        )
        if self._server is not None:
            await self.server.send_to_all([msg], NodeType.BEACON, peer.peer_node_id)

    async def broadcast_uncompact_blocks(
        self, uncompact_interval_scan: int, target_uncompact_proofs: int, sanitize_weight_proof_only: bool
    ) -> None:
        try:
            while not self._shut_down:
                while self.sync_store.get_sync_mode() or self.sync_store.get_long_sync():
                    if self._shut_down:
                        return None
                    await asyncio.sleep(30)

                broadcast_list: List[timelord_protocol.RequestCompactProofOfTime] = []

                self.log.info("Getting random heights for bluebox to compact")
                heights = await self.block_store.get_random_not_compactified(target_uncompact_proofs)
                self.log.info("Heights found for bluebox to compact: [%s]" % ", ".join(map(str, heights)))

                for h in heights:
                    headers = await self.blockchain.get_header_blocks_in_range(h, h)
                    records: Dict[bytes32, BlockRecord] = {}
                    if sanitize_weight_proof_only:
                        records = await self.blockchain.get_block_records_in_range(h, h)
                    for header in headers.values():
                        expected_header_hash = self.blockchain.height_to_hash(header.height)
                        if header.header_hash != expected_header_hash:
                            continue
                        if sanitize_weight_proof_only:
                            assert header.header_hash in records
                            record = records[header.header_hash]
                        for sub_slot in header.finished_sub_slots:
                            if (
                                sub_slot.proofs.challenge_chain_slot_proof.witness_type > 0
                                or not sub_slot.proofs.challenge_chain_slot_proof.normalized_to_identity
                            ):
                                broadcast_list.append(
                                    timelord_protocol.RequestCompactProofOfTime(
                                        sub_slot.challenge_chain.challenge_chain_end_of_slot_vdf,
                                        header.header_hash,
                                        header.height,
                                        uint8(CompressibleVDFField.CC_EOS_VDF),
                                    )
                                )
                            if sub_slot.proofs.infused_challenge_chain_slot_proof is not None and (
                                sub_slot.proofs.infused_challenge_chain_slot_proof.witness_type > 0
                                or not sub_slot.proofs.infused_challenge_chain_slot_proof.normalized_to_identity
                            ):
                                assert sub_slot.infused_challenge_chain is not None
                                broadcast_list.append(
                                    timelord_protocol.RequestCompactProofOfTime(
                                        sub_slot.infused_challenge_chain.infused_challenge_chain_end_of_slot_vdf,
                                        header.header_hash,
                                        header.height,
                                        uint8(CompressibleVDFField.ICC_EOS_VDF),
                                    )
                                )
                        # Running in 'sanitize_weight_proof_only' ignores CC_SP_VDF and CC_IP_VDF
                        # unless this is a challenge block.
                        if sanitize_weight_proof_only:
                            if not record.is_challenge_block(self.constants):
                                continue
                        if header.challenge_chain_sp_proof is not None and (
                            header.challenge_chain_sp_proof.witness_type > 0
                            or not header.challenge_chain_sp_proof.normalized_to_identity
                        ):
                            assert header.reward_chain_block.challenge_chain_sp_vdf is not None
                            broadcast_list.append(
                                timelord_protocol.RequestCompactProofOfTime(
                                    header.reward_chain_block.challenge_chain_sp_vdf,
                                    header.header_hash,
                                    header.height,
                                    uint8(CompressibleVDFField.CC_SP_VDF),
                                )
                            )

                        if (
                            header.challenge_chain_ip_proof.witness_type > 0
                            or not header.challenge_chain_ip_proof.normalized_to_identity
                        ):
                            broadcast_list.append(
                                timelord_protocol.RequestCompactProofOfTime(
                                    header.reward_chain_block.challenge_chain_ip_vdf,
                                    header.header_hash,
                                    header.height,
                                    uint8(CompressibleVDFField.CC_IP_VDF),
                                )
                            )

                if len(broadcast_list) > target_uncompact_proofs:
                    broadcast_list = broadcast_list[:target_uncompact_proofs]
                if self.sync_store.get_sync_mode() or self.sync_store.get_long_sync():
                    continue
                if self._server is not None:
                    self.log.info(f"Broadcasting {len(broadcast_list)} items to the bluebox")
                    msgs = []
                    for new_pot in broadcast_list:
                        msg = make_msg(ProtocolMessageTypes.request_compact_proof_of_time, new_pot)
                        msgs.append(msg)
                    await self.server.send_to_all(msgs, NodeType.TIMELORD)
                await asyncio.sleep(uncompact_interval_scan)
        except Exception as e:
            error_stack = traceback.format_exc()
            self.log.error(f"Exception in broadcast_uncompact_blocks: {e}")
            self.log.error(f"Exception Stack: {error_stack}")


async def node_next_block_check(
    peer: WSBpxConnection, potential_peek: uint32, blockchain: BlockchainInterface
) -> bool:
    block_response: Optional[Any] = await peer.call_api(
        BeaconAPI.request_block, beacon_protocol.RequestBlock(potential_peek)
    )
    if block_response is not None and isinstance(block_response, beacon_protocol.RespondBlock):
        peak = blockchain.get_peak()
        if peak is not None and block_response.block.prev_header_hash == peak.header_hash:
            return True
    return False
