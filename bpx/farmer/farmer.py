from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
import contextlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp
from blspy import AugSchemeMPL, G1Element, G2Element, PrivateKey

from bpx.consensus.constants import ConsensusConstants
from bpx.daemon.keychain_proxy import KeychainProxy, connect_to_keychain_and_validate, wrap_local_keychain
from bpx.plot_sync.delta import Delta
from bpx.plot_sync.receiver import Receiver
from bpx.protocols import farmer_protocol, harvester_protocol
from bpx.protocols.protocol_message_types import ProtocolMessageTypes
from bpx.rpc.rpc_server import StateChangedProtocol, default_get_connections
from bpx.server.outbound_message import NodeType, make_msg
from bpx.server.server import BpxServer, ssl_context_for_root
from bpx.server.ws_connection import WSBpxConnection
from bpx.ssl.create_ssl import get_mozilla_ca_crt
from bpx.types.blockchain_format.proof_of_space import ProofOfSpace
from bpx.types.blockchain_format.sized_bytes import bytes32
from bpx.util.byte_types import hexstr_to_bytes
from bpx.util.config import load_config
from bpx.util.errors import KeychainProxyConnectionFailure
from bpx.util.hash import std_hash
from bpx.util.ints import uint8, uint16, uint64
from bpx.util.keychain import Keychain
from bpx.util.derive_keys import (
    master_sk_to_farmer_sk,
    master_sk_to_pool_sk,
)

log = logging.getLogger(__name__)

"""
HARVESTER PROTOCOL (FARMER <-> HARVESTER)
"""


class Farmer:
    def __init__(
        self,
        root_path: Path,
        farmer_config: Dict[str, Any],
        consensus_constants: ConsensusConstants,
        local_keychain: Optional[Keychain] = None,
    ):
        self.keychain_proxy: Optional[KeychainProxy] = None
        self.local_keychain = local_keychain
        self._root_path = root_path
        self.config = farmer_config
        # Keep track of all sps, keyed on challenge chain signage point hash
        self.sps: Dict[bytes32, List[farmer_protocol.NewSignagePoint]] = {}

        # Keep track of harvester plot identifier (str), target sp index, and PoSpace for each challenge
        self.proofs_of_space: Dict[bytes32, List[Tuple[str, ProofOfSpace]]] = {}

        # Quality string to plot identifier and challenge_hash, for use with harvester.RequestSignatures
        self.quality_str_to_identifiers: Dict[bytes32, Tuple[str, bytes32, bytes32, bytes32]] = {}

        # number of responses to each signage point
        self.number_of_responses: Dict[bytes32, int] = {}

        # A dictionary of keys to time added. These keys refer to keys in the above 4 dictionaries. This is used
        # to periodically clear the memory
        self.cache_add_time: Dict[bytes32, uint64] = {}

        self.plot_sync_receivers: Dict[bytes32, Receiver] = {}

        self.cache_clear_task: Optional[asyncio.Task[None]] = None
        self.refresh_keys_task: Optional[asyncio.Task[None]] = None
        self.constants = consensus_constants
        self._shut_down = False
        self.server: Any = None
        self.state_changed_callback: Optional[StateChangedProtocol] = None
        self.log = log

        self.started = False
        self.harvester_handshake_task: Optional[asyncio.Task[None]] = None

        self.all_root_sks: List[PrivateKey] = []

    def get_connections(self, request_node_type: Optional[NodeType]) -> List[Dict[str, Any]]:
        return default_get_connections(server=self.server, request_node_type=request_node_type)

    async def ensure_keychain_proxy(self) -> KeychainProxy:
        if self.keychain_proxy is None:
            if self.local_keychain:
                self.keychain_proxy = wrap_local_keychain(self.local_keychain, log=self.log)
            else:
                self.keychain_proxy = await connect_to_keychain_and_validate(self._root_path, self.log)
                if not self.keychain_proxy:
                    raise KeychainProxyConnectionFailure()
        return self.keychain_proxy

    async def get_all_private_keys(self) -> List[Tuple[PrivateKey, bytes]]:
        keychain_proxy = await self.ensure_keychain_proxy()
        return await keychain_proxy.get_all_private_keys()

    async def setup_keys(self) -> bool:
        no_keys_error_str = "No keys exist. Please run 'bpx keys generate' or open the UI."
        try:
            self.all_root_sks = [sk for sk, _ in await self.get_all_private_keys()]
        except KeychainProxyConnectionFailure:
            return False

        self._private_keys = [master_sk_to_farmer_sk(sk) for sk in self.all_root_sks] + [
            master_sk_to_pool_sk(sk) for sk in self.all_root_sks
        ]

        if len(self.get_public_keys()) == 0:
            log.warning(no_keys_error_str)
            return False

        config = load_config(self._root_path, "config.yaml")
        self.config = config["farmer"]

        self.pool_public_keys = [G1Element.from_bytes(bytes.fromhex(pk)) for pk in self.config["pool_public_keys"]]

        self.pool_sks_map = {bytes(key.get_g1()): key for key in self.get_private_keys()}

        if len(self.pool_sks_map) == 0:
            log.warning(no_keys_error_str)
            return False

        return True

    async def _start(self) -> None:
        async def start_task() -> None:
            # `Farmer.setup_keys` returns `False` if there are no keys setup yet. In this case we just try until it
            # succeeds or until we need to shut down.
            while not self._shut_down:
                if await self.setup_keys():
                    self.cache_clear_task = asyncio.create_task(self._periodically_clear_cache_and_refresh_task())
                    self.refresh_keys_task = asyncio.create_task(self._refresh_keys())
                    log.debug("start_task: initialized")
                    self.started = True
                    return
                await asyncio.sleep(1)

        asyncio.create_task(start_task())

    def _close(self) -> None:
        cancel_task_safe(task=self.refresh_keys_task, log=log)
        self._shut_down = True

    async def _await_closed(self, shutting_down: bool = True) -> None:
        if self.cache_clear_task is not None:
            await self.cache_clear_task
        if self.refresh_keys_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self.refresh_keys_task
        if shutting_down and self.keychain_proxy is not None:
            proxy = self.keychain_proxy
            self.keychain_proxy = None
            await proxy.close()
            await asyncio.sleep(0.5)  # https://docs.aiohttp.org/en/stable/client_advanced.html#graceful-shutdown
        self.started = False

    def _set_state_changed_callback(self, callback: StateChangedProtocol) -> None:
        self.state_changed_callback = callback

    async def on_connect(self, peer: WSBpxConnection) -> None:
        self.state_changed("add_connection", {})

        async def handshake_task() -> None:
            # Wait until the task in `Farmer._start` is done so that we have keys available for the handshake. Bail out
            # early if we need to shut down or if the harvester is not longer connected.
            while not self.started and not self._shut_down and peer in self.server.get_connections():
                await asyncio.sleep(1)

            if self._shut_down:
                log.debug("handshake_task: shutdown")
                self.harvester_handshake_task = None
                return

            if peer not in self.server.get_connections():
                log.debug("handshake_task: disconnected")
                self.harvester_handshake_task = None
                return

            # Sends a handshake to the harvester
            handshake = harvester_protocol.HarvesterHandshake(
                self.get_public_keys(),
                self.pool_public_keys,
            )
            msg = make_msg(ProtocolMessageTypes.harvester_handshake, handshake)
            await peer.send_message(msg)
            self.harvester_handshake_task = None

        if peer.connection_type is NodeType.HARVESTER:
            self.plot_sync_receivers[peer.peer_node_id] = Receiver(peer, self.plot_sync_callback)
            self.harvester_handshake_task = asyncio.create_task(handshake_task())            

    def set_server(self, server: BpxServer) -> None:
        self.server = server

    def state_changed(self, change: str, data: Dict[str, Any]) -> None:
        if self.state_changed_callback is not None:
            self.state_changed_callback(change, data)

    def on_disconnect(self, connection: WSBpxConnection) -> None:
        self.log.info(f"peer disconnected {connection.get_peer_logging()}")
        self.state_changed("close_connection", {})
        if connection.connection_type is NodeType.HARVESTER:
            del self.plot_sync_receivers[connection.peer_node_id]
            self.state_changed("harvester_removed", {"node_id": connection.peer_node_id})

    async def plot_sync_callback(self, peer_id: bytes32, delta: Optional[Delta]) -> None:
        log.debug(f"plot_sync_callback: peer_id {peer_id}, delta {delta}")
        receiver: Receiver = self.plot_sync_receivers[peer_id]
        self.state_changed("harvester_update", receiver.to_dict(True))

    def get_public_keys(self) -> List[G1Element]:
        return [child_sk.get_g1() for child_sk in self._private_keys]

    def get_private_keys(self) -> List[PrivateKey]:
        return self._private_keys

    async def get_harvesters(self, counts_only: bool = False) -> Dict[str, Any]:
        harvesters: List[Dict[str, Any]] = []
        for connection in self.server.get_connections(NodeType.HARVESTER):
            self.log.debug(f"get_harvesters host: {connection.peer_host}, node_id: {connection.peer_node_id}")
            receiver = self.plot_sync_receivers.get(connection.peer_node_id)
            if receiver is not None:
                harvesters.append(receiver.to_dict(counts_only))
            else:
                self.log.debug(
                    f"get_harvesters invalid peer: {connection.peer_host}, node_id: {connection.peer_node_id}"
                )

        return {"harvesters": harvesters}

    def get_receiver(self, node_id: bytes32) -> Receiver:
        receiver: Optional[Receiver] = self.plot_sync_receivers.get(node_id)
        if receiver is None:
            raise KeyError(f"Receiver missing for {node_id}")
        return receiver

    async def _periodically_clear_cache_and_refresh_task(self) -> None:
        time_slept = 0
        refresh_slept = 0
        while not self._shut_down:
            try:
                if time_slept > self.constants.SUB_SLOT_TIME_TARGET:
                    now = time.time()
                    removed_keys: List[bytes32] = []
                    for key, add_time in self.cache_add_time.items():
                        if now - float(add_time) > self.constants.SUB_SLOT_TIME_TARGET * 3:
                            self.sps.pop(key, None)
                            self.proofs_of_space.pop(key, None)
                            self.quality_str_to_identifiers.pop(key, None)
                            self.number_of_responses.pop(key, None)
                            removed_keys.append(key)
                    for key in removed_keys:
                        self.cache_add_time.pop(key, None)
                    time_slept = 0
                    log.debug(
                        f"Cleared farmer cache. Num sps: {len(self.sps)} {len(self.proofs_of_space)} "
                        f"{len(self.quality_str_to_identifiers)} {len(self.number_of_responses)}"
                    )
                time_slept += 1
                refresh_slept += 1
                # Periodically refresh GUI to show the correct download/upload rate.
                if refresh_slept >= 30:
                    self.state_changed("add_connection", {})
                    refresh_slept = 0

            except Exception:
                log.error(f"_periodically_clear_cache_and_refresh_task failed: {traceback.format_exc()}")

            await asyncio.sleep(1)
    
    async def _refresh_keys(self) -> None:
        try:
            while True:
                await asyncio.sleep(30)
                await self.setup_keys()
        except asyncio.CancelledError:
            pass
