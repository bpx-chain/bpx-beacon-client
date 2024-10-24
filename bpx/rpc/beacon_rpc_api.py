from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from bpx.consensus.block_record import BlockRecord
from bpx.consensus.pos_quality import UI_ACTUAL_SPACE_CONSTANT_FACTOR
from bpx.beacon.beacon import Beacon
from bpx.rpc.rpc_server import Endpoint, EndpointResult
from bpx.server.outbound_message import NodeType
from bpx.types.blockchain_format.sized_bytes import bytes32
from bpx.types.full_block import FullBlock
from bpx.types.unfinished_header_block import UnfinishedHeaderBlock
from bpx.util.byte_types import hexstr_to_bytes
from bpx.util.ints import uint32, uint64, uint128
from bpx.util.log_exceptions import log_exceptions
from bpx.util.ws_message import WsRpcMessage, create_payload_dict


class BeaconRpcApi:
    def __init__(self, service: Beacon) -> None:
        self.service = service
        self.service_name = "bpx_beacon"
        self.cached_blockchain_state: Optional[Dict[str, Any]] = None

    def get_routes(self) -> Dict[str, Endpoint]:
        return {
            # Blockchain
            "/get_blockchain_state": self.get_blockchain_state,
            "/get_block": self.get_block,
            "/get_blocks": self.get_blocks,
            "/get_block_count_metrics": self.get_block_count_metrics,
            "/get_block_record_by_height": self.get_block_record_by_height,
            "/get_block_record": self.get_block_record,
            "/get_block_records": self.get_block_records,
            "/get_unfinished_block_headers": self.get_unfinished_block_headers,
            "/get_network_space": self.get_network_space,
            "/get_network_info": self.get_network_info,
            "/get_recent_signage_point_or_eos": self.get_recent_signage_point_or_eos,
            "/get_coinbase": self.get_coinbase,
            "/set_coinbase": self.set_coinbase,
        }

    async def _state_changed(self, change: str, change_data: Optional[Dict[str, Any]] = None) -> List[WsRpcMessage]:
        if change_data is None:
            change_data = {}

        payloads = []
        if change == "new_peak" or change == "sync_mode":
            data = await self.get_blockchain_state({})
            assert data is not None
            payloads.append(
                create_payload_dict(
                    "get_blockchain_state",
                    data,
                    self.service_name,
                    "ui",
                )
            )
            payloads.append(
                create_payload_dict(
                    "get_blockchain_state",
                    data,
                    self.service_name,
                    "metrics",
                )
            )

        if change in ("block", "signage_point"):
            payloads.append(create_payload_dict(change, change_data, self.service_name, "metrics"))

        return payloads

    async def get_blockchain_state(self, _: Dict[str, Any]) -> EndpointResult:
        """
        Returns a summary of the node's view of the blockchain.
        """
        node_id = self.service.server.node_id.hex()
        if self.service.initialized is False:
            res = {
                "blockchain_state": {
                    "peak": None,
                    "genesis_challenge_initialized": self.service.initialized,
                    "sync": {
                        "sync_mode": False,
                        "synced": False,
                        "sync_tip_height": 0,
                        "sync_progress_height": 0,
                        "ec_synced": False,
                    },
                    "difficulty": 0,
                    "sub_slot_iters": 0,
                    "space": 0,
                    "node_id": node_id,
                    "ec_conn": False,
                },
            }
            return res
        peak: Optional[BlockRecord] = self.service.blockchain.get_peak()

        if peak is not None and peak.height > 0:
            difficulty = uint64(peak.weight - self.service.blockchain.block_record(peak.prev_hash).weight)
            sub_slot_iters = peak.sub_slot_iters
        else:
            difficulty = self.service.constants.DIFFICULTY_STARTING
            sub_slot_iters = self.service.constants.SUB_SLOT_ITERS_STARTING

        sync_mode: bool = self.service.sync_store.get_sync_mode() or self.service.sync_store.get_long_sync()

        sync_tip_height: Optional[uint32] = uint32(0)
        if sync_mode:
            target_peak = self.service.sync_store.target_peak
            if target_peak is not None:
                sync_tip_height = target_peak.height
            if peak is not None:
                sync_progress_height: uint32 = peak.height
                # Don't display we're syncing towards 0, instead show 'Syncing height/height'
                # until sync_store retrieves the correct number.
                if sync_tip_height == uint32(0):
                    sync_tip_height = peak.height
            else:
                sync_progress_height = uint32(0)
        else:
            sync_progress_height = uint32(0)

        if peak is not None and peak.height > 1:
            newer_block_hex = peak.header_hash.hex()
            # Average over the last day
            header_hash = self.service.blockchain.height_to_hash(uint32(max(1, peak.height - 4608)))
            assert header_hash is not None
            older_block_hex = header_hash.hex()
            space = await self.get_network_space(
                {"newer_block_header_hash": newer_block_hex, "older_block_header_hash": older_block_hex}
            )
        else:
            space = {"space": uint128(0)}

        if self.service.server is not None:
            is_connected = len(self.service.server.get_connections(NodeType.BEACON)) > 0
        else:
            is_connected = False
        synced = await self.service.synced() and is_connected
        
        ec_conn = self.service.execution_client.connected
        ec_synced = not self.service.execution_client.syncing and ec_conn and synced

        assert space is not None
        response = {
            "blockchain_state": {
                "peak": peak,
                "genesis_challenge_initialized": self.service.initialized,
                "sync": {
                    "sync_mode": sync_mode,
                    "synced": synced,
                    "sync_tip_height": sync_tip_height,
                    "sync_progress_height": sync_progress_height,
                    "ec_synced": ec_synced,
                },
                "difficulty": difficulty,
                "sub_slot_iters": sub_slot_iters,
                "space": space["space"],
                "node_id": node_id,
                "ec_conn": ec_conn,
            },
        }
        self.cached_blockchain_state = dict(response["blockchain_state"])
        return response

    async def get_network_info(self, _: Dict[str, Any]) -> EndpointResult:
        network_name = self.service.config["selected_network"]
        return {"network_name": network_name}

    async def get_recent_signage_point_or_eos(self, request: Dict[str, Any]) -> EndpointResult:
        if "sp_hash" not in request:
            challenge_hash: bytes32 = bytes32.from_hexstr(request["challenge_hash"])
            # This is the case of getting an end of slot
            eos_tuple = self.service.beacon_store.recent_eos.get(challenge_hash)
            if not eos_tuple:
                raise ValueError(f"Did not find eos {challenge_hash.hex()} in cache")
            eos, time_received = eos_tuple

            # If it's still in the beacon client store, it's not reverted
            if self.service.beacon_store.get_sub_slot(eos.challenge_chain.get_hash()):
                return {"eos": eos, "time_received": time_received, "reverted": False}

            # Otherwise we can backtrack from peak to find it in the blockchain
            curr: Optional[BlockRecord] = self.service.blockchain.get_peak()
            if curr is None:
                raise ValueError("No blocks in the chain")

            number_of_slots_searched = 0
            while number_of_slots_searched < 10:
                if curr.first_in_sub_slot:
                    assert curr.finished_challenge_slot_hashes is not None
                    if curr.finished_challenge_slot_hashes[-1] == eos.challenge_chain.get_hash():
                        # Found this slot in the blockchain
                        return {"eos": eos, "time_received": time_received, "reverted": False}
                    number_of_slots_searched += len(curr.finished_challenge_slot_hashes)
                curr = self.service.blockchain.try_block_record(curr.prev_hash)
                if curr is None:
                    # Got to the beginning of the blockchain without finding the slot
                    return {"eos": eos, "time_received": time_received, "reverted": True}

            # Backtracked through 10 slots but still did not find it
            return {"eos": eos, "time_received": time_received, "reverted": True}

        # Now we handle the case of getting a signage point
        sp_hash: bytes32 = bytes32.from_hexstr(request["sp_hash"])
        sp_tuple = self.service.beacon_store.recent_signage_points.get(sp_hash)
        if sp_tuple is None:
            raise ValueError(f"Did not find sp {sp_hash.hex()} in cache")

        sp, time_received = sp_tuple
        assert sp.rc_vdf is not None, "Not an EOS, the signage point rewards chain VDF must not be None"
        assert sp.cc_vdf is not None, "Not an EOS, the signage point challenge chain VDF must not be None"

        # If it's still in the beacon client store, it's not reverted
        if self.service.beacon_store.get_signage_point(sp_hash):
            return {"signage_point": sp, "time_received": time_received, "reverted": False}

        # Otherwise we can backtrack from peak to find it in the blockchain
        rc_challenge: bytes32 = sp.rc_vdf.challenge
        next_b: Optional[BlockRecord] = None
        curr_b_optional: Optional[BlockRecord] = self.service.blockchain.get_peak()
        assert curr_b_optional is not None
        curr_b: BlockRecord = curr_b_optional

        for _ in range(200):
            sp_total_iters = sp.cc_vdf.number_of_iterations + curr_b.ip_sub_slot_total_iters(self.service.constants)
            if curr_b.reward_infusion_new_challenge == rc_challenge:
                if next_b is None:
                    return {"signage_point": sp, "time_received": time_received, "reverted": False}
                next_b_total_iters = next_b.ip_sub_slot_total_iters(self.service.constants) + next_b.ip_iters(
                    self.service.constants
                )

                return {
                    "signage_point": sp,
                    "time_received": time_received,
                    "reverted": sp_total_iters > next_b_total_iters,
                }
            if curr_b.finished_reward_slot_hashes is not None:
                assert curr_b.finished_challenge_slot_hashes is not None
                for eos_rc in curr_b.finished_challenge_slot_hashes:
                    if eos_rc == rc_challenge:
                        if next_b is None:
                            return {"signage_point": sp, "time_received": time_received, "reverted": False}
                        next_b_total_iters = next_b.ip_sub_slot_total_iters(self.service.constants) + next_b.ip_iters(
                            self.service.constants
                        )
                        return {
                            "signage_point": sp,
                            "time_received": time_received,
                            "reverted": sp_total_iters > next_b_total_iters,
                        }
            next_b = curr_b
            curr_b_optional = self.service.blockchain.try_block_record(curr_b.prev_hash)
            if curr_b_optional is None:
                break
            curr_b = curr_b_optional

        return {"signage_point": sp, "time_received": time_received, "reverted": True}

    async def get_block(self, request: Dict[str, Any]) -> EndpointResult:
        if "header_hash" not in request:
            raise ValueError("No header_hash in request")
        header_hash = bytes32.from_hexstr(request["header_hash"])

        block: Optional[FullBlock] = await self.service.block_store.get_full_block(header_hash)
        if block is None:
            raise ValueError(f"Block {header_hash.hex()} not found")

        return {"block": block}

    async def get_blocks(self, request: Dict[str, Any]) -> EndpointResult:
        if "start" not in request:
            raise ValueError("No start in request")
        if "end" not in request:
            raise ValueError("No end in request")
        exclude_hh = False
        if "exclude_header_hash" in request:
            exclude_hh = request["exclude_header_hash"]
        exclude_reorged = False
        if "exclude_reorged" in request:
            exclude_reorged = request["exclude_reorged"]

        start = int(request["start"])
        end = int(request["end"])
        block_range = []
        for a in range(start, end):
            block_range.append(uint32(a))
        blocks: List[FullBlock] = await self.service.block_store.get_full_blocks_at(block_range)
        json_blocks = []
        for block in blocks:
            hh: bytes32 = block.header_hash
            if exclude_reorged and self.service.blockchain.height_to_hash(block.height) != hh:
                # Don't include forked (reorged) blocks
                continue
            json = block.to_json_dict()
            if not exclude_hh:
                json["header_hash"] = hh.hex()
            json_blocks.append(json)
        return {"blocks": json_blocks}

    async def get_block_count_metrics(self, _: Dict[str, Any]) -> EndpointResult:
        compact_blocks = 0
        uncompact_blocks = 0
        with log_exceptions(self.service.log, consume=True):
            compact_blocks = await self.service.block_store.count_compactified_blocks()
            uncompact_blocks = await self.service.block_store.count_uncompactified_blocks()

        return {
            "metrics": {
                "compact_blocks": compact_blocks,
                "uncompact_blocks": uncompact_blocks,
            }
        }

    async def get_block_records(self, request: Dict[str, Any]) -> EndpointResult:
        if "start" not in request:
            raise ValueError("No start in request")
        if "end" not in request:
            raise ValueError("No end in request")

        start = int(request["start"])
        end = int(request["end"])
        records = []
        peak_height = self.service.blockchain.get_peak_height()
        if peak_height is None:
            raise ValueError("Peak is None")

        for a in range(start, end):
            if peak_height < uint32(a):
                self.service.log.warning("requested block is higher than known peak ")
                break
            header_hash: Optional[bytes32] = self.service.blockchain.height_to_hash(uint32(a))
            if header_hash is None:
                raise ValueError(f"Height not in blockchain: {a}")
            record: Optional[BlockRecord] = self.service.blockchain.try_block_record(header_hash)
            if record is None:
                # Fetch from DB
                record = await self.service.blockchain.block_store.get_block_record(header_hash)
            if record is None:
                raise ValueError(f"Block {header_hash.hex()} does not exist")

            records.append(record)
        return {"block_records": records}

    async def get_block_record_by_height(self, request: Dict[str, Any]) -> EndpointResult:
        if "height" not in request:
            raise ValueError("No height in request")
        height = request["height"]
        header_height = uint32(int(height))
        peak_height = self.service.blockchain.get_peak_height()
        if peak_height is None or header_height > peak_height:
            raise ValueError(f"Block height {height} not found in chain")
        header_hash: Optional[bytes32] = self.service.blockchain.height_to_hash(header_height)
        if header_hash is None:
            raise ValueError(f"Block hash {height} not found in chain")
        record: Optional[BlockRecord] = self.service.blockchain.try_block_record(header_hash)
        if record is None:
            # Fetch from DB
            record = await self.service.blockchain.block_store.get_block_record(header_hash)
        if record is None:
            raise ValueError(f"Block {header_hash} does not exist")
        return {"block_record": record}

    async def get_block_record(self, request: Dict[str, Any]) -> EndpointResult:
        if "header_hash" not in request:
            raise ValueError("header_hash not in request")
        header_hash_str = request["header_hash"]
        header_hash = bytes32.from_hexstr(header_hash_str)
        record: Optional[BlockRecord] = self.service.blockchain.try_block_record(header_hash)
        if record is None:
            # Fetch from DB
            record = await self.service.blockchain.block_store.get_block_record(header_hash)
        if record is None:
            raise ValueError(f"Block {header_hash.hex()} does not exist")

        return {"block_record": record}

    async def get_unfinished_block_headers(self, _request: Dict[str, Any]) -> EndpointResult:
        peak: Optional[BlockRecord] = self.service.blockchain.get_peak()
        if peak is None:
            return {"headers": []}

        response_headers: List[UnfinishedHeaderBlock] = []
        for ub_height, block, _ in (self.service.beacon_store.get_unfinished_blocks()).values():
            if ub_height == peak.height:
                unfinished_header_block = UnfinishedHeaderBlock(
                    block.finished_sub_slots,
                    block.reward_chain_block,
                    block.challenge_chain_sp_proof,
                    block.reward_chain_sp_proof,
                    block.foliage,
                    block.foliage_transaction_block,
                    block.execution_payload,
                )
                response_headers.append(unfinished_header_block)
        return {"headers": response_headers}

    async def get_network_space(self, request: Dict[str, Any]) -> EndpointResult:
        """
        Retrieves an estimate of total space validating the chain
        between two block header hashes.
        """
        if "newer_block_header_hash" not in request or "older_block_header_hash" not in request:
            raise ValueError("Invalid request. newer_block_header_hash and older_block_header_hash required")
        newer_block_hex = request["newer_block_header_hash"]
        older_block_hex = request["older_block_header_hash"]

        if newer_block_hex == older_block_hex:
            raise ValueError("New and old must not be the same")

        newer_block_bytes = bytes32.from_hexstr(newer_block_hex)
        older_block_bytes = bytes32.from_hexstr(older_block_hex)

        newer_block = await self.service.block_store.get_block_record(newer_block_bytes)
        if newer_block is None:
            # It's possible that the peak block has not yet been committed to the DB, so as a fallback, check memory
            try:
                newer_block = self.service.blockchain.block_record(newer_block_bytes)
            except KeyError:
                raise ValueError(f"Newer block {newer_block_hex} not found")
        older_block = await self.service.block_store.get_block_record(older_block_bytes)
        if older_block is None:
            raise ValueError(f"Older block {older_block_hex} not found")
        delta_weight = newer_block.weight - older_block.weight

        delta_iters = newer_block.total_iters - older_block.total_iters
        weight_div_iters = delta_weight / delta_iters
        additional_difficulty_constant = self.service.constants.DIFFICULTY_CONSTANT_FACTOR
        eligible_plots_filter_multiplier = 2**self.service.constants.NUMBER_ZERO_BITS_PLOT_FILTER
        network_space_bytes_estimate = (
            UI_ACTUAL_SPACE_CONSTANT_FACTOR
            * weight_div_iters
            * additional_difficulty_constant
            * eligible_plots_filter_multiplier
        )
        return {"space": uint128(int(network_space_bytes_estimate))}
    
    async def get_coinbase(self, request: Dict[str, Any]) -> EndpointResult:
        result = await self.service.get_coinbase()
        return {"coinbase": result}

    async def set_coinbase(self, request: Dict[str, Any]) -> EndpointResult:
        self.service.set_coinbase(request["coinbase"])
        return {}
