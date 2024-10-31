from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from bpx.consensus.block_record import BlockRecord
from bpx.beacon.signage_point import SignagePoint
from bpx.rpc.rpc_client import RpcClient
from bpx.types.blockchain_format.sized_bytes import bytes32
from bpx.types.end_of_slot_bundle import EndOfSubSlotBundle
from bpx.types.full_block import FullBlock
from bpx.types.unfinished_header_block import UnfinishedHeaderBlock
from bpx.util.byte_types import hexstr_to_bytes
from bpx.util.ints import uint32, uint64


class BeaconRpcClient(RpcClient):
    async def get_blockchain_state(self) -> Dict:
        response = await self.fetch("get_blockchain_state", {})
        if response["blockchain_state"]["peak"] is not None:
            response["blockchain_state"]["peak"] = BlockRecord.from_json_dict(response["blockchain_state"]["peak"])
        return response["blockchain_state"]

    async def get_block(self, header_hash) -> Optional[FullBlock]:
        try:
            response = await self.fetch("get_block", {"header_hash": header_hash.hex()})
        except Exception:
            return None
        return FullBlock.from_json_dict(response["block"])

    async def get_blocks(self, start: int, end: int, exclude_reorged: bool = False) -> List[FullBlock]:
        response = await self.fetch(
            "get_blocks", {"start": start, "end": end, "exclude_header_hash": True, "exclude_reorged": exclude_reorged}
        )
        return [FullBlock.from_json_dict(block) for block in response["blocks"]]

    async def get_block_record_by_height(self, height) -> Optional[BlockRecord]:
        try:
            response = await self.fetch("get_block_record_by_height", {"height": height})
        except Exception:
            return None
        return BlockRecord.from_json_dict(response["block_record"])

    async def get_block_record(self, header_hash) -> Optional[BlockRecord]:
        try:
            response = await self.fetch("get_block_record", {"header_hash": header_hash.hex()})
            if response["block_record"] is None:
                return None
        except Exception:
            return None
        return BlockRecord.from_json_dict(response["block_record"])

    async def get_unfinished_block_headers(self) -> List[UnfinishedHeaderBlock]:
        response = await self.fetch("get_unfinished_block_headers", {})
        return [UnfinishedHeaderBlock.from_json_dict(r) for r in response["headers"]]

    async def get_network_space(
        self, newer_block_header_hash: bytes32, older_block_header_hash: bytes32
    ) -> Optional[uint64]:
        try:
            network_space_bytes_estimate = await self.fetch(
                "get_network_space",
                {
                    "newer_block_header_hash": newer_block_header_hash.hex(),
                    "older_block_header_hash": older_block_header_hash.hex(),
                },
            )
        except Exception:
            return None
        return network_space_bytes_estimate["space"]

    async def get_block_records(self, start: int, end: int) -> List:
        try:
            response = await self.fetch("get_block_records", {"start": start, "end": end})
            if response["block_records"] is None:
                return []
        except Exception:
            return []
        # TODO: return block records
        return response["block_records"]

    async def get_recent_signage_point_or_eos(
        self, sp_hash: Optional[bytes32], challenge_hash: Optional[bytes32]
    ) -> Optional[Any]:
        try:
            if sp_hash is not None:
                assert challenge_hash is None
                response = await self.fetch("get_recent_signage_point_or_eos", {"sp_hash": sp_hash.hex()})
                return {
                    "signage_point": SignagePoint.from_json_dict(response["signage_point"]),
                    "time_received": response["time_received"],
                    "reverted": response["reverted"],
                }
            else:
                assert challenge_hash is not None
                response = await self.fetch("get_recent_signage_point_or_eos", {"challenge_hash": challenge_hash.hex()})
                return {
                    "eos": EndOfSubSlotBundle.from_json_dict(response["eos"]),
                    "time_received": response["time_received"],
                    "reverted": response["reverted"],
                }
        except Exception:
            return None
    
    async def get_coinbase(self) -> Dict[str, Any]:
        return await self.fetch("get_coinbase", {})

    async def set_coinbase(
        self,
        coinbase: str,
    ) -> Dict[str, Any]:
        return await self.fetch("set_coinbase", {"coinbase": coinbase})
