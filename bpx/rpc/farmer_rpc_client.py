from __future__ import annotations

from typing import Any, Dict, List, Optional, cast

from bpx.rpc.farmer_rpc_api import PlotInfoRequestData, PlotPathRequestData
from bpx.rpc.rpc_client import RpcClient
from bpx.types.blockchain_format.sized_bytes import bytes32
from bpx.util.misc import dataclass_to_json_dict


class FarmerRpcClient(RpcClient):
    async def get_signage_point(self, sp_hash: bytes32) -> Optional[Dict[str, Any]]:
        try:
            return await self.fetch("get_signage_point", {"sp_hash": sp_hash.hex()})
        except ValueError:
            return None

    async def get_signage_points(self) -> List[Dict[str, Any]]:
        return cast(List[Dict[str, Any]], (await self.fetch("get_signage_points", {}))["signage_points"])

    async def get_harvesters(self) -> Dict[str, Any]:
        return await self.fetch("get_harvesters", {})

    async def get_harvesters_summary(self) -> Dict[str, object]:
        return await self.fetch("get_harvesters_summary", {})

    async def get_harvester_plots_valid(self, request: PlotInfoRequestData) -> Dict[str, Any]:
        return await self.fetch("get_harvester_plots_valid", dataclass_to_json_dict(request))

    async def get_harvester_plots_invalid(self, request: PlotPathRequestData) -> Dict[str, Any]:
        return await self.fetch("get_harvester_plots_invalid", dataclass_to_json_dict(request))

    async def get_harvester_plots_keys_missing(self, request: PlotPathRequestData) -> Dict[str, Any]:
        return await self.fetch("get_harvester_plots_keys_missing", dataclass_to_json_dict(request))

    async def get_harvester_plots_duplicates(self, request: PlotPathRequestData) -> Dict[str, Any]:
        return await self.fetch("get_harvester_plots_duplicates", dataclass_to_json_dict(request))
 