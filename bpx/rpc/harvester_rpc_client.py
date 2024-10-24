from __future__ import annotations

from typing import Any, Dict, List

from bpx.rpc.rpc_client import RpcClient


class HarvesterRpcClient(RpcClient):
    async def get_plots(self) -> Dict[str, Any]:
        return await self.fetch("get_plots", {})

    async def refresh_plots(self) -> None:
        await self.fetch("refresh_plots", {})

    async def delete_plot(self, filename: str) -> bool:
        return (await self.fetch("delete_plot", {"filename": filename}))["success"]

    async def add_plot_directory(self, dirname: str) -> bool:
        return (await self.fetch("add_plot_directory", {"dirname": dirname}))["success"]

    async def get_plot_directories(self) -> List[str]:
        return (await self.fetch("get_plot_directories", {}))["directories"]

    async def remove_plot_directory(self, dirname: str) -> bool:
        return (await self.fetch("remove_plot_directory", {"dirname": dirname}))["success"]
    
    async def get_harvester_config(self) -> Dict[str, Any]:
        return await self.fetch("get_harvester_config", {})

    async def update_harvester_config(self, config: Dict[str, Any]) -> bool:
        response = await self.fetch("update_harvester_config", config)
        # TODO: casting due to lack of type checked deserialization
        result = cast(bool, response["success"])
        return result
