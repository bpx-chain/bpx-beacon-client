from __future__ import annotations

import dataclasses
import operator
from typing import Any, Callable, Dict, List, Optional

from typing_extensions import Protocol

from bpx.farmer.farmer import Farmer
from bpx.plot_sync.receiver import Receiver
from bpx.protocols.harvester_protocol import Plot
from bpx.rpc.rpc_server import Endpoint, EndpointResult
from bpx.types.blockchain_format.sized_bytes import bytes32
from bpx.util.byte_types import hexstr_to_bytes
from bpx.util.ints import uint32
from bpx.util.paginator import Paginator
from bpx.util.streamable import Streamable, streamable
from bpx.util.ws_message import WsRpcMessage, create_payload_dict


class PaginatedRequestData(Protocol):
    @property
    def node_id(self) -> bytes32:
        pass

    @property
    def page(self) -> uint32:
        pass

    @property
    def page_size(self) -> uint32:
        pass


@streamable
@dataclasses.dataclass(frozen=True)
class FilterItem(Streamable):
    key: str
    value: Optional[str]


@streamable
@dataclasses.dataclass(frozen=True)
class PlotInfoRequestData(Streamable):
    node_id: bytes32
    page: uint32
    page_size: uint32
    filter: List[FilterItem] = dataclasses.field(default_factory=list)
    sort_key: str = "filename"
    reverse: bool = False


@streamable
@dataclasses.dataclass(frozen=True)
class PlotPathRequestData(Streamable):
    node_id: bytes32
    page: uint32
    page_size: uint32
    filter: List[str] = dataclasses.field(default_factory=list)
    reverse: bool = False


def paginated_plot_request(source: List[Any], request: PaginatedRequestData) -> Dict[str, object]:
    paginator: Paginator = Paginator(source, request.page_size)
    return {
        "node_id": request.node_id.hex(),
        "page": request.page,
        "page_count": paginator.page_count(),
        "total_count": len(source),
        "plots": paginator.get_page(request.page),
    }


def plot_matches_filter(plot: Plot, filter_item: FilterItem) -> bool:
    plot_attribute = getattr(plot, filter_item.key)
    if filter_item.value is None:
        return plot_attribute is None
    else:
        return filter_item.value in str(plot_attribute)


class FarmerRpcApi:
    def __init__(self, farmer: Farmer):
        self.service = farmer
        self.service_name = "bpx_farmer"

    def get_routes(self) -> Dict[str, Endpoint]:
        return {
            "/get_signage_point": self.get_signage_point,
            "/get_signage_points": self.get_signage_points,
            "/get_harvesters": self.get_harvesters,
            "/get_harvesters_summary": self.get_harvesters_summary,
            "/get_harvester_plots_valid": self.get_harvester_plots_valid,
            "/get_harvester_plots_invalid": self.get_harvester_plots_invalid,
            "/get_harvester_plots_keys_missing": self.get_harvester_plots_keys_missing,
            "/get_harvester_plots_duplicates": self.get_harvester_plots_duplicates,
        }

    async def _state_changed(self, change: str, change_data: Optional[Dict[str, Any]]) -> List[WsRpcMessage]:
        payloads = []

        if change_data is None:
            # TODO: maybe something better?
            pass
        elif change == "new_signage_point":
            sp_hash = change_data["sp_hash"]
            data = await self.get_signage_point({"sp_hash": sp_hash.hex()})
            payloads.append(
                create_payload_dict(
                    "new_signage_point",
                    data,
                    self.service_name,
                    "ui",
                )
            )
            payloads.append(
                create_payload_dict(
                    "new_signage_point",
                    data,
                    self.service_name,
                    "metrics",
                )
            )
        elif change == "new_farming_info":
            payloads.append(
                create_payload_dict(
                    "new_farming_info",
                    change_data,
                    self.service_name,
                    "ui",
                )
            )
            payloads.append(
                create_payload_dict(
                    "new_farming_info",
                    change_data,
                    self.service_name,
                    "metrics",
                )
            )
        elif change == "harvester_update":
            payloads.append(
                create_payload_dict(
                    "harvester_update",
                    change_data,
                    self.service_name,
                    "ui",
                )
            )
            payloads.append(
                create_payload_dict(
                    "harvester_update",
                    change_data,
                    self.service_name,
                    "metrics",
                )
            )
        elif change == "harvester_removed":
            payloads.append(
                create_payload_dict(
                    "harvester_removed",
                    change_data,
                    self.service_name,
                    "ui",
                )
            )
            payloads.append(
                create_payload_dict(
                    "harvester_removed",
                    change_data,
                    self.service_name,
                    "metrics",
                )
            )
        elif change == "proof":
            payloads.append(
                create_payload_dict(
                    "proof",
                    change_data,
                    self.service_name,
                    "metrics",
                )
            )
        elif change == "add_connection":
            payloads.append(
                create_payload_dict(
                    "add_connection",
                    change_data,
                    self.service_name,
                    "metrics",
                )
            )
        elif change == "close_connection":
            payloads.append(
                create_payload_dict(
                    "close_connection",
                    change_data,
                    self.service_name,
                    "metrics",
                )
            )

        return payloads

    async def get_signage_point(self, request: Dict[str, Any]) -> EndpointResult:
        sp_hash = hexstr_to_bytes(request["sp_hash"])
        for _, sps in self.service.sps.items():
            for sp in sps:
                if sp.challenge_chain_sp == sp_hash:
                    pospaces = self.service.proofs_of_space.get(sp.challenge_chain_sp, [])
                    return {
                        "signage_point": {
                            "challenge_hash": sp.challenge_hash,
                            "challenge_chain_sp": sp.challenge_chain_sp,
                            "reward_chain_sp": sp.reward_chain_sp,
                            "difficulty": sp.difficulty,
                            "sub_slot_iters": sp.sub_slot_iters,
                            "signage_point_index": sp.signage_point_index,
                        },
                        "proofs": pospaces,
                    }
        raise ValueError(f"Signage point {sp_hash.hex()} not found")

    async def get_signage_points(self, _: Dict[str, Any]) -> EndpointResult:
        result: List[Dict[str, Any]] = []
        for sps in self.service.sps.values():
            for sp in sps:
                pospaces = self.service.proofs_of_space.get(sp.challenge_chain_sp, [])
                result.append(
                    {
                        "signage_point": {
                            "challenge_hash": sp.challenge_hash,
                            "challenge_chain_sp": sp.challenge_chain_sp,
                            "reward_chain_sp": sp.reward_chain_sp,
                            "difficulty": sp.difficulty,
                            "sub_slot_iters": sp.sub_slot_iters,
                            "signage_point_index": sp.signage_point_index,
                        },
                        "proofs": pospaces,
                    }
                )
        return {"signage_points": result}

    async def get_harvesters(self, request: Dict[str, Any]) -> EndpointResult:
        return await self.service.get_harvesters(False)

    async def get_harvesters_summary(self, _: Dict[str, object]) -> EndpointResult:
        return await self.service.get_harvesters(True)

    async def get_harvester_plots_valid(self, request_dict: Dict[str, object]) -> EndpointResult:
        # TODO: Consider having a extra List[PlotInfo] in Receiver to avoid rebuilding the list for each call
        request = PlotInfoRequestData.from_json_dict(request_dict)
        plot_list = list(self.service.get_receiver(request.node_id).plots().values())
        # Apply filter
        plot_list = [
            plot for plot in plot_list if all(plot_matches_filter(plot, filter_item) for filter_item in request.filter)
        ]
        restricted_sort_keys: List[str] = ["pool_contract_puzzle_hash", "pool_public_key", "plot_public_key"]
        # Apply sort_key and reverse if sort_key is not restricted
        if request.sort_key in restricted_sort_keys:
            raise KeyError(f"Can't sort by optional attributes: {restricted_sort_keys}")
        # Sort by plot_id also by default since its unique
        plot_list = sorted(plot_list, key=operator.attrgetter(request.sort_key, "plot_id"), reverse=request.reverse)
        return paginated_plot_request(plot_list, request)

    def paginated_plot_path_request(
        self, source_func: Callable[[Receiver], List[str]], request_dict: Dict[str, object]
    ) -> Dict[str, object]:
        request: PlotPathRequestData = PlotPathRequestData.from_json_dict(request_dict)
        receiver = self.service.get_receiver(request.node_id)
        source = source_func(receiver)
        # Apply filter
        source = [plot for plot in source if all(filter_item in plot for filter_item in request.filter)]
        # Apply reverse
        source = sorted(source, reverse=request.reverse)
        return paginated_plot_request(source, request)

    async def get_harvester_plots_invalid(self, request_dict: Dict[str, object]) -> EndpointResult:
        return self.paginated_plot_path_request(Receiver.invalid, request_dict)

    async def get_harvester_plots_keys_missing(self, request_dict: Dict[str, object]) -> EndpointResult:
        return self.paginated_plot_path_request(Receiver.keys_missing, request_dict)

    async def get_harvester_plots_duplicates(self, request_dict: Dict[str, object]) -> EndpointResult:
        return self.paginated_plot_path_request(Receiver.duplicates, request_dict)
