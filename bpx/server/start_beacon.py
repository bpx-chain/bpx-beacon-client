from __future__ import annotations

import logging
import os
import pathlib
import sys
from multiprocessing import freeze_support
from typing import Any, Dict, List, Optional, Tuple

from bpx.consensus.constants import ConsensusConstants
from bpx.consensus.default_constants import DEFAULT_CONSTANTS
from bpx.beacon.beacon import Beacon
from bpx.beacon.beacon_api import BeaconAPI
from bpx.rpc.beacon_rpc_api import BeaconRpcApi
from bpx.server.outbound_message import NodeType
from bpx.server.start_service import RpcInfo, Service, async_run
from bpx.util.bpx_logging import initialize_service_logging
from bpx.util.config import load_config, load_config_cli
from bpx.util.default_root import DEFAULT_ROOT_PATH
from bpx.util.ints import uint16
from bpx.util.task_timing import maybe_manage_task_instrumentation

# See: https://bugs.python.org/issue29288
"".encode("idna")

SERVICE_NAME = "beacon"
log = logging.getLogger(__name__)


def create_beacon_service(
    root_path: pathlib.Path,
    config: Dict[str, Any],
    consensus_constants: ConsensusConstants,
    connect_to_daemon: bool = True,
    override_capabilities: Optional[List[Tuple[uint16, str]]] = None,
) -> Service[Beacon]:
    service_config = config[SERVICE_NAME]

    beacon = Beacon(
        service_config,
        root_path=root_path,
        consensus_constants=consensus_constants,
    )
    api = BeaconAPI(beacon)

    upnp_list = []
    if service_config["enable_upnp"]:
        upnp_list = [service_config["port"]]
    network_id = service_config["selected_network"]
    rpc_info: Optional[RpcInfo] = None
    if service_config["start_rpc_server"]:
        rpc_info = (BeaconRpcApi, service_config["rpc_port"])
    return Service(
        root_path=root_path,
        config=config,
        node=api.beacon,
        peer_api=api,
        node_type=NodeType.BEACON,
        advertised_port=service_config["port"],
        service_name=SERVICE_NAME,
        upnp_ports=upnp_list,
        server_listen_ports=[service_config["port"]],
        on_connect_callback=beacon.on_connect,
        network_id=network_id,
        rpc_info=rpc_info,
        connect_to_daemon=connect_to_daemon,
        override_capabilities=override_capabilities,
    )


async def async_main(service_config: Dict[str, Any]) -> int:
    # TODO: refactor to avoid the double load
    config = load_config(DEFAULT_ROOT_PATH, "config.yaml")
    config[SERVICE_NAME] = service_config
    overrides = service_config["network_overrides"]["constants"][service_config["selected_network"]]
    updated_constants = DEFAULT_CONSTANTS.replace_str_to_bytes(**overrides)
    initialize_service_logging(service_name=SERVICE_NAME, config=config)
    service = create_beacon_service(DEFAULT_ROOT_PATH, config, updated_constants)
    await service.setup_process_global_state()
    await service.run()

    return 0


def main() -> int:
    freeze_support()

    with maybe_manage_task_instrumentation(enable=os.environ.get("BPX_INSTRUMENT_NODE") is not None):
        service_config = load_config_cli(DEFAULT_ROOT_PATH, "config.yaml", SERVICE_NAME)
        target_peer_count = service_config.get("target_peer_count", 80) - service_config.get(
            "target_outbound_peer_count", 8
        )
        if target_peer_count < 0:
            target_peer_count = None
        if not service_config.get("use_bpx_loop_policy", True):
            target_peer_count = None
        return async_run(coro=async_main(service_config), connection_limit=target_peer_count)


if __name__ == "__main__":
    sys.exit(main())
