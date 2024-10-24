from __future__ import annotations

import logging
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, List, Optional, Tuple, Type, TypeVar

from aiohttp import ClientConnectorError

from bpx.daemon.keychain_proxy import KeychainProxy, connect_to_keychain_and_validate
from bpx.rpc.farmer_rpc_client import FarmerRpcClient
from bpx.rpc.beacon_rpc_client import BeaconRpcClient
from bpx.rpc.harvester_rpc_client import HarvesterRpcClient
from bpx.rpc.rpc_client import RpcClient
from bpx.types.blockchain_format.sized_bytes import bytes32
from bpx.util.config import load_config
from bpx.util.default_root import DEFAULT_ROOT_PATH
from bpx.util.ints import uint16
from bpx.util.keychain import KeyData

NODE_TYPES: Dict[str, Type[RpcClient]] = {
    "farmer": FarmerRpcClient,
    "beacon": BeaconRpcClient,
    "harvester": HarvesterRpcClient,
}

node_config_section_names: Dict[Type[RpcClient], str] = {
    FarmerRpcClient: "farmer",
    BeaconRpcClient: "beacon",
    HarvesterRpcClient: "harvester",
}


_T_RpcClient = TypeVar("_T_RpcClient", bound=RpcClient)


async def validate_client_connection(
    rpc_client: RpcClient,
    node_type: str,
    rpc_port: int,
    root_path: Path,
    fingerprint: Optional[int],
) -> Optional[int]:
    try:
        await rpc_client.healthz()
    except ClientConnectorError:
        print(f"Connection error. Check if {node_type.replace('_', ' ')} rpc is running at {rpc_port}")
        print(f"This is normal if {node_type.replace('_', ' ')} is still starting up")
        rpc_client.close()
    await rpc_client.await_closed()  # if close is not already called this does nothing
    return fingerprint


@asynccontextmanager
async def get_any_service_client(
    client_type: Type[_T_RpcClient],
    rpc_port: Optional[int] = None,
    root_path: Path = DEFAULT_ROOT_PATH,
    fingerprint: Optional[int] = None,
) -> AsyncIterator[Tuple[Optional[_T_RpcClient], Dict[str, Any], Optional[int]]]:
    """
    Yields a tuple with a RpcClient for the applicable node type a dictionary of the node's configuration,
    and a fingerprint if applicable. However, if connecting to the node fails then we will return None for
    the RpcClient.
    """

    node_type = node_config_section_names.get(client_type)
    if node_type is None:
        # Click already checks this, so this should never happen
        raise ValueError(f"Invalid client type requested: {client_type.__name__}")
    # load variables from config file
    config = load_config(root_path, "config.yaml")
    self_hostname = config["self_hostname"]
    if rpc_port is None:
        rpc_port = config[node_type]["rpc_port"]
    # select node client type based on string
    node_client = await client_type.create(self_hostname, uint16(rpc_port), root_path, config)
    try:
        # check if we can connect to node, and if we can then validate
        # fingerprint access, otherwise return fingerprint and shutdown client
        fingerprint = await validate_client_connection(
            node_client, node_type, rpc_port, root_path, fingerprint
        )
        if node_client.session.closed:
            yield None, config, fingerprint
        else:
            yield node_client, config, fingerprint
    except Exception as e:  # this is only here to make the errors more user-friendly.
        print(f"Exception from '{node_type}' {e}:\n{traceback.format_exc()}")

    finally:
        node_client.close()  # this can run even if already closed, will just do nothing.
        await node_client.await_closed()
