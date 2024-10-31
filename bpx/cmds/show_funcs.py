from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from bpx.rpc.beacon_rpc_client import BeaconRpcClient


async def print_blockchain_state(node_client: BeaconRpcClient, config: Dict[str, Any]) -> bool:
    import time

    from bpx.consensus.block_record import BlockRecord
    from bpx.util.ints import uint64
    from bpx.util.misc import format_bytes

    blockchain_state = await node_client.get_blockchain_state()
    if blockchain_state is None:
        print("There is no blockchain found yet. Try again shortly")
        return True
    peak: Optional[BlockRecord] = blockchain_state["peak"]
    node_id = blockchain_state["node_id"]
    difficulty = blockchain_state["difficulty"]
    sub_slot_iters = blockchain_state["sub_slot_iters"]
    synced = blockchain_state["sync"]["synced"]
    sync_mode = blockchain_state["sync"]["sync_mode"]
    ec_conn = blockchain_state["ec_conn"]
    ec_synced = blockchain_state["sync"]["ec_synced"]
    num_blocks: int = 10
    network_name = config["selected_network"]
    genesis_challenge = config["farmer"]["network_overrides"]["constants"][network_name]["GENESIS_CHALLENGE"]
    beacon_port = config["beacon"]["port"]
    beacon_rpc_port = config["beacon"]["rpc_port"]

    print(f"Network: {network_name}    Port: {beacon_port}   RPC Port: {beacon_rpc_port}")
    print(f"Node ID: {node_id}")
    print(f"Genesis Challenge: {genesis_challenge}")

    if synced:
        print("\nBeacon Client Status: Synced")
    elif peak is not None and sync_mode:
        sync_max_block = blockchain_state["sync"]["sync_tip_height"]
        sync_current_block = blockchain_state["sync"]["sync_progress_height"]
        print(
            f"\nBeacon Client Status: Syncing {sync_current_block}/{sync_max_block} "
            f"({sync_max_block - sync_current_block} behind)."
        )
    elif peak is not None:
        print(f"\nBeacon Client Status: Not Synced. Peak height: {peak.height}")
    else:
        print("\nSearching for an initial chain\n")
        print("You may be able to expedite with 'bpx peer beacon -a host:port' using a known node.\n")

    if peak is not None:
        if not ec_conn:
            print("Execution Client Status: Offline")
        elif not ec_synced:
            print("Execution Client Status: Syncing")
        else:
            print("Execution Client Status: Synced")

        print(f"\nPeak: Height: {peak.height}\n      Hash: {peak.header_hash}")
        
        if peak.is_transaction_block:
            peak_time = peak.timestamp
        else:
            peak_hash = peak.header_hash
            curr = await node_client.get_block_record(peak_hash)
            while curr is not None and not curr.is_transaction_block:
                curr = await node_client.get_block_record(curr.prev_hash)
            if curr is not None:
                peak_time = curr.timestamp
            else:
                peak_time = uint64(0)
        peak_time_struct = time.struct_time(time.localtime(peak_time))

        print(
            "      Time:",
            f"{time.strftime('%a %b %d %Y %T %Z', peak_time_struct)}",
        )

        print("\nEstimated network space: ", end="")
        print(format_bytes(blockchain_state["space"]))
        print(f"Current difficulty: {difficulty}")
        print(f"Current VDF sub_slot_iters: {sub_slot_iters}")
        print("\n  Height: |   Hash:")

        added_blocks: List[BlockRecord] = []
        curr = await node_client.get_block_record(peak.header_hash)
        while curr is not None and len(added_blocks) < num_blocks:
            added_blocks.append(curr)
            if curr.height > 0:
                curr = await node_client.get_block_record(curr.prev_hash)
            else:
                curr = None

        for b in added_blocks:
            print(f"{b.height:>9} | {b.header_hash}")
    else:
        print("Blockchain has no blocks yet")
    return False


async def print_block_from_hash(
    node_client: BeaconRpcClient, config: Dict[str, Any], block_by_header_hash: str
) -> None:
    import time

    from bpx.consensus.block_record import BlockRecord
    from bpx.types.blockchain_format.sized_bytes import bytes32
    from bpx.types.full_block import FullBlock
    from bpx.util.byte_types import hexstr_to_bytes

    block: Optional[BlockRecord] = await node_client.get_block_record(hexstr_to_bytes(block_by_header_hash))
    full_block: Optional[FullBlock] = await node_client.get_block(hexstr_to_bytes(block_by_header_hash))
    # Would like to have a verbose flag for this
    if block is not None:
        assert full_block is not None
        prev_b = await node_client.get_block_record(block.prev_hash)
        if prev_b is not None:
            difficulty = block.weight - prev_b.weight
        else:
            difficulty = block.weight
        if block.is_transaction_block:
            assert full_block.transactions_info is not None
            block_time = time.struct_time(
                time.localtime(
                    full_block.foliage_transaction_block.timestamp if full_block.foliage_transaction_block else None
                )
            )
            block_time_string = time.strftime("%a %b %d %Y %T %Z", block_time)
        else:
            block_time_string = "Not a transaction block"
        pool_pk = (
            full_block.reward_chain_block.proof_of_space.pool_public_key
            if full_block.reward_chain_block.proof_of_space.pool_public_key is not None
            else "None"
        )
        print(
            f"Block Height           {block.height}\n"
            f"Header Hash            0x{block.header_hash.hex()}\n"
            f"Timestamp              {block_time_string}\n"
            f"Weight                 {block.weight}\n"
            f"Previous Block         0x{block.prev_hash.hex()}\n"
            f"Difficulty             {difficulty}\n"
            f"Sub-slot iters         {block.sub_slot_iters}\n"
            f"Total VDF Iterations   {block.total_iters}\n"
            f"Deficit                {block.deficit}\n"
            f"PoSpace 'k' Size       {full_block.reward_chain_block.proof_of_space.size}\n"
            f"Plot Public Key        0x{full_block.reward_chain_block.proof_of_space.plot_public_key}\n"
            f"Pool Public Key        {pool_pk}\n"
        )
    else:
        print("Block with header hash", block_by_header_hash, "not found")

async def show_async(
    rpc_port: Optional[int],
    root_path: Path,
    print_state: bool,
    block_header_hash_by_height: str,
    block_by_header_hash: str,
) -> None:
    from bpx.cmds.cmds_util import get_any_service_client

    async with get_any_service_client(BeaconRpcClient, rpc_port, root_path) as node_config_fp:
        node_client, config, _ = node_config_fp
        if node_client is not None:
            # Check State
            if print_state:
                if await print_blockchain_state(node_client, config) is True:
                    return None  # if no blockchain is found
            # Get Block Information
            if block_header_hash_by_height != "":
                block_header = await node_client.get_block_record_by_height(block_header_hash_by_height)
                if block_header is not None:
                    print(f"Header hash of block {block_header_hash_by_height}: {block_header.header_hash.hex()}")
                else:
                    print("Block height", block_header_hash_by_height, "not found")
            if block_by_header_hash != "":
                await print_block_from_hash(node_client, config, block_by_header_hash)
