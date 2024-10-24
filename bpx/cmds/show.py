from __future__ import annotations

from typing import Optional

import click

from bpx.cmds.show_funcs import show_async


@click.command("show", short_help="Show node information", no_args_is_help=True)
@click.option(
    "-p",
    "--rpc-port",
    help=(
        "Set the port where the Beacon Client is hosting the RPC interface. "
        "See the rpc_port under beacon in config.yaml"
    ),
    type=int,
    default=None,
)
@click.option("-s", "--state", help="Show the current state of the blockchain", is_flag=True, type=bool, default=False)
@click.option(
    "-bh", "--block-header-hash-by-height", help="Look up a block header hash by block height", type=str, default=""
)
@click.option("-b", "--block-by-header-hash", help="Look up a block by block header hash", type=str, default="")
@click.pass_context
def show_cmd(
    ctx: click.Context,
    rpc_port: Optional[int],
    state: bool,
    block_header_hash_by_height: str,
    block_by_header_hash: str,
) -> None:
    import asyncio

    asyncio.run(
        show_async(
            rpc_port,
            ctx.obj["root_path"],
            state,
            block_header_hash_by_height,
            block_by_header_hash,
        )
    )
