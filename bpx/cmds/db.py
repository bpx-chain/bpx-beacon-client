from __future__ import annotations

from pathlib import Path
from typing import Optional

import click

from bpx.cmds.db_backup_func import db_backup_func
from bpx.cmds.db_validate_func import db_validate_func


@click.group("db", short_help="Manage the blockchain database")
def db_cmd() -> None:
    pass


@db_cmd.command("validate", short_help="validate the blockchain database. Does not verify proofs")
@click.option("--db", "in_db_path", default=None, type=click.Path(), help="Specifies which database file to validate")
@click.option(
    "--validate-blocks",
    default=False,
    is_flag=True,
    help="validate consistency of properties of the encoded blocks and block records",
)
@click.pass_context
def db_validate_cmd(ctx: click.Context, in_db_path: Optional[str], validate_blocks: bool) -> None:
    try:
        db_validate_func(
            Path(ctx.obj["root_path"]),
            None if in_db_path is None else Path(in_db_path),
            validate_blocks=validate_blocks,
        )
    except RuntimeError as e:
        print(f"FAILED: {e}")


@db_cmd.command("backup", short_help="backup the blockchain database using VACUUM INTO command")
@click.option("--backup_file", "db_backup_file", default=None, type=click.Path(), help="Specifies the backup file")
@click.option("--no_indexes", default=False, is_flag=True, help="Create backup without indexes")
@click.pass_context
def db_backup_cmd(ctx: click.Context, db_backup_file: Optional[str], no_indexes: bool) -> None:
    try:
        db_backup_func(
            Path(ctx.obj["root_path"]),
            None if db_backup_file is None else Path(db_backup_file),
            no_indexes=no_indexes,
        )
    except RuntimeError as e:
        print(f"FAILED: {e}")
