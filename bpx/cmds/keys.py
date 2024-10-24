from __future__ import annotations

from typing import Optional, Tuple

import click


@click.group("keys", short_help="Manage your keys")
@click.pass_context
def keys_cmd(ctx: click.Context) -> None:
    """Create, delete, view and use your key pairs"""
    from pathlib import Path

    root_path: Path = ctx.obj["root_path"]
    if not root_path.is_dir():
        raise RuntimeError("Please initialize (or migrate) your config directory with bpx init")


@keys_cmd.command("generate", short_help="Generates and adds a key to keychain")
@click.option(
    "--label",
    "-l",
    default=None,
    help="Enter the label for the key",
    type=str,
    required=False,
)
@click.pass_context
def generate_cmd(ctx: click.Context, label: Optional[str]) -> None:
    from .init_funcs import check_keys
    from .keys_funcs import generate_and_add

    generate_and_add(label)
    check_keys(ctx.obj["root_path"])


@keys_cmd.command("show", short_help="Displays all the keys in keychain or the key with the given fingerprint")
@click.option(
    "--show-mnemonic-seed", help="Show the mnemonic seed of the keys", default=False, show_default=True, is_flag=True
)
@click.option(
    "--json",
    "-j",
    help=("Displays all the keys in keychain as JSON"),
    default=False,
    show_default=True,
    is_flag=True,
)
@click.option(
    "--fingerprint",
    "-f",
    help="Enter the fingerprint of the key you want to view",
    type=int,
    required=False,
    default=None,
)
@click.pass_context
def show_cmd(
    ctx: click.Context,
    show_mnemonic_seed: bool,
    json: bool,
    fingerprint: Optional[int],
) -> None:
    from .keys_funcs import show_keys

    show_keys(ctx.obj["root_path"], show_mnemonic_seed, json, fingerprint)


@keys_cmd.command("add", short_help="Add a private key by mnemonic")
@click.option(
    "--filename",
    "-f",
    default=None,
    help="The filename containing the secret key mnemonic to add",
    type=str,
    required=False,
)
@click.option(
    "--label",
    "-l",
    default=None,
    help="Enter the label for the key",
    type=str,
    required=False,
)
@click.pass_context
def add_cmd(ctx: click.Context, filename: str, label: Optional[str]) -> None:
    from .init_funcs import check_keys
    from .keys_funcs import query_and_add_private_key_seed

    mnemonic = None
    if filename:
        from pathlib import Path

        mnemonic = Path(filename).read_text().rstrip()

    query_and_add_private_key_seed(mnemonic, label)
    check_keys(ctx.obj["root_path"])


@keys_cmd.group("label", short_help="Manage your key labels")
def label_cmd() -> None:
    pass


@label_cmd.command("show", short_help="Show the labels of all available keys")
def show_label_cmd() -> None:
    from .keys_funcs import show_all_key_labels

    show_all_key_labels()


@label_cmd.command("set", short_help="Set the label of a key")
@click.option(
    "--fingerprint",
    "-f",
    help="Enter the fingerprint of the key you want to use",
    type=int,
    required=True,
)
@click.option(
    "--label",
    "-l",
    help="Enter the new label for the key",
    type=str,
    required=True,
)
def set_label_cmd(fingerprint: int, label: str) -> None:
    from .keys_funcs import set_key_label

    set_key_label(fingerprint, label)


@label_cmd.command("delete", short_help="Delete the label of a key")
@click.option(
    "--fingerprint",
    "-f",
    help="Enter the fingerprint of the key you want to use",
    type=int,
    required=True,
)
def delete_label_cmd(fingerprint: int) -> None:
    from .keys_funcs import delete_key_label

    delete_key_label(fingerprint)


@keys_cmd.command("delete", short_help="Delete a key by its pk fingerprint in hex form")
@click.option(
    "--fingerprint",
    "-f",
    default=None,
    help="Enter the fingerprint of the key you want to use",
    type=int,
    required=True,
)
@click.pass_context
def delete_cmd(ctx: click.Context, fingerprint: int) -> None:
    from .init_funcs import check_keys
    from .keys_funcs import delete

    delete(fingerprint)
    check_keys(ctx.obj["root_path"])


@keys_cmd.command("delete_all", short_help="Delete all private keys in keychain")
def delete_all_cmd() -> None:
    from bpx.util.keychain import Keychain

    Keychain().delete_all_keys()


@keys_cmd.command("generate_and_print", short_help="Generates but does NOT add to keychain")
def generate_and_print_cmd() -> None:
    from .keys_funcs import generate_and_print

    generate_and_print()
