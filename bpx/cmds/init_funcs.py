from __future__ import annotations

import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from bpx import __version__
from bpx.cmds.configure import configure
from bpx.ssl.create_ssl import create_all_ssl
from bpx.util.config import (
    create_default_bpx_config,
    initial_config_file,
    load_config,
    lock_and_load_config,
    save_config,
    unflatten_properties,
)
from bpx.util.db_version import set_db_version
from bpx.util.keychain import Keychain
from bpx.util.path import path_from_root
from bpx.util.ssl_check import (
    DEFAULT_PERMISSIONS_CERT_FILE,
    DEFAULT_PERMISSIONS_KEY_FILE,
    RESTRICT_MASK_CERT_FILE,
    RESTRICT_MASK_KEY_FILE,
    check_and_fix_permissions_for_ssl_file,
    fix_ssl,
)
from bpx.util.derive_keys import master_sk_to_pool_sk


def check_keys(new_root: Path, keychain: Optional[Keychain] = None) -> None:
    if keychain is None:
        keychain = Keychain()
    all_sks = keychain.get_all_private_keys()
    if len(all_sks) == 0:
        print("No keys are present in the keychain. Generate them with 'bpx keys generate'")
        return None

    with lock_and_load_config(new_root, "config.yaml") as config:
        pool_child_pubkeys = [master_sk_to_pool_sk(sk).get_g1() for sk, _ in all_sks]

        # Set the pool pks in the farmer
        pool_pubkeys_hex = set(bytes(pk).hex() for pk in pool_child_pubkeys)
        if "pool_public_keys" in config["farmer"]:
            for pk_hex in config["farmer"]["pool_public_keys"]:
                # Add original ones in config
                pool_pubkeys_hex.add(pk_hex)

        config["farmer"]["pool_public_keys"] = pool_pubkeys_hex
        save_config(new_root, "config.yaml", config)


def copy_files_rec(old_path: Path, new_path: Path) -> None:
    if old_path.is_file():
        print(f"{new_path}")
        new_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(old_path, new_path)
    elif old_path.is_dir():
        for old_path_child in old_path.iterdir():
            new_path_child = new_path / old_path_child.name
            copy_files_rec(old_path_child, new_path_child)

def copy_cert_files(cert_path: Path, new_path: Path) -> None:
    for old_path_child in cert_path.glob("*.crt"):
        new_path_child = new_path / old_path_child.name
        copy_files_rec(old_path_child, new_path_child)
        check_and_fix_permissions_for_ssl_file(new_path_child, RESTRICT_MASK_CERT_FILE, DEFAULT_PERMISSIONS_CERT_FILE)

    for old_path_child in cert_path.glob("*.key"):
        new_path_child = new_path / old_path_child.name
        copy_files_rec(old_path_child, new_path_child)
        check_and_fix_permissions_for_ssl_file(new_path_child, RESTRICT_MASK_KEY_FILE, DEFAULT_PERMISSIONS_KEY_FILE)


def init(
    create_certs: Optional[Path],
    root_path: Path,
    fix_ssl_permissions: bool = False,
    testnet: bool = False,
) -> Optional[int]:
    if create_certs is not None:
        if root_path.exists():
            if os.path.isdir(create_certs):
                ca_dir: Path = root_path / "config/ssl/ca"
                if ca_dir.exists():
                    print(f"Deleting your OLD CA in {ca_dir}")
                    shutil.rmtree(ca_dir)
                print(f"Copying your CA from {create_certs} to {ca_dir}")
                copy_cert_files(create_certs, ca_dir)
                create_all_ssl(root_path)
            else:
                print(f"** Directory {create_certs} does not exist **")
        else:
            print(f"** {root_path} does not exist. Executing core init **")
            # sanity check here to prevent infinite recursion
            if (
                bpx_init(
                    root_path,
                    fix_ssl_permissions=fix_ssl_permissions,
                    testnet=testnet,
                )
                == 0
                and root_path.exists()
            ):
                return init(create_certs, root_path, fix_ssl_permissions)

            print(f"** {root_path} was not created. Exiting **")
            return -1
    else:
        return bpx_init(root_path, fix_ssl_permissions=fix_ssl_permissions, testnet=testnet)

    return None


def bpx_version_number() -> Tuple[str, str, str, str]:
    scm_full_version = __version__
    left_full_version = scm_full_version.split("+")

    version = left_full_version[0].split(".")

    scm_major_version = version[0]
    scm_minor_version = version[1]
    if len(version) > 2:
        smc_patch_version = version[2]
        patch_release_number = smc_patch_version
    else:
        smc_patch_version = ""

    major_release_number = scm_major_version
    minor_release_number = scm_minor_version
    dev_release_number = ""

    # If this is a beta dev release - get which beta it is
    if "0b" in scm_minor_version:
        original_minor_ver_list = scm_minor_version.split("0b")
        major_release_number = str(1 - int(scm_major_version))  # decrement the major release for beta
        minor_release_number = scm_major_version
        patch_release_number = original_minor_ver_list[1]
        if smc_patch_version and "dev" in smc_patch_version:
            dev_release_number = "." + smc_patch_version
    elif "0rc" in version[1]:
        original_minor_ver_list = scm_minor_version.split("0rc")
        major_release_number = str(1 - int(scm_major_version))  # decrement the major release for release candidate
        minor_release_number = str(int(scm_major_version) + 1)  # RC is 0.2.1 for RC 1
        patch_release_number = original_minor_ver_list[1]
        if smc_patch_version and "dev" in smc_patch_version:
            dev_release_number = "." + smc_patch_version
    else:
        major_release_number = scm_major_version
        minor_release_number = scm_minor_version
        patch_release_number = smc_patch_version
        dev_release_number = ""

    install_release_number = major_release_number + "." + minor_release_number
    if len(patch_release_number) > 0:
        install_release_number += "." + patch_release_number
    if len(dev_release_number) > 0:
        install_release_number += dev_release_number

    return major_release_number, minor_release_number, patch_release_number, dev_release_number


def bpx_full_version_str() -> str:
    major, minor, patch, dev = bpx_version_number()
    return f"{major}.{minor}.{patch}{dev}"


def bpx_init(
    root_path: Path,
    *,
    should_check_keys: bool = True,
    fix_ssl_permissions: bool = False,
    testnet: bool = False,
) -> int:
    """
    Standard first run initialization or migration steps. Handles config creation,
    generation of SSL certs.

    should_check_keys can be set to False to avoid blocking when accessing a passphrase
    protected Keychain. When launching the daemon from the GUI, we want the GUI to
    handle unlocking the keychain.
    """
    bpx_root = os.environ.get("BPX_ROOT", None)
    if bpx_root is not None:
        print(f"BPX_ROOT is set to {bpx_root}")

    print(f"BPX directory {root_path}")
    if root_path.is_dir() and Path(root_path / "config" / "config.yaml").exists():
        # This is reached if BPX_ROOT is set, or if user has run bpx init twice
        # before a new update.
        if testnet:
            configure(
                root_path,
                set_farmer_peer="",
                set_node_introducer="",
                set_beacon_port="",
                set_harvester_port="",
                set_log_level="",
                enable_upnp="",
                set_outbound_peer_count="",
                set_peer_count="",
                testnet="true",
                peer_connect_timeout="",
                crawler_db_path="",
                crawler_minimum_version_count=None,
                seeder_domain_name="",
                seeder_nameserver="",
            )
        if fix_ssl_permissions:
            fix_ssl(root_path)
        if should_check_keys:
            check_keys(root_path)
        print(f"{root_path} already exists, no migration action taken")
        return -1

    create_default_bpx_config(root_path)
    if testnet:
        configure(
            root_path,
            set_farmer_peer="",
            set_node_introducer="",
            set_beacon_port="",
            set_harvester_port="",
            set_log_level="",
            enable_upnp="",
            set_outbound_peer_count="",
            set_peer_count="",
            testnet="true",
            peer_connect_timeout="",
            crawler_db_path="",
            crawler_minimum_version_count=None,
            seeder_domain_name="",
            seeder_nameserver="",
        )
    create_all_ssl(root_path)
    if fix_ssl_permissions:
        fix_ssl(root_path)
    if should_check_keys:
        check_keys(root_path)

    config: Dict[str, Any]

    db_path_replaced: str
    config = load_config(root_path, "config.yaml")["beacon"]
    db_path_replaced = config["database_path"].replace("CHALLENGE", config["selected_network"])
    db_path = path_from_root(root_path, db_path_replaced)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # create new v1 db file
        with sqlite3.connect(db_path) as connection:
            set_db_version(connection, 1)
    except sqlite3.OperationalError:
        # db already exists, so we're good
        pass

    print("")
    print("To see your keys, run 'bpx keys show --show-mnemonic-seed'")

    return 0
