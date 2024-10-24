from __future__ import annotations

import json
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from blspy import AugSchemeMPL, G1Element, G2Element, PrivateKey

from bpx.cmds.passphrase_funcs import obtain_current_passphrase
from bpx.util.config import load_config
from bpx.util.errors import KeychainException
from bpx.util.file_keyring import MAX_LABEL_LENGTH
from bpx.util.ints import uint32
from bpx.util.keychain import Keychain, KeyData, bytes_to_mnemonic, generate_mnemonic, mnemonic_to_seed
from bpx.util.keyring_wrapper import KeyringWrapper
from bpx.util.derive_keys import (
    master_sk_to_farmer_sk,
    master_sk_to_pool_sk,
)


def unlock_keyring() -> None:
    """
    Used to unlock the keyring interactively, if necessary
    """

    try:
        if KeyringWrapper.get_shared_instance().has_master_passphrase():
            obtain_current_passphrase(use_passphrase_cache=True)
    except Exception as e:
        print(f"Unable to unlock the keyring: {e}")
        sys.exit(1)


def generate_and_print() -> str:
    """
    Generates a seed for a private key, and prints the mnemonic to the terminal.
    """

    mnemonic = generate_mnemonic()
    print("Generating private key. Mnemonic (24 secret words):")
    print(mnemonic)
    print("Note that this key has not been added to the keychain. Run bpx keys add")
    return mnemonic


def generate_and_add(label: Optional[str]) -> None:
    """
    Generates a seed for a private key, prints the mnemonic to the terminal, and adds the key to the keyring.
    """
    unlock_keyring()
    print("Generating private key")
    query_and_add_private_key_seed(mnemonic=generate_mnemonic(), label=label)


def query_and_add_private_key_seed(mnemonic: Optional[str], label: Optional[str] = None) -> None:
    unlock_keyring()
    if mnemonic is None:
        mnemonic = input("Enter the mnemonic you want to use: ")
    if label is None:
        label = input("Enter the label you want to assign to this key (Press Enter to skip): ")
    if len(label) == 0:
        label = None
    add_private_key_seed(mnemonic, label)


def add_private_key_seed(mnemonic: str, label: Optional[str]) -> None:
    """
    Add a private key seed to the keyring, with the given mnemonic and an optional label.
    """
    unlock_keyring()
    try:
        sk = Keychain().add_private_key(mnemonic, label)
        fingerprint = sk.get_g1().get_fingerprint()
        print(f"Added private key with public key fingerprint {fingerprint}")

    except (ValueError, KeychainException) as e:
        print(e)


def show_all_key_labels() -> None:
    unlock_keyring()
    fingerprint_width = 11

    def print_line(fingerprint: str, label: str) -> None:
        fingerprint_text = ("{0:<" + str(fingerprint_width) + "}").format(fingerprint)
        label_text = ("{0:<" + str(MAX_LABEL_LENGTH) + "}").format(label)
        print("| " + fingerprint_text + " | " + label_text + " |")

    keys = Keychain().get_keys()

    if len(keys) == 0:
        sys.exit("No keys are present in the keychain. Generate them with 'bpx keys generate'")

    print_line("fingerprint", "label")
    print_line("-" * fingerprint_width, "-" * MAX_LABEL_LENGTH)

    for key_data in keys:
        print_line(str(key_data.fingerprint), key_data.label or "No label assigned")


def set_key_label(fingerprint: int, label: str) -> None:
    unlock_keyring()
    try:
        Keychain().set_label(fingerprint, label)
        print(f"label {label!r} assigned to {fingerprint!r}")
    except Exception as e:
        sys.exit(f"Error: {e}")


def delete_key_label(fingerprint: int) -> None:
    unlock_keyring()
    try:
        Keychain().delete_label(fingerprint)
        print(f"label removed for {fingerprint!r}")
    except Exception as e:
        sys.exit(f"Error: {e}")


def show_keys(
    root_path: Path, show_mnemonic: bool, json_output: bool, fingerprint: Optional[int]
) -> None:
    """
    Prints all keys and mnemonics (if available).
    """
    unlock_keyring()
    config = load_config(root_path, "config.yaml")
    if fingerprint is None:
        all_keys = Keychain().get_keys(True)
    else:
        all_keys = [Keychain().get_key(fingerprint, True)]

    if len(all_keys) == 0:
        if json_output:
            print(json.dumps({"keys": []}))
        else:
            print("There are no saved private keys")
        return None

    if not json_output:
        msg = "Showing all public keys derived from your master seed and private key:"
        if show_mnemonic:
            msg = "Showing all public and private keys"
        print(msg)

    def process_key_data(key_data: KeyData) -> Dict[str, Any]:
        key: Dict[str, Any] = {}
        sk = key_data.private_key
        if key_data.label is not None:
            key["label"] = key_data.label

        key["fingerprint"] = key_data.fingerprint
        key["master_pk"] = bytes(key_data.public_key).hex()
        key["farmer_pk"] = bytes(master_sk_to_farmer_sk(sk).get_g1()).hex()
        key["pool_pk"] = bytes(master_sk_to_pool_sk(sk).get_g1()).hex()

        if show_mnemonic:
            key["master_sk"] = bytes(sk).hex()
            key["mnemonic"] = bytes_to_mnemonic(key_data.entropy)
        return key

    keys = [process_key_data(key) for key in all_keys]

    if json_output:
        print(json.dumps({"keys": list(keys)}))
    else:
        for key in keys:
            print("")
            if "label" in key:
                print("Label:", key["label"])
            print("Fingerprint:", key["fingerprint"])
            print("Master public key (m):", key["master_pk"])
            print("Farmer public key (m/12381/8444/0/0):", key["farmer_pk"])
            print("Pool public key (m/12381/8444/1/0):", key["pool_pk"])
            if show_mnemonic:
                print("Master private key (m):", key["master_sk"])
                print("  Mnemonic seed (24 secret words):")
                print(key["mnemonic"])


def delete(fingerprint: int) -> None:
    """
    Delete a key by its public key fingerprint (which is an integer).
    """
    unlock_keyring()
    print(f"Deleting private_key with fingerprint {fingerprint}")
    Keychain().delete_key_by_fingerprint(fingerprint)
