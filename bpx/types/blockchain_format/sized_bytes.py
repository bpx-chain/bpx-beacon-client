from __future__ import annotations

from bpx.util.byte_types import SizedBytes


class bytes20(SizedBytes):
    _size = 20


class bytes32(SizedBytes):
    _size = 32


class bytes100(SizedBytes):
    _size = 100


class bytes256(SizedBytes):
    _size = 256
