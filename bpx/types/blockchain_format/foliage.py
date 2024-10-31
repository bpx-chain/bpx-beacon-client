from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from blspy import G2Element

from bpx.types.blockchain_format.sized_bytes import bytes20, bytes32
from bpx.util.ints import uint64
from bpx.util.streamable import Streamable, streamable


@streamable
@dataclass(frozen=True)
class FoliageTransactionBlock(Streamable):
    prev_transaction_block_hash: bytes32
    timestamp: uint64
    execution_block_hash: bytes32


@streamable
@dataclass(frozen=True)
class FoliageBlockData(Streamable):
    # Part of the block that is signed by the plot key
    unfinished_reward_block_hash: bytes32
    coinbase: bytes20
    extension_data: bytes32  # Used for future updates. Can be any 32 byte value initially


@streamable
@dataclass(frozen=True)
class Foliage(Streamable):
    # The entire foliage block, containing signature and the unsigned back pointer
    # The hash of this is the "header hash". Note that for unfinished blocks, the prev_block_hash
    # Is the prev from the signage point, and can be replaced with a more recent block
    prev_block_hash: bytes32
    reward_block_hash: bytes32
    foliage_block_data: FoliageBlockData
    foliage_block_data_signature: G2Element
    foliage_transaction_block_hash: Optional[bytes32]
    foliage_transaction_block_signature: Optional[G2Element]
