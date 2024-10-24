from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from blspy import G2Element

from bpx.types.blockchain_format.sized_bytes import bytes20, bytes32, bytes256
from bpx.util.ints import uint64, uint256
from bpx.util.streamable import Streamable, streamable


@streamable
@dataclass(frozen=True)
class WithdrawalV1(Streamable):
    index: uint64
    validatorIndex: uint64
    address: bytes20
    amount: uint64


@streamable
@dataclass(frozen=True)
class ExecutionPayloadV2(Streamable):
    parentHash: bytes32
    feeRecipient: bytes20
    stateRoot: bytes32
    receiptsRoot: bytes32
    logsBloom: bytes256
    prevRandao: bytes32
    blockNumber: uint64
    gasLimit: uint64
    gasUsed: uint64
    timestamp: uint64
    extraData: bytes
    baseFeePerGas: uint256
    blockHash: bytes32
    transactions: List[bytes]
    withdrawals: List[WithdrawalV1]
