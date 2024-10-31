from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from bpx.types.blockchain_format.foliage import Foliage, FoliageTransactionBlock
from bpx.types.blockchain_format.reward_chain_block import RewardChainBlockUnfinished
from bpx.types.blockchain_format.sized_bytes import bytes32
from bpx.types.blockchain_format.vdf import VDFProof
from bpx.types.end_of_slot_bundle import EndOfSubSlotBundle
from bpx.util.ints import uint128
from bpx.util.streamable import Streamable, streamable
from bpx.types.blockchain_format.execution_payload import ExecutionPayloadV2


@streamable
@dataclass(frozen=True)
class UnfinishedHeaderBlock(Streamable):
    finished_sub_slots: List[EndOfSubSlotBundle]  # If first sb
    reward_chain_block: RewardChainBlockUnfinished  # Reward chain trunk data
    challenge_chain_sp_proof: Optional[VDFProof]  # If not first sp in sub-slot
    reward_chain_sp_proof: Optional[VDFProof]  # If not first sp in sub-slot
    foliage: Foliage  # Reward chain foliage data
    foliage_transaction_block: Optional[FoliageTransactionBlock]  # Reward chain foliage data (tx block)
    execution_payload: Optional[ExecutionPayloadV2]

    @property
    def prev_header_hash(self) -> bytes32:
        return self.foliage.prev_block_hash

    @property
    def header_hash(self) -> bytes32:
        return self.foliage.get_hash()

    @property
    def total_iters(self) -> uint128:
        return self.reward_chain_block.total_iters
