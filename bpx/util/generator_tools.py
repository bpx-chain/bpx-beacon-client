from __future__ import annotations

from typing import Any, Iterator, List, Optional, Tuple

from bpx.types.full_block import FullBlock
from bpx.types.header_block import HeaderBlock

def get_block_header(block: FullBlock) -> HeaderBlock:
    return HeaderBlock(
        block.finished_sub_slots,
        block.reward_chain_block,
        block.challenge_chain_sp_proof,
        block.challenge_chain_ip_proof,
        block.reward_chain_sp_proof,
        block.reward_chain_ip_proof,
        block.infused_challenge_chain_ip_proof,
        block.foliage,
        block.foliage_transaction_block,
        block.execution_payload,
    )

def list_to_batches(list_to_split: List[Any], batch_size: int) -> Iterator[Tuple[int, List[Any]]]:
    if batch_size <= 0:
        raise ValueError("list_to_batches: batch_size must be greater than 0.")
    total_size = len(list_to_split)
    if total_size == 0:
        return iter(())
    for batch_start in range(0, total_size, batch_size):
        batch_end = min(batch_start + batch_size, total_size)
        yield total_size - batch_end, list_to_split[batch_start:batch_end]
