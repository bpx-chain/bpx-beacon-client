from __future__ import annotations

import logging
import random
from dataclasses import replace
from typing import Callable, Dict, List, Optional, Tuple

import blspy
from blspy import G1Element, G2Element

from bpx.consensus.block_record import BlockRecord
from bpx.consensus.blockchain_interface import BlockchainInterface
from bpx.consensus.constants import ConsensusConstants
from bpx.beacon.signage_point import SignagePoint
from bpx.types.blockchain_format.foliage import Foliage, FoliageBlockData, FoliageTransactionBlock
from bpx.types.blockchain_format.proof_of_space import ProofOfSpace
from bpx.types.blockchain_format.reward_chain_block import RewardChainBlock, RewardChainBlockUnfinished
from bpx.types.blockchain_format.sized_bytes import bytes20, bytes32
from bpx.types.blockchain_format.vdf import VDFInfo, VDFProof
from bpx.types.end_of_slot_bundle import EndOfSubSlotBundle
from bpx.types.full_block import FullBlock
from bpx.types.unfinished_block import UnfinishedBlock
from bpx.util.hash import std_hash
from bpx.util.ints import uint8, uint32, uint64, uint128
from bpx.util.prev_transaction_block import get_prev_transaction_block
from bpx.util.recursive_replace import recursive_replace
from bpx.beacon.execution_client import ExecutionClient

log = logging.getLogger(__name__)

def create_foliage(
    constants: ConsensusConstants,
    execution_client: ExecutionClient,
    reward_block_unfinished: RewardChainBlockUnfinished,
    prev_block: Optional[BlockRecord],
    blocks: BlockchainInterface,
    total_iters_sp: uint128,
    timestamp: uint64,
    coinbase: bytes20,
    get_plot_signature: Callable[[bytes32, G1Element], G2Element],
    seed: bytes = b"",
) -> Foliage:
    """
    Creates a foliage for a given reward chain block. This is called at the signage point, so some of this information may be
    tweaked at the infusion point.

    Args:
        constants: consensus constants being used for this chain
        execution_client: execution client instance
        reward_block_unfinished: the reward block to look at, potentially at the signage point
        prev_block: the previous block at the signage point
        blocks: dict from header hash to blocks, of all ancestor blocks
        total_iters_sp: total iters at the signage point
        timestamp: timestamp to put into the foliage block
        coinbase: where to pay out farmer rewards
        get_plot_signature: retrieve the signature corresponding to the plot public key
        seed: seed to randomize block

    """

    if prev_block is not None:
        res = get_prev_transaction_block(prev_block, blocks, total_iters_sp)
        is_transaction_block: bool = res[0]
        prev_transaction_block: Optional[BlockRecord] = res[1]
        execution_payload = execution_client.get_payload(prev_transaction_block)
        execution_block_hash = execution_payload.blockHash
    else:
        # Genesis is a transaction block
        prev_transaction_block = None
        is_transaction_block = True
        execution_payload = None
        execution_block_hash = constants.GENESIS_EXECUTION_BLOCK_HASH
    
    random.seed(seed)
    # Use the extension data to create different blocks based on header hash
    extension_data: bytes32 = bytes32(random.randint(0, 100000000).to_bytes(32, "big"))
    if prev_block is None:
        height: uint32 = uint32(0)
    else:
        height = uint32(prev_block.height + 1)

    foliage_data = FoliageBlockData(
        reward_block_unfinished.get_hash(),
        coinbase,
        extension_data,
    )

    foliage_block_data_signature: G2Element = get_plot_signature(
        foliage_data.get_hash(),
        reward_block_unfinished.proof_of_space.plot_public_key,
    )

    prev_block_hash: bytes32 = constants.GENESIS_CHALLENGE
    if height != 0:
        assert prev_block is not None
        prev_block_hash = prev_block.header_hash
    
    foliage_transaction_block_hash: Optional[bytes32]

    if is_transaction_block:
        if prev_transaction_block is None:
            prev_transaction_block_hash: bytes32 = constants.GENESIS_CHALLENGE
        else:
            prev_transaction_block_hash = prev_transaction_block.header_hash
        foliage_transaction_block: Optional[FoliageTransactionBlock] = FoliageTransactionBlock(
            prev_transaction_block_hash,
            timestamp,
            execution_block_hash,
        )
        assert foliage_transaction_block is not None
        
        foliage_transaction_block_hash = foliage_transaction_block.get_hash()
        foliage_transaction_block_signature: Optional[G2Element] = get_plot_signature(
            foliage_transaction_block_hash, reward_block_unfinished.proof_of_space.plot_public_key
        )
        assert foliage_transaction_block_signature is not None
    else:
        foliage_transaction_block_hash = None
        foliage_transaction_block_signature = None
        foliage_transaction_block = None
    assert (foliage_transaction_block_hash is None) == (foliage_transaction_block_signature is None)

    foliage = Foliage(
        prev_block_hash,
        reward_block_unfinished.get_hash(),
        foliage_data,
        foliage_block_data_signature,
        foliage_transaction_block_hash,
        foliage_transaction_block_signature,
    )

    return foliage, foliage_transaction_block, execution_payload


def create_unfinished_block(
    constants: ConsensusConstants,
    execution_client: ExecutionClient,
    sub_slot_start_total_iters: uint128,
    sub_slot_iters: uint64,
    signage_point_index: uint8,
    sp_iters: uint64,
    ip_iters: uint64,
    proof_of_space: ProofOfSpace,
    slot_cc_challenge: bytes32,
    coinbase: bytes20,
    get_plot_signature: Callable[[bytes32, G1Element], G2Element],
    signage_point: SignagePoint,
    timestamp: uint64,
    blocks: BlockchainInterface,
    seed: bytes = b"",
    prev_block: Optional[BlockRecord] = None,
    finished_sub_slots_input: Optional[List[EndOfSubSlotBundle]] = None,
) -> UnfinishedBlock:
    """
    Creates a new unfinished block using all the information available at the signage point. This will have to be
    modified using information from the infusion point.

    Args:
        constants: consensus constants being used for this chain
        execution_client: execution client instance
        sub_slot_start_total_iters: the starting sub-slot iters at the signage point sub-slot
        sub_slot_iters: sub-slot-iters at the infusion point epoch
        signage_point_index: signage point index of the block to create
        sp_iters: sp_iters of the block to create
        ip_iters: ip_iters of the block to create
        proof_of_space: proof of space of the block to create
        slot_cc_challenge: challenge hash at the sp sub-slot
        coinbase: where to pay out farmer rewards
        get_plot_signature: function that returns signature corresponding to plot public key
        signage_point: signage point information (VDFs)
        prev_block: previous block (already in chain) from the signage point
        timestamp: timestamp to add to the foliage block, if created
        blocks: dictionary from header hash to SBR of all included SBR
        seed: seed to randomize chain
        finished_sub_slots_input: finished_sub_slots at the signage point

    Returns:

    """
    if coinbase == bytes20.fromhex("0000000000000000000000000000000000000000"):
        raise RuntimeError("Unable to create new block: coinbase not set")
    
    if finished_sub_slots_input is None:
        finished_sub_slots: List[EndOfSubSlotBundle] = []
    else:
        finished_sub_slots = finished_sub_slots_input.copy()
    overflow: bool = sp_iters > ip_iters
    total_iters_sp: uint128 = uint128(sub_slot_start_total_iters + sp_iters)
    is_genesis: bool = prev_block is None

    new_sub_slot: bool = len(finished_sub_slots) > 0

    cc_sp_hash: bytes32 = slot_cc_challenge

    # Only enters this if statement if we are in testing mode (making VDF proofs here)
    if signage_point.cc_vdf is not None:
        assert signage_point.rc_vdf is not None
        cc_sp_hash = signage_point.cc_vdf.output.get_hash()
        rc_sp_hash = signage_point.rc_vdf.output.get_hash()
    else:
        if new_sub_slot:
            rc_sp_hash = finished_sub_slots[-1].reward_chain.get_hash()
        else:
            if is_genesis:
                rc_sp_hash = constants.GENESIS_CHALLENGE
            else:
                assert prev_block is not None
                assert blocks is not None
                curr = prev_block
                while not curr.first_in_sub_slot:
                    curr = blocks.block_record(curr.prev_hash)
                assert curr.finished_reward_slot_hashes is not None
                rc_sp_hash = curr.finished_reward_slot_hashes[-1]
        signage_point = SignagePoint(None, None, None, None)

    cc_sp_signature: Optional[G2Element] = get_plot_signature(cc_sp_hash, proof_of_space.plot_public_key)
    rc_sp_signature: Optional[G2Element] = get_plot_signature(rc_sp_hash, proof_of_space.plot_public_key)
    assert cc_sp_signature is not None
    assert rc_sp_signature is not None
    assert blspy.AugSchemeMPL.verify(proof_of_space.plot_public_key, cc_sp_hash, cc_sp_signature)

    total_iters = uint128(sub_slot_start_total_iters + ip_iters + (sub_slot_iters if overflow else 0))

    rc_block = RewardChainBlockUnfinished(
        total_iters,
        signage_point_index,
        slot_cc_challenge,
        proof_of_space,
        signage_point.cc_vdf,
        cc_sp_signature,
        signage_point.rc_vdf,
        rc_sp_signature,
    )
    
    (foliage, foliage_transaction_block, execution_payload) = create_foliage(
        constants,
        execution_client,
        rc_block,
        prev_block,
        blocks,
        total_iters_sp,
        timestamp,
        coinbase,
        get_plot_signature,
        seed,
    )
    return UnfinishedBlock(
        finished_sub_slots,
        rc_block,
        signage_point.cc_proof,
        signage_point.rc_proof,
        foliage,
        foliage_transaction_block,
        execution_payload,
    )


def unfinished_block_to_full_block(
    unfinished_block: UnfinishedBlock,
    cc_ip_vdf: VDFInfo,
    cc_ip_proof: VDFProof,
    rc_ip_vdf: VDFInfo,
    rc_ip_proof: VDFProof,
    icc_ip_vdf: Optional[VDFInfo],
    icc_ip_proof: Optional[VDFProof],
    finished_sub_slots: List[EndOfSubSlotBundle],
    prev_block: Optional[BlockRecord],
    blocks: BlockchainInterface,
    total_iters_sp: uint128,
    difficulty: uint64,
) -> FullBlock:
    """
    Converts an unfinished block to a finished block. Includes all the infusion point VDFs as well as tweaking
    other properties (height, weight, sub-slots, etc)

    Args:
        unfinished_block: the unfinished block to finish
        cc_ip_vdf: the challenge chain vdf info at the infusion point
        cc_ip_proof: the challenge chain proof
        rc_ip_vdf: the reward chain vdf info at the infusion point
        rc_ip_proof: the reward chain proof
        icc_ip_vdf: the infused challenge chain vdf info at the infusion point
        icc_ip_proof: the infused challenge chain proof
        finished_sub_slots: finished sub slots from the prev block to the infusion point
        prev_block: prev block from the infusion point
        blocks: dictionary from header hash to SBR of all included SBR
        total_iters_sp: total iters at the signage point
        difficulty: difficulty at the infusion point

    """
    # Replace things that need to be replaced, since foliage blocks did not necessarily have the latest information
    if prev_block is None:
        is_transaction_block = True
        new_weight = uint128(difficulty)
        new_height = uint32(0)
        new_foliage = unfinished_block.foliage
        new_foliage_transaction_block = unfinished_block.foliage_transaction_block
        new_execution_payload = unfinished_block.execution_payload
    else:
        is_transaction_block, _ = get_prev_transaction_block(prev_block, blocks, total_iters_sp)
        new_weight = uint128(prev_block.weight + difficulty)
        new_height = uint32(prev_block.height + 1)
        if is_transaction_block:
            new_fbh = unfinished_block.foliage.foliage_transaction_block_hash
            new_fbs = unfinished_block.foliage.foliage_transaction_block_signature
            new_foliage_transaction_block = unfinished_block.foliage_transaction_block
            new_execution_payload = unfinished_block.execution_payload
        else:
            new_fbh = None
            new_fbs = None
            new_foliage_transaction_block = None
            new_execution_payload = None
        assert (new_fbh is None) == (new_fbs is None)
        new_foliage = replace(
            unfinished_block.foliage,
            prev_block_hash=prev_block.header_hash,
            foliage_transaction_block_hash=new_fbh,
            foliage_transaction_block_signature=new_fbs,
        )
    ret = FullBlock(
        finished_sub_slots,
        RewardChainBlock(
            new_weight,
            new_height,
            unfinished_block.reward_chain_block.total_iters,
            unfinished_block.reward_chain_block.signage_point_index,
            unfinished_block.reward_chain_block.pos_ss_cc_challenge_hash,
            unfinished_block.reward_chain_block.proof_of_space,
            unfinished_block.reward_chain_block.challenge_chain_sp_vdf,
            unfinished_block.reward_chain_block.challenge_chain_sp_signature,
            cc_ip_vdf,
            unfinished_block.reward_chain_block.reward_chain_sp_vdf,
            unfinished_block.reward_chain_block.reward_chain_sp_signature,
            rc_ip_vdf,
            icc_ip_vdf,
            is_transaction_block,
        ),
        unfinished_block.challenge_chain_sp_proof,
        cc_ip_proof,
        unfinished_block.reward_chain_sp_proof,
        rc_ip_proof,
        icc_ip_proof,
        new_foliage,
        new_foliage_transaction_block,
        new_execution_payload,
    )
    ret = recursive_replace(
        ret,
        "foliage.reward_block_hash",
        ret.reward_chain_block.get_hash(),
    )
    return ret
