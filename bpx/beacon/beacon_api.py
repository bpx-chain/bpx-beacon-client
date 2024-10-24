from __future__ import annotations

import asyncio
import dataclasses
import functools
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from secrets import token_bytes
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

from blspy import AugSchemeMPL, G1Element, G2Element

from bpx.consensus.block_creation import create_unfinished_block
from bpx.consensus.block_record import BlockRecord
from bpx.consensus.pot_iterations import calculate_ip_iters, calculate_iterations_quality, calculate_sp_iters
from bpx.beacon.signage_point import SignagePoint
from bpx.protocols import farmer_protocol, beacon_protocol, introducer_protocol, timelord_protocol
from bpx.protocols.beacon_protocol import RejectBlock, RejectBlocks
from bpx.protocols.protocol_message_types import ProtocolMessageTypes
from bpx.server.outbound_message import Message, make_msg
from bpx.server.server import BpxServer
from bpx.server.ws_connection import WSBpxConnection
from bpx.types.blockchain_format.proof_of_space import verify_and_get_quality_string
from bpx.types.blockchain_format.sized_bytes import bytes20, bytes32
from bpx.types.blockchain_format.sub_epoch_summary import SubEpochSummary
from bpx.types.end_of_slot_bundle import EndOfSubSlotBundle
from bpx.types.full_block import FullBlock
from bpx.types.peer_info import PeerInfo
from bpx.types.unfinished_block import UnfinishedBlock
from bpx.util.api_decorators import api_request
from bpx.util.hash import std_hash
from bpx.util.ints import uint8, uint32, uint64, uint128
from bpx.util.limited_semaphore import LimitedSemaphoreFullError

if TYPE_CHECKING:
    from bpx.beacon.beacon import Beacon
else:
    Beacon = object


class BeaconAPI:
    beacon: Beacon
    executor: ThreadPoolExecutor

    def __init__(self, beacon: Beacon) -> None:
        self.beacon = beacon
        self.executor = ThreadPoolExecutor(max_workers=1)

    @property
    def server(self) -> BpxServer:
        assert self.beacon.server is not None
        return self.beacon.server

    @property
    def log(self) -> logging.Logger:
        return self.beacon.log

    @property
    def api_ready(self) -> bool:
        return self.beacon.initialized

    @api_request(peer_required=True, reply_types=[ProtocolMessageTypes.respond_peers])
    async def request_peers(
        self, _request: beacon_protocol.RequestPeers, peer: WSBpxConnection
    ) -> Optional[Message]:
        if peer.peer_server_port is None:
            return None
        peer_info = PeerInfo(peer.peer_host, peer.peer_server_port)
        if self.beacon.beacon_peers is not None:
            msg = await self.beacon.beacon_peers.request_peers(peer_info)
            return msg
        return None

    @api_request(peer_required=True)
    async def respond_peers(
        self, request: beacon_protocol.RespondPeers, peer: WSBpxConnection
    ) -> Optional[Message]:
        self.log.debug(f"Received {len(request.peer_list)} peers")
        if self.beacon.beacon_peers is not None:
            await self.beacon.beacon_peers.respond_peers(request, peer.get_peer_info(), True)
        return None

    @api_request(peer_required=True)
    async def respond_peers_introducer(
        self, request: introducer_protocol.RespondPeersIntroducer, peer: WSBpxConnection
    ) -> Optional[Message]:
        self.log.debug(f"Received {len(request.peer_list)} peers from introducer")
        if self.beacon.beacon_peers is not None:
            await self.beacon.beacon_peers.respond_peers(request, peer.get_peer_info(), False)

        await peer.close()
        return None

    @api_request(peer_required=True, execute_task=True)
    async def new_peak(self, request: beacon_protocol.NewPeak, peer: WSBpxConnection) -> None:
        """
        A peer notifies us that they have added a new peak to their blockchain. If we don't have it,
        we can ask for it.
        """
        # this semaphore limits the number of tasks that can call new_peak() at
        # the same time, since it can be expensive
        try:
            async with self.beacon.new_peak_sem.acquire():
                await self.beacon.new_peak(request, peer)
        except LimitedSemaphoreFullError:
            return None

        return None

    @api_request(reply_types=[ProtocolMessageTypes.respond_proof_of_weight])
    async def request_proof_of_weight(self, request: beacon_protocol.RequestProofOfWeight) -> Optional[Message]:
        if self.beacon.weight_proof_handler is None:
            return None
        if not self.beacon.blockchain.contains_block(request.tip):
            self.log.error(f"got weight proof request for unknown peak {request.tip}")
            return None
        if request.tip in self.beacon.pow_creation:
            event = self.beacon.pow_creation[request.tip]
            await event.wait()
            wp = await self.beacon.weight_proof_handler.get_proof_of_weight(request.tip)
        else:
            event = asyncio.Event()
            self.beacon.pow_creation[request.tip] = event
            wp = await self.beacon.weight_proof_handler.get_proof_of_weight(request.tip)
            event.set()
        tips = list(self.beacon.pow_creation.keys())

        if len(tips) > 4:
            # Remove old from cache
            for i in range(0, 4):
                self.beacon.pow_creation.pop(tips[i])

        if wp is None:
            self.log.error(f"failed creating weight proof for peak {request.tip}")
            return None

        # Serialization of wp is slow
        if (
            self.beacon.beacon_store.serialized_wp_message_tip is not None
            and self.beacon.beacon_store.serialized_wp_message_tip == request.tip
        ):
            return self.beacon.beacon_store.serialized_wp_message
        message = make_msg(
            ProtocolMessageTypes.respond_proof_of_weight, beacon_protocol.RespondProofOfWeight(wp, request.tip)
        )
        self.beacon.beacon_store.serialized_wp_message_tip = request.tip
        self.beacon.beacon_store.serialized_wp_message = message
        return message

    @api_request()
    async def respond_proof_of_weight(self, request: beacon_protocol.RespondProofOfWeight) -> Optional[Message]:
        self.log.warning("Received proof of weight too late.")
        return None

    @api_request(reply_types=[ProtocolMessageTypes.respond_block, ProtocolMessageTypes.reject_block])
    async def request_block(self, request: beacon_protocol.RequestBlock) -> Optional[Message]:
        if not self.beacon.blockchain.contains_height(request.height):
            reject = RejectBlock(request.height)
            msg = make_msg(ProtocolMessageTypes.reject_block, reject)
            return msg
        header_hash: Optional[bytes32] = self.beacon.blockchain.height_to_hash(request.height)
        if header_hash is None:
            return make_msg(ProtocolMessageTypes.reject_block, RejectBlock(request.height))

        block: Optional[FullBlock] = await self.beacon.block_store.get_full_block(header_hash)
        if block is not None:
            return make_msg(ProtocolMessageTypes.respond_block, beacon_protocol.RespondBlock(block))
        return make_msg(ProtocolMessageTypes.reject_block, RejectBlock(request.height))

    @api_request(reply_types=[ProtocolMessageTypes.respond_blocks, ProtocolMessageTypes.reject_blocks])
    async def request_blocks(self, request: beacon_protocol.RequestBlocks) -> Optional[Message]:
        # note that we treat the request range as *inclusive*, but we check the
        # size before we bump end_height. So MAX_BLOCK_COUNT_PER_REQUESTS is off
        # by one
        if (
            request.end_height < request.start_height
            or request.end_height - request.start_height > self.beacon.constants.MAX_BLOCK_COUNT_PER_REQUESTS
        ):
            reject = RejectBlocks(request.start_height, request.end_height)
            msg: Message = make_msg(ProtocolMessageTypes.reject_blocks, reject)
            return msg
        for i in range(request.start_height, request.end_height + 1):
            if not self.beacon.blockchain.contains_height(uint32(i)):
                reject = RejectBlocks(request.start_height, request.end_height)
                msg = make_msg(ProtocolMessageTypes.reject_blocks, reject)
                return msg

        blocks_bytes: List[bytes] = []
        for i in range(request.start_height, request.end_height + 1):
            header_hash_i = self.beacon.blockchain.height_to_hash(uint32(i))
            if header_hash_i is None:
                reject = RejectBlocks(request.start_height, request.end_height)
                return make_msg(ProtocolMessageTypes.reject_blocks, reject)
            block_bytes: Optional[bytes] = await self.beacon.block_store.get_full_block_bytes(header_hash_i)
            if block_bytes is None:
                reject = RejectBlocks(request.start_height, request.end_height)
                msg = make_msg(ProtocolMessageTypes.reject_blocks, reject)
                return msg

            blocks_bytes.append(block_bytes)

        respond_blocks_manually_streamed: bytes = (
            bytes(uint32(request.start_height))
            + bytes(uint32(request.end_height))
            + len(blocks_bytes).to_bytes(4, "big", signed=False)
        )
        for block_bytes in blocks_bytes:
            respond_blocks_manually_streamed += block_bytes
        msg = make_msg(ProtocolMessageTypes.respond_blocks, respond_blocks_manually_streamed)

        return msg

    @api_request()
    async def reject_block(self, request: beacon_protocol.RejectBlock) -> None:
        self.log.debug(f"reject_block {request.height}")

    @api_request()
    async def reject_blocks(self, request: beacon_protocol.RejectBlocks) -> None:
        self.log.debug(f"reject_blocks {request.start_height} {request.end_height}")

    @api_request()
    async def respond_blocks(self, request: beacon_protocol.RespondBlocks) -> None:
        self.log.warning("Received unsolicited/late blocks")
        return None

    @api_request(peer_required=True)
    async def respond_block(
        self,
        respond_block: beacon_protocol.RespondBlock,
        peer: WSBpxConnection,
    ) -> Optional[Message]:
        """
        Receive a full block from a peer beacon client (or ourselves).
        """

        self.log.warning(f"Received unsolicited/late block from peer {peer.get_peer_logging()}")
        return None

    @api_request()
    async def new_unfinished_block(
        self, new_unfinished_block: beacon_protocol.NewUnfinishedBlock
    ) -> Optional[Message]:
        # Ignore if syncing
        if self.beacon.sync_store.get_sync_mode() or self.beacon.execution_client.syncing:
            return None
        block_hash = new_unfinished_block.unfinished_reward_hash
        if self.beacon.beacon_store.get_unfinished_block(block_hash) is not None:
            return None

        # This prevents us from downloading the same block from many peers
        if block_hash in self.beacon.beacon_store.requesting_unfinished_blocks:
            return None

        msg = make_msg(
            ProtocolMessageTypes.request_unfinished_block,
            beacon_protocol.RequestUnfinishedBlock(block_hash),
        )
        self.beacon.beacon_store.requesting_unfinished_blocks.add(block_hash)

        # However, we want to eventually download from other peers, if this peer does not respond
        # Todo: keep track of who it was
        async def eventually_clear() -> None:
            await asyncio.sleep(5)
            if block_hash in self.beacon.beacon_store.requesting_unfinished_blocks:
                self.beacon.beacon_store.requesting_unfinished_blocks.remove(block_hash)

        asyncio.create_task(eventually_clear())

        return msg

    @api_request(reply_types=[ProtocolMessageTypes.respond_unfinished_block])
    async def request_unfinished_block(
        self, request_unfinished_block: beacon_protocol.RequestUnfinishedBlock
    ) -> Optional[Message]:
        unfinished_block: Optional[UnfinishedBlock] = self.beacon.beacon_store.get_unfinished_block(
            request_unfinished_block.unfinished_reward_hash
        )
        if unfinished_block is not None:
            msg = make_msg(
                ProtocolMessageTypes.respond_unfinished_block,
                beacon_protocol.RespondUnfinishedBlock(unfinished_block),
            )
            return msg
        return None

    @api_request(peer_required=True, bytes_required=True)
    async def respond_unfinished_block(
        self,
        respond_unfinished_block: beacon_protocol.RespondUnfinishedBlock,
        peer: WSBpxConnection,
        respond_unfinished_block_bytes: bytes = b"",
    ) -> Optional[Message]:
        if self.beacon.sync_store.get_sync_mode():
            return None
        await self.beacon.add_unfinished_block(
            respond_unfinished_block.unfinished_block, peer, block_bytes=respond_unfinished_block_bytes
        )
        return None

    @api_request(peer_required=True)
    async def new_signage_point_or_end_of_sub_slot(
        self, new_sp: beacon_protocol.NewSignagePointOrEndOfSubSlot, peer: WSBpxConnection
    ) -> Optional[Message]:
        # Ignore if syncing
        if self.beacon.sync_store.get_sync_mode() or self.beacon.execution_client.syncing:
            return None
        if (
            self.beacon.beacon_store.get_signage_point_by_index(
                new_sp.challenge_hash,
                new_sp.index_from_challenge,
                new_sp.last_rc_infusion,
            )
            is not None
        ):
            return None
        if self.beacon.beacon_store.have_newer_signage_point(
            new_sp.challenge_hash, new_sp.index_from_challenge, new_sp.last_rc_infusion
        ):
            return None

        if new_sp.index_from_challenge == 0 and new_sp.prev_challenge_hash is not None:
            if self.beacon.beacon_store.get_sub_slot(new_sp.prev_challenge_hash) is None:
                collected_eos = []
                challenge_hash_to_request = new_sp.challenge_hash
                last_rc = new_sp.last_rc_infusion
                num_non_empty_sub_slots_seen = 0
                for _ in range(30):
                    if num_non_empty_sub_slots_seen >= 3:
                        self.log.debug("Diverged from peer. Don't have the same blocks")
                        return None
                    # If this is an end of sub slot, and we don't have the prev, request the prev instead
                    # We want to catch up to the latest slot so we can receive signage points
                    beacon_request = beacon_protocol.RequestSignagePointOrEndOfSubSlot(
                        challenge_hash_to_request, uint8(0), last_rc
                    )
                    response = await peer.call_api(
                        BeaconAPI.request_signage_point_or_end_of_sub_slot, beacon_request, timeout=10
                    )
                    if not isinstance(response, beacon_protocol.RespondEndOfSubSlot):
                        self.beacon.log.debug(f"Invalid response for slot {response}")
                        return None
                    collected_eos.append(response)
                    if (
                        self.beacon.beacon_store.get_sub_slot(
                            response.end_of_slot_bundle.challenge_chain.challenge_chain_end_of_slot_vdf.challenge
                        )
                        is not None
                        or response.end_of_slot_bundle.challenge_chain.challenge_chain_end_of_slot_vdf.challenge
                        == self.beacon.constants.GENESIS_CHALLENGE
                    ):
                        for eos in reversed(collected_eos):
                            await self.respond_end_of_sub_slot(eos, peer)
                        return None
                    if (
                        response.end_of_slot_bundle.challenge_chain.challenge_chain_end_of_slot_vdf.number_of_iterations
                        != response.end_of_slot_bundle.reward_chain.end_of_slot_vdf.number_of_iterations
                    ):
                        num_non_empty_sub_slots_seen += 1
                    challenge_hash_to_request = (
                        response.end_of_slot_bundle.challenge_chain.challenge_chain_end_of_slot_vdf.challenge
                    )
                    last_rc = response.end_of_slot_bundle.reward_chain.end_of_slot_vdf.challenge
                self.beacon.log.warning("Failed to catch up in sub-slots")
                return None

        if new_sp.index_from_challenge > 0:
            if (
                new_sp.challenge_hash != self.beacon.constants.GENESIS_CHALLENGE
                and self.beacon.beacon_store.get_sub_slot(new_sp.challenge_hash) is None
            ):
                # If this is a normal signage point,, and we don't have the end of sub slot, request the end of sub slot
                beacon_request = beacon_protocol.RequestSignagePointOrEndOfSubSlot(
                    new_sp.challenge_hash, uint8(0), new_sp.last_rc_infusion
                )
                return make_msg(ProtocolMessageTypes.request_signage_point_or_end_of_sub_slot, beacon_request)

        # Otherwise (we have the prev or the end of sub slot), request it normally
        beacon_request = beacon_protocol.RequestSignagePointOrEndOfSubSlot(
            new_sp.challenge_hash, new_sp.index_from_challenge, new_sp.last_rc_infusion
        )

        return make_msg(ProtocolMessageTypes.request_signage_point_or_end_of_sub_slot, beacon_request)

    @api_request(reply_types=[ProtocolMessageTypes.respond_signage_point, ProtocolMessageTypes.respond_end_of_sub_slot])
    async def request_signage_point_or_end_of_sub_slot(
        self, request: beacon_protocol.RequestSignagePointOrEndOfSubSlot
    ) -> Optional[Message]:
        if request.index_from_challenge == 0:
            sub_slot: Optional[Tuple[EndOfSubSlotBundle, int, uint128]] = self.beacon.beacon_store.get_sub_slot(
                request.challenge_hash
            )
            if sub_slot is not None:
                return make_msg(
                    ProtocolMessageTypes.respond_end_of_sub_slot,
                    beacon_protocol.RespondEndOfSubSlot(sub_slot[0]),
                )
        else:
            if self.beacon.beacon_store.get_sub_slot(request.challenge_hash) is None:
                if request.challenge_hash != self.beacon.constants.GENESIS_CHALLENGE:
                    self.log.info(f"Don't have challenge hash {request.challenge_hash}")

            sp: Optional[SignagePoint] = self.beacon.beacon_store.get_signage_point_by_index(
                request.challenge_hash,
                request.index_from_challenge,
                request.last_rc_infusion,
            )
            if sp is not None:
                assert (
                    sp.cc_vdf is not None
                    and sp.cc_proof is not None
                    and sp.rc_vdf is not None
                    and sp.rc_proof is not None
                )
                beacon_response = beacon_protocol.RespondSignagePoint(
                    request.index_from_challenge,
                    sp.cc_vdf,
                    sp.cc_proof,
                    sp.rc_vdf,
                    sp.rc_proof,
                )
                return make_msg(ProtocolMessageTypes.respond_signage_point, beacon_response)
            else:
                self.log.info(f"Don't have signage point {request}")
        return None

    @api_request(peer_required=True)
    async def respond_signage_point(
        self, request: beacon_protocol.RespondSignagePoint, peer: WSBpxConnection
    ) -> Optional[Message]:
        if self.beacon.sync_store.get_sync_mode():
            return None
        async with self.beacon.timelord_lock:
            # Already have signage point

            if self.beacon.beacon_store.have_newer_signage_point(
                request.challenge_chain_vdf.challenge,
                request.index_from_challenge,
                request.reward_chain_vdf.challenge,
            ):
                return None
            existing_sp = self.beacon.beacon_store.get_signage_point(
                request.challenge_chain_vdf.output.get_hash()
            )
            if existing_sp is not None and existing_sp.rc_vdf == request.reward_chain_vdf:
                return None
            peak = self.beacon.blockchain.get_peak()
            if peak is not None and peak.height > self.beacon.constants.MAX_SUB_SLOT_BLOCKS:
                next_sub_slot_iters = self.beacon.blockchain.get_next_slot_iters(peak.header_hash, True)
                sub_slots_for_peak = await self.beacon.blockchain.get_sp_and_ip_sub_slots(peak.header_hash)
                assert sub_slots_for_peak is not None
                ip_sub_slot: Optional[EndOfSubSlotBundle] = sub_slots_for_peak[1]
            else:
                sub_slot_iters = self.beacon.constants.SUB_SLOT_ITERS_STARTING
                next_sub_slot_iters = sub_slot_iters
                ip_sub_slot = None

            added = self.beacon.beacon_store.new_signage_point(
                request.index_from_challenge,
                self.beacon.blockchain,
                self.beacon.blockchain.get_peak(),
                next_sub_slot_iters,
                SignagePoint(
                    request.challenge_chain_vdf,
                    request.challenge_chain_proof,
                    request.reward_chain_vdf,
                    request.reward_chain_proof,
                ),
            )

            if added:
                await self.beacon.signage_point_post_processing(request, peer, ip_sub_slot)
            else:
                self.log.debug(
                    f"Signage point {request.index_from_challenge} not added, CC challenge: "
                    f"{request.challenge_chain_vdf.challenge}, RC challenge: {request.reward_chain_vdf.challenge}"
                )

            return None

    @api_request(peer_required=True)
    async def respond_end_of_sub_slot(
        self, request: beacon_protocol.RespondEndOfSubSlot, peer: WSBpxConnection
    ) -> Optional[Message]:
        if self.beacon.sync_store.get_sync_mode():
            return None
        msg, _ = await self.beacon.add_end_of_sub_slot(request.end_of_slot_bundle, peer)
        return msg

    # FARMER PROTOCOL
    @api_request(peer_required=True)
    async def declare_proof_of_space(
        self, request: farmer_protocol.DeclareProofOfSpace, peer: WSBpxConnection
    ) -> Optional[Message]:
        """
        Creates a block body and header, with the proof of space and sends the hash of the header data
        back to the farmer.
        """
        if self.beacon.sync_store.get_sync_mode():
            return None

        async with self.beacon.timelord_lock:
            sp_vdfs: Optional[SignagePoint] = self.beacon.beacon_store.get_signage_point(
                request.challenge_chain_sp
            )

            if sp_vdfs is None:
                self.log.warning(f"Received proof of space for an unknown signage point {request.challenge_chain_sp}")
                return None
            if request.signage_point_index > 0:
                assert sp_vdfs.rc_vdf is not None
                if sp_vdfs.rc_vdf.output.get_hash() != request.reward_chain_sp:
                    self.log.debug(
                        f"Received proof of space for a potentially old signage point {request.challenge_chain_sp}. "
                        f"Current sp: {sp_vdfs.rc_vdf.output.get_hash()}"
                    )
                    return None

            if request.signage_point_index == 0:
                cc_challenge_hash: bytes32 = request.challenge_chain_sp
            else:
                assert sp_vdfs.cc_vdf is not None
                cc_challenge_hash = sp_vdfs.cc_vdf.challenge

            pos_sub_slot: Optional[Tuple[EndOfSubSlotBundle, int, uint128]] = None
            if request.challenge_hash != self.beacon.constants.GENESIS_CHALLENGE:
                # Checks that the proof of space is a response to a recent challenge and valid SP
                pos_sub_slot = self.beacon.beacon_store.get_sub_slot(cc_challenge_hash)
                if pos_sub_slot is None:
                    self.log.warning(f"Received proof of space for an unknown sub slot: {request}")
                    return None
                total_iters_pos_slot: uint128 = pos_sub_slot[2]
            else:
                total_iters_pos_slot = uint128(0)
            assert cc_challenge_hash == request.challenge_hash

            # Now we know that the proof of space has a signage point either:
            # 1. In the previous sub-slot of the peak (overflow)
            # 2. In the same sub-slot as the peak
            # 3. In a future sub-slot that we already know of

            # Checks that the proof of space is valid
            quality_string: Optional[bytes32] = verify_and_get_quality_string(
                request.proof_of_space, self.beacon.constants, cc_challenge_hash, request.challenge_chain_sp
            )
            assert quality_string is not None and len(quality_string) == 32
            
            async with self.beacon._blockchain_lock_high_priority:
                peak: Optional[BlockRecord] = self.beacon.blockchain.get_peak()

            def get_plot_sig(to_sign: bytes32, _extra: G1Element) -> G2Element:
                if to_sign == request.challenge_chain_sp:
                    return request.challenge_chain_sp_signature
                elif to_sign == request.reward_chain_sp:
                    return request.reward_chain_sp_signature
                return G2Element()

            prev_b: Optional[BlockRecord] = peak

            # Finds the previous block from the signage point, ensuring that the reward chain VDF is correct
            if prev_b is not None:
                if request.signage_point_index == 0:
                    if pos_sub_slot is None:
                        self.log.warning("Pos sub slot is None")
                        return None
                    rc_challenge = pos_sub_slot[0].reward_chain.end_of_slot_vdf.challenge
                else:
                    assert sp_vdfs.rc_vdf is not None
                    rc_challenge = sp_vdfs.rc_vdf.challenge

                # Backtrack through empty sub-slots
                for eos, _, _ in reversed(self.beacon.beacon_store.finished_sub_slots):
                    if eos is not None and eos.reward_chain.get_hash() == rc_challenge:
                        rc_challenge = eos.reward_chain.end_of_slot_vdf.challenge

                found = False
                attempts = 0
                while prev_b is not None and attempts < 10:
                    if prev_b.reward_infusion_new_challenge == rc_challenge:
                        found = True
                        break
                    if prev_b.finished_reward_slot_hashes is not None and len(prev_b.finished_reward_slot_hashes) > 0:
                        if prev_b.finished_reward_slot_hashes[-1] == rc_challenge:
                            # This block includes a sub-slot which is where our SP vdf starts. Go back one more
                            # to find the prev block
                            prev_b = self.beacon.blockchain.try_block_record(prev_b.prev_hash)
                            found = True
                            break
                    prev_b = self.beacon.blockchain.try_block_record(prev_b.prev_hash)
                    attempts += 1
                if not found:
                    self.log.warning("Did not find a previous block with the correct reward chain hash")
                    return None

            try:
                finished_sub_slots: Optional[
                    List[EndOfSubSlotBundle]
                ] = self.beacon.beacon_store.get_finished_sub_slots(
                    self.beacon.blockchain, prev_b, cc_challenge_hash
                )
                if finished_sub_slots is None:
                    return None

                if (
                    len(finished_sub_slots) > 0
                    and pos_sub_slot is not None
                    and finished_sub_slots[-1] != pos_sub_slot[0]
                ):
                    self.log.error("Have different sub-slots than is required to farm this block")
                    return None
            except ValueError as e:
                self.log.warning(f"Value Error: {e}")
                return None

            if peak is None or peak.height <= self.beacon.constants.MAX_SUB_SLOT_BLOCKS:
                difficulty = self.beacon.constants.DIFFICULTY_STARTING
                sub_slot_iters = self.beacon.constants.SUB_SLOT_ITERS_STARTING
            else:
                difficulty = uint64(peak.weight - self.beacon.blockchain.block_record(peak.prev_hash).weight)
                sub_slot_iters = peak.sub_slot_iters
                for sub_slot in finished_sub_slots:
                    if sub_slot.challenge_chain.new_difficulty is not None:
                        difficulty = sub_slot.challenge_chain.new_difficulty
                    if sub_slot.challenge_chain.new_sub_slot_iters is not None:
                        sub_slot_iters = sub_slot.challenge_chain.new_sub_slot_iters

            required_iters: uint64 = calculate_iterations_quality(
                self.beacon.constants.DIFFICULTY_CONSTANT_FACTOR,
                quality_string,
                request.proof_of_space.size,
                difficulty,
                request.challenge_chain_sp,
            )
            sp_iters: uint64 = calculate_sp_iters(self.beacon.constants, sub_slot_iters, request.signage_point_index)
            ip_iters: uint64 = calculate_ip_iters(
                self.beacon.constants,
                sub_slot_iters,
                request.signage_point_index,
                required_iters,
            )
            
            # The block's timestamp must be greater than the previous transaction block's timestamp
            timestamp = uint64(int(time.time()))
            curr: Optional[BlockRecord] = prev_b
            while curr is not None and not curr.is_transaction_block and curr.height != 0:
                curr = self.beacon.blockchain.try_block_record(curr.prev_hash)
            if curr is not None:
                assert curr.timestamp is not None
                if timestamp <= curr.timestamp:
                    timestamp = uint64(int(curr.timestamp + 1))

            self.log.info("Starting to make the unfinished block")
            unfinished_block: UnfinishedBlock = create_unfinished_block(
                self.beacon.constants,
                self.beacon.execution_client,
                total_iters_pos_slot,
                sub_slot_iters,
                request.signage_point_index,
                sp_iters,
                ip_iters,
                request.proof_of_space,
                cc_challenge_hash,
                bytes20.from_hexstr(self.beacon.config["coinbase"]),
                get_plot_sig,
                sp_vdfs,
                timestamp,
                self.beacon.blockchain,
                b"",
                prev_b,
                finished_sub_slots,
            )
            self.log.info("Made the unfinished block")
            if prev_b is not None:
                height: uint32 = uint32(prev_b.height + 1)
            else:
                height = uint32(0)
            self.beacon.beacon_store.add_candidate_block(quality_string, height, unfinished_block)

            foliage_sb_data_hash = unfinished_block.foliage.foliage_block_data.get_hash()
            if unfinished_block.is_transaction_block():
                foliage_transaction_block_hash = unfinished_block.foliage.foliage_transaction_block_hash
            else:
                foliage_transaction_block_hash = bytes32([0] * 32)
            assert foliage_transaction_block_hash is not None

            message = farmer_protocol.RequestSignedValues(
                quality_string,
                foliage_sb_data_hash,
                foliage_transaction_block_hash,
            )
            await peer.send_message(make_msg(ProtocolMessageTypes.request_signed_values, message))

        return None

    @api_request(peer_required=True)
    async def signed_values(
        self, farmer_request: farmer_protocol.SignedValues, peer: WSBpxConnection
    ) -> Optional[Message]:
        """
        Signature of header hash, by the harvester. This is enough to create an unfinished
        block, which only needs a Proof of Time to be finished. If the signature is valid,
        we call the unfinished_block routine.
        """
        candidate_tuple: Optional[Tuple[uint32, UnfinishedBlock]] = self.beacon.beacon_store.get_candidate_block(
            farmer_request.quality_string
        )

        if candidate_tuple is None:
            self.log.warning(f"Quality string {farmer_request.quality_string} not found in database")
            return None
        height, candidate = candidate_tuple

        if not AugSchemeMPL.verify(
            candidate.reward_chain_block.proof_of_space.plot_public_key,
            candidate.foliage.foliage_block_data.get_hash(),
            farmer_request.foliage_block_data_signature,
        ):
            self.log.warning("Signature not valid. There might be a collision in plots. Ignore this during tests.")
            return None

        fsb2 = dataclasses.replace(
            candidate.foliage,
            foliage_block_data_signature=farmer_request.foliage_block_data_signature,
        )
        if candidate.is_transaction_block():
            fsb2 = dataclasses.replace(
                fsb2, foliage_transaction_block_signature=farmer_request.foliage_transaction_block_signature
            )

        new_candidate = dataclasses.replace(candidate, foliage=fsb2)

        # Propagate to ourselves (which validates and does further propagations)
        try:
            await self.beacon.add_unfinished_block(new_candidate, None, True)
        except Exception as e:
	        self.beacon.log.error(f"Error farming block {e} {new_candidate}")
            
        return None

    # TIMELORD PROTOCOL
    @api_request(peer_required=True)
    async def new_infusion_point_vdf(
        self, request: timelord_protocol.NewInfusionPointVDF, peer: WSBpxConnection
    ) -> Optional[Message]:
        if self.beacon.sync_store.get_sync_mode():
            return None
        # Lookup unfinished blocks
        async with self.beacon.timelord_lock:
            return await self.beacon.new_infusion_point_vdf(request, peer)

    @api_request(peer_required=True)
    async def new_signage_point_vdf(
        self, request: timelord_protocol.NewSignagePointVDF, peer: WSBpxConnection
    ) -> None:
        if self.beacon.sync_store.get_sync_mode():
            return None

        beacon_message = beacon_protocol.RespondSignagePoint(
            request.index_from_challenge,
            request.challenge_chain_sp_vdf,
            request.challenge_chain_sp_proof,
            request.reward_chain_sp_vdf,
            request.reward_chain_sp_proof,
        )
        await self.respond_signage_point(beacon_message, peer)

    @api_request(peer_required=True)
    async def new_end_of_sub_slot_vdf(
        self, request: timelord_protocol.NewEndOfSubSlotVDF, peer: WSBpxConnection
    ) -> Optional[Message]:
        if self.beacon.sync_store.get_sync_mode():
            return None
        if (
            self.beacon.beacon_store.get_sub_slot(request.end_of_sub_slot_bundle.challenge_chain.get_hash())
            is not None
        ):
            return None
        # Calls our own internal message to handle the end of sub slot, and potentially broadcasts to other peers.
        msg, added = await self.beacon.add_end_of_sub_slot(request.end_of_sub_slot_bundle, peer)
        if not added:
            self.log.error(
                f"Was not able to add end of sub-slot: "
                f"{request.end_of_sub_slot_bundle.challenge_chain.challenge_chain_end_of_slot_vdf.challenge}. "
                f"Re-sending new-peak to timelord"
            )
            await self.beacon.send_peak_to_timelords(peer=peer)
            return None
        else:
            return msg

    @api_request()
    async def respond_compact_proof_of_time(self, request: timelord_protocol.RespondCompactProofOfTime) -> None:
        if self.beacon.sync_store.get_sync_mode():
            return None
        await self.beacon.add_compact_proof_of_time(request)
        return None

    @api_request(peer_required=True, bytes_required=True, execute_task=True)
    async def new_compact_vdf(
        self, request: beacon_protocol.NewCompactVDF, peer: WSBpxConnection, request_bytes: bytes = b""
    ) -> None:
        if self.beacon.sync_store.get_sync_mode():
            return None

        name = std_hash(request_bytes)
        if name in self.beacon.compact_vdf_requests:
            self.log.debug(f"Ignoring NewCompactVDF: {request}, already requested")
            return None
        self.beacon.compact_vdf_requests.add(name)

        # this semaphore will only allow a limited number of tasks call
        # new_compact_vdf() at a time, since it can be expensive
        try:
            async with self.beacon.compact_vdf_sem.acquire():
                try:
                    await self.beacon.new_compact_vdf(request, peer)
                finally:
                    self.beacon.compact_vdf_requests.remove(name)
        except LimitedSemaphoreFullError:
            self.log.debug(f"Ignoring NewCompactVDF: {request}, _waiters")
            return None

        return None

    @api_request(peer_required=True, reply_types=[ProtocolMessageTypes.respond_compact_vdf])
    async def request_compact_vdf(self, request: beacon_protocol.RequestCompactVDF, peer: WSBpxConnection) -> None:
        if self.beacon.sync_store.get_sync_mode():
            return None
        await self.beacon.request_compact_vdf(request, peer)
        return None

    @api_request(peer_required=True)
    async def respond_compact_vdf(self, request: beacon_protocol.RespondCompactVDF, peer: WSBpxConnection) -> None:
        if self.beacon.sync_store.get_sync_mode():
            return None
        await self.beacon.add_compact_vdf(request, peer)
        return None
