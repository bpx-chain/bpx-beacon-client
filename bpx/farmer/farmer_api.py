from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from blspy import AugSchemeMPL, G2Element, PrivateKey

from bpx import __version__
from bpx.consensus.pot_iterations import calculate_iterations_quality, calculate_sp_interval_iters
from bpx.farmer.farmer import Farmer
from bpx.harvester.harvester_api import HarvesterAPI
from bpx.protocols import farmer_protocol, harvester_protocol
from bpx.protocols.harvester_protocol import (
    PlotSyncDone,
    PlotSyncPathList,
    PlotSyncPlotList,
    PlotSyncStart,
)
from bpx.protocols.protocol_message_types import ProtocolMessageTypes
from bpx.server.outbound_message import NodeType, make_msg
from bpx.server.server import ssl_context_for_root
from bpx.server.ws_connection import WSBpxConnection
from bpx.ssl.create_ssl import get_mozilla_ca_crt
from bpx.types.blockchain_format.proof_of_space import (
    generate_plot_public_key,
    generate_taproot_sk,
    verify_and_get_quality_string,
)
from bpx.util.api_decorators import api_request
from bpx.util.ints import uint32, uint64


class FarmerAPI:
    farmer: Farmer

    def __init__(self, farmer) -> None:
        self.farmer = farmer

    @api_request(peer_required=True)
    async def new_proof_of_space(self, new_proof_of_space: harvester_protocol.NewProofOfSpace, peer: WSBpxConnection):
        """
        This is a response from the harvester, for a NewChallenge. Here we check if the proof
        of space is sufficiently good, and if so, we ask for the whole proof.
        """
        if new_proof_of_space.sp_hash not in self.farmer.number_of_responses:
            self.farmer.number_of_responses[new_proof_of_space.sp_hash] = 0
            self.farmer.cache_add_time[new_proof_of_space.sp_hash] = uint64(int(time.time()))

        max_pos_per_sp = 5

        if self.farmer.config.get("selected_network") != "mainnet":
            # This is meant to make testnets more stable, when difficulty is very low
            if self.farmer.number_of_responses[new_proof_of_space.sp_hash] > max_pos_per_sp:
                self.farmer.log.info(
                    f"Surpassed {max_pos_per_sp} PoSpace for one SP, no longer submitting PoSpace for signage point "
                    f"{new_proof_of_space.sp_hash}"
                )
                return None

        if new_proof_of_space.sp_hash not in self.farmer.sps:
            self.farmer.log.warning(
                f"Received response for a signage point that we do not have {new_proof_of_space.sp_hash}"
            )
            return None

        sps = self.farmer.sps[new_proof_of_space.sp_hash]
        for sp in sps:
            computed_quality_string = verify_and_get_quality_string(
                new_proof_of_space.proof,
                self.farmer.constants,
                new_proof_of_space.challenge_hash,
                new_proof_of_space.sp_hash,
            )
            if computed_quality_string is None:
                self.farmer.log.error(f"Invalid proof of space {new_proof_of_space.proof}")
                return None

            self.farmer.number_of_responses[new_proof_of_space.sp_hash] += 1

            required_iters: uint64 = calculate_iterations_quality(
                self.farmer.constants.DIFFICULTY_CONSTANT_FACTOR,
                computed_quality_string,
                new_proof_of_space.proof.size,
                sp.difficulty,
                new_proof_of_space.sp_hash,
            )

            # If the iters are good enough to make a block, proceed with the block making flow
            if required_iters < calculate_sp_interval_iters(self.farmer.constants, sp.sub_slot_iters):
                # Proceed at getting the signatures for this PoSpace
                request = harvester_protocol.RequestSignatures(
                    new_proof_of_space.plot_identifier,
                    new_proof_of_space.challenge_hash,
                    new_proof_of_space.sp_hash,
                    [sp.challenge_chain_sp, sp.reward_chain_sp],
                )

                if new_proof_of_space.sp_hash not in self.farmer.proofs_of_space:
                    self.farmer.proofs_of_space[new_proof_of_space.sp_hash] = []
                self.farmer.proofs_of_space[new_proof_of_space.sp_hash].append(
                    (
                        new_proof_of_space.plot_identifier,
                        new_proof_of_space.proof,
                    )
                )
                self.farmer.cache_add_time[new_proof_of_space.sp_hash] = uint64(int(time.time()))
                self.farmer.quality_str_to_identifiers[computed_quality_string] = (
                    new_proof_of_space.plot_identifier,
                    new_proof_of_space.challenge_hash,
                    new_proof_of_space.sp_hash,
                    peer.peer_node_id,
                )
                self.farmer.cache_add_time[computed_quality_string] = uint64(int(time.time()))

                await peer.send_message(make_msg(ProtocolMessageTypes.request_signatures, request))
                return

    @api_request()
    async def respond_signatures(self, response: harvester_protocol.RespondSignatures):
        """
        There are two cases: receiving signatures for sps, or receiving signatures for the block.
        """
        if response.sp_hash not in self.farmer.sps:
            self.farmer.log.warning(f"Do not have challenge hash {response.challenge_hash}")
            return None
        is_sp_signatures: bool = False
        sps = self.farmer.sps[response.sp_hash]
        signage_point_index = sps[0].signage_point_index
        found_sp_hash_debug = False
        for sp_candidate in sps:
            if response.sp_hash == response.message_signatures[0][0]:
                found_sp_hash_debug = True
                if sp_candidate.reward_chain_sp == response.message_signatures[1][0]:
                    is_sp_signatures = True
        if found_sp_hash_debug:
            assert is_sp_signatures

        pospace = None
        for plot_identifier, candidate_pospace in self.farmer.proofs_of_space[response.sp_hash]:
            if plot_identifier == response.plot_identifier:
                pospace = candidate_pospace
        assert pospace is not None
        include_taproot: bool = pospace.pool_contract_puzzle_hash is not None

        computed_quality_string = verify_and_get_quality_string(
            pospace, self.farmer.constants, response.challenge_hash, response.sp_hash
        )
        if computed_quality_string is None:
            self.farmer.log.warning(f"Have invalid PoSpace {pospace}")
            return None

        if is_sp_signatures:
            (
                challenge_chain_sp,
                challenge_chain_sp_harv_sig,
            ) = response.message_signatures[0]
            reward_chain_sp, reward_chain_sp_harv_sig = response.message_signatures[1]
            for sk in self.farmer.get_private_keys():
                pk = sk.get_g1()
                if pk == response.farmer_pk:
                    agg_pk = generate_plot_public_key(response.local_pk, pk, include_taproot)
                    assert agg_pk == pospace.plot_public_key
                    if include_taproot:
                        taproot_sk: PrivateKey = generate_taproot_sk(response.local_pk, pk)
                        taproot_share_cc_sp: G2Element = AugSchemeMPL.sign(taproot_sk, challenge_chain_sp, agg_pk)
                        taproot_share_rc_sp: G2Element = AugSchemeMPL.sign(taproot_sk, reward_chain_sp, agg_pk)
                    else:
                        taproot_share_cc_sp = G2Element()
                        taproot_share_rc_sp = G2Element()
                    farmer_share_cc_sp = AugSchemeMPL.sign(sk, challenge_chain_sp, agg_pk)
                    agg_sig_cc_sp = AugSchemeMPL.aggregate(
                        [challenge_chain_sp_harv_sig, farmer_share_cc_sp, taproot_share_cc_sp]
                    )
                    assert AugSchemeMPL.verify(agg_pk, challenge_chain_sp, agg_sig_cc_sp)

                    # This means it passes the sp filter
                    farmer_share_rc_sp = AugSchemeMPL.sign(sk, reward_chain_sp, agg_pk)
                    agg_sig_rc_sp = AugSchemeMPL.aggregate(
                        [reward_chain_sp_harv_sig, farmer_share_rc_sp, taproot_share_rc_sp]
                    )
                    assert AugSchemeMPL.verify(agg_pk, reward_chain_sp, agg_sig_rc_sp)

                    request = farmer_protocol.DeclareProofOfSpace(
                        response.challenge_hash,
                        challenge_chain_sp,
                        signage_point_index,
                        reward_chain_sp,
                        pospace,
                        agg_sig_cc_sp,
                        agg_sig_rc_sp,
                    )
                    self.farmer.state_changed("proof", {"proof": request, "passed_filter": True})
                    msg = make_msg(ProtocolMessageTypes.declare_proof_of_space, request)
                    await self.farmer.server.send_to_all([msg], NodeType.BEACON)
                    return None

        else:
            # This is a response with block signatures
            for sk in self.farmer.get_private_keys():
                (
                    foliage_block_data_hash,
                    foliage_sig_harvester,
                ) = response.message_signatures[0]
                (
                    foliage_transaction_block_hash,
                    foliage_transaction_block_sig_harvester,
                ) = response.message_signatures[1]
                pk = sk.get_g1()
                if pk == response.farmer_pk:
                    agg_pk = generate_plot_public_key(response.local_pk, pk, include_taproot)
                    assert agg_pk == pospace.plot_public_key
                    if include_taproot:
                        taproot_sk = generate_taproot_sk(response.local_pk, pk)
                        foliage_sig_taproot: G2Element = AugSchemeMPL.sign(taproot_sk, foliage_block_data_hash, agg_pk)
                        foliage_transaction_block_sig_taproot: G2Element = AugSchemeMPL.sign(
                            taproot_sk, foliage_transaction_block_hash, agg_pk
                        )
                    else:
                        foliage_sig_taproot = G2Element()
                        foliage_transaction_block_sig_taproot = G2Element()

                    foliage_sig_farmer = AugSchemeMPL.sign(sk, foliage_block_data_hash, agg_pk)
                    foliage_transaction_block_sig_farmer = AugSchemeMPL.sign(sk, foliage_transaction_block_hash, agg_pk)

                    foliage_agg_sig = AugSchemeMPL.aggregate(
                        [foliage_sig_harvester, foliage_sig_farmer, foliage_sig_taproot]
                    )
                    foliage_block_agg_sig = AugSchemeMPL.aggregate(
                        [
                            foliage_transaction_block_sig_harvester,
                            foliage_transaction_block_sig_farmer,
                            foliage_transaction_block_sig_taproot,
                        ]
                    )
                    assert AugSchemeMPL.verify(agg_pk, foliage_block_data_hash, foliage_agg_sig)
                    assert AugSchemeMPL.verify(agg_pk, foliage_transaction_block_hash, foliage_block_agg_sig)

                    request_to_nodes = farmer_protocol.SignedValues(
                        computed_quality_string,
                        foliage_agg_sig,
                        foliage_block_agg_sig,
                    )

                    msg = make_msg(ProtocolMessageTypes.signed_values, request_to_nodes)
                    await self.farmer.server.send_to_all([msg], NodeType.BEACON)

    """
    FARMER PROTOCOL (FARMER <-> FULL NODE)
    """

    @api_request()
    async def new_signage_point(self, new_signage_point: farmer_protocol.NewSignagePoint):
        message = harvester_protocol.NewSignagePointHarvester(
            new_signage_point.challenge_hash,
            new_signage_point.difficulty,
            new_signage_point.sub_slot_iters,
            new_signage_point.signage_point_index,
            new_signage_point.challenge_chain_sp,
        )

        msg = make_msg(ProtocolMessageTypes.new_signage_point_harvester, message)
        await self.farmer.server.send_to_all([msg], NodeType.HARVESTER)
        if new_signage_point.challenge_chain_sp not in self.farmer.sps:
            self.farmer.sps[new_signage_point.challenge_chain_sp] = []

        if new_signage_point in self.farmer.sps[new_signage_point.challenge_chain_sp]:
            self.farmer.log.debug(f"Duplicate signage point {new_signage_point.signage_point_index}")
            return

        self.farmer.sps[new_signage_point.challenge_chain_sp].append(new_signage_point)
        self.farmer.cache_add_time[new_signage_point.challenge_chain_sp] = uint64(int(time.time()))
        self.farmer.state_changed("new_signage_point", {"sp_hash": new_signage_point.challenge_chain_sp})

    @api_request()
    async def request_signed_values(self, beacon_request: farmer_protocol.RequestSignedValues):
        if beacon_request.quality_string not in self.farmer.quality_str_to_identifiers:
            self.farmer.log.error(f"Do not have quality string {beacon_request.quality_string}")
            return None

        (plot_identifier, challenge_hash, sp_hash, node_id) = self.farmer.quality_str_to_identifiers[
            beacon_request.quality_string
        ]
        request = harvester_protocol.RequestSignatures(
            plot_identifier,
            challenge_hash,
            sp_hash,
            [beacon_request.foliage_block_data_hash, beacon_request.foliage_transaction_block_hash],
        )

        msg = make_msg(ProtocolMessageTypes.request_signatures, request)
        await self.farmer.server.send_to_specific([msg], node_id)

    @api_request()
    async def farming_info(self, request: farmer_protocol.FarmingInfo):
        self.farmer.state_changed(
            "new_farming_info",
            {
                "farming_info": {
                    "challenge_hash": request.challenge_hash,
                    "signage_point": request.sp_hash,
                    "passed_filter": request.passed,
                    "proofs": request.proofs,
                    "total_plots": request.total_plots,
                    "timestamp": request.timestamp,
                }
            },
        )

    @api_request(peer_required=True)
    async def respond_plots(self, _: harvester_protocol.RespondPlots, peer: WSBpxConnection):
        self.farmer.log.warning(f"Respond plots came too late from: {peer.get_peer_logging()}")

    @api_request(peer_required=True)
    async def plot_sync_start(self, message: PlotSyncStart, peer: WSBpxConnection):
        await self.farmer.plot_sync_receivers[peer.peer_node_id].sync_started(message)

    @api_request(peer_required=True)
    async def plot_sync_loaded(self, message: PlotSyncPlotList, peer: WSBpxConnection):
        await self.farmer.plot_sync_receivers[peer.peer_node_id].process_loaded(message)

    @api_request(peer_required=True)
    async def plot_sync_removed(self, message: PlotSyncPathList, peer: WSBpxConnection):
        await self.farmer.plot_sync_receivers[peer.peer_node_id].process_removed(message)

    @api_request(peer_required=True)
    async def plot_sync_invalid(self, message: PlotSyncPathList, peer: WSBpxConnection):
        await self.farmer.plot_sync_receivers[peer.peer_node_id].process_invalid(message)

    @api_request(peer_required=True)
    async def plot_sync_keys_missing(self, message: PlotSyncPathList, peer: WSBpxConnection):
        await self.farmer.plot_sync_receivers[peer.peer_node_id].process_keys_missing(message)

    @api_request(peer_required=True)
    async def plot_sync_duplicates(self, message: PlotSyncPathList, peer: WSBpxConnection):
        await self.farmer.plot_sync_receivers[peer.peer_node_id].process_duplicates(message)

    @api_request(peer_required=True)
    async def plot_sync_done(self, message: PlotSyncDone, peer: WSBpxConnection):
        await self.farmer.plot_sync_receivers[peer.peer_node_id].sync_done(message)
