from __future__ import annotations

from enum import Enum


class ProtocolMessageTypes(Enum):
    # Shared protocol (all services)
    handshake = 1

    # Harvester protocol (harvester <-> farmer)
    harvester_handshake = 2
    new_proof_of_space = 3
    request_signatures = 4
    respond_signatures = 5
    new_signage_point_harvester = 6
    request_plots = 7
    respond_plots = 8
    plot_sync_start = 9
    plot_sync_loaded = 10
    plot_sync_removed = 11
    plot_sync_invalid = 12
    plot_sync_keys_missing = 13
    plot_sync_duplicates = 14
    plot_sync_done = 15
    plot_sync_response = 16

    # Farmer protocol (farmer <-> beacon)
    new_signage_point = 17
    declare_proof_of_space = 18
    request_signed_values = 19
    signed_values = 20
    farming_info = 21

    # Timelord protocol (timelord <-> beacon)
    new_peak_timelord = 22
    new_unfinished_block_timelord = 23
    new_infusion_point_vdf = 24
    new_signage_point_vdf = 25
    new_end_of_sub_slot_vdf = 26
    request_compact_proof_of_time = 27
    respond_compact_proof_of_time = 28

    # Beacon client protocol (beacon <-> beacon)
    new_peak = 29
    request_proof_of_weight = 30
    respond_proof_of_weight = 31
    request_block = 32
    respond_block = 33
    reject_block = 34
    request_blocks = 35
    respond_blocks = 36
    reject_blocks = 37
    new_unfinished_block = 38
    request_unfinished_block = 39
    respond_unfinished_block = 40
    new_signage_point_or_end_of_sub_slot = 41
    request_signage_point_or_end_of_sub_slot = 42
    respond_signage_point = 43
    respond_end_of_sub_slot = 44
    request_compact_vdf = 45
    respond_compact_vdf = 46
    new_compact_vdf = 47
    request_peers = 48
    respond_peers = 49
    none_response = 50

    # Introducer protocol (introducer <-> beacon)
    request_peers_introducer = 51
    respond_peers_introducer = 52
