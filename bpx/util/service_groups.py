from __future__ import annotations

from typing import Generator, KeysView

SERVICES_FOR_GROUP = {
    "all": [
        "bpx_harvester",
        "bpx_timelord_launcher",
        "bpx_timelord",
        "bpx_farmer",
        "bpx_beacon",
    ],
    "beacon": ["bpx_beacon"],
    "harvester": ["bpx_harvester"],
    "farmer": ["bpx_harvester", "bpx_farmer", "bpx_beacon"],
    "farmer-only": ["bpx_farmer"],
    "timelord": ["bpx_timelord_launcher", "bpx_timelord", "bpx_beacon"],
    "timelord-only": ["bpx_timelord"],
    "timelord-launcher-only": ["bpx_timelord_launcher"],
    "introducer": ["bpx_introducer"],
    "crawler": ["bpx_crawler"],
    "seeder": ["bpx_crawler", "bpx_seeder"],
    "seeder-only": ["bpx_seeder"],
}


def all_groups() -> KeysView[str]:
    return SERVICES_FOR_GROUP.keys()


def services_for_groups(groups) -> Generator[str, None, None]:
    for group in groups:
        for service in SERVICES_FOR_GROUP[group]:
            yield service


def validate_service(service: str) -> bool:
    return any(service in _ for _ in SERVICES_FOR_GROUP.values())
