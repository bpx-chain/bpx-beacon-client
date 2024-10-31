from __future__ import annotations

from typing import Optional

from bpx.protocols import beacon_protocol
from bpx.seeder.crawler import Crawler
from bpx.server.outbound_message import Message
from bpx.server.server import BpxServer
from bpx.server.ws_connection import WSBpxConnection
from bpx.util.api_decorators import api_request


class CrawlerAPI:
    crawler: Crawler

    def __init__(self, crawler):
        self.crawler = crawler

    def __getattr__(self, attr_name: str):
        async def invoke(*args, **kwargs):
            pass

        return invoke

    @property
    def server(self) -> BpxServer:
        assert self.crawler.server is not None
        return self.crawler.server

    @property
    def log(self):
        return self.crawler.log

    @api_request(peer_required=True)
    async def request_peers(self, _request: beacon_protocol.RequestPeers, peer: WSBpxConnection):
        pass

    @api_request(peer_required=True)
    async def respond_peers(
        self, request: beacon_protocol.RespondPeers, peer: WSBpxConnection
    ) -> Optional[Message]:
        pass

    @api_request(peer_required=True)
    async def new_peak(self, request: beacon_protocol.NewPeak, peer: WSBpxConnection) -> Optional[Message]:
        await self.crawler.new_peak(request, peer)
        return None

    @api_request(peer_required=True)
    async def new_signage_point_or_end_of_sub_slot(
        self, new_sp: beacon_protocol.NewSignagePointOrEndOfSubSlot, peer: WSBpxConnection
    ) -> Optional[Message]:
        pass

    @api_request()
    async def new_unfinished_block(
        self, new_unfinished_block: beacon_protocol.NewUnfinishedBlock
    ) -> Optional[Message]:
        pass

    @api_request(peer_required=True)
    async def new_compact_vdf(self, request: beacon_protocol.NewCompactVDF, peer: WSBpxConnection):
        pass

    @api_request()
    async def request_proof_of_weight(self, request: beacon_protocol.RequestProofOfWeight) -> Optional[Message]:
        pass

    @api_request()
    async def request_block(self, request: beacon_protocol.RequestBlock) -> Optional[Message]:
        pass

    @api_request()
    async def request_blocks(self, request: beacon_protocol.RequestBlocks) -> Optional[Message]:
        pass

    @api_request()
    async def request_unfinished_block(
        self, request_unfinished_block: beacon_protocol.RequestUnfinishedBlock
    ) -> Optional[Message]:
        pass

    @api_request()
    async def request_signage_point_or_end_of_sub_slot(
        self, request: beacon_protocol.RequestSignagePointOrEndOfSubSlot
    ) -> Optional[Message]:
        pass
