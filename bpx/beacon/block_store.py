from __future__ import annotations

import dataclasses
import logging
import sqlite3
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import typing_extensions
import zstd

from bpx.consensus.block_record import BlockRecord
from bpx.types.blockchain_format.sized_bytes import bytes32
from bpx.types.full_block import FullBlock
from bpx.types.weight_proof import SubEpochChallengeSegment, SubEpochSegments
from bpx.util.db_wrapper import DbWrapper
from bpx.util.errors import Err
from bpx.util.ints import uint32
from bpx.util.lru_cache import LRUCache

log = logging.getLogger(__name__)


@typing_extensions.final
@dataclasses.dataclass
class BlockStore:
    block_cache: LRUCache[bytes32, FullBlock]
    db_wrapper: DbWrapper
    ses_challenge_cache: LRUCache[bytes32, List[SubEpochChallengeSegment]]

    @classmethod
    async def create(cls, db_wrapper: DbWrapper) -> BlockStore:
        self = cls(LRUCache(1000), db_wrapper, LRUCache(50))

        async with self.db_wrapper.writer_maybe_transaction() as conn:
            log.info("DB: Creating block store tables and indexes.")
            # TODO: most data in block is duplicated in block_record. The only
            # reason for this is that our parsing of a FullBlock is so slow,
            # it's faster to store duplicate data to parse less when we just
            # need the BlockRecord. Once we fix the parsing (and data structure)
            # of FullBlock, this can use less space
            await conn.execute(
                "CREATE TABLE IF NOT EXISTS full_blocks("
                "header_hash blob PRIMARY KEY,"
                "prev_hash blob,"
                "height bigint,"
                "sub_epoch_summary blob,"
                "is_fully_compactified tinyint,"
                "in_main_chain tinyint,"
                "block blob,"
                "block_record blob)"
            )

            # This is a single-row table containing the hash of the current
            # peak. The "key" field is there to make update statements simple
            await conn.execute("CREATE TABLE IF NOT EXISTS current_peak(key int PRIMARY KEY, hash blob)")

            # If any of these indices are altered, they should also be altered
            # in the bpx/cmds/db_upgrade.py file
            log.info("DB: Creating index height")
            await conn.execute("CREATE INDEX IF NOT EXISTS height on full_blocks(height)")

            # Sub epoch segments for weight proofs
            await conn.execute(
                "CREATE TABLE IF NOT EXISTS sub_epoch_segments("
                "ses_block_hash blob PRIMARY KEY,"
                "challenge_segments blob)"
            )

            # If any of these indices are altered, they should also be altered
            # in the bpx/cmds/db_upgrade.py file
            log.info("DB: Creating index is_fully_compactified")
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS is_fully_compactified ON"
                " full_blocks(is_fully_compactified, in_main_chain) WHERE in_main_chain=1"
            )
            log.info("DB: Creating index main_chain")
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS main_chain ON full_blocks(height, in_main_chain) WHERE in_main_chain=1"
            )

        return self

    def maybe_from_hex(self, field: Union[bytes, str]) -> bytes32:
        assert isinstance(field, bytes)
        return bytes32(field)

    def maybe_to_hex(self, field: bytes) -> Any:
        return field

    def compress(self, block: FullBlock) -> bytes:
        ret: bytes = zstd.compress(bytes(block))
        return ret

    def maybe_decompress(self, block_bytes: bytes) -> FullBlock:
        ret: FullBlock = FullBlock.from_bytes(zstd.decompress(block_bytes))
        return ret


    async def rollback(self, height: int) -> None:
        async with self.db_wrapper.writer_maybe_transaction() as conn:
            await conn.execute(
                "UPDATE full_blocks SET in_main_chain=0 WHERE height>? AND in_main_chain=1", (height,)
            )

    async def set_in_chain(self, header_hashes: List[Tuple[bytes32]]) -> None:
        async with self.db_wrapper.writer_maybe_transaction() as conn:
            async with await conn.executemany(
                "UPDATE full_blocks SET in_main_chain=1 WHERE header_hash=?", header_hashes
            ) as cursor:
                if cursor.rowcount != len(header_hashes):
                    raise RuntimeError(f"The blockchain database is corrupt. All of {header_hashes} should exist")

    async def replace_proof(self, header_hash: bytes32, block: FullBlock) -> None:
        assert header_hash == block.header_hash

        block_bytes: bytes
        block_bytes = self.compress(block)

        self.block_cache.put(header_hash, block)

        async with self.db_wrapper.writer_maybe_transaction() as conn:
            await conn.execute(
                "UPDATE full_blocks SET block=?,is_fully_compactified=? WHERE header_hash=?",
                (
                    block_bytes,
                    int(block.is_fully_compactified()),
                    self.maybe_to_hex(header_hash),
                ),
            )

    async def add_full_block(self, header_hash: bytes32, block: FullBlock, block_record: BlockRecord) -> None:
        self.block_cache.put(header_hash, block)

        ses: Optional[bytes] = (
            None
            if block_record.sub_epoch_summary_included is None
            else bytes(block_record.sub_epoch_summary_included)
        )

        async with self.db_wrapper.writer_maybe_transaction() as conn:
            await conn.execute(
                "INSERT OR IGNORE INTO full_blocks VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    header_hash,
                    block.prev_header_hash,
                    block.height,
                    ses,
                    int(block.is_fully_compactified()),
                    False,  # in_main_chain
                    self.compress(block),
                    bytes(block_record),
                ),
            )

    async def persist_sub_epoch_challenge_segments(
        self, ses_block_hash: bytes32, segments: List[SubEpochChallengeSegment]
    ) -> None:
        async with self.db_wrapper.writer_maybe_transaction() as conn:
            await conn.execute(
                "INSERT OR REPLACE INTO sub_epoch_segments VALUES(?, ?)",
                (self.maybe_to_hex(ses_block_hash), bytes(SubEpochSegments(segments))),
            )

    async def get_sub_epoch_challenge_segments(
        self,
        ses_block_hash: bytes32,
    ) -> Optional[List[SubEpochChallengeSegment]]:
        cached: Optional[List[SubEpochChallengeSegment]] = self.ses_challenge_cache.get(ses_block_hash)
        if cached is not None:
            return cached

        async with self.db_wrapper.reader_no_transaction() as conn:
            async with conn.execute(
                "SELECT challenge_segments from sub_epoch_segments WHERE ses_block_hash=?",
                (self.maybe_to_hex(ses_block_hash),),
            ) as cursor:
                row = await cursor.fetchone()

        if row is not None:
            challenge_segments: List[SubEpochChallengeSegment] = SubEpochSegments.from_bytes(row[0]).challenge_segments
            self.ses_challenge_cache.put(ses_block_hash, challenge_segments)
            return challenge_segments
        return None

    def rollback_cache_block(self, header_hash: bytes32) -> None:
        try:
            self.block_cache.remove(header_hash)
        except KeyError:
            # this is best effort. When rolling back, we may not have added the
            # block to the cache yet
            pass

    async def get_full_block(self, header_hash: bytes32) -> Optional[FullBlock]:
        cached: Optional[FullBlock] = self.block_cache.get(header_hash)
        if cached is not None:
            return cached
        async with self.db_wrapper.reader_no_transaction() as conn:
            async with conn.execute(
                "SELECT block from full_blocks WHERE header_hash=?", (self.maybe_to_hex(header_hash),)
            ) as cursor:
                row = await cursor.fetchone()
        if row is not None:
            block = self.maybe_decompress(row[0])
            self.block_cache.put(header_hash, block)
            return block
        return None

    async def get_full_block_bytes(self, header_hash: bytes32) -> Optional[bytes]:
        cached = self.block_cache.get(header_hash)
        if cached is not None:
            return bytes(cached)
        async with self.db_wrapper.reader_no_transaction() as conn:
            async with conn.execute(
                "SELECT block from full_blocks WHERE header_hash=?", (self.maybe_to_hex(header_hash),)
            ) as cursor:
                row = await cursor.fetchone()
        if row is not None:
            ret: bytes = zstd.decompress(row[0])
            return ret

        return None

    async def get_full_blocks_at(self, heights: List[uint32]) -> List[FullBlock]:
        if len(heights) == 0:
            return []

        formatted_str = f'SELECT block from full_blocks WHERE height in ({"?," * (len(heights) - 1)}?)'
        async with self.db_wrapper.reader_no_transaction() as conn:
            async with conn.execute(formatted_str, heights) as cursor:
                ret: List[FullBlock] = []
                for row in await cursor.fetchall():
                    ret.append(self.maybe_decompress(row[0]))
                return ret

    async def get_block_records_by_hash(self, header_hashes: List[bytes32]) -> List[BlockRecord]:
        """
        Returns a list of Block Records, ordered by the same order in which header_hashes are passed in.
        Throws an exception if the blocks are not present
        """
        if len(header_hashes) == 0:
            return []

        all_blocks: Dict[bytes32, BlockRecord] = {}
        async with self.db_wrapper.reader_no_transaction() as conn:
            async with conn.execute(
                "SELECT header_hash,block_record FROM full_blocks "
                f'WHERE header_hash in ({"?," * (len(header_hashes) - 1)}?)',
                header_hashes,
            ) as cursor:
                for row in await cursor.fetchall():
                    header_hash = bytes32(row[0])
                    all_blocks[header_hash] = BlockRecord.from_bytes(row[1])

        ret: List[BlockRecord] = []
        for hh in header_hashes:
            if hh not in all_blocks:
                raise ValueError(f"Header hash {hh} not in the blockchain")
            ret.append(all_blocks[hh])
        return ret

    async def get_blocks_by_hash(self, header_hashes: List[bytes32]) -> List[FullBlock]:
        """
        Returns a list of Full Blocks blocks, ordered by the same order in which header_hashes are passed in.
        Throws an exception if the blocks are not present
        """

        if len(header_hashes) == 0:
            return []

        header_hashes_db: Sequence[Union[bytes32, str]]
        header_hashes_db = header_hashes
        formatted_str = (
            f'SELECT header_hash, block from full_blocks WHERE header_hash in ({"?," * (len(header_hashes_db) - 1)}?)'
        )
        all_blocks: Dict[bytes32, FullBlock] = {}
        async with self.db_wrapper.reader_no_transaction() as conn:
            async with conn.execute(formatted_str, header_hashes_db) as cursor:
                for row in await cursor.fetchall():
                    header_hash = self.maybe_from_hex(row[0])
                    full_block: FullBlock = self.maybe_decompress(row[1])
                    all_blocks[header_hash] = full_block
                    self.block_cache.put(header_hash, full_block)
        ret: List[FullBlock] = []
        for hh in header_hashes:
            if hh not in all_blocks:
                raise ValueError(f"Header hash {hh} not in the blockchain")
            ret.append(all_blocks[hh])
        return ret

    async def get_block_record(self, header_hash: bytes32) -> Optional[BlockRecord]:
        async with self.db_wrapper.reader_no_transaction() as conn:
            async with conn.execute(
                "SELECT block_record FROM full_blocks WHERE header_hash=?",
                (header_hash,),
            ) as cursor:
                row = await cursor.fetchone()
        if row is not None:
            return BlockRecord.from_bytes(row[0])

        return None

    async def get_block_records_in_range(
        self,
        start: int,
        stop: int,
    ) -> Dict[bytes32, BlockRecord]:
        """
        Returns a dictionary with all blocks in range between start and stop
        if present.
        """

        ret: Dict[bytes32, BlockRecord] = {}
        async with self.db_wrapper.reader_no_transaction() as conn:
            async with conn.execute(
                "SELECT header_hash, block_record FROM full_blocks WHERE height >= ? AND height <= ?",
                (start, stop),
            ) as cursor:
                for row in await cursor.fetchall():
                    header_hash = bytes32(row[0])
                    ret[header_hash] = BlockRecord.from_bytes(row[1])
                    
        return ret

    async def get_peak(self) -> Optional[Tuple[bytes32, uint32]]:
            async with self.db_wrapper.reader_no_transaction() as conn:
                async with conn.execute("SELECT hash FROM current_peak WHERE key = 0") as cursor:
                    peak_row = await cursor.fetchone()
            if peak_row is None:
                return None
            async with self.db_wrapper.reader_no_transaction() as conn:
                async with conn.execute("SELECT height FROM full_blocks WHERE header_hash=?", (peak_row[0],)) as cursor:
                    peak_height = await cursor.fetchone()
            if peak_height is None:
                return None
            return bytes32(peak_row[0]), uint32(peak_height[0])

    async def get_block_records_close_to_peak(
        self, blocks_n: int
    ) -> Tuple[Dict[bytes32, BlockRecord], Optional[bytes32]]:
        """
        Returns a dictionary with all blocks that have height >= peak height - blocks_n, as well as the
        peak header hash.
        """

        peak = await self.get_peak()
        if peak is None:
            return {}, None

        ret: Dict[bytes32, BlockRecord] = {}
        async with self.db_wrapper.reader_no_transaction() as conn:
            async with conn.execute(
                "SELECT header_hash, block_record FROM full_blocks WHERE height >= ?",
                (peak[1] - blocks_n,),
            ) as cursor:
                for row in await cursor.fetchall():
                    header_hash = bytes32(row[0])
                    ret[header_hash] = BlockRecord.from_bytes(row[1])

        return ret, peak[0]

    async def set_peak(self, header_hash: bytes32) -> None:
        # We need to be in a sqlite transaction here.
        # Note: we do not commit this to the database yet

        # Note: we use the key field as 0 just to ensure all inserts replace the existing row
        async with self.db_wrapper.writer_maybe_transaction() as conn:
            await conn.execute("INSERT OR REPLACE INTO current_peak VALUES(?, ?)", (0, header_hash))

    async def is_fully_compactified(self, header_hash: bytes32) -> Optional[bool]:
        async with self.db_wrapper.writer_maybe_transaction() as conn:
            async with conn.execute(
                "SELECT is_fully_compactified from full_blocks WHERE header_hash=?", (self.maybe_to_hex(header_hash),)
            ) as cursor:
                row = await cursor.fetchone()
        if row is None:
            return None
        return bool(row[0])

    async def get_random_not_compactified(self, number: int) -> List[int]:
        async with self.db_wrapper.reader_no_transaction() as conn:
            async with conn.execute(
                f"SELECT height FROM full_blocks WHERE in_main_chain=1 AND is_fully_compactified=0 "
                f"ORDER BY RANDOM() LIMIT {number}"
            ) as cursor:
                rows = await cursor.fetchall()

        heights = [int(row[0]) for row in rows]

        return heights

    async def count_compactified_blocks(self) -> int:
        async with self.db_wrapper.reader_no_transaction() as conn:
            async with conn.execute(
                "select count(*) from full_blocks where is_fully_compactified=1 and in_main_chain=1"
            ) as cursor:
                row = await cursor.fetchone()

        assert row is not None

        [count] = row
        return int(count)

    async def count_uncompactified_blocks(self) -> int:
        async with self.db_wrapper.reader_no_transaction() as conn:
            async with conn.execute(
                "select count(*) from full_blocks where is_fully_compactified=0 and in_main_chain=1"
            ) as cursor:
                row = await cursor.fetchone()

        assert row is not None

        [count] = row
        return int(count)
