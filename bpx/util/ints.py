from __future__ import annotations

from bpx.util.struct_stream import StructStream, parse_metadata_from_name


@parse_metadata_from_name
class uint8(StructStream):
    pass


@parse_metadata_from_name
class int16(StructStream):
    pass


@parse_metadata_from_name
class uint16(StructStream):
    pass


@parse_metadata_from_name
class uint32(StructStream):
    pass


@parse_metadata_from_name
class uint64(StructStream):
    pass


@parse_metadata_from_name
class uint128(StructStream):
    pass


@parse_metadata_from_name
class uint256(StructStream):
    pass
