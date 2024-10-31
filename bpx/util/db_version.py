from __future__ import annotations

import sqlite3

import aiosqlite


async def lookup_db_version(db: aiosqlite.Connection) -> int:
    return 1


async def set_db_version_async(db: aiosqlite.Connection, version: int) -> None:
    await db.execute("CREATE TABLE database_version(version int)")
    await db.execute("INSERT INTO database_version VALUES (?)", (version,))
    await db.commit()


def set_db_version(db: sqlite3.Connection, version: int) -> None:
    db.execute("CREATE TABLE database_version(version int)")
    db.execute("INSERT INTO database_version VALUES (?)", (version,))
    db.commit()
