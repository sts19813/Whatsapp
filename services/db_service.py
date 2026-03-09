import sqlite3
from pathlib import Path
from typing import Any, Optional

DB_PATH = Path("turismo_bot.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phone TEXT UNIQUE NOT NULL,
                display_name TEXT,
                municipality TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phone TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                message_type TEXT NOT NULL DEFAULT 'text',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phone TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                importance INTEGER NOT NULL DEFAULT 1,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(phone, key)
            );

            CREATE TABLE IF NOT EXISTS location_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phone TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                source TEXT NOT NULL DEFAULT 'whatsapp_location',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS geocode_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT UNIQUE NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                provider TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            """
        )


def upsert_user(phone: str, display_name: Optional[str] = None, municipality: Optional[str] = None) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO users (phone, display_name, municipality)
            VALUES (?, ?, ?)
            ON CONFLICT(phone) DO UPDATE SET
                display_name = COALESCE(excluded.display_name, users.display_name),
                municipality = COALESCE(excluded.municipality, users.municipality),
                updated_at = CURRENT_TIMESTAMP
            """,
            (phone, display_name, municipality),
        )


def set_user_municipality(phone: str, municipality: str) -> None:
    upsert_user(phone=phone, municipality=municipality)


def get_user(phone: str) -> Optional[dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE phone = ?", (phone,)).fetchone()
        return dict(row) if row else None


def save_message(phone: str, role: str, content: str, message_type: str = "text") -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO messages (phone, role, content, message_type) VALUES (?, ?, ?, ?)",
            (phone, role, content, message_type),
        )


def get_recent_messages(phone: str, limit: int = 16) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT role, content, message_type, created_at
            FROM messages
            WHERE phone = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (phone, limit),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


def upsert_memory(phone: str, key: str, value: str, importance: int = 2) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO memories (phone, key, value, importance)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(phone, key) DO UPDATE SET
                value = excluded.value,
                importance = excluded.importance,
                updated_at = CURRENT_TIMESTAMP
            """,
            (phone, key, value, importance),
        )


def get_memories(phone: str, limit: int = 12) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT key, value, importance, updated_at
            FROM memories
            WHERE phone = ?
            ORDER BY importance DESC, updated_at DESC
            LIMIT ?
            """,
            (phone, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def get_memory_value(phone: str, key: str) -> Optional[str]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT value FROM memories WHERE phone = ? AND key = ? LIMIT 1",
            (phone, key),
        ).fetchone()
    return row["value"] if row else None


def save_location(phone: str, latitude: float, longitude: float, source: str = "whatsapp_location") -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO location_history (phone, latitude, longitude, source) VALUES (?, ?, ?, ?)",
            (phone, latitude, longitude, source),
        )


def get_latest_location(phone: str) -> Optional[dict[str, float]]:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT latitude, longitude
            FROM location_history
            WHERE phone = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (phone,),
        ).fetchone()
    return {"lat": row["latitude"], "lon": row["longitude"]} if row else None


def get_cached_geocode(query: str) -> Optional[dict[str, float]]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT latitude, longitude FROM geocode_cache WHERE query = ?",
            (query,),
        ).fetchone()
    return {"lat": row["latitude"], "lon": row["longitude"]} if row else None


def save_geocode_cache(query: str, latitude: float, longitude: float, provider: str) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO geocode_cache (query, latitude, longitude, provider)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(query) DO UPDATE SET
                latitude = excluded.latitude,
                longitude = excluded.longitude,
                provider = excluded.provider,
                updated_at = CURRENT_TIMESTAMP
            """,
            (query, latitude, longitude, provider),
        )
