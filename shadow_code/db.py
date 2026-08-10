"""SQLite session persistence for shadow-code.

Stores conversation sessions (messages, metadata) in a local SQLite database.
Uses WAL mode for safe concurrent access and foreign keys for referential integrity.

Database location: ~/.shadow-code/sessions.db

Usage:
    from .db import Database

    db = Database()
    sid = db.create_session("shadow-gemma:latest")
    db.add_message(sid, "user", "Hello")
    db.add_message(sid, "assistant", "Hi there!")
    sessions = db.list_sessions()
    session = db.get_session(sid)
    db.close()
"""

import sqlite3
from datetime import datetime
from pathlib import Path


class DatabaseError(Exception):
    """Raised when a database operation fails."""

    pass


def default_db_path() -> str:
    """Default session database path: ~/.shadow-code/sessions.db."""
    db_dir = Path("~/.shadow-code").expanduser()
    db_dir.mkdir(parents=True, exist_ok=True)
    return str(db_dir / "sessions.db")


class Database:
    """SQLite database for persisting conversation sessions."""

    def __init__(self, db_path: str | None = None):
        """Open (or create) the session database.

        Args:
            db_path: Path to the SQLite database file. Defaults to
                     ~/.shadow-code/sessions.db
        """
        if db_path is None:
            db_path = default_db_path()

        self._path = db_path
        try:
            self._conn = sqlite3.connect(db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._create_tables()
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to open database at {db_path}: {e}") from e

    def _create_tables(self):
        """Create tables if they don't exist."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT    NOT NULL DEFAULT '',
                model      TEXT    NOT NULL,
                created_at TEXT    NOT NULL,
                updated_at TEXT    NOT NULL,
                total_tokens INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id INTEGER NOT NULL,
                role       TEXT    NOT NULL,
                content    TEXT    NOT NULL,
                timestamp  TEXT    NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_messages_session
                ON messages(session_id);
        """)
        self._conn.commit()

    def create_session(self, model: str, name: str = "") -> int:
        """Create a new session and return its ID.

        Args:
            model: The model name used for this session.
            name: Optional human-readable session name.

        Returns:
            The integer ID of the newly created session.
        """
        now = datetime.now().isoformat()
        try:
            cur = self._conn.execute(
                "INSERT INTO sessions (name, model, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (name, model, now, now),
            )
            self._conn.commit()
            return cur.lastrowid or 0
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to create session: {e}") from e

    def add_message(self, session_id: int, role: str, content: str):
        """Add a message to an existing session.

        Args:
            session_id: The session to add the message to.
            role: Message role ("user", "assistant", or "system").
            content: The message text.
        """
        now = datetime.now().isoformat()
        try:
            self._conn.execute(
                "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, role, content, now),
            )
            self._conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to add message: {e}") from e

    def update_session_tokens(self, session_id: int, total_tokens: int):
        """Update the token count for a session.

        Args:
            session_id: The session to update.
            total_tokens: The current total prompt token count.
        """
        try:
            self._conn.execute(
                "UPDATE sessions SET total_tokens = ?, updated_at = ? WHERE id = ?",
                (total_tokens, datetime.now().isoformat(), session_id),
            )
            self._conn.commit()
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to update tokens: {e}") from e

    def get_session(self, session_id: int) -> dict | None:
        """Load a session with all its messages.

        Args:
            session_id: The session to load.

        Returns:
            A dict with session metadata and a 'messages' list, or None if
            the session does not exist.
        """
        try:
            row = self._conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            if row is None:
                return None

            messages = self._conn.execute(
                "SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()

            return {
                "id": row["id"],
                "name": row["name"],
                "model": row["model"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "total_tokens": row["total_tokens"],
                "messages": [
                    {
                        "role": m["role"],
                        "content": m["content"],
                        "timestamp": m["timestamp"],
                    }
                    for m in messages
                ],
            }
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to get session {session_id}: {e}") from e

    def list_sessions(self, limit: int = 50) -> list[dict]:
        """List session summaries, most recent first.

        Args:
            limit: Maximum number of sessions to return.

        Returns:
            A list of dicts with session metadata (no messages).
        """
        try:
            rows = self._conn.execute(
                "SELECT s.id, s.name, s.model, s.created_at, s.updated_at, "
                "s.total_tokens, COUNT(m.id) AS message_count "
                "FROM sessions s LEFT JOIN messages m ON s.id = m.session_id "
                "GROUP BY s.id ORDER BY s.updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

            return [
                {
                    "id": r["id"],
                    "name": r["name"],
                    "model": r["model"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                    "total_tokens": r["total_tokens"],
                    "message_count": r["message_count"],
                }
                for r in rows
            ]
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to list sessions: {e}") from e

    def delete_session(self, session_id: int) -> bool:
        """Delete a session and all its messages.

        Args:
            session_id: The session to delete.

        Returns:
            True if the session existed and was deleted, False otherwise.
        """
        try:
            cur = self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            self._conn.commit()
            return cur.rowcount > 0
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to delete session {session_id}: {e}") from e

    def rename_session(self, session_id: int, name: str) -> bool:
        """Rename a session.

        Args:
            session_id: The session to rename.
            name: The new name.

        Returns:
            True if the session existed and was renamed, False otherwise.
        """
        try:
            cur = self._conn.execute(
                "UPDATE sessions SET name = ?, updated_at = ? WHERE id = ?",
                (name, datetime.now().isoformat(), session_id),
            )
            self._conn.commit()
            return cur.rowcount > 0
        except sqlite3.Error as e:
            raise DatabaseError(f"Failed to rename session {session_id}: {e}") from e

    def close(self):
        """Close the database connection."""
        import contextlib

        with contextlib.suppress(sqlite3.Error):
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
