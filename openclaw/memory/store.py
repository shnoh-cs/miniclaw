"""Vector memory store using SQLite + numpy."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class MemoryChunk:
    """A chunk of text with its embedding and metadata."""

    id: int = 0
    file_path: str = ""
    line_start: int = 0
    line_end: int = 0
    text: str = ""
    embedding: np.ndarray | None = None
    score: float = 0.0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class MemoryStore:
    """SQLite-backed vector store with embedding cache."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._ensure_schema()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _ensure_schema(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                line_start INTEGER NOT NULL,
                line_end INTEGER NOT NULL,
                text TEXT NOT NULL,
                embedding BLOB,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_path);

            CREATE TABLE IF NOT EXISTS embedding_cache (
                text_hash TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS fts_index (
                id INTEGER PRIMARY KEY,
                text TEXT NOT NULL
            );
        """)

        # FTS5 for BM25 keyword search
        try:
            conn.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS fts_search
                USING fts5(text, content='fts_index', content_rowid='id')
            """)
        except sqlite3.OperationalError:
            pass  # FTS5 not available, keyword search will degrade

        conn.commit()

    def upsert_chunk(self, chunk: MemoryChunk) -> int:
        """Insert or update a chunk. Returns the chunk ID."""
        conn = self._get_conn()
        embedding_blob = chunk.embedding.tobytes() if chunk.embedding is not None else None

        if chunk.id > 0:
            conn.execute(
                "UPDATE chunks SET text=?, embedding=?, updated_at=? WHERE id=?",
                (chunk.text, embedding_blob, time.time(), chunk.id),
            )
            # Update FTS
            try:
                conn.execute("DELETE FROM fts_search WHERE rowid=?", (chunk.id,))
                conn.execute("INSERT INTO fts_search(rowid, text) VALUES (?, ?)", (chunk.id, chunk.text))
            except sqlite3.OperationalError:
                pass
            conn.commit()
            return chunk.id

        cursor = conn.execute(
            "INSERT INTO chunks (file_path, line_start, line_end, text, embedding, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (chunk.file_path, chunk.line_start, chunk.line_end, chunk.text,
             embedding_blob, chunk.created_at, chunk.updated_at),
        )
        chunk_id = cursor.lastrowid

        # Update FTS index
        conn.execute(
            "INSERT INTO fts_index (id, text) VALUES (?, ?)",
            (chunk_id, chunk.text),
        )
        try:
            conn.execute(
                "INSERT INTO fts_search(rowid, text) VALUES (?, ?)",
                (chunk_id, chunk.text),
            )
        except sqlite3.OperationalError:
            pass

        conn.commit()
        return chunk_id

    def get_all_embeddings(self) -> list[tuple[int, np.ndarray]]:
        """Get all chunk IDs and their embeddings for vector search."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL"
        ).fetchall()

        results = []
        for row_id, blob in rows:
            if blob:
                emb = np.frombuffer(blob, dtype=np.float32)
                results.append((row_id, emb))
        return results

    def get_chunks_by_ids(self, ids: list[int]) -> list[MemoryChunk]:
        """Get chunks by their IDs."""
        if not ids:
            return []
        conn = self._get_conn()
        placeholders = ",".join("?" for _ in ids)
        rows = conn.execute(
            f"SELECT id, file_path, line_start, line_end, text, created_at, updated_at "
            f"FROM chunks WHERE id IN ({placeholders})",
            ids,
        ).fetchall()

        return [
            MemoryChunk(
                id=r[0], file_path=r[1], line_start=r[2], line_end=r[3],
                text=r[4], created_at=r[5], updated_at=r[6],
            )
            for r in rows
        ]

    def bm25_search(self, query: str, limit: int = 20) -> list[tuple[int, float]]:
        """BM25 keyword search using FTS5. Returns (chunk_id, rank)."""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                "SELECT rowid, rank FROM fts_search WHERE fts_search MATCH ? "
                "ORDER BY rank LIMIT ?",
                (query, limit),
            ).fetchall()
            return [(r[0], r[1]) for r in rows]
        except sqlite3.OperationalError:
            return []  # FTS5 not available

    def delete_by_file(self, file_path: str) -> int:
        """Delete all chunks for a file. Returns count deleted."""
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM chunks WHERE file_path=?", (file_path,))
        conn.commit()
        return cursor.rowcount

    def get_cached_embedding(self, text_hash: str) -> np.ndarray | None:
        """Get a cached embedding by text hash."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT embedding FROM embedding_cache WHERE text_hash=?",
            (text_hash,),
        ).fetchone()
        if row and row[0]:
            return np.frombuffer(row[0], dtype=np.float32)
        return None

    def cache_embedding(self, text_hash: str, embedding: np.ndarray) -> None:
        """Cache an embedding."""
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO embedding_cache (text_hash, embedding, created_at) "
            "VALUES (?, ?, ?)",
            (text_hash, embedding.tobytes(), time.time()),
        )
        conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
