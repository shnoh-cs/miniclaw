"""Vector memory store using SQLite + numpy."""

from __future__ import annotations

import hashlib
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
    source_type: str = "file"  # "file" | "session"


class MemoryStore:
    """SQLite-backed vector store with embedding cache."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        # Cached embedding matrix for fast batch cosine similarity
        self._emb_cache_ids: list[int] | None = None
        self._emb_cache_matrix: np.ndarray | None = None
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

            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
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

        # Migration: add source_type column if not present
        try:
            conn.execute("SELECT source_type FROM chunks LIMIT 1")
        except sqlite3.OperationalError:
            conn.execute("ALTER TABLE chunks ADD COLUMN source_type TEXT NOT NULL DEFAULT 'file'")

        conn.commit()

    def _invalidate_embedding_cache(self) -> None:
        """Invalidate the in-memory embedding matrix cache."""
        self._emb_cache_ids = None
        self._emb_cache_matrix = None

    def upsert_chunk(self, chunk: MemoryChunk) -> int:
        """Insert or update a chunk. Returns the chunk ID."""
        self._invalidate_embedding_cache()
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

        source_type = getattr(chunk, "source_type", "file")
        cursor = conn.execute(
            "INSERT INTO chunks (file_path, line_start, line_end, text, embedding, created_at, updated_at, source_type) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (chunk.file_path, chunk.line_start, chunk.line_end, chunk.text,
             embedding_blob, chunk.created_at, chunk.updated_at, source_type),
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
            f"SELECT id, file_path, line_start, line_end, text, created_at, updated_at, source_type "
            f"FROM chunks WHERE id IN ({placeholders})",
            ids,
        ).fetchall()

        return [
            MemoryChunk(
                id=r[0], file_path=r[1], line_start=r[2], line_end=r[3],
                text=r[4], created_at=r[5], updated_at=r[6],
                source_type=r[7] if len(r) > 7 else "file",
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

    def get_all_embeddings_cached(self) -> tuple[list[int], np.ndarray] | None:
        """Return a pre-built (ids, matrix) pair for fast batch cosine similarity.

        The result is cached in memory and invalidated on insert/update/delete.
        Returns None if no embeddings exist.
        """
        if self._emb_cache_ids is not None and self._emb_cache_matrix is not None:
            return (self._emb_cache_ids, self._emb_cache_matrix)

        all_embs = self.get_all_embeddings()
        if not all_embs:
            return None

        ids = [cid for cid, _ in all_embs]
        matrix = np.vstack([emb for _, emb in all_embs])
        # Pre-compute norms for the matrix (stored alongside for caller use)
        self._emb_cache_ids = ids
        self._emb_cache_matrix = matrix
        return (ids, matrix)

    def delete_by_file(self, file_path: str) -> int:
        """Delete all chunks for a file. Returns count deleted."""
        self._invalidate_embedding_cache()
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

    def get_metadata(self, key: str) -> str | None:
        """Get a metadata value by key."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT value FROM metadata WHERE key=?", (key,)
        ).fetchone()
        return row[0] if row else None

    def set_metadata(self, key: str, value: str) -> None:
        """Set a metadata value."""
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
            (key, value),
        )
        conn.commit()

    @staticmethod
    def compute_fingerprint(
        embedding_model: str,
        base_url: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> str:
        """Compute a fingerprint from embedding config for change detection.

        If any of these parameters change, existing embeddings are incompatible
        and the entire index must be rebuilt.
        """
        raw = f"{embedding_model}|{base_url}|{chunk_size}|{chunk_overlap}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def check_fingerprint(self, fingerprint: str) -> bool:
        """Check if stored fingerprint matches. Returns True if match or first run."""
        stored = self.get_metadata("embedding_fingerprint")
        if stored is None:
            # First run — store and continue
            self.set_metadata("embedding_fingerprint", fingerprint)
            return True
        return stored == fingerprint

    def reset_index(self, new_fingerprint: str) -> None:
        """Drop all chunks and embedding cache, store new fingerprint.

        Called when embedding model/endpoint/chunking config changes.
        """
        self._invalidate_embedding_cache()
        conn = self._get_conn()
        conn.executescript("""
            DELETE FROM chunks;
            DELETE FROM embedding_cache;
            DELETE FROM fts_index;
        """)
        try:
            conn.execute("DELETE FROM fts_search")
        except sqlite3.OperationalError:
            pass
        conn.commit()
        self.set_metadata("embedding_fingerprint", new_fingerprint)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None
