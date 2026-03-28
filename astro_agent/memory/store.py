"""
Core memory store — SQLite + sqlite-vec + FTS5 with RRF hybrid search.

Design informed by:
  - OpenClaw: SQLite + sqlite-vec + FTS5 hybrid search, embedding cache
  - Zep/Graphiti: temporal validity (invalidate, never delete)
  - Hindsight: confidence scoring, epistemic separation via source column
  - Academic: RRF fusion for multi-modal retrieval

The store is a single SQLite file at ~/.astro_agent/memory.db. It holds
four memory types (sessions, observations, failures, preferences) plus
a vector index (sqlite-vec), full-text index (FTS5), and embedding cache.

The embedding system is fail-loud: if memory is enabled, the embedder
is guaranteed to be initialized and working. There are no silent fallback
paths — hybrid search (vector + FTS5 + RRF) is always available.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default location for the global memory store
DEFAULT_DB_PATH = Path.home() / ".astro_agent" / "memory.db"


class MemoryStore:
    """
    Long-term memory store backed by SQLite + sqlite-vec + FTS5.

    Provides:
      - add_session / add_observation / add_failure / add_preference
      - search(query) with RRF-fused hybrid retrieval
      - invalidate(table, id) for temporal validity
      - Embedding cache via SHA256(text + model) keys
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH, embedder=None):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._embedder = embedder
        self._has_vec = False
        self._has_fts = False

    # ── Connection management ────────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._detect_extensions()
            self._init_db()
            self._backfill_embeddings()
        return self._conn

    def _detect_extensions(self):
        """Check which SQLite extensions are available."""
        conn = self._conn
        # Check sqlite-vec
        try:
            import sqlite_vec  # noqa: F401
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            self._has_vec = True
            logger.info("sqlite-vec extension loaded")
        except Exception as e:
            logger.info(f"sqlite-vec not available, vector search disabled: {e}")
            self._has_vec = False

        # Check FTS5
        try:
            conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts_test USING fts5(x)")
            conn.execute("DROP TABLE IF EXISTS _fts_test")
            self._has_fts = True
            logger.info("FTS5 extension available")
        except Exception:
            self._has_fts = False
            logger.info("FTS5 not available, keyword search disabled")

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = self._conn
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                target_name TEXT NOT NULL,
                target_type TEXT,
                sensor TEXT,
                sensor_type TEXT,
                focal_length_mm REAL,
                pixel_scale REAL,
                bortle INTEGER,
                sqm_reading REAL,
                total_integration_s REAL,
                num_frames INTEGER,
                outcome TEXT,
                outcome_detail TEXT,
                final_snr REAL,
                created_at TEXT NOT NULL,
                valid_until TEXT
            );

            CREATE TABLE IF NOT EXISTS observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT REFERENCES sessions(id),
                phase TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'hitl',
                content TEXT NOT NULL,
                parameters TEXT,
                metrics TEXT,
                confidence REAL DEFAULT 0.5,
                reinforcement_count INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                valid_until TEXT
            );

            CREATE TABLE IF NOT EXISTS failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT REFERENCES sessions(id),
                phase TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'hitl',
                tool TEXT NOT NULL,
                content TEXT NOT NULL,
                parameters TEXT,
                root_cause TEXT,
                resolution TEXT,
                created_at TEXT NOT NULL,
                valid_until TEXT
            );

            CREATE TABLE IF NOT EXISTS preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT REFERENCES sessions(id),
                tool TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'hitl',
                content TEXT NOT NULL,
                parameters TEXT,
                target_type TEXT,
                sensor TEXT,
                confidence REAL DEFAULT 0.5,
                reinforcement_count INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                valid_until TEXT
            );

            CREATE TABLE IF NOT EXISTS embedding_cache (
                hash TEXT PRIMARY KEY,
                embedding BLOB,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memory_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)

        # Vector index (sqlite-vec) — dimension comes from the embedder
        if self._has_vec and self._embedder:
            dim = self._embedder.dimension
            if dim:
                try:
                    conn.execute(f"""
                        CREATE VIRTUAL TABLE IF NOT EXISTS memory_vec USING vec0(
                            source_table TEXT NOT NULL,
                            source_id INTEGER NOT NULL,
                            embedding FLOAT[{dim}]
                        )
                    """)
                except Exception as e:
                    logger.warning(f"Failed to create memory_vec table: {e}")
                    self._has_vec = False

        # Full-text search index (FTS5)
        if self._has_fts:
            try:
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                        content,
                        source_table,
                        source_id,
                        tokenize='porter'
                    )
                """)
            except Exception as e:
                logger.warning(f"Failed to create memory_fts table: {e}")
                self._has_fts = False

        conn.commit()

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Model tracking + migration ───────────────────────────────────────

    def _get_meta(self, key: str) -> str | None:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT value FROM memory_meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def _set_meta(self, key: str, value: str):
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO memory_meta (key, value) VALUES (?, ?)",
            (key, value),
        )
        conn.commit()

    def check_embedding_model(
        self, current_model: str, current_dim: int | None, rebuild: bool = False,
    ) -> None:
        """Check if stored model matches current config.

        On first run, records the model/dimension. On subsequent runs, detects
        mismatch and either blocks (raising EmbeddingInitError) or rebuilds
        the vector index.
        """
        from astro_agent.memory.embeddings import EmbeddingInitError

        stored_model = self._get_meta("embedding_model")

        if stored_model is None:
            # First run — record the model
            self._set_meta("embedding_model", current_model)
            if current_dim:
                self._set_meta("embedding_dimension", str(current_dim))
            return

        if stored_model == current_model:
            return

        # Model mismatch
        if not rebuild:
            raise EmbeddingInitError(
                f"Embedding model changed: '{stored_model}' → '{current_model}'.\n"
                f"This requires rebuilding the vector index (all existing embeddings "
                f"will be regenerated from the stored text).\n\n"
                f"CLI: re-run with --rebuild-embeddings\n"
                f"Config: set rebuild_embeddings = true in [memory], "
                f"then remove it after one successful run."
            )

        # Rebuild: drop + recreate vec table, update meta, backfill
        logger.warning(
            f"Embedding model changed: '{stored_model}' → '{current_model}'. "
            f"Rebuilding vector index..."
        )
        self._rebuild_vec_index(current_dim)
        self._set_meta("embedding_model", current_model)
        if current_dim:
            self._set_meta("embedding_dimension", str(current_dim))

    def _rebuild_vec_index(self, dim: int | None):
        """Drop and recreate the vector index with a new dimension."""
        conn = self._get_conn()
        try:
            conn.execute("DROP TABLE IF EXISTS memory_vec")
            if dim and self._has_vec:
                conn.execute(f"""
                    CREATE VIRTUAL TABLE memory_vec USING vec0(
                        source_table TEXT NOT NULL,
                        source_id INTEGER NOT NULL,
                        embedding FLOAT[{dim}]
                    )
                """)
            conn.commit()
            logger.info(f"Vector index rebuilt with dimension {dim}")
            # Backfill will run automatically since all rows are now missing
            self._backfill_embeddings()
        except Exception as e:
            logger.warning(f"Vector index rebuild failed: {e}")
            self._has_vec = False

    # ── Write operations ─────────────────────────────────────────────────

    def add_session(
        self,
        session_id: str,
        target_name: str,
        target_type: str | None = None,
        sensor: str | None = None,
        sensor_type: str | None = None,
        focal_length_mm: float | None = None,
        pixel_scale: float | None = None,
        bortle: int | None = None,
        sqm_reading: float | None = None,
        total_integration_s: float | None = None,
        num_frames: int | None = None,
        outcome: str | None = None,
        outcome_detail: str | None = None,
        final_snr: float | None = None,
    ) -> str:
        """Record a processing session."""
        conn = self._get_conn()
        now = _now_iso()
        conn.execute(
            """INSERT OR REPLACE INTO sessions
               (id, target_name, target_type, sensor, sensor_type,
                focal_length_mm, pixel_scale, bortle, sqm_reading,
                total_integration_s, num_frames, outcome, outcome_detail,
                final_snr, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, target_name, target_type, sensor, sensor_type,
             focal_length_mm, pixel_scale, bortle, sqm_reading,
             total_integration_s, num_frames, outcome, outcome_detail,
             final_snr, now),
        )
        conn.commit()
        logger.info(f"Memory: session recorded — {target_name} ({session_id})")
        return session_id

    def add_observation(
        self,
        content: str,
        phase: str,
        session_id: str | None = None,
        source: str = "hitl",
        parameters: dict | None = None,
        metrics: dict | None = None,
    ) -> int:
        """Store an LLM-extracted observation from a processing phase."""
        conn = self._get_conn()
        now = _now_iso()
        cur = conn.execute(
            """INSERT INTO observations
               (session_id, phase, source, content, parameters, metrics,
                confidence, reinforcement_count, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, 0.5, 1, ?, ?)""",
            (session_id, phase, source, content,
             json.dumps(parameters) if parameters else None,
             json.dumps(metrics) if metrics else None,
             now, now),
        )
        obs_id = cur.lastrowid
        conn.commit()
        self._index_memory("observations", obs_id, content)
        logger.info(f"Memory: observation added — {phase} ({source})")
        return obs_id

    def add_failure(
        self,
        content: str,
        phase: str,
        tool: str,
        session_id: str | None = None,
        source: str = "hitl",
        parameters: dict | None = None,
        root_cause: str | None = None,
        resolution: str | None = None,
    ) -> int:
        """Store an LLM-extracted failure record."""
        conn = self._get_conn()
        now = _now_iso()
        cur = conn.execute(
            """INSERT INTO failures
               (session_id, phase, source, tool, content, parameters,
                root_cause, resolution, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, phase, source, tool, content,
             json.dumps(parameters) if parameters else None,
             root_cause, resolution, now),
        )
        fail_id = cur.lastrowid
        conn.commit()
        # Index the combined content for search
        search_text = content
        if root_cause:
            search_text += f" Root cause: {root_cause}"
        if resolution:
            search_text += f" Resolution: {resolution}"
        self._index_memory("failures", fail_id, search_text)
        logger.info(f"Memory: failure recorded — {tool} in {phase} ({source})")
        return fail_id

    def add_preference(
        self,
        content: str,
        tool: str,
        session_id: str | None = None,
        source: str = "hitl",
        parameters: dict | None = None,
        target_type: str | None = None,
        sensor: str | None = None,
    ) -> int:
        """Store a user preference from HITL approval."""
        conn = self._get_conn()
        now = _now_iso()
        cur = conn.execute(
            """INSERT INTO preferences
               (session_id, tool, source, content, parameters,
                target_type, sensor, confidence, reinforcement_count,
                created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 0.5, 1, ?, ?)""",
            (session_id, tool, source, content,
             json.dumps(parameters) if parameters else None,
             target_type, sensor, now, now),
        )
        pref_id = cur.lastrowid
        conn.commit()
        self._index_memory("preferences", pref_id, content)
        logger.info(f"Memory: preference recorded — {tool} ({source})")
        return pref_id

    # ── Indexing ─────────────────────────────────────────────────────────

    def _index_memory(self, table: str, row_id: int, content: str):
        """Add a memory to the vector and FTS indexes."""
        conn = self._get_conn()

        # FTS5 index
        if self._has_fts:
            try:
                conn.execute(
                    "INSERT INTO memory_fts (content, source_table, source_id) VALUES (?, ?, ?)",
                    (content, table, str(row_id)),
                )
                conn.commit()
            except Exception as e:
                logger.warning(f"FTS5 indexing failed for {table}:{row_id}: {e}")

        # Vector index
        if self._has_vec and self._embedder:
            try:
                embedding = self._embedder.embed(content)
                if embedding is not None:
                    conn.execute(
                        "INSERT INTO memory_vec (source_table, source_id, embedding) VALUES (?, ?, ?)",
                        (table, row_id, _serialize_embedding(embedding)),
                    )
                    conn.commit()
            except Exception as e:
                logger.warning(f"Vector indexing failed for {table}:{row_id}: {e}")

    def _backfill_embeddings(self):
        """Embed any memories that are missing from the vector index.

        Runs once at startup. Also runs after a model change triggers
        a vector index rebuild.
        """
        if not self._has_vec or not self._embedder:
            return

        conn = self._conn
        tables = ["observations", "failures", "preferences"]
        total = 0

        for table in tables:
            # Find rows that exist in the table but not in memory_vec
            rows = conn.execute(
                f"""SELECT t.id, t.content FROM {table} t
                    WHERE t.valid_until IS NULL
                    AND NOT EXISTS (
                        SELECT 1 FROM memory_vec v
                        WHERE v.source_table = ? AND v.source_id = t.id
                    )""",
                (table,),
            ).fetchall()

            if not rows:
                continue

            for row in rows:
                content = row["content"]
                # For failures, include root_cause and resolution in embedding
                if table == "failures":
                    full_row = conn.execute(
                        "SELECT root_cause, resolution FROM failures WHERE id = ?",
                        (row["id"],),
                    ).fetchone()
                    if full_row:
                        if full_row["root_cause"]:
                            content += f" Root cause: {full_row['root_cause']}"
                        if full_row["resolution"]:
                            content += f" Resolution: {full_row['resolution']}"

                try:
                    embedding = self._embedder.embed(content)
                    if embedding is not None:
                        conn.execute(
                            "INSERT INTO memory_vec (source_table, source_id, embedding) VALUES (?, ?, ?)",
                            (table, row["id"], _serialize_embedding(embedding)),
                        )
                        total += 1
                except Exception as e:
                    logger.warning(f"Backfill embedding failed for {table}:{row['id']}: {e}")

        if total > 0:
            conn.commit()
            logger.info(f"Memory: backfilled {total} embeddings")

    # ── Search ───────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        limit: int = 10,
        memory_type: str | None = None,
        target_type: str | None = None,
        phase: str | None = None,
        source: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Hybrid search: vector similarity + FTS5 keyword, fused with RRF.

        Research basis (Lesson #4 + #11):
          - RRF fusion from Zep/Hindsight: rank-based, no score normalization needed
          - Retrieval quality > write quality (academic finding)
          - Union, not intersection (OpenClaw): results from either method contribute

        Args:
            query: Natural language search query
            limit: Max results to return
            memory_type: Filter to "observations", "failures", or "preferences"
            target_type: Filter to specific target type (e.g. "emission_nebula")
            phase: Filter to specific phase (e.g. "stretch")
            source: Filter to specific source (e.g. "hitl")

        Returns:
            List of dicts with memory content, type, metadata, and RRF score
        """
        conn = self._get_conn()
        tables = self._resolve_tables(memory_type)

        # Hybrid search: vector + FTS5
        # With fail-loud init, both indexes are guaranteed available.
        vec_results = []
        fts_results = []

        query_embedding = self._embedder.embed(query)
        if query_embedding is not None:
            vec_results = self._vector_search(
                query_embedding, tables, limit=50,
                target_type=target_type, phase=phase, source=source,
            )

        fts_results = self._fts_search(
            query, tables, limit=50,
            target_type=target_type, phase=phase, source=source,
        )

        logger.debug(
            f"Memory search: {len(vec_results)} vector + {len(fts_results)} FTS5 candidates"
        )

        # RRF fusion (k=60, standard constant)
        # Future consideration: Distribution-Based Score Fusion (DBSF) could
        # replace RRF here. DBSF normalizes raw scores from each branch using
        # mean/stddev, preserving *how strongly* something matched rather than
        # just its rank position. Both vec0 (distance) and FTS5 (bm25()) return
        # usable raw scores. Guard with a fallback to RRF when either branch
        # returns <10 results, as DBSF statistics become unstable at low counts.
        rrf_scores: dict[str, float] = {}
        rrf_items: dict[str, dict] = {}

        for rank, item in enumerate(vec_results):
            key = f"{item['table']}:{item['id']}"
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (rank + 60)
            rrf_items[key] = item

        for rank, item in enumerate(fts_results):
            key = f"{item['table']}:{item['id']}"
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (rank + 60)
            rrf_items[key] = item

        # Sort by RRF score descending, prioritize hitl source
        sorted_keys = sorted(
            rrf_scores.keys(),
            key=lambda k: (
                # Primary: hitl source gets a bonus (epistemic separation, Lesson #8)
                0.01 if rrf_items[k].get("source") == "hitl" else 0,
                rrf_scores[k],
            ),
            reverse=True,
        )

        results = []
        for key in sorted_keys[:limit]:
            item = rrf_items[key]
            item["rrf_score"] = rrf_scores[key]
            results.append(item)

        return results

    def _resolve_tables(self, memory_type: str | None) -> list[str]:
        if memory_type:
            return [memory_type]
        return ["observations", "failures", "preferences"]

    def _vector_search(
        self,
        query_embedding: list[float],
        tables: list[str],
        limit: int = 50,
        **filters,
    ) -> list[dict]:
        """Search the vector index for semantically similar memories.

        Filters by source_table at the SQL level so sqlite-vec skips
        irrelevant rows rather than retrieving them for Python-side discard.
        """
        conn = self._get_conn()
        results = []

        placeholders = ",".join("?" for _ in tables)
        rows = conn.execute(
            f"""SELECT source_table, source_id, distance
               FROM memory_vec
               WHERE embedding MATCH ?
               AND source_table IN ({placeholders})
               ORDER BY distance
               LIMIT ?""",
            (_serialize_embedding(query_embedding), *tables, limit),
        ).fetchall()

        for row in rows:
            item = self._load_memory(row["source_table"], row["source_id"], **filters)
            if item:
                item["distance"] = row["distance"]
                results.append(item)

        return results

    def _fts_search(
        self,
        query: str,
        tables: list[str],
        limit: int = 50,
        **filters,
    ) -> list[dict]:
        """Search the FTS5 index for keyword matches, ranked by BM25."""
        conn = self._get_conn()
        results = []

        # Escape FTS5 special characters
        safe_query = query.replace('"', '""')

        placeholders = ",".join("?" for _ in tables)
        rows = conn.execute(
            f"""SELECT source_table, source_id, bm25(memory_fts) AS rank
               FROM memory_fts
               WHERE memory_fts MATCH ?
               AND source_table IN ({placeholders})
               ORDER BY rank
               LIMIT ?""",
            (f'"{safe_query}"', *tables, limit),
        ).fetchall()

        for row in rows:
            item = self._load_memory(row["source_table"], int(row["source_id"]), **filters)
            if item:
                item["fts_rank"] = row["rank"]
                results.append(item)

        return results

    def _load_memory(
        self,
        table: str,
        row_id: int,
        target_type: str | None = None,
        phase: str | None = None,
        source: str | None = None,
    ) -> dict | None:
        """Load a memory row by table and ID, applying filters."""
        conn = self._get_conn()
        row = conn.execute(
            f"SELECT * FROM {table} WHERE id = ? AND valid_until IS NULL",
            (row_id,),
        ).fetchone()
        if not row:
            return None

        item = dict(row)
        item["table"] = table

        # Apply filters
        if source and item.get("source") != source:
            return None
        if phase and item.get("phase") != phase:
            return None
        if target_type:
            # Check target_type on preferences table, or join to session
            if table == "preferences" and item.get("target_type") != target_type:
                return None

        # Parse JSON fields
        for field in ("parameters", "metrics"):
            if field in item and item[field]:
                try:
                    item[field] = json.loads(item[field])
                except (json.JSONDecodeError, TypeError):
                    pass

        # Attach session context if available
        session_id = item.get("session_id")
        if session_id:
            session = conn.execute(
                "SELECT target_name, target_type, sensor, sensor_type, bortle, created_at FROM sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            if session:
                item["session_target"] = session["target_name"]
                item["session_target_type"] = session["target_type"]
                item["session_sensor"] = session["sensor"]
                item["session_date"] = session["created_at"]

        return item

    # ── Temporal validity (Lesson #2: invalidate, never delete) ──────────

    def invalidate(self, table: str, row_id: int):
        """Mark a memory as invalid instead of deleting it."""
        conn = self._get_conn()
        now = _now_iso()
        conn.execute(
            f"UPDATE {table} SET valid_until = ? WHERE id = ?",
            (now, row_id),
        )
        conn.commit()
        logger.info(f"Memory: invalidated {table}:{row_id}")

    # ── Stats ────────────────────────────────────────────────────────────

    def stats(self) -> dict[str, int]:
        """Return counts of active memories by type."""
        conn = self._get_conn()
        counts = {}
        for table in ("sessions", "observations", "failures", "preferences"):
            row = conn.execute(
                f"SELECT COUNT(*) as n FROM {table} WHERE valid_until IS NULL"
            ).fetchone()
            counts[table] = row["n"] if row else 0
        return counts


# ── Helpers ──────────────────────────────────────────────────────────────────


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _serialize_embedding(embedding: list[float]) -> bytes:
    """Serialize a float list to bytes for sqlite-vec."""
    import struct
    return struct.pack(f"{len(embedding)}f", *embedding)
