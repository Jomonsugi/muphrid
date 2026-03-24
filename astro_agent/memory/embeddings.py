"""
Embedding generation via Ollama with SHA256 cache.

Design informed by:
  - OpenClaw: SHA256(text + model) cache key, LRU eviction at 50K entries
  - Lesson #12: embedding cache critical since Qwen3-Embedding-8B is compute-heavy

Lazy initialization — no Ollama connection until the first embed() call.
Graceful degradation: if Ollama is unavailable, embed() returns None and
the memory store falls back to FTS5-only search.
"""

from __future__ import annotations

import hashlib
import logging
import struct
from typing import Any

logger = logging.getLogger(__name__)

# Max cached embeddings before LRU eviction
_MAX_CACHE_SIZE = 50_000


class OllamaEmbedder:
    """
    Generate embeddings via Ollama's local API with persistent SHA256 cache.

    The cache lives in the same memory.db (embedding_cache table) so it
    persists across sessions without a separate file.
    """

    def __init__(self, model: str = "qwen3-embedding", db_conn=None):
        self._model = model
        self._db_conn = db_conn
        self._client = None  # lazy init
        self._initialized = False
        self._available = True
        self._dimension: int | None = None

    def _init_client(self):
        """Lazy-initialize the Ollama client on first use."""
        if self._initialized:
            return
        self._initialized = True
        try:
            import ollama
            self._client = ollama.Client()
            # Verify the model is available with a test embedding
            test = self._client.embed(model=self._model, input="test")
            if test and test.get("embeddings"):
                self._dimension = len(test["embeddings"][0])
                logger.info(
                    f"Ollama embedder ready: model={self._model}, "
                    f"dimension={self._dimension}"
                )
            else:
                raise RuntimeError("Empty embedding response from Ollama")
        except ImportError:
            logger.warning(
                "ollama package not installed. Memory vector search disabled. "
                "Install with: uv add ollama"
            )
            self._available = False
        except Exception as e:
            logger.warning(
                f"Ollama not available for embeddings: {e}. "
                f"Ensure Ollama is running (ollama serve) and the model is pulled "
                f"(ollama pull {self._model}). Memory will use FTS5-only search."
            )
            self._available = False

    @property
    def dimension(self) -> int | None:
        """Return the embedding dimension, or None if not yet initialized."""
        if not self._initialized:
            self._init_client()
        return self._dimension

    def embed(self, text: str) -> list[float] | None:
        """
        Generate an embedding for text, using cache when available.

        Returns None if Ollama is unavailable — the memory store gracefully
        degrades to FTS5-only search.
        """
        if not self._initialized:
            self._init_client()
        if not self._available:
            return None

        # Check cache first
        cache_key = self._cache_key(text)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        # Generate embedding
        try:
            result = self._client.embed(model=self._model, input=text)
            if result and result.get("embeddings"):
                embedding = result["embeddings"][0]
                self._cache_put(cache_key, embedding)
                return embedding
            return None
        except Exception as e:
            logger.warning(f"Embedding generation failed: {e}")
            return None

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        """Generate embeddings for multiple texts."""
        return [self.embed(text) for text in texts]

    # ── Cache operations ─────────────────────────────────────────────────

    def _cache_key(self, text: str) -> str:
        """SHA256(text + model) as cache key."""
        raw = f"{text}|{self._model}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _cache_get(self, key: str) -> list[float] | None:
        """Look up a cached embedding."""
        if not self._db_conn:
            return None
        try:
            row = self._db_conn.execute(
                "SELECT embedding FROM embedding_cache WHERE hash = ?",
                (key,),
            ).fetchone()
            if row and row[0]:
                return _deserialize_embedding(row[0])
        except Exception:
            pass
        return None

    def _cache_put(self, key: str, embedding: list[float]):
        """Store an embedding in the cache."""
        if not self._db_conn:
            return
        try:
            from datetime import datetime, timezone
            now = datetime.now(timezone.utc).isoformat()
            blob = _serialize_embedding(embedding)
            self._db_conn.execute(
                "INSERT OR REPLACE INTO embedding_cache (hash, embedding, created_at) VALUES (?, ?, ?)",
                (key, blob, now),
            )
            self._db_conn.commit()

            # LRU eviction if cache is too large
            count = self._db_conn.execute(
                "SELECT COUNT(*) FROM embedding_cache"
            ).fetchone()[0]
            if count > _MAX_CACHE_SIZE:
                self._db_conn.execute(
                    f"""DELETE FROM embedding_cache WHERE hash IN (
                        SELECT hash FROM embedding_cache
                        ORDER BY created_at ASC
                        LIMIT {count - _MAX_CACHE_SIZE}
                    )"""
                )
                self._db_conn.commit()
                logger.info(f"Embedding cache evicted {count - _MAX_CACHE_SIZE} old entries")
        except Exception as e:
            logger.warning(f"Embedding cache write failed: {e}")


# ── Serialization helpers ────────────────────────────────────────────────────


def _serialize_embedding(embedding: list[float]) -> bytes:
    """Pack float list into bytes."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def _deserialize_embedding(blob: bytes) -> list[float]:
    """Unpack bytes into float list."""
    n = len(blob) // 4  # 4 bytes per float32
    return list(struct.unpack(f"{n}f", blob))
