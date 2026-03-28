"""
Multi-backend embedding system with SHA256 cache.

Supports two providers:
  - Ollama: external service, large models (e.g. qwen3-embedding, 4096 dims)
  - FastEmbed: in-process ONNX inference, no external service needed

Provider and model are configured in processing.toml — this is a one-time
choice because switching invalidates the entire vector index.

Design informed by:
  - OpenClaw: SHA256(text + model) cache key, LRU eviction at 50K entries
  - Lesson #12: embedding cache critical since large models are compute-heavy
"""

from __future__ import annotations

import hashlib
import logging
import shutil
import struct
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astro_agent.config import Settings

logger = logging.getLogger(__name__)

# Max cached embeddings before LRU eviction
_MAX_CACHE_SIZE = 50_000


# ── Exception ────────────────────────────────────────────────────────────────


class EmbeddingInitError(RuntimeError):
    """Raised when the embedding system cannot initialize.

    Messages are human-readable and include actionable fix instructions.
    """


# ── Ollama auto-start ────────────────────────────────────────────────────────


def _ensure_ollama_running(timeout: float = 15.0) -> None:
    """Start Ollama if it isn't already serving. Raises EmbeddingInitError on failure."""
    # Quick check: is it already running?
    try:
        urllib.request.urlopen("http://localhost:11434", timeout=2)
        return
    except Exception:
        pass

    # Not running — try to start it
    ollama_bin = shutil.which("ollama")
    if not ollama_bin:
        raise EmbeddingInitError(
            "Ollama binary not found on PATH.\n\n"
            "Either install Ollama (https://ollama.com) or switch to FastEmbed:\n"
            "  In processing.toml [memory]: embedding_provider = \"fastembed\""
        )

    logger.info("Ollama not running — starting 'ollama serve' in background...")
    subprocess.Popen(
        [ollama_bin, "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for it to come up
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen("http://localhost:11434", timeout=2)
            logger.info("Ollama is now running")
            return
        except Exception:
            time.sleep(0.5)

    raise EmbeddingInitError(
        f"Ollama did not start within {timeout}s.\n\n"
        "Try starting it manually: ollama serve\n"
        "Or switch to FastEmbed in processing.toml [memory]: "
        "embedding_provider = \"fastembed\""
    )


# ── Backend: Ollama ──────────────────────────────────────��───────────────────


class OllamaEmbedder:
    """Generate embeddings via Ollama's local API.

    Raises EmbeddingInitError if Ollama is unavailable or the model can't be loaded.
    """

    def __init__(self, model: str = "qwen3-embedding"):
        self._model = model
        self._client = None
        self._dimension: int | None = None
        self._init_client()

    def _init_client(self):
        """Connect to Ollama and verify the model works."""
        _ensure_ollama_running()
        try:
            import ollama
        except ImportError:
            raise EmbeddingInitError(
                "ollama Python package not installed.\n\n"
                "Install with: uv add ollama"
            )
        self._client = ollama.Client()
        try:
            test = self._client.embed(model=self._model, input="test")
        except Exception as e:
            raise EmbeddingInitError(
                f"Ollama model '{self._model}' is not available: {e}\n\n"
                f"Pull it with: ollama pull {self._model}\n"
                f"Or choose a different model in processing.toml [memory] embedding_model"
            )
        if test and test.get("embeddings"):
            self._dimension = len(test["embeddings"][0])
            logger.info(
                f"Ollama embedder ready: model={self._model}, "
                f"dimension={self._dimension}"
            )
        else:
            raise EmbeddingInitError(
                f"Ollama returned empty embeddings for model '{self._model}'."
            )

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def embed(self, text: str) -> list[float] | None:
        """Generate an embedding. Returns None on transient failure."""
        try:
            result = self._client.embed(model=self._model, input=text)
            if result and result.get("embeddings"):
                return result["embeddings"][0]
            return None
        except Exception as e:
            logger.warning(f"Ollama embedding failed: {e}")
            return None

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        return [self.embed(text) for text in texts]


# ── Backend: FastEmbed ───────────────────────────────────────────────────────


class FastEmbedEmbedder:
    """In-process embeddings via fastembed (ONNX). No external service needed.

    Raises EmbeddingInitError if fastembed is not installed or the model fails to load.
    """

    def __init__(self, model: str = "snowflake/snowflake-arctic-embed-l"):
        self._model = model
        self._fe_model = None
        self._dimension: int | None = None
        self._init_model()

    def _init_model(self):
        """Load the ONNX model (downloads on first use)."""
        try:
            from fastembed import TextEmbedding
        except ImportError:
            raise EmbeddingInitError(
                "fastembed package not installed.\n\n"
                "Install with: uv add fastembed"
            )
        try:
            self._fe_model = TextEmbedding(model_name=self._model)
        except Exception as e:
            raise EmbeddingInitError(
                f"FastEmbed model '{self._model}' failed to load: {e}\n\n"
                f"Available models can be listed with:\n"
                f"  python -c \"from fastembed import TextEmbedding; "
                f"print([m['model'] for m in TextEmbedding.list_supported_models()])\"\n\n"
                f"Or choose a different model in processing.toml [memory] embedding_model"
            )
        # Discover dimension with a test embedding
        try:
            test_results = list(self._fe_model.embed(["test"]))
            if test_results:
                self._dimension = len(test_results[0])
                logger.info(
                    f"FastEmbed embedder ready: model={self._model}, "
                    f"dimension={self._dimension}"
                )
            else:
                raise EmbeddingInitError(
                    f"FastEmbed returned empty embeddings for model '{self._model}'."
                )
        except EmbeddingInitError:
            raise
        except Exception as e:
            raise EmbeddingInitError(
                f"FastEmbed test embedding failed for '{self._model}': {e}"
            )

    @property
    def model_id(self) -> str:
        return self._model

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def embed(self, text: str) -> list[float] | None:
        """Generate an embedding. Returns None on transient failure."""
        try:
            results = list(self._fe_model.embed([text]))
            if results:
                return results[0].tolist()
            return None
        except Exception as e:
            logger.warning(f"FastEmbed embedding failed: {e}")
            return None

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        try:
            results = list(self._fe_model.embed(texts))
            return [r.tolist() for r in results]
        except Exception as e:
            logger.warning(f"FastEmbed batch embedding failed: {e}")
            return [self.embed(text) for text in texts]


# ── Cache wrapper ────────────────────────────────────────────────────────────


class CachedEmbedder:
    """Decorator that adds SHA256 SQLite caching to any embedder.

    Cache key: SHA256(text + "|" + model_id). Switching models naturally
    invalidates stale entries. LRU eviction at _MAX_CACHE_SIZE entries.
    """

    def __init__(self, inner: OllamaEmbedder | FastEmbedEmbedder, db_conn=None):
        self._inner = inner
        self._db_conn = db_conn

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    @property
    def dimension(self) -> int | None:
        return self._inner.dimension

    def embed(self, text: str) -> list[float] | None:
        """Generate an embedding, checking cache first."""
        cache_key = self._cache_key(text)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        embedding = self._inner.embed(text)
        if embedding is not None:
            self._cache_put(cache_key, embedding)
        return embedding

    def embed_batch(self, texts: list[str]) -> list[list[float] | None]:
        return [self.embed(text) for text in texts]

    # ── Cache operations ───────────────────────────���─────────────────────

    def _cache_key(self, text: str) -> str:
        """SHA256(text + model) as cache key."""
        raw = f"{text}|{self._inner.model_id}"
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


# ── Factory ──────────────────────────────────────────────────────────────────


def create_embedder(provider: str, model: str) -> OllamaEmbedder | FastEmbedEmbedder:
    """Create the appropriate embedder backend.

    Raises EmbeddingInitError if the provider is unknown or initialization fails.
    """
    if provider == "ollama":
        return OllamaEmbedder(model=model)
    elif provider == "fastembed":
        return FastEmbedEmbedder(model=model)
    else:
        raise EmbeddingInitError(
            f"Unknown embedding provider: '{provider}'.\n\n"
            f"Valid providers: 'ollama', 'fastembed'\n"
            f"Set in processing.toml [memory] embedding_provider"
        )


# ── Centralized init ────────────────────────────────────────────────────────


def init_memory_system(settings: Settings, rebuild_embeddings: bool = False) -> None:
    """Initialize the complete memory system: embedder + store + tool registration.

    This is the single entry point for both CLI and Gradio. It replaces the
    duplicated init blocks that previously existed in both.

    Raises EmbeddingInitError if:
      - The embedding backend cannot start
      - The configured model/provider changed and rebuild_embeddings is False
    """
    from astro_agent.graph.hitl import set_memory_enabled
    from astro_agent.graph.registry import register_memory_tool
    from astro_agent.memory.store import MemoryStore
    from astro_agent.tools.utility.t33_memory_search import set_memory_store

    # Create the raw embedder (validates provider + model, connects to service)
    inner = create_embedder(settings.memory_embedding_provider, settings.memory_embedding_model)

    # Create the store (sets up DB, detects extensions)
    store = MemoryStore(db_path=settings.memory_db_path, embedder=inner)

    # Check for model mismatch before wiring everything up
    store.check_embedding_model(
        current_model=settings.memory_embedding_model,
        current_dim=inner.dimension,
        rebuild=rebuild_embeddings,
    )

    # Wrap with cache using the store's DB connection
    cached = CachedEmbedder(inner, db_conn=store._get_conn())
    store._embedder = cached

    # Wire up the memory tool
    set_memory_store(store)
    set_memory_enabled(True)
    register_memory_tool()
    logger.info(
        f"Long-term memory initialized: provider={settings.memory_embedding_provider}, "
        f"model={settings.memory_embedding_model}, dimension={inner.dimension}"
    )


# ── Serialization helpers ────────────────────────────────────────────────────


def _serialize_embedding(embedding: list[float]) -> bytes:
    """Pack float list into bytes."""
    return struct.pack(f"{len(embedding)}f", *embedding)


def _deserialize_embedding(blob: bytes) -> list[float]:
    """Unpack bytes into float list."""
    n = len(blob) // 4  # 4 bytes per float32
    return list(struct.unpack(f"{n}f", blob))
