"""Environment-backed cache settings (Postgres, Redis, embedder selection)."""

from __future__ import annotations

import os
import secrets
import warnings
from typing import Any, Literal

from .types import EmbedderType

_DEFAULT_LOG_DIGEST_KEY = secrets.token_hex(32)

_VALID_EMBEDDER_TYPES: frozenset[str] = frozenset(
    ("huggingface", "openai", "cohere", "voyage", "ollama")
)
_VALID_RESPONSE_MODES: frozenset[str] = frozenset(("buffered", "tee"))
_VALID_HIT_RESPONSE_MODES: frozenset[str] = frozenset(("single", "stream"))

# Sentinel used to distinguish "not supplied" from "explicitly None" for optional
# fields that must fall back to an environment variable when omitted.
_UNSET: Any = object()


def _parse_optional_float(value: str | None, name: str) -> float | None:
    """Parse an optional float environment variable.

    Args:
        value: Raw string from os.getenv, or None.
        name: Variable name used in error messages.

    Returns:
        Parsed float, or None when value is None or empty string.

    Raises:
        ValueError: When value is present but cannot be converted to float.
    """
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        raise ValueError(
            f"{name} must be a float, got {value!r}"
        )


def _parse_optional_int(value: str | None, name: str) -> int | None:
    """Parse an optional integer environment variable.

    Args:
        value: Raw string from os.getenv, or None.
        name: Variable name used in error messages.

    Returns:
        Parsed int, or None when value is None or empty string.

    Raises:
        ValueError: When value is present but cannot be converted to int.
    """
    if value is None or value.strip() == "":
        return None
    try:
        return int(value)
    except ValueError:
        raise ValueError(
            f"{name} must be an integer, got {value!r}"
        )


def _parse_bool(value: str | None, default: bool) -> bool:
    """Parse a boolean environment variable from common truthy string values.

    Args:
        value: Raw string from os.getenv, or None.
        default: Fallback when value is None or empty.

    Returns:
        True when value is one of ``"1"``, ``"true"``, ``"yes"`` (case-insensitive),
        False when it is ``"0"``, ``"false"``, ``"no"``, or the default otherwise.
    """
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in ("1", "true", "yes")


class CacheSettings:
    """Cache configuration loaded from constructor arguments or environment variables.

    All fields accept explicit constructor arguments. When a field is omitted, the
    constructor reads the corresponding ``SEMANTIC_CACHE_*`` environment variable and
    applies the documented default.

    Call ``CacheSettings.from_env()`` (or the convenience alias ``get_cache_settings()``)
    to construct a fully environment-driven instance.
    """

    # ------------------------------------------------------------------ #
    # Constructor
    # ------------------------------------------------------------------ #

    def __init__(  # noqa: PLR0912, PLR0913, PLR0915
        self,
        *,
        disable_proxy_app_docs: bool | None = None,
        top_k_candidates: int | None = None,
        threshold: float | None = None,
        rejection_threshold: Any = _UNSET,
        pg_uri: str | None = None,
        pg_ensure_schema: bool | None = None,
        redis_uri: str | None = None,
        redis_ttl_seconds: int | None = None,
        pg_ttl_days: Any = _UNSET,
        pgvector_hnsw_m: int | None = None,
        pgvector_hnsw_ef_construction: int | None = None,
        pgvector_hnsw_ef_search: Any = _UNSET,
        pg_pool_size: int | None = None,
        pg_pool_max_overflow: int | None = None,
        embed_timeout_seconds: Any = _UNSET,
        store_timeout_seconds: Any = _UNSET,
        upstream_timeout_seconds: Any = _UNSET,
        embedder_type: EmbedderType | None = None,
        hugging_face_api_key: Any = _UNSET,
        openai_api_key: Any = _UNSET,
        cohere_api_key: Any = _UNSET,
        cohere_embedding_model: Any = _UNSET,
        cohere_embedding_dimensions: Any = _UNSET,
        cohere_input_type: Any = _UNSET,
        voyage_api_key: Any = _UNSET,
        voyage_embedding_model: Any = _UNSET,
        voyage_embedding_dimensions: Any = _UNSET,
        voyage_input_type: Any = _UNSET,
        ollama_api_key: Any = _UNSET,
        ollama_base_url: str | None = None,
        ollama_embedding_model: Any = _UNSET,
        ollama_embedding_dimensions: Any = _UNSET,
        circuit_breaker_429_enabled: bool | None = None,
        circuit_breaker_429_consecutive_limit: int | None = None,
        circuit_breaker_429_open_seconds: float | None = None,
        middleware_flight_lock_max_entries: int | None = None,
        middleware_flight_lock_acquire_timeout_seconds: Any = _UNSET,
        require_cache_scope: bool | None = None,
        cache_authorized_requests: bool | None = None,
        log_digest_key: str | None = None,
        response_mode: Literal["buffered", "tee"] | None = None,
        hit_response_mode: Any = _UNSET,
        hit_stream_chunk_size: int | None = None,
    ) -> None:
        """Build a ``CacheSettings`` instance from explicit arguments or environment.

        All keyword arguments are optional. When omitted, each field is read from the
        matching ``SEMANTIC_CACHE_*`` (or provider-specific) environment variable and
        falls back to its documented default.

        Use the sentinel value ``_UNSET`` (the module default) to distinguish a field
        that was not supplied (falls back to env) from one explicitly set to ``None``
        (disables the feature).

        Args:
            disable_proxy_app_docs: Hide docs URLs for the proxy app.
            top_k_candidates: Max nearest-neighbor candidates from pgvector (>= 1).
            threshold: Primary cosine similarity gate [0.0, 1.0].
            rejection_threshold: Optional stricter second-stage cutoff. Must be >=
                threshold when set. Pass ``None`` explicitly to disable.
            pg_uri: PostgreSQL URI with pgvector extension.
            pg_ensure_schema: When True (default), create cache tables and indexes on
                first use via ``AsyncPgVectorStore.ensure_schema``. Set False when
                schema is managed externally (migrations, DBA) or the DB role lacks
                DDL privileges.
            redis_uri: Redis URI; empty or whitespace-only disables Redis.
            redis_ttl_seconds: Default TTL for Redis-cached responses in seconds.
            pg_ttl_days: Optional TTL for Postgres cache rows in fractional days.
                Pass ``None`` explicitly to disable expiry.
            pgvector_hnsw_m: HNSW graph connectivity for new pgvector indexes.
            pgvector_hnsw_ef_construction: HNSW build candidate list size.
            pgvector_hnsw_ef_search: Optional per-query HNSW search breadth override.
                Pass ``None`` explicitly to use the database default.
            pg_pool_size: Postgres connection pool minimum size.
            pg_pool_max_overflow: Postgres pool overflow slots above pool_size.
            embed_timeout_seconds: Timeout budget for embedder calls. Pass ``None``
                explicitly to disable.
            store_timeout_seconds: Timeout budget for Postgres/Redis ops. Pass
                ``None`` explicitly to disable.
            upstream_timeout_seconds: Timeout budget for upstream ASGI call. Pass
                ``None`` explicitly to disable.
            embedder_type: Embedder backend (``openai``, ``cohere``, ``voyage``,
                ``huggingface``, ``ollama``).
            hugging_face_api_key: Hugging Face API key.
            openai_api_key: OpenAI API key.
            cohere_api_key: Cohere API key.
            cohere_embedding_model: Cohere embedding model id.
            cohere_embedding_dimensions: Cohere embedding vector width.
            cohere_input_type: Cohere input_type hint.
            voyage_api_key: Voyage API key.
            voyage_embedding_model: Voyage embedding model id.
            voyage_embedding_dimensions: Voyage embedding vector width.
            voyage_input_type: Voyage input_type hint.
            ollama_api_key: Ollama API key.
            ollama_base_url: OpenAI-compatible API root for Ollama (must include /v1).
            ollama_embedding_model: Ollama embedding model id (required when
                embedder_type is ``ollama``).
            ollama_embedding_dimensions: Ollama embedding vector width (required when
                embedder_type is ``ollama``).
            circuit_breaker_429_enabled: Enable the 429-triggered circuit breaker.
            circuit_breaker_429_consecutive_limit: Consecutive 429s to open circuit.
            circuit_breaker_429_open_seconds: Cooldown window in seconds.
            middleware_flight_lock_max_entries: Max in-flight lock keys retained.
            middleware_flight_lock_acquire_timeout_seconds: Max wait for flight lock.
                Pass ``None`` explicitly to wait indefinitely.
            require_cache_scope: When True, every request must supply a non-empty
                scope (multi-tenant mode).
            cache_authorized_requests: When True, cache requests with Authorization
                header.
            log_digest_key: HMAC secret for prompt-derived log fields.
            response_mode: Miss delivery mode (``buffered`` or ``tee``).
            hit_response_mode: Hit delivery mode (``single`` or ``stream``).
            hit_stream_chunk_size: Chunk byte size for stream hit delivery.

        Raises:
            ValueError: When mandatory validation fails (e.g. invalid enum value,
                out-of-range numeric, or Ollama missing required fields).
        """
        _u = _UNSET

        # Whether hit_response_mode was set explicitly by the caller (vs env-derived).
        _hit_rm_explicit = hit_response_mode is not _u

        # ---- simple boolean flags ----------------------------------------
        self.disable_proxy_app_docs: bool = (
            disable_proxy_app_docs
            if disable_proxy_app_docs is not None
            else _parse_bool(
                os.getenv("SEMANTIC_CACHE_DISABLE_PROXY_APP_DOCS"), default=True
            )
        )

        # ---- numeric: top_k_candidates ------------------------------------
        _raw_top_k: int
        if top_k_candidates is not None:
            _raw_top_k = top_k_candidates
        else:
            _env_top_k = os.getenv("SEMANTIC_CACHE_TOP_K_CANDIDATES")
            _raw_top_k = int(_env_top_k) if _env_top_k and _env_top_k.strip() else 1
        if _raw_top_k < 1:
            raise ValueError(
                f"top_k_candidates must be >= 1, got {_raw_top_k!r}"
            )
        self.top_k_candidates: int = _raw_top_k

        # ---- threshold ---------------------------------------------------
        _raw_threshold: float
        if threshold is not None:
            _raw_threshold = threshold
        else:
            _env_t = os.getenv("SEMANTIC_CACHE_THRESHOLD")
            _raw_threshold = float(_env_t) if _env_t and _env_t.strip() else 0.95
        if not (0.0 <= _raw_threshold <= 1.0):
            raise ValueError(
                f"threshold must be in [0.0, 1.0], got {_raw_threshold!r}"
            )
        self.threshold: float = _raw_threshold

        # ---- rejection_threshold -----------------------------------------
        if rejection_threshold is _u:
            _rt_raw: float | None = _parse_optional_float(
                os.getenv("SEMANTIC_CACHE_REJECTION_THRESHOLD"),
                "SEMANTIC_CACHE_REJECTION_THRESHOLD",
            )
        else:
            _rt_raw = rejection_threshold
        if _rt_raw is not None and not (0.0 <= _rt_raw <= 1.0):
            raise ValueError(
                f"rejection_threshold must be in [0.0, 1.0], got {_rt_raw!r}"
            )
        self.rejection_threshold: float | None = _rt_raw

        # ---- pg_uri ------------------------------------------------------
        if pg_uri is not None:
            self.pg_uri: str = pg_uri
        else:
            self.pg_uri = os.getenv("SEMANTIC_CACHE_PG_URI", "")
        if self.pg_uri.strip() == "":
            raise ValueError("SEMANTIC_CACHE_PG_URI is required")

        # ---- pg_ensure_schema --------------------------------------------
        if pg_ensure_schema is not None:
            self.pg_ensure_schema: bool = pg_ensure_schema
        else:
            self.pg_ensure_schema = _parse_bool(
                os.getenv("SEMANTIC_CACHE_PG_ENSURE_SCHEMA"), default=True
            )

        # ---- redis_uri ---------------------------------------------------
        if redis_uri is not None:
            self.redis_uri: str = redis_uri
        else:
            self.redis_uri = os.getenv("SEMANTIC_CACHE_REDIS_URI", "")

        # ---- redis_ttl_seconds -------------------------------------------
        if redis_ttl_seconds is not None:
            self.redis_ttl_seconds: int = redis_ttl_seconds
        else:
            _env_rttl = os.getenv("SEMANTIC_CACHE_REDIS_TTL_SECONDS")
            self.redis_ttl_seconds = (
                int(_env_rttl) if _env_rttl and _env_rttl.strip() else 3600
            )

        # ---- pg_ttl_days -------------------------------------------------
        if pg_ttl_days is _u:
            self.pg_ttl_days: float | None = _parse_optional_float(
                os.getenv("SEMANTIC_CACHE_PG_TTL_DAYS"),
                "SEMANTIC_CACHE_PG_TTL_DAYS",
            )
        else:
            self.pg_ttl_days = pg_ttl_days
        if self.pg_ttl_days is not None and self.pg_ttl_days <= 0.0:
            raise ValueError(
                f"pg_ttl_days must be > 0.0, got {self.pg_ttl_days!r}"
            )

        # ---- pgvector_hnsw_m ---------------------------------------------
        if pgvector_hnsw_m is not None:
            _raw_hm = pgvector_hnsw_m
        else:
            _env_hm = os.getenv("SEMANTIC_CACHE_PGVECTOR_HNSW_M")
            _raw_hm = int(_env_hm) if _env_hm and _env_hm.strip() else 16
        if _raw_hm < 2:
            raise ValueError(
                f"pgvector_hnsw_m must be >= 2, got {_raw_hm!r}"
            )
        self.pgvector_hnsw_m: int = _raw_hm

        # ---- pgvector_hnsw_ef_construction --------------------------------
        if pgvector_hnsw_ef_construction is not None:
            _raw_hef = pgvector_hnsw_ef_construction
        else:
            _env_hef = os.getenv("SEMANTIC_CACHE_PGVECTOR_HNSW_EF_CONSTRUCTION")
            _raw_hef = int(_env_hef) if _env_hef and _env_hef.strip() else 64
        if _raw_hef < 4:
            raise ValueError(
                f"pgvector_hnsw_ef_construction must be >= 4, got {_raw_hef!r}"
            )
        self.pgvector_hnsw_ef_construction: int = _raw_hef

        # ---- pgvector_hnsw_ef_search -------------------------------------
        if pgvector_hnsw_ef_search is _u:
            self.pgvector_hnsw_ef_search: int | None = _parse_optional_int(
                os.getenv("SEMANTIC_CACHE_PGVECTOR_HNSW_EF_SEARCH"),
                "SEMANTIC_CACHE_PGVECTOR_HNSW_EF_SEARCH",
            )
        else:
            self.pgvector_hnsw_ef_search = pgvector_hnsw_ef_search
        if (
            self.pgvector_hnsw_ef_search is not None
            and self.pgvector_hnsw_ef_search < 1
        ):
            raise ValueError(
                "pgvector_hnsw_ef_search must be >= 1, "
                f"got {self.pgvector_hnsw_ef_search!r}"
            )

        # ---- pg_pool_size / overflow -------------------------------------
        if pg_pool_size is not None:
            self.pg_pool_size: int = pg_pool_size
        else:
            _env_ps = os.getenv("SEMANTIC_CACHE_PG_POOL_SIZE")
            self.pg_pool_size = int(_env_ps) if _env_ps and _env_ps.strip() else 10

        if pg_pool_max_overflow is not None:
            self.pg_pool_max_overflow: int = pg_pool_max_overflow
        else:
            _env_po = os.getenv("SEMANTIC_CACHE_PG_POOL_MAX_OVERFLOW")
            self.pg_pool_max_overflow = (
                int(_env_po) if _env_po and _env_po.strip() else 20
            )

        # ---- embed_timeout_seconds ---------------------------------------
        if embed_timeout_seconds is _u:
            _env_et = os.getenv("SEMANTIC_CACHE_EMBED_TIMEOUT_SECONDS")
            self.embed_timeout_seconds: float | None = (
                float(_env_et) if _env_et and _env_et.strip() else 10.0
            )
        else:
            self.embed_timeout_seconds = embed_timeout_seconds
        if self.embed_timeout_seconds is not None and self.embed_timeout_seconds <= 0:
            raise ValueError(
                "embed_timeout_seconds must be > 0 when set, "
                f"got {self.embed_timeout_seconds!r}"
            )

        # ---- store_timeout_seconds ---------------------------------------
        if store_timeout_seconds is _u:
            _env_st = os.getenv("SEMANTIC_CACHE_STORE_TIMEOUT_SECONDS")
            self.store_timeout_seconds: float | None = (
                float(_env_st) if _env_st and _env_st.strip() else 5.0
            )
        else:
            self.store_timeout_seconds = store_timeout_seconds
        if self.store_timeout_seconds is not None and self.store_timeout_seconds <= 0:
            raise ValueError(
                "store_timeout_seconds must be > 0 when set, "
                f"got {self.store_timeout_seconds!r}"
            )

        # ---- upstream_timeout_seconds ------------------------------------
        if upstream_timeout_seconds is _u:
            self.upstream_timeout_seconds: float | None = _parse_optional_float(
                os.getenv("SEMANTIC_CACHE_UPSTREAM_TIMEOUT_SECONDS"),
                "SEMANTIC_CACHE_UPSTREAM_TIMEOUT_SECONDS",
            )
        else:
            self.upstream_timeout_seconds = upstream_timeout_seconds
        if (
            self.upstream_timeout_seconds is not None
            and self.upstream_timeout_seconds <= 0
        ):
            raise ValueError(
                "upstream_timeout_seconds must be > 0 when set, "
                f"got {self.upstream_timeout_seconds!r}"
            )

        # ---- embedder_type -----------------------------------------------
        if embedder_type is not None:
            _raw_et = embedder_type
        else:
            _raw_et = os.getenv("SEMANTIC_CACHE_EMBEDDER_TYPE", "huggingface")
        if _raw_et not in _VALID_EMBEDDER_TYPES:
            raise ValueError(
                f"embedder_type must be one of {sorted(_VALID_EMBEDDER_TYPES)}, "
                f"got {_raw_et!r}"
            )
        self.embedder_type: EmbedderType = _raw_et  # type: ignore[assignment]

        # ---- API keys ----------------------------------------------------
        _hf_env = (
            os.getenv("HUGGINGFACE_API_KEY")
            or os.getenv("SEMANTIC_CACHE_HUGGING_FACE_API_KEY")
        ) or None
        self.hugging_face_api_key: str | None = (
            hugging_face_api_key if hugging_face_api_key is not _u else _hf_env
        )

        _oa_env = (
            os.getenv("OPENAI_API_KEY") or os.getenv("SEMANTIC_CACHE_OPENAI_API_KEY")
        ) or None
        self.openai_api_key: str | None = (
            openai_api_key if openai_api_key is not _u else _oa_env
        )

        _co_env = (
            os.getenv("COHERE_API_KEY") or os.getenv("SEMANTIC_CACHE_COHERE_API_KEY")
        ) or None
        self.cohere_api_key: str | None = (
            cohere_api_key if cohere_api_key is not _u else _co_env
        )

        self.cohere_embedding_model: str | None = (
            cohere_embedding_model
            if cohere_embedding_model is not _u
            else (os.getenv("SEMANTIC_CACHE_COHERE_EMBEDDING_MODEL") or None)
        )
        _ced: int | None = (
            cohere_embedding_dimensions
            if cohere_embedding_dimensions is not _u
            else _parse_optional_int(
                os.getenv("SEMANTIC_CACHE_COHERE_EMBEDDING_DIMENSIONS"),
                "SEMANTIC_CACHE_COHERE_EMBEDDING_DIMENSIONS",
            )
        )
        if _ced is not None and _ced < 1:
            raise ValueError(
                f"cohere_embedding_dimensions must be >= 1, got {_ced!r}"
            )
        self.cohere_embedding_dimensions: int | None = _ced
        self.cohere_input_type: str | None = (
            cohere_input_type
            if cohere_input_type is not _u
            else (os.getenv("SEMANTIC_CACHE_COHERE_INPUT_TYPE") or None)
        )

        _vo_env = (
            os.getenv("VOYAGE_API_KEY") or os.getenv("SEMANTIC_CACHE_VOYAGE_API_KEY")
        ) or None
        self.voyage_api_key: str | None = (
            voyage_api_key if voyage_api_key is not _u else _vo_env
        )

        self.voyage_embedding_model: str | None = (
            voyage_embedding_model
            if voyage_embedding_model is not _u
            else (os.getenv("SEMANTIC_CACHE_VOYAGE_EMBEDDING_MODEL") or None)
        )
        _ved: int | None = (
            voyage_embedding_dimensions
            if voyage_embedding_dimensions is not _u
            else _parse_optional_int(
                os.getenv("SEMANTIC_CACHE_VOYAGE_EMBEDDING_DIMENSIONS"),
                "SEMANTIC_CACHE_VOYAGE_EMBEDDING_DIMENSIONS",
            )
        )
        if _ved is not None and _ved < 1:
            raise ValueError(
                f"voyage_embedding_dimensions must be >= 1, got {_ved!r}"
            )
        self.voyage_embedding_dimensions: int | None = _ved
        self.voyage_input_type: str | None = (
            voyage_input_type
            if voyage_input_type is not _u
            else (os.getenv("SEMANTIC_CACHE_VOYAGE_INPUT_TYPE") or None)
        )

        _ol_env = (
            os.getenv("OLLAMA_API_KEY") or os.getenv("SEMANTIC_CACHE_OLLAMA_API_KEY")
        ) or None
        self.ollama_api_key: str | None = (
            ollama_api_key if ollama_api_key is not _u else _ol_env
        )

        if ollama_base_url is not None:
            self.ollama_base_url: str = ollama_base_url
        else:
            self.ollama_base_url = os.getenv(
                "SEMANTIC_CACHE_OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"
            )

        self.ollama_embedding_model: str | None = (
            ollama_embedding_model
            if ollama_embedding_model is not _u
            else (os.getenv("SEMANTIC_CACHE_OLLAMA_EMBEDDING_MODEL") or None)
        )
        _oed: int | None = (
            ollama_embedding_dimensions
            if ollama_embedding_dimensions is not _u
            else _parse_optional_int(
                os.getenv("SEMANTIC_CACHE_OLLAMA_EMBEDDING_DIMENSIONS"),
                "SEMANTIC_CACHE_OLLAMA_EMBEDDING_DIMENSIONS",
            )
        )
        if _oed is not None and _oed < 1:
            raise ValueError(
                f"ollama_embedding_dimensions must be >= 1, got {_oed!r}"
            )
        self.ollama_embedding_dimensions: int | None = _oed

        # ---- circuit breaker ---------------------------------------------
        self.circuit_breaker_429_enabled: bool = (
            circuit_breaker_429_enabled
            if circuit_breaker_429_enabled is not None
            else _parse_bool(
                os.getenv("SEMANTIC_CACHE_CIRCUIT_BREAKER_429_ENABLED"), default=False
            )
        )
        if circuit_breaker_429_consecutive_limit is not None:
            _raw_cbl = circuit_breaker_429_consecutive_limit
        else:
            _env_cbl = os.getenv("SEMANTIC_CACHE_CIRCUIT_BREAKER_429_CONSECUTIVE_LIMIT")
            _raw_cbl = int(_env_cbl) if _env_cbl and _env_cbl.strip() else 5
        if _raw_cbl < 1:
            raise ValueError(
                "circuit_breaker_429_consecutive_limit must be >= 1, "
                f"got {_raw_cbl!r}"
            )
        self.circuit_breaker_429_consecutive_limit: int = _raw_cbl

        if circuit_breaker_429_open_seconds is not None:
            _raw_cos = circuit_breaker_429_open_seconds
        else:
            _env_cos = os.getenv("SEMANTIC_CACHE_CIRCUIT_BREAKER_429_OPEN_SECONDS")
            _raw_cos = float(_env_cos) if _env_cos and _env_cos.strip() else 60.0
        if _raw_cos <= 0:
            raise ValueError(
                "circuit_breaker_429_open_seconds must be > 0, "
                f"got {_raw_cos!r}"
            )
        self.circuit_breaker_429_open_seconds: float = _raw_cos

        # ---- flight lock -------------------------------------------------
        if middleware_flight_lock_max_entries is not None:
            _raw_fle = middleware_flight_lock_max_entries
        else:
            _env_fle = os.getenv("SEMANTIC_CACHE_MIDDLEWARE_FLIGHT_LOCK_MAX_ENTRIES")
            _raw_fle = int(_env_fle) if _env_fle and _env_fle.strip() else 4096
        if _raw_fle < 1:
            raise ValueError(
                "middleware_flight_lock_max_entries must be >= 1, "
                f"got {_raw_fle!r}"
            )
        self.middleware_flight_lock_max_entries: int = _raw_fle

        if middleware_flight_lock_acquire_timeout_seconds is _u:
            self.middleware_flight_lock_acquire_timeout_seconds: float | None = (
                _parse_optional_float(
                    os.getenv(
                        "SEMANTIC_CACHE_MIDDLEWARE_FLIGHT_LOCK_ACQUIRE_TIMEOUT_SECONDS"
                    ),
                    "SEMANTIC_CACHE_MIDDLEWARE_FLIGHT_LOCK_ACQUIRE_TIMEOUT_SECONDS",
                )
            )
        else:
            self.middleware_flight_lock_acquire_timeout_seconds = (
                middleware_flight_lock_acquire_timeout_seconds
            )
        if (
            self.middleware_flight_lock_acquire_timeout_seconds is not None
            and self.middleware_flight_lock_acquire_timeout_seconds <= 0
        ):
            raise ValueError(
                "middleware_flight_lock_acquire_timeout_seconds must be > 0 when "
                "set, got "
                f"{self.middleware_flight_lock_acquire_timeout_seconds!r}"
            )

        # ---- security/tenancy flags --------------------------------------
        self.require_cache_scope: bool = (
            require_cache_scope
            if require_cache_scope is not None
            else _parse_bool(
                os.getenv("SEMANTIC_CACHE_REQUIRE_CACHE_SCOPE"), default=False
            )
        )
        self.cache_authorized_requests: bool = (
            cache_authorized_requests
            if cache_authorized_requests is not None
            else _parse_bool(
                os.getenv("SEMANTIC_CACHE_CACHE_AUTHORIZED_REQUESTS"), default=False
            )
        )

        # ---- log_digest_key ----------------------------------------------
        _raw_ldk = (
            log_digest_key
            if log_digest_key is not None
            else os.getenv("SEMANTIC_CACHE_LOG_DIGEST_KEY", "")
        )
        self.log_digest_key: str = (
            _raw_ldk if _raw_ldk and _raw_ldk.strip() else _DEFAULT_LOG_DIGEST_KEY
        )

        # ---- response_mode -----------------------------------------------
        if response_mode is not None:
            _raw_rm: str = response_mode
        else:
            _raw_rm = os.getenv("SEMANTIC_CACHE_RESPONSE_MODE", "buffered")
        if _raw_rm not in _VALID_RESPONSE_MODES:
            raise ValueError(
                f"response_mode must be one of {sorted(_VALID_RESPONSE_MODES)}, "
                f"got {_raw_rm!r}"
            )
        self.response_mode: Literal["buffered", "tee"] = _raw_rm  # type: ignore[assignment]

        # ---- hit_response_mode -------------------------------------------
        if hit_response_mode is not _u:
            if hit_response_mode is None:
                _raw_hrm = "single"
                _hit_rm_explicit = False
            else:
                _raw_hrm = hit_response_mode
                _hit_rm_explicit = True
        else:
            _env_hrm = os.getenv("SEMANTIC_CACHE_HIT_RESPONSE_MODE", "")
            _env_hrm_stripped = _env_hrm.strip()
            _raw_hrm = _env_hrm_stripped if _env_hrm_stripped else "single"
            _hit_rm_explicit = bool(_env_hrm_stripped)
        if _raw_hrm not in _VALID_HIT_RESPONSE_MODES:
            raise ValueError(
                f"hit_response_mode must be one of "
                f"{sorted(_VALID_HIT_RESPONSE_MODES)}, got {_raw_hrm!r}"
            )
        # Auto-default: when response_mode is tee and hit_response_mode was not
        # explicitly set, coerce to "stream" for symmetric delivery.
        if self.response_mode == "tee" and not _hit_rm_explicit:
            _raw_hrm = "stream"
        self.hit_response_mode: Literal["single", "stream"] = _raw_hrm  # type: ignore[assignment]

        # ---- hit_stream_chunk_size ---------------------------------------
        if hit_stream_chunk_size is not None:
            _raw_hsc = hit_stream_chunk_size
        else:
            _env_hsc = os.getenv("SEMANTIC_CACHE_HIT_STREAM_CHUNK_SIZE")
            _raw_hsc = int(_env_hsc) if _env_hsc and _env_hsc.strip() else 0
        if _raw_hsc < 0:
            raise ValueError(
                f"hit_stream_chunk_size must be >= 0, got {_raw_hsc!r}"
            )
        self.hit_stream_chunk_size: int = _raw_hsc

        # ------------------------------------------------------------------ #
        # Cross-field validation
        # ------------------------------------------------------------------ #
        self._validate_rejection_threshold()
        self._validate_ollama_embedding_settings()

    # ------------------------------------------------------------------ #
    # Validators
    # ------------------------------------------------------------------ #

    def _validate_rejection_threshold(self) -> None:
        """Validate the rejection threshold against the primary threshold.

        Raises:
            ValueError: When rejection_threshold < threshold.
        """
        if self.rejection_threshold is None:
            return
        if self.rejection_threshold < self.threshold:
            raise ValueError(
                "rejection_threshold must be >= threshold when set "
                f"(got rejection_threshold={self.rejection_threshold!r}, "
                f"threshold={self.threshold!r}). "
                "Otherwise the second stage cannot reject any candidate that passed "
                "the primary similarity gate."
            )
        if self.rejection_threshold == self.threshold:
            warnings.warn(
                (
                    "rejection_threshold equals threshold; the second similarity "
                    "stage cannot reject any candidate that passed the primary gate. "
                    "Set rejection_threshold strictly above threshold for a "
                    "meaningful second stage, or unset rejection_threshold to use a "
                    "single threshold."
                ),
                UserWarning,
                stacklevel=3,
            )

    def _validate_ollama_embedding_settings(self) -> None:
        """Ensure Ollama embedder settings include model id and vector width.

        Raises:
            ValueError: When embedder_type is ``ollama`` but model or dimensions are
                missing.
        """
        if self.embedder_type != "ollama":
            return
        if (
            self.ollama_embedding_model is None
            or not self.ollama_embedding_model.strip()
        ):
            raise ValueError(
                "ollama_embedding_model is required when embedder_type is 'ollama' "
                "(set SEMANTIC_CACHE_OLLAMA_EMBEDDING_MODEL)."
            )
        if self.ollama_embedding_dimensions is None:
            raise ValueError(
                "ollama_embedding_dimensions is required when embedder_type is "
                "'ollama' (set SEMANTIC_CACHE_OLLAMA_EMBEDDING_DIMENSIONS)."
            )

    # ------------------------------------------------------------------ #
    # Factory helpers
    # ------------------------------------------------------------------ #

    @classmethod
    def from_env(cls) -> "CacheSettings":
        """Construct ``CacheSettings`` entirely from environment variables.

        Returns:
            Settings instance populated from ``SEMANTIC_CACHE_*`` env vars.
        """
        return cls()

    def replace(self, **kwargs: Any) -> "CacheSettings":
        """Return a new ``CacheSettings`` instance with selected fields overridden.

        All fields not present in ``kwargs`` are copied from this instance. This is
        the native replacement for the pydantic ``model_copy(update={...})`` pattern.

        Args:
            **kwargs: Field names and new values to override.

        Returns:
            New ``CacheSettings`` with the specified fields replaced.

        Examples:
            >>> settings = CacheSettings(threshold=0.9)
            >>> faster = settings.replace(embed_timeout_seconds=2.0)
        """
        current = {
            field: getattr(self, field)
            for field in (
                "disable_proxy_app_docs",
                "top_k_candidates",
                "threshold",
                "rejection_threshold",
                "pg_uri",
                "pg_ensure_schema",
                "redis_uri",
                "redis_ttl_seconds",
                "pg_ttl_days",
                "pgvector_hnsw_m",
                "pgvector_hnsw_ef_construction",
                "pgvector_hnsw_ef_search",
                "pg_pool_size",
                "pg_pool_max_overflow",
                "embed_timeout_seconds",
                "store_timeout_seconds",
                "upstream_timeout_seconds",
                "embedder_type",
                "hugging_face_api_key",
                "openai_api_key",
                "cohere_api_key",
                "cohere_embedding_model",
                "cohere_embedding_dimensions",
                "cohere_input_type",
                "voyage_api_key",
                "voyage_embedding_model",
                "voyage_embedding_dimensions",
                "voyage_input_type",
                "ollama_api_key",
                "ollama_base_url",
                "ollama_embedding_model",
                "ollama_embedding_dimensions",
                "circuit_breaker_429_enabled",
                "circuit_breaker_429_consecutive_limit",
                "circuit_breaker_429_open_seconds",
                "middleware_flight_lock_max_entries",
                "middleware_flight_lock_acquire_timeout_seconds",
                "require_cache_scope",
                "cache_authorized_requests",
                "log_digest_key",
                "response_mode",
                "hit_response_mode",
                "hit_stream_chunk_size",
            )
        }
        current.update(kwargs)
        return CacheSettings(**current)

    def __repr__(self) -> str:
        """Return a developer-friendly representation omitting sensitive fields."""
        return (
            f"CacheSettings("
            f"embedder_type={self.embedder_type!r}, "
            f"threshold={self.threshold!r}, "
            f"pg_uri=<redacted>, "
            f"response_mode={self.response_mode!r}"
            f")"
        )


def get_cache_settings() -> CacheSettings:
    """Load ``CacheSettings`` from process environment variables.

    Variables use the ``SEMANTIC_CACHE_`` prefix (see ``CacheSettings`` fields).

    Returns:
        Settings instance populated from environment variables.
    """
    return CacheSettings.from_env()
