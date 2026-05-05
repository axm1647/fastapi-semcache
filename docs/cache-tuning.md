## Cache behavior and similarity tuning

`SemanticCache` uses a two-stage retrieval model so you can trade off recall and precision without changing application code.

### Stage 1: nearest-neighbor search (top-k)

The first stage embeds the query and runs a pgvector similarity search:

- **`SEMANTIC_CACHE_THRESHOLD`** (`CacheSettings.threshold`):
  - Primary similarity gate in \[0.0, 1.0].
  - Candidates below this value are discarded by the vector store.
- **`SEMANTIC_CACHE_TOP_K_CANDIDATES`** (`CacheSettings.top_k_candidates`):
  - Maximum number of nearest neighbors returned from pgvector after applying the primary threshold.
  - Defaults to `1` for single-hit behavior.

After this stage you get up to `top_k_candidates` `CacheEntry` rows ordered from highest to lowest similarity, all with `similarity >= threshold`.

### Stage 2: optional rejection threshold

The second stage can apply a stricter similarity cutoff on the in-memory candidates:

- **`SEMANTIC_CACHE_REJECTION_THRESHOLD`** (`CacheSettings.rejection_threshold`):
  - When unset (`None` or empty env var), behavior matches the original single-threshold model: the best remaining candidate is accepted.
  - When set, the cache scans the candidates in order and selects the **first** entry whose `similarity >= rejection_threshold`.
  - If **no** candidate passes this stricter bar, the cache returns a **miss** (`is_hit=False`).

This is useful when you want to:

- Keep `SEMANTIC_CACHE_THRESHOLD` lower (for example `0.80`) to allow more candidates into the first stage.
- Enforce a higher bar (for example `0.90`) for actually serving a cached response.

### Example configurations

- **Strict, precision-first cache:**

  - `SEMANTIC_CACHE_THRESHOLD=0.90`
  - `SEMANTIC_CACHE_TOP_K_CANDIDATES=1`
  - `SEMANTIC_CACHE_REJECTION_THRESHOLD=` (unset)

  Only very similar neighbors are considered, and the single best one is accepted if it passes `0.90`.

- **More recall with a second-stage guard:**

  - `SEMANTIC_CACHE_THRESHOLD=0.80`
  - `SEMANTIC_CACHE_TOP_K_CANDIDATES=5`
  - `SEMANTIC_CACHE_REJECTION_THRESHOLD=0.90`

  Up to five neighbors with similarity at least `0.80` are fetched; the cache only serves a hit if at least one of them has similarity `>= 0.90`, otherwise it falls back to a miss.

### Notes

- If `SEMANTIC_CACHE_TOP_K_CANDIDATES` is less than `1`, it is treated as `1` internally.
- All thresholds are clamped to the inclusive range \[0.0, 1.0] by `CacheSettings`.

