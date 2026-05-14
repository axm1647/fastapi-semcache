# When not to use fastapi-semcache

Semantic caching is not a silver bullet. Before integrating `fastapi-semcache`, consider whether your use case actually benefits.

## Low hit-rate scenarios

**Mostly unique queries.** If every request asks something genuinely new (creative writing, one-off data analysis, ad-hoc exploration) cache hit rates will be low and the overhead of embedding + pgvector lookup may not pay off. A rough rule of thumb: if you do not expect a meaningful fraction of queries to be semantically similar to a prior query within your TTL window, skip the cache.

**Highly personalised responses.** If the same question requires different answers for different users or context (personalised recommendations, user-specific summaries), semantic caching can serve stale or wrong responses to the wrong user. Scope partitioning (`extract_scope`) helps here, but if every user's partition is essentially unique you gain little and pay the embedding cost on every request.

## Data freshness concerns

**Constantly changing data.** Real-time prices, live policy documents, rapidly-evolving feeds: cached responses become stale quickly and can actively mislead. If your upstream data changes faster than your cache TTL, caching is risky without aggressive invalidation.

**Strict correctness requirements.** In legal, medical, financial, or compliance contexts, a semantically similar but not identical query may require a materially different response. The similarity threshold is tunable but no threshold eliminates false positives entirely. If a wrong cached answer is worse than a slow correct one, cache selectively or not at all.

## Architecture mismatches

**Non-JSON responses.** The middleware stores and replays JSON object responses. Binary responses, file downloads, HTML pages, and plain-text bodies will not be cached. Streaming-only endpoints (SSE, WebSockets) are also outside the current scope.

**Side-effectful POST endpoints.** The middleware intercepts `POST` by default. If your POST endpoints mutate state (creating records, triggering workflows), returning a cached response silently skips the side effect. Either exclude those paths with `path_prefix` / a custom `extract_query` that returns `None`, or disable the middleware on those routes entirely.

**Already cached at a higher layer.** If your upstream LLM provider, API gateway, or CDN already caches responses aggressively, adding semantic caching is likely redundant. Check your cache hit metrics there first.

## Cost and operational tradeoffs

**Tiny LLM budgets.** If LLM costs are already negligible relative to engineering time, the complexity of maintaining threshold tuning, monitoring hit rates, and handling invalidation may not be worth the saving.

**No capacity to tune and monitor.** Semantic caching requires ongoing attention: threshold calibration, hit/miss ratio monitoring, and cache invalidation when upstream data changes. If your team does not have bandwidth for that, simpler solutions are less risky.

## When traditional caching is enough

If your queries are exact-match (same prompt, same response every time) traditional HTTP caching or exact-key Redis caching is simpler, faster, and cheaper. Semantic caching earns its keep when you need to match **similar intent** across paraphrased or reformatted queries, not verbatim strings.

Alternatives worth considering in these cases:

- HTTP cache headers (`Cache-Control`, `ETag`) for idempotent GET endpoints
- Exact-key caching (Redis, in-memory dict) for deterministic prompt/response pairs
- Rate limiting and circuit breakers only, if the goal is cost control rather than latency
- Prompt normalisation at the application layer before hitting the LLM, to increase exact-match rates without a vector database
