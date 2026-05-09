# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.21] - 2026-05-09

### Added

- `trusted_extract_scope_from_server_side` in `semanticcache.middleware.core.extractors`: reads cache scope only from `request.state.cache_scope` or `request.state.tenant_id` for use after trusted auth or gateway middleware sets `request.state`.

### Changed

- Renamed `default_extract_scope` to `default_extract_scope_from_request_context` (same behavior: `X-Semantic-Cache-Scope` header and JSON `cache_scope` / `tenant_id`). Documented that this path trusts client-supplied values and is aimed at single-tenant or trusted-proxy setups.
- `SemanticCacheMiddleware` documentation now refers to the renamed default extractor and points multi-tenant deployments at a custom `extract_scope` or the new trusted helper.

### Removed

- **`default_extract_scope`** export from `semanticcache.middleware.core` (use **`default_extract_scope_from_request_context`** instead).

### Documentation

- README quick-start and `docs/cache-tuning.md`: clarified scope trust boundaries and added an example wrapping `trusted_extract_scope_from_server_side` for `extract_scope`.
