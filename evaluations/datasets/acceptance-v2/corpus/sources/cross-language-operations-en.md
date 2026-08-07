# Cross-language Retrieval Operations

Chinese and English questions share the same immutable bilingual corpus revision. Retrieval
may match evidence written in another language, but the final answer follows the user's chosen
response language. Cache identity stores only a SHA-256 digest of the normalized query and the
complete retrieval configuration; raw query text is never stored in cache telemetry.

The cache operations contract identifier is `CACHE-XLG-2026`.
