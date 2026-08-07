# Atlas Knowledge API Technical Specification

The production query endpoint is `POST /v2/knowledge/query`. Every request must carry the
tenant header `X-Atlas-Tenant`. The request budget is 9,000 milliseconds and the context
builder may select at most 12 chunks. Official acceptance traffic bypasses the retrieval cache.

The stable specification identifier is `SPEC-ATLAS-2026-08`. Responses must cite the selected
evidence chunk identifiers; the service must not expose local filesystem paths.
