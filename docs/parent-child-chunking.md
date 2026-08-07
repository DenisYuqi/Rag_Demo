# Parent/child chunking

Ingestion derives two revision-scoped levels from each normalized extraction block:

- Parents default to 1536 tokens, do not overlap, and never cross an extraction block.
- Children default to 512 tokens with 128 tokens of overlap. Every child is wholly contained in one parent.
- Only children are embedded and written to Chroma and BM25. Parents are stored in SQLite in `parent_chunks` under the same immutable revision.

Retrieval, fusion, reranking, evidence assessment, and citations continue to use child IDs and child text. After evidence approval, context construction resolves the selected children to their parents, preserves child ranking, removes duplicate parents, and supplies the larger parent text to generation. The supporting child ID remains the only allowed citation identity.

Configure the hierarchy with:

```dotenv
RAG_MVP_CHUNK_TARGET_TOKENS=512
RAG_MVP_CHUNK_OVERLAP_TOKENS=128
RAG_MVP_PARENT_CHUNK_TARGET_TOKENS=1536
```

The parent target must be at least the child target, and child overlap must be smaller than the child target. Restart the service after changing these values.

## Migration and reindexing

Parent mappings are part of the chunk, dense-index, BM25, cache, and revision identities. Existing single-level revisions are intentionally incompatible and are not upgraded in place.

After deploying this version:

1. Restart the application so the additive SQLite migration creates `parent_chunks`.
2. Run a full reindex for each retrieval profile/data root. Do not reuse or manually activate an earlier revision.
3. Confirm readiness and perform a grounded query before removing old revision artifacts through the normal lifecycle cleanup.

Publication fails closed if a child references a missing parent, if an unreferenced parent is present, or if the SQLite parent inventory differs from the staged revision digest. A failed unpublished revision is cleaned up without changing the active revision.
