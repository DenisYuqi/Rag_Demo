# Selectable retrieval profiles

The workbench exposes a **Retrieval profile / 检索模型** selector with two independent choices:

- `openai-api` keeps the existing OpenAI-compatible embedding and optional listwise reranking path. The HTTP API remains permanently bound to this profile.
- `bge-local` uses `BAAI/bge-m3` for 1024-dimensional L2-normalized embeddings and `BAAI/bge-reranker-v2-m3` for local cross-encoder reranking. Answer generation still uses the configured OpenAI-compatible generation model.

Each profile has its own data root, document catalog, ingestion jobs, sessions, Chroma index, and BM25 snapshot. Upload, reindex, delete, and Chat operations apply only to the profile selected in the workbench. Switching profiles never queries an index created by the other embedding model.

## First local use

FlagEmbedding loads models lazily. The first BGE ingestion loads/downloads `BAAI/bge-m3`; the first `hybrid-rerank` question loads/downloads `BAAI/bge-reranker-v2-m3`. These artifacts require several gigabytes of persistent disk and memory, and cold loading can exceed ordinary request deadlines. Production deployments should provision the configured `RAG_MVP_BGE_MODEL_CACHE_DIR` before serving traffic or warm both models after startup.

Compose mounts the `rag-mvp-model-cache` volume at `/var/cache/huggingface`. The application data volume separately stores the BGE corpus under `/var/lib/rag-mvp/profiles/bge-local` by default.

## Local settings

The model names, data/cache roots, device, FP16 mode, batch sizes, and maximum input lengths use the `RAG_MVP_BGE_*` environment variables documented in `.env.example`. Keep FP16 disabled on CPU. On a compatible accelerator, set `RAG_MVP_BGE_DEVICE` to the device accepted by FlagEmbedding (for example `cuda:0`) and enable FP16 only after validating model output and memory behavior.

To remove the BGE choice without affecting the OpenAI index, set `RAG_MVP_BGE_PROFILE_ENABLED=false`. Changing `RAG_MVP_DEFAULT_RETRIEVAL_PROFILE` only changes the initial workbench selection; it does not change the HTTP API provider.
