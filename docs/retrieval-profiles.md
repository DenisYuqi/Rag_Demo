# Selectable retrieval profiles

The workbench exposes a **Retrieval profile / 检索模型** selector with two independent choices:

- `openai-api` keeps the existing OpenAI-compatible embedding and optional listwise reranking path. Evaluation HTTP calls without an explicit profile remain bound to this profile for backward compatibility.
- `bge-local` uses `BAAI/bge-m3` for 1024-dimensional L2-normalized embeddings and `BAAI/bge-reranker-v2-m3` for local cross-encoder reranking. Answer generation still uses the configured OpenAI-compatible generation model.

Each profile has its own data root, document catalog, ingestion jobs, sessions, Chroma index, and BM25 snapshot. Upload, reindex, delete, and Chat operations apply only to the profile selected in the workbench. Switching profiles never queries an index created by the other embedding model.

## Profile-specific Evaluation

The Evaluation and Comparison tabs follow the same workbench selector. Switching from `openai-api` to `bge-local` refreshes the dataset, plan, run, comparison, progress, and report views from the BGE profile's isolated database and artifact directories. Existing run selections are discarded during a switch, so a run identifier from one profile is never resolved against the other profile.

BGE evaluation installs each registered corpus into a per-run workspace beneath the BGE data root, then executes retrieval with `BAAI/bge-m3` and reranking with `BAAI/bge-reranker-v2-m3`. Generation remains on the configured OpenAI-compatible model. BGE runs can therefore have longer cold-start latency and still incur generation API cost even though embedding and reranking are local. Static OpenAI release evidence is not presented as BGE evidence.

Evaluation and Comparison API routes accept `?retrieval_profile=bge-local`; omitting the parameter preserves the existing `openai-api` behavior. Same-origin report and artifact links rendered in the BGE view include that qualifier, including polling and downloads. Unknown profile identifiers fail closed with `evaluation_unavailable` or `comparison_unavailable` rather than falling back to another profile.

Evaluation scoring is restart-bound configuration shared by both retrieval profiles. `RAG_MVP_EVALUATION_SCORER_BACKEND=legacy` preserves the deterministic scorer. Setting it to `ragas` uses the configured OpenAI-compatible judge for semantic faithfulness and final-context precision while retaining deterministic compliance, style, refusal, safety, and operational checks. The Evidence Assessor remains enabled in both modes. Use `RAG_MVP_EVALUATION_RAGAS_JUDGE_MODEL` to override the generation model for judging, and bound judge work with the timeout, retry, and concurrency settings documented in `.env.example`. Reports record the backend and judge identity; scores from different backends are not comparison-compatible.

## First local use

FlagEmbedding loads models lazily. The first BGE ingestion loads/downloads `BAAI/bge-m3`; the first `hybrid-rerank` question loads/downloads `BAAI/bge-reranker-v2-m3`. These artifacts require several gigabytes of persistent disk and memory, and cold loading can exceed ordinary request deadlines. Production deployments should provision the configured `RAG_MVP_BGE_MODEL_CACHE_DIR` before serving traffic or warm both models after startup.

Compose mounts the `rag-mvp-model-cache` volume at `/var/cache/huggingface`. The application data volume separately stores the BGE corpus under `/var/lib/rag-mvp/profiles/bge-local` by default.

## Local settings

The model names, data/cache roots, device, FP16 mode, batch sizes, maximum input lengths, and local-provider/QA stage deadlines use the `RAG_MVP_BGE_*` environment variables documented in `.env.example`. The BGE profile defaults to larger provider, total-request, reranking, and evidence-assessment budgets than `openai-api`; changing them does not alter the API profile. Keep FP16 disabled on CPU. On a compatible accelerator, set `RAG_MVP_BGE_DEVICE` to the device accepted by FlagEmbedding (for example `cuda:0`) and enable FP16 only after validating model output and memory behavior.

For the tested Windows CUDA setup, dependency pins, verification commands, performance observations, and troubleshooting guidance, see [BGE local inference CUDA acceleration](bge-cuda-acceleration.md).

To remove the BGE choice without affecting the OpenAI index, set `RAG_MVP_BGE_PROFILE_ENABLED=false`. Changing `RAG_MVP_DEFAULT_RETRIEVAL_PROFILE` only changes the initial workbench selection; it does not change the HTTP API provider.
