# 配置指南

应用配置由 Pydantic Settings 校验，字段统一使用 `RAG_MVP_` 环境变量前缀。本文列出日常最常用的选项；可复制的完整基线见 [`.env.example`](../.env.example)，类型、默认值和交叉约束以 [`src/rag_mvp/config/settings.py`](../src/rag_mvp/config/settings.py) 为准。

## 加载顺序

应用会读取仓库根目录的 `.env`；进程环境变量的优先级高于 dotenv 文件。配置在进程启动时加载并缓存，修改后必须重启应用。

建议用途：

- `.env.example`：可提交的安全模板，不包含真实凭据。
- `.env`：本机配置、密钥和个人覆盖，可从示例文件复制，不提交 Git。
- 生产环境：平台环境变量或 Secret 文件，不依赖工作目录中的 dotenv。

## 最小可用配置

源代码方式启动时，可在 `.env` 中配置：

```dotenv
RAG_MVP_PROVIDER_BACKEND=openai
RAG_MVP_OPENAI_API_KEY=replace-with-your-key
RAG_MVP_OPENAI_BASE_URL=https://api.openai.com/v1
RAG_MVP_EMBEDDING_MODEL=text-embedding-3-small
RAG_MVP_EMBEDDING_DIMENSION=1536
RAG_MVP_GENERATION_MODEL=gpt-4.1-mini
```

也可以只提供密钥文件路径：

```dotenv
RAG_MVP_OPENAI_API_KEY_FILE=C:/secure/location/openai_api_key
```

当直接密钥和文件同时存在时，直接配置的密钥优先。密钥文件使用 UTF-8 读取并去除首尾空白。

`offline` 是配置对象和确定性测试的安全默认值，但可执行应用只在 `openai` provider 配置完整时装配真实摄取和问答服务。未配置 provider 时进程仍能启动并响应 `/healthz`，而 `/readyz` 会诚实返回未就绪。

## 服务与存储

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `RAG_MVP_ENVIRONMENT` | `development` | `development`、`test` 或 `production` |
| `RAG_MVP_HOST` | `127.0.0.1` | ASGI 监听地址 |
| `RAG_MVP_PORT` | `8000` | ASGI 端口 |
| `RAG_MVP_DATA_ROOT` | `data` | OpenAI profile 的持久化根目录，不能是文件系统根目录 |
| `RAG_MVP_WORKBENCH_ENABLED` | `true` | 是否挂载 Gradio |
| `RAG_MVP_WORKBENCH_PATH` | `/workbench` | 工作台路径，不能覆盖 API 或运维端点 |
| `RAG_MVP_UPLOAD_MAX_BYTES` | `26214400` | 上传上限，默认 25 MiB |
| `RAG_MVP_LOG_LEVEL` | `INFO` | `DEBUG`、`INFO`、`WARNING` 或 `ERROR` |

不要让两个进程、多个 Uvicorn worker 或多个容器同时写入同一 `RAG_MVP_DATA_ROOT`。应用通过 `locks/writer.lock` 强制支持的单写入拓扑。

## OpenAI-compatible Provider

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `RAG_MVP_PROVIDER_BACKEND` | `offline` | 可执行服务使用 `openai` |
| `RAG_MVP_OPENAI_BASE_URL` | `https://api.openai.com/v1` | 兼容服务的 `/v1` 基地址 |
| `RAG_MVP_OPENAI_API_KEY` | 未设置 | 直接凭据，仅建议本地开发使用 |
| `RAG_MVP_OPENAI_API_KEY_FILE` | 未设置 | UTF-8 Secret 文件路径 |
| `RAG_MVP_OPENAI_PROXY_URL` | 未设置 | 可选 HTTP(S) provider 代理，按敏感值处理 |
| `RAG_MVP_OPENAI_SEND_DIMENSIONS` | `true` | 嵌入请求是否发送 dimensions 参数 |
| `RAG_MVP_OPENAI_MAX_TOKENS_PARAMETER` | `max_completion_tokens` | 兼容端点要求的 token 参数名 |
| `RAG_MVP_EMBEDDING_MODEL` | `text-embedding-3-small` | 嵌入模型 |
| `RAG_MVP_EMBEDDING_DIMENSION` | `1536` | 向量维度，必须与模型输出及已有索引一致 |
| `RAG_MVP_GENERATION_MODEL` | `gpt-4.1-mini` | 答案生成模型 |
| `RAG_MVP_RERANKING_MODEL` | 未设置 | OpenAI-compatible 列表式重排模型 |
| `RAG_MVP_PROVIDER_TIMEOUT_SECONDS` | `8` | 单次 provider 超时上限 |
| `RAG_MVP_PROVIDER_RETRY_LIMIT` | `1` | 额外重试次数，范围 0–5 |

若把默认检索模式设为 `hybrid-rerank`，必须同时配置 `RAG_MVP_RERANKING_MODEL`。更换嵌入模型或维度后不能复用旧向量索引，必须重建。

## 检索与父子分块

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `RAG_MVP_DEFAULT_RETRIEVAL_MODE` | `hybrid` | `dense`、`hybrid` 或 `hybrid-rerank` |
| `RAG_MVP_DENSE_CANDIDATE_LIMIT` | `20` | 稠密候选数 |
| `RAG_MVP_LEXICAL_CANDIDATE_LIMIT` | `20` | BM25 候选数 |
| `RAG_MVP_RERANK_CANDIDATE_LIMIT` | `10` | 进入重排的候选数 |
| `RAG_MVP_CONTEXT_CHUNK_LIMIT` | `5` | 最终上下文候选上限 |
| `RAG_MVP_RRF_K` | `60` | RRF 常数 |
| `RAG_MVP_DENSE_WEIGHT` | `1.0` | 稠密通道融合权重 |
| `RAG_MVP_LEXICAL_WEIGHT` | `1.0` | 词法通道融合权重 |
| `RAG_MVP_ALLOW_SINGLE_RETRIEVER_DEGRADATION` | `false` | 一个检索通道失败时是否降级 |
| `RAG_MVP_RETRIEVAL_CACHE_ENABLED` | `false` | 是否启用进程内检索缓存 |
| `RAG_MVP_RETRIEVAL_CACHE_MAX_ENTRIES` | `256` | 缓存条目上限 |
| `RAG_MVP_RETRIEVAL_CACHE_TTL_SECONDS` | `300` | 缓存 TTL |
| `RAG_MVP_CHUNK_TARGET_TOKENS` | `512` | 子块目标大小 |
| `RAG_MVP_CHUNK_OVERLAP_TOKENS` | `128` | 子块重叠 |
| `RAG_MVP_PARENT_CHUNK_TARGET_TOKENS` | `1536` | 父块目标大小 |

约束：重叠必须小于子块，父块不能小于子块，重排候选数不能小于上下文块数。修改分块配置后需要完整重建索引，不能在原修订上混用。原理见[父子分块与重建索引](parent-child-chunking.md)。

## BGE 本地检索

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `RAG_MVP_BGE_PROFILE_ENABLED` | `true` | 是否在工作台提供 BGE profile |
| `RAG_MVP_DEFAULT_RETRIEVAL_PROFILE` | `openai-api` | 工作台首次选择的 profile |
| `RAG_MVP_BGE_DATA_ROOT` | `<data_root>/profiles/bge-local` | BGE 独立数据根目录 |
| `RAG_MVP_BGE_EMBEDDING_MODEL` | `BAAI/bge-m3` | 本地嵌入模型 |
| `RAG_MVP_BGE_EMBEDDING_DIMENSION` | `1024` | `bge-m3` 固定使用 1024 |
| `RAG_MVP_BGE_RERANKING_MODEL` | `BAAI/bge-reranker-v2-m3` | 本地交叉编码重排模型 |
| `RAG_MVP_BGE_DEVICE` | `auto` | FlagEmbedding 接受的设备，如 `cpu`、`cuda:0` |
| `RAG_MVP_BGE_USE_FP16` | `false` | GPU 可验证后启用；CPU 必须保持 false |
| `RAG_MVP_BGE_MODEL_CACHE_DIR` | 未设置 | Hugging Face 模型缓存目录 |

BGE 的 batch size、最大输入长度和较长的阶段预算均可通过 `RAG_MVP_BGE_*` 变量调整，完整列表见 `.env.example`。BGE 数据根目录必须与主数据目录不同。首次调用会下载/加载模型，应预留数 GB 磁盘和内存；GPU 设置见[BGE CUDA 加速](bge-cuda-acceleration.md)。

## OCR

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `RAG_MVP_OCR_ENABLED` | `true` | 是否允许对无可用文本的 PDF 页面执行 OCR |
| `RAG_MVP_OCR_LANGUAGES` | `chi_sim+eng` | 传给 Tesseract 的语言组合 |

启用 OCR 还要求操作系统已经安装 Tesseract 和对应语言数据。Docker 镜像内置英文与简体中文包。关闭 OCR 后，扫描型 PDF 可能因没有可用文本而摄取失败。

## QA 截止时间与并发

问答使用总截止时间和阶段预算。主要变量包括：

- `RAG_MVP_QA_MAX_ACTIVE`、`RAG_MVP_QA_MAX_QUEUE`
- `RAG_MVP_QA_DEADLINE_SECONDS`
- `RAG_MVP_QA_QUEUE_BUDGET_SECONDS`
- `RAG_MVP_QA_VALIDATION_BUDGET_SECONDS`
- `RAG_MVP_QA_RETRIEVAL_BUDGET_SECONDS`
- `RAG_MVP_QA_EMBEDDING_BUDGET_SECONDS`
- `RAG_MVP_QA_EVIDENCE_ASSESSMENT_BUDGET_SECONDS`
- `RAG_MVP_QA_GENERATION_BUDGET_SECONDS`
- `RAG_MVP_QA_FINALIZATION_BUDGET_SECONDS`
- `RAG_MVP_RERANK_DEADLINE_SECONDS`

每个阶段预算必须小于总截止时间，生成预算与最终处理预算之和不能超过总截止时间。不要只通过放大总超时掩盖 provider、CPU 回退或模型冷加载问题；先用请求诊断和阶段指标定位瓶颈。

## 评测

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `RAG_MVP_EVALUATION_DATASET_ROOT` | `evaluations/datasets` | 版本化数据集目录 |
| `RAG_MVP_EVALUATION_RELEASE_ROOT` | `evaluations/releases` | 发布证据目录 |
| `RAG_MVP_EVALUATION_MAX_ACTIVE_JOBS` | `1` | 同时运行的评测任务数，范围 1–4 |
| `RAG_MVP_EVALUATION_SCORER_BACKEND` | `legacy` | `legacy` 或 `ragas` |
| `RAG_MVP_EVALUATION_RAGAS_JUDGE_MODEL` | 未设置 | 为空时复用生成模型 |
| `RAG_MVP_EVALUATION_RAGAS_MAX_CONCURRENCY` | `2` | judge 最大并发 |

`legacy` 使用确定性评分。`ragas` 会调用 OpenAI-compatible judge 评估语义指标，因此要求 provider 为 `openai`，也会增加时间和费用。不同评分后端的分数不应直接作为同一口径比较。

## 可观测性

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `RAG_MVP_TELEMETRY_EXPORTER` | `none` | `none`、`console` 或 `otlp` |
| `RAG_MVP_TELEMETRY_OTLP_TRACES_ENDPOINT` | 未设置 | 使用 `otlp` 时必填的 HTTP(S) `/v1/traces` 地址 |
| `RAG_MVP_TELEMETRY_EXPORT_TIMEOUT_SECONDS` | `5` | 导出超时 |
| `RAG_MVP_PRICING_VERSION` | `unconfigured` | 成本证据使用的显式定价版本 |

OTLP 地址不能包含用户信息、密码、查询参数或 fragment。应用日志、API 响应和诊断只暴露允许字段并执行脱敏，但运维系统仍应按敏感数据系统管理日志与 trace 存储。

## 容器专用变量

Compose 还读取一些不属于应用 Settings 的变量：

- `RAG_MVP_IMAGE`、`RAG_MVP_APP_VERSION`、`RAG_MVP_SOURCE_REVISION`
- `RAG_MVP_BIND_ADDRESS`、`RAG_MVP_HOST_PORT`、`RAG_MVP_CONTAINER_PORT`
- `RAG_MVP_DATA_VOLUME`、`RAG_MVP_MODEL_CACHE_VOLUME`
- `RAG_MVP_OPENAI_API_KEY_SECRET_FILE`

其中 `RAG_MVP_OPENAI_API_KEY_SECRET_FILE` 是宿主机文件路径，Compose 将其挂载为 `/run/secrets/openai_api_key`，再通过容器内的 `RAG_MVP_OPENAI_API_KEY_FILE` 读取。不要把宿主机路径误写成容器内路径。

## 校验当前配置

不打印任何 Secret 的安全检查：

```powershell
uv run python -c "from rag_mvp.config.settings import Settings; s=Settings(); print(s.safe_dump()); print('configuration_id=', s.configuration_identity); print('provider_errors=', s.provider_readiness_errors())"
```

运行配置回归测试：

```powershell
uv run pytest tests/unit/config/test_settings.py -q
```
