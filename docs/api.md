# API 指南

服务使用 FastAPI，默认基地址为 `http://127.0.0.1:8000`。完整、与当前代码同步的请求/响应 Schema 可在 `/docs` 或 `/openapi.json` 查看；本文重点说明调用顺序和容易忽略的契约。

## 通用约定

- 业务 API 前缀：`/api/v1`
- JSON：`application/json`
- 文档上传：`multipart/form-data`
- 问答响应：`application/x-ndjson`，客户端应关闭代理缓冲并逐行读取
- 错误响应：`{"error":{"code":"safe_reason_code"}}`
- 异步摄取和评测创建成功时返回 `202 Accepted`
- `Location` 响应头指向摄取任务状态地址
- ID 是不透明值，客户端不应解析或自行构造
- 服务当前没有内置用户认证；`owner_id` 是会话归属字段，不是身份凭据

## 端点总览

### 运维

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/healthz` | 进程存活检查 |
| `GET` | `/readyz` | 依赖和组件就绪检查 |
| `GET` | `/metrics` | Prometheus 指标，不出现在 OpenAPI Schema 中 |

### 文档与索引

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/v1/documents` | 上传文档并创建摄取任务 |
| `GET` | `/api/v1/ingestion-jobs/{job_id}` | 查询摄取任务状态 |
| `GET` | `/api/v1/documents` | 查询活动索引和活动文档 |
| `DELETE` | `/api/v1/documents/{source_id}` | 删除文档并发布新索引修订 |
| `POST` | `/api/v1/index/rebuild` | 基于活动文档完整重建索引 |

### 问答

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST` | `/api/v1/qa` | 执行证据驱动问答并返回 NDJSON 事件 |

### 评测与诊断

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/v1/evaluation-datasets` | 列出已校验数据集 |
| `GET` | `/api/v1/evaluation-plans` | 列出注册的执行计划 |
| `GET` | `/api/v1/evaluations` | 列出评测运行 |
| `POST` | `/api/v1/evaluations` | 启动评测 |
| `GET` | `/api/v1/evaluations/{run_id}` | 查询评测状态和进度 |
| `GET` | `/api/v1/evaluations/{run_id}/summary` | 获取隐私安全摘要与门禁状态 |
| `GET` | `/api/v1/evaluations/{run_id}/failed-cases` | 获取白名单化的失败诊断 |
| `GET` | `/api/v1/evaluations/{run_id}/artifacts` | 获取已验证制品清单 |
| `GET` | `/api/v1/evaluations/{run_id}/artifacts/{artifact_id}` | 下载单个不可变制品 |
| `GET` | `/api/v1/reports/{run_id}.{report_format}` | 下载 JSON 或 HTML 报告 |
| `GET` | `/api/v1/diagnostics/requests/{request_id}` | 获取近期请求的隐私安全诊断 |

### 候选方案对比

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `GET` | `/api/v1/comparison-plans` | 列出受控对比计划 |
| `GET` | `/api/v1/comparisons` | 列出不可变对比运行 |
| `POST` | `/api/v1/comparisons` | 启动一个已注册对比 |
| `GET` | `/api/v1/comparisons/{comparison_id}` | 查询对比状态和进度 |
| `GET` | `/api/v1/comparisons/{comparison_id}/summary` | 获取对比证据和结论 |
| `GET` | `/api/v1/comparisons/{comparison_id}/artifacts` | 获取已验证制品清单 |
| `GET` | `/api/v1/comparisons/{comparison_id}/artifacts/{artifact_id}` | 下载单个不可变制品 |

评测和对比端点默认使用 `openai-api`。需要操作 BGE 隔离环境时追加 `?retrieval_profile=bge-local`；未知 profile 会返回不可用错误，不会回退到另一索引。文档与问答 HTTP API 不接受该参数，切换 profile 请使用工作台。

## 上传与轮询示例

上传支持 PDF、Markdown 和 UTF-8 文本，默认上限为 25 MiB。`source_key` 用于标识同一逻辑来源，`display_title` 用于安全展示；两者均可省略。

```bash
curl -i -X POST "http://127.0.0.1:8000/api/v1/documents" \
  -F "file=@README.md;type=text/markdown" \
  -F "source_key=employee-policy" \
  -F "display_title=Employee Policy"
```

响应状态为 `202`，主体示例：

```json
{
  "job_id": "job_opaque_id",
  "operation": "upload",
  "status": "queued",
  "stage": "queued",
  "source_id": null,
  "document_version": null,
  "ocr_page_count": 0,
  "chunk_count": 0,
  "active_index_revision": null,
  "stage_timings_ms": {},
  "warnings": [],
  "safe_error_code": null,
  "failed_stage": null,
  "created_at": "2026-01-01T00:00:00Z",
  "updated_at": "2026-01-01T00:00:00Z"
}
```

读取响应头中的 `Location` 并轮询，直到 `status` 为 `succeeded` 或 `failed`：

```bash
curl "http://127.0.0.1:8000/api/v1/ingestion-jobs/job_opaque_id"
```

只有成功发布后文档才会出现在活动列表中：

```bash
curl "http://127.0.0.1:8000/api/v1/documents"
```

更新同一逻辑文档时使用相同的 `source_key` 再次上传。索引发布失败时，先前活动版本和修订不会改变。

## 问答示例

首次请求不传 `session_id`，服务会创建会话。`mode` 可选值为 `dense`、`hybrid`、`hybrid-rerank`，省略时使用服务配置；`requested_language` 可选 `en` 或 `zh-CN`。

```bash
curl -N -X POST "http://127.0.0.1:8000/api/v1/qa" \
  -H "Content-Type: application/json" \
  -H "X-RAG-Cache-Policy: use" \
  -d '{
    "owner_id": "demo-user",
    "question": "员工每年有多少天年假？",
    "mode": "hybrid",
    "requested_language": "zh-CN"
  }'
```

响应每行都是一个完整 JSON 事件，`kind` 为 `answer`、`refusal` 或 `error`，且终态事件的 `terminal` 为 `true`。答案事件包含 `session_id`、内容、声明、引用和隐私安全诊断。后续对话复用返回的 `session_id`：

```bash
curl -N -X POST "http://127.0.0.1:8000/api/v1/qa" \
  -H "Content-Type: application/json" \
  -d '{
    "owner_id": "demo-user",
    "session_id": "session_0123456789abcdef0123456789abcdef",
    "question": "如果我年中入职呢？"
  }'
```

同一个 `session_id` 必须始终搭配创建它的 `owner_id`。当索引未就绪、证据不足、问题超出范围、检测到注入或依赖失败时，服务可能安全拒答或返回错误事件；客户端不应把“没有答案”当作空字符串处理。

缓存策略通过 `X-RAG-Cache-Policy` 传递：

- `use`：按服务配置使用缓存。
- `bypass`：绕过缓存，常用于可重复评测。

## 启动评测示例

先获取数据集与计划：

```bash
curl "http://127.0.0.1:8000/api/v1/evaluation-datasets"
curl "http://127.0.0.1:8000/api/v1/evaluation-plans"
```

再使用目录中返回的精确 ID 和版本启动：

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/evaluations" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "mvp-bilingual-rag",
    "dataset_version": "1.0.0",
    "plan_id": null
  }'
```

评测和对比可能调用付费模型。执行前应核对数据集规模、模型、评分后端、最大并发和定价版本；不要根据示例 ID 假设本地目录内容。

## 就绪状态排查

不要只检查 `/healthz`。以下命令能显示每个组件的安全状态原因：

```bash
curl -i "http://127.0.0.1:8000/readyz"
```

常见情况：

- `provider_credentials_missing`：没有加载 API 密钥或密钥文件。
- `index_not_ready`：尚未成功摄取文档或活动修订不可用。
- `qa_not_composed`：provider 未满足装配条件，问答服务没有创建。
- `storage_not_writable`：数据根目录不可写。
- `telemetry_otlp_endpoint_missing`：启用了 OTLP，但没有配置 traces endpoint。

这些原因码故意不包含密钥、文档内容、提示词或内部路径。更深的排查应结合结构化日志、`/metrics` 和请求诊断端点。
