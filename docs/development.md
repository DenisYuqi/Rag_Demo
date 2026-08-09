# 开发指南

本文面向需要修改、测试或扩展 RAG Assistant MVP 的开发者。项目要求 Python 3.12，并使用 uv 和已提交的 `uv.lock` 管理可复现环境。

## 初始化开发环境

```powershell
git clone <repository-url>
Set-Location Rag_Demo
uv sync
uv lock --check
```

`uv sync` 默认安装 `dev` dependency group，包括 pytest、Ruff、mypy、pytest-cov 和 respx。不要在 `.venv` 中手工执行未记录的 `pip install`；新增或修改依赖应更新 `pyproject.toml`，随后提交新的 `uv.lock`。

本地处理扫描 PDF 还需要 Tesseract。确认安装和语言包：

```powershell
tesseract --version
tesseract --list-langs
```

输出至少应包含 `eng` 和 `chi_sim`。不开发 OCR 功能时可以设置 `RAG_MVP_OCR_ENABLED=false`，但相应扫描 PDF 无法建立索引。

## 本地配置与启动

```powershell
Copy-Item .env.example .env
```

启动前在 `.env.local` 中写入：

```dotenv
RAG_MVP_OPENAI_API_KEY=replace-with-your-key
```

不要将真实密钥放进命令历史或提交到 Git。然后启动：

```powershell
uv run rag-mvp
```

应用入口是 `rag_mvp.__main__:main`，以单个 Uvicorn worker 运行 `rag_mvp.api.app:create_executable_app`。不要通过 `--workers` 启动多个共享本地数据目录的进程。

配置变更需要重启进程。常见本地地址：

- 工作台：<http://127.0.0.1:8000/workbench>
- OpenAPI：<http://127.0.0.1:8000/docs>
- 就绪状态：<http://127.0.0.1:8000/readyz>

## 代码组织与依赖方向

```text
api/ui  -> application services -> domain contracts
                                  -> providers
                                  -> storage repositories
                                  -> observability/safety
```

建议保持以下边界：

- `domain` 放不可变业务模型和枚举，不依赖 FastAPI 或 Gradio。
- `providers` 通过协议和路由封装外部模型，不让 SDK 类型进入领域层。
- `storage` 负责 SQLite、文件布局和制品持久化，外部输入必须经过路径校验。
- `ingestion`、`retrieval`、`qa`、`evaluation` 编排业务阶段，不直接构造 HTTP 响应。
- `api` 与 `ui` 是适配层，只暴露允许字段和安全错误码。
- 新的诊断字段必须经过隐私审查和回归测试，不能记录问题原文、文档正文、密钥或不受控路径。

运行时通过 `api/composition.py` 装配 OpenAI-compatible profile 和可选 BGE profile。测试通常直接注入 fake provider、临时 SQLite 和临时数据根目录，不访问真实网络。

## 测试矩阵

测试按作用域组织：

| 目录/文件 | 内容 |
| --- | --- |
| `tests/unit` | 领域逻辑、服务、存储、provider、性能和评测单元测试 |
| `tests/api` | FastAPI 契约、状态码、流式输出和组合测试 |
| `tests/integration` | 摄取、检索、问答、生命周期、发布和并发流程 |
| `tests/ui` | Gradio 组件、回调、来源预览和仪表板 |
| `tests/privacy` | 敏感信息与制品扫描回归 |
| `tests/test_openai_live.py` | 显式选择的真实 provider 付费测试 |

常用命令：

```powershell
# 全量确定性测试
uv run pytest

# 快速单元与 API 回归
uv run pytest tests/unit tests/api -q

# 指定标记
uv run pytest -m integration
uv run pytest -m api
uv run pytest -m ui
uv run pytest -m privacy

# 单个测试文件或用例
uv run pytest tests/api/test_documents.py -q
uv run pytest tests/api/test_documents.py::test_upload_list_rebuild_and_delete_contract -q

# 覆盖率
uv run pytest --cov=rag_mvp --cov-report=term-missing
```

真实外部服务测试可能产生费用，只有在明确核对当前环境、模型与密钥后才运行：

```powershell
uv run pytest -m live tests/test_openai_live.py
```

常规测试必须保持离线、确定性，并使用 fake 或受控 stub。不要让新测试因本机已有 `.env.local` 而意外访问网络；构造 `Settings` 时可使用 `_env_file=None` 并显式传入测试值。

## 静态质量检查

```powershell
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
uv lock --check
```

Ruff 目标版本是 Python 3.12，行宽 100；mypy 使用 strict 模式。若需要自动格式化：

```powershell
uv run ruff format src tests
```

格式化是机械变更，提交前仍要复查 diff，避免混入不相关文件。

## 常见开发流程

### 修改文档摄取

重点运行：

```powershell
uv run pytest tests/unit/ingestion tests/api/test_documents.py tests/integration/test_ingestion_service.py -q
```

同时检查上传格式、大小、UTF-8、PDF 加密/损坏、OCR 回退、父子块完整性、失败修订清理和活动修订原子性。

### 修改检索或问答

```powershell
uv run pytest tests/unit/retrieval tests/unit/qa tests/api/test_qa.py tests/integration/test_qa_pipeline.py -q
```

应覆盖三种检索模式、revision identity、缓存策略、阶段 deadline、证据不足拒答、提示词注入、引用一致性和脱敏输出。

### 修改 UI

```powershell
uv run pytest tests/ui -q
```

工作台包含 `Chat`、`Documents`、`Evaluation` 和只读 `README` 页签。修改回调时要保持浏览器状态不含原文敏感数据，并同时验证两个 retrieval profile 的切换行为。

### 修改 BGE 适配器

```powershell
uv run pytest tests/unit/providers/test_bge_adapters.py tests/ui/test_retrieval_profiles.py -q
```

模型加载是惰性的且可能占用数 GB。单元测试不应下载真实模型；设备和 CUDA smoke test 按[BGE CUDA 加速](bge-cuda-acceleration.md)执行。

## 评测工具

校验版本化数据集：

```powershell
uv run python -m rag_mvp.evaluation.validate_dataset evaluations/datasets/mvp-v1
```

查看运行器参数：

```powershell
uv run python -m rag_mvp.evaluation.run_evaluation --help
uv run python -m rag_mvp.evaluation.verify_report --help
uv run python -m rag_mvp.performance.run_load_test --help
uv run python -m rag_mvp.safety.scan_artifacts --help
```

评测结果和发布证据包含配置、代码修订、provider 尝试和内容摘要。不要手工编辑已生成的不可变制品；变更数据集、配置或代码后创建新的 run ID。

## 数据与索引注意事项

- 本地 `data/`、`.env`、`.env.local`、缓存和虚拟环境已被 Git 忽略。
- 不要把真实业务文档、API 密钥或包含 PII 的日志复制到测试 fixture 或评测制品。
- 修改分块、嵌入模型、维度或规范化逻辑后，用新修订完整重建索引。
- 删除测试数据前确认目标是显式的临时目录；不要对未解析的数据根目录执行递归删除。
- Compose 的应用数据卷和模型缓存卷用途不同。停止容器不会自动删除它们。

## 提交前检查

至少完成：

```powershell
uv lock --check
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src
uv run pytest
git diff --check
git status --short
```

若全量测试因明确的外部条件无法运行，在变更说明中列出已执行的精确命令、未执行范围和原因。
