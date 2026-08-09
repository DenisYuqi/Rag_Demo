# BGE 本地推理 CUDA 加速

本文记录 Windows 开发环境中 `bge-local` 检索配置的 CUDA 加速方案。该方案覆盖
`BAAI/bge-m3` 稠密向量生成和 `BAAI/bge-reranker-v2-m3` 重排；最终答案生成仍由配置的
OpenAI-compatible 模型完成，因此总问答耗时还包含远程生成和网络耗时。

## 适用范围

当前验证环境如下：

- Windows、Python 3.12、uv 项目环境；
- NVIDIA RTX 2000 Ada Generation Laptop GPU，8 GB 显存；
- NVIDIA 驱动可支持 CUDA 12.8；
- PyTorch `2.11.0+cu128`、FlagEmbedding `1.4.0`、Transformers `4.57.6`。

PyTorch CUDA wheel 自带所需的 CUDA 运行库，本机不需要为了本项目单独安装完整 CUDA
Toolkit，但仍需要可兼容 CUDA 12.8 的 NVIDIA 驱动。安装矩阵以
[PyTorch 官方说明](https://pytorch.org/get-started/previous-versions/)为准。

本文的 `cu128` 来源只对 Windows 生效。Linux 和容器部署仍使用各自的默认依赖来源；
容器 GPU 透传、NVIDIA Container Toolkit 和镜像构建不在本文范围内。

## 项目依赖配置

项目在 `pyproject.toml` 中直接约束 Windows 版 PyTorch，并通过 uv 的显式索引获取 CUDA
wheel：

```toml
[project]
dependencies = [
  # ...
  "torch==2.11.0; sys_platform == 'win32'",
  "transformers>=4.44,<5",
]

[tool.uv.sources]
torch = [
  { index = "pytorch-cu128", marker = "sys_platform == 'win32'" },
]

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

`explicit = true` 可防止其他依赖被意外地从 PyTorch 索引解析。平台 marker 则避免 Windows
CUDA wheel 影响非 Windows 环境。详细机制参见
[uv 的 PyTorch 集成文档](https://docs.astral.sh/uv/guides/integration/pytorch/)。

`transformers<5` 是兼容性要求。FlagEmbedding 1.4 的重排实现仍使用 Transformers 4.x
tokenizer 接口；单独升级到 Transformers 5.x 会导致 `XLMRobertaTokenizer has no
attribute prepare_for_model`，使 `hybrid-rerank` 失败。修改这些版本后必须同时更新并提交
`uv.lock`，不要在虚拟环境中单独执行未记录的 `pip install`。

## 安装或同步

在仓库根目录执行：

```powershell
uv sync
uv lock --check
```

CUDA 版 Torch 和运行库体积较大，第一次下载可能需要数分钟；中断后再次执行 `uv sync`
会复用已完成的缓存。安装完成后重启应用，已经运行的 Python 进程不会自动切换到新 Torch
或重新读取 `.env`。

## 应用配置

在本机 `.env` 中设置：

```dotenv
RAG_MVP_BGE_PROFILE_ENABLED=true
RAG_MVP_BGE_DEVICE=cuda:0
RAG_MVP_BGE_USE_FP16=true
RAG_MVP_BGE_EMBEDDING_BATCH_SIZE=8
RAG_MVP_BGE_RERANKING_BATCH_SIZE=8
```

`cuda:0` 指向第一块 NVIDIA GPU。FP16 可降低显存占用并提高本次验证硬件上的吞吐；若切回
CPU，应同时改为 `RAG_MVP_BGE_DEVICE=cpu` 和
`RAG_MVP_BGE_USE_FP16=false`。批量大小需要结合文档长度和显存调整，发生 CUDA OOM 时应先
降低两个 batch size。

## 验证 CUDA

执行以下命令检查 PyTorch、CUDA runtime 和实际设备：

```powershell
.\.venv\Scripts\python.exe -c "import torch; print('torch=', torch.__version__); print('cuda_runtime=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

本次验证输出为：

```text
torch= 2.11.0+cu128
cuda_runtime= 12.8
cuda_available= True
device= NVIDIA RTX 2000 Ada Generation Laptop GPU
```

还应确认应用确实读取了 GPU 配置：

```powershell
.\.venv\Scripts\python.exe -c "from rag_mvp.config.settings import Settings; s=Settings(); print(s.bge_device, s.bge_use_fp16)"
```

预期结果是 `cuda:0 True`。随后启动应用，在 UI 中选择 `bge-local`，分别执行一次 `dense`
和 `hybrid-rerank` 查询。可同时运行 `nvidia-smi`，确认 Python 进程出现并占用显存。

## 本机性能记录

以下数据采集于前述 RTX 2000 Ada 环境，batch size 为 8。它们是用于确认 GPU 路径的本地
烟雾测试，不是严格的跨硬件基准；输入长度、缓存状态和远程答案生成都会改变端到端结果。

| 阶段 | 模型加载 | 首次 batch 8 推理 | 预热后 batch 8 推理 | 约占用显存 |
| --- | ---: | ---: | ---: | ---: |
| 稠密嵌入 | 4.609 s | 0.816 s | 0.028 s | 1.09 GB |
| 重排 | 2.607 s | 1.117 s | 0.028 s | 1.09 GB |

优化前，CPU 路径中一次证据评估的 BGE 批量嵌入观测值约为 7.9 秒。GPU 路径预热后的本地
模型计算已降到几十毫秒量级。不过模型按需加载，因此进程启动后的第一次 `dense` 或
`hybrid-rerank` 请求仍可能花费数秒；`hybrid-rerank` 首次使用还需要加载第二个模型。后续
请求才代表稳定的热路径性能。

端到端耗时不能直接等同于表中的本地推理耗时。检索、SQLite/Chroma 访问、证据评估、远程
答案生成和网络延迟仍在请求关键路径上。

## 故障排查

### `torch.cuda.is_available()` 为 `False`

检查是否误装了带 `+cpu` 后缀的 Torch、NVIDIA 驱动是否正常，以及启动应用时是否使用了
项目的 `.venv`。重新执行 `uv sync` 后必须重启应用。

### Torch 模块或包元数据不完整

若旧环境曾被不同包管理器覆盖，可能出现 `torch has no attribute __version__`、缺少
`RECORD` 或 `importlib.metadata.version("torch")` 异常。优先执行：

```powershell
uv sync --reinstall-package torch --link-mode copy
```

若仍失败，应在确认没有项目进程使用该环境后重建 `.venv`，再执行 `uv sync`。不要手工混用
CPU 和 CUDA wheel。

### 第一次请求仍超过 deadline

先连续执行两次相同模式的请求，区分模型冷加载与稳定性能。确认模型已经缓存到
`RAG_MVP_BGE_MODEL_CACHE_DIR`，并根据实际冷启动时间调整启动流程或预热策略。不要仅通过
放大 deadline 掩盖 `cuda_available=False` 或模型回落到 CPU 的问题。

### CUDA out of memory

先降低 `RAG_MVP_BGE_EMBEDDING_BATCH_SIZE` 和 `RAG_MVP_BGE_RERANKING_BATCH_SIZE`，再缩短
允许的最大输入长度。还应使用 `nvidia-smi` 排查其他进程的显存占用。8 GB 显存可以运行
本次配置，但可用余量取决于两个模型是否同时驻留及其他 GPU 进程。

## 回归检查

依赖或设备配置变更后至少运行：

```powershell
uv lock --check
.\.venv\Scripts\python.exe -m pytest tests\unit\providers\test_bge_adapters.py tests\unit\config\test_settings.py -q
.\.venv\Scripts\python.exe -m pytest tests\ui\test_mount.py tests\ui\test_chat.py -q
```

本次变更对应的检查结果为 69 项测试通过。
