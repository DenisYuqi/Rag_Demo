# 评测资产说明

本目录保存 RAG MVP 的评测输入、验收配置、运行结果和历史发布证据。这里的大部分 JSON、JSONL 和 HTML 文件是机器输入或生成制品，不是需要逐份阅读的项目文档。

## 快速入口

- [Release v2 验收说明](../docs/release-v2-acceptance.md)：当前验收流程和命令。
- [当前 Release v2 交付包](../deliverables/release-v2-bge-20260809-r3/)：当前发布报告及证据。
- [检索策略对比报告](../deliverables/release-v2-bge-20260809-r3/retrieval-comparison.md)：vector-only、hybrid 与 hybrid+rerank 的量化结果和结论。
- [问题诊断数据](two-issue-diagnosis.json)：两项问题的权威机器可读记录。
- [问题诊断报告](two-issue-diagnosis-report.md)：上述问题记录的人类可读说明。
- [Phase 12 历史发布](releases/phase12_20260807t030340z-954bb3e2/README.md)：已封存的旧版发布证据。

## 目录说明

| 路径 | 内容 | 维护方式 |
| --- | --- | --- |
| `acceptance/` | 候选配置模板和最终选定配置 | 人工维护；配置变化时新增或更新明确版本 |
| `datasets/` | 评测案例、语料、manifest 和源文件 | 版本化维护；已用于发布的数据集保持不变 |
| `logging/` | 隐私安全日志样例和结构化字段字典 | 人工维护，并与日志契约同步 |
| `performance/` | 性能与负载验收场景 | 人工维护，并与验收标准同步 |
| `pricing/` | 带日期和版本的模型价格目录 | 价格来源变化时新增版本，不覆盖历史版本 |
| `privacy/` | 仅含合成数据的隐私扫描样例 | 人工维护；禁止加入真实个人信息或密钥 |
| `results/` | 评测运行产生的原始报告、诊断和逐案例证据 | 生成制品；保留被发布或问题诊断引用的运行 |
| `releases/` | 已封存的历史发布包及 manifest | 不修改、不重命名；新发布使用新目录 |

## 权威来源

- `two-issue-diagnosis.json` 是问题诊断的权威数据；Markdown 报告仅用于阅读。
- 数据集和发布目录中的 `manifest.json` 或 `release-manifest.json` 记录版本、文件身份和完整性信息。
- `report.html`、`report.json`、`summary.json`、`diagnostics.json` 等结果文件由评测流程生成，不应手工修改。
- 当前交付包位于仓库根目录的 `deliverables/`；`releases/` 下的内容仅代表对应时间点的历史状态。

## 命名约定

- 人工维护的新文件优先使用小写 kebab-case，例如 `two-issue-diagnosis.json`。
- 数据集使用用途和版本组合命名，例如 `mvp-v1`、`acceptance-v2`。
- 定价文件包含提供方、用途、日期和可选版本号。
- 运行目录中的下划线、时间戳和运行 ID 是制品身份的一部分，不做格式化重命名。
- `.gitattributes`、`README.md` 等标准控制或入口文件不受小写 kebab-case 约束。
- 已封存发布即使包含旧命名也保持原样，以免破坏 manifest 哈希和历史复现。

## 清理原则

清理 `results/` 前，先确认目标运行没有被问题诊断、发布 manifest、测试夹具或文档引用。被引用的运行和所有封存发布必须保留；无引用的中间运行可以移入外部归档或在确认无复现需求后删除。
