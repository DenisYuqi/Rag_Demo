# To Improve

## Evidence Assessor：引入可审计的规则策略层 （Drools）

状态：提议，尚未实现。

详细问题分析和修复验证见 [Evidence Assessor 问题分析与解决方案报告](./evidence-assessor-issues-report.md)。

### 背景与证据

当前 Evidence Assessor 已经是语义判断与内嵌规则的混合实现，包括响应指令过滤、断言切分、current/withdrawn/authority 识别、冲突判断、相似度阈值和最终拒答策略。规则目前分散在代码和正则表达式中，不容易独立版本化、解释和校准。

BGE/Ragas 评估 `eval_d3d76490c9ef444c8ad00d3cb829ffe4` 的 Quality Gates 为 5/5 passed，但 Refusal Appropriateness 为 22/24。两个扣分用例均正确拒答，但拒答原因分类不准确：

| Case | 期望原因 | 实际原因 |
| --- | --- | --- |
| `accept-zh-009` | `out-of-scope` | `low-confidence` |
| `accept-en-009` | `out-of-scope` | `low-confidence` |

根因是 assessor 将“知识库中没有相关主题”和“存在相关证据但不足以支持事实”都压缩成无支持证据，下游拒答策略无法恢复二者的区别。

### 建议方向

保留 BGE 负责跨语言、同义改写和 OCR 噪声下的语义相关性，在其后增加一个类型化、版本化的 `EvidencePolicyEngine`。规则层负责确定性的资格、权威性、冲突和拒答原因判断，不用纯规则替换语义模型。

建议处理流程：

```text
Query
  -> 事实与响应指令分离
  -> BGE 断言级相关性
  -> EvidencePolicyEngine
       - source lifecycle
       - authority/current/withdrawn
       - identifier/value/unit
       - negation/modality
       - applicability/date/scope
       - out-of-scope vs low-confidence
  -> RefusalPolicy
```

规则优先级：

1. 安全、候选注册表和索引修订有效性。
2. current/withdrawn/authority 硬约束。
3. 同一事实槽位、相同适用范围下的结构化冲突。
4. 精确标识符、数值、币种、时间和单位支持。
5. BGE 语义支持。
6. `out-of-scope`、`low-confidence`、`conflicting-evidence` 等拒答原因分类。

### 决策模型

不要通过任意 bonus 将所有判断混入一个相似度分数。至少分别保留：

- `semantic_relevance`
- `fact_support`
- `authority_status`
- `lifecycle_status`
- `conflict_status`
- `scope_status`

规则引擎输出应包含稳定的决策、原因代码、命中规则 ID 和必要的安全诊断，而不是修改原始 embedding 分数。

拒答原因可按以下原则区分：

- 没有合格支持，Top-K 中也没有查询实体、主题或结构化槽位覆盖：`out-of-scope`。
- 存在主题相关候选，但事实支持未达到要求：`low-confidence`。
- 相同事实槽位和适用范围出现无法由 authority/lifecycle 消解的不同值：`conflicting-evidence`。

### 适合规则化的判断

- 响应语言、格式或安全指令不作为待检索事实。
- withdrawn/draft 不得作为现行规范依据，除非问题明确询问历史或草案。
- current authoritative evidence 优先于已撤回候选。
- 精确代码、Header、前缀、数值和单位的联合匹配。
- `must`/`must not`、允许/禁止等同一命题的模态冲突。
- 生效日期、地区、部门和对象范围的适用性判断。
- 拒答原因和可审计解释代码的选择。

BGE 仍负责中英文释义、同义表达、隐式语义关系和 OCR 噪声文本，避免规则数量失控。

### 约束与风险

- 生产规则不得读取 evaluation dataset 的 `authoritative_evidence_ids`、`approved_propositions` 或 `support_anchor_groups`，避免评估标签泄漏。
- authority、lifecycle、生效日期和适用范围应来自摄取阶段的通用文档元数据或受控来源注册表。
- 精确标识符命中不能单独视为事实支持，还必须具有主题或关系约束，避免无关文档中的相同代码误命中。
- 数值冲突必须先确认主体、属性、单位和适用范围相同，避免将不同年份、地区或员工类别误判为冲突。
- 规则定义必须使用固定枚举和受控条件，不允许任意表达式或 `eval`。
- 每次规则或阈值调整都必须改变 assessor/policy identity，确保评估和缓存不会混用不同策略。

### 实施顺序

1. 抽取当前内嵌规则为类型化决策和稳定规则 ID，保持现有行为不变。
2. 保留未达支持阈值的最佳相关性、实体覆盖和候选诊断，避免过早压缩为零。
3. 首先实现 `out-of-scope` 与 `low-confidence` 分类，覆盖上述两个失败用例。
4. 增加精确标识符、数值/单位和同槽位冲突规则。
5. 以 shadow mode 同时记录旧决策和规则策略决策，不影响线上回答。
6. 在双语、跨语言、OCR、权威/撤回、数值冲突和越界数据集上完成校准后再启用。

### 验收标准

- `accept-zh-009` 和 `accept-en-009` 返回 `out-of-scope`，而不是 `low-confidence`。
- 现有 24 条 BGE acceptance 的回答/拒答结果无回归。
- Answer Compliance、Faithfulness、Context Precision 和 Style 不下降。
- Refusal Appropriateness 达到 24/24，且每条规则决策可以追溯到规则版本和命中规则 ID。
- 未知、缺失或矛盾元数据保持 fail-closed，不通过默认值伪造 authority 或适用范围。

## RAG 改进路线审查（2026-08-09）

状态：代码审查完成，以下事项尚未实现。

### 审查范围与结论边界

主要检查了文档摄取、分块、索引输入、混合检索、查询理解、重排、上下文组装、FAQ、知识图谱、多模态解析和可观测性。以下结论是当前实现的代码审查结果，所有改进项都必须先经过项目自身的离线评估和候选对比。

总体判断：Rag_Demo 的检索主干已经较为完整。它已经具有父子分块、Dense + BM25、加权 RRF、可选重排、BGE 本地 profile、查询历史扩展、严格引用、原子索引修订、检索缓存、阶段预算和较完整的评估体系。当前最值得改进的不是重写主干，而是修正上下文窗口、提高结构信息利用率、减少重复上下文，并在低召回时做受控扩展。

### 能力盘点

| 能力 | 当前状态 | 可采用的改进方向 | 判断 |
| --- | --- | --- | --- |
| 父子分块 | 已实现；子块检索、父块生成 | 已实现；并可补充父块、相邻块和关系块 | 主干无需重做，但父块截断方式需要立即修正 |
| 混合检索 | Dense + BM25 + weighted RRF；单路失败可降级 | Vector + keyword + weighted RRF | 已基本对齐 |
| 重排 | OpenAI-compatible 或 BGE cross-encoder | 多 provider 重排、阈值降级、综合分数和 MMR | 应补候选多样性，不必复制 provider 数量 |
| 查询处理 | 基于规则识别追问，只拼接历史用户问题 | LLM 改写 + 意图分类 + 图片理解；低召回时本地 query expansion | 应增加受控的低召回扩展；LLM 改写保持可选 |
| 分块策略 | 保留 Markdown section path/PDF 页边界，块内固定 token 切分 | 文档画像驱动 heading/heuristic/legacy 多级策略，失败自动回退并提供预览诊断 | Rag_Demo 可增量引入结构感知与诊断，不应直接替换现有稳定分块 |
| 索引文本 | Embedding/BM25 主要使用 `chunk.text`；标题和 section path 作为元数据 | 标题和 heading breadcrumb 会加入索引文本，但不污染展示正文 | 应增加版本化的 `index_text` |
| 上下文去重 | 同一 parent 只保留最高排名 child | 精确/近似重复过滤，重排后 MMR；还能补相邻/关系块 | 应在上下文预算前增加近重复抑制和多样化 |
| 检索范围 | 每个 retrieval profile 基本是一个全局 corpus | 可按 KB、文档、tag 和 tenant 限定范围，并支持多 KB fan-out | 首先补 `source_id`/文档级过滤；多租户不是当前 MVP 必需项 |
| 文档与多模态 | PDF、Markdown、TXT；PDF 可选择性 OCR | DOC/DOCX、表格、图片 OCR/Caption、多种解析引擎等 | 按真实语料需求逐类增加，不建议一次性复制 |
| FAQ / Graph | 没有专用 FAQ 索引和知识图谱检索 | FAQ 专用元数据/负问题过滤；实体图谱与 chunk 检索并行 | 都是条件型能力，优先级低于基础检索正确性 |
| 评估与发布证据 | 数据集版本、质量门、候选对比、不可变报告较完整 | 提供在线 evaluation 与 Langfuse 链路 | Rag_Demo 现有体系是优势，应作为所有改造的准入门 |

### P0：修正父块上下文截断，确保命中事实进入生成上下文

#### 现状与风险

`src/rag_mvp/qa/context.py` 先把命中的 child 展开为 parent，再按 `maximum_tokens_per_chunk` 截断。但当前截断固定取 `parent.text` 的前缀：

```python
text = parent.text if not truncated else parent.text[: spans[token_count][0]]
```

默认父块目标为 1536 token，而单个生成上下文块默认最多 500 token。若命中的 child 位于父块中后部，Evidence Assessor 评估的是正确 child，但生成模型收到的可能只是父块开头，命中事实会被截掉。这会造成“检索与 assessor 正确、生成仍拒答或漏答”的隐蔽问题。现有单元测试只验证前缀截断，没有覆盖父块尾部命中。

成熟的检索后上下文组装通常会补充 parent/nearby/related chunk；Rag_Demo 更适合保留现有父子模型，但改为 **hit-anchored window**。

#### 建议实现

1. 摄取时为 child 持久化相对 parent 的 token 或字符区间，PDF 也必须有相对区间，不能只依赖页码。
2. `ContextBuilder` 以命中 child 为中心分配左右窗口，窗口必须完整覆盖 child；剩余预算再向前后扩展。
3. 同一 parent 有多个命中 child 时，先合并相交/相邻区间；相距很远时允许生成两个有界窗口，而不是只保留最高排名 child 后丢弃另一处事实。
4. 引用身份仍使用命中的 child ID；parent/window 只负责提供生成上下文。
5. 更新 chunk/schema、context selection 和 cache identity 版本，并完整重建索引，禁止新旧 offset 语义混用。

#### 验收标准

- 构造 1536-token parent，把唯一答案放在 token 900 之后；在 500-token 单块预算下，生成上下文必须包含命中 child 的完整文本。
- parent 头部、中部、尾部命中，以及中英文/OCR 文本都通过。
- 同一 parent 的多个远距离命中不会被静默丢弃，并且总 token 预算仍严格生效。
- 引用继续指向 active revision 中的 child，现有 citation/grounding 校验无回归。
- 增加 `context_hit_coverage` 诊断，验收数据集必须为 100%。

### P0：为检索引入版本化的标题与章节上下文

#### 现状与差距

Rag_Demo 已在 `ChunkLocator.section_path` 保存 Markdown 章节路径，也保存 `display_title`，但 `EmbeddingStage` 实际发送的是 `chunk.text`，BM25 同样以正文为主。对于“安装”“限制”“例外”等多个文档都会重复出现的通用段落，正文缺少标题/章节语义，召回容易混淆。

建议将 document title 和 heading breadcrumb 加入独立索引文本，同时与原始内容分开保存，避免引用和展示正文被前缀污染。

#### 建议实现

- 增加独立、确定性的 `index_text`：建议格式为 `title + section breadcrumb + child text`；`Chunk.text` 保持原始可引用正文。
- Dense 和 BM25 必须使用同一版本的结构增强输入；rerank 可使用带标题/章节的 enriched passage，但 Evidence Assessor 与引用仍读取原正文及明确元数据。
- 单独保存 `embedding_input_digest`/`index_text_digest`。不能继续只用正文 `content_digest` 命中 embedding cache，否则不同标题或章节下的相同正文会错误复用向量。
- 将 enrichment 版本加入 derivation config、index revision、retrieval cache identity 和评估报告；改变格式必须重建索引。
- 对标题/面包屑设置长度上限并去重，防止超长标题挤占正文或成为关键词噪声。

#### 验收标准

- 两个文档含相同正文、不同标题/章节时，查询指定章节能稳定召回正确来源。
- 原文 digest、引用文本和 source locator 不因索引前缀改变。
- embedding cache 不会跨不同 `index_text` 误命中。
- 在现有双语 acceptance 上无回归，并新增 repeated-heading、generic-section 和跨文档同文案例。

### P0：重排后增加近重复抑制与上下文多样化

#### 现状与差距

当前 `ContextBuilder` 只按 `parent_chunk_id` 去重。不同 parent、重复上传或多个版本相似文档仍可能占满 5 个上下文名额。建议在检索合并中增加内容签名、部分重叠过滤，并在重排后用 MMR 抑制高冗余候选，从而提高有限 context budget 的利用率。

#### 建议实现

1. 在 rerank 之后、parent expansion 之前增加确定性的 exact digest/content-signature 去重。
2. 对剩余候选做有界近重复判断；首版可采用规范化 token overlap/Jaccard，避免立刻引入新的向量调用。
3. 再做 source-aware MMR 或配额选择，优先保留高相关且能覆盖不同来源/章节的证据；精确标识符和权威来源规则优先于多样性惩罚。
4. 记录 `pre_selection_ids`、`post_selection_ids`、drop reason、每来源数量和平均冗余度。不要复用当前要求 pre/post 集合完全相同的 rerank 诊断字段。

#### 验收标准

- 重复上传、重叠 child 和高度相似 parent 不会重复消耗上下文名额。
- 需要两份不同来源才能回答的测试中，两侧证据都被保留。
- current/authoritative 证据不会仅因与 withdrawn/draft 文本相似而被去除。
- 多样化步骤完全确定、受 deadline 约束，并具有独立版本和可审计 drop reason。

### P1：增加自适应分块与只读预览诊断

Rag_Demo 当前已经保留 Markdown heading block 和 PDF page block，不能简单描述为“完全不感知结构”；差距在于 block 内仍固定按 token 窗口切分，没有基于文档画像选择 heading/heuristic/fallback，也没有分块结果质量诊断。

建议分阶段实现：

1. 先增加只读 chunk preview：显示 parent/child 数量、长度分布、overlap、section path、页码、极小/极大块、预估 embedding token 和策略版本，不写数据库、不调用模型。
2. 再增加文档 profiler，识别 Markdown 标题、编号章节、中英文章标题、分页符、代码块、表格和异常 OCR 行。
3. 策略链从最保守的 `heading -> current-token-splitter` 开始；只有验证器确认覆盖完整、无大量微小块、无超限块时才接受新策略，否则回退当前实现。
4. 分块策略必须进入 derivation digest 和 index revision；配置变更只能通过新修订重建。

验收数据应包含 Markdown 文档、扫描 PDF、长叙述文本、表格、代码块、中英混合与 OCR 碎行。除端到端指标外，还要检查字符覆盖、顺序、重复率、parent-child 包含关系和 chunk size 分布。

### P1：增加受控的低召回 query expansion

Rag_Demo 的 `QueryRewriter` 只在规则识别为追问时拼接有限数量的历史**用户**问题，安全边界清晰，但它不会生成同义词、缩写展开、关键词变体，也没有意图分类。建议在初次召回不足时生成少量本地变体并并发检索；LLM/VLM 自包含改写和意图判断保持为可选能力。

建议先实现低风险版本：

- 仅在初检结果数、top score、Evidence Assessor 主题覆盖等信号显示低召回时触发。
- 使用确定性变体：去问句词、保留精确标识符、缩写/全称词典、引号短语、中文分词关键词；每个请求限制变体数、候选数、并发和总 deadline。
- 每个变体作为独立检索 channel，统一用版本化 RRF 合并；缓存 identity 必须包含规范化变体集合和 expansion policy identity。
- 记录每个变体新增的唯一相关命中，若长期没有增益则自动停用该规则。

LLM rewrite/intent 可以作为后续 feature flag，但不要采用未经约束的完整历史策略。继续禁止把未验证的 assistant 输出拼入检索 query；模型输出必须结构化校验、经过 injection 检查，并有失败回退、成本上限和独立评估。

### P1：支持请求级检索范围与元数据过滤

当前一个 Rag_Demo retrieval profile 对应一个整体 corpus，问答 API 没有一等公民的文档范围。随着文档增加，用户很难表达“只查这几份文件”，也无法从检索层保证范围隔离。建议逐步支持 KB、文档、tag 等 scope。

建议按 MVP 范围实施：

1. 首先支持 `source_ids` allowlist 和可选 `document_kinds`，以后再增加受控 tags；暂不引入完整 tenant/RBAC。
2. Dense、BM25、rerank、Evidence Assessor 和 citation registry 都必须验证相同 scope，不能只在最终结果中过滤。
3. scope 纳入 canonical request、retrieval cache key、诊断和评估证据。
4. 对不存在、已删除或不属于 active revision 的 source fail closed。

验收要求 scope leakage 为 0，并覆盖 Dense/BM25 单路降级、缓存命中、重排和旧 revision 场景。

### P2：按语料需求扩展解析、多模态、FAQ 和图谱

以下扩展能力不应排在检索正确性之前：

- **Office/HTML/表格**：按真实上传量选择 DOCX、XLSX、PPTX、HTML/CSV；表格应保留表头、行列关系和单元格 locator，不能只转成无结构纯文本。
- **图片与多模态**：将 image OCR、caption 和原图引用建成独立 chunk 类型，保留 page/position 关系；先建立图片问答与引用评估集，再启用 VLM。
- **FAQ 专用索引**：若 FAQ 是主要语料，再增加 question/answer/aliases/negative_questions 模型和独立分数校准，避免用任意 bonus 混入通用相似度。
- **Knowledge Graph**：仅在实体关系或 multi-hop 数据集证明 chunk RAG 明显不足时引入；图谱命中最终仍要回到可引用 source chunk，不能把未溯源的图节点直接当事实。
- **可插拔向量存储与多 worker**：当单进程 SQLite/Chroma 的容量或吞吐数据确实不达标时，再抽象外部 vector store、任务队列和分布式锁；不要仅为功能对齐而增加运维复杂度。

### 不建议重复建设的部分

- 不重写现有 weighted RRF、BGE reranker 和 parent-child 主干；先修复窗口选择并做消融实验。
- 不因存在更多模型/向量库选项，就扩大当前 MVP 的 provider surface；这会显著增加组合测试和故障模式。
- 不为更丰富的链路界面替换现有 OpenTelemetry/Prometheus/成本账本。可补充 query-expansion 命中、context hit coverage、冗余度和 scope 分布等**不含原始敏感文本**的诊断。
- 不默认引入 web search、Agent tool loop、tenant/RBAC 或完整 Wiki 系统；它们不是当前文档 RAG 正确性的前置条件。
- 不直接采用重排阈值自动下调。Rag_Demo 当前是证据优先、低置信拒答语义，阈值降级必须证明不会降低拒答适当性。

### 推荐实施顺序

1. **正确性阶段**：hit-anchored parent window、相关 schema/identity 版本和尾部命中回归集。
2. **上下文质量阶段**：结构增强 `index_text`、近重复抑制、MMR/source diversity 和诊断字段。
3. **召回阶段**：chunk preview、自适应策略、低召回 query expansion、请求级 source scope。
4. **条件能力阶段**：Office/表格、多模态、FAQ、GraphRAG 和外部向量库，逐项用数据决定是否进入生产。

每一阶段都应运行现有 BGE/OpenAI profile 的 acceptance、Ragas/legacy scorer、性能证据和候选对比。至少新增以下固定回归集合：父块尾部命中、跨文档重复段落、同正文不同标题、缩写/全称、低召回扩展、scope 越界、current-vs-withdrawn 近重复，以及表格/图片（启用对应能力后）。
