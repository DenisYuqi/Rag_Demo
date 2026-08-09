# Evidence Assessor 问题分析与解决方案报告

生成日期：2026-08-09  
范围：BGE Local 检索配置、Evidence Assessor、Refusal Policy、Ragas 评估结果  
状态：Issue 1 已验证解决；Issue 2 已形成方案、尚未实现

## 1. 执行摘要

本轮分析发现两个相互关联但性质不同的问题：

1. Evidence Assessor 使用整段 child chunk 做支持和冲突判断，导致相关事实被长文本稀释，并被同一 chunk 中无关的否定句污染，最终错误拒答四条本应回答的用例。
2. Assessor 将“知识库完全没有相关主题”和“存在相关候选但证据强度不足”都压缩为无支持证据，导致 `out-of-scope` 被错误归类为 `low-confidence`。

Issue 1 已通过断言级评估修复并完成全量 BGE/Ragas 验证。Issue 2 不影响是否拒答，但影响拒答原因和用户指导的准确性，建议通过可审计的规则策略层解决。

## 2. 评估证据

| 指标 | 修复前 `eval_f0cb9e8...` | 修复后 `eval_d3d7649...` |
| --- | ---: | ---: |
| Quality Gate | failed | passed |
| Faithfulness | 14/14，100% | 18/18，100% |
| Context Precision | 约 100% | 17/18，94.44% |
| Answer Compliance | 14/18，77.78%，failed | 18/18，100%，passed |
| Style | 22/24，91.67% | 24/24，100% |
| Refusal Appropriateness | 18/24，75%，failed | 22/24，91.67%，passed |
| 执行失败用例 | 0 | 0 |

修复后的完整评估共执行 24 条用例，全部成功完成，Advanced Quality Gate 的五项指标全部通过。

## 3. Issue 1：整段 chunk 评估导致误拒答和误冲突

### 3.1 现象

修复前，以下四条可回答用例被错误拒答：

| Case | 类型 | 主要表现 |
| --- | --- | --- |
| `accept-zh-002` | 跨语言技术规范 | 正确技术事实已召回，但被错误冲突判断拦截 |
| `accept-zh-006` | 技术规范、多事实 | 正确证据存在，但无关否定句影响冲突判断 |
| `accept-en-006` | 英问中证据、架构 | 目标 chunk 为 rerank Top-1，但整段 embedding 支持不足 |
| `accept-en-007` | 技术规范 | 正确 primary evidence 被无关文本污染后判为冲突 |

这四条误拒答直接造成 Answer Compliance 从 18/18 降为 14/18，同时也造成四条 `answer_expected_but_refusal_emitted`。

### 3.2 根因

原实现对整个 child chunk 生成一个 embedding，并使用整段文本执行否定和冲突判断，产生三个问题：

1. **语义稀释**：目标事实只占长 chunk 的一小部分，尤其在英问中答的跨语言场景中，相似度被其他段落内容稀释。
2. **无关否定污染**：例如技术文档中与当前问题无关的 `must not expose local filesystem paths`，会让整个 chunk 被视为负向证据。
3. **问题指令误分解**：`请用中文回答` 等响应指令可能被拆成需要证据支持的额外事实。

### 3.3 已实施解决方案

- 将候选 chunk 按中英文句号、问号、感叹号、分号和段落边界拆成有限数量的断言。
- 每个问题事实与候选的最佳匹配断言计算语义支持度，不再使用整段 chunk 的单一相似度。
- 冲突判断只比较与当前事实最佳匹配的断言，避免无关否定句污染整段证据。
- 从问题事实分解中排除响应语言指令。
- 保留 rerank Top-K 边界，并继续使用 current/withdrawn/authority 元数据进行权威性消歧。
- 对“用户要求无条件选择撤回草案”的场景保留显式冲突规则，防止断言切分后弱化安全拒答。
- 将 assessor identity 升级为 `semantic-authority-assertion-rerank-v3`，防止不同策略的评估证据混用。

### 3.4 验证结果

- 四条误拒答用例全部恢复为 answerable。
- 24/24 条 BGE acceptance 的回答/拒答决策与数据集预期一致。
- Answer Compliance 从 77.78% 提升到 100%。
- Quality Gates 从 failed 提升到 5/5 passed。
- QA、并发和评估计划相关测试共 159 项通过。

### 3.5 状态

**已验证解决，代码当前保持未提交。**

## 4. Issue 2：`out-of-scope` 被归类为 `low-confidence`

### 4.1 现象

修复后的评估仍有两条 Refusal Appropriateness 扣分：

| Case | 期望结果 | 实际结果 | 是否正确拒答 |
| --- | --- | --- | --- |
| `accept-zh-009` | `out-of-scope` | `low-confidence` | 是 |
| `accept-en-009` | `out-of-scope` | `low-confidence` | 是 |

系统在两条用例中都正确选择了拒答，但使用了错误的原因和 `low-confidence` 指导模板。因此 Refusal Appropriateness 为 22/24，而不是 24/24。

### 4.2 根因

当前 assessor 在没有候选达到支持阈值时输出 `support_score=0` 和空 supporting chunk 集合。这个结果丢失了下列中间信息：

- 未达阈值候选的最佳语义相关度。
- 查询实体、主题和精确标识符在 Top-K 中的覆盖情况。
- 候选是“主题相关但事实不足”，还是“主题完全无关”。
- 是否存在可解释为适用范围不明、版本不明或事实槽位缺失的证据。

下游 `RefusalPolicy` 只能看到“没有支持证据”，因而统一返回 `low-confidence`，无法恢复 `out-of-scope`。

### 4.3 建议解决方案

保留 BGE 负责跨语言和语义相关性，在 assessor 与拒答策略之间增加类型化、版本化的 `EvidencePolicyEngine`。规则层不替换 BGE，也不通过任意 bonus 修改 embedding 分数，而是分别保留并判断：

- `semantic_relevance`
- `fact_support`
- `authority_status`
- `lifecycle_status`
- `conflict_status`
- `scope_status`
- 查询实体、标识符、数值和单位覆盖

建议拒答决策表：

| 条件 | 决策 |
| --- | --- |
| 无合格支持，Top-K 中也无查询实体、主题或结构化槽位覆盖 | `out-of-scope` |
| 存在主题相关候选，但事实支持未达到阈值 | `low-confidence` |
| 同一事实槽位、单位和适用范围出现无法消解的不同值 | `conflicting-evidence` |
| current authoritative evidence 可消解 withdrawn/draft | 使用 current evidence，不产生冲突拒答 |
| 用户要求无条件采用 withdrawn/draft | `conflicting-evidence` |

规则引擎应输出稳定的规则 ID、原因代码和安全诊断，例如：

```text
decision=out-of-scope
rule_id=scope.no-topic-or-entity-coverage.v1
semantic_support=false
topic_coverage=false
structured_anchor_coverage=false
```

### 4.4 实施约束

- 不得在生产规则中读取 evaluation dataset 的 `authoritative_evidence_ids`、`approved_propositions` 或 `support_anchor_groups`，避免评估标签泄漏。
- authority、lifecycle、生效日期和适用范围必须来自通用摄取元数据或受控来源注册表。
- 精确代码或数值命中必须同时满足主题、属性或关系约束，不能单独作为事实支持。
- 数值冲突必须先确认主体、属性、单位、时间和适用范围相同。
- 规则条件必须使用固定枚举和受控操作符，禁止任意表达式和 `eval`。
- 规则版本变化必须进入 assessor/policy identity 和评估 provenance。

### 4.5 推荐实施步骤

1. 在不改变线上决策的前提下，保留未达支持阈值的最佳相关性和候选特征。
2. 抽取当前 authority、withdrawn、否定和冲突规则，赋予稳定规则 ID。
3. 首先实现 `out-of-scope` 与 `low-confidence` 分类。
4. 以 shadow mode 记录旧策略和新规则策略的差异，不影响线上回答。
5. 在双语、跨语言、OCR、权威/撤回、数值冲突和越界用例上校准。
6. 达到验收条件后再启用规则策略作为正式决策路径。

### 4.6 验收标准

- `accept-zh-009` 和 `accept-en-009` 返回 `out-of-scope`。
- Refusal Appropriateness 达到 24/24。
- 现有 24 条 BGE acceptance 的回答/拒答结果无回归。
- Answer Compliance 保持 18/18。
- Faithfulness、Context Precision 和 Style 不下降到各自门槛以下。
- 每个规则决策均可追溯到规则版本、命中规则 ID 和安全的输入特征摘要。

### 4.7 状态

**待实现。建议作为下一项 Evidence Assessor 改进进入 shadow-mode 验证。**

## 5. 结论

两个问题需要分开处理：

- Issue 1 是证据支持粒度错误，已通过断言级语义评估修复，并由完整 BGE/Ragas run 验证。
- Issue 2 是拒答原因所需的信息在 assessor 输出阶段丢失，适合通过显式规则策略层解决。

因此推荐的长期架构不是“纯规则 assessor”，而是：**BGE 负责语义召回和断言相关性，规则策略层负责证据资格、权威性、冲突、适用范围和拒答原因。**
