from __future__ import annotations

import math

import pytest

from rag_mvp.evaluation.grounding_metrics import (
    CONTEXT_PRECISION_SCORER_VERSION,
    FAITHFULNESS_SCORER_VERSION,
    ContextPrecisionScorer,
    EvidenceVerdict,
    FactSupportAssessment,
    FaithfulnessScorer,
    MetricInputError,
    MetricName,
    adjudicated_text_support,
    aggregate_metric,
)


def _fact(
    fact_id: str,
    supported: bool,
    evidence_ids: tuple[str, ...] = (),
) -> FactSupportAssessment:
    return FactSupportAssessment(
        fact_id=fact_id,
        supported=supported,
        rationale="support_verified" if supported else "support_not_verified",
        evidence_chunk_ids=evidence_ids,
    )


def test_faithfulness_uses_all_factual_units_and_retains_audit_evidence() -> None:
    result = FaithfulnessScorer().score(
        case_id="case-en-1",
        answerable=True,
        response_outcome="answer",
        factual_units=(
            _fact("fact-1", True, ("chunk-1",)),
            _fact("fact-2", False),
            _fact("fact-3", True, ("chunk-2", "chunk-3")),
        ),
    )

    assert result.metric is MetricName.FAITHFULNESS
    assert result.scorer_version == FAITHFULNESS_SCORER_VERSION
    assert result.eligible
    assert result.score == 2 / 3
    assert result.numerator == 2
    assert result.denominator == 3
    assert result.rationale == "supported_factual_units=2; factual_units=3"
    assert tuple(item.reference_id for item in result.evidence) == (
        "fact-1",
        "fact-2",
        "fact-3",
    )
    assert tuple(item.verdict for item in result.evidence) == (
        EvidenceVerdict.SUPPORTED,
        EvidenceVerdict.UNSUPPORTED,
        EvidenceVerdict.SUPPORTED,
    )
    assert result.evidence[2].evidence_references == ("chunk-2", "chunk-3")


@pytest.mark.parametrize(
    ("answerable", "outcome", "facts", "reason"),
    [
        (False, "answer", (_fact("fact-1", True, ("chunk-1",)),), "case_not_answerable"),
        (True, "refusal", (), "response_not_answer"),
        (True, "answer", (), "no_factual_units"),
    ],
)
def test_faithfulness_applies_strict_eligibility(
    answerable: bool,
    outcome: str,
    facts: tuple[FactSupportAssessment, ...],
    reason: str,
) -> None:
    result = FaithfulnessScorer().score(
        case_id="case-1",
        answerable=answerable,
        response_outcome=outcome,
        factual_units=facts,
    )

    assert not result.eligible
    assert result.score is None
    assert result.denominator is None
    assert result.evidence == ()
    assert result.rationale == reason


def test_faithfulness_rejects_unadjudicated_or_duplicate_facts() -> None:
    with pytest.raises(ValueError, match="supported_fact_requires_evidence"):
        _fact("fact-1", True)

    duplicate = _fact("fact-1", True, ("chunk-1",))
    with pytest.raises(MetricInputError, match="duplicate_factual_unit"):
        FaithfulnessScorer().score(
            case_id="case-1",
            answerable=True,
            response_outcome="answer",
            factual_units=(duplicate, duplicate),
        )


def test_context_precision_is_rank_aware_average_precision() -> None:
    result = ContextPrecisionScorer().score(
        case_id="case-zh-1",
        answerable=True,
        retrieved_evidence_ids=("chunk-a", "chunk-noise", "chunk-b"),
        authoritative_evidence_ids=("chunk-a", "chunk-b"),
    )

    expected_numerator = 1.0 + (2 / 3)
    assert result.metric is MetricName.CONTEXT_PRECISION
    assert result.scorer_version == CONTEXT_PRECISION_SCORER_VERSION
    assert result.eligible
    assert math.isclose(result.numerator or 0, expected_numerator)
    assert math.isclose(result.score or 0, expected_numerator / 2)
    assert result.denominator == 2
    assert tuple(item.rank for item in result.evidence) == (1, 2, 3)
    assert tuple(item.verdict for item in result.evidence) == (
        EvidenceVerdict.RELEVANT,
        EvidenceVerdict.IRRELEVANT,
        EvidenceVerdict.RELEVANT,
    )
    assert "average_precision_contribution=" in result.rationale


@pytest.mark.parametrize("retrieved", [(), ("chunk-noise",)])
def test_missing_authoritative_context_is_an_eligible_zero(
    retrieved: tuple[str, ...],
) -> None:
    result = ContextPrecisionScorer().score(
        case_id="case-1",
        answerable=True,
        retrieved_evidence_ids=retrieved,
        authoritative_evidence_ids=("chunk-authoritative",),
    )

    assert result.eligible
    assert result.score == 0
    assert result.denominator == 1
    assert result.evidence[-1].reference_id == "chunk-authoritative"
    assert result.evidence[-1].verdict is EvidenceVerdict.MISSING


def test_context_precision_requires_an_answerable_case_and_authoritative_mapping() -> None:
    non_answerable = ContextPrecisionScorer().score(
        case_id="case-refusal",
        answerable=False,
        retrieved_evidence_ids=(),
        authoritative_evidence_ids=(),
    )
    missing_mapping = ContextPrecisionScorer().score(
        case_id="case-answerable",
        answerable=True,
        retrieved_evidence_ids=("chunk-1",),
        authoritative_evidence_ids=(),
    )

    assert not non_answerable.eligible
    assert non_answerable.rationale == "case_not_answerable"
    assert not missing_mapping.eligible
    assert missing_mapping.rationale == "no_authoritative_evidence"

    with pytest.raises(ValueError, match="retrieved_evidence_ids_duplicate"):
        ContextPrecisionScorer().score(
            case_id="case-answerable",
            answerable=True,
            retrieved_evidence_ids=("chunk-1", "chunk-1"),
            authoritative_evidence_ids=("chunk-1",),
        )


def test_aggregate_uses_only_eligible_cases_without_rounding() -> None:
    scorer = FaithfulnessScorer()
    results = (
        scorer.score(
            case_id="case-1",
            answerable=True,
            response_outcome="answer",
            factual_units=(
                _fact("fact-1", True, ("chunk-1",)),
                _fact("fact-2", False),
                _fact("fact-3", False),
            ),
        ),
        scorer.score(
            case_id="case-2",
            answerable=True,
            response_outcome="answer",
            factual_units=(_fact("fact-1", True, ("chunk-1",)),),
        ),
        scorer.score(
            case_id="case-3",
            answerable=False,
            response_outcome="refusal",
            factual_units=(),
        ),
    )

    aggregate = aggregate_metric(
        results,
        metric=MetricName.FAITHFULNESS,
        scorer_version=scorer.version,
    )

    assert aggregate.eligible_cases == 2
    assert aggregate.total_cases == 3
    assert aggregate.score == ((1 / 3) + 1.0) / 2
    assert aggregate.value == aggregate.score


def test_aggregate_with_no_eligible_cases_keeps_value_unknown() -> None:
    result = FaithfulnessScorer().score(
        case_id="case-1",
        answerable=False,
        response_outcome="refusal",
        factual_units=(),
    )

    aggregate = aggregate_metric(
        (result,),
        metric=MetricName.FAITHFULNESS,
        scorer_version=FAITHFULNESS_SCORER_VERSION,
    )

    assert aggregate.eligible_cases == 0
    assert aggregate.score is None


@pytest.mark.parametrize(
    ("candidate", "reference"),
    [
        ("At least 12 chunks are selected.", "At most 12 chunks are selected."),
        (
            "Evaluation tasks can modify the online active index.",
            "Evaluation tasks cannot modify the online active index.",
        ),
        ("Switch the revision before validation.", "Switch the revision after validation."),
        ("评估任务可以修改在线活动索引。", "评估任务不能修改在线活动索引。"),
        ("新索引完成校验前切换活动修订。", "新索引完成校验后切换活动修订。"),
        (
            "One atomic revision deletion removes the validated index.",
            "A validated index is activated with one atomic revision switch.",
        ),
        (
            "The authoritative code may be OPS-RAG-7421.",
            "The authoritative code is OPS-RAG-7421.",
        ),
        (
            "The authoritative code might be OPS-RAG-7421.",
            "The authoritative code is OPS-RAG-7421.",
        ),
        (
            "The authoritative code could be OPS-RAG-7421.",
            "The authoritative code is OPS-RAG-7421.",
        ),
        (
            "Perhaps the authoritative code is OPS-RAG-7421.",
            "The authoritative code is OPS-RAG-7421.",
        ),
        (
            "权威代码可能是 OPS-RAG-7421。",
            "权威代码是 OPS-RAG-7421。",
        ),
        (
            "At most three paid evaluation jobs may run concurrently.",
            "At most two paid evaluation jobs may run concurrently.",
        ),
        (
            "Two atomic revision switches activate the validated index.",
            "One atomic revision switch activates the validated index.",
        ),
        ("The cap is USD 1,800.", "The cap is CNY 1,800."),
        ("The submission window is 30 hours.", "The submission window is 30 calendar days."),
        ("At most 12 jobs are selected.", "At most 12 chunks are selected."),
        (
            "The endpoint is GET /v2/knowledge/query.",
            "The endpoint is POST /v2/knowledge/query.",
        ),
    ],
)
def test_adjudicated_support_rejects_polarity_and_comparator_reversals(
    candidate: str,
    reference: str,
) -> None:
    assert not adjudicated_text_support(candidate, reference)


@pytest.mark.parametrize(
    ("candidate", "reference"),
    [
        (
            "Use OPS-RAG-7421 as the authoritative escalation code.",
            "The authoritative RAG escalation code is OPS-RAG-7421.",
        ),
        (
            "One atomic revision switch activates the validated index.",
            "A validated index is activated with one atomic revision switch.",
        ),
        ("No more than 12 chunks may be selected.", "At most 12 chunks are selected."),
        ("评估任务不得修改在线活动索引。", "评估任务不能修改在线活动索引。"),
        ("现行权威上限是人民币 1,800 元。", "当前权威上限是人民币 1,800 元。"),
        (
            "新索引在完成校验后通过一次原子操作切换活动修订。",
            "新索引完成校验后, 通过一次原子操作切换为活动修订。",
        ),
    ],
)
def test_adjudicated_support_accepts_anchor_preserving_paraphrases(
    candidate: str,
    reference: str,
) -> None:
    assert adjudicated_text_support(candidate, reference)


def test_authored_anchor_groups_allow_an_explicit_bounded_paraphrase() -> None:
    assert adjudicated_text_support(
        "索引标识必须以 idx_ 前缀开头.",
        "索引修订标识必须使用 idx_ 前缀.",
        support_anchor_groups=(
            ("索引标识", "索引修订标识"),
            ("idx_",),
            ("前缀", "以 idx_ 前缀开头"),
        ),
        approved_propositions=(
            "索引修订标识必须使用 idx_ 前缀.",
            "索引标识必须以 idx_ 前缀开头.",
        ),
    )


@pytest.mark.parametrize(
    ("candidate", "reference", "anchor_groups"),
    [
        (
            "The authoritative code is OPS-RAG-7421, and the Moon is made of cheese.",
            "The authoritative RAG escalation code is OPS-RAG-7421.",
            (("authoritative code", "escalation code"), ("OPS-RAG-7421",)),
        ),
        (
            "The Moon is made of cheese, and the authoritative code is OPS-RAG-7421.",
            "The authoritative RAG escalation code is OPS-RAG-7421.",
            (("authoritative code", "escalation code"), ("OPS-RAG-7421",)),
        ),
        (
            "The authoritative RAG escalation code is OPS-RAG-7421. "
            "The escalation happens on the Moon.",
            "The authoritative RAG escalation code is OPS-RAG-7421.",
            (("authoritative code", "escalation code"), ("OPS-RAG-7421",)),
        ),
        (
            "当前权威上限是人民币 1,800 元, 并且月球由奶酪构成.",
            "当前权威上限是人民币 1,800 元.",
            (("权威上限",), ("人民币",), ("1,800",)),
        ),
        (
            "月球由奶酪构成, 当前权威上限是人民币 1,800 元.",
            "当前权威上限是人民币 1,800 元.",
            (("权威上限",), ("人民币",), ("1,800",)),
        ),
    ],
)
def test_adjudicated_support_rejects_unsupported_compound_claims(
    candidate: str,
    reference: str,
    anchor_groups: tuple[tuple[str, ...], ...],
) -> None:
    assert not adjudicated_text_support(
        candidate,
        reference,
        support_anchor_groups=anchor_groups,
    )


@pytest.mark.parametrize(
    ("candidate", "reference"),
    [
        (
            "According to the source, the authoritative code is OPS-RAG-7421.",
            "The authoritative code is OPS-RAG-7421.",
        ),
        (
            "根据权威文档显示, 当前权威上限是人民币 1,800 元.",
            "当前权威上限是人民币 1,800 元.",
        ),
    ],
)
def test_adjudicated_support_allows_bounded_nonsubstantive_connective_prefixes(
    candidate: str,
    reference: str,
) -> None:
    assert adjudicated_text_support(candidate, reference)


@pytest.mark.parametrize(
    ("candidate", "reference", "groups", "approved"),
    [
        (
            "Escalation owns the RAG Operations Desk.",
            "The RAG Operations Desk owns escalation.",
            (("escalation",), ("owns", "responsible for"), ("RAG Operations Desk",)),
            (
                "The RAG Operations Desk owns escalation.",
                "The RAG Operations Desk is responsible for escalation.",
            ),
        ),
        (
            "RAG Operations Desk 由升级事务负责。",
            "升级事务由 RAG Operations Desk 负责。",
            (("升级事务",), ("负责",), ("RAG Operations Desk",)),
            (
                "升级事务由 RAG Operations Desk 负责。",
                "RAG Operations Desk 负责升级事务。",
            ),
        ),
        (
            "One atomic revision switch is activated with a validated index.",
            "A validated index is activated with one atomic revision switch.",
            (
                ("validated index",),
                ("activated", "activates"),
                ("one",),
                ("atomic",),
                ("revision",),
            ),
            (
                "A validated index is activated with one atomic revision switch.",
                "One atomic revision switch activates the validated index.",
            ),
        ),
        (
            "活动修订校验后通过原子操作切换为新索引。",
            "新索引校验后通过原子操作切换为活动修订。",
            (("校验后",), ("原子",), ("切换",), ("活动修订",)),
            ("新索引校验后通过原子操作切换为活动修订。",),
        ),
        (
            "提交应在 30 个自然日内报销表。",
            "报销表应在 30 个自然日内提交。",
            (("提交",), ("30",), ("自然日",)),
            ("报销表应在 30 个自然日内提交。",),
        ),
    ],
)
def test_approved_proposition_contract_rejects_directional_reversals(
    candidate: str,
    reference: str,
    groups: tuple[tuple[str, ...], ...],
    approved: tuple[str, ...],
) -> None:
    assert not adjudicated_text_support(
        candidate,
        reference,
        support_anchor_groups=groups,
        approved_propositions=approved,
    )


@pytest.mark.parametrize("separator", ["\n", ": ", " - ", " — "])
def test_approved_proposition_contract_rejects_appended_reversal_clauses(
    separator: str,
) -> None:
    reference = "The RAG Operations Desk owns escalation."
    reversal = "Escalation owns the RAG Operations Desk."
    assert not adjudicated_text_support(
        f"{reference}{separator}{reversal}",
        reference,
        support_anchor_groups=(
            ("escalation",),
            ("owns", "responsible for"),
            ("RAG Operations Desk",),
        ),
        approved_propositions=(
            reference,
            "The RAG Operations Desk is responsible for escalation.",
        ),
    )


@pytest.mark.parametrize(
    ("candidate", "reference", "groups"),
    [
        (
            "escalation code OPS-RAG-7421",
            "The authoritative RAG escalation code is OPS-RAG-7421.",
            (("escalation code",), ("OPS-RAG-7421",)),
        ),
        (
            "升级事务 负责 RAG Operations Desk",
            "升级事务由 RAG Operations Desk 负责。",
            (("升级事务",), ("负责",), ("RAG Operations Desk",)),
        ),
    ],
)
def test_approved_proposition_contract_rejects_keyword_salads(
    candidate: str,
    reference: str,
    groups: tuple[tuple[str, ...], ...],
) -> None:
    assert not adjudicated_text_support(
        candidate,
        reference,
        support_anchor_groups=groups,
        approved_propositions=(reference,),
    )


@pytest.mark.parametrize(
    ("candidate", "reference", "groups"),
    [
        (
            "One atomic revision switch activates the validated index.",
            "A validated index is activated with one atomic revision switch.",
            (
                ("validated index",),
                ("activated", "activates"),
                ("one",),
                ("atomic",),
                ("revision",),
            ),
        ),
        (
            "RAG Operations Desk 负责升级事务。",
            "升级事务由 RAG Operations Desk 负责。",
            (("升级事务",), ("负责",), ("RAG Operations Desk",)),
        ),
    ],
)
def test_approved_proposition_contract_preserves_audited_directional_paraphrases(
    candidate: str,
    reference: str,
    groups: tuple[tuple[str, ...], ...],
) -> None:
    assert adjudicated_text_support(
        candidate,
        reference,
        support_anchor_groups=groups,
        approved_propositions=(reference, candidate),
    )


@pytest.mark.parametrize(
    ("candidate", "reference", "groups", "approved"),
    [
        (
            "The authoritative RAG escalation code is `OPS-RAG-7421`.",
            "The authoritative RAG escalation code is OPS-RAG-7421.",
            (("escalation code",), ("OPS-RAG-7421",)),
            ("The authoritative RAG escalation code is OPS-RAG-7421.",),
        ),
        (
            "当前有效政策规定\N{FULLWIDTH COMMA}境内航班经济舱票价报销上限为人民币1800元。",
            "当前权威上限是人民币 1,800 元。",
            (
                ("权威上限", "境内航班经济舱票价报销上限"),
                ("人民币",),
                ("1,800",),
            ),
            (
                "当前权威上限是人民币 1,800 元。",
                "当前有效政策规定\N{FULLWIDTH COMMA}境内航班经济舱票价报销上限为人民币 1,800 元。",
            ),
        ),
        (
            "索引修订标识必须以前缀 `idx_` 开头。",
            "索引修订标识必须使用 idx_ 前缀。",
            (
                ("索引修订标识",),
                ("前缀", "以前缀 `idx_` 开头"),
                ("idx_",),
            ),
            (
                "索引修订标识必须使用 idx_ 前缀。",
                "索引修订标识必须以前缀 `idx_` 开头。",
            ),
        ),
    ],
)
def test_approved_propositions_accept_only_safe_formatting_and_audited_source_surfaces(
    candidate: str,
    reference: str,
    groups: tuple[tuple[str, ...], ...],
    approved: tuple[str, ...],
) -> None:
    assert adjudicated_text_support(
        candidate,
        reference,
        support_anchor_groups=groups,
        approved_propositions=approved,
    )


def test_ordered_anchor_frame_accepts_only_a_complete_conservative_proposition() -> None:
    reference = "At most 12 chunks are selected."
    groups = (
        ("at most", "no more than"),
        ("12",),
        ("chunks",),
        ("selected",),
    )

    assert adjudicated_text_support(
        "No more than 12 chunks are selected.",
        reference,
        support_anchor_groups=groups,
        approved_propositions=(reference,),
    )
    assert not adjudicated_text_support(
        "no more than 12 chunks selected",
        reference,
        support_anchor_groups=groups,
        approved_propositions=(reference,),
    )


@pytest.mark.parametrize(
    "candidate",
    [
        "The authoritative RAG escalation code is `OPS-RAG-7421`: the Moon owns escalation.",
        "权威的 RAG 升级代码是 `OPS-RAG-7421`\N{FULLWIDTH COMMA}责任团队是 RAG Operations Desk。",
        "当前有效政策规定\N{FULLWIDTH COMMA}境内航班经济舱票价报销上限为人民币1800元"
        " - 月球由奶酪构成。",
    ],
)
def test_formatting_and_source_surfaces_cannot_launder_appended_claims(
    candidate: str,
) -> None:
    if "1800" in candidate:
        reference = "当前权威上限是人民币 1,800 元。"
        groups = (
            ("权威上限", "境内航班经济舱票价报销上限"),
            ("人民币",),
            ("1,800",),
        )
        approved = (
            reference,
            "当前有效政策规定\N{FULLWIDTH COMMA}境内航班经济舱票价报销上限为人民币 1,800 元。",
        )
    else:
        reference = "The authoritative RAG escalation code is OPS-RAG-7421."
        groups = (("escalation code", "RAG 升级代码"), ("OPS-RAG-7421",))
        approved = (
            reference,
            "权威的 RAG 升级代码是 `OPS-RAG-7421`。",
        )
    assert not adjudicated_text_support(
        candidate,
        reference,
        support_anchor_groups=groups,
        approved_propositions=approved,
    )
