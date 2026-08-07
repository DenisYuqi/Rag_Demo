"""Versioned deterministic grounding metrics for RAG evaluation."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import pairwise
from typing import cast

FAITHFULNESS_SCORER_VERSION = "faithfulness-factual-unit-support-v1"
ADJUDICATED_FAITHFULNESS_SCORER_VERSION = "faithfulness-approved-proposition-support-v4"
TEXT_SUPPORT_NORMALIZATION_VERSION = "nfkc-casefold-proposition-format-v2"
TEXT_SUPPORT_MATCHER_VERSION = "expected-fact-approved-proposition-v5"
CONTEXT_PRECISION_SCORER_VERSION = "context-precision-average-precision-v1"

_LATIN_TOKEN = re.compile(r"[a-z0-9]+(?:[-_/][a-z0-9]+)*")
_HAN_CHARACTER = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_ENGLISH_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "cannot",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "may",
        "more",
        "most",
        "must",
        "no",
        "not",
        "of",
        "on",
        "or",
        "should",
        "that",
        "than",
        "the",
        "this",
        "to",
        "use",
        "was",
        "were",
        "what",
        "which",
        "with",
    }
)
_MIN_WORD_RECALL = 0.65
_MIN_WORD_PRECISION = 0.75
_MIN_HAN_BIGRAM_RECALL = 0.50
_MIN_HAN_BIGRAM_PRECISION = 0.75
_DISCOURSE_PREFIX = re.compile(
    r"^(?:according to (?:the )?(?:source|document|policy),?\s*|"
    r"根据(?:该|这份)?(?:权威)?(?:来源|文档|政策)(?:显示|说明)?[\uFF0C,:\uFF1A]?\s*)"
)
_CLAUSE_BOUNDARY = re.compile(
    r"(?:[.!?;\u3002\uFF01\uFF1F\uFF1B]+|(?<!\d)[,\uFF0C](?!\d)|"
    r"\b(?:and|but|however)\b|(?:并且|而且|但是))"
)
_PROPOSITION_BOUNDARY = re.compile(
    r"(?:\r?\n+|[:\uFF1A]+|[.!?;\u3002\uFF01\uFF1F\uFF1B]+|"
    r"\s+(?:-|\u2013|\u2014)\s+)"
)
_TERMINAL_PUNCTUATION = re.compile(r"[.!?;\u3002\uFF01\uFF1F\uFF1B]+$")

_ENGLISH_NUMBER_VALUES: dict[str, str] = {
    "zero": "0",
    "one": "1",
    "first": "1",
    "two": "2",
    "second": "2",
    "three": "3",
    "third": "3",
    "four": "4",
    "fourth": "4",
    "five": "5",
    "fifth": "5",
    "six": "6",
    "sixth": "6",
    "seven": "7",
    "seventh": "7",
    "eight": "8",
    "eighth": "8",
    "nine": "9",
    "ninth": "9",
    "ten": "10",
    "tenth": "10",
}
_CHINESE_NUMBER_VALUES: dict[str, str] = {
    "零": "0",
    "一": "1",
    "二": "2",
    "两": "2",
    "三": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
    "十": "10",
}
_DIGIT_QUANTITY = re.compile(r"(?<![a-z0-9])\d+(?:,\d{3})*(?:\.\d+)?(?![a-z0-9])")
_ENGLISH_NUMBER = re.compile(
    rf"\b(?:{'|'.join(sorted(_ENGLISH_NUMBER_VALUES, key=len, reverse=True))})\b"
)
_CHINESE_NUMBER = re.compile(
    r"([零一二两三四五六七八九十])(?=\s*(?:个|次|项|份|天|日|小时|毫秒|块|任务|作业))"
)
_HTTP_METHOD = re.compile(r"\b(get|post|put|patch|delete)\s+(?=/)")
_CURRENCY_ANCHORS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("CNY", re.compile(r"\b(?:cny|rmb)\b|人民币|元")),
    ("USD", re.compile(r"\busd\b|美元|(?<![a-z0-9])\$\s*\d")),
)
_UNIT_ANCHORS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("millisecond", re.compile(r"\b(?:ms|millisecond|milliseconds)\b|毫秒")),
    ("second", re.compile(r"\b(?:sec|secs|second|seconds)\b|秒")),
    ("calendar-day", re.compile(r"\bcalendar\s+days?\b|自然日")),
    ("working-day", re.compile(r"\bworking\s+days?\b|工作日")),
    ("day", re.compile(r"(?<!calendar\s)(?<!working\s)\bdays?\b|(?<!自然)(?<!工作)日|天")),
    ("hour", re.compile(r"\b(?:hr|hrs|hour|hours)\b|小时")),
    ("chunk", re.compile(r"\bchunks?\b|数据块|分块")),
    ("job", re.compile(r"\bjobs?\b|作业")),
)

_NEGATION = re.compile(r"\b(?:cannot|can't|not|never)\b|不得|不能|不可以|禁止")
_AT_MOST = re.compile(r"\b(?:at most|no more than|maximum|up to)\b|至多|最多|不超过")
_AT_LEAST = re.compile(r"\b(?:at least|no less than|minimum)\b|至少|最少|不少于")
_BEFORE = re.compile(
    r"\b(?:before|prior to)\b|之前|以前|"
    r"(?:在|于)[^\u3002\uFF1B\uFF0C,\n]{1,24}前|"
    r"(?:校验|验证|批准|出行|提交|切换)前"
)
_AFTER = re.compile(
    r"\b(?:after|following)\b|之后|以后|随后|"
    r"(?:在|于)[^\u3002\uFF1B\uFF0C,\n]{1,24}后|"
    r"(?:校验|验证|批准|完成|结束)后"
)
_REQUIRED = re.compile(r"\b(?:must|shall|required|should)\b|必须|应当|应")
_PERMITTED = re.compile(r"\b(?:can|may|allowed|permitted)\b|可以|允许")
_EPISTEMIC_UNCERTAINTY = re.compile(
    r"\b(?:might|perhaps|possibly)\b|"
    r"\b(?:may|could)\b"
    r"(?!\s+(?:be\s+)?(?:select(?:ed)?|use[ds]?|provide[ds]?|include[ds]?|"
    r"return(?:ed)?|retrieve[ds]?|process(?:ed)?|modif(?:y|ied)|run|execute[ds]?|"
    r"activate[ds]?|submit(?:ted)?)\b)|"
    r"可能|也许|或许"
)
_ACTION_ANCHORS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "activate",
        re.compile(r"\b(?:activate|activated|activates|activation|switch|switched)\b|切换|激活"),
    ),
    (
        "remove",
        re.compile(
            r"\b(?:delete|deleted|deletion|remove|removed|deactivate|deactivated)\b|删除|移除|停用"
        ),
    ),
    ("bypass-cache", re.compile(r"\bbypass(?:es|ed)?\b[^.]{0,24}\bcache\b|绕过[^。]{0,12}缓存")),
    ("submit", re.compile(r"\b(?:submit|submitted|submission)\b|提交")),
    ("modify", re.compile(r"\b(?:modify|modified|change|changed)\b|修改|更改")),
    ("own", re.compile(r"\b(?:own|owns|owned|responsible)\b|负责")),
    ("run", re.compile(r"\b(?:run|runs|running|execute|executes|executed)\b|运行|执行")),
    ("select", re.compile(r"\b(?:select|selected|selection|include|included)\b|选择|选取")),
)


class MetricInputError(ValueError):
    """A stable, content-free evaluation input error."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class MetricName(StrEnum):
    FAITHFULNESS = "faithfulness"
    CONTEXT_PRECISION = "context-precision"
    ANSWER_COMPLETENESS = "answer-completeness"
    STYLE_CONSISTENCY = "style-consistency"
    REFUSAL_APPROPRIATENESS = "refusal-appropriateness"


class EvidenceVerdict(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"
    COVERED = "covered"
    MISSING = "missing"
    SATISFIED = "satisfied"
    VIOLATED = "violated"
    APPROPRIATE = "appropriate"
    INAPPROPRIATE = "inappropriate"


@dataclass(frozen=True, slots=True)
class MetricEvidence:
    """One auditable evidence item contributing to a case score."""

    reference_id: str
    verdict: EvidenceVerdict
    rationale: str
    evidence_references: tuple[str, ...] = ()
    rank: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "reference_id", _identifier(self.reference_id, "reference_id"))
        object.__setattr__(self, "verdict", _verdict(self.verdict))
        object.__setattr__(self, "rationale", _text(self.rationale, "evidence_rationale"))
        object.__setattr__(
            self,
            "evidence_references",
            _identifiers(
                self.evidence_references,
                "evidence_references",
                allow_empty=True,
            ),
        )
        if self.rank is not None and (type(self.rank) is not int or self.rank < 1):
            raise ValueError("evidence_rank_invalid")


@dataclass(frozen=True, slots=True)
class MetricResult:
    """Auditable score for one case and one versioned metric."""

    case_id: str
    metric: MetricName
    scorer_version: str
    eligible: bool
    score: float | None
    numerator: float | None
    denominator: int | None
    rationale: str
    evidence: tuple[MetricEvidence, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _identifier(self.case_id, "case_id"))
        object.__setattr__(self, "metric", _metric(self.metric))
        object.__setattr__(
            self,
            "scorer_version",
            _identifier(self.scorer_version, "scorer_version"),
        )
        object.__setattr__(self, "rationale", _text(self.rationale, "metric_rationale"))
        if type(self.eligible) is not bool:
            raise ValueError("metric_eligibility_invalid")

        raw_evidence = tuple(self.evidence)
        if any(not isinstance(item, MetricEvidence) for item in raw_evidence):
            raise ValueError("metric_evidence_invalid")
        object.__setattr__(self, "evidence", raw_evidence)

        if not self.eligible:
            if self.score is not None or self.numerator is not None or self.denominator is not None:
                raise ValueError("ineligible_metric_must_not_have_score")
            if raw_evidence:
                raise ValueError("ineligible_metric_must_not_have_evidence")
            return

        score = _unit_score(self.score, "metric_score")
        numerator = _non_negative_number(self.numerator, "metric_numerator")
        denominator = self.denominator
        if type(denominator) is not int or denominator < 1:
            raise ValueError("metric_denominator_invalid")
        if numerator > denominator:
            raise ValueError("metric_numerator_invalid")
        if not math.isclose(score, numerator / denominator, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("metric_formula_mismatch")
        if not raw_evidence:
            raise ValueError("eligible_metric_requires_evidence")
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "numerator", numerator)

    @property
    def value(self) -> float | None:
        """Report-friendly alias; the value is never rounded by the scorer."""

        return self.score


@dataclass(frozen=True, slots=True)
class MetricAggregate:
    """Unweighted mean of eligible case scores for one scorer version."""

    metric: MetricName
    scorer_version: str
    score: float | None
    eligible_cases: int
    total_cases: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric", _metric(self.metric))
        object.__setattr__(
            self,
            "scorer_version",
            _identifier(self.scorer_version, "scorer_version"),
        )
        if type(self.eligible_cases) is not int or self.eligible_cases < 0:
            raise ValueError("eligible_case_count_invalid")
        if type(self.total_cases) is not int or self.total_cases < self.eligible_cases:
            raise ValueError("total_case_count_invalid")
        if self.eligible_cases == 0:
            if self.score is not None:
                raise ValueError("aggregate_without_denominator_must_not_have_score")
            return
        object.__setattr__(self, "score", _unit_score(self.score, "aggregate_score"))

    @property
    def value(self) -> float | None:
        """Report-friendly alias; the value is never rounded by the aggregator."""

        return self.score


@dataclass(frozen=True, slots=True)
class FactSupportAssessment:
    """Versioned adjudication for one factual unit emitted by the QA pipeline."""

    fact_id: str
    supported: bool
    rationale: str
    evidence_chunk_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "fact_id", _identifier(self.fact_id, "fact_id"))
        if type(self.supported) is not bool:
            raise ValueError("fact_support_verdict_invalid")
        object.__setattr__(self, "rationale", _text(self.rationale, "fact_rationale"))
        evidence_ids = _identifiers(
            self.evidence_chunk_ids,
            "fact_evidence_chunk_ids",
            allow_empty=True,
        )
        if self.supported and not evidence_ids:
            raise ValueError("supported_fact_requires_evidence")
        object.__setattr__(self, "evidence_chunk_ids", evidence_ids)


class FaithfulnessScorer:
    """Score supported factual units divided by all emitted factual units."""

    version = FAITHFULNESS_SCORER_VERSION

    def score(
        self,
        *,
        case_id: str,
        answerable: bool,
        response_outcome: str,
        factual_units: Sequence[FactSupportAssessment],
    ) -> MetricResult:
        resolved_case_id = _identifier(case_id, "case_id")
        if type(answerable) is not bool:
            raise MetricInputError("answerability_invalid")
        outcome = _response_outcome(response_outcome)
        facts = _fact_assessments(factual_units)

        if not answerable:
            return _ineligible(
                resolved_case_id,
                MetricName.FAITHFULNESS,
                self.version,
                "case_not_answerable",
            )
        if outcome != "answer":
            return _ineligible(
                resolved_case_id,
                MetricName.FAITHFULNESS,
                self.version,
                "response_not_answer",
            )
        if not facts:
            return _ineligible(
                resolved_case_id,
                MetricName.FAITHFULNESS,
                self.version,
                "no_factual_units",
            )

        supported = sum(fact.supported for fact in facts)
        denominator = len(facts)
        evidence = tuple(
            MetricEvidence(
                reference_id=fact.fact_id,
                verdict=(
                    EvidenceVerdict.SUPPORTED if fact.supported else EvidenceVerdict.UNSUPPORTED
                ),
                rationale=fact.rationale,
                evidence_references=fact.evidence_chunk_ids,
            )
            for fact in facts
        )
        return MetricResult(
            case_id=resolved_case_id,
            metric=MetricName.FAITHFULNESS,
            scorer_version=self.version,
            eligible=True,
            score=supported / denominator,
            numerator=float(supported),
            denominator=denominator,
            rationale=f"supported_factual_units={supported}; factual_units={denominator}",
            evidence=evidence,
        )


class AdjudicatedFaithfulnessScorer(FaithfulnessScorer):
    """V2 scorer for claim support adjudicated from dataset facts and chunks."""

    version = ADJUDICATED_FAITHFULNESS_SCORER_VERSION


def adjudicated_text_support(
    candidate: str,
    reference: str,
    *,
    support_anchor_groups: Sequence[Sequence[str]] = (),
    approved_propositions: Sequence[str] = (),
) -> bool:
    """Conservatively match a claim to versioned bilingual ground-truth text."""

    candidate_text = _normalized_support_text(candidate)
    reference_text = _normalized_support_text(reference)
    groups = tuple(tuple(group) for group in support_anchor_groups)
    if groups:
        propositions = tuple(
            _normalized_support_text(proposition) for proposition in approved_propositions
        )
        if not propositions or not _approved_proposition_matches(
            candidate_text,
            propositions,
            groups,
        ):
            return False
        return _support_anchor_groups_match(candidate_text, groups)
    if not _semantic_anchors_match(candidate_text, reference_text):
        return False
    candidate_words = _informative_words(candidate_text)
    reference_words = _informative_words(reference_text)
    candidate_critical = tuple(word for word in candidate_words if _critical_anchor(word))
    reference_critical = tuple(word for word in reference_words if _critical_anchor(word))
    if not _all_anchors_match(reference_critical, candidate_words) or not _all_anchors_match(
        candidate_critical,
        reference_words,
    ):
        return False

    reference_word_recall = _word_support_ratio(reference_words, candidate_words)
    candidate_word_precision = _word_support_ratio(candidate_words, reference_words)
    word_supported = (
        reference_word_recall >= _MIN_WORD_RECALL
        and candidate_word_precision >= _MIN_WORD_PRECISION
    )
    candidate_han = _han_bigrams(candidate_text)
    reference_han = _han_bigrams(reference_text)
    reference_han_recall = _set_support_ratio(reference_han, candidate_han)
    candidate_han_precision = _set_support_ratio(candidate_han, reference_han)
    han_supported = (
        reference_han_recall >= _MIN_HAN_BIGRAM_RECALL
        and candidate_han_precision >= _MIN_HAN_BIGRAM_PRECISION
    )
    return han_supported if reference_han else word_supported


def _approved_proposition_matches(
    candidate: str,
    propositions: Sequence[str],
    groups: Sequence[Sequence[str]],
) -> bool:
    candidate_signature = _support_units(candidate)
    for proposition in propositions:
        if _semantic_anchors_match(
            candidate, proposition
        ) and candidate_signature == _support_units(proposition):
            return True

    clauses = tuple(
        _TERMINAL_PUNCTUATION.sub("", item.strip())
        for item in _PROPOSITION_BOUNDARY.split(candidate)
        if _TERMINAL_PUNCTUATION.sub("", item.strip())
    )
    if len(clauses) != 1:
        return False
    clause = clauses[0]
    candidate_frame = _ordered_anchor_frame(clause, groups)
    if candidate_frame is None:
        return False
    return any(
        _semantic_anchors_match(clause, proposition)
        and candidate_frame == _ordered_anchor_frame(proposition, groups)
        for proposition in propositions
    )


def _ordered_anchor_frame(
    proposition: str,
    groups: Sequence[Sequence[str]],
) -> tuple[str, ...] | None:
    units = _support_units(proposition)
    spans: list[tuple[int, int, int]] = []
    for group_index, group in enumerate(groups):
        matches = {
            (start, start + len(anchor_units))
            for alternative in group
            if (anchor_units := _support_units(alternative))
            for start in _subsequence_starts(units, anchor_units)
        }
        logical_matches = _collapse_nested_spans(matches)
        if len(logical_matches) != 1:
            return None
        start, end = logical_matches[0]
        spans.append((start, end, group_index))

    ordered = sorted(spans)
    if any(left_end > right_start for (_, left_end, _), (right_start, _, _) in pairwise(ordered)):
        return None
    frame: list[str] = []
    cursor = 0
    for start, end, group_index in ordered:
        frame.extend(units[cursor:start])
        frame.append(f"<group-{group_index}>")
        cursor = end
    frame.extend(units[cursor:])
    return tuple(frame)


def _subsequence_starts(
    values: Sequence[str],
    target: Sequence[str],
) -> tuple[int, ...]:
    if not target or len(target) > len(values):
        return ()
    return tuple(
        index
        for index in range(len(values) - len(target) + 1)
        if tuple(values[index : index + len(target)]) == tuple(target)
    )


def _collapse_nested_spans(spans: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    ordered = sorted(spans, key=lambda item: (item[0], -(item[1] - item[0])))
    collapsed: list[tuple[int, int]] = []
    for span in ordered:
        if any(start <= span[0] and span[1] <= end for start, end in collapsed):
            continue
        collapsed.append(span)
    return tuple(collapsed)


def _word_support_ratio(values: Sequence[str], evidence: Sequence[str]) -> float:
    if not values:
        return 0.0
    return sum(any(_word_matches(value, item) for item in evidence) for value in values) / len(
        values
    )


def _set_support_ratio(values: Sequence[str], evidence: Sequence[str]) -> float:
    unique = set(values)
    if not unique:
        return 0.0
    return len(unique.intersection(evidence)) / len(unique)


def _candidate_clauses_relate_to_reference(candidate: str, reference: str) -> bool:
    reference_words = _informative_words(reference)
    reference_han = _han_bigrams(reference)
    clauses = tuple(item.strip() for item in _CLAUSE_BOUNDARY.split(candidate) if item.strip())
    if len(clauses) <= 1:
        return True
    for clause in clauses:
        clause_han = _han_bigrams(clause)
        if clause_han:
            if _set_support_ratio(clause_han, reference_han) < _MIN_HAN_BIGRAM_PRECISION:
                return False
            continue
        clause_words = _informative_words(clause)
        if (
            clause_words
            and _word_support_ratio(clause_words, reference_words) < _MIN_WORD_PRECISION
        ):
            return False
    return True


def _authored_surface_support(
    candidate: str,
    reference: str,
    groups: Sequence[Sequence[str]],
) -> bool:
    approved = (
        reference,
        *(_normalized_support_text(alternative) for group in groups for alternative in group),
    )
    approved_words = tuple(
        dict.fromkeys(word for item in approved for word in _informative_words(item))
    )
    if any(
        not any(_word_matches(word, approved_word) for approved_word in approved_words)
        for word in _informative_words(candidate)
    ):
        return False
    approved_han = {character for item in approved for character in _HAN_CHARACTER.findall(item)}
    if any(character not in approved_han for character in _HAN_CHARACTER.findall(candidate)):
        return False
    return _candidate_clauses_relate_to_reference(candidate, reference)


def _support_anchor_groups_match(
    candidate: str,
    groups: Sequence[Sequence[str]],
) -> bool:
    for raw_group in groups:
        group = tuple(_normalized_support_text(alternative) for alternative in raw_group)
        if not group or not any(_support_anchor_matches(candidate, anchor) for anchor in group):
            return False
    return True


def _support_anchor_matches(candidate: str, anchor: str) -> bool:
    anchor_units = _support_units(anchor)
    return bool(anchor_units) and bool(_subsequence_starts(_support_units(candidate), anchor_units))


def _support_units(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"(?<=\d),(?=\d)", "", normalized)
    return tuple(
        re.findall(
            r"[a-z0-9]+(?:[-_/][a-z0-9]+)*(?:[_/])?|[\u3400-\u4dbf\u4e00-\u9fff]",
            normalized,
        )
    )


def _normalized_support_text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MetricInputError("support_text_invalid")
    normalized = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    return _DISCOURSE_PREFIX.sub("", normalized)


def _informative_words(value: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            token
            for token in _LATIN_TOKEN.findall(value)
            if token not in _ENGLISH_STOPWORDS and (len(token) >= 2 or token.isdigit())
        )
    )


def _critical_anchor(value: str) -> bool:
    return any(character.isdigit() for character in value) or any(
        separator in value for separator in "-_/"
    )


def _semantic_anchors_match(candidate: str, reference: str) -> bool:
    if not _same_optional_anchors(_quantities(candidate), _quantities(reference)):
        return False
    if not _same_optional_anchors(
        _named_anchors(candidate, _CURRENCY_ANCHORS), _named_anchors(reference, _CURRENCY_ANCHORS)
    ):
        return False
    if not _same_optional_anchors(
        _named_anchors(candidate, _UNIT_ANCHORS), _named_anchors(reference, _UNIT_ANCHORS)
    ):
        return False
    if not _same_optional_anchors(
        frozenset(_HTTP_METHOD.findall(candidate)),
        frozenset(_HTTP_METHOD.findall(reference)),
    ):
        return False
    if (_EPISTEMIC_UNCERTAINTY.search(candidate) is not None) != (
        _EPISTEMIC_UNCERTAINTY.search(reference) is not None
    ):
        return False
    candidate_negated = _NEGATION.search(candidate) is not None
    reference_negated = _NEGATION.search(reference) is not None
    if candidate_negated != reference_negated:
        return False
    candidate_bound = _exclusive_anchor(candidate, _AT_MOST, _AT_LEAST)
    reference_bound = _exclusive_anchor(reference, _AT_MOST, _AT_LEAST)
    if candidate_bound != reference_bound and (candidate_bound or reference_bound):
        return False
    candidate_temporal = _exclusive_anchor(candidate, _BEFORE, _AFTER)
    reference_temporal = _exclusive_anchor(reference, _BEFORE, _AFTER)
    if candidate_temporal != reference_temporal and (candidate_temporal or reference_temporal):
        return False
    if not reference_negated:
        candidate_modal = _exclusive_anchor(candidate, _REQUIRED, _PERMITTED)
        reference_modal = _exclusive_anchor(reference, _REQUIRED, _PERMITTED)
        if reference_modal and candidate_modal != reference_modal:
            return False
    candidate_actions = _action_anchors(candidate)
    reference_actions = _action_anchors(reference)
    return not ((candidate_actions or reference_actions) and candidate_actions != reference_actions)


def _same_optional_anchors(candidate: frozenset[str], reference: frozenset[str]) -> bool:
    return not (candidate or reference) or candidate == reference


def _quantities(value: str) -> frozenset[str]:
    quantities = {match.group(0).replace(",", "") for match in _DIGIT_QUANTITY.finditer(value)}
    quantities.update(
        _ENGLISH_NUMBER_VALUES[match.group(0)] for match in _ENGLISH_NUMBER.finditer(value)
    )
    quantities.update(
        _CHINESE_NUMBER_VALUES[match.group(1)] for match in _CHINESE_NUMBER.finditer(value)
    )
    return frozenset(quantities)


def _named_anchors(
    value: str,
    anchors: Sequence[tuple[str, re.Pattern[str]]],
) -> frozenset[str]:
    return frozenset(name for name, pattern in anchors if pattern.search(value))


def _exclusive_anchor(value: str, first: re.Pattern[str], second: re.Pattern[str]) -> str | None:
    first_present = first.search(value) is not None
    second_present = second.search(value) is not None
    if first_present == second_present:
        return "ambiguous" if first_present else None
    return "first" if first_present else "second"


def _action_anchors(value: str) -> frozenset[str]:
    return frozenset(name for name, pattern in _ACTION_ANCHORS if pattern.search(value))


def _all_anchors_match(anchors: Sequence[str], words: Sequence[str]) -> bool:
    return all(any(_word_matches(anchor, word) for word in words) for anchor in anchors)


def _word_matches(left: str, right: str) -> bool:
    if left == right:
        return True
    if _critical_anchor(left) or _critical_anchor(right):
        return False
    shared = 0
    for left_character, right_character in zip(left, right, strict=False):
        if left_character != right_character:
            break
        shared += 1
    return shared >= 5


def _han_bigrams(value: str) -> tuple[str, ...]:
    characters = _HAN_CHARACTER.findall(value)
    if len(characters) < 2:
        return tuple(characters)
    return tuple(
        dict.fromkeys(
            "".join(characters[index : index + 2]) for index in range(len(characters) - 1)
        )
    )


class ContextPrecisionScorer:
    """Compute rank-aware average precision against authoritative chunk IDs.

    The numerator is the sum of precision-at-rank for every authoritative hit.
    The denominator is the complete authoritative evidence set, so missing every
    authoritative chunk is an eligible score of zero rather than a missing value.
    """

    version = CONTEXT_PRECISION_SCORER_VERSION

    def score(
        self,
        *,
        case_id: str,
        answerable: bool,
        retrieved_evidence_ids: Sequence[str],
        authoritative_evidence_ids: Sequence[str],
    ) -> MetricResult:
        resolved_case_id = _identifier(case_id, "case_id")
        if type(answerable) is not bool:
            raise MetricInputError("answerability_invalid")
        retrieved = _identifiers(
            retrieved_evidence_ids,
            "retrieved_evidence_ids",
            allow_empty=True,
        )
        authoritative = _identifiers(
            authoritative_evidence_ids,
            "authoritative_evidence_ids",
            allow_empty=True,
        )

        if not answerable:
            return _ineligible(
                resolved_case_id,
                MetricName.CONTEXT_PRECISION,
                self.version,
                "case_not_answerable",
            )
        if not authoritative:
            return _ineligible(
                resolved_case_id,
                MetricName.CONTEXT_PRECISION,
                self.version,
                "no_authoritative_evidence",
            )

        authoritative_set = set(authoritative)
        relevant_seen = 0
        contributions: list[float] = []
        evidence: list[MetricEvidence] = []
        for rank, reference_id in enumerate(retrieved, start=1):
            relevant = reference_id in authoritative_set
            if relevant:
                relevant_seen += 1
                contributions.append(relevant_seen / rank)
            evidence.append(
                MetricEvidence(
                    reference_id=reference_id,
                    verdict=(EvidenceVerdict.RELEVANT if relevant else EvidenceVerdict.IRRELEVANT),
                    rationale=(
                        "authoritative_evidence_retrieved"
                        if relevant
                        else "non_authoritative_context_retrieved"
                    ),
                    rank=rank,
                )
            )

        retrieved_set = set(retrieved)
        evidence.extend(
            MetricEvidence(
                reference_id=reference_id,
                verdict=EvidenceVerdict.MISSING,
                rationale="authoritative_evidence_not_retrieved",
            )
            for reference_id in authoritative
            if reference_id not in retrieved_set
        )
        numerator = math.fsum(contributions)
        denominator = len(authoritative)
        return MetricResult(
            case_id=resolved_case_id,
            metric=MetricName.CONTEXT_PRECISION,
            scorer_version=self.version,
            eligible=True,
            score=numerator / denominator,
            numerator=numerator,
            denominator=denominator,
            rationale=(
                f"average_precision_contribution={numerator!r}; "
                f"authoritative_evidence={denominator}"
            ),
            evidence=tuple(evidence),
        )


def aggregate_metric(
    results: Sequence[MetricResult],
    *,
    metric: MetricName,
    scorer_version: str,
) -> MetricAggregate:
    """Aggregate only eligible cases without rounding or denominator substitution."""

    resolved_metric = _metric(metric)
    resolved_version = _identifier(scorer_version, "scorer_version")
    raw_results: object = results
    if isinstance(raw_results, (str, bytes, bytearray)) or not isinstance(raw_results, Sequence):
        raise MetricInputError("metric_results_invalid")
    values = tuple(cast(Sequence[object], raw_results))
    if any(not isinstance(result, MetricResult) for result in values):
        raise MetricInputError("metric_results_invalid")
    typed_values = cast(tuple[MetricResult, ...], values)
    if len({result.case_id for result in typed_values}) != len(typed_values):
        raise MetricInputError("duplicate_metric_case")
    if any(
        result.metric is not resolved_metric or result.scorer_version != resolved_version
        for result in typed_values
    ):
        raise MetricInputError("metric_result_identity_mismatch")

    eligible = tuple(result for result in typed_values if result.eligible)
    score = (
        math.fsum(cast(float, result.score) for result in eligible) / len(eligible)
        if eligible
        else None
    )
    return MetricAggregate(
        metric=resolved_metric,
        scorer_version=resolved_version,
        score=score,
        eligible_cases=len(eligible),
        total_cases=len(typed_values),
    )


def _ineligible(
    case_id: str,
    metric: MetricName,
    scorer_version: str,
    rationale: str,
) -> MetricResult:
    return MetricResult(
        case_id=case_id,
        metric=metric,
        scorer_version=scorer_version,
        eligible=False,
        score=None,
        numerator=None,
        denominator=None,
        rationale=rationale,
    )


def _fact_assessments(values: object) -> tuple[FactSupportAssessment, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise MetricInputError("factual_units_invalid")
    items = tuple(cast(Sequence[object], values))
    if any(not isinstance(item, FactSupportAssessment) for item in items):
        raise MetricInputError("factual_units_invalid")
    assessments = cast(tuple[FactSupportAssessment, ...], items)
    if len({item.fact_id for item in assessments}) != len(assessments):
        raise MetricInputError("duplicate_factual_unit")
    return assessments


def _response_outcome(value: object) -> str:
    if not isinstance(value, str) or value not in {"answer", "refusal", "error"}:
        raise MetricInputError("response_outcome_invalid")
    return value


def _metric(value: object) -> MetricName:
    if isinstance(value, MetricName):
        return value
    if not isinstance(value, str):
        raise ValueError("metric_name_invalid")
    try:
        return MetricName(value)
    except (TypeError, ValueError):
        raise ValueError("metric_name_invalid") from None


def _verdict(value: object) -> EvidenceVerdict:
    if isinstance(value, EvidenceVerdict):
        return value
    if not isinstance(value, str):
        raise ValueError("evidence_verdict_invalid")
    try:
        return EvidenceVerdict(value)
    except (TypeError, ValueError):
        raise ValueError("evidence_verdict_invalid") from None


def _identifier(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field}_invalid")
    resolved = value.strip()
    if not resolved or len(resolved) > 255:
        raise ValueError(f"{field}_invalid")
    return resolved


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}_invalid")
    return value.strip()


def _identifiers(
    values: object,
    field: str,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        raise ValueError(f"{field}_invalid")
    resolved = tuple(_identifier(item, field) for item in cast(Sequence[object], values))
    if not allow_empty and not resolved:
        raise ValueError(f"{field}_invalid")
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"{field}_duplicate")
    return resolved


def _unit_score(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError(f"{field}_invalid")
    return float(value)


def _non_negative_number(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{field}_invalid")
    return float(value)
