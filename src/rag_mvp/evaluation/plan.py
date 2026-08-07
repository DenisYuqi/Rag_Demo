"""Safe dataset resolution and reproducible evaluation-plan construction."""

from __future__ import annotations

import hashlib
import platform
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from rag_mvp.config.settings import Settings
from rag_mvp.domain.qa import ConversationRole
from rag_mvp.domain.retrieval import RetrievalMode
from rag_mvp.evaluation.answer_metrics import (
    ANSWER_COMPLETENESS_SCORER_VERSION,
    REFUSAL_APPROPRIATENESS_SCORER_VERSION,
    STYLE_CONSISTENCY_SCORER_VERSION,
)
from rag_mvp.evaluation.dataset import (
    CorpusDerivation,
    DatasetValidationError,
    EvaluationDataset,
    load_dataset,
)
from rag_mvp.evaluation.grounding_metrics import (
    CONTEXT_PRECISION_SCORER_VERSION,
    FAITHFULNESS_SCORER_VERSION,
    MetricName,
)
from rag_mvp.evaluation.quality_gate import QUALITY_GATE_VERSION
from rag_mvp.evaluation.runner import (
    EvaluationCaseInput,
    EvaluationConversationTurn,
    EvaluationEnvironment,
    EvaluationRunIdentity,
    EvaluationRunPlan,
)
from rag_mvp.evaluation.scoring import SCORING_PIPELINE_VERSION
from rag_mvp.ingestion.chunking import CHUNKING_VERSION, TOKENIZER_VERSION
from rag_mvp.ingestion.indexing import INDEX_EXTRACTION_VERSION
from rag_mvp.ingestion.normalization import NORMALIZATION_VERSION
from rag_mvp.providers.models import GenerationFormat, NormalizationPolicy
from rag_mvp.qa.context import CONTEXT_SELECTION_VERSION, CONTEXT_TOKENIZER_VERSION
from rag_mvp.qa.evidence_assessor import FACT_EVIDENCE_ASSESSOR_VERSION
from rag_mvp.qa.grounding import GROUNDING_VALIDATOR_VERSION
from rag_mvp.qa.prompt import (
    GENERATOR_OUTPUT_SCHEMA_VERSION,
    GENERATOR_PROMPT_VERSION,
    GeneratorPromptBuilder,
)
from rag_mvp.qa.query_rewrite import QUERY_REWRITE_VERSION
from rag_mvp.qa.refusal import REFUSAL_POLICY_VERSION
from rag_mvp.retrieval.fusion import RrfConfig
from rag_mvp.retrieval.rerank import RerankStage
from rag_mvp.retrieval.service import DEGRADATION_POLICY_VERSION, RRF_TIE_POLICY_VERSION

SOURCE_CODE_IDENTITY_VERSION = "rag-mvp-source-sha256-v1"
PLAN_IDENTITY_VERSION = "evaluation-plan-identity-v1"
OPENAI_COMPATIBLE_ADAPTER_VERSION = "openai-compatible-v1"
EVALUATION_RANDOM_SEED = 0

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,254}$")
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,254}$")
_SEMANTIC_VERSION = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")
_IGNORED_SOURCE_SUFFIXES = frozenset({".pyc", ".pyo"})


class EvaluationPlanError(ValueError):
    """A stable, content-free dataset or plan construction error."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class EvaluationDatasetRegistry:
    """Resolve one fully validated immutable dataset without implicit upgrades."""

    root: Path

    def __init__(self, root: str | Path) -> None:
        object.__setattr__(self, "root", Path(root))

    def resolve(self, dataset_id: str, version: str | None = None) -> EvaluationDataset:
        resolved_id = _safe_identifier(dataset_id, "dataset_id")
        resolved_version = _safe_version(version) if version is not None else None
        datasets = self._scan()
        matches = tuple(
            dataset
            for dataset in datasets
            if dataset.manifest.dataset_id == resolved_id
            and (resolved_version is None or dataset.manifest.version == resolved_version)
        )
        if not matches:
            raise EvaluationPlanError("evaluation_dataset_not_found")
        if len(matches) != 1:
            raise EvaluationPlanError("evaluation_dataset_ambiguous")
        return matches[0]

    def list(self) -> tuple[EvaluationDataset, ...]:
        """Return the validated, immutable catalog without selecting or upgrading a version."""

        return self._scan()

    def _scan(self) -> tuple[EvaluationDataset, ...]:
        if self.root.is_symlink():
            raise EvaluationPlanError("evaluation_dataset_registry_unsafe")
        try:
            root = self.root.resolve(strict=True)
        except (OSError, RuntimeError):
            raise EvaluationPlanError("evaluation_dataset_registry_missing") from None
        if not root.is_dir():
            raise EvaluationPlanError("evaluation_dataset_registry_missing")
        try:
            entries = tuple(sorted(root.iterdir(), key=lambda path: path.name))
        except OSError:
            raise EvaluationPlanError("evaluation_dataset_registry_unreadable") from None

        datasets: list[EvaluationDataset] = []
        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.is_symlink():
                raise EvaluationPlanError("evaluation_dataset_registry_unsafe")
            if not entry.is_dir():
                continue
            manifest = entry / "manifest.json"
            if manifest.is_symlink() or not manifest.is_file():
                raise EvaluationPlanError("evaluation_dataset_registry_entry_invalid")
            try:
                dataset = load_dataset(entry, acceptance_mode=True)
            except (DatasetValidationError, OSError, ValueError):
                raise EvaluationPlanError("evaluation_dataset_registry_entry_invalid") from None
            if dataset.root != entry.resolve():
                raise EvaluationPlanError("evaluation_dataset_registry_unsafe")
            datasets.append(dataset)
        return tuple(datasets)


def source_code_revision(package_root: str | Path | None = None) -> str:
    """Hash paths and bytes for all non-generated files in the ``rag_mvp`` package."""

    unresolved = (
        Path(package_root) if package_root is not None else Path(__file__).resolve().parents[1]
    )
    if unresolved.is_symlink():
        raise EvaluationPlanError("source_package_unsafe")
    try:
        root = unresolved.resolve(strict=True)
    except (OSError, RuntimeError):
        raise EvaluationPlanError("source_package_missing") from None
    if not root.is_dir():
        raise EvaluationPlanError("source_package_missing")

    source_files: list[tuple[str, Path]] = []
    try:
        entries = tuple(root.rglob("*"))
    except OSError:
        raise EvaluationPlanError("source_package_unreadable") from None
    for entry in entries:
        relative_path = entry.relative_to(root)
        if entry.is_symlink():
            raise EvaluationPlanError("source_package_unsafe")
        if (
            "__pycache__" in relative_path.parts
            or entry.suffix.casefold() in _IGNORED_SOURCE_SUFFIXES
        ):
            continue
        if entry.is_file():
            source_files.append((relative_path.as_posix(), entry))
    if not source_files:
        raise EvaluationPlanError("source_package_empty")

    digest = hashlib.sha256()
    digest.update(SOURCE_CODE_IDENTITY_VERSION.encode("ascii"))
    digest.update(b"\0")
    for source_relative, path in sorted(source_files):
        try:
            payload = path.read_bytes()
        except OSError:
            raise EvaluationPlanError("source_package_unreadable") from None
        encoded_path = source_relative.encode("utf-8")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return f"sha256:{digest.hexdigest()}"


def build_evaluation_plan(
    dataset: EvaluationDataset,
    settings: Settings,
    run_id: str,
) -> EvaluationRunPlan:
    """Build an acceptance plan whose identity contains every configured RAG seam."""

    if not isinstance(dataset, EvaluationDataset):
        raise EvaluationPlanError("evaluation_dataset_invalid")
    if not isinstance(settings, Settings):
        raise EvaluationPlanError("evaluation_settings_invalid")
    if not isinstance(run_id, str) or _SAFE_RUN_ID.fullmatch(run_id) is None:
        raise EvaluationPlanError("evaluation_run_id_invalid")
    _require_matching_derivation(dataset, settings)

    retrieval_mode = RetrievalMode(settings.default_retrieval_mode)
    prompt_builder = GeneratorPromptBuilder()
    provider_alias, adapter_version = _provider_identity(settings)
    derivation = dataset.corpus.manifest.derivation
    cases = tuple(
        EvaluationCaseInput(
            case_id=case.case_id,
            question=case.question,
            language=case.language.value,
            history=tuple(
                EvaluationConversationTurn(
                    role=ConversationRole(turn.role),
                    content=turn.content,
                )
                for turn in case.history
            ),
            retrieval_mode=retrieval_mode,
        )
        for case in dataset.cases
    )

    identity = EvaluationRunIdentity(
        dataset_id=dataset.manifest.dataset_id,
        dataset_version=dataset.manifest.version,
        dataset_hash=dataset.manifest.content_hash,
        corpus_version=dataset.corpus.manifest.version,
        corpus_hash=dataset.corpus.manifest.content_hash,
        configuration_id=settings.configuration_identity,
        code_revision=source_code_revision(),
        prompt_versions={
            "generation": GENERATOR_PROMPT_VERSION,
            "generation-output-schema": GENERATOR_OUTPUT_SCHEMA_VERSION,
            "query-rewrite": QUERY_REWRITE_VERSION,
            "reranking": RerankStage.PROMPT_VERSION,
            "reranking-parser": RerankStage.PARSER_VERSION,
        },
        provider_identities={
            "backend": settings.provider_backend,
            "embedding": provider_alias,
            "generation": provider_alias,
            "reranking": provider_alias if settings.reranking_model is not None else "disabled",
            "adapter": adapter_version,
        },
        model_identities={
            "embedding": settings.embedding_model,
            "generation": settings.generation_model,
            "reranking": settings.reranking_model or "disabled",
        },
        generation_settings={
            "temperature": 0.0,
            "maximum_question_characters": prompt_builder.maximum_question_characters,
            "maximum_output_tokens": prompt_builder.maximum_output_tokens,
            "response_format": GenerationFormat.JSON_OBJECT.value,
            "max_tokens_parameter": settings.openai_max_tokens_parameter,
            "provider_timeout_seconds": settings.provider_timeout_seconds,
            "provider_retry_limit": settings.provider_retry_limit,
            "qa_deadline_seconds": settings.qa_deadline_seconds,
            "qa_generation_budget_seconds": settings.qa_generation_budget_seconds,
            "qa_finalization_budget_seconds": settings.qa_finalization_budget_seconds,
        },
        embedding_identity={
            "provider": provider_alias,
            "model": settings.embedding_model,
            "dimension": settings.embedding_dimension,
            "normalization": NormalizationPolicy.NONE.value,
            "adapter_version": adapter_version,
            "send_dimensions": settings.openai_send_dimensions,
        },
        chunking_identity={
            "extraction_version": derivation.extraction_version,
            "normalization_version": derivation.normalization_version,
            "chunking_version": derivation.chunking_version,
            "tokenizer_version": derivation.tokenizer_version,
            "target_tokens": derivation.target_tokens,
            "overlap_tokens": derivation.overlap_tokens,
            "ocr_enabled": settings.ocr_enabled,
            "ocr_languages": settings.ocr_languages,
        },
        retrieval_configuration={
            "identity_version": PLAN_IDENTITY_VERSION,
            "mode": retrieval_mode.value,
            "dense_candidate_limit": settings.dense_candidate_limit,
            "lexical_candidate_limit": settings.lexical_candidate_limit,
            "rerank_candidate_limit": settings.rerank_candidate_limit,
            "context_chunk_limit": settings.context_chunk_limit,
            "rrf_k": settings.rrf_k,
            "dense_weight": settings.dense_weight,
            "lexical_weight": settings.lexical_weight,
            "rrf_version": RrfConfig().version,
            "rrf_tie_policy_version": RRF_TIE_POLICY_VERSION,
            "reranking_enabled": settings.reranking_model is not None,
            "rerank_deadline_seconds": settings.rerank_deadline_seconds,
            "allow_single_retriever_degradation": (settings.allow_single_retriever_degradation),
            "degradation_policy_version": DEGRADATION_POLICY_VERSION,
            "retrieval_cache_enabled": settings.retrieval_cache_enabled,
            "qa_retrieval_budget_seconds": settings.qa_retrieval_budget_seconds,
            "qa_embedding_budget_seconds": settings.qa_embedding_budget_seconds,
            "qa_dense_retrieval_budget_seconds": settings.qa_dense_retrieval_budget_seconds,
            "qa_bm25_budget_seconds": settings.qa_bm25_budget_seconds,
            "qa_fusion_budget_seconds": settings.qa_fusion_budget_seconds,
            "qa_evidence_assessment_budget_seconds": (
                settings.qa_evidence_assessment_budget_seconds
            ),
            "context_selection_version": CONTEXT_SELECTION_VERSION,
            "context_tokenizer_version": CONTEXT_TOKENIZER_VERSION,
            "fact_evidence_assessor_version": FACT_EVIDENCE_ASSESSOR_VERSION,
            "minimum_support_score": settings.qa_minimum_support_score,
            "grounding_validator_version": GROUNDING_VALIDATOR_VERSION,
            "refusal_policy_version": REFUSAL_POLICY_VERSION,
        },
        scorer_versions={
            MetricName.FAITHFULNESS.value: FAITHFULNESS_SCORER_VERSION,
            MetricName.CONTEXT_PRECISION.value: CONTEXT_PRECISION_SCORER_VERSION,
            MetricName.ANSWER_COMPLETENESS.value: ANSWER_COMPLETENESS_SCORER_VERSION,
            MetricName.STYLE_CONSISTENCY.value: STYLE_CONSISTENCY_SCORER_VERSION,
            MetricName.REFUSAL_APPROPRIATENESS.value: (REFUSAL_APPROPRIATENESS_SCORER_VERSION),
            "scoring-pipeline": SCORING_PIPELINE_VERSION,
            "quality-gate": QUALITY_GATE_VERSION,
        },
        pricing_version=settings.pricing_version,
        random_seeds={
            "case-order": EVALUATION_RANDOM_SEED,
            "scoring": EVALUATION_RANDOM_SEED,
        },
        environment=EvaluationEnvironment(
            python_version=platform.python_version(),
            platform=f"{platform.system() or 'unknown'}-{platform.machine() or 'unknown'}",
            deployment=settings.environment,
        ),
        cache_policy="bypass",
    )
    try:
        return EvaluationRunPlan(run_id=run_id, identity=identity, cases=cases)
    except ValidationError:
        raise EvaluationPlanError("evaluation_plan_invalid") from None


def _require_matching_derivation(dataset: EvaluationDataset, settings: Settings) -> None:
    derivation = dataset.corpus.manifest.derivation
    expected = CorpusDerivation(
        target_tokens=settings.chunk_target_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
    )
    if derivation != expected or (
        derivation.extraction_version != INDEX_EXTRACTION_VERSION
        or derivation.normalization_version != NORMALIZATION_VERSION
        or derivation.chunking_version != CHUNKING_VERSION
        or derivation.tokenizer_version != TOKENIZER_VERSION
    ):
        raise EvaluationPlanError("dataset_settings_derivation_mismatch")


def _provider_identity(settings: Settings) -> tuple[str, str]:
    if settings.provider_backend == "offline":
        return "offline", "offline-config-v1"
    endpoint_digest = hashlib.sha256(
        settings.openai_base_url.rstrip("/").encode("utf-8")
    ).hexdigest()[:16]
    return f"openai-compatible-{endpoint_digest}", OPENAI_COMPATIBLE_ADAPTER_VERSION


def _safe_identifier(value: object, field: str) -> str:
    if not isinstance(value, str) or _SAFE_ID.fullmatch(value) is None:
        raise EvaluationPlanError(f"{field}_invalid")
    return value


def _safe_version(value: object) -> str:
    if not isinstance(value, str) or _SEMANTIC_VERSION.fullmatch(value) is None:
        raise EvaluationPlanError("dataset_version_invalid")
    return value


__all__ = [
    "EVALUATION_RANDOM_SEED",
    "OPENAI_COMPATIBLE_ADAPTER_VERSION",
    "PLAN_IDENTITY_VERSION",
    "SOURCE_CODE_IDENTITY_VERSION",
    "EvaluationDatasetRegistry",
    "EvaluationPlanError",
    "build_evaluation_plan",
    "source_code_revision",
]
