"""Immutable, versioned evaluation-dataset loading and validation."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

from pydantic import Field, StringConstraints, ValidationError, field_validator, model_validator

from rag_mvp.domain._base import DomainModel, Identifier, NonEmptyText
from rag_mvp.domain.ingestion import Chunk, ChunkLocator, DocumentKind, ExtractionMethod
from rag_mvp.ingestion.chunking import (
    CHUNKING_VERSION,
    TOKENIZER_VERSION,
    ChunkingConfig,
    chunk_document,
)
from rag_mvp.ingestion.extractors import ExtractedBlock, ExtractedDocument, extract_utf8_text
from rag_mvp.ingestion.indexing import INDEX_EXTRACTION_VERSION
from rag_mvp.ingestion.normalization import NORMALIZATION_VERSION, normalize_document

DATASET_SCHEMA_VERSION = "rag-evaluation-dataset-v1"
CORPUS_SCHEMA_VERSION = "rag-evaluation-corpus-v1"
HASH_ALGORITHM = "sha256-canonical-json-v1"

type SemanticVersion = Annotated[
    str,
    StringConstraints(pattern=r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"),
]
type Sha256Digest = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
type ChunkContentDigest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class DatasetValidationError(ValueError):
    """Raised when dataset provenance or eligibility cannot be trusted."""


class EvaluationLanguage(StrEnum):
    ENGLISH = "en"
    CHINESE = "zh"
    MIXED = "mixed"


class Answerability(StrEnum):
    ANSWERABLE = "answerable"
    UNANSWERABLE = "unanswerable"
    REQUIRED_REFUSAL = "required-refusal"


class EvaluationCategory(StrEnum):
    ANSWERABLE_CHINESE = "answerable-zh"
    ANSWERABLE_ENGLISH = "answerable-en"
    MULTI_TURN = "multi-turn"
    OCR = "ocr"
    UNANSWERABLE = "unanswerable"
    REQUIRED_REFUSAL = "required-refusal"
    INJECTION = "injection"
    PII = "pii"


class EvaluationMetric(StrEnum):
    FAITHFULNESS = "faithfulness"
    CONTEXT_PRECISION = "context-precision"
    ANSWER_COMPLETENESS = "answer-completeness"
    STYLE_CONSISTENCY = "style-consistency"
    REFUSAL_APPROPRIATENESS = "refusal-appropriateness"


class StyleExpectation(StrEnum):
    ANSWER_IN_REQUEST_LANGUAGE = "answer-in-request-language"
    CITATIONS_REQUIRED = "citations-required"
    CONCISE = "concise"
    REFUSAL_CONCISE = "refusal-concise"
    PII_REDACTED = "pii-redacted"


class CorpusSnapshotFormat(StrEnum):
    SOURCE = "source"
    FROZEN_OCR_PAGE = "frozen-ocr-page-v1"


ACCEPTANCE_REQUIRED_CATEGORIES: frozenset[EvaluationCategory] = frozenset(EvaluationCategory)
ACCEPTANCE_REQUIRED_METRICS: frozenset[EvaluationMetric] = frozenset(EvaluationMetric)


def _require_unique(values: tuple[object, ...], *, label: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{label} must be unique")


def _validate_relative_path(value: str) -> str:
    candidate = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or candidate.is_absolute()
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError("path must be a safe POSIX path relative to the dataset")
    return value


def _validate_optional_relative_path(value: str | None) -> str | None:
    return None if value is None else _validate_relative_path(value)


class ConversationTurn(DomainModel):
    role: Literal["user", "assistant"]
    content: NonEmptyText


class ExpectedFact(DomainModel):
    fact_id: Identifier
    text: NonEmptyText
    evidence_ids: Annotated[tuple[Identifier, ...], Field(min_length=1)]

    @field_validator("evidence_ids")
    @classmethod
    def evidence_ids_are_unique(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        _require_unique(value, label="expected-fact evidence IDs")
        return value


class EvaluationCase(DomainModel):
    case_id: Identifier
    question: NonEmptyText
    language: EvaluationLanguage
    answerability: Answerability
    category: EvaluationCategory
    expected_facts: tuple[ExpectedFact, ...] = ()
    authoritative_evidence_ids: tuple[Identifier, ...] = ()
    style_expectations: Annotated[tuple[StyleExpectation, ...], Field(min_length=1)]
    history: tuple[ConversationTurn, ...] = ()

    @model_validator(mode="after")
    def validate_case_contract(self) -> EvaluationCase:
        _require_unique(tuple(fact.fact_id for fact in self.expected_facts), label="fact IDs")
        _require_unique(self.authoritative_evidence_ids, label="authoritative evidence IDs")
        _require_unique(self.style_expectations, label="style expectations")

        fact_evidence = {
            evidence_id for fact in self.expected_facts for evidence_id in fact.evidence_ids
        }
        authoritative = set(self.authoritative_evidence_ids)
        if self.answerability is Answerability.ANSWERABLE:
            if not self.expected_facts or not authoritative:
                raise ValueError("answerable cases require facts and authoritative evidence")
            if fact_evidence != authoritative:
                raise ValueError(
                    "expected facts must map exactly to the authoritative evidence IDs"
                )
        elif self.expected_facts or authoritative:
            raise ValueError("refusal cases cannot declare answer facts or authoritative evidence")

        answerable_categories = {
            EvaluationCategory.ANSWERABLE_CHINESE,
            EvaluationCategory.ANSWERABLE_ENGLISH,
            EvaluationCategory.MULTI_TURN,
            EvaluationCategory.OCR,
            EvaluationCategory.PII,
        }
        if (
            self.category in answerable_categories
            and self.answerability is not Answerability.ANSWERABLE
        ):
            raise ValueError("the selected category requires an answerable label")
        if (
            self.category is EvaluationCategory.UNANSWERABLE
            and self.answerability is not Answerability.UNANSWERABLE
        ):
            raise ValueError("the unanswerable category requires an unanswerable label")
        if (
            self.category in {EvaluationCategory.REQUIRED_REFUSAL, EvaluationCategory.INJECTION}
            and self.answerability is not Answerability.REQUIRED_REFUSAL
        ):
            raise ValueError("the selected category requires a required-refusal label")
        if self.category is EvaluationCategory.MULTI_TURN and not self.history:
            raise ValueError("multi-turn cases require prior conversation history")
        if (
            self.category is EvaluationCategory.ANSWERABLE_CHINESE
            and self.language is not EvaluationLanguage.CHINESE
        ):
            raise ValueError("answerable-zh cases must request Chinese")
        if (
            self.category is EvaluationCategory.ANSWERABLE_ENGLISH
            and self.language is not EvaluationLanguage.ENGLISH
        ):
            raise ValueError("answerable-en cases must request English")
        if (
            self.category is EvaluationCategory.PII
            and StyleExpectation.PII_REDACTED not in self.style_expectations
        ):
            raise ValueError("PII cases must require redaction")
        return self


class CorpusReference(DomainModel):
    snapshot_id: Identifier
    version: SemanticVersion
    content_hash: Sha256Digest
    manifest_file: str

    _safe_manifest_file = field_validator("manifest_file")(_validate_relative_path)


class DatasetManifest(DomainModel):
    schema_version: Literal["rag-evaluation-dataset-v1"] = "rag-evaluation-dataset-v1"
    dataset_id: Identifier
    version: SemanticVersion
    content_hash: Sha256Digest
    hash_algorithm: Literal["sha256-canonical-json-v1"] = "sha256-canonical-json-v1"
    cases_file: str = "cases.jsonl"
    case_count: Annotated[int, Field(gt=0)]
    corpus: CorpusReference
    required_categories: Annotated[tuple[EvaluationCategory, ...], Field(min_length=1)]
    required_metrics: Annotated[tuple[EvaluationMetric, ...], Field(min_length=1)]

    _safe_cases_file = field_validator("cases_file")(_validate_relative_path)

    @model_validator(mode="after")
    def declared_requirements_are_unique(self) -> DatasetManifest:
        _require_unique(self.required_categories, label="required categories")
        _require_unique(self.required_metrics, label="required metrics")
        return self


class CorpusDocument(DomainModel):
    source_id: Identifier
    source_key: Identifier
    display_title: Identifier
    document_version: Annotated[int, Field(gt=0)]
    source_path: str
    media_type: Identifier
    kind: DocumentKind
    snapshot_format: CorpusSnapshotFormat = CorpusSnapshotFormat.SOURCE
    derivation_artifact_path: str | None = None
    derivation_artifact_hash: Sha256Digest | None = None
    language: EvaluationLanguage
    extraction_method: ExtractionMethod
    content_hash: Sha256Digest

    _safe_source_path = field_validator("source_path")(_validate_relative_path)
    _safe_derivation_artifact_path = field_validator("derivation_artifact_path")(
        _validate_optional_relative_path
    )

    @model_validator(mode="after")
    def validate_snapshot_format(self) -> CorpusDocument:
        if self.snapshot_format is CorpusSnapshotFormat.SOURCE:
            if self.kind not in {DocumentKind.MARKDOWN, DocumentKind.TEXT}:
                raise ValueError("source snapshots currently support Markdown or UTF-8 text")
            if self.extraction_method is not ExtractionMethod.TEXT:
                raise ValueError("text source snapshots require text extraction")
            if (
                self.derivation_artifact_path is not None
                or self.derivation_artifact_hash is not None
            ):
                raise ValueError("source snapshots cannot declare a derivation artifact")
        elif (
            self.kind is not DocumentKind.PDF or self.extraction_method is not ExtractionMethod.OCR
        ):
            raise ValueError("frozen OCR pages require a PDF document and OCR extraction")
        elif self.derivation_artifact_path is None or self.derivation_artifact_hash is None:
            raise ValueError("frozen OCR pages require a hashed derivation artifact")
        return self


class CorpusChunk(DomainModel):
    chunk_id: Identifier
    source_id: Identifier
    document_version: Annotated[int, Field(gt=0)]
    ordinal: Annotated[int, Field(ge=0)]
    text: NonEmptyText
    content_digest: ChunkContentDigest
    locator: ChunkLocator
    language: EvaluationLanguage
    extraction_method: ExtractionMethod
    token_count: Annotated[int, Field(gt=0)]

    def to_domain_chunk(self) -> Chunk:
        """Materialize the exact chunk accepted by the production index stager."""

        return Chunk(
            chunk_id=self.chunk_id,
            source_id=self.source_id,
            document_version=self.document_version,
            ordinal=self.ordinal,
            text=self.text,
            content_digest=self.content_digest,
            locator=self.locator,
            token_count=self.token_count,
        )


class CorpusDerivation(DomainModel):
    extraction_version: Literal["extraction-v1"] = "extraction-v1"
    normalization_version: Literal["unicode-nfc-lines-v1"] = "unicode-nfc-lines-v1"
    chunking_version: Literal["structure-page-token-v1"] = "structure-page-token-v1"
    tokenizer_version: Literal["unicode-word-cjk-v1"] = "unicode-word-cjk-v1"
    target_tokens: Annotated[int, Field(gt=0)] = 500
    overlap_tokens: Annotated[int, Field(ge=0)] = 80

    @model_validator(mode="after")
    def validate_chunk_bounds(self) -> CorpusDerivation:
        if self.overlap_tokens >= self.target_tokens:
            raise ValueError("corpus overlap must be below the target token count")
        return self


class CorpusSnapshotManifest(DomainModel):
    schema_version: Literal["rag-evaluation-corpus-v1"] = "rag-evaluation-corpus-v1"
    snapshot_id: Identifier
    version: SemanticVersion
    content_hash: Sha256Digest
    hash_algorithm: Literal["sha256-canonical-json-v1"] = "sha256-canonical-json-v1"
    documents_file: str = "documents.jsonl"
    chunks_file: str = "chunks.jsonl"
    derivation: CorpusDerivation
    active_sources: dict[Identifier, Annotated[int, Field(gt=0)]]
    document_count: Annotated[int, Field(gt=0)]
    chunk_count: Annotated[int, Field(gt=0)]

    _safe_documents_file = field_validator("documents_file")(_validate_relative_path)
    _safe_chunks_file = field_validator("chunks_file")(_validate_relative_path)


class CorpusSnapshot(DomainModel):
    manifest: CorpusSnapshotManifest
    documents: tuple[CorpusDocument, ...]
    chunks: tuple[CorpusChunk, ...]

    @property
    def chunks_by_id(self) -> dict[str, CorpusChunk]:
        return {chunk.chunk_id: chunk for chunk in self.chunks}


class EvaluationDataset(DomainModel):
    root: Path
    manifest: DatasetManifest
    cases: tuple[EvaluationCase, ...]
    corpus: CorpusSnapshot
    category_counts: dict[EvaluationCategory, int]
    metric_eligibility_counts: dict[EvaluationMetric, int]

    def eligible_cases(self, metric: EvaluationMetric | str) -> tuple[EvaluationCase, ...]:
        resolved = EvaluationMetric(metric)
        return tuple(case for case in self.cases if _is_eligible(case, resolved))

    @property
    def production_chunks(self) -> tuple[Chunk, ...]:
        """Return the verified chunks for direct staging by ``RevisionStager``."""

        return tuple(chunk.to_domain_chunk() for chunk in self.corpus.chunks)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _canonical_text(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value.lstrip("\ufeff"))
    return normalized.replace("\r\n", "\n").replace("\r", "\n")


def calculate_source_content_hash(path: Path, media_type: str) -> str:
    """Hash a source artifact without making Git newline behavior part of its identity."""

    if media_type.startswith("text/") or media_type in {
        "application/json",
        "application/markdown",
    }:
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeError as exc:
            raise DatasetValidationError("source artifact is not valid UTF-8 text") from exc
        return _sha256_bytes(_canonical_text(content).encode("utf-8"))
    return _sha256_bytes(path.read_bytes())


def calculate_chunk_content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def calculate_corpus_content_hash(
    manifest: CorpusSnapshotManifest,
    documents: tuple[CorpusDocument, ...],
    chunks: tuple[CorpusChunk, ...],
) -> str:
    payload = {
        "manifest": manifest.model_dump(mode="json", exclude={"content_hash"}),
        "documents": [document.model_dump(mode="json") for document in documents],
        "chunks": [chunk.model_dump(mode="json") for chunk in chunks],
    }
    return _sha256_bytes(_canonical_json_bytes(payload))


def calculate_dataset_content_hash(
    manifest: DatasetManifest,
    cases: tuple[EvaluationCase, ...],
) -> str:
    payload = {
        "manifest": manifest.model_dump(mode="json", exclude={"content_hash"}),
        "cases": [case.model_dump(mode="json") for case in cases],
    }
    return _sha256_bytes(_canonical_json_bytes(payload))


compute_corpus_content_hash = calculate_corpus_content_hash
compute_dataset_content_hash = calculate_dataset_content_hash


def _resolve_relative(root: Path, value: str) -> Path:
    parts = PurePosixPath(_validate_relative_path(value)).parts
    resolved_root = root.resolve()
    candidate = resolved_root.joinpath(*parts).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise DatasetValidationError("dataset path escapes its declared root") from exc
    return candidate


def _read_object(path: Path, model_type: type[DomainModel], *, label: str) -> DomainModel:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetValidationError(f"missing {label}") from exc
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(f"invalid {label}") from exc
    if not isinstance(payload, dict):
        raise DatasetValidationError(f"{label} must be a JSON object")
    try:
        return model_type.model_validate(payload)
    except ValidationError as exc:
        raise DatasetValidationError(f"invalid {label} schema") from exc


def _read_jsonl(
    path: Path,
    model_type: type[DomainModel],
    *,
    label: str,
) -> tuple[DomainModel, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise DatasetValidationError(f"missing {label}") from exc
    except UnicodeError as exc:
        raise DatasetValidationError(f"invalid UTF-8 in {label}") from exc
    if not lines:
        raise DatasetValidationError(f"{label} must not be empty")

    records: list[DomainModel] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise DatasetValidationError(f"blank line in {label} at line {line_number}")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetValidationError(f"invalid JSON in {label} at line {line_number}") from exc
        if not isinstance(payload, dict):
            raise DatasetValidationError(f"invalid record in {label} at line {line_number}")
        try:
            records.append(model_type.model_validate(payload))
        except ValidationError as exc:
            raise DatasetValidationError(
                f"invalid schema in {label} at line {line_number}"
            ) from exc
    return tuple(records)


def _is_eligible(case: EvaluationCase, metric: EvaluationMetric) -> bool:
    if metric in {
        EvaluationMetric.FAITHFULNESS,
        EvaluationMetric.CONTEXT_PRECISION,
        EvaluationMetric.ANSWER_COMPLETENESS,
    }:
        return (
            case.answerability is Answerability.ANSWERABLE
            and bool(case.expected_facts)
            and bool(case.authoritative_evidence_ids)
        )
    if metric is EvaluationMetric.STYLE_CONSISTENCY:
        return bool(case.style_expectations)
    return True


def _derive_extracted_document(
    corpus_root: Path,
    document: CorpusDocument,
) -> ExtractedDocument:
    try:
        if document.snapshot_format is CorpusSnapshotFormat.SOURCE:
            source_path = _resolve_relative(corpus_root, document.source_path)
            source_content = source_path.read_bytes()
            extracted = extract_utf8_text(source_content, kind=document.kind)
        else:
            assert document.derivation_artifact_path is not None
            artifact_path = _resolve_relative(corpus_root, document.derivation_artifact_path)
            ocr_text = artifact_path.read_text(encoding="utf-8-sig")
            if not ocr_text.strip():
                raise ValueError("empty frozen OCR page")
            extracted = ExtractedDocument(
                kind=DocumentKind.PDF,
                blocks=(
                    ExtractedBlock(
                        text=ocr_text,
                        page_number=1,
                        extraction_method=ExtractionMethod.OCR,
                    ),
                ),
                ocr_page_count=1,
            )
        return normalize_document(extracted)
    except Exception as exc:
        raise DatasetValidationError("corpus source derivation failed") from exc


def _derive_document_chunks(
    corpus_root: Path,
    document: CorpusDocument,
    derivation: CorpusDerivation,
) -> tuple[Chunk, ...]:
    try:
        normalized = _derive_extracted_document(corpus_root, document)
        return chunk_document(
            normalized,
            source_id=document.source_id,
            document_version=document.document_version,
            config=ChunkingConfig(
                target_tokens=derivation.target_tokens,
                overlap_tokens=derivation.overlap_tokens,
                version=derivation.chunking_version,
                tokenizer_version=derivation.tokenizer_version,
            ),
        )
    except Exception as exc:
        raise DatasetValidationError("corpus source derivation failed") from exc


def _validate_reproduced_chunks(
    corpus_root: Path,
    manifest: CorpusSnapshotManifest,
    documents: tuple[CorpusDocument, ...],
    chunks: tuple[CorpusChunk, ...],
) -> None:
    derivation = manifest.derivation
    if (
        derivation.extraction_version != INDEX_EXTRACTION_VERSION
        or derivation.normalization_version != NORMALIZATION_VERSION
        or derivation.chunking_version != CHUNKING_VERSION
        or derivation.tokenizer_version != TOKENIZER_VERSION
    ):
        raise DatasetValidationError("corpus derivation identity is not supported by this code")

    chunks_by_source: dict[str, list[CorpusChunk]] = {}
    for chunk in chunks:
        chunks_by_source.setdefault(chunk.source_id, []).append(chunk)

    for document in documents:
        declared = tuple(
            sorted(chunks_by_source.get(document.source_id, ()), key=lambda item: item.ordinal)
        )
        derived = _derive_document_chunks(corpus_root, document, derivation)
        if len(declared) != len(derived):
            raise DatasetValidationError("corpus chunk inventory is not reproducible")
        for declared_chunk, derived_chunk in zip(declared, derived, strict=True):
            expected_method = (
                ExtractionMethod.OCR
                if document.snapshot_format is CorpusSnapshotFormat.FROZEN_OCR_PAGE
                else ExtractionMethod.TEXT
            )
            if (
                declared_chunk.to_domain_chunk() != derived_chunk
                or declared_chunk.extraction_method is not expected_method
            ):
                raise DatasetValidationError("corpus chunk does not match production derivation")


def _validate_corpus(
    dataset_root: Path,
    reference: CorpusReference,
) -> CorpusSnapshot:
    manifest_path = _resolve_relative(dataset_root, reference.manifest_file)
    manifest_value = _read_object(
        manifest_path,
        CorpusSnapshotManifest,
        label="corpus manifest",
    )
    assert isinstance(manifest_value, CorpusSnapshotManifest)
    manifest = manifest_value
    if (
        manifest.snapshot_id != reference.snapshot_id
        or manifest.version != reference.version
        or manifest.content_hash != reference.content_hash
    ):
        raise DatasetValidationError("corpus reference does not match the corpus manifest")

    corpus_root = manifest_path.parent
    document_values = _read_jsonl(
        _resolve_relative(corpus_root, manifest.documents_file),
        CorpusDocument,
        label="corpus documents",
    )
    chunk_values = _read_jsonl(
        _resolve_relative(corpus_root, manifest.chunks_file),
        CorpusChunk,
        label="corpus chunks",
    )
    documents = tuple(value for value in document_values if isinstance(value, CorpusDocument))
    chunks = tuple(value for value in chunk_values if isinstance(value, CorpusChunk))
    if len(documents) != manifest.document_count or len(chunks) != manifest.chunk_count:
        raise DatasetValidationError("corpus record counts do not match its manifest")

    source_ids = tuple(document.source_id for document in documents)
    source_keys = tuple(document.source_key for document in documents)
    chunk_ids = tuple(chunk.chunk_id for chunk in chunks)
    try:
        _require_unique(source_ids, label="corpus source IDs")
        _require_unique(source_keys, label="corpus source keys")
        _require_unique(chunk_ids, label="corpus chunk IDs")
    except ValueError as exc:
        raise DatasetValidationError(str(exc)) from exc

    documents_by_id = {document.source_id: document for document in documents}
    active_sources = {document.source_id: document.document_version for document in documents}
    if manifest.active_sources != active_sources:
        raise DatasetValidationError("corpus active sources do not match its document inventory")
    for document in documents:
        source_path = _resolve_relative(corpus_root, document.source_path)
        if not source_path.is_file():
            raise DatasetValidationError("corpus source artifact is missing")
        if calculate_source_content_hash(source_path, document.media_type) != document.content_hash:
            raise DatasetValidationError("corpus source artifact hash mismatch")
        if document.derivation_artifact_path is not None:
            assert document.derivation_artifact_hash is not None
            derivation_path = _resolve_relative(corpus_root, document.derivation_artifact_path)
            if not derivation_path.is_file():
                raise DatasetValidationError("corpus derivation artifact is missing")
            if (
                calculate_source_content_hash(derivation_path, "text/plain")
                != document.derivation_artifact_hash
            ):
                raise DatasetValidationError("corpus derivation artifact hash mismatch")

    for chunk in chunks:
        referenced_document = documents_by_id.get(chunk.source_id)
        if (
            referenced_document is None
            or referenced_document.document_version != chunk.document_version
        ):
            raise DatasetValidationError("corpus chunk references an unknown document version")
        if calculate_chunk_content_hash(chunk.text) != chunk.content_digest:
            raise DatasetValidationError("corpus chunk content hash mismatch")

    _validate_reproduced_chunks(corpus_root, manifest, documents, chunks)

    calculated_hash = calculate_corpus_content_hash(manifest, documents, chunks)
    if calculated_hash != manifest.content_hash:
        raise DatasetValidationError("corpus snapshot content hash mismatch")
    return CorpusSnapshot(manifest=manifest, documents=documents, chunks=chunks)


def _validate_dataset_contract(
    manifest: DatasetManifest,
    cases: tuple[EvaluationCase, ...],
    corpus: CorpusSnapshot,
    *,
    acceptance_mode: bool,
) -> tuple[dict[EvaluationCategory, int], dict[EvaluationMetric, int]]:
    if len(cases) != manifest.case_count:
        raise DatasetValidationError("case count does not match the dataset manifest")
    case_ids = tuple(case.case_id for case in cases)
    try:
        _require_unique(case_ids, label="case IDs")
    except ValueError as exc:
        raise DatasetValidationError(str(exc)) from exc

    chunks_by_id = corpus.chunks_by_id
    for case in cases:
        missing = set(case.authoritative_evidence_ids) - chunks_by_id.keys()
        if missing:
            raise DatasetValidationError("case references unknown authoritative evidence")
        if case.category is EvaluationCategory.OCR and not any(
            chunks_by_id[evidence_id].extraction_method is ExtractionMethod.OCR
            for evidence_id in case.authoritative_evidence_ids
        ):
            raise DatasetValidationError("OCR cases require authoritative OCR evidence")

    category_counts = dict(Counter(case.category for case in cases))
    required_categories = set(manifest.required_categories)
    if acceptance_mode and not required_categories >= ACCEPTANCE_REQUIRED_CATEGORIES:
        raise DatasetValidationError("acceptance manifest omits required categories")
    categories_to_validate = required_categories
    if any(category_counts.get(category, 0) < 1 for category in categories_to_validate):
        raise DatasetValidationError("a required category has no eligible case")

    metric_counts = {
        metric: sum(_is_eligible(case, metric) for case in cases) for metric in EvaluationMetric
    }
    required_metrics = set(manifest.required_metrics)
    if acceptance_mode and not required_metrics >= ACCEPTANCE_REQUIRED_METRICS:
        raise DatasetValidationError("acceptance manifest omits required metrics")
    metrics_to_validate = required_metrics
    if any(metric_counts.get(metric, 0) < 1 for metric in metrics_to_validate):
        raise DatasetValidationError("a required metric has no eligible denominator")
    return category_counts, metric_counts


def load_dataset(
    path: str | Path,
    *,
    expected_corpus_version: str | None = None,
    acceptance_mode: bool = True,
) -> EvaluationDataset:
    """Load and fully validate an immutable dataset directory."""

    root = Path(path).resolve()
    if not root.is_dir():
        raise DatasetValidationError("dataset root does not exist")
    manifest_value = _read_object(root / "manifest.json", DatasetManifest, label="dataset manifest")
    assert isinstance(manifest_value, DatasetManifest)
    manifest = manifest_value
    if expected_corpus_version is not None and manifest.corpus.version != expected_corpus_version:
        raise DatasetValidationError("dataset corpus version does not match the requested corpus")

    case_values = _read_jsonl(
        _resolve_relative(root, manifest.cases_file),
        EvaluationCase,
        label="evaluation cases",
    )
    cases = tuple(value for value in case_values if isinstance(value, EvaluationCase))
    corpus = _validate_corpus(root, manifest.corpus)
    if calculate_dataset_content_hash(manifest, cases) != manifest.content_hash:
        raise DatasetValidationError("dataset content hash mismatch")
    category_counts, metric_counts = _validate_dataset_contract(
        manifest,
        cases,
        corpus,
        acceptance_mode=acceptance_mode,
    )
    return EvaluationDataset(
        root=root,
        manifest=manifest,
        cases=cases,
        corpus=corpus,
        category_counts=category_counts,
        metric_eligibility_counts=metric_counts,
    )


def validate_dataset(
    path: str | Path,
    *,
    expected_corpus_version: str | None = None,
    acceptance_mode: bool = True,
) -> EvaluationDataset:
    """Validate a dataset and return the trusted, immutable loaded value."""

    return load_dataset(
        path,
        expected_corpus_version=expected_corpus_version,
        acceptance_mode=acceptance_mode,
    )


def materialize_production_chunks(dataset: EvaluationDataset) -> tuple[Chunk, ...]:
    """Return loader-verified domain chunks for production index staging."""

    return dataset.production_chunks


def materialize_production_documents(
    dataset: EvaluationDataset,
) -> tuple[tuple[CorpusDocument, bytes, ExtractedDocument], ...]:
    """Materialize validated source bytes and normalized production extraction values."""

    corpus_manifest_path = _resolve_relative(
        dataset.root,
        dataset.manifest.corpus.manifest_file,
    )
    corpus_root = corpus_manifest_path.parent
    return tuple(
        (
            document,
            _resolve_relative(corpus_root, document.source_path).read_bytes(),
            _derive_extracted_document(corpus_root, document),
        )
        for document in dataset.corpus.documents
    )


__all__ = [
    "ACCEPTANCE_REQUIRED_CATEGORIES",
    "ACCEPTANCE_REQUIRED_METRICS",
    "Answerability",
    "ConversationTurn",
    "CorpusChunk",
    "CorpusDerivation",
    "CorpusDocument",
    "CorpusReference",
    "CorpusSnapshot",
    "CorpusSnapshotFormat",
    "CorpusSnapshotManifest",
    "DatasetManifest",
    "DatasetValidationError",
    "EvaluationCase",
    "EvaluationCategory",
    "EvaluationDataset",
    "EvaluationLanguage",
    "EvaluationMetric",
    "ExpectedFact",
    "StyleExpectation",
    "calculate_chunk_content_hash",
    "calculate_corpus_content_hash",
    "calculate_dataset_content_hash",
    "calculate_source_content_hash",
    "compute_corpus_content_hash",
    "compute_dataset_content_hash",
    "load_dataset",
    "materialize_production_chunks",
    "materialize_production_documents",
    "validate_dataset",
]
