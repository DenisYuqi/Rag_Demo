from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from rag_mvp.domain.ingestion import DocumentKind, ExtractionMethod
from rag_mvp.evaluation.dataset import (
    ACCEPTANCE_REQUIRED_CATEGORIES,
    ACCEPTANCE_REQUIRED_METRICS,
    CorpusSnapshotFormat,
    DatasetManifest,
    DatasetValidationError,
    EvaluationCase,
    EvaluationCategory,
    EvaluationMetric,
    calculate_dataset_content_hash,
    load_dataset,
    materialize_production_chunks,
    materialize_production_documents,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATASET_ROOT = REPOSITORY_ROOT / "evaluations" / "datasets" / "mvp-v1"


def _copy_dataset(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    copied = tmp_path / "mvp-v1"
    shutil.copytree(DATASET_ROOT, copied)
    return copied


def _load_case_payloads(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in (root / "cases.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def _write_case_payloads(root: Path, payloads: list[dict[str, Any]]) -> None:
    serialized = "\n".join(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) for payload in payloads
    )
    (root / "cases.jsonl").write_text(f"{serialized}\n", encoding="utf-8")


def _refresh_dataset_hash(root: Path) -> None:
    path = root / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest = DatasetManifest.model_validate(payload)
    cases = tuple(EvaluationCase.model_validate(item) for item in _load_case_payloads(root))
    payload["content_hash"] = calculate_dataset_content_hash(manifest, cases)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def test_loads_versioned_bilingual_dataset_with_all_denominators() -> None:
    dataset = load_dataset(DATASET_ROOT, expected_corpus_version="1.0.0")

    assert dataset.manifest.dataset_id == "mvp-bilingual-rag"
    assert dataset.manifest.version == "1.0.0"
    assert len(dataset.cases) == 8
    assert set(dataset.category_counts) == ACCEPTANCE_REQUIRED_CATEGORIES
    assert all(dataset.category_counts[category] >= 1 for category in EvaluationCategory)
    assert set(dataset.metric_eligibility_counts) == ACCEPTANCE_REQUIRED_METRICS
    assert all(dataset.metric_eligibility_counts[metric] >= 1 for metric in EvaluationMetric)

    evidence_ids = {
        evidence_id for case in dataset.cases for evidence_id in case.authoritative_evidence_ids
    }
    assert evidence_ids <= dataset.corpus.chunks_by_id.keys()


def test_corpus_materializes_exact_production_chunks_and_documents() -> None:
    dataset = load_dataset(DATASET_ROOT)

    production_chunks = materialize_production_chunks(dataset)
    assert production_chunks == tuple(chunk.to_domain_chunk() for chunk in dataset.corpus.chunks)
    assert all(chunk.chunk_id.startswith("chk_") for chunk in production_chunks)
    assert dataset.corpus.manifest.active_sources == {
        document.source_id: document.document_version for document in dataset.corpus.documents
    }

    documents = materialize_production_documents(dataset)
    assert len(documents) == dataset.corpus.manifest.document_count
    ocr_metadata, ocr_source, ocr_document = next(
        item
        for item in documents
        if item[0].snapshot_format is CorpusSnapshotFormat.FROZEN_OCR_PAGE
    )
    assert ocr_metadata.kind is DocumentKind.PDF
    assert ocr_metadata.extraction_method is ExtractionMethod.OCR
    assert ocr_source.startswith(b"%PDF-1.4")
    assert ocr_document.ocr_page_count == 1
    assert ocr_document.blocks[0].page_number == 1
    assert ocr_document.blocks[0].extraction_method is ExtractionMethod.OCR


def test_rejects_case_content_tampering_without_a_new_version(tmp_path: Path) -> None:
    copied = _copy_dataset(tmp_path)
    payloads = _load_case_payloads(copied)
    payloads[0]["question"] = "This silently changed question must invalidate the fixture."
    _write_case_payloads(copied, payloads)

    with pytest.raises(DatasetValidationError, match="dataset content hash mismatch"):
        load_dataset(copied)


def test_rejects_duplicate_case_ids_even_when_hash_is_recomputed(tmp_path: Path) -> None:
    copied = _copy_dataset(tmp_path)
    payloads = _load_case_payloads(copied)
    payloads[-1]["case_id"] = payloads[0]["case_id"]
    _write_case_payloads(copied, payloads)
    _refresh_dataset_hash(copied)

    with pytest.raises(DatasetValidationError, match="case IDs must be unique"):
        load_dataset(copied)


def test_acceptance_manifest_cannot_omit_a_required_category(tmp_path: Path) -> None:
    copied = _copy_dataset(tmp_path)
    path = copied / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["required_categories"].remove(EvaluationCategory.PII.value)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _refresh_dataset_hash(copied)

    with pytest.raises(DatasetValidationError, match="omits required categories"):
        load_dataset(copied)


def test_rejects_unknown_authoritative_evidence_with_a_valid_hash(tmp_path: Path) -> None:
    copied = _copy_dataset(tmp_path)
    payloads = _load_case_payloads(copied)
    payloads[0]["expected_facts"][0]["evidence_ids"] = ["chk_missing"]
    payloads[0]["authoritative_evidence_ids"] = ["chk_missing"]
    _write_case_payloads(copied, payloads)
    _refresh_dataset_hash(copied)

    with pytest.raises(DatasetValidationError, match="unknown authoritative evidence"):
        load_dataset(copied)


def test_rejects_corpus_version_mismatch() -> None:
    with pytest.raises(DatasetValidationError, match="corpus version"):
        load_dataset(DATASET_ROOT, expected_corpus_version="2.0.0")


def test_rejects_changed_source_and_frozen_ocr_artifacts(tmp_path: Path) -> None:
    copied = _copy_dataset(tmp_path)
    source = copied / "corpus" / "sources" / "benefits-policy-en.md"
    source.write_text(source.read_text(encoding="utf-8") + "\nChanged.\n", encoding="utf-8")
    with pytest.raises(DatasetValidationError, match="source artifact hash mismatch"):
        load_dataset(copied)

    copied = _copy_dataset(tmp_path / "ocr")
    transcript = copied / "corpus" / "sources" / "scanned-expense-notice-ocr.txt"
    transcript.write_text(
        transcript.read_text(encoding="utf-8").replace("OCR-7421", "OCR-0000"),
        encoding="utf-8",
    )
    with pytest.raises(DatasetValidationError, match="derivation artifact hash mismatch"):
        load_dataset(copied)
