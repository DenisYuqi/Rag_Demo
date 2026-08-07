"""Isolated comparison workspaces and full immutable-index reuse identities."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator

from rag_mvp.domain._base import DomainModel, Identifier, SafeScalar
from rag_mvp.domain.ingestion import IndexRevision, IndexRevisionStatus
from rag_mvp.evaluation.dataset import EvaluationDataset
from rag_mvp.evaluation.runner import EvaluationRunPlan
from rag_mvp.retrieval.snapshot import (
    CHUNK_SET_DIGEST_ALGORITHM,
    RECORD_DIGEST_ALGORITHM,
    chunk_record_digest,
    chunk_set_digest,
    parent_chunk_record_digest,
    parent_set_digest,
)
from rag_mvp.retrieval.tokenizer import BILINGUAL_TOKENIZER_IDENTITY

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,254}$")


class EvaluationEnvironmentError(RuntimeError):
    """A content-free isolation or reuse-integrity failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class EvaluationReuseIdentityEntry(DomainModel):
    name: Identifier
    value: SafeScalar


class EvaluationSourceVersion(DomainModel):
    source_id: Identifier
    version: Annotated[int, Field(gt=0)]


class EvaluationIndexReuseKey(DomainModel):
    """Every corpus/chunking/embedding field required for safe index reuse."""

    schema_version: str = "evaluation-index-reuse-v2"
    corpus_version: Identifier
    corpus_hash: Identifier
    active_sources: tuple[EvaluationSourceVersion, ...]
    chunk_set_digest: Identifier
    chunk_set_digest_algorithm: str = CHUNK_SET_DIGEST_ALGORITHM
    record_digest_algorithm: str = RECORD_DIGEST_ALGORITHM
    chunk_count: Annotated[int, Field(gt=0)]
    parent_chunk_set_digest: Identifier
    parent_chunk_count: Annotated[int, Field(gt=0)]
    embedding_identity: tuple[EvaluationReuseIdentityEntry, ...]
    chunking_identity: tuple[EvaluationReuseIdentityEntry, ...]
    index_identity: tuple[EvaluationReuseIdentityEntry, ...]

    @field_validator("active_sources")
    @classmethod
    def sources_are_canonical(
        cls,
        value: tuple[EvaluationSourceVersion, ...],
    ) -> tuple[EvaluationSourceVersion, ...]:
        if not value or len({item.source_id for item in value}) != len(value):
            raise ValueError("evaluation reuse sources are empty or duplicate")
        return tuple(sorted(value, key=lambda item: item.source_id))

    @field_validator("embedding_identity", "chunking_identity", "index_identity")
    @classmethod
    def entries_are_canonical(
        cls,
        value: tuple[EvaluationReuseIdentityEntry, ...],
    ) -> tuple[EvaluationReuseIdentityEntry, ...]:
        if not value or len({item.name for item in value}) != len(value):
            raise ValueError("evaluation reuse identity is empty or duplicate")
        return tuple(sorted(value, key=lambda item: item.name))

    @property
    def digest(self) -> str:
        payload = self.model_dump(mode="json")
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"

    @classmethod
    def from_plan(
        cls,
        plan: EvaluationRunPlan,
        dataset: EvaluationDataset,
    ) -> EvaluationIndexReuseKey:
        if (
            plan.identity.dataset_id != dataset.manifest.dataset_id
            or plan.identity.dataset_version != dataset.manifest.version
            or plan.identity.dataset_hash != dataset.manifest.content_hash
            or plan.identity.corpus_version != dataset.corpus.manifest.version
            or plan.identity.corpus_hash != dataset.corpus.manifest.content_hash
        ):
            raise EvaluationEnvironmentError("evaluation_reuse_dataset_identity_mismatch")
        titles = {
            document.source_id: document.display_title for document in dataset.corpus.documents
        }
        try:
            records = {
                chunk.chunk_id: chunk_record_digest(chunk, titles[chunk.source_id])
                for chunk in dataset.production_chunks
            }
        except KeyError:
            raise EvaluationEnvironmentError("evaluation_reuse_chunk_source_missing") from None
        parent_records = {
            parent.parent_chunk_id: parent_chunk_record_digest(parent)
            for parent in dataset.production_parents
        }
        if not parent_records:
            raise EvaluationEnvironmentError("evaluation_reuse_parent_inventory_missing")
        return cls(
            corpus_version=dataset.corpus.manifest.version,
            corpus_hash=dataset.corpus.manifest.content_hash,
            active_sources=tuple(
                EvaluationSourceVersion(source_id=source_id, version=version)
                for source_id, version in dataset.corpus.manifest.active_sources.items()
            ),
            chunk_set_digest=chunk_set_digest(records),
            chunk_count=len(records),
            parent_chunk_set_digest=parent_set_digest(parent_records),
            parent_chunk_count=len(parent_records),
            embedding_identity=_entries(plan.identity.embedding_identity),
            chunking_identity=_entries(plan.identity.chunking_identity),
            index_identity=_entries(
                {
                    "dense_metric": "cosine",
                    "dense_schema_version": "chroma-revision-v2",
                    "lexical_algorithm_version": "bm25-okapi-v1",
                    "lexical_b": 0.75,
                    "lexical_k1": 1.5,
                    "lexical_schema_version": "bm25-snapshot-v4",
                    "lexical_tokenizer_identity": BILINGUAL_TOKENIZER_IDENTITY,
                }
            ),
        )

    def verify_revision(self, revision: IndexRevision) -> None:
        """Reject reuse unless the active immutable revision matches every index seam."""

        embedding = _entry_mapping(self.embedding_identity)
        chunking = _entry_mapping(self.chunking_identity)
        index = _entry_mapping(self.index_identity)
        expected_sources = {item.source_id: item.version for item in self.active_sources}
        actual_embedding = revision.embedding_space
        if (
            revision.status is not IndexRevisionStatus.ACTIVE
            or revision.published_at is None
            or revision.active_sources != expected_sources
            or revision.chunk_set_digest != self.chunk_set_digest
            or revision.chunk_count != self.chunk_count
            or revision.parent_chunk_set_digest != self.parent_chunk_set_digest
            or revision.parent_chunk_count != self.parent_chunk_count
            or revision.extraction_version != chunking.get("extraction_version")
            or revision.chunking_version != chunking.get("chunking_version")
            or revision.tokenizer_version != index.get("lexical_tokenizer_identity")
            or actual_embedding.provider_alias != embedding.get("provider")
            or actual_embedding.model != embedding.get("model")
            or actual_embedding.dimension != embedding.get("dimension")
            or actual_embedding.normalization != embedding.get("normalization")
            or actual_embedding.adapter_version != embedding.get("adapter_version")
            or revision.dense_schema_version != index.get("dense_schema_version")
            or revision.dense_metric != index.get("dense_metric")
            or revision.lexical_schema_version != index.get("lexical_schema_version")
            or revision.lexical_algorithm_version != index.get("lexical_algorithm_version")
            or revision.lexical_k1 != index.get("lexical_k1")
            or revision.lexical_b != index.get("lexical_b")
            or revision.record_digest_algorithm != self.record_digest_algorithm
        ):
            raise EvaluationEnvironmentError("evaluation_index_reuse_identity_mismatch")


@dataclass(frozen=True, slots=True)
class EvaluationSuiteWorkspaceManager:
    """Derive server-owned roots while rejecting ancestor symlink escapes."""

    online_data_root: Path

    def workspace_for(
        self,
        suite_id: str,
        reuse_key: EvaluationIndexReuseKey,
    ) -> Path:
        if not isinstance(suite_id, str) or _SAFE_ID.fullmatch(suite_id) is None:
            raise EvaluationEnvironmentError("evaluation_suite_id_invalid")
        if not isinstance(reuse_key, EvaluationIndexReuseKey):
            raise EvaluationEnvironmentError("evaluation_index_reuse_key_invalid")
        online_root = self.online_data_root.expanduser().resolve()
        suite_parent = (online_root / "evaluations" / "suites").resolve()
        suite_root = (suite_parent / suite_id).resolve()
        indexes_root = (suite_root / "indexes").resolve()
        workspace = (indexes_root / reuse_key.digest.removeprefix("sha256:")[:32]).resolve()
        online_index_root = (online_root / "indexes").resolve()
        if (
            not suite_parent.is_relative_to(online_root)
            or not suite_root.is_relative_to(suite_parent)
            or not indexes_root.is_relative_to(suite_root)
            or not workspace.is_relative_to(indexes_root)
            or workspace == online_root
            or workspace.is_relative_to(online_index_root)
        ):
            raise EvaluationEnvironmentError("evaluation_suite_workspace_unsafe")
        return workspace


def _entries(values: dict[str, SafeScalar]) -> tuple[EvaluationReuseIdentityEntry, ...]:
    if not values:
        raise EvaluationEnvironmentError("evaluation_reuse_identity_missing")
    return tuple(
        EvaluationReuseIdentityEntry(name=name, value=value)
        for name, value in sorted(values.items())
    )


def _entry_mapping(
    values: tuple[EvaluationReuseIdentityEntry, ...],
) -> dict[str, SafeScalar]:
    return {item.name: item.value for item in values}


__all__ = [
    "EvaluationEnvironmentError",
    "EvaluationIndexReuseKey",
    "EvaluationReuseIdentityEntry",
    "EvaluationSourceVersion",
    "EvaluationSuiteWorkspaceManager",
]
