"""Knowledge ingestion pipeline."""

from rag_mvp.ingestion.chunking import ChunkingConfig, chunk_document
from rag_mvp.ingestion.embedding import (
    EmbeddingStage,
    EmbeddingStageError,
    EmbeddingStageResult,
)
from rag_mvp.ingestion.extractors import (
    ExtractedBlock,
    ExtractedDocument,
    ExtractionError,
    PageUsabilityPolicy,
    TesseractOcrAdapter,
    extract_pdf,
    extract_utf8_text,
)
from rag_mvp.ingestion.indexing import (
    INDEX_EXTRACTION_VERSION,
    IndexingError,
    RevisionPublisher,
    RevisionStager,
)
from rag_mvp.ingestion.normalization import (
    NORMALIZATION_VERSION,
    canonical_document_digest,
    normalize_document,
    normalize_text,
)
from rag_mvp.ingestion.service import (
    IngestionRecoveryError,
    IngestionService,
    IngestionSubmissionError,
    RecoveryReport,
)
from rag_mvp.ingestion.validation import (
    UploadValidationError,
    ValidatedUpload,
    validate_upload,
)
from rag_mvp.ingestion.versioning import (
    DeletedSourceError,
    SourceVersionDisposition,
    SourceVersioningService,
    SourceVersionRegistration,
    derivation_config_digest,
)

__all__ = [
    "INDEX_EXTRACTION_VERSION",
    "NORMALIZATION_VERSION",
    "ChunkingConfig",
    "DeletedSourceError",
    "EmbeddingStage",
    "EmbeddingStageError",
    "EmbeddingStageResult",
    "ExtractedBlock",
    "ExtractedDocument",
    "ExtractionError",
    "IndexingError",
    "IngestionRecoveryError",
    "IngestionService",
    "IngestionSubmissionError",
    "PageUsabilityPolicy",
    "RecoveryReport",
    "RevisionPublisher",
    "RevisionStager",
    "SourceVersionDisposition",
    "SourceVersionRegistration",
    "SourceVersioningService",
    "TesseractOcrAdapter",
    "UploadValidationError",
    "ValidatedUpload",
    "canonical_document_digest",
    "chunk_document",
    "derivation_config_digest",
    "extract_pdf",
    "extract_utf8_text",
    "normalize_document",
    "normalize_text",
    "validate_upload",
]
