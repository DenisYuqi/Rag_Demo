"""Knowledge ingestion pipeline."""

from rag_mvp.ingestion.chunking import ChunkingConfig, chunk_document
from rag_mvp.ingestion.extractors import (
    ExtractedBlock,
    ExtractedDocument,
    ExtractionError,
    PageUsabilityPolicy,
    TesseractOcrAdapter,
    extract_pdf,
    extract_utf8_text,
)
from rag_mvp.ingestion.normalization import (
    NORMALIZATION_VERSION,
    canonical_document_digest,
    normalize_document,
    normalize_text,
)
from rag_mvp.ingestion.validation import (
    UploadValidationError,
    ValidatedUpload,
    validate_upload,
)

__all__ = [
    "NORMALIZATION_VERSION",
    "ChunkingConfig",
    "ExtractedBlock",
    "ExtractedDocument",
    "ExtractionError",
    "PageUsabilityPolicy",
    "TesseractOcrAdapter",
    "UploadValidationError",
    "ValidatedUpload",
    "canonical_document_digest",
    "chunk_document",
    "extract_pdf",
    "extract_utf8_text",
    "normalize_document",
    "normalize_text",
    "validate_upload",
]
