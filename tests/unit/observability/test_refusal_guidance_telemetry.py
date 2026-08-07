from __future__ import annotations

from pathlib import Path

from rag_mvp.config.settings import Settings
from rag_mvp.domain.qa import (
    RefusalReason,
    SafeQADiagnostics,
    StreamEventKind,
    ValidatedStreamEvent,
)
from rag_mvp.observability.runtime import _diagnostic_metadata


def test_runtime_telemetry_projects_only_safe_refusal_guidance_identity(
    tmp_path: Path,
) -> None:
    pii_fixture = "person@example.com"
    event = ValidatedStreamEvent(
        request_id="request-guided-refusal",
        session_id="session-guided-refusal",
        sequence=0,
        kind=StreamEventKind.REFUSAL,
        response_language="en",
        content="I cannot complete this request safely. Please ask about available documents.",
        reason=RefusalReason.PROMPT_INJECTION,
        diagnostics=SafeQADiagnostics(
            metadata={
                "refusal_reason_code": "prompt-injection",
                "refusal_guidance_reason_code": "prompt-injection",
                "refusal_guidance_template_id": ("refusal-guidance-v1.prompt-injection.en"),
                "refusal_guidance_catalog_version": "refusal-guidance-v1",
                "refusal_guidance_present": True,
                "refusal_guidance_language": "en",
                "input_policy": "override_policy",
                "question": f"raw trigger for {pii_fixture}",
            }
        ),
        terminal=True,
    )
    settings = Settings(data_root=tmp_path / "data", _env_file=None)

    metadata = _diagnostic_metadata(event, settings)

    assert metadata == {
        "configuration_id": settings.configuration_identity,
        "citation_count": 0,
        "refusal_reason_code": "prompt-injection",
        "refusal_guidance_reason_code": "prompt-injection",
        "refusal_guidance_template_id": "refusal-guidance-v1.prompt-injection.en",
        "refusal_guidance_catalog_version": "refusal-guidance-v1",
        "refusal_guidance_language": "en",
        "refusal_guidance_present": True,
    }
    assert pii_fixture not in repr(metadata)
    assert "input_policy" not in metadata
