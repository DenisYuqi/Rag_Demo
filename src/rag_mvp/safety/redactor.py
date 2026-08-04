"""Sensitive-value redaction with deterministic overlap handling."""

from __future__ import annotations

from collections.abc import Iterable

from rag_mvp.safety.detectors import DEFAULT_DETECTORS
from rag_mvp.safety.models import DetectionSpan, Detector, RedactionResult


class RedactionError(RuntimeError):
    """Raised when content cannot be deterministically and safely redacted."""


def resolve_overlaps(spans: Iterable[DetectionSpan]) -> tuple[DetectionSpan, ...]:
    """Resolve overlaps into safe unions with deterministic category precedence.

    All intersecting spans are replaced as one union so that selecting a
    higher-priority detector can never leave a lower-priority sensitive suffix
    visible.  The union's placeholder comes from the highest priority span,
    then the longest span, then stable category/detector ordering.
    """

    ordered = sorted(spans, key=lambda span: (span.start, span.end))
    if not ordered:
        return ()

    clusters: list[list[DetectionSpan]] = []
    current = [ordered[0]]
    cluster_end = ordered[0].end
    for span in ordered[1:]:
        if span.start < cluster_end:
            current.append(span)
            cluster_end = max(cluster_end, span.end)
            continue
        clusters.append(current)
        current = [span]
        cluster_end = span.end
    clusters.append(current)

    resolved: list[DetectionSpan] = []
    for cluster in clusters:
        winner = min(
            cluster,
            key=lambda span: (
                -span.priority,
                -span.length,
                span.kind.value,
                span.detector,
                span.start,
            ),
        )
        resolved.append(
            DetectionSpan(
                start=min(span.start for span in cluster),
                end=max(span.end for span in cluster),
                kind=winner.kind,
                detector=winner.detector,
                priority=winner.priority,
            )
        )
    return tuple(resolved)


class Redactor:
    """Run a registry of detectors and replace complete sensitive spans."""

    def __init__(self, detectors: Iterable[Detector] | None = None) -> None:
        self._detectors = tuple(DEFAULT_DETECTORS if detectors is None else detectors)

    @property
    def initialized(self) -> bool:
        """Whether at least one detector rule is installed."""

        return bool(self._detectors)

    @property
    def detectors(self) -> tuple[Detector, ...]:
        """Installed detectors in deterministic execution order."""

        return self._detectors

    def detect(self, text: str) -> tuple[DetectionSpan, ...]:
        """Detect and resolve sensitive spans without retaining matched values."""

        if not isinstance(text, str):
            raise RedactionError("redaction input must be text")
        if not self.initialized:
            raise RedactionError("redactor is not initialized")

        candidates: list[DetectionSpan] = []
        try:
            for detector in self._detectors:
                for span in detector.detect(text):
                    if span.end > len(text):
                        raise RedactionError("detector returned an out-of-bounds span")
                    candidates.append(span)
        except RedactionError:
            raise
        except Exception as error:
            raise RedactionError("a detector failed") from error
        return resolve_overlaps(candidates)

    def redact(self, text: str) -> RedactionResult:
        """Return redacted text and content-free detection metadata."""

        spans = self.detect(text)
        if not spans:
            return RedactionResult(text, (), len(text))

        parts: list[str] = []
        cursor = 0
        for span in spans:
            parts.append(text[cursor : span.start])
            parts.append(span.placeholder)
            cursor = span.end
        parts.append(text[cursor:])
        return RedactionResult("".join(parts), spans, len(text))


DEFAULT_REDACTOR = Redactor()


def redact_text(text: str, *, redactor: Redactor = DEFAULT_REDACTOR) -> str:
    """Redact a string using the default detector registry."""

    return redactor.redact(text).redacted_text
