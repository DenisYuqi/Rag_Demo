"""Offline verification CLI for a canonical JSON report and adjacent HTML."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import cast

from .html_report import HtmlReportError, load_html_report, verify_html_parity
from .json_report import (
    ReportSerializationError,
    ReportValidationError,
    canonical_report_document,
    load_json_report,
)


class ReportVerificationError(ValueError):
    """A safe offline verification failure that never includes report content."""


@dataclass(frozen=True, slots=True)
class VerificationResult:
    json_path: Path
    html_path: Path
    run_id: str
    schema_version: str
    content_hash: str
    final_gate_passed: bool

    @property
    def valid(self) -> bool:
        return True

    @property
    def html_verified(self) -> bool:
        return True


def verify_report(
    json_path: Path | str,
    html_path: Path | str | None = None,
) -> VerificationResult:
    """Verify schema, canonical JSON bytes, and exact adjacent-HTML parity."""

    source = Path(json_path).expanduser().resolve()
    if source.suffix.casefold() != ".json":
        raise ReportVerificationError("source report path must end in .json")
    companion = (
        source.with_suffix(".html") if html_path is None else Path(html_path).expanduser().resolve()
    )
    if companion.suffix.casefold() not in {".html", ".htm"}:
        raise ReportVerificationError("companion report path must be HTML")

    report = load_json_report(source)
    canonical = canonical_report_document(report)
    try:
        persisted = source.read_bytes()
    except OSError as error:
        raise ReportVerificationError("canonical JSON report is unavailable") from error
    if persisted != canonical:
        raise ReportVerificationError("JSON report is valid but not canonically serialized")

    html = load_html_report(companion)
    verify_html_parity(report, html)
    gate = cast(dict[str, object], report["gate"])
    return VerificationResult(
        json_path=source,
        html_path=companion,
        run_id=cast(str, report["run_id"]),
        schema_version=cast(str, report["schema_version"]),
        content_hash=f"sha256:{sha256(persisted).hexdigest()}",
        final_gate_passed=cast(bool, gate["final_passed"]),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m rag_mvp.evaluation.verify_report",
        description="Verify one canonical evaluation JSON report and its adjacent HTML report.",
    )
    parser.add_argument("report_json", type=Path, help="path to the source-of-truth .json report")
    parser.add_argument(
        "--html",
        type=Path,
        default=None,
        help="optional HTML path; defaults to the JSON path with a .html suffix",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = verify_report(arguments.report_json, arguments.html)
    except (
        HtmlReportError,
        OSError,
        ReportSerializationError,
        ReportValidationError,
        ReportVerificationError,
        TypeError,
        UnicodeError,
        ValueError,
    ) as error:
        print(f"report verification failed: {error}", file=sys.stderr)
        return 1
    gate = "pass" if result.final_gate_passed else "fail"
    print(
        "report verified: "
        f"run_id={result.run_id} schema={result.schema_version} "
        f"content_hash={result.content_hash} final_gate={gate}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the module CLI
    raise SystemExit(main())
