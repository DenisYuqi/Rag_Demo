"""Standalone HTML rendering with exact schema-v2 JSON parity."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape
from pydantic import BaseModel

from rag_mvp.safety.redactor import DEFAULT_REDACTOR, Redactor

from .html_report import (
    HtmlReportParityError,
    _display_value,
    _parse_html,
    _pointer_token,
    _resolve_pointer,
    load_html_report,
)
from .json_report import (
    JsonObject,
    ReportSerializationError,
    ReportWriteError,
    _atomic_write_text,
    canonical_json_value,
    decode_json_report,
)
from .report_v2 import (
    canonical_report_json_v2,
    prepare_report_v2,
    validate_report_v2,
)

REPORT_TEMPLATE_NAME_V2 = "evaluation-report-v2.html.j2"
_CANONICAL_SECTIONS_V2 = (
    "provenance",
    "acceptance_contract",
    "gates",
    "performance_evidence",
    "operations_summary",
    "category_results",
    "failed_cases",
    "artifacts",
    "limitations",
)
_REQUIRED_VISIBLE_POINTERS_V2 = frozenset(
    {
        "/run_id",
        "/schema_version",
        "/generated_at",
        "/configuration_id",
        "/status",
        "/accepted",
        *{f"/{section}" for section in _CANONICAL_SECTIONS_V2},
    }
)


def render_html_report_v2(
    report: Mapping[str, object] | BaseModel,
    *,
    redactor: Redactor = DEFAULT_REDACTOR,
) -> str:
    """Render a privacy-filtered standalone HTML projection of schema v2."""

    prepared = prepare_report_v2(report, redactor=redactor)
    canonical = canonical_report_json_v2(prepared)
    template = _template_environment_v2().get_template(REPORT_TEMPLATE_NAME_V2)
    rendered = template.render(
        report=prepared,
        canonical_json=canonical,
        canonical_sections=_CANONICAL_SECTIONS_V2,
    )
    _verify_prepared_html_parity_v2(prepared, rendered)
    return rendered


def write_html_report_v2(
    report: Mapping[str, object] | BaseModel,
    output_path: Path | str,
    *,
    overwrite: bool = False,
    redactor: Redactor = DEFAULT_REDACTOR,
) -> Path:
    """Atomically publish a v2 HTML report without replacing by default."""

    target = Path(output_path)
    if target.suffix.casefold() not in {".html", ".htm"}:
        raise ReportWriteError("schema-v2 HTML report path must end in .html or .htm")
    rendered = render_html_report_v2(report, redactor=redactor)
    return _atomic_write_text(target, rendered + "\n", overwrite=overwrite)


def extract_embedded_report_v2(html: str) -> JsonObject:
    """Extract and validate the canonical schema-v2 JSON embedded in HTML."""

    parser = _parse_html(html)
    try:
        raw = decode_json_report(parser.embedded_json)
    except (ReportSerializationError, ValueError) as error:
        raise HtmlReportParityError("HTML contains invalid embedded schema-v2 JSON") from error
    if not isinstance(raw, dict):
        raise HtmlReportParityError("HTML embedded schema-v2 report is not an object")
    try:
        return validate_report_v2(cast(Mapping[str, object], raw))
    except (ReportSerializationError, ValueError) as error:
        raise HtmlReportParityError("HTML embedded report does not satisfy schema v2") from error


def verify_html_parity_v2(
    report: Mapping[str, object] | BaseModel,
    html: str,
    *,
    redactor: Redactor = DEFAULT_REDACTOR,
) -> None:
    """Verify exact embedded bytes and every marked visible v2 value."""

    prepared = prepare_report_v2(report, redactor=redactor)
    _verify_prepared_html_parity_v2(prepared, html)


def load_html_report_v2(path: Path | str) -> str:
    """Read a bounded UTF-8 HTML artifact without mutating it."""

    return load_html_report(path)


def _verify_prepared_html_parity_v2(report: JsonObject, html: str) -> None:
    parser = _parse_html(html)
    canonical = canonical_report_json_v2(report)
    if parser.embedded_json != canonical:
        raise HtmlReportParityError(
            "HTML embedded schema-v2 JSON differs from the canonical source"
        )
    embedded = extract_embedded_report_v2(html)
    if embedded != report:
        raise HtmlReportParityError("HTML embedded values differ from the schema-v2 source")

    missing = _REQUIRED_VISIBLE_POINTERS_V2.difference(parser.markers)
    if missing:
        raise HtmlReportParityError("HTML is missing required visible schema-v2 values")
    for pointer, rendered_value in parser.markers.items():
        try:
            source_value = _resolve_pointer(report, pointer)
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise HtmlReportParityError("HTML references an unknown schema-v2 value") from error
        if rendered_value != _display_value(source_value):
            raise HtmlReportParityError(
                "a visible HTML value differs from the schema-v2 JSON source"
            )


def _template_environment_v2() -> Environment:
    environment = Environment(
        loader=PackageLoader("rag_mvp.evaluation", "templates"),
        autoescape=select_autoescape(enabled_extensions=("html", "j2"), default=True),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    environment.filters["canonical_value"] = canonical_json_value
    environment.filters["display_value"] = _display_value
    environment.filters["pointer_token"] = _pointer_token
    return environment


class HtmlReportRendererV2:
    """Injectable schema-v2 renderer for bounded report workers."""

    def __init__(self, *, redactor: Redactor = DEFAULT_REDACTOR) -> None:
        self._redactor = redactor

    def render(self, report: Mapping[str, object] | BaseModel) -> str:
        return render_html_report_v2(report, redactor=self._redactor)

    def generate(
        self,
        report: Mapping[str, object] | BaseModel,
        output_path: Path | str,
        *,
        overwrite: bool = False,
    ) -> Path:
        return write_html_report_v2(
            report,
            output_path,
            overwrite=overwrite,
            redactor=self._redactor,
        )
