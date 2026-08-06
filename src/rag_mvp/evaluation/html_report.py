"""Human-readable HTML rendering with exact JSON value parity."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import cast

from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape
from pydantic import BaseModel

from rag_mvp.safety.redactor import DEFAULT_REDACTOR, Redactor

from .json_report import (
    MAX_REPORT_BYTES,
    JsonObject,
    JsonValue,
    ReportSerializationError,
    ReportWriteError,
    _atomic_write_text,
    canonical_json_value,
    canonical_report_json,
    decode_json_report,
    prepare_report,
    validate_report,
)

REPORT_TEMPLATE_NAME = "evaluation-report-v1.html.j2"
MAX_HTML_REPORT_BYTES = MAX_REPORT_BYTES * 2
_CANONICAL_SECTIONS = (
    "provenance",
    "configuration",
    "thresholds",
    "metrics",
    "failed_cases",
    "performance",
    "cost",
    "privacy",
    "issues",
    "gate",
)
_REQUIRED_VISIBLE_POINTERS = frozenset(
    {
        "/run_id",
        "/schema_version",
        "/generated_at",
        "/gate/final_passed",
        *{f"/{section}" for section in _CANONICAL_SECTIONS},
    }
)


class HtmlReportError(ValueError):
    """Base error for malformed or unsafe HTML report artifacts."""


class HtmlReportParityError(HtmlReportError):
    """Raised when an HTML value cannot be reconciled with source JSON."""


@dataclass(frozen=True, slots=True)
class _MarkerCapture:
    pointer: str
    tag: str
    text: list[str]


class _ReportHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.embedded_fragments: list[str] = []
        self.markers: dict[str, str] = {}
        self._embedded_open = False
        self._embedded_seen = False
        self._marker: _MarkerCapture | None = None
        self._marker_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if attributes.get("id") == "evaluation-report-json":
            if (
                tag != "script"
                or attributes.get("type") != "application/json"
                or self._embedded_seen
                or self._embedded_open
            ):
                raise HtmlReportParityError("HTML has an invalid canonical JSON container")
            self._embedded_seen = True
            self._embedded_open = True

        pointer = attributes.get("data-report-pointer")
        if pointer is not None:
            if self._marker is not None or pointer in self.markers:
                raise HtmlReportParityError("HTML contains duplicate or nested value markers")
            self._marker = _MarkerCapture(pointer=pointer, tag=tag, text=[])
            self._marker_depth = 1
        elif self._marker is not None:
            self._marker_depth += 1

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if dict(attrs).get("data-report-pointer") is not None:
            raise HtmlReportParityError("HTML value marker must contain text")
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._embedded_open:
            self._embedded_open = False
        if self._marker is None:
            return
        self._marker_depth -= 1
        if self._marker_depth < 0:
            raise HtmlReportParityError("HTML value marker nesting is invalid")
        if self._marker_depth == 0:
            if tag != self._marker.tag:
                raise HtmlReportParityError("HTML value marker closing tag is invalid")
            self.markers[self._marker.pointer] = "".join(self._marker.text)
            self._marker = None

    def handle_data(self, data: str) -> None:
        if self._embedded_open:
            self.embedded_fragments.append(data)
        if self._marker is not None:
            self._marker.text.append(data)

    def finish(self) -> None:
        if self._embedded_open or not self._embedded_seen:
            raise HtmlReportParityError("HTML canonical JSON container is missing or unclosed")
        if self._marker is not None:
            raise HtmlReportParityError("HTML value marker is unclosed")

    @property
    def embedded_json(self) -> str:
        return "".join(self.embedded_fragments)


def render_html_report(
    report: Mapping[str, object] | BaseModel,
    *,
    redactor: Redactor = DEFAULT_REDACTOR,
) -> str:
    """Render one standalone HTML document from a redacted validated report."""

    prepared = prepare_report(report, redactor=redactor)
    canonical = canonical_report_json(prepared)
    environment = _template_environment()
    template = environment.get_template(REPORT_TEMPLATE_NAME)
    rendered = template.render(
        report=prepared,
        canonical_json=canonical,
        canonical_sections=_CANONICAL_SECTIONS,
    )
    _verify_prepared_html_parity(prepared, rendered)
    return rendered


def write_html_report(
    report: Mapping[str, object] | BaseModel,
    output_path: Path | str,
    *,
    overwrite: bool = False,
    redactor: Redactor = DEFAULT_REDACTOR,
) -> Path:
    """Render and atomically publish an HTML report without replacing by default."""

    target = Path(output_path)
    if target.suffix.casefold() not in {".html", ".htm"}:
        raise ReportWriteError("HTML report path must end in .html or .htm")
    rendered = render_html_report(report, redactor=redactor)
    return _atomic_write_text(target, rendered + "\n", overwrite=overwrite)


def extract_embedded_report(html: str) -> JsonObject:
    """Extract and validate the canonical source JSON embedded in an HTML report."""

    parser = _parse_html(html)
    try:
        raw = decode_json_report(parser.embedded_json)
    except (ReportSerializationError, ValueError) as error:
        raise HtmlReportParityError("HTML contains invalid embedded report JSON") from error
    if not isinstance(raw, dict):
        raise HtmlReportParityError("HTML embedded report JSON is not an object")
    try:
        return validate_report(cast(Mapping[str, object], raw))
    except (ReportSerializationError, ValueError) as error:
        raise HtmlReportParityError("HTML embedded report does not satisfy schema v1") from error


def verify_html_parity(
    report: Mapping[str, object] | BaseModel,
    html: str,
    *,
    redactor: Redactor = DEFAULT_REDACTOR,
) -> None:
    """Verify exact embedded bytes and every marked visible value against JSON."""

    prepared = prepare_report(report, redactor=redactor)
    _verify_prepared_html_parity(prepared, html)


def load_html_report(path: Path | str) -> str:
    source = Path(path)
    try:
        size = source.stat().st_size
    except OSError as error:
        raise HtmlReportError("HTML report file is unavailable") from error
    if size <= 0 or size > MAX_HTML_REPORT_BYTES:
        raise HtmlReportError("HTML report size is outside allowed bounds")
    try:
        return source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise HtmlReportError("HTML report is not valid UTF-8") from error


def _verify_prepared_html_parity(report: JsonObject, html: str) -> None:
    parser = _parse_html(html)
    canonical = canonical_report_json(report)
    if parser.embedded_json != canonical:
        raise HtmlReportParityError("HTML embedded JSON differs from the canonical source")
    embedded = extract_embedded_report(html)
    if embedded != report:
        raise HtmlReportParityError("HTML embedded values differ from the JSON source")

    missing = _REQUIRED_VISIBLE_POINTERS.difference(parser.markers)
    if missing:
        raise HtmlReportParityError("HTML is missing required visible report values")
    for pointer, rendered_value in parser.markers.items():
        try:
            source_value = _resolve_pointer(report, pointer)
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise HtmlReportParityError("HTML references an unknown JSON value") from error
        expected = _display_value(source_value)
        if rendered_value != expected:
            raise HtmlReportParityError("a visible HTML value differs from the JSON source")


def _parse_html(html: str) -> _ReportHtmlParser:
    if not isinstance(html, str):
        raise TypeError("HTML report must be a string")
    if not html or len(html.encode("utf-8")) > MAX_HTML_REPORT_BYTES:
        raise HtmlReportError("HTML report size is outside allowed bounds")
    parser = _ReportHtmlParser()
    try:
        parser.feed(html)
        parser.close()
        parser.finish()
    except HtmlReportError:
        raise
    except (UnicodeError, ValueError) as error:
        raise HtmlReportError("HTML report cannot be parsed") from error
    return parser


def _resolve_pointer(document: JsonValue, pointer: str) -> JsonValue:
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise ValueError("JSON pointer must start with a slash")
    current: JsonValue = document
    for encoded in pointer[1:].split("/"):
        token = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict):
            current = current[token]
        elif isinstance(current, list):
            if not token.isascii() or not token.isdecimal():
                raise ValueError("array JSON pointer token is invalid")
            current = current[int(token)]
        else:
            raise TypeError("JSON pointer traverses a scalar")
    return current


def _display_value(value: object) -> str:
    if isinstance(value, dict | list):
        return canonical_json_value(value)
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "PASS" if value else "FAIL"
    if isinstance(value, str):
        return value
    return canonical_json_value(value)


def _pointer_token(value: object) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def _template_environment() -> Environment:
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


class HtmlReportRenderer:
    """Injectable synchronous renderer suitable for the bounded report pool."""

    def __init__(self, *, redactor: Redactor = DEFAULT_REDACTOR) -> None:
        self._redactor = redactor

    def render(self, report: Mapping[str, object] | BaseModel) -> str:
        return render_html_report(report, redactor=self._redactor)

    def generate(
        self,
        report: Mapping[str, object] | BaseModel,
        output_path: Path | str,
        *,
        overwrite: bool = False,
    ) -> Path:
        return write_html_report(
            report,
            output_path,
            overwrite=overwrite,
            redactor=self._redactor,
        )
