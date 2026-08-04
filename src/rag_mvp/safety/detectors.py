"""Deterministic detectors for the privacy classes required by the MVP."""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Iterator
from datetime import date
from typing import Final

from rag_mvp.safety.models import DetectionSpan, Detector, SensitiveKind


def _span(
    match: re.Match[str],
    *,
    kind: SensitiveKind,
    detector: str,
    priority: int,
    group: int | str = 0,
) -> DetectionSpan:
    start, end = match.span(group)
    return DetectionSpan(start, end, kind, detector, priority)


class EmailDetector:
    """Detect conventional ASCII email addresses in bilingual text."""

    name = "email"
    priority = 70
    _pattern: Final[re.Pattern[str]] = re.compile(
        r"(?<![A-Za-z0-9.!#$%&'*+/=?^_`{|}~-])"
        r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+"
        r"@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
        r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+"
        r"(?![A-Za-z0-9-])"
    )

    @staticmethod
    def _valid(match: re.Match[str]) -> bool:
        local_part = match.group(0).partition("@")[0]
        return (
            not local_part.startswith(".")
            and not local_part.endswith(".")
            and ".." not in local_part
        )

    def detect(self, text: str) -> tuple[DetectionSpan, ...]:
        return tuple(
            _span(
                match,
                kind=SensitiveKind.EMAIL,
                detector=self.name,
                priority=self.priority,
            )
            for match in self._pattern.finditer(text)
            if self._valid(match)
        )


class PhoneDetector:
    """Detect Chinese mobile/landline and explicitly formatted international phones."""

    name = "phone"
    priority = 50
    _candidate: Final[re.Pattern[str]] = re.compile(
        r"(?<![\w+])"
        r"(?:\+?\d{1,3}[ .-]?)?"
        r"(?:\(\d{2,4}\)[ .-]?)?"
        r"\d(?:[ .-]?\d){6,14}"
        r"(?!\w)"
    )
    _china_mobile: Final[re.Pattern[str]] = re.compile(r"(?:86)?1[3-9]\d{9}\Z")
    _china_landline: Final[re.Pattern[str]] = re.compile(r"(?:86)?0\d{9,11}\Z")

    @staticmethod
    def _trimmed_span(match: re.Match[str]) -> tuple[int, int, str]:
        raw = match.group(0)
        leading = len(raw) - len(raw.lstrip())
        trimmed = raw.strip()
        return match.start() + leading, match.start() + leading + len(trimmed), trimmed

    @classmethod
    def _valid_candidate(cls, raw: str) -> bool:
        digits = "".join(
            character for character in raw if character.isascii() and character.isdigit()
        )
        if not 7 <= len(digits) <= 15:
            return False
        if cls._china_mobile.fullmatch(digits) or cls._china_landline.fullmatch(digits):
            return True
        # General international/national matching is intentionally limited to
        # visibly formatted values of a plausible full-number length to avoid
        # treating dates, SSNs, and arbitrary numeric IDs as phones.
        return len(digits) >= 10 and (
            raw.startswith("+") or any(character in raw for character in " ()-.")
        )

    def detect(self, text: str) -> tuple[DetectionSpan, ...]:
        spans: list[DetectionSpan] = []
        for match in self._candidate.finditer(text):
            start, end, raw = self._trimmed_span(match)
            if self._valid_candidate(raw):
                spans.append(
                    DetectionSpan(start, end, SensitiveKind.PHONE, self.name, self.priority)
                )
        return tuple(spans)


_CHINESE_ID_WEIGHTS: Final[tuple[int, ...]] = (
    7,
    9,
    10,
    5,
    8,
    4,
    2,
    1,
    6,
    3,
    7,
    9,
    10,
    5,
    8,
    4,
    2,
)
_CHINESE_ID_CHECKS: Final[str] = "10X98765432"


class ChineseNationalIdDetector:
    """Detect checksum-valid 18-character PRC resident identity numbers."""

    name = "chinese_national_id"
    priority = 90
    _candidate: Final[re.Pattern[str]] = re.compile(
        r"(?<![0-9A-Za-z])[1-9]\d{16}[0-9Xx](?![0-9A-Za-z])"
    )

    @staticmethod
    def _valid_date(value: str) -> bool:
        try:
            parsed = date(int(value[:4]), int(value[4:6]), int(value[6:8]))
        except ValueError:
            return False
        return date(1800, 1, 1) <= parsed <= date.today()

    @classmethod
    def is_valid(cls, value: str) -> bool:
        normalized = value.upper()
        if len(normalized) != 18 or not normalized[:17].isdigit():
            return False
        if normalized[:6] == "000000" or normalized[14:17] == "000":
            return False
        if not cls._valid_date(normalized[6:14]):
            return False
        checksum_index = (
            sum(
                int(character) * weight
                for character, weight in zip(normalized[:17], _CHINESE_ID_WEIGHTS, strict=True)
            )
            % 11
        )
        return normalized[-1] == _CHINESE_ID_CHECKS[checksum_index]

    def detect(self, text: str) -> tuple[DetectionSpan, ...]:
        return tuple(
            _span(
                match,
                kind=SensitiveKind.CHINESE_ID,
                detector=self.name,
                priority=self.priority,
            )
            for match in self._candidate.finditer(text)
            if self.is_valid(match.group(0))
        )


class SSNDetector:
    """Detect structurally valid United States Social Security numbers."""

    name = "us_ssn"
    priority = 90
    _candidate: Final[re.Pattern[str]] = re.compile(
        r"(?<!\d)(?P<area>\d{3})-(?P<group>\d{2})-(?P<serial>\d{4})(?!\d)"
    )

    @staticmethod
    def _valid(match: re.Match[str]) -> bool:
        area = int(match.group("area"))
        group = int(match.group("group"))
        serial = int(match.group("serial"))
        return area not in {0, 666} and area < 900 and group != 0 and serial != 0

    def detect(self, text: str) -> tuple[DetectionSpan, ...]:
        return tuple(
            _span(
                match,
                kind=SensitiveKind.SSN,
                detector=self.name,
                priority=self.priority,
            )
            for match in self._candidate.finditer(text)
            if self._valid(match)
        )


class PaymentCardDetector:
    """Detect plausible 13-19 digit payment-card values using Luhn validation."""

    name = "payment_card"
    priority = 85
    _candidate: Final[re.Pattern[str]] = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")

    @staticmethod
    def passes_luhn(value: str) -> bool:
        digits = [int(character) for character in value if character.isdigit()]
        if not 13 <= len(digits) <= 19 or len(set(digits)) == 1:
            return False
        checksum = 0
        parity = len(digits) % 2
        for index, digit in enumerate(digits):
            if index % 2 == parity:
                digit *= 2
                if digit > 9:
                    digit -= 9
            checksum += digit
        return checksum % 10 == 0

    def detect(self, text: str) -> tuple[DetectionSpan, ...]:
        return tuple(
            _span(
                match,
                kind=SensitiveKind.PAYMENT_CARD,
                detector=self.name,
                priority=self.priority,
            )
            for match in self._candidate.finditer(text)
            if self.passes_luhn(match.group(0))
        )


class IPAddressDetector:
    """Detect validated IPv4 and IPv6 addresses."""

    name = "ip_address"
    priority = 65
    _ipv4_candidate: Final[re.Pattern[str]] = re.compile(
        r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?!\w)(?!\.\d)"
    )
    _ipv6_candidate: Final[re.Pattern[str]] = re.compile(
        r"(?<![0-9A-Fa-f:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}"
        r"(?![0-9A-Fa-f:])"
    )

    @staticmethod
    def _validated(
        matches: Iterator[re.Match[str]], kind: SensitiveKind, version: int
    ) -> Iterator[DetectionSpan]:
        for match in matches:
            value = match.group(0)
            try:
                parsed = ipaddress.ip_address(value)
            except ValueError:
                continue
            if parsed.version == version:
                yield DetectionSpan(match.start(), match.end(), kind, "ip_address", 65)

    def detect(self, text: str) -> tuple[DetectionSpan, ...]:
        ipv4 = self._validated(self._ipv4_candidate.finditer(text), SensitiveKind.IPV4, 4)
        ipv6 = self._validated(self._ipv6_candidate.finditer(text), SensitiveKind.IPV6, 6)
        return tuple((*ipv4, *ipv6))


_PRIVATE_KEY_BLOCK: Final[re.Pattern[str]] = re.compile(
    r"-----BEGIN (?P<label>[A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?)-----"
    r"[\s\S]*?"
    r"-----END (?P=label)-----",
    re.IGNORECASE,
)


class PrivateKeyDetector:
    """Detect a complete PEM private-key block."""

    name = "private_key"
    priority = 120

    def detect(self, text: str) -> tuple[DetectionSpan, ...]:
        return tuple(
            _span(
                match,
                kind=SensitiveKind.SECRET,
                detector=self.name,
                priority=self.priority,
            )
            for match in _PRIVATE_KEY_BLOCK.finditer(text)
        )


class SecretDetector:
    """Detect common authorization, API-key, token, and password values."""

    name = "secret"
    priority = 110
    _captured_patterns: Final[tuple[re.Pattern[str], ...]] = (
        re.compile(
            r"(?i)\b(?:authorization\s*:\s*)?(?:bearer|basic)\s+"
            r"(?P<value>[A-Za-z0-9._~+/=-]+)"
        ),
        re.compile(
            r"(?ix)\b(?:(?:[a-z][a-z0-9_-]*_)?api[ _-]?key|access[_-]?token|"
            r"auth[_-]?token|client[_-]?secret|"
            r"password|passwd|pwd|secret)\s*[:=]\s*"
            r"(?P<value>\"[^\"\r\n]+\"|'[^'\r\n]+'|[^\s,;}{\]]+)"
        ),
    )
    _standalone_patterns: Final[tuple[re.Pattern[str], ...]] = (
        re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{16,}(?![A-Za-z0-9])"),
        re.compile(r"(?<![A-Z0-9])AKIA[A-Z0-9]{16}(?![A-Z0-9])"),
        re.compile(r"(?<![A-Za-z0-9])AIza[0-9A-Za-z_-]{20,}(?![A-Za-z0-9])"),
        re.compile(r"(?<![A-Za-z0-9])gh[pousr]_[A-Za-z0-9]{20,}(?![A-Za-z0-9])"),
        re.compile(r"(?<![A-Za-z0-9])xox[baprs]-[A-Za-z0-9-]{16,}(?![A-Za-z0-9])"),
    )

    def detect(self, text: str) -> tuple[DetectionSpan, ...]:
        spans: list[DetectionSpan] = []
        for pattern in self._captured_patterns:
            spans.extend(
                _span(
                    match,
                    kind=SensitiveKind.SECRET,
                    detector=self.name,
                    priority=self.priority,
                    group="value",
                )
                for match in pattern.finditer(text)
            )
        for pattern in self._standalone_patterns:
            spans.extend(
                _span(
                    match,
                    kind=SensitiveKind.SECRET,
                    detector=self.name,
                    priority=self.priority,
                )
                for match in pattern.finditer(text)
            )
        return tuple(spans)


DEFAULT_DETECTORS: Final[tuple[Detector, ...]] = (
    PrivateKeyDetector(),
    SecretDetector(),
    ChineseNationalIdDetector(),
    SSNDetector(),
    PaymentCardDetector(),
    EmailDetector(),
    IPAddressDetector(),
    PhoneDetector(),
)


def has_unclosed_private_key(text: str) -> bool:
    """Return whether ``text`` contains a PEM private-key start without its end."""

    begin_pattern = re.compile(
        r"-----BEGIN (?P<label>[A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?)-----", re.IGNORECASE
    )
    for begin in begin_pattern.finditer(text):
        label = re.escape(begin.group("label"))
        end_pattern = re.compile(rf"-----END {label}-----", re.IGNORECASE)
        if end_pattern.search(text, begin.end()) is None:
            return True
    return False
