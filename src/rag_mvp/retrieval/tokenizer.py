"""Versioned bilingual lexical tokenization."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.resources
import re
from dataclasses import dataclass, field

import jieba

_LATIN = re.compile(r"[A-Za-z0-9]+(?:[._'-][A-Za-z0-9]+)*")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
JIEBA_PACKAGE_VERSION = "0.42.1"
JIEBA_DICTIONARY_SHA256 = "7197c3211ddd98962b036cdf40324d1ea2bfaa12bd028e68faa70111a88e12a8"
TOKENIZER_IMPLEMENTATION_VERSION = "latin-jieba-cjk-ngram-v2"
BILINGUAL_TOKENIZER_IDENTITY = (
    f"{TOKENIZER_IMPLEMENTATION_VERSION}:jieba-{JIEBA_PACKAGE_VERSION}:"
    f"dict-sha256-{JIEBA_DICTIONARY_SHA256}:hmm-false"
)


@dataclass(frozen=True, slots=True)
class BilingualTokenizer:
    version: str = BILINGUAL_TOKENIZER_IDENTITY
    _jieba: jieba.Tokenizer = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.version != BILINGUAL_TOKENIZER_IDENTITY:
            raise ValueError("unsupported_tokenizer_identity")
        try:
            package_version = importlib.metadata.version("jieba")
            dictionary = importlib.resources.files("jieba").joinpath("dict.txt").read_bytes()
        except (ImportError, OSError):
            raise ValueError("unsupported_tokenizer_implementation") from None
        if (
            package_version != JIEBA_PACKAGE_VERSION
            or hashlib.sha256(dictionary).hexdigest() != JIEBA_DICTIONARY_SHA256
        ):
            raise ValueError("unsupported_tokenizer_implementation")
        tokenizer = jieba.Tokenizer()
        tokenizer.initialize()
        object.__setattr__(self, "_jieba", tokenizer)

    def tokenize(self, text: str) -> tuple[str, ...]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        tokens: list[str] = [match.group(0).casefold() for match in _LATIN.finditer(text)]
        for match in _CJK.finditer(text):
            sequence = match.group(0)
            tokens.append(sequence)
            tokens.extend(piece for piece in self._jieba.cut(sequence, HMM=False) if piece.strip())
            tokens.extend(sequence)
            tokens.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
        return tuple(tokens)
