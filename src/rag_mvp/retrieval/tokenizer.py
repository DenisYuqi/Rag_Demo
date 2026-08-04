"""Versioned bilingual lexical tokenization."""

from __future__ import annotations

import re
from dataclasses import dataclass

import jieba

_LATIN = re.compile(r"[A-Za-z0-9]+(?:[._'-][A-Za-z0-9]+)*")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")


@dataclass(frozen=True, slots=True)
class BilingualTokenizer:
    version: str = "jieba-cjk-ngram-v1"

    def tokenize(self, text: str) -> tuple[str, ...]:
        tokens: list[str] = [match.group(0).casefold() for match in _LATIN.finditer(text)]
        for match in _CJK.finditer(text):
            sequence = match.group(0)
            tokens.append(sequence)
            tokens.extend(piece for piece in jieba.cut(sequence, HMM=False) if piece.strip())
            tokens.extend(sequence)
            tokens.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
        return tuple(tokens)
