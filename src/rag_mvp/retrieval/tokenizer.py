"""Versioned bilingual lexical tokenization."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import jieba

_LATIN = re.compile(r"[A-Za-z0-9]+(?:[._'-][A-Za-z0-9]+)*")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
BILINGUAL_TOKENIZER_IDENTITY = "jieba-cjk-ngram-v1"


@dataclass(frozen=True, slots=True)
class BilingualTokenizer:
    version: str = BILINGUAL_TOKENIZER_IDENTITY
    _jieba: jieba.Tokenizer = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.version != BILINGUAL_TOKENIZER_IDENTITY:
            raise ValueError("unsupported_tokenizer_identity")
        tokenizer = jieba.Tokenizer()
        tokenizer.initialize()
        object.__setattr__(self, "_jieba", tokenizer)

    def tokenize(self, text: str) -> tuple[str, ...]:
        tokens: list[str] = [match.group(0).casefold() for match in _LATIN.finditer(text)]
        for match in _CJK.finditer(text):
            sequence = match.group(0)
            tokens.append(sequence)
            tokens.extend(piece for piece in self._jieba.cut(sequence, HMM=False) if piece.strip())
            tokens.extend(sequence)
            tokens.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
        return tuple(tokens)
