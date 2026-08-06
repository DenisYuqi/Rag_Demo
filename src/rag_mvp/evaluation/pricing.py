"""Pinned standard OpenAI pricing used by the 2026-08 acceptance evidence."""

from __future__ import annotations

from decimal import Decimal

from rag_mvp.domain.evaluation import ModelPricing
from rag_mvp.observability.costs import PricingCatalog

OPENAI_STANDARD_PRICING_VERSION = "openai-standard-2026-08-07"
OPENAI_PRICING_SOURCES = (
    "https://openai.com/index/introducing-gpt-5-4/",
    "https://developers.openai.com/api/docs/models/gpt-4.1-mini",
    "https://developers.openai.com/api/docs/models/text-embedding-3-small",
)

_STANDARD_USD_PER_MILLION: dict[str, tuple[Decimal, Decimal | None]] = {
    "gpt-5.4": (Decimal("2.50"), Decimal("15.00")),
    "gpt-4.1-mini": (Decimal("0.40"), Decimal("1.60")),
    "text-embedding-3-small": (Decimal("0.02"), None),
}


def openai_standard_pricing_catalog(
    *,
    provider: str,
    models: tuple[str, ...],
) -> PricingCatalog:
    """Build a catalog only for exact models covered by the pinned public rate card.

    Unknown model names remain absent so downstream estimates correctly report
    ``pricing-not-found`` instead of silently applying a nearby model's rate.
    """

    unique_models = tuple(dict.fromkeys(models))
    entries = tuple(
        ModelPricing(
            pricing_version=OPENAI_STANDARD_PRICING_VERSION,
            provider=provider,
            model=model,
            currency="USD",
            input_per_million=_STANDARD_USD_PER_MILLION[model][0],
            output_per_million=_STANDARD_USD_PER_MILLION[model][1],
        )
        for model in unique_models
        if model in _STANDARD_USD_PER_MILLION
    )
    return PricingCatalog(version=OPENAI_STANDARD_PRICING_VERSION, entries=entries)


__all__ = [
    "OPENAI_PRICING_SOURCES",
    "OPENAI_STANDARD_PRICING_VERSION",
    "openai_standard_pricing_catalog",
]
