"""Command-line validator for immutable RAG evaluation datasets."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from rag_mvp.evaluation.dataset import DatasetValidationError, load_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="path to a versioned dataset directory")
    parser.add_argument(
        "--corpus-version",
        help="require the dataset to reference this exact corpus semantic version",
    )
    parser.add_argument(
        "--non-acceptance",
        action="store_true",
        help="validate only requirements declared by the manifest",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        dataset = load_dataset(
            args.dataset,
            expected_corpus_version=args.corpus_version,
            acceptance_mode=not args.non_acceptance,
        )
    except (DatasetValidationError, OSError) as exc:
        print(
            json.dumps(
                {
                    "valid": False,
                    "error": type(exc).__name__,
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1

    print(
        json.dumps(
            {
                "valid": True,
                "dataset_id": dataset.manifest.dataset_id,
                "dataset_version": dataset.manifest.version,
                "dataset_content_hash": dataset.manifest.content_hash,
                "corpus_snapshot_id": dataset.corpus.manifest.snapshot_id,
                "corpus_version": dataset.corpus.manifest.version,
                "corpus_content_hash": dataset.corpus.manifest.content_hash,
                "case_count": len(dataset.cases),
                "category_counts": {
                    category.value: count
                    for category, count in sorted(
                        dataset.category_counts.items(),
                        key=lambda item: item[0].value,
                    )
                },
                "metric_eligibility_counts": {
                    metric.value: count
                    for metric, count in sorted(
                        dataset.metric_eligibility_counts.items(),
                        key=lambda item: item[0].value,
                    )
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through ``python -m``
    raise SystemExit(main())
