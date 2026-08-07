from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "statement",
    (
        "import rag_mvp.evaluation.plan; import rag_mvp.api.qa",
        "import rag_mvp.api.qa; import rag_mvp.evaluation.plan",
    ),
)
def test_evaluation_and_api_import_order_is_cycle_safe(statement: str) -> None:
    completed = subprocess.run(  # noqa: S603 - fixed interpreter and test-only statements
        [sys.executable, "-c", statement],
        capture_output=True,
        check=False,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
