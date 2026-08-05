from __future__ import annotations

import pytest

from rag_mvp.providers.models import Deadline, ProviderCallContext


@pytest.fixture
def provider_context() -> ProviderCallContext:
    return ProviderCallContext(
        request_id="request-test",
        operation_id="operation-test",
        deadline=Deadline.after(5),
    )
