"""Command-line entry point for the application."""

from __future__ import annotations

import uvicorn

from rag_mvp.config.settings import get_settings


def main() -> None:
    """Run the single-process ASGI server."""
    settings = get_settings()
    uvicorn.run(
        "rag_mvp.api.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
