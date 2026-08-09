# syntax=docker/dockerfile:1.7@sha256:a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e

ARG PYTHON_IMAGE="docker.io/library/python:3.12.13-slim-trixie@sha256:646fb0bca3dd3ea1bcc6feb72c17ed16eed6e10cffc732fcc1478bd3e7f02d7b"
ARG UV_IMAGE="ghcr.io/astral-sh/uv:0.12.1@sha256:cf4eedcaa81655197f625739489effcbe71b61ceb1506f332c3facae5deceded"

FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE} AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /build

COPY --from=uv /uv /uvx /usr/local/bin/
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

FROM ${PYTHON_IMAGE} AS runtime

ARG APP_VERSION="0.1.0"
ARG SOURCE_REVISION="unknown"
ARG DEBIAN_SNAPSHOT="20260803T000000Z"

LABEL org.opencontainers.image.title="rag-mvp" \
      org.opencontainers.image.description="Bilingual, evidence-grounded RAG assistant MVP" \
      org.opencontainers.image.version="${APP_VERSION}" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      org.opencontainers.image.source="https://github.com/DenisYuqi/Rag_Demo" \
      org.opencontainers.image.licenses="LicenseRef-Proprietary" \
      org.opencontainers.image.base.name="docker.io/library/python:3.12.13-slim-trixie" \
      org.opencontainers.image.base.digest="sha256:646fb0bca3dd3ea1bcc6feb72c17ed16eed6e10cffc732fcc1478bd3e7f02d7b"

RUN sed -i \
        -e "s|URIs: http://deb.debian.org/debian-security|URIs: http://snapshot.debian.org/archive/debian-security/${DEBIAN_SNAPSHOT}|" \
        -e "s|URIs: http://deb.debian.org/debian|URIs: http://snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}|" \
        -e "s|^Suites: trixie trixie-updates$|Suites: trixie|" \
        /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Retries=3 -o Acquire::Check-Valid-Until=false update \
    && install_attempt=1 \
    && until DEBIAN_FRONTEND=noninteractive apt-get \
        -o Acquire::Retries=3 -o Acquire::Check-Valid-Until=false install \
        --yes --no-install-recommends \
        ca-certificates=20250419 \
        libgomp1=14.2.0-19 \
        tesseract-ocr=5.5.0-1+b1 \
        tesseract-ocr-chi-sim=1:4.1.0-2 \
        tesseract-ocr-eng=1:4.1.0-2; do \
        if [ "${install_attempt}" -ge 5 ]; then exit 1; fi; \
        install_attempt=$((install_attempt + 1)); \
        sleep $((install_attempt * 2)); \
    done \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 rag-mvp \
    && useradd --system --uid 10001 --gid 10001 \
        --home-dir /home/rag-mvp --create-home \
        --shell /usr/sbin/nologin rag-mvp \
    && install --directory --owner=10001 --group=10001 --mode=0700 /var/lib/rag-mvp

COPY --from=builder --chown=10001:10001 /opt/venv /opt/venv
COPY --chown=10001:10001 \
    evaluations/privacy/supported-fixtures-v1.json \
    /opt/venv/lib/python3.12/evaluations/privacy/supported-fixtures-v1.json
COPY --chown=10001:10001 \
    evaluations/pricing/openai-comparison-standard-2026-08-07-v1.json \
    /opt/venv/lib/python3.12/evaluations/pricing/openai-comparison-standard-2026-08-07-v1.json

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    RAG_MVP_ENVIRONMENT=production \
    RAG_MVP_HOST=0.0.0.0 \
    RAG_MVP_PORT=8000 \
    RAG_MVP_DATA_ROOT=/var/lib/rag-mvp

WORKDIR /home/rag-mvp
USER 10001:10001

EXPOSE 8000
STOPSIGNAL SIGTERM

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["/opt/venv/bin/python", "-c", "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"RAG_MVP_PORT\", \"8000\")}/healthz', timeout=2).read()"]

CMD ["/opt/venv/bin/rag-mvp"]
