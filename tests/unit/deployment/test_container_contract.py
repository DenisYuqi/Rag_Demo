"""Static regression tests for the local container deployment contract."""

from __future__ import annotations

import re
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _read_repository_file(name: str) -> str:
    return (REPOSITORY_ROOT / name).read_text(encoding="utf-8")


def _meaningful_dockerignore_lines() -> list[str]:
    return [
        line.strip()
        for line in _read_repository_file(".dockerignore").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _top_level_mapping_keys(text: str, section: str) -> list[str]:
    """Read direct child keys from a simple top-level YAML mapping."""

    match = re.search(rf"(?ms)^{re.escape(section)}:\s*\n(?P<body>.*?)(?=^\S|\Z)", text)
    assert match is not None, f"missing top-level {section!r} section"
    return re.findall(r"(?m)^  ([A-Za-z0-9_-]+):(?:\s|$)", match.group("body"))


def test_dockerignore_sensitive_exclusions_are_last_matching_rules() -> None:
    lines = _meaningful_dockerignore_lines()
    required_sensitive_rules = {
        ".env",
        ".env.*",
        "**/.env",
        "**/.env.*",
        "secrets",
        "secrets/**",
        "**/secrets",
        "**/secrets/**",
        "**/*.key",
        "**/*.pem",
        "**/*.p12",
        "**/*.pfx",
        "**/id_rsa*",
        "**/*credential*",
        "**/*secret*",
    }

    assert required_sensitive_rules <= set(lines)
    assert "!src/**" not in lines
    assert {"!src/**/*.py", "!src/**/*.json", "!src/**/*.j2"} <= set(lines)
    last_allow_index = max(index for index, line in enumerate(lines) if line.startswith("!"))
    first_sensitive_index = min(
        index for index, line in enumerate(lines) if line in required_sensitive_rules
    )
    assert first_sensitive_index > last_allow_index
    assert not any(line.startswith("!") for line in lines[first_sensitive_index:])
    assert lines[-1] == "**/*.egg-info/**"
    assert {
        "**/__pycache__",
        "**/__pycache__/**",
        "**/*.py[cod]",
        "**/*.egg-info",
        "**/*.egg-info/**",
    } <= set(lines[first_sensitive_index:])


def test_dockerfile_is_pinned_multistage_nonroot_and_revision_labeled() -> None:
    dockerfile = _read_repository_file("Dockerfile")

    assert re.search(
        r"(?m)^# syntax=docker/dockerfile:1\.7@sha256:[0-9a-f]{64}$",
        dockerfile,
    )
    image_args = re.findall(r'^ARG\s+\w+_IMAGE="([^"]+)"', dockerfile, flags=re.MULTILINE)
    assert len(image_args) == 2
    assert all(re.search(r"@sha256:[0-9a-f]{64}$", image) for image in image_args)
    assert len(re.findall(r"(?m)^FROM\s+", dockerfile)) >= 3
    assert "uv sync --frozen --no-dev" in dockerfile
    assert 'ARG DEBIAN_SNAPSHOT="20260803T000000Z"' in dockerfile
    assert "snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT}" in dockerfile
    assert "s|^Suites: trixie trixie-updates$|Suites: trixie|" in dockerfile
    assert 'if [ "${install_attempt}" -ge 5 ]; then exit 1; fi' in dockerfile
    assert "tesseract-ocr=5.5.0-1+b1" in dockerfile
    assert re.search(r'(?m)^ARG SOURCE_REVISION="unknown"$', dockerfile)
    assert 'org.opencontainers.image.revision="${SOURCE_REVISION}"' in dockerfile
    assert re.search(r"(?m)^USER 10001:10001$", dockerfile)
    assert re.search(r"(?m)^STOPSIGNAL SIGTERM$", dockerfile)
    assert 'CMD ["/opt/venv/bin/rag-mvp"]' in dockerfile
    assert "OPENAI_API_KEY" not in dockerfile


def test_compose_has_one_hardened_loopback_application_service() -> None:
    compose = _read_repository_file("compose.yaml")
    placeholder = _read_repository_file("docker/provider-key.placeholder")

    assert _top_level_mapping_keys(compose, "services") == ["app"]
    assert "container_name: rag-mvp" in compose
    assert "init: true" not in compose
    assert 'user: "10001:10001"' in compose
    assert "read_only: true" in compose
    assert re.search(r"(?ms)^    cap_drop:\s*\n      - ALL$", compose)
    assert "- no-new-privileges:true" in compose
    assert re.search(r"(?ms)^    deploy:\s*\n      replicas: 1$", compose)
    assert "${RAG_MVP_BIND_ADDRESS:-127.0.0.1}" in compose
    assert "${RAG_MVP_HOST_PORT:-8000}" in compose
    assert "source: app_data" in compose
    assert "target: /var/lib/rag-mvp" in compose
    assert "RAG_MVP_OPENAI_API_KEY_FILE: /run/secrets/openai_api_key" in compose
    assert "RAG_MVP_OPENAI_API_KEY:" not in compose
    for setting in (
        "RAG_MVP_PROVIDER_TIMEOUT_SECONDS",
        "RAG_MVP_PROVIDER_RETRY_LIMIT",
        "RAG_MVP_QA_DEADLINE_SECONDS",
        "RAG_MVP_QA_RETRIEVAL_BUDGET_SECONDS",
        "RAG_MVP_QA_EMBEDDING_BUDGET_SECONDS",
        "RAG_MVP_QA_EVIDENCE_ASSESSMENT_BUDGET_SECONDS",
        "RAG_MVP_QA_GENERATION_BUDGET_SECONDS",
        "RAG_MVP_QA_FINALIZATION_BUDGET_SECONDS",
    ):
        assert setting in compose
        assert setting in _read_repository_file(".env.example")
    for setting, default in (
        ("RAG_MVP_QA_DEADLINE_SECONDS", "9.5"),
        ("RAG_MVP_QA_RETRIEVAL_BUDGET_SECONDS", "5.0"),
        ("RAG_MVP_QA_EMBEDDING_BUDGET_SECONDS", "4.5"),
        ("RAG_MVP_QA_EVIDENCE_ASSESSMENT_BUDGET_SECONDS", "5.0"),
        ("RAG_MVP_QA_GENERATION_BUDGET_SECONDS", "6.0"),
        ("RAG_MVP_QA_FINALIZATION_BUDGET_SECONDS", "0.6"),
    ):
        assert f"{setting}: ${{{setting}:-{default}}}" in compose
        assert f"{setting}={default}" in _read_repository_file(".env.example")
    assert "RAG_MVP_TELEMETRY_EXPORTER:" in compose
    assert "RAG_MVP_TELEMETRY_OTLP_TRACES_ENDPOINT:" in compose
    assert "RAG_MVP_SERVER_SHUTDOWN_GRACE_SECONDS:" in compose
    assert "RAG_MVP_SHUTDOWN_GRACE_SECONDS:" in compose
    assert re.search(r"(?m)^    stop_grace_period: 20s$", compose)
    assert "RAG_MVP_STOP_GRACE_PERIOD" not in compose
    assert "SOURCE_REVISION: ${RAG_MVP_SOURCE_REVISION:-unknown}" in compose
    assert "org.opencontainers.image.revision" not in compose
    assert placeholder.strip() == ""


def test_runbook_asserts_revision_and_recreate_persistence_evidence() -> None:
    runbook = _read_repository_file("docs/local-container-runbook.md")
    powershell = "\n".join(re.findall(r"```powershell\n(.*?)```", runbook, re.DOTALL))

    assert "OCI revision mismatch" in runbook
    assert "Release evidence inputs must be committed and clean" in runbook
    for release_input in (
        ".trivyignore",
        "docs/local-container-runbook.md",
        "docs/container-security-review.md",
        "docker/provider-key.placeholder",
        "evaluations/datasets/mvp-v1/corpus/sources/benefits-policy-en.md",
        "evaluations/performance/acceptance-scenarios-v1.json",
        "evaluations/pricing/openai-standard-2026-08-07.json",
        "evaluations/privacy/supported-fixtures-v1.json",
    ):
        assert release_input in runbook
    assert "Docker context contains an untracked or missing allowlisted source file" in runbook
    assert "Runtime secret path must remain outside the repository" in runbook
    assert "COMPOSE_DISABLE_ENV_FILE" in runbook
    assert "compose-resolved.json" in runbook
    assert "stop_grace_period -ne '20s'" in runbook
    assert "Unsupported release image platform" in runbook
    assert "Assert-ContainerImage $firstContainerId" in runbook
    assert "Assert-ContainerImage $secondContainerId" in runbook
    assert "Assert-ContainerImage $restoredContainerId" in runbook
    assert "Container did not complete a clean graceful stop" in runbook
    assert "$shutdownEvents[-1].counts.failed_tasks -ne 0" in runbook
    assert "--force-recreate --no-deps app" in runbook
    assert "container_id_changed = $true" in runbook
    assert "reingested = $false" in runbook
    assert "citation_identity_equal = $true" in runbook
    assert "$health.status -eq 'alive'" in runbook
    assert "[Text.Encoding]::UTF8.GetString($qaResponse.Content)" in runbook
    assert "LocalApplicationData" in runbook
    assert "trivy-high-critical.raw.json" in runbook
    assert "trivy-secret.gate.json" in runbook
    assert "trivy-critical-policy.gate.json" in runbook
    assert "[version]'0.73.0'" in runbook
    assert "application-first-secret-leak-gate.json" in runbook
    assert "application-second-secret-leak-gate.json" in runbook
    assert "function Capture-DockerLogs" in runbook
    assert "Capture-AndScanContainerLogs" in runbook
    assert "$ErrorActionPreference = 'Continue'" in runbook
    assert not re.search(r"docker logs .*2>&1 \| Set-Content", runbook)
    assert runbook.count("$null -ne $_.PSObject.Properties['event']") == 2
    assert "successful_shutdown_events = $shutdownEvents.Count" in runbook
    assert "application-first-secret-leak-gate.json' 1" in runbook
    assert "application-second-secret-leak-gate.json' 2" in runbook
    assert "$applicationLogText.Contains($secretText)" in runbook
    assert "ReadAllText($secretPath, [Text.Encoding]::UTF8).Trim()" in runbook
    assert "Chroma server route must remain absent" in runbook
    assert re.search(r"alpine:3\.22\.1@sha256:[0-9a-f]{64}", runbook)
    assert runbook.count("--network none") >= 3
    assert runbook.count("--user 10001:10001") >= 3
    assert "Initialize restore volume ownership" in runbook
    assert '--mount "type=volume,src=$restoreVolume,dst=/var/lib/rag-mvp"' in runbook
    assert "Verify restore volume is empty" in runbook
    assert "$helperImage find /data -mindepth 1 -maxdepth 1 -print -quit" in runbook
    assert "$helperImage tar -xzf /backup/restore.tgz -C /data" in runbook
    assert "$helperImage sh -ec" not in runbook
    assert "$imageId" in runbook
    assert "$retainedVolumeNames = @($retainedVolumes | ForEach-Object" in runbook
    assert "@($retainedVolumes.Name)" not in runbook
    assert "docker compose down --remove-orphans" in runbook
    assert not re.search(r"(?m)^\s*docker compose down .*--volumes", powershell)
    assert not re.search(r"(?m)^\s*docker volume rm(?:\s|$)", powershell)
