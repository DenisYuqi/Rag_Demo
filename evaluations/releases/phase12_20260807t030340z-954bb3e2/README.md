# RAG MVP Phase 12 accepted release

Release ID: `20260807t030340z-954bb3e2`

This directory is the immutable Phase 12 publication set for the accepted single-instance
candidate. The accepted runtime source is Git/OCI revision
`333150f731f1882b52cd504a44ef62c8b15a7188`; the later commit that publishes this directory
is evidence metadata and is not the source baked into the image.

## Accepted result

- Local image reference: `rag-mvp:phase12-333150f731f1`
- Local image ID: `sha256:592adf63fcb0795a2197c67386d9d00e2da012c56bc62a2e303270796e6cfe29`
- Image digest scope: local container engine; no remotely pullable registry digest was verified.
- Platform and runtime identity: `linux/amd64`, `10001:10001`
- Configuration ID: `d78625c2d01ee0e3`
- Dataset/corpus: `mvp-bilingual-rag` / `mvp-synthetic-corpus`, both version `1.0.0`
- Quality: Faithfulness `1.0`, Context Precision `1.0`, Completeness `1.0`, Style
  `0.9583333333333334`, Refusal Appropriateness `1.0`; final gate passed.
- Load: one instance, concurrency `5`, `509` attempts, `504` successes, `5` errors,
  P90 complete latency `4268.061692999936 ms`, error rate `0.009823182711198428`;
  evidence decision valid and passed.
- Load cost: `USD 0.76331826`; projected cost per 1,000 successful calls
  `USD 1.514520357142857142857142857` using pricing version
  `openai-standard-2026-08-07`.
- Privacy: 107 privacy tests passed; 120 publishable captured files had zero supported fixture
  matches, zero semantic detector matches, and zero scan errors.
- Security: zero embedded secrets and zero unresolved Critical findings. Raw Trivy reports are
  retained outside this release directory because third-party package metadata contains public
  maintainer email addresses; only the content-minimized security result is recorded in the
  manifest.

`release-manifest.json` is the authoritative identity, threshold, provenance, cost, and artifact
hash record. The JSON/HTML evaluation reports and performance bundle are exact byte copies from
the accepted image run. `privacy-scan.json` is the final LF-normalized captured-surface privacy
record.

## Offline verification (no model charges)

Run these commands from the repository root with PowerShell:

```powershell
uv sync --frozen

$releaseRoot = (Resolve-Path `
  'evaluations/releases/phase12_20260807t030340z-954bb3e2').Path

uv run python -m rag_mvp.evaluation.verify_report `
  (Join-Path $releaseRoot 'evaluation-report.json') `
  --html (Join-Path $releaseRoot 'evaluation-report.html')

uv run python -c "from rag_mvp.performance.evidence_bundle import load_performance_evidence_bundle as load; b=load(r'$releaseRoot\performance-evidence.json'); assert b['decision']['valid'] and b['decision']['passed']; print('performance bundle verified')"

$manifest = Get-Content -Raw -LiteralPath `
  (Join-Path $releaseRoot 'release-manifest.json') | ConvertFrom-Json
foreach ($artifact in $manifest.artifacts.PSObject.Properties) {
  $actual = 'sha256:' + (Get-FileHash -Algorithm SHA256 -LiteralPath `
    (Join-Path $releaseRoot $artifact.Value.path)).Hash.ToLowerInvariant()
  if ($actual -ne [string] $artifact.Value.sha256) {
    throw "artifact hash mismatch: $($artifact.Value.path)"
  }
}

uv run pytest -q -m privacy
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest
openspec validate build-rag-assistant-mvp --strict
```

The accepted results for the last four engineering gates are: 248 files formatted, Ruff passed,
Mypy passed for 120 source files, and Pytest `1086 passed, 1 skipped`. The skipped test is the
explicitly opt-in paid OpenAI live test; the versioned real-provider quality and load runs are the
published paid evidence instead.

## Rebuild and assert the exact candidate image

Use a detached worktree so the release publication commit is not confused with the accepted
runtime source:

```powershell
$acceptedRevision = '333150f731f1882b52cd504a44ef62c8b15a7188'
$acceptedImage = 'rag-mvp:phase12-333150f731f1'
$acceptedImageId = 'sha256:592adf63fcb0795a2197c67386d9d00e2da012c56bc62a2e303270796e6cfe29'
$worktree = Join-Path (Split-Path -Parent (Get-Location)) 'rag-mvp-phase12-source'

git worktree add --detach $worktree $acceptedRevision
Set-Location $worktree
docker build --progress plain `
  --build-arg "SOURCE_REVISION=$acceptedRevision" `
  --build-arg 'APP_VERSION=0.1.0' `
  --tag $acceptedImage .

$image = docker image inspect $acceptedImage --format '{{json .}}' | ConvertFrom-Json
if ([string] $image.Id -ne $acceptedImageId) { throw 'image ID mismatch' }
if ([string] $image.Config.Labels.'org.opencontainers.image.revision' -ne $acceptedRevision) {
  throw 'OCI revision mismatch'
}
if ([string] $image.Os -ne 'linux' -or [string] $image.Architecture -ne 'amd64') {
  throw 'platform mismatch'
}
if ([string] $image.Config.User -ne '10001:10001') { throw 'runtime identity mismatch' }
```

The base images and Debian snapshot are pinned in `Dockerfile`. The image-ID assertion is
deliberately strict: a different ID is a different candidate and must not be represented as this
accepted release.

## Reproduce the real-provider quality gate (paid)

This creates a fresh volume and a fresh report. Model output and provider latency can vary, so it
reproduces the versioned gate, not byte-identical model output. Put the provider credential in a
file outside the repository; never paste its value into a command or report.

```powershell
$repositoryRoot = (Get-Location).Path
$acceptedRevision = '333150f731f1882b52cd504a44ef62c8b15a7188'
$acceptedImageId = 'sha256:592adf63fcb0795a2197c67386d9d00e2da012c56bc62a2e303270796e6cfe29'
$credentialFile = (Resolve-Path 'D:/private/rag-mvp-provider-key').Path
$datasetRoot = (Resolve-Path 'evaluations/datasets/mvp-v1').Path
$issuesFile = (Resolve-Path 'evaluations/phase10-issues.json').Path
$qualityOutput = Join-Path $env:LOCALAPPDATA 'rag-mvp/reproduction/phase12-quality'
$qualityVolume = 'rag-mvp-phase12-reproduction-quality'
$qualityRunId = 'phase12-reproduction-quality'
New-Item -ItemType Directory -Force -Path $qualityOutput | Out-Null
docker volume create $qualityVolume

$acceptedEnvironment = @(
  'RAG_MVP_SERVICE_VERSION=0.1.0',
  'RAG_MVP_WORKBENCH_ENABLED=false',
  'RAG_MVP_PROVIDER_BACKEND=openai',
  'RAG_MVP_OPENAI_API_KEY_FILE=/run/secrets/openai_api_key',
  'RAG_MVP_OPENAI_BASE_URL=https://api.openai.com/v1',
  'RAG_MVP_OPENAI_SEND_DIMENSIONS=true',
  'RAG_MVP_OPENAI_MAX_TOKENS_PARAMETER=max_completion_tokens',
  'RAG_MVP_EMBEDDING_MODEL=text-embedding-3-small',
  'RAG_MVP_EMBEDDING_DIMENSION=1536',
  'RAG_MVP_GENERATION_MODEL=gpt-5.4',
  'RAG_MVP_RERANKING_MODEL=',
  'RAG_MVP_PROVIDER_TIMEOUT_SECONDS=8',
  'RAG_MVP_PROVIDER_RETRY_LIMIT=1',
  'RAG_MVP_QA_DEADLINE_SECONDS=9.5',
  'RAG_MVP_QA_RETRIEVAL_BUDGET_SECONDS=5.0',
  'RAG_MVP_QA_EMBEDDING_BUDGET_SECONDS=4.5',
  'RAG_MVP_QA_EVIDENCE_ASSESSMENT_BUDGET_SECONDS=5.0',
  'RAG_MVP_QA_GENERATION_BUDGET_SECONDS=5.0',
  'RAG_MVP_QA_FINALIZATION_BUDGET_SECONDS=0.1',
  'RAG_MVP_RETRIEVAL_CACHE_ENABLED=false',
  'RAG_MVP_OCR_ENABLED=true',
  'RAG_MVP_OCR_LANGUAGES=chi_sim+eng',
  'RAG_MVP_TELEMETRY_EXPORTER=console',
  'RAG_MVP_PRICING_VERSION=openai-standard-2026-08-07'
)

$qualityArgs = @(
  'run', '--rm', '--user', '10001:10001', '--read-only',
  '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges:true',
  '--pids-limit', '256', '--tmpfs', '/tmp:rw,noexec,nosuid,size=268435456',
  '--mount', "type=volume,source=$qualityVolume,target=/var/lib/rag-mvp",
  '--mount', "type=bind,source=$qualityOutput,target=/acceptance/output",
  '--mount', "type=bind,source=$datasetRoot,target=/inputs/dataset,readonly",
  '--mount', "type=bind,source=$issuesFile,target=/inputs/issues.json,readonly",
  '--mount', "type=bind,source=$credentialFile,target=/run/secrets/openai_api_key,readonly"
)
foreach ($setting in $acceptedEnvironment) { $qualityArgs += @('--env', $setting) }
$qualityArgs += @(
  $acceptedImageId, 'python', '-m', 'rag_mvp.evaluation.run_evaluation',
  '/inputs/dataset', '--data-root', '/var/lib/rag-mvp',
  '--output-root', '/acceptance/output', '--run-id', $qualityRunId,
  '--profile', 'accepted', '--issues', '/inputs/issues.json', '--require-final-pass'
)
docker @qualityArgs
if ($LASTEXITCODE -ne 0) { throw 'quality reproduction failed' }

$qualityRunRoot = Join-Path $qualityOutput $qualityRunId
docker run --rm --user 10001:10001 --read-only --cap-drop ALL `
  --security-opt no-new-privileges:true --pids-limit 64 `
  --mount "type=bind,source=$qualityRunRoot,target=/evidence,readonly" `
  $acceptedImageId python -m rag_mvp.evaluation.verify_report `
  /evidence/report.json --html /evidence/report.html
if ($LASTEXITCODE -ne 0) { throw 'quality report verification failed' }
```

The accepted quality run cost `USD 0.01346762`; future charges depend on provider pricing and
actual token use.

## Reproduce the five-concurrent-request load gate (paid)

Run this after the quality command so the named volume contains the installed, immutable corpus
and active indexes. Confirm current quota and your own cost cap before executing
`--confirm-acceptance-run`.

```powershell
$networkName = 'rag-mvp-phase12-reproduction-load'
$applicationName = 'rag-mvp-phase12-reproduction-app'
$loadOutput = Join-Path $env:LOCALAPPDATA 'rag-mvp/reproduction/phase12-load'
$scenarioFile = (Resolve-Path 'evaluations/performance/acceptance-scenarios-v1.json').Path
$pricingFile = (Resolve-Path 'evaluations/pricing/openai-standard-2026-08-07.json').Path
New-Item -ItemType Directory -Force -Path $loadOutput | Out-Null
docker network create --driver bridge $networkName

$appArgs = @(
  'create', '--name', $applicationName, '--network', $networkName,
  '--network-alias', 'app', '--user', '10001:10001', '--read-only',
  '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges:true',
  '--pids-limit', '256', '--stop-timeout', '20',
  '--tmpfs', '/tmp:rw,noexec,nosuid,size=268435456',
  '--mount', "type=volume,source=$qualityVolume,target=/var/lib/rag-mvp",
  '--mount', "type=bind,source=$credentialFile,target=/run/secrets/openai_api_key,readonly"
)
foreach ($setting in $acceptedEnvironment) { $appArgs += @('--env', $setting) }
$appArgs += $acceptedImageId
$applicationId = (docker @appArgs | Select-Object -Last 1).Trim()
if ($LASTEXITCODE -ne 0) { throw 'application create failed' }
docker start $applicationId
if ($LASTEXITCODE -ne 0) { throw 'application start failed' }

$ready = $null
for ($attempt = 1; $attempt -le 45; $attempt++) {
  $readyText = docker run --rm --network $networkName $acceptedImageId python -c `
    "import urllib.request; print(urllib.request.urlopen('http://app:8000/readyz', timeout=3).read().decode())"
  if ($LASTEXITCODE -eq 0) {
    $ready = $readyText | ConvertFrom-Json
    if ([string] $ready.status -eq 'ready') { break }
  }
  Start-Sleep -Seconds 2
}
if ($null -eq $ready -or [string] $ready.status -ne 'ready') {
  throw 'application did not become ready'
}
if ([string] $ready.configuration_id -ne 'd78625c2d01ee0e3') {
  throw 'configuration ID mismatch'
}
$instanceIdentity = [string] $ready.instance_identity

$models = @(
  'bm25={"algorithm":"bm25-okapi-v1","b":0.75,"k1":1.5,"schema":"bm25-snapshot-v3","tokenizer":"latin-jieba-cjk-ngram-v2:jieba-0.42.1:dict-sha256-7197c3211ddd98962b036cdf40324d1ea2bfaa12bd028e68faa70111a88e12a8:hmm-false"}',
  'dense={"metric":"cosine","schema":"chroma-revision-v1"}',
  'embedding={"adapter":"openai-compatible-v1","dimension":1536,"model":"text-embedding-3-small","normalization":"none","provider":"openai-compatible-d9617135d6fdd0a2"}',
  'generation=openai-compatible-d9617135d6fdd0a2/gpt-5.4/openai-compatible-v1',
  'rrf={"dense_weight":1.0,"k":60,"lexical_weight":1.0,"tie_policy":"rrf-score-best-rank-chunk-id-v1","version":"weighted-rrf-v1"}'
)

$loadArgs = @(
  'run', '--rm', '--network', $networkName, '--user', '10001:10001',
  '--read-only', '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges:true',
  '--pids-limit', '128', '--tmpfs', '/tmp:rw,noexec,nosuid,size=67108864',
  '--mount', "type=bind,source=$loadOutput,target=/evidence",
  '--mount', "type=bind,source=$scenarioFile,target=/inputs/scenarios.json,readonly",
  '--mount', "type=bind,source=$pricingFile,target=/inputs/pricing.json,readonly",
  $acceptedImageId, 'python', '-m', 'rag_mvp.performance.run_load_test',
  '--base-url', 'http://app:8000', '--scenario-file', '/inputs/scenarios.json',
  '--output', '/evidence/performance-evidence.json',
  '--run-id', 'phase12-reproduction-load', '--code-revision', $acceptedRevision,
  '--configuration-id', 'd78625c2d01ee0e3', '--service-version', '0.1.0'
)
foreach ($model in $models) {
  $loadArgs += @('--model', $model.Replace('"', '\"'))
}
$loadArgs += @(
  '--instance-identity', $instanceIdentity,
  '--expected-workload-digest', 'sha256:af1e02ceed740f3e85af145a4922532403115dab6bd666564fd02f9c0471d65f',
  '--confirm-acceptance-run', '--metric-reference', 'metrics-before-load',
  '--log-reference', 'application-final-log',
  '--pricing-evidence', '/inputs/pricing.json', '--warmup-attempts', '5',
  '--concurrency', '5', '--target-successes', '500', '--max-attempts', '600',
  '--retry-limit', '1', '--request-timeout-seconds', '15', '--instance-count', '1'
)
docker @loadArgs
$loadExitCode = $LASTEXITCODE
docker logs $applicationId 2>&1 | Set-Content -Encoding utf8 `
  (Join-Path $loadOutput 'application-final.log')
docker stop --time 20 $applicationId
docker rm $applicationId
docker network rm $networkName
if ($loadExitCode -ne 0) { throw 'load decision did not pass' }

uv run python -c "from rag_mvp.performance.evidence_bundle import load_performance_evidence_bundle as load; b=load(r'$loadOutput\performance-evidence.json'); assert b['decision']['valid'] and b['decision']['passed']; print('reproduced performance bundle verified')"
```

The accepted preflight estimated approximately `USD 0.85` for 505 calls and `USD 1.02` for the
605-call maximum, under a `USD 5` cap. The exact figures are in the manifest. These are evidence
from the accepted run, not a guarantee for a future run.

## Reproduce the release privacy seal

The generic scanner safely handles structured JSON numbers but not the evaluation HTML's visible
and embedded typed latency values. Scan the five non-HTML files with the generic profile, then use
the exact `html-report-structured-numeric-v1` verifier below for the HTML. The verifier removes
only typed numeric JSON nodes from detector input; exact fixture matching still uses the original
HTML bytes.

```powershell
$releaseRoot = (Resolve-Path `
  'evaluations/releases/phase12_20260807t030340z-954bb3e2').Path
$acceptedImageId = 'sha256:592adf63fcb0795a2197c67386d9d00e2da012c56bc62a2e303270796e6cfe29'
$fixtureFile = (Resolve-Path 'evaluations/privacy/supported-fixtures-v1.json').Path
docker run --rm --user 10001:10001 --read-only --cap-drop ALL `
  --security-opt no-new-privileges:true --pids-limit 64 `
  --mount "type=bind,source=$releaseRoot,target=/release,readonly" `
  --mount "type=bind,source=$fixtureFile,target=/inputs/fixtures.json,readonly" `
  $acceptedImageId python -m rag_mvp.safety.scan_artifacts /release `
  --fixtures /inputs/fixtures.json --exclude /release/evaluation-report.html
if ($LASTEXITCODE -ne 0) { throw 'non-HTML release privacy scan failed' }

$semanticVerifier = Join-Path $env:TEMP 'rag-mvp-scan-semantic-html.py'
$semanticVerifierSource = @'
from __future__ import annotations

import hashlib
import html
import json
import sys
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

from rag_mvp.safety.scan_artifacts import (
    _scan_text,
    _without_json_numbers,
    load_fixture_set,
)


class _SemanticHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._structured_tags: list[str] = []
        self._json_script_depth = 0
        self._json_script_parts: list[str] = []
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {name.casefold(): value for name, value in attrs}
        if attributes.get("data-report-pointer") is not None:
            self._structured_tags.append(tag.casefold())
        if tag.casefold() == "script" and attributes.get("type") == "application/json":
            self._json_script_depth += 1
        for name in ("alt", "title", "href", "src", "value"):
            value = attributes.get(name)
            if value:
                self.parts.append(value)

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "script" and self._json_script_depth:
            self._json_script_depth -= 1
            if self._json_script_depth == 0:
                self._append_structured("".join(self._json_script_parts))
                self._json_script_parts.clear()
        if self._structured_tags and self._structured_tags[-1] == tag.casefold():
            self._structured_tags.pop()

    def handle_data(self, data: str) -> None:
        if self._json_script_depth:
            self._json_script_parts.append(data)
        elif self._structured_tags:
            self._append_structured(data)
        else:
            self.parts.append(data)

    def _append_structured(self, data: str) -> None:
        decoded = html.unescape(data).strip()
        if not decoded:
            return
        try:
            value = json.loads(decoded)
        except (json.JSONDecodeError, TypeError, ValueError):
            self.parts.append(decoded)
            return
        if isinstance(value, int | float) and not isinstance(value, bool):
            self.parts.append("null")
            return
        self.parts.append(
            json.dumps(
                _without_json_numbers(value),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
        )


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: scan-semantic-html.py HTML FIXTURES", file=sys.stderr)
        return 2
    html_path = Path(sys.argv[1])
    fixture_path = Path(sys.argv[2])
    try:
        raw = html_path.read_bytes()
        content = raw.decode("utf-8")
        fixture_set = load_fixture_set(fixture_path)
        parser = _SemanticHtmlParser()
        parser.feed(content)
        parser.close()
        detector_text = "\n".join(parser.parts)
        fixture_counts, detector_counts = _scan_text(
            f"{html_path.name}\n{content}",
            fixture_set.fixtures,
            detector_text=f"{html_path.name}\n{detector_text}",
        )
    except (OSError, UnicodeError, TypeError, ValueError):
        print("semantic HTML privacy scan failed", file=sys.stderr)
        return 1

    fixture_total = sum(fixture_counts.values())
    detector_total = sum(detector_counts.values())
    result = {
        "schema_version": "phase12-semantic-html-privacy-scan-v1",
        "profile": "html-report-structured-numeric-v1",
        "source_sha256": f"sha256:{hashlib.sha256(raw).hexdigest()}",
        "semantic_projection_sha256": (
            f"sha256:{hashlib.sha256(detector_text.encode('utf-8')).hexdigest()}"
        ),
        "fixture_version": fixture_set.version,
        "fixture_sha256": fixture_set.digest,
        "categories": {
            "fixture": dict(sorted(Counter(fixture_counts).items())),
            "detector": dict(sorted(Counter(detector_counts).items())),
        },
        "counts": {
            "fixture_matches": fixture_total,
            "detector_matches": detector_total,
            "errors": 0,
        },
        "passed": fixture_total == 0 and detector_total == 0,
    }
    print(json.dumps(result, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
'@
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$verifierWriter = New-Object System.IO.StreamWriter(
  $semanticVerifier, $false, $utf8NoBom
)
$verifierWriter.Write($semanticVerifierSource + "`n")
$verifierWriter.Dispose()
$verifierHash = 'sha256:' + (Get-FileHash -Algorithm SHA256 -LiteralPath `
  $semanticVerifier).Hash.ToLowerInvariant()
if ($verifierHash -ne 'sha256:744d4ef3ba43aecb606965e018d1ee2e0d710f669f48c171fe9a6a3a89b77a3a') {
  throw 'semantic verifier source hash mismatch'
}

docker run --rm --user 10001:10001 --read-only --cap-drop ALL `
  --security-opt no-new-privileges:true --pids-limit 64 `
  --mount "type=bind,source=$releaseRoot,target=/release,readonly" `
  --mount "type=bind,source=$fixtureFile,target=/inputs/fixtures.json,readonly" `
  --mount "type=bind,source=$semanticVerifier,target=/tools/scan-semantic-html.py,readonly" `
  $acceptedImageId python /tools/scan-semantic-html.py `
  /release/evaluation-report.html /inputs/fixtures.json
if ($LASTEXITCODE -ne 0) { throw 'semantic HTML privacy scan failed' }

Get-ChildItem -LiteralPath $releaseRoot -File | Sort-Object Name | ForEach-Object {
  [pscustomobject]@{
    path = $_.Name
    sha256 = 'sha256:' + (Get-FileHash -Algorithm SHA256 -LiteralPath `
      $_.FullName).Hash.ToLowerInvariant()
  }
} | ConvertTo-Json
```

The final six-file seal is stored outside the release directory so neither the manifest nor the
privacy record attempts to hash itself. After sealing, changing any of the six files invalidates
the release.

## Known limitations

- The MVP is local, single-process, single-instance, and single-writer. Embedded SQLite, Chroma,
  and BM25 storage are not horizontally scalable.
- There is no authentication. The service is intended for loopback/trusted local use only and
  must not be exposed directly to the internet.
- The accepted image identity is local-engine evidence, not a verified remote registry artifact.
- Reranking is disabled; accepted retrieval is deterministic hybrid BM25+dense weighted RRF.
- The dataset has eight fixed synthetic bilingual cases and does not establish open-domain quality.
- The project's supported PII detector is not a general DLP system.
- Provider latency, quota, pricing, and model output can vary. Full quality/load reproduction makes
  paid API calls.
- The accepted error rate is about `0.9823%`, below but close to the strict `<1%` threshold.
- One Chinese answerable case missed its individual style check; aggregate style and the final
  versioned quality gate still passed.
- The accepted load is measured attempt 2. Attempt 1 achieved 504 successes and acceptable latency
  but was rejected because one timed-out provider attempt made cost evidence incomplete. Its raw
  evidence is retained externally and is not substituted into this accepted bundle.
- The raw HTML's 12 generic phone/card candidates are six typed latency numbers rendered once in a
  visible structured block and once in canonical embedded JSON. The semantic profile proves zero
  fixture or detector matches without weakening exact matching on strings.
- Raw Trivy JSON is excluded from publication because it includes public third-party maintainer
  email metadata. The security gate summary still records the exact scanner identity and finding
  counts.
