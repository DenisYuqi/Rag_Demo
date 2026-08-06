# Local container deployment and evidence runbook

Run every command below in the same PowerShell session from the repository
root. The procedure creates a uniquely named, previously absent data volume for
each run. It never removes a volume. The application has no authentication, so
keep the default `127.0.0.1` binding and never publish it directly to a LAN or
the internet.

All retained evidence, scan output, and backups are written below the current
user's `LocalApplicationData` directory, outside the repository and outside the
Docker build context. The sample corpus is non-sensitive; do not substitute
private documents when producing release evidence.

## Initialize an isolated evidence run

Install Docker with Compose v2, Git, and Trivy 0.73.0 or newer. Initialize a
unique run, assert that its data volume did not already exist, and create it
with identifying labels:

```powershell
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

function Assert-NativeSuccess([string] $Operation) {
    if ($LASTEXITCODE -ne 0) {
        throw "$Operation failed with exit code $LASTEXITCODE."
    }
}

$repoRoot = (Resolve-Path '.').Path
$expectedRevision = (git rev-parse --verify HEAD).Trim()
Assert-NativeSuccess 'Resolve Git revision'
$releaseInputPaths = @(
    '.dockerignore', '.env.example', '.trivyignore', 'Dockerfile', 'compose.yaml',
    'pyproject.toml', 'uv.lock', 'src', 'docs/local-container-runbook.md',
    'docs/container-security-review.md',
    'docker/provider-key.placeholder',
    'evaluations/datasets/mvp-v1/corpus/sources/benefits-policy-en.md',
    'evaluations/performance/acceptance-scenarios-v1.json',
    'evaluations/pricing/openai-standard-2026-08-07.json',
    'evaluations/privacy/supported-fixtures-v1.json'
)
$buildInputStatus = @(git status --porcelain=v1 --untracked-files=all -- $releaseInputPaths)
Assert-NativeSuccess 'Check exact release evidence inputs'
if ($buildInputStatus.Count -ne 0) {
    throw 'Release evidence inputs must be committed and clean before evidence collection.'
}

$runId = ((Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ') + '-' +
    [Guid]::NewGuid().ToString('N').Substring(0, 8)).ToLowerInvariant()
$localRoot = Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'rag-mvp'
$evidenceRoot = Join-Path $localRoot (Join-Path 'evidence' $runId)
$backupDir = Join-Path $localRoot (Join-Path 'backups' $runId)

$repoPrefix = [IO.Path]::GetFullPath($repoRoot).TrimEnd('\') + '\'
foreach ($retainedPath in @($evidenceRoot, $backupDir)) {
    $fullRetainedPath = [IO.Path]::GetFullPath($retainedPath)
    if ($fullRetainedPath.StartsWith($repoPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Retained output must be outside the repository: $fullRetainedPath"
    }
    New-Item -ItemType Directory -Force -Path $fullRetainedPath | Out-Null
}

function Write-JsonEvidence([string] $Name, [object] $Value) {
    $json = $Value | ConvertTo-Json -Depth 50
    [IO.File]::WriteAllText(
        (Join-Path $evidenceRoot $Name),
        $json,
        [Text.UTF8Encoding]::new($false)
    )
}

$buildInputFiles = @(git ls-files -- $releaseInputPaths)
Assert-NativeSuccess 'Enumerate committed release evidence inputs'
$buildInputManifest = @($buildInputFiles | Sort-Object | ForEach-Object {
    [ordered]@{
        path = $_.Replace('\', '/')
        sha256 = (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLowerInvariant()
    }
})
Write-JsonEvidence 'container-build-inputs.json' $buildInputManifest

$actualContextFiles = @('pyproject.toml', 'uv.lock') + @(
    Get-ChildItem -LiteralPath (Join-Path $repoRoot 'src') -Recurse -File |
        Where-Object { $_.Extension -in @('.py', '.json', '.j2') } |
        ForEach-Object { $_.FullName.Substring($repoPrefix.Length).Replace('\', '/') }
)
$trackedContextFiles = @(git ls-files -- pyproject.toml uv.lock src | Where-Object {
    $_ -in @('pyproject.toml', 'uv.lock') -or [IO.Path]::GetExtension($_) -in @('.py', '.json', '.j2')
})
Assert-NativeSuccess 'Enumerate tracked Docker context files'
$contextDelta = @(Compare-Object `
    -ReferenceObject @($trackedContextFiles | Sort-Object) `
    -DifferenceObject @($actualContextFiles | Sort-Object))
if ($contextDelta.Count -ne 0) {
    throw 'Docker context contains an untracked or missing allowlisted source file.'
}

$dataVolume = "rag-mvp-phase11-$runId"
$existingVolumes = @(docker volume ls --quiet)
Assert-NativeSuccess 'List Docker volumes'
if ($existingVolumes -contains $dataVolume) {
    throw "Refusing to reuse existing volume $dataVolume."
}

$createdVolume = (docker volume create `
    --label 'com.rag-mvp.purpose=phase11-evidence' `
    --label "com.rag-mvp.run-id=$runId" `
    $dataVolume).Trim()
Assert-NativeSuccess 'Create clean evidence volume'
if ($createdVolume -ne $dataVolume) {
    throw "Docker created unexpected volume $createdVolume."
}
$volumeInspect = @(docker volume inspect $dataVolume | ConvertFrom-Json)
Assert-NativeSuccess 'Inspect clean evidence volume'
if ($volumeInspect.Count -ne 1 -or $volumeInspect[0].Name -ne $dataVolume) {
    throw 'The clean evidence volume could not be confirmed.'
}
Write-JsonEvidence 'volume-created.json' $volumeInspect[0]

$env:RAG_MVP_DATA_VOLUME = $dataVolume
$env:RAG_MVP_SOURCE_REVISION = $expectedRevision
$env:COMPOSE_DISABLE_ENV_FILE = 'true'
$env:RAG_MVP_BIND_ADDRESS = '127.0.0.1'
$env:RAG_MVP_HOST_PORT = '8000'
$imageRef = if ($env:RAG_MVP_IMAGE) { $env:RAG_MVP_IMAGE } else { 'rag-mvp:dev' }
$env:RAG_MVP_IMAGE = $imageRef

$manifest = [ordered]@{
    schema_version = 'rag-mvp-phase11-container-evidence/v1'
    run_id = $runId
    started_at_utc = (Get-Date).ToUniversalTime().ToString('o')
    repository_revision = $expectedRevision
    image_ref = $imageRef
    source_data_volume = $dataVolume
    restored_data_volume = $null
    container_ids = @()
    ingestion_job_id = $null
    citation_identity = @()
    security_gates = [ordered]@{}
    artifacts = @()
}
Write-JsonEvidence 'manifest.in-progress.json' $manifest
```

The GUID suffix makes collisions extremely unlikely; the explicit pre-create
list check and post-create inspection turn non-reuse into recorded evidence.
Do not replace `$dataVolume` with the shared Compose fallback.

## Configure the runtime secret

Store the provider key outside the repository. An existing non-empty path in
`RAG_MVP_OPENAI_API_KEY_SECRET_FILE` is reused without reading it to the
terminal; otherwise these commands prompt without displaying the value and
write UTF-8 without a BOM:

```powershell
$configuredSecret = $env:RAG_MVP_OPENAI_API_KEY_SECRET_FILE
if ($configuredSecret -and (Test-Path -LiteralPath $configuredSecret) -and
    (Get-Item -LiteralPath $configuredSecret).Length -gt 0) {
    $secretPath = (Resolve-Path -LiteralPath $configuredSecret).Path
} else {
    $secretRoot = Join-Path $localRoot 'secrets'
    New-Item -ItemType Directory -Force -Path $secretRoot | Out-Null
    $secretPath = Join-Path $secretRoot 'openai_api_key'
    $secureKey = Read-Host 'OpenAI-compatible API key' -AsSecureString
    $keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
    try {
        $keyText = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer)
        [IO.File]::WriteAllText($secretPath, $keyText, [Text.UTF8Encoding]::new($false))
    } finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
        Remove-Variable keyText, keyPointer, secureKey -ErrorAction SilentlyContinue
    }
}
$secretPath = [IO.Path]::GetFullPath($secretPath)
if ($secretPath -eq $repoRoot -or
    $secretPath.StartsWith($repoPrefix, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Runtime secret path must remain outside the repository.'
}
$env:RAG_MVP_OPENAI_API_KEY_SECRET_FILE = $secretPath
```

Compose mounts this file as `/run/secrets/openai_api_key`; the credential is
not rendered by `docker compose config` and is not an image build argument. If
the file is absent or empty, the process remains live but `/readyz` safely
reports `provider_credentials_missing`.

## Validate, build, and assert image identity

The release-relevant build must carry the exact checked-out revision. Printing
the label is insufficient: the following block fails if it differs from Git
`HEAD` or if the runtime user is not UID/GID 10001.

```powershell
docker compose config --quiet
Assert-NativeSuccess 'Validate Compose model'
$resolvedCompose = docker compose config --format json | ConvertFrom-Json
Assert-NativeSuccess 'Resolve controlled Compose model'
if ($resolvedCompose.services.app.stop_grace_period -ne '20s') {
    throw 'Compose stop grace period must remain pinned to 20 seconds.'
}
$resolvedCompose.secrets.openai_api_key.file = '[EXTERNAL_SECRET_FILE]'
Write-JsonEvidence 'compose-resolved.json' $resolvedCompose
docker compose build --pull
Assert-NativeSuccess 'Build image'

$imageInspect = @(docker image inspect $imageRef | ConvertFrom-Json)
Assert-NativeSuccess 'Inspect built image'
if ($imageInspect.Count -ne 1) { throw "Expected exactly one image for $imageRef." }
$actualRevision = [string]$imageInspect[0].Config.Labels.'org.opencontainers.image.revision'
if ($actualRevision -ne $expectedRevision) {
    throw "OCI revision mismatch: expected $expectedRevision, found $actualRevision."
}
$imageId = [string]$imageInspect[0].Id
if ($imageInspect[0].Os -ne 'linux' -or $imageInspect[0].Architecture -ne 'amd64') {
    throw "Unsupported release image platform: $($imageInspect[0].Os)/$($imageInspect[0].Architecture)."
}

$runtimeIdentity = (docker run --rm --network none --entrypoint /usr/bin/id $imageId).Trim()
Assert-NativeSuccess 'Check runtime identity'
if ($runtimeIdentity -notmatch 'uid=10001\(' -or $runtimeIdentity -notmatch 'gid=10001\(') {
    throw "Image is not configured for UID/GID 10001: $runtimeIdentity"
}

$identityEvidence = [ordered]@{
    expected_revision = $expectedRevision
    actual_revision = $actualRevision
    image_id = $imageId
    os = $imageInspect[0].Os
    architecture = $imageInspect[0].Architecture
    repo_digests = @($imageInspect[0].RepoDigests)
    runtime_identity = $runtimeIdentity
}
Write-JsonEvidence 'image-identity.json' $identityEvidence

function Assert-ContainerImage([string] $ContainerId) {
    $containerImageId = (docker inspect --format '{{.Image}}' $ContainerId).Trim()
    Assert-NativeSuccess 'Inspect running container image ID'
    if ($containerImageId -ne $imageId) {
        throw "Container image changed: expected $imageId, found $containerImageId."
    }
}

function Assert-CleanContainerStop([string] $ContainerId, [string] $EvidenceName) {
    $stoppedInspect = @(docker inspect $ContainerId | ConvertFrom-Json)
    Assert-NativeSuccess 'Inspect stopped container state'
    if ($stoppedInspect.Count -ne 1 -or $stoppedInspect[0].State.Running -or
        $stoppedInspect[0].State.OOMKilled -or $stoppedInspect[0].State.ExitCode -ne 0) {
        throw 'Container did not complete a clean graceful stop.'
    }
    Write-JsonEvidence $EvidenceName ([ordered]@{
        container_id = $ContainerId
        exit_code = $stoppedInspect[0].State.ExitCode
        oom_killed = $stoppedInspect[0].State.OOMKilled
        finished_at = $stoppedInspect[0].State.FinishedAt
        passed = $true
    })
}
```

## Start, ingest, and retain grounded QA evidence

Start the single application service and retain its first container ID. Wait
for liveness before uploading the version-controlled, non-sensitive sample:

```powershell
$baseUrl = "http://127.0.0.1:$($env:RAG_MVP_HOST_PORT)"
docker compose up --detach
Assert-NativeSuccess 'Start application'
$firstContainerId = (docker compose ps --quiet app).Trim()
Assert-NativeSuccess 'Read first container ID'
if (-not $firstContainerId) { throw 'Compose did not return an application container ID.' }
Assert-ContainerImage $firstContainerId
$manifest.container_ids += $firstContainerId

$health = $null
for ($attempt = 0; $attempt -lt 60; $attempt++) {
    try {
        $health = Invoke-RestMethod "$baseUrl/healthz"
        if ($health.status -eq 'alive') { break }
    } catch {
        $health = $null
    }
    Start-Sleep -Seconds 1
}
if ($null -eq $health -or $health.status -ne 'alive') {
    throw 'Service did not become live within 60 seconds.'
}
Write-JsonEvidence 'health-before-ingest.json' $health

$samplePath = (Resolve-Path `
    'evaluations/datasets/mvp-v1/corpus/sources/benefits-policy-en.md').Path
$uploadJson = curl.exe --fail-with-body --silent --show-error `
    --request POST "$baseUrl/api/v1/documents" `
    --form "file=@$samplePath;type=text/markdown" `
    --form 'source_key=container-smoke-benefits-v1' `
    --form 'display_title=Container smoke benefits policy'
Assert-NativeSuccess 'Submit sample ingestion'
$upload = $uploadJson | ConvertFrom-Json
Write-JsonEvidence 'ingestion-submission.json' $upload

$jobUrl = "$baseUrl/api/v1/ingestion-jobs/$($upload.job_id)"
$job = $null
for ($attempt = 0; $attempt -lt 120; $attempt++) {
    Start-Sleep -Seconds 1
    $job = Invoke-RestMethod $jobUrl
    if ($job.status -in @('succeeded', 'failed')) { break }
}
if ($null -eq $job -or $job.status -ne 'succeeded') {
    throw "Ingestion failed with safe code: $($job.safe_error_code)"
}
$manifest.ingestion_job_id = $upload.job_id
Write-JsonEvidence 'ingestion-completed.json' $job
```

Wait for all readiness components, then ask one cache-bypassed grounded
question. The response and exact citation identity are retained as evidence.

```powershell
function Wait-RagReady([string] $EvidenceName) {
    $ready = $null
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        try {
            $ready = Invoke-RestMethod "$baseUrl/readyz"
            if ($ready.status -eq 'ready') { break }
        } catch {
            $ready = $null
        }
        Start-Sleep -Seconds 1
    }
    if ($null -eq $ready -or $ready.status -ne 'ready') {
        throw 'Service did not become ready within 60 seconds.'
    }
    Write-JsonEvidence $EvidenceName $ready
}

function Invoke-GroundedSmokeQa([string] $EvidenceName) {
    $qaBody = @{
        owner_id = 'container-smoke-owner'
        question = 'How many paid annual-leave days do full-time employees receive each calendar year?'
        mode = 'hybrid'
        requested_language = 'en'
    } | ConvertTo-Json -Compress
    $qaResponse = Invoke-WebRequest -UseBasicParsing `
        -Method POST `
        -Uri "$baseUrl/api/v1/qa" `
        -Headers @{ 'X-RAG-Cache-Policy' = 'bypass' } `
        -ContentType 'application/json' `
        -Body $qaBody
    $qaContent = if ($qaResponse.Content -is [byte[]]) {
        [Text.Encoding]::UTF8.GetString($qaResponse.Content)
    } else {
        [string]$qaResponse.Content
    }
    $qaEvent = $qaContent.Trim() | ConvertFrom-Json
    if ($qaEvent.kind -ne 'answer' -or @($qaEvent.citations).Count -lt 1) {
        throw 'Grounded smoke QA did not return a cited answer.'
    }
    Write-JsonEvidence $EvidenceName $qaEvent
    return $qaEvent
}

function Capture-DockerLogs([string] $ContainerId, [string] $LogPath) {
    $previousErrorActionPreference = $ErrorActionPreference
    $dockerLogsExitCode = 1
    $capturedLines = @()
    try {
        # Windows PowerShell surfaces a native process's stderr as ErrorRecord
        # objects. Container applications legitimately use stderr for logs, so
        # judge this command by Docker's exit code and preserve both streams.
        $ErrorActionPreference = 'Continue'
        $capturedLines = @(docker logs $ContainerId 2>&1 | ForEach-Object {
            $_.ToString()
        })
        $dockerLogsExitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
    }
    if ($dockerLogsExitCode -ne 0) {
        throw "Capture logs for container $ContainerId failed with exit code $dockerLogsExitCode."
    }
    [IO.File]::WriteAllLines(
        $LogPath,
        [string[]]$capturedLines,
        [Text.UTF8Encoding]::new($false)
    )
}

function Capture-AndScanContainerLogs(
    [string] $ContainerId,
    [string] $LogName,
    [string] $GateName,
    [int] $ExpectedShutdownCount
) {
    $logPath = Join-Path $evidenceRoot $LogName
    Capture-DockerLogs $ContainerId $logPath
    $secretText = [IO.File]::ReadAllText($secretPath, [Text.Encoding]::UTF8).Trim()
    $applicationLogText = [IO.File]::ReadAllText($logPath, [Text.Encoding]::UTF8)
    try {
        if (-not $secretText -or $applicationLogText.Contains($secretText)) {
            throw "Provider secret leakage gate failed for $LogName."
        }
        $shutdownEvents = @(Get-Content -LiteralPath $logPath | ForEach-Object {
            try { $_ | ConvertFrom-Json } catch { $null }
        } | Where-Object {
            $null -ne $_ -and $null -ne $_.PSObject.Properties['event'] -and
            $_.event -eq 'runtime.shutdown.sequence.completed'
        })
        $invalidShutdownEvents = @($shutdownEvents | Where-Object {
            $_.outcome -ne 'succeeded' -or $_.counts.pending_tasks -ne 0 -or
            $_.counts.failed_tasks -ne 0
        })
        if ($shutdownEvents.Count -ne $ExpectedShutdownCount -or
            $invalidShutdownEvents.Count -ne 0) {
            throw "Container $ContainerId has incomplete shutdown evidence."
        }
        Write-JsonEvidence $GateName ([ordered]@{
            container_id = $ContainerId
            scanned_artifact = $LogName
            raw_provider_secret_matches = 0
            successful_shutdown_events = $shutdownEvents.Count
            passed = $true
        })
    } finally {
        Remove-Variable secretText, applicationLogText -ErrorAction SilentlyContinue
    }
}

Wait-RagReady 'ready-after-ingest.json'
$firstQa = Invoke-GroundedSmokeQa 'qa-before-recreate.json'
$firstCitationIdentity = @($firstQa.citations | ForEach-Object {
    [ordered]@{
        source_title = $_.source_title
        document_version = $_.document_version
        chunk_id = $_.chunk_id
    }
})
$manifest.citation_identity = $firstCitationIdentity
Write-JsonEvidence 'citation-identity-before-recreate.json' $firstCitationIdentity
```

## Prove persistence across a forced recreate

This section deliberately contains no ingestion command. It force-recreates
the application against the same `$dataVolume`, asserts that the container ID
changed, waits for readiness from persisted state, and repeats the same
cache-bypassed question.

```powershell
if ($env:RAG_MVP_DATA_VOLUME -ne $dataVolume) {
    throw 'Data-volume selection changed before the persistence check.'
}
docker compose stop app
Assert-NativeSuccess 'Stop first application container before recreate'
Assert-CleanContainerStop $firstContainerId 'stop-before-recreate.json'
Capture-AndScanContainerLogs `
    $firstContainerId 'application-first.log.jsonl' `
    'application-first-secret-leak-gate.json' 1
docker compose up --detach --force-recreate --no-deps app
Assert-NativeSuccess 'Force-recreate application'
$secondContainerId = (docker compose ps --quiet app).Trim()
Assert-NativeSuccess 'Read recreated container ID'
if (-not $secondContainerId -or $secondContainerId -eq $firstContainerId) {
    throw 'The application container was not recreated.'
}
Assert-ContainerImage $secondContainerId
$manifest.container_ids += $secondContainerId

Wait-RagReady 'ready-after-recreate.json'
$secondQa = Invoke-GroundedSmokeQa 'qa-after-recreate.json'
$secondCitationIdentity = @($secondQa.citations | ForEach-Object {
    [ordered]@{
        source_title = $_.source_title
        document_version = $_.document_version
        chunk_id = $_.chunk_id
    }
})
Write-JsonEvidence 'citation-identity-after-recreate.json' $secondCitationIdentity

$firstCitationJson = $firstCitationIdentity | ConvertTo-Json -Depth 10 -Compress
$secondCitationJson = $secondCitationIdentity | ConvertTo-Json -Depth 10 -Compress
if ($secondCitationJson -ne $firstCitationJson) {
    throw 'The same grounded question did not return the same persisted citation identity.'
}

Write-JsonEvidence 'recreate-persistence.json' ([ordered]@{
    data_volume = $dataVolume
    container_id_before = $firstContainerId
    container_id_after = $secondContainerId
    container_id_changed = $true
    reingested = $false
    citation_identity_equal = $true
})
```

## Back up outside the repository

Stop the service before archiving so SQLite, Chroma, BM25, source artifacts,
metadata, and reports form one consistent snapshot. The provider secret is a
separate mount and is not included. The backup remains under
`LocalApplicationData`, not `$PWD`.

```powershell
docker compose stop app
Assert-NativeSuccess 'Stop application for backup'
Assert-CleanContainerStop $secondContainerId 'stop-before-backup.json'
$backupName = "rag-mvp-data-$runId.tgz"
$backupPath = Join-Path $backupDir $backupName
$helperImage = 'docker.io/library/alpine:3.22.1@sha256:4bcff63911fcb4448bd4fdacec207030997caf25e9bea4045fa6c8c44de311d1'
docker run --rm --platform linux/amd64 --network none --read-only `
    --user 10001:10001 --cap-drop ALL `
    --security-opt no-new-privileges `
    --mount "type=volume,src=$dataVolume,dst=/data,readonly" `
    --mount "type=bind,src=$backupDir,dst=/backup" `
    $helperImage tar -czf "/backup/$backupName" -C /data .
Assert-NativeSuccess 'Back up data volume'
if (-not (Test-Path -LiteralPath $backupPath)) { throw 'Backup archive was not created.' }
$backupHash = (Get-FileHash -LiteralPath $backupPath -Algorithm SHA256).Hash.ToLowerInvariant()
Write-JsonEvidence 'backup.json' ([ordered]@{
    path = $backupPath
    sha256 = $backupHash
    source_volume = $dataVolume
})
docker compose start app
Assert-NativeSuccess 'Restart application after backup'
Wait-RagReady 'ready-after-backup.json'
```

Backups contain uploaded document content and must receive the same access
controls as the source corpus.

## Restore into another new volume

Restore into a second, previously absent named volume. This avoids overwriting
the current volume and preserves both copies for rollback and audit.

```powershell
docker compose stop app
Assert-NativeSuccess 'Stop source-volume deployment before restore'
Assert-CleanContainerStop $secondContainerId 'stop-before-restore.json'
Capture-AndScanContainerLogs `
    $secondContainerId 'application-second.log.jsonl' `
    'application-second-secret-leak-gate.json' 2
docker compose down --remove-orphans
Assert-NativeSuccess 'Remove source-volume deployment'
$restoreVolume = "rag-mvp-phase11-restored-$runId"
$existingVolumes = @(docker volume ls --quiet)
Assert-NativeSuccess 'List Docker volumes before restore'
if ($existingVolumes -contains $restoreVolume) {
    throw "Refusing to overwrite existing restore volume $restoreVolume."
}
docker volume create `
    --label 'com.rag-mvp.purpose=phase11-restored-evidence' `
    --label "com.rag-mvp.run-id=$runId" `
    $restoreVolume | Out-Null
Assert-NativeSuccess 'Create restore volume'

# Mounting an empty named volume at the image's pre-owned data directory lets
# Docker initialize the volume root with the runtime UID/GID and mode. The
# command itself has no network, runs as the non-root runtime user, and cannot
# mutate the read-only image filesystem.
docker run --rm --platform linux/amd64 --network none --read-only `
    --user 10001:10001 --cap-drop ALL `
    --security-opt no-new-privileges `
    --mount "type=volume,src=$restoreVolume,dst=/var/lib/rag-mvp" `
    --entrypoint /usr/bin/id `
    $imageId | Out-Null
Assert-NativeSuccess 'Initialize restore volume ownership'

$restoreEntries = @(docker run --rm --platform linux/amd64 --network none --read-only `
    --user 10001:10001 --cap-drop ALL `
    --security-opt no-new-privileges `
    --mount "type=volume,src=$restoreVolume,dst=/data,readonly" `
    $helperImage find /data -mindepth 1 -maxdepth 1 -print -quit)
Assert-NativeSuccess 'Verify restore volume is empty'
if ($restoreEntries.Count -ne 0) {
    throw 'Restore volume was not empty after ownership initialization.'
}

docker run --rm --platform linux/amd64 --network none --read-only `
    --user 10001:10001 --cap-drop ALL `
    --security-opt no-new-privileges `
    --mount "type=volume,src=$restoreVolume,dst=/data" `
    --mount "type=bind,src=$backupPath,dst=/backup/restore.tgz,readonly" `
    $helperImage tar -xzf /backup/restore.tgz -C /data
Assert-NativeSuccess 'Restore data into empty volume'

$env:RAG_MVP_DATA_VOLUME = $restoreVolume
$manifest.restored_data_volume = $restoreVolume
docker compose up --detach
Assert-NativeSuccess 'Start restored deployment'
$restoredContainerId = (docker compose ps --quiet app).Trim()
Assert-NativeSuccess 'Read restored container ID'
Assert-ContainerImage $restoredContainerId
$manifest.container_ids += $restoredContainerId
Wait-RagReady 'ready-after-restore.json'
$restoredQa = Invoke-GroundedSmokeQa 'qa-after-restore.json'
$restoredCitationIdentity = @($restoredQa.citations | ForEach-Object {
    [ordered]@{
        source_title = $_.source_title
        document_version = $_.document_version
        chunk_id = $_.chunk_id
    }
})
if (($restoredCitationIdentity | ConvertTo-Json -Depth 10 -Compress) -ne
    $firstCitationJson) {
    throw 'Restored data did not produce the original citation identity.'
}
Write-JsonEvidence 'restore-persistence.json' ([ordered]@{
    restored_volume = $restoreVolume
    reingested = $false
    citation_identity_equal = $true
})
```

## Retain raw security evidence and enforce release gates

The raw vulnerability report must include every database match, including
reviewed exceptions. Point it at an intentionally empty ignore file. The
Critical policy gate then explicitly uses the repository's reviewed
`.trivyignore`. All three machine-readable reports stay in the external
evidence directory.

```powershell
$trivyCommand = Get-Command trivy -ErrorAction SilentlyContinue | Select-Object -First 1
if ($null -eq $trivyCommand) {
    $wingetTrivy = Get-ChildItem `
        (Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) `
            'Microsoft\WinGet\Packages\AquaSecurity.Trivy_*\trivy.exe') `
        -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $wingetTrivy) { throw 'Trivy executable was not found.' }
    $trivyExe = $wingetTrivy.FullName
} else {
    $trivyExe = $trivyCommand.Source
}
$trivyVersionPath = Join-Path $evidenceRoot 'trivy-version.txt'
$trivyVersionJson = & $trivyExe version --format json
Assert-NativeSuccess 'Record Trivy version'
$trivyVersionJson | Set-Content -Encoding utf8 $trivyVersionPath
$trivyVersion = $trivyVersionJson | ConvertFrom-Json
if ([version]$trivyVersion.Version -lt [version]'0.73.0') {
    throw "Trivy 0.73.0 or newer is required; found $($trivyVersion.Version)."
}

$emptyIgnorePath = Join-Path $evidenceRoot 'trivy-empty-ignore.txt'
[IO.File]::WriteAllText($emptyIgnorePath, '', [Text.UTF8Encoding]::new($false))
$rawScanPath = Join-Path $evidenceRoot 'trivy-high-critical.raw.json'
$secretGatePath = Join-Path $evidenceRoot 'trivy-secret.gate.json'
$criticalGatePath = Join-Path $evidenceRoot 'trivy-critical-policy.gate.json'

$chromaRouteStatus = curl.exe --silent --output NUL --write-out '%{http_code}' `
    "$baseUrl/api/v2/tenants/default/databases/default/collections"
Assert-NativeSuccess 'Probe absent Chroma server route'
if ($chromaRouteStatus -ne '404') { throw 'Chroma server route must remain absent.' }
Write-JsonEvidence 'chroma-route-absence.json' ([ordered]@{
    route = '/api/v2/tenants/default/databases/default/collections'
    http_status = 404
    passed = $true
})

& $trivyExe image --scanners vuln --ignorefile $emptyIgnorePath `
    --severity HIGH,CRITICAL --format json --output $rawScanPath $imageId
Assert-NativeSuccess 'Capture raw HIGH/CRITICAL vulnerability report'
& $trivyExe image --scanners secret --exit-code 1 `
    --format json --output $secretGatePath $imageId
Assert-NativeSuccess 'Enforce image secret gate'
& $trivyExe image --scanners vuln --ignorefile .trivyignore --exit-code 1 `
    --severity CRITICAL --format json --output $criticalGatePath $imageId
Assert-NativeSuccess 'Enforce unresolved Critical vulnerability gate'

$manifest.security_gates = [ordered]@{
    raw_high_critical_report = $rawScanPath
    secret_gate_report = $secretGatePath
    critical_policy_gate_report = $criticalGatePath
    secret_gate_passed = $true
    critical_policy_gate_passed = $true
    exception_review = 'docs/container-security-review.md'
}
```

Review every raw finding against
`docs/container-security-review.md`. A database or base-image update requires a
fresh review; a successful policy gate is not a substitute for the raw report.

## Stop safely and seal the manifest

Stop and remove only the Compose container/network. Never pass `--volumes` or
run `docker volume rm` in this procedure. Finally, prove that both named
volumes remain, hash every retained evidence artifact, and seal the manifest.

```powershell
docker compose stop app
Assert-NativeSuccess 'Gracefully stop application'
Assert-CleanContainerStop $restoredContainerId 'stop-final.json'
$finalLogPath = Join-Path $evidenceRoot 'application-final.log.jsonl'
Capture-DockerLogs $restoredContainerId $finalLogPath
$shutdownEvents = @(Get-Content -LiteralPath $finalLogPath | ForEach-Object {
    try { $_ | ConvertFrom-Json } catch { $null }
} | Where-Object {
    $null -ne $_ -and $null -ne $_.PSObject.Properties['event'] -and
    $_.event -eq 'runtime.shutdown.sequence.completed'
})
if ($shutdownEvents.Count -ne 1 -or $shutdownEvents[-1].outcome -ne 'succeeded' -or
    $shutdownEvents[-1].counts.pending_tasks -ne 0 -or
    $shutdownEvents[-1].counts.failed_tasks -ne 0) {
    throw 'Final runtime shutdown did not prove completed resource cleanup.'
}
$secretText = [IO.File]::ReadAllText($secretPath, [Text.Encoding]::UTF8).Trim()
$finalLogText = [IO.File]::ReadAllText($finalLogPath, [Text.Encoding]::UTF8)
try {
    if (-not $secretText -or $finalLogText.Contains($secretText)) {
        throw 'Final provider secret leakage gate failed.'
    }
} finally {
    Remove-Variable secretText, finalLogText -ErrorAction SilentlyContinue
}
Write-JsonEvidence 'shutdown-cleanup-gate.json' ([ordered]@{
    event = 'runtime.shutdown.sequence.completed'
    pending_tasks = 0
    failed_tasks = 0
    successful_shutdown_events = 1
    raw_provider_secret_matches = 0
    passed = $true
})
docker compose down --remove-orphans
Assert-NativeSuccess 'Remove stopped Compose container and network'

$retainedVolumeDocument = docker volume inspect $dataVolume $restoreVolume | ConvertFrom-Json
Assert-NativeSuccess 'Confirm evidence volumes remain'
$retainedVolumes = @()
foreach ($retainedVolume in $retainedVolumeDocument) {
    $retainedVolumes += $retainedVolume
}
$retainedVolumeNames = @($retainedVolumes | ForEach-Object { $_.Name })
if ($retainedVolumeNames -notcontains $dataVolume -or
    $retainedVolumeNames -notcontains $restoreVolume) {
    throw 'One or more evidence volumes were unexpectedly removed.'
}
Write-JsonEvidence 'retained-volumes.json' $retainedVolumes

$manifest.completed_at_utc = (Get-Date).ToUniversalTime().ToString('o')
$manifest.artifacts = @(Get-ChildItem -LiteralPath $evidenceRoot -File |
    Where-Object { $_.Name -notin @('manifest.in-progress.json', 'manifest.json', 'manifest.sha256') } |
    Sort-Object Name |
    ForEach-Object {
        [ordered]@{
            name = $_.Name
            sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        }
    })
Write-JsonEvidence 'manifest.json' $manifest
$manifestHash = (Get-FileHash (Join-Path $evidenceRoot 'manifest.json') `
    -Algorithm SHA256).Hash.ToLowerInvariant()
[IO.File]::WriteAllText(
    (Join-Path $evidenceRoot 'manifest.sha256'),
    "$manifestHash  manifest.json`n",
    [Text.UTF8Encoding]::new($false)
)
Remove-Item (Join-Path $evidenceRoot 'manifest.in-progress.json')

Remove-Item Env:RAG_MVP_OPENAI_API_KEY_SECRET_FILE -ErrorAction SilentlyContinue
Remove-Item Env:RAG_MVP_SOURCE_REVISION -ErrorAction SilentlyContinue
Remove-Item Env:RAG_MVP_DATA_VOLUME -ErrorAction SilentlyContinue
Remove-Item Env:RAG_MVP_IMAGE -ErrorAction SilentlyContinue
Remove-Item Env:RAG_MVP_BIND_ADDRESS -ErrorAction SilentlyContinue
Remove-Item Env:RAG_MVP_HOST_PORT -ErrorAction SilentlyContinue
Remove-Item Env:COMPOSE_DISABLE_ENV_FILE -ErrorAction SilentlyContinue
Write-Host "Evidence retained at: $evidenceRoot"
Write-Host "Backup retained at:   $backupPath"
Write-Host "Volumes retained:     $dataVolume, $restoreVolume"
```

## Single-instance limit

Do not use `--scale app=2`, multiple Compose projects against the same volume,
multiple Uvicorn workers, or a second writer. `container_name`, one declared
replica, one mounted read-write volume, and the application writer lock enforce
the supported local topology. Horizontal replicas require a networked vector
store plus shared metadata, locking, and artifact services; that architecture
is outside this local deployment.
