# Container security review

This review accompanies Trivy 0.73.0 results for the pinned Linux `amd64`
runtime image. The release procedure stores reports outside the repository at
`%LOCALAPPDATA%\rag-mvp\evidence\<run-id>`. It retains an unfiltered raw scan,
then runs the secret and unresolved-Critical policy gates. An exception means
the vulnerable execution path is not present or reachable in this deployment;
it is not a claim that the upstream package has been patched.

| Finding | Disposition | Deployment-specific evidence |
| --- | --- | --- |
| `CVE-2026-45829` (`chromadb`) | Not affected | The issue requires the Chroma HTTP collection endpoint and attacker-controlled `trust_remote_code`. This image uses only an in-process `PersistentClient`; it does not start or mount the Chroma server/API. The only listener is the RAG FastAPI process, and `/api/v2/tenants/.../collections` is absent. |
| `CVE-2026-6653` (`libxml2`) | Not affected | The advisory is for the MinGW build. The accepted image is Linux `amd64`, not MinGW/Windows. |
| `CVE-2026-8376` (`perl-base`) | Not affected | The affected heap-overflow path is limited to 32-bit Perl. The accepted image is Linux `amd64`. |
| `CVE-2026-42496` (`perl-base`) | Not affected | The application never invokes Perl or `Archive::Tar`; uploads support PDF, Markdown, and text and are processed by Python/Tesseract. The container is read-only apart from its bounded data volume and tmpfs. |
| `CVE-2026-57433` (`perl-base`) | Not affected | The application never invokes Perl `Storable`; no RAG route deserializes Perl data. |
| `CVE-2026-13221` (`perl-base`) | Not affected | The application never invokes the Perl regular-expression engine. Python/RE2-compatible application paths do not call this executable or library. |
| `CVE-2026-58016` (`libglib2.0-0t64`) | Not affected | The affected GDBus XML-introspection function is not used. The image has no D-Bus daemon or D-Bus endpoint; GLib is present only as a transitive OCR dependency. |

The Chroma exception must be removed if a Chroma HTTP server is introduced, if
arbitrary embedding-function configuration is accepted, or when upstream ships
a fixed release. The OS exceptions must be re-reviewed on every base-image or
Trivy database update and removed as soon as fixed packages are available.

## Verification commands

Run the raw scan first with an empty ignore file so retained evidence exposes
every database match. `$evidenceRoot`, `$emptyIgnorePath`, and the immutable
`$imageId` are
initialized by `docs/local-container-runbook.md`:

```powershell
trivy image --scanners vuln --ignorefile $emptyIgnorePath `
    --severity HIGH,CRITICAL --format json `
    --output (Join-Path $evidenceRoot 'trivy-high-critical.raw.json') $imageId
```

Then run the release gates from the repository root. The vulnerability gate
names the reviewed `.trivyignore` explicitly, and both gate reports are retained
beside the raw report:

```powershell
trivy image --scanners secret --exit-code 1 --format json `
    --output (Join-Path $evidenceRoot 'trivy-secret.gate.json') $imageId
trivy image --scanners vuln --ignorefile .trivyignore --exit-code 1 `
    --severity CRITICAL --format json `
    --output (Join-Path $evidenceRoot 'trivy-critical-policy.gate.json') $imageId
```

The evidence manifest records these paths and their SHA-256 hashes. Re-run the
raw scan and exception review whenever the base digest, dependency lock, or
Trivy database changes.

The runbook also refuses dirty release-evidence inputs, compares every
allowlisted Docker-context source file with the Git index, and requires the
runtime secret path to resolve outside the repository. Each application
container is stopped, inspected, and has its complete logs scanned for the raw
provider secret before it can be removed. The Compose stop grace period is
pinned to 20 seconds, while validated server and application budgets must total
less than that boundary.

Confirm the Chroma server route is absent against a running application:

```powershell
curl.exe --silent --output NUL --write-out '%{http_code}' `
  http://127.0.0.1:8000/api/v2/tenants/default/databases/default/collections
```

The main runbook fails unless this route returns `404`, records the result, and
also asserts that the inspected release image is Linux `amd64`; both checks are
required preconditions for the reviewed exceptions above.
