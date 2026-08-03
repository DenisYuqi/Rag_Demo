# ADDED Requirements

### Requirement: Reproducible production container
The project SHALL provide a multi-stage Dockerfile that builds an OCI-compatible image from pinned dependencies. The runtime image MUST run as a non-root user, contain only required runtime artifacts, handle termination signals, and expose only the configured application port.

#### Scenario: Build from a clean checkout
- **WHEN** the documented Docker build command runs with lock files present
- **THEN** it SHALL produce a runnable image labeled with application version and source revision

#### Scenario: Inspect runtime identity
- **WHEN** the container starts
- **THEN** the application process SHALL run as a non-root user

### Requirement: External configuration and secrets
Runtime behavior SHALL be configured through documented environment variables and mounted files. Credentials MUST NOT be embedded in the image, Compose file, source-controlled defaults, image history, diagnostics, or logs.

#### Scenario: Start without a required secret
- **WHEN** required model-provider credentials are absent
- **THEN** the container SHALL remain unhealthy and expose only a safe configuration reason

#### Scenario: Supply runtime configuration
- **WHEN** valid environment variables or secret-file references are supplied
- **THEN** the same image SHALL start without a rebuild

### Requirement: Persistent local data volume
Chroma, BM25, uploaded-source artifacts, metadata, and evaluation reports SHALL reside below one configurable data root that can be mounted as a Docker volume. The service SHALL validate that the root is writable before readiness succeeds.

#### Scenario: Restart the container
- **WHEN** a container with indexed documents is replaced using the same mounted volume
- **THEN** the new container SHALL load the same active index and reports without re-ingestion

#### Scenario: Data volume is not writable
- **WHEN** the configured data root cannot be written
- **THEN** readiness SHALL fail and indexing or QA traffic MUST NOT proceed

### Requirement: Single-instance embedded Chroma operation
The supplied local deployment SHALL run exactly one application replica and one writer against the embedded Chroma data. Documentation MUST state that horizontal replicas require migration to a networked vector store and shared metadata services.

#### Scenario: Start with Docker Compose
- **WHEN** the documented Compose command runs
- **THEN** exactly one application service instance SHALL mount the persistent data volume read-write

#### Scenario: User requests horizontal scaling
- **WHEN** an operator attempts to scale the embedded deployment above one replica
- **THEN** documentation and startup locking SHALL prevent unsupported concurrent writers or clearly fail before serving traffic

### Requirement: Health and graceful lifecycle
The application SHALL expose separate `/healthz` and `/readyz` endpoints. Liveness SHALL verify process health; readiness SHALL require valid configuration, writable storage, validated indexes, required model providers, and safety controls. Shutdown SHALL stop accepting traffic, cancel or drain work within a grace period, flush telemetry, and close persistent stores.

#### Scenario: Container is initializing
- **WHEN** required components are not ready
- **THEN** liveness MAY succeed while readiness fails

#### Scenario: Container receives termination
- **WHEN** the runtime sends a termination signal
- **THEN** the application SHALL become unready and close resources within the configured grace period

### Requirement: Local Compose deployment
The project SHALL provide a Docker Compose configuration and `.env.example` that expose the workbench/API, mount persistent data, configure health checks, and accept external model credentials. The documentation SHALL provide build, smoke-test, monitor, log-inspection, backup, restore, and stop commands.

#### Scenario: Follow the local deployment guide
- **WHEN** a user supplies valid credentials and executes the documented commands
- **THEN** the service SHALL become ready, retain data across restarts, and answer a smoke-test question after ingestion

### Requirement: Container security and artifact checks
66 The project SHALL scan the built image for known high or critical vulnerabilities and inspect it for embedded credentials before release. Unresolved critical vulnerabilities or detected secrets MUST fail the container release gate.

#### Scenario: Release checks find a critical issue
- **WHEN** image scanning reports an unresolved critical vulnerability or a secret is found in image layers
- **THEN** the release verification SHALL fail

### Requirement: Cloud-portable boundaries
The container SHALL avoid hard-coded local hostnames, storage paths, model vendors, and telemetry backends. Provider, storage-root, base-URL, and telemetry-export configuration MUST remain external so a future cloud deployment can reuse the image or application modules.

#### Scenario: Change a compatible model endpoint
- **WHEN** an operator supplies a different OpenAI-compatible base URL and model configuration
- **THEN** the container SHALL use it without source modification or image rebuild

#### Scenario: Future managed deployment is designed
- **WHEN** a large change adds cloud-managed storage, telemetry, or ingress
- **THEN** the RAG domain service SHALL remain independent of those infrastructure implementations
