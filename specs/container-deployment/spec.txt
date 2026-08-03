# ADDED Requirements

### Requirement: Reproducible production container
4 The project SHALL provide a multi-stage Dockerfile that builds an OCI-compatible image from pinned dependencies.
5 The runtime image MUST run as a non-root user, contain only required runtime artifacts, handle termination signals, and expose only the configured application port.

### Scenario: Build from a clean checkout
6 → WHEN the documented Docker build command runs with lock files present
8 → THEN it SHALL produce a runnable image labeled with application version and source revision

### Scenario: Inspect runtime identity
10 → WHEN the container starts
11 → THEN the application process SHALL run as a non-root user

### Requirement: External configuration and secrets
14 Runtime behavior SHALL be configured through documented environment variables and mounted files. Credentials MUST NOT be embedded in the image, Compose file, source-controlled defaults, image history, diagnostics, or logs.

### Scenario: Start without a required secret
16 → WHEN required model-provider credentials are absent
18 → THEN the container SHALL remain unhealthy and expose only a safe configuration reason

### Scenario: Supply runtime configuration
21 → WHEN valid environment variables or secret-file references are supplied
23 → THEN the same image SHALL start without a rebuild

### Requirement: Persistent local data volume
25 Chroma, BN25, uploaded-source artifacts, metadata, and evaluation reports SHALL reside below one configurable data root that can be mounted as a Docker volume. The service SHALL validate that the root is writable before readiness succeeds.

### Scenario: Restart the container
28 → WHEN a container with indexed documents is replaced using the same mounted volume
30 → THEN the new container SHALL load the same active index and reports without re-ingestion

### Scenario: Data volume is not writable
33 → WHEN the configured data root cannot be written
35 → THEN readiness SHALL fail and indexing or QA traffic MUST NOT proceed

### Requirement: Single-instance embedded Chroma operation
37 The supplied local deployment SHALL run exactly one application replica and one writer against the embedded Chroma data. Documentation MUST state that horizontal replicas require migration to a networked vector store and shared metadata services.

### Scenario: Start with Docker Compose
39 → WHEN the documented Compose command runs
41 → THEN exactly one application service instance SHALL mount the persistent data volume read-write

### Scenario: User requests horizontal scaling
43 → WHEN an operator attempts to scale the embedded deployment above one replica
45 → THEN documentation and startup locking SHALL prevent unsupported concurrent writers or clearly fail before serving traffic

### Requirement: Health and graceful lifecycle
47 The application SHALL expose separate `/healthz` and `/readyz` endpoints. Liveness SHALL verify process health; readiness SHALL require valid configuration, writable storage, validated indexes, required model providers, and safety controls. Shutdown SHALL stop accepting traffic, cancel or drain work within a grace period, flush telemetry, and close persistent stores.

### Scenario: Container is initializing
49 → WHEN required components are not ready
51 → THEN Liveness MAY succeed while readiness fails

### Scenario: Container receives termination
54 → WHEN the runtime sends a termination signal
56 → THEN the application SHALL become unready and close resources within the configured grace period

### Requirement: Local Compose deployment
59 The project SHALL provide a Docker Compose configuration and `.env.example` that expose the workbench/API, mount persistent data, configure health checks, and accept external model credentials. The documentation SHALL provide build, start, monitor, log-inspection, backup, restore, and stop commands.

### Scenario: Follow the local deployment guide
61 → WHEN a user supplies valid credentials and executes the documented commands
63 → THEN the service SHALL start, retain data across restarts, and answer a smoke-test question after startup

### Requirement: Container security and artifact checks
66 The project SHALL scan the built image for known high or critical vulnerabilities and inspect it for embedded credentials before release. Unresolved critical vulnerabilities or detected secrets MUST fail the container release.

### Scenario: Release checks find a critical issue
68 → WHEN image scanning reports an unresolved critical vulnerability or a secret is found in image layers
71 → THEN the release pipeline SHALL fail

### Requirement: Cloud-portable boundaries
73 The container SHALL avoid hard-coded hostnames, storage paths, model vendors, and telemetry backends. Provider, storage bucket, port, and telemetry-support configuration MUST remain external so a future cloud deployment can reuse the image or application modules.

### Scenario: Change a compatible model endpoint
76 → WHEN an operator supplies a different OpenAI-compatible base URL and model configuration
78 → THEN the container SHALL accept the new model source without modification or ingress rework

> Partial bottom text cutoff in screenshot:
> the domain services SHALL use structured logging, telemetry for ingress implementations