## Context

See `proposal.md` for motivation and `specs/profile-aware-evaluation/spec.md` for observable requirements. The executable currently composes Evaluation only around the OpenAI profile. Production evaluation/comparison executors also default to an OpenAI composition factory, while the workbench stores one Evaluation gateway and generates unqualified same-origin artifact URLs. The BGE profile already has an isolated database, corpus, model router, and data root that can safely own a second Evaluation service.

## Goals / Non-Goals

**Goals:**

- Reuse the existing Evaluation application, repositories, artifact verification, and dashboards for both profiles.
- Ensure isolated evaluation workspaces execute through the same retrieval backend as their owning online profile.
- Preserve existing unqualified HTTP endpoints and default API ownership.
- Make profile changes reset browser selection state before rendering another profile's data.

**Non-Goals:**

- Combine OpenAI and BGE runs in one comparison or copy historical runs between profiles.
- Change evaluation datasets, metrics, scoring, report schemas, or quality gates.
- Add a distributed BGE evaluation worker or expose arbitrary profile identifiers.

## Decisions

### Extract reusable Evaluation composition and inject the retrieval runtime factory

Refactor the existing Evaluation construction into a helper that receives truthful profile settings, repositories, layout, redactor, and an isolated composition factory. The OpenAI profile supplies the current OpenAI factory; BGE supplies a factory that composes an already-derived BGE settings object without reapplying the online BGE data root. The same factory is injected into both production Evaluation and Comparison executors.

This keeps plans, reports, provider ledgers, and configuration identities on the existing production path. Building a separate BGE-only evaluation runner was rejected because it would duplicate evidence and release invariants.

### Store one Evaluation service inside every complete profile composition

`compose_openai_services` continues to include Evaluation by default. `compose_bge_services` gains the same opt-in parameter and includes Evaluation for the online BGE profile, while isolated evaluation workspaces always request `include_evaluation=False` to prevent recursive supervisor construction. Each service uses its profile database plus `<profile-root>/evaluations/...` directories.

Static sealed release evidence remains attached only to `openai-api`; BGE catalogs contain generated BGE runs and verified BGE artifacts, avoiding presentation of historical OpenAI release evidence as local-model output.

### Extend the workbench registry without breaking legacy callers

`WorkbenchServices` gains an immutable mapping of profile identifiers to Evaluation gateways and an `evaluations_for(profile_id)` resolver. The existing `evaluations` field remains the default/legacy gateway. Profile construction accepts a separate evaluation mapping so the existing Chat/Documents tuple contract remains compatible with tests and callers.

Callbacks accept a profile identifier for every Evaluation and Comparison method, resolve once, and fail safely on unknown identifiers. Dashboard renderers receive the resolved profile identifier only to produce profile-qualified links; evaluation data continues to come from the gateway.

### Use an optional allowlisted query qualifier for same-origin downloads

Evaluation and Comparison API dependencies inspect an optional `retrieval_profile` query parameter. Missing input resolves to `openai-api`; a registered value resolves to that profile's Evaluation service; unknown values fail safely. Generated workbench links append the qualifier for non-default profiles.

An entirely new route tree was rejected because it would duplicate all evaluation/comparison handlers. Encoding the profile in run IDs was rejected because identifiers are opaque evidence identities and must not become routing authority.

### Own auxiliary Evaluation lifecycle explicitly

The application runtime retains the default `evaluation_service` used by existing APIs and adds the profile service registry. Startup and shutdown deduplicate services by object identity, start all services before accepting traffic, and close each owned service once. Auxiliary services are not hidden inside QA cleanup, which keeps failures and ownership observable.

## Risks / Trade-offs

- [BGE evaluation is substantially slower and consumes more local memory] → Reuse lazy bounded adapters, existing evaluation admission limits, and the BGE-specific deadlines.
- [Run IDs are not globally namespaced across profile databases] → Require the explicit profile qualifier and never search/fallback across registries.
- [Existing workbench code is concurrently changing] → Use additive service APIs and narrow callback/event patches; preserve unrelated README/dashboard changes.
- [Comparison variants may request a retrieval configuration unsupported by the BGE adapter] → Materialization and composition fail closed with existing safe comparison errors; no silent OpenAI fallback is allowed.
- [Two supervisors may reference the same static dataset directory] → Treat datasets as read-only inputs while keeping workspaces, ledgers, and artifacts profile-local.

## Migration Plan

1. Deploy with `openai-api` as the default and verify existing Evaluation HTTP routes and reports are unchanged.
2. Enable `bge-local`; the new profile evaluation directories are created beneath its existing data root.
3. Select `bge-local` in the workbench and run a small evaluation to warm models and produce the first local report.
4. Verify profile-qualified artifact links and model identities before running larger suites.

Rollback removes the auxiliary Evaluation registry and disables BGE Evaluation composition. Existing OpenAI evidence is unaffected; BGE evaluation data can remain in its isolated profile root for a later retry.
