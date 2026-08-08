## Why

The workbench retrieval selector currently changes Chat and Documents only, while Evaluation remains permanently bound to the OpenAI-backed service. Operators therefore cannot generate, inspect, or compare evaluation evidence for the selected `bge-local` retrieval profile.

## What Changes

- Compose an independent Evaluation service for every enabled retrieval profile, using that profile's models, data root, run repository, and report/artifact directories.
- Keep HTTP Evaluation APIs bound to `openai-api` for backward compatibility while exposing the profile registry to the workbench.
- Route all workbench Evaluation and Comparison operations through the explicitly selected retrieval profile.
- Refresh profile-specific datasets, plans, runs, comparisons, summaries, and report downloads when the selector changes, without leaking selection state or run identifiers across profiles.
- Preserve truthful evaluation configuration/model identities so BGE reports identify BGE-M3 embedding and reranking rather than the API retrieval models.
- Start and close auxiliary profile Evaluation services through the existing application lifecycle.

## Capabilities

### New Capabilities

- `profile-aware-evaluation`: Profile-isolated Evaluation and Comparison execution, evidence, reports, lifecycle, and workbench routing for `openai-api` and `bge-local`.

### Modified Capabilities

None.

## Impact

- Evaluation composition and executable lifecycle under `src/rag_mvp/api`.
- Workbench service registry, callbacks, Evaluation/Comparison dashboards, and Gradio event inputs under `src/rag_mvp/ui`.
- Profile-aware evaluation report/run storage beneath each profile data root.
- API, composition, lifecycle, and UI regression tests plus operator documentation.
