## 1. Profile Evaluation Composition

- [x] 1.1 Extract reusable Evaluation service composition with injectable isolated retrieval factories
- [x] 1.2 Compose BGE Evaluation and Comparison executors with BGE runtime factories, isolated storage, and truthful identities
- [x] 1.3 Add composition tests for profile isolation, factory routing, static release handling, and disabled BGE behavior

## 2. Runtime Lifecycle and HTTP Access

- [x] 2.1 Register default and auxiliary Evaluation services in the application runtime with deduplicated startup and shutdown
- [x] 2.2 Resolve optional allowlisted retrieval-profile qualifiers in Evaluation and Comparison HTTP routes while preserving OpenAI defaults
- [x] 2.3 Add API and lifecycle tests for default compatibility, BGE artifacts, unknown profiles, startup, and single close ownership

## 3. Workbench Profile Routing

- [x] 3.1 Extend WorkbenchServices with immutable profile Evaluation gateways and safe explicit resolution
- [x] 3.2 Route every Evaluation and Comparison callback through the selected profile and generate qualified download links
- [x] 3.3 Wire the shared retrieval selector into Evaluation/Comparison events and reset profile-specific browser selections on change
- [x] 3.4 Add UI tests for profile catalogs, starts, polling, switching, unknown identifiers, and report links

## 4. Documentation and Verification

- [x] 4.1 Document profile-specific evaluation storage, execution cost/latency, and report selection behavior
- [x] 4.2 Run targeted/full tests, static checks, Compose validation, and strict OpenSpec validation
- [x] 4.3 Commit only profile-aware Evaluation changes without including concurrent workspace edits
