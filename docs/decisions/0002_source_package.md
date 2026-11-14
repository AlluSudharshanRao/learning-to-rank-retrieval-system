# ADR 0002 Separate source code from runtime artifacts

**Status:** accepted. CPU checks and HPC encoder/exact-search regression passed.

## Context

The initial standalone smoke scripts established correctness on HPC but executed work at module import, embedded experiment inputs, and depended on implicit scratch paths. Copying code manually made it difficult to keep local source, execution evidence, and versioned configuration aligned.

## Decision

Move reusable behavior into `src/ltr_system`. Keep four thin source-checkout wrappers with their original filenames. Bundle the pinned model revision and illustrative fixture as package data. Expose explicit runtime-root and logging arguments. Keep heavyweight dependencies lazy so CLI help and package import work without CUDA, checkpoints, or network access.

Use focused CPU tests and lint/format checks. Keep GPU regression as an explicit Slurm step. Accept equivalent tie orderings in exact-search validation while checking each returned ID's score and the complete ranking. Write JSON metadata through an atomic temporary-file replacement.

## Consequences

Source transfers must include `src/`, not just `scripts/`. The explicit wrapper path bootstrap is intentional for `python -I`; it adds only the checkout source tree. Editable installs support development, and wheel configuration includes default JSON files. Runtime artifacts remain under the selected project root; existing HPC environments need no reinstall for a source update.

The refactor preserves the fixture and encoder semantics. Validation includes the CPU suite, package build, and HPC regression. Multi-file artifact publication, full dependency locking, and production serving are outside this refactor.
