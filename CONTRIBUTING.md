# Development workflow

Use small, reviewable changes. Keep code, tests, and user-facing documentation in the same change. Separate correctness checks from quality evaluations and performance measurements.

## Local checks

Use a dedicated Python 3.11 environment for the pinned project stack. From the repository root:

```bash
python -m pip install -e . -r requirements/dev.txt
python -m ruff check .
python -m ruff format --check .
python -m unittest discover -s tests -v
```

The base package requires NumPy. FAISS integration tests require `faiss-cpu==1.15.1`; pooling tests require PyTorch. Missing optional dependencies produce explicit skips. Full retrieval setup follows the HPC runbook. The CPU CI job installs FAISS, and a second job tests pooling with CPU PyTorch; neither allocates a GPU or downloads model weights.

## Code conventions

- Put reusable behavior in `src/ltr_system/`; keep `scripts/` as thin entry points.
- Parse CLI arguments before optional imports so `--help` works without CUDA or model packages.
- Keep imports free of downloads, file writes, and GPU allocation.
- Use explicit exceptions for invalid runtime inputs. Use runtime validation rather than Python assert statements for external inputs.
- Validate dimensions, finite values, vector normalization, unique IDs, and model revisions at artifact boundaries.
- Store checkpoint identity in versioned configuration; machine-specific paths belong in ignored artifacts.
- Use UTF-8 text, LF line endings, type annotations, and Ruff formatting.

## Verification and review

Test failure cases and invariants: tied scores, ID mapping, invalid inputs, padding, and interrupted metadata writes. Test behavior at module and artifact boundaries. Record the test environment and any skipped checks. A passing CPU test suite does not establish GPU compatibility or ranking quality.

Before a commit, inspect `git status`, `git diff`, and staged files. Commit source, package configuration, tests, and public docs. Keep Git-ignored local notes out of commits. Do not commit datasets, credentials, caches, environment directories, checkpoints, or runtime indexes.

Dataset and checkpoint redistribution follows the upstream terms. A repository license is pending.
