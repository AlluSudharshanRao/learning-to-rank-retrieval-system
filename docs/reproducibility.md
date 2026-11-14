# Reproducibility

## Code and environment

Record a code commit, dirty-worktree status, Python version, OS, installed distribution versions, installation indexes, and hardware. `requirements/retrieval-verified.txt` lists directly observed HPC package versions, not a complete lockfile. The complete HPC dependency snapshot remains in runtime storage and is pending repository review.

The minimal package declares NumPy; GPU PyTorch is installed separately from its explicit CUDA wheel index. This prevents a base-package install from silently choosing a CPU or different CUDA build. Constrain the already verified numerical stack when resolving retrieval dependencies.

## Model identity

`src/ltr_system/defaults/encoder.json` pins E5 to the full revision previously downloaded on HPC. Downloads use the configured commit rather than the moving main branch. Runtime manifests must agree with this configuration. Existing manifests without `schema_version` remain readable if their identity and snapshot path match; newly written manifests include schema version 1.

The encoder loads local Safetensors with remote code disabled and Hub offline mode. Its provenance JSON records the texts, IDs, tokenizer length limit, model revision, PyTorch build, GPU, and job ID. Runtime paths are machine-specific and excluded from Git.

## Artifact expectations

| Artifact | Purpose | Verification |
| --- | --- | --- |
| `artifacts/models/e5-small-v2.json` | Checkpoint identity and local snapshot | Match configured repository and revision |
| `artifacts/smoke/encoder_vectors.npz` | Query/passages and stable IDs | Finite normalized vectors and aligned int64 IDs |
| `artifacts/smoke/encoder_provenance.json` | Context for newly generated embeddings | Inspect alongside the vector artifact |
| `artifacts/smoke/exact.faiss` | Exact index | Compare scores and IDs before and after reload |

JSON metadata replacement is atomic on a single filesystem. Vector and FAISS outputs are currently small smoke-test files; multi-file transactional publication and resumable corpus shards are future pipeline work. A smoke rerun replaces only those named project artifacts. Never load an untrusted FAISS index.

## Validation scope

The original GPU, encoder, and FAISS checks passed on HPC. The refactored encoder and exact-search implementation also passed HPC regression on the same fixture, with matching scores to six printed decimals. Unit tests use synthetic vectors; they do not establish Recall, MRR, NDCG, throughput, or latency. No model training or full-corpus evaluation has been completed.
