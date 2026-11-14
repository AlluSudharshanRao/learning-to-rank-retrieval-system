# Validation record

## Environments

| Environment | Python | NumPy | PyTorch | FAISS | Device |
| --- | --- | --- | --- | --- | --- |
| NYU HPC | 3.11.18 | 2.2.6 | 2.9.1+cu128 | 1.15.1 | NVIDIA L40S |
| Local tests | 3.14.4 | 2.4.4 | 2.11.0+cpu | 1.15.1 | CPU |

The local test environment differs from the HPC runtime. GPU validation was performed on the HPC environment listed above.

## Local checks

| Check | Result |
| --- | --- |
| Unit and integration suite | 19 tests passed; no skips |
| Ruff lint | Passed |
| Ruff format | Passed for 13 Python files |
| FAISS command with synthetic vectors | Score, ID, and reload checks passed |
| Wheel build | Passed |
| Packaged model configuration | Loaded successfully in an isolated subprocess |
| Public documentation links | Passed |

Test coverage includes invalid dimensions, non-finite vectors, normalization, duplicate IDs, score/ID alignment, tied rankings, FAISS persistence, masked pooling, atomic metadata replacement, and CLI help outside the source directory.

## HPC checks

| Check | Run | Result |
| --- | --- | --- |
| CUDA forward and backward | Job 22876502 | Passed |
| Initial encoder | Jobs 23126518 and 23128889 | Passed |
| Refactored encoder | Job 23142396 | Passed |
| Exact FAISS search | Before and after encoder regression | Passed |
| Index save and reload | Three-vector fixture | Passed |

Encoder revision: `ffb93f3bd4047442299a41ebb6fa998a38507c52`.

The encoder produced token IDs of shape `(4, 22)`, valid token counts `[15, 22, 22, 11]`, hidden states of shape `(4, 22, 384)`, and normalized embeddings of shape `(4, 384)`.

| Passage ID | Cosine similarity |
| --- | ---: |
| 101 | 0.852339 |
| 102 | 0.821197 |
| 103 | 0.709573 |

The refactored output matched the initial scores to six printed decimal places. FAISS returned the same scores and IDs as direct dot products, including after serialization. These checks cover the illustrative fixture; they do not establish corpus-scale retrieval quality, performance, or bitwise equivalence across environments.

FAISS loaded its generic library after specialized AVX modules were unavailable. Correctness checks passed with that backend. CPU implementation and thread settings must be recorded for performance measurements.

## Remaining validation

- Execute the GitHub workflow once a remote is connected.
- Validate the Slurm batch template on Linux. Local MSYS execution was blocked by a host permission error; interactive srun checks passed separately.
- Collect the full HPC dependency snapshot for reproducible environment reconstruction.
- Add dataset-scale evaluation when ingestion is implemented.

## Design documentation

Five Mermaid flowcharts were parsed successfully with Mermaid 11.12.1: the README overview and four detailed system-design views. Relative documentation links passed validation. The diagrams distinguish implemented checks from planned pipelines; they do not add runtime functionality or benchmark results.
