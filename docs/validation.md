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

## Dataset source downloader

Local suite: 23 tests passed, including four new downloader tests. Synthetic archives cover complete transfers, member inventory and hashing, incomplete transfers preserving an existing file, unsafe/duplicate/missing members, corrupted gzip trailers, and temporary-file cleanup. Ruff lint and formatting checks passed. The official archive has not yet been downloaded or validated on HPC.

## Record preparation

Official download completed on HPC with exit code 0:0 in 00:09:34; archive inventory and gzip checks passed. Local suite now passes 28 tests. New parser tests cover numeric ID zero, malformed fields, duplicate IDs and judgments, invalid UTF-8, invalid labels, missing query/passage references, existing-output refusal, and extraction/publication with failed-run cleanup. The end-to-end fixture overrides only the official corpus count to two records. Ruff checks passed. Official record preparation remains pending.

## Duplicate judgment policy

The first official preparation run stopped on an identical repeated training judgment. Parser version 2 accepts identical repeats with explicit counts and rejects conflicting labels. All 29 local tests passed, including source preservation and both judgment roles; Ruff passed. Corrected HPC preparation is pending. No full official integrity report is available yet.

## Official record validation completed

Parser version 2 completed on HPC with exit code 0:0 in 00:01:37. The integrity report records 8,841,823 passages, 808,731 training queries, and 6,980 development-small queries. Training qrels contain 532,761 rows, 532,751 unique judgments, and 10 identical duplicate rows. Development-small contains 7,437 unique judgments without duplicates. Duplicate source IDs, conflicting judgments, missing judgment references, and train/final query ID overlap are zero. Normalized-query text overlap and query splits remain pending. Extracted source TSVs retain their original bytes.

## Query splitting

The local suite passes 34 tests; Ruff passes. Five split tests cover normalization and group selection, stable expansion and source-order independence, insufficient data, end-to-end deduplicated output without a final-judgment input, and rejection of changed source data. Official split execution and its counts remain pending.

## Official query split completed

Split version 1 completed on HPC with exit code 0:0 in 00:00:06. The manifest reports 50,000 training queries with 52,867 unique judgments, 2,000 validation queries with 2,113 unique judgments, and 6,980 final queries. There are 502,865 eligible normalized query groups; 305,792 source training queries lack judgments and are excluded from supervised selection. No eligible judged query matched the final query IDs or normalized text. Selected splits have zero ID and normalized-text overlap. Final judgments were not read by splitting. The run used Unicode database 14.0.0. Supplementary TREC overlap remains unchecked; this is still a pre-training gate for the supplementary held-out protocol. These are data preparation results, not retrieval quality measurements.

## Binary metric implementation

All 42 local tests passed, including eight metric tests for hand-calculated values, cutoff and empty-ranking behavior, duplicate-ID rejection, input validation, and fixed-query aggregation. Ruff passed. No real retrieval rankings have been scored, and graded TREC/reference-evaluator verification is pending.

## File-based evaluator

All 46 local tests passed, including four adapter tests for CLI parity, rank validation, judgment-label handling, and failed-run report suppression. Ruff passed. No real retrieval result file has been scored; HPC execution and independent reference parity remain pending.

## Evaluator HPC verification and BM25 runtime

All 12 metric/adapter tests passed under the HPC retrieval environment in 1.286 seconds, including isolated CLI integration. This is synthetic correctness evidence, not reference-tool parity or model quality.

A separate scratch BM25 environment has Python 3.12.16, OpenJDK 21.0.10-internal, Pyserini 2.4.0, NumPy 2.2.6, and PyTorch 2.9.1+cpu. Pip reported no broken requirements. On a CPU allocation, NumPy imported in 1.177 seconds; full LuceneSearcher/Java integration passed in 74.575 seconds. These single-run startup observations are not retrieval latency benchmarks. Index construction and search remain pending.

## BM25 synthetic check implementation

Added scripts/bm25_smoke_test.py and src/ltr_system/bm25_smoke.py. The check uses the documented JsonCollection CLI to build a temporary three-document index, sets k1=0.9 and b=0.4, verifies count and external IDs for unique terms, verifies an absent-term query, and reopens the index to compare results. Saves artifacts/smoke/bm25_check.json only on success. Actual Pyserini indexing/search execution remains pending on HPC; local checks cover import-safe CLI help and existing regressions, not Lucene integration.

## BM25 lifecycle correction

HPC job 23179958 passed synthetic index/search/reopen assertions and wrote a report, then failed during temporary-index deletion (exit 1:0). The local correction retains per-run smoke artifacts, closes searchers, and writes job-associated check.json inside each run directory. All 46 local tests and Ruff passed; corrected HPC execution is pending. Earlier references to temporary-index cleanup and the shared bm25_check.json report describe the original version and are superseded by this correction.

## BM25 smoke completed and sharding implementation

Corrected smoke job 23180928 completed in 00:01:08 with exit 0:0. Logs confirm three indexed documents, zero indexing errors, and successful search/reopen checks. The report is retained with its per-run index. Full-corpus quality is unmeasured.

Added streaming TSV-to-JSONL conversion with verified input identity, preserved IDs/text, deterministic shards, per-shard hashes and staged publication. All 50 local tests and Ruff passed. Official-data conversion remains pending.

## Full lexical indexing implementation

Corpus conversion verified on HPC: 8,841,823 records, 89 shards, last shard 41,823. Added index builder with pre/post source verification, separate index/verification processes, staged publication and retained failure logs. All 53 local tests and Ruff passed; full Lucene build and runtime resource usage remain unverified. Local tests exercise input validation and CLI loading, not full Java indexing.

## Full index completed; validation runner implemented

HPC index job 23187037 completed in 00:02:36 with exit 0:0, verifying 8,841,823 documents. Index subprocess duration including startup: 88.170653 seconds. Four threads, 16 GB heap, English/Porter analysis, stopword removal; no raw text, positions or document vectors. Source hash matches the validated collection.

All 56 local tests and Ruff passed after adding validation retrieval/scoring. Three new tests use a simulated search backend for serialization/evaluator integration, empty queries, invalid scores/IDs, duplicate results and tie preservation. They do not establish real Lucene relevance quality. Official retrieval remains pending.

## First full-corpus validation retrieval completed

Job23188133 completed in00:02:02, exit0:0. All2000 validation queries have results. Measured MRR@10=0.161909, binary NDCG@10=0.205845; Recall@20/50/100/1000=0.436250/0.560583/0.643917/0.849333. See results.md for protocol and limitations. Values are rounded from user-provided HPC evaluator output, not independently retrieved artifact copies.

## E5 encoding pilot implementation

Added bounded GPU encoding pilot using the pinned model and verified corpus prefix. All 58 local tests and Ruff passed. New tests cover prefix selection/ID handling; they do not run CUDA or establish encoding performance. Real GPU pilot timing, memory usage and vectors remain pending.

## Encoding pilot completed on HPC

Job23190504 passed on L40S, exit0:0 in00:01:27. The verified-prefix10000-passage FP32 batch64 pilot records4.754380seconds timed encoding and2103.323537passages/sec; peak allocated463351808bytes and reserved788529152bytes. Shape/finite/unit-norm checks passed. Full corpus encoding, representative workload profiling and dense retrieval quality remain pending. GitHub push/CI for commit10c5059 was reported successful before this pilot.

## Passage-length profiler implementation

Added seeded corpus-row sampling and CPU token-length profiling with saved sample/IDs and nearest-rank statistics. All60 local tests and Ruff passed. Two new tests cover selection repeatability, coverage without replacement, input-count failures and the exact512token truncation boundary. Actual pinned-tokenizer profiling remains pending on HPC.

## Saved-sample encoding pilot extension

Official sampled length profile passed (job23192851):10000rows, mean79.3199tokens, maximum289, zero above512 in sample. Extended GPU pilot to verified saved samples and repeated timing. All62 local tests and Ruff passed, including two new sample identity tests. GPU comparisons of batch64/256 remain pending; no speedup claim yet.

## Sample GPU comparison and vector agreement completed

Job23193970 passed on L40S. Three repetitions each: batch64 median5.904316s/1693.676367passagessec, batch256 median6.686149s/1495.629267passagessec. Same10000sample IDs and10000x384 vectors verified; maxcomponentdifference3.2782554626464844e-7; minimum pairedcosine0.9999998807907104; allclose atol1e-5/rtol1e-4 passed. Initial full-encoding choice is batch64 FP32; fullcorpus encoding and dense relevance remain pending.

## Resumable dense encoding implementation

Added shardwise FP32 encoding, explicit vector-map closure, staged publication, checksummed ID/vector artifacts, strict configuration matching and verified completed-shard reuse. All66 local tests and Ruff passed. The failure-injection test caught a Windows traceback retaining a memory-mapped file; explicit unmapping now permits cleanup. No full corpus shard has yet been encoded by this command. Next gate: run shard0, then rerun shard0 and confirm verified skip.

## First dense shard and restart verified on HPC

Job23248207 completed in00:02:02, exit0:0, publishing100,000 passage vectors at384dimensions FP32 on L40S, batch64. Manifest reports50.898146seconds shard processing including parsing, tokenization, encoding, writes/checks, source rehash and IDs. Job23248932 completed in00:01:15, exit0:0; its log explicitly confirms verified completed shard and skipping encoder for part-00000. Model loading still occurs before skip checks. Remaining88shards and aggregate coverage validation remain pending.

## Full-range dense encoding job completed

Job23249408 completed with exit0:0 in01:17:03. Logs confirm publication through part-00088 with41,823 records and completion of range[0,89). This job reused the already encoded first shard; its elapsed time is not a clean from-scratch encoding benchmark. An independent all-shard coverage audit is pending. Added CPU audit and three synthetic tests; all69 local tests and Ruff passed. Dense retrieval quality remains unmeasured.
