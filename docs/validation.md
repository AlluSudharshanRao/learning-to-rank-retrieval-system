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

## Exact dense index implementation

Added exact FAISS index builder and isolated CLI. All72 local tests and Ruff passed, including canonical ID conversion, real FAISS exact-ranking/save/reload checks on synthetic data, and rejection of modified embedding artifacts without publication. Full-corpus builder execution remains pending. Embedding coverage on HPC previously passed with8,841,823rows,89shards,384dimensions and ordered source-ID alignment; vector files total13,581,051,520bytes including headers.

## Exact dense index completed; query encoder implemented

Index job23282570 completed with exit0:0 in00:01:23. Manifest confirms8,841,823vectors at384dimensions, FAISS1.15.1 IDMap2FlatIP,89 reconstruction checks after reload, and13,651,774,802byte saved index. Build/verification scope took70.122530seconds. This verifies index structure, not dense relevance quality.

Added validation-only E5 query encoder. All74 local tests and Ruff passed, including two input tests for ID order/query prefixes, changed input and duplicate query rejection. GPU query encoding and full-corpus dense validation retrieval remain pending.

## Validation query encoding completed

HPC job 23302557 completed successfully in 68 seconds. The query manifest
confirms 2,000 vectors, 384 dimensions, FP32, masked mean pooling and L2
normalization with the pinned E5 checkpoint. The validation query checksum
matches the frozen split. No final queries or judgments were read.

## Exact dense retrieval implementation

Added a CPU exact search runner with checksummed input validation,
configuration and ordered-query alignment, staged output publication and
the existing binary relevance evaluator. Synthetic tests exercise full
search-to-evaluation execution, batch-size agreement, checksum rejection,
reordered IDs and invalid vector norms. Full-corpus execution and quality
comparison with BM25 are pending.

## Exact dense retrieval completed on HPC

Job 23620628 completed with exit 0:0 in 14 minutes 29 seconds. The reported
binary evaluation covers all 2,000 validation queries with zero missing
or empty rankings. MRR@10 is 0.317547, NDCG@10 is 0.381328, and Recall at
20/50/100/1000 is 0.707167/0.828917/0.879083/0.968500. These exceed the
current BM25 aggregate values on the same validation population. Full run
manifest inspection remains pending; no serving latency or statistical
significance claim is made.

## Exact search timing and reference evaluation gate

The completed exact validation run manifest reports 2,000 queries,
2,000,000 retrieved candidates, batch size 32 and four CPU threads with
FAISS 1.15.1. Search calls took 825.801958 seconds; the retrieval loop took
826.391006 seconds. Approximate search-only offline throughput was
2.42 queries/second. This excludes encoding, index loading, checksums and
scoring, and does not establish serving latency. The final query set was
not read. Saved evaluation SHA-256:
`2efcf0f24bf87ff5a4f5e491263b9ffb6299fde2c785b9ab12e5ca63895b4f48`.

Installed pytrec-eval-terrier 0.5.11 into a separate project tools directory
on HPC. Import and rank-2/rank-11 metric sanity checks passed. Added a
saved-run comparison command with independent TSV parsing and per-query
as well as aggregate checks. Full reference comparison is pending HPC.

## Independent evaluator agreement confirmed

HPC job 23625746 completed with exit 0:0 in six seconds. Both
`bm25-validation-default` and `e5-exact-validation-default` passed the
comparison against pytrec-eval-terrier 0.5.11 on the same 2,000 validation
queries. Maximum per-query absolute error was 0.0 for RR@10, binary
NDCG@10, and Recall@20/50/100/1000 in both runs: 24,000 per-query metric
comparisons in total. Macro aggregates also passed the 1e-12 absolute
tolerance. Query and judgment checksums matched across runs; final queries
were not read. This confirms agreement for metric computation on these
saved rankings. It does not establish statistical significance, retrieval
implementation correctness, or generalization.

## ANN construction pilot

CI for commit 2b01306 passed, as confirmed by the user. Added a first-shard
HNSW construction pilot with audited input checks, retained external IDs,
reload/reconstruction checks and exact-score validation on stored-vector
probes. The pilot is not a full-corpus relevance evaluation. HPC execution,
full-corpus HNSW indexing and matched search benchmarks remain pending.

## HNSW pilot completed on HPC

Job 23655213 passed in 23 seconds (exit 0:0). Manifest confirms 100,000
vectors, 384 dimensions, M=32, efConstruction=200, successful reload and
ID reconstruction/score checks. Graph add calls took 12.004085 seconds;
index bytes were 181,620,879. Sixty-four stored-vector probes at efSearch=64
had mean exact top-ten neighbor overlap 0.98125. This validates the pilot
mechanics, not held-out relevance, full-corpus ANN quality or search latency.

## Full-corpus HNSW builder implementation

Added a separate full-corpus HNSW builder using audited source shards,
small add batches, checksum rechecks, staged publication and reload/ID/score
checks. Synthetic tests cover multiple shards with noncontiguous external
IDs, tampering in a later shard without publication, and coverage count
mismatch. Full-corpus construction and real-query ANN evaluation are pending.

## Full HNSW construction passed

Job 23657686 completed with exit 0:0 in 22 minutes 16 seconds. Manifest
confirms all 8,841,823 vectors in 89 shards with M=32, efConstruction=200
and eight threads. Reload and 89 reconstruction checks passed; index size
is 16,057,891,835 bytes. Added a validation retrieval runner with exact-run
identity binding, judged relevance scoring and per-query neighbor overlap.
Real-query HNSW execution remains pending.

## HNSW efSearch=1024 validation completed

Job 23729240 passed in 59 seconds, exit 0:0. All 2,000 validation queries
returned 1,000 candidates. Judged MRR@10 was 0.31089801587301585 and
Recall@1000 was 0.951. Exact-neighbor recall at 10/100/1000 was
0.9755/0.9687/0.9153245. Search-call time was 2.1995858828304335 seconds
for one offline pass. No matched speedup or serving percentile is claimed.
The next controlled quality run increases efSearch to 2048 without
changing the graph, query population, batch size or thread count.

## HNSW efSearch=2048 validation completed

Job 23729437 passed in 56 seconds with exit 0:0. Manifest identities match
the ef1024 run. MRR@10 was 0.31427162698412703; judged Recall@1000 was
0.96025. Neighbor recall at 10/100/1000 was 0.986/0.980865/0.9573705.
Search calls took 4.781392026809044 seconds. Both quality measures improved,
but separate hosts and single passes preclude a controlled timing ratio.
Next quality setting is ef4096 with graph, queries, batch and threads fixed.

## HNSW efSearch=4096 validation completed

Job 23729642 passed in 66 seconds, exit 0:0. MRR@10 was
0.31504742063492064; judged Recall@1000 was 0.9625. Exact-neighbor recall
at 10/100/1000 was 0.99015/0.986745/0.9769855. Single-pass search calls
took 17.768442114815116 seconds. All three HNSW settings have measured
real-query relevance and exact-neighbor overlap. Matched performance
benchmarking remains pending; no measured serving latency exists yet.

## Matched search benchmark implementation

Downloaded all five quality-run reports and verified each evaluation JSON
against its run manifest hash locally. Added a matched CPU benchmark for
exact plus HNSW ef1024/2048/4096: four repetitions, common query orders,
rotating algorithm order, explicit warmup, full validation batch throughput
and actual single-query timing on a fixed 200-query sample. Raw timings,
per-repetition summaries and scope metadata are retained. Synthetic tests
cover scheduling, timer/result handling, percentile math, real small-index
execution and tampered query rejection. Full HPC benchmarking is pending.

## Matched benchmark passed on HPC

Job 23730406 completed in 42 minutes 50 seconds with exit 0:0. All four
algorithms completed four repetitions on cs661 with four CPU threads.
Median batch QPS was 4.233586 exact, 863.083329 ef1024, 415.198655 ef2048
and 177.204698 ef4096. Pooled single-query search p95 was 547.978210,
6.383914, 12.642729 and 27.224447 ms respectively. Results include actual
single-query measurements, not amortized batch times. Raw report transfer,
per-repetition inspection and graph generation remain pending.

## Verified benchmark plots

Downloaded the raw report for job 23730406. Verified four trials and 800
single-query observations per algorithm, matching query orders, compatible
index/query identities across quality and benchmark reports, and timing
summaries recomputed from raw samples. Added a plotting command with four
source-validation tests. All 105 local tests and Ruff checks passed.
Generated and visually inspected three PNG/SVG figure pairs plus portable
JSON/CSV summaries. README and architecture status now reflect the measured
offline pipeline. Latest ANN/reporting changes have not yet run in CI.
