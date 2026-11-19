# Matched CPU search benchmark

## Purpose

Compare exact E5 search with the fixed full-corpus HNSW graph at
efSearch 1024, 2048 and 4096. This controls for the different nodes and
single-pass timing scopes in the initial relevance runs. It measures warm,
in-process CPU search, not complete retrieval service latency.

## Workloads and controls

| Setting | Initial protocol |
| --- | --- |
| Candidate count | 1,000 for every algorithm |
| Query vectors | Existing 2,000 frozen validation queries |
| Batch throughput | All queries, batches of 32 |
| Single-query latency | Fixed sample of 200 validation queries |
| CPU search threads | Four |
| Measured repetitions | Four |
| Random seed | 2026 |
| Warmup per algorithm per repetition | Two batch calls and two single-query calls |
| Algorithm order | Seed-shuffled initial order, rotated across repetitions |
| Query order | Shuffled each repetition, shared across algorithms |

The same exact and HNSW index objects remain resident in one CPU process
on one node. Changing efSearch does not reload or rebuild HNSW. Index,
query and split checksums and compatible corpus/configuration identities
are checked before benchmarking and rechecked before publishing results.
The runner records graph settings, library versions, CPU model, affinity,
host, thread environment and job ID. It sets OMP/OpenBLAS/MKL thread
limits before importing NumPy and FAISS in the isolated CLI process, and
sets the FAISS OpenMP limit explicitly. Actual allocated CPU topology is
recorded rather than assumed. Other jobs may share the node, so this is
not a claim of exclusive machine access.

Four repetitions balance each algorithm's trial position once. The same
200 latency queries are reshuffled and reused across repetitions, giving
800 timing observations per algorithm. These are repeated observations,
not 800 independent query samples or users. No confidence interval for
user-facing latency is implied. Quality tuning uses validation data; the
final query set remains reserved and judgments are not read by this runner.

## Timing boundaries

Each query batch is gathered into a contiguous array before the timer.
Only the synchronous Python FAISS search call is timed. Result shape,
finite/ordered scores, nonnegative IDs and unique returned IDs are checked
after the timer. Output serialization, query preparation, model encoding,
index loading, checksums and relevance evaluation are excluded.

Batch queries/second is calculated from the sum of timed search calls
for a full repetition. The report retains each repetition and summarizes
the median rate. No per-query latency is inferred from batch time.
Single-query percentiles use actual one-query search calls. NumPy's linear
percentile method is recorded; pooled and per-repetition p50/p95 values
are retained alongside raw samples and sampled query order.

These are search-call percentiles without network, encoding, queueing,
concurrency, document fetches or reranking. They are not service p50/p95.
The benchmark is a warm workload, not a cold-start or index-loading study.
Alternating algorithms does not remove every cache/frequency/noisy-neighbor
effect; repeated measurements and metadata make remaining limits explicit.

## Execution and outputs

Entry point: `scripts/benchmark_search.py`. Initial allocation: four CPUs,
96 GB RAM, three hours. Both full indexes and ID mappings are resident,
so this benchmark needs more memory than an individual HNSW query run.
The allocation is a conservative limit, not measured consumption.
No GPU is used. Runtime may be dominated by repeated exact search.

Output: `artifacts/benchmarks/search-matched-v1.json`. The runner rejects
an existing output path and writes the report only after all trials and
checksum rechecks pass. Failed/interrupted trials do not publish a partial
report and must restart. Process peak RSS covers both resident indexes and
benchmark overhead; it is not an isolated index-memory measurement. Index
file bytes are already recorded by their build manifests.

Report raw trials, median batch throughput and single-query percentiles
alongside the existing quality sweep before selecting a configuration.
A further load-test phase is needed for full serving throughput and latency.
No matched performance results exist until the HPC benchmark completes.

## First matched execution

The matched benchmark passed on HPC in job 23730406. Summary metrics and
per-repetition batch times are recorded in results.md. These supersede
uncontrolled timing comparisons from separate quality jobs. Search-only
p95 and batch throughput must retain their workload labels; full serving
latency remains unmeasured. Raw report inspection and reproducible plots
are the next reporting gate.
