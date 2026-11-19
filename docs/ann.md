# HNSW indexing experiments

The exact E5 baseline is the reference for ANN experiments. It uses the
same normalized FP32 passage and query vectors throughout this phase.
Changing the encoder or truncation policy at the same time would confound
the effect of indexing.

## First gate: one-shard construction pilot

`scripts/build_hnsw_pilot.py` builds an index over `part-00000` from the
audited embedding corpus. This is a corpus-prefix pilot, normally 100,000
passages, not a representative sample or a full-corpus relevance run.
The initial configuration uses `IndexIDMap2(IndexHNSWFlat)` with inner
product, M=32, efConstruction=200 and four CPU threads. These are starting
settings rather than empirically optimal choices.

| Parameter | Role | Main tradeoff |
| --- | --- | --- |
| M | Graph connectivity; base-layer capacity is 2M in this implementation | More graph memory and build work; potentially improved navigation |
| efConstruction | Build-time exploration | Additional build cost may improve graph quality |
| efSearch | Search-time exploration | More search work may recover more exact neighbors |

HNSWFlat retains FP32 vectors. Graph links add memory overhead; this is
not vector compression. Passage IDs are retained through the IDMap2 wrapper.
The pilot verifies the coverage/configuration/shard hashes, vector norms,
ID counts and file checksums before building. It publishes a new index
and manifest only after reloading, reconstructing stored vectors by ID,
checking returned IDs and scores, and rechecking input hashes.

Up to 64 evenly spaced stored passage vectors serve as sanity probes.
An exact index over the same shard provides top-ten neighbor sets for a
reported ID-overlap check. No minimum overlap is assumed, and ties may
change exact membership. This is an easy stored-vector test and must not
be used to claim held-out query accuracy. Probe efSearch is 64. Build
order is source row order with 10,000-vector add calls. Parallel insertion
can produce different graphs; saving a checksum identifies the actual
artifact rather than promising bitwise repeatable construction.

Output: `artifacts/indexes/e5-hnsw-pilot-m32-efc200/` containing
`hnsw.faiss` and `manifest.json`. The manifest separates graph-add time
from exact-reference additions, loading, writing and validation. Existing
output is not overwritten. An interrupted build has no construction
checkpoint and must restart. Inputs must remain immutable throughout.
The builder uses the existing embedding writer lock; a stale lock after
forced termination requires inspection before removal.

Run on a CPU Slurm node. A 16 GB allocation is a conservative initial
budget for the one-shard graph plus its temporary exact reference and
Python/FAISS overhead; measured peak memory will guide later sizing.
No GPU, new model weights or recomputed embeddings are required.

## Subsequent gates

After the pilot passes, build and verify a full-corpus index. Sweep
search settings with the same 2,000 validation queries. Measure neighbor
recall against saved exact results separately from judged relevance metrics.
Only compare timing under matched hardware, query batches, thread counts,
warmup and timing scope. A later single-query latency benchmark needs its
own exact baseline; previous offline throughput is not serving p50/p95.
Track build time, index bytes and host memory alongside search quality.
Final evaluation queries remain reserved during tuning.

Sources: [FAISS index documentation](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
and [FAISS HNSW tutorial](https://github.com/facebookresearch/faiss/blob/main/tutorial/python/6-HNSW.py).

## Pilot execution status

The first-shard pilot passed on HPC in job 23655213. It built and reloaded
100,000 vectors; 64 stored-vector probes had mean top-ten exact-neighbor
ID overlap 0.98125 at efSearch=64. Graph additions took 12.004085 seconds,
and the saved index is 181,620,879 bytes. These observations support the
full-corpus construction gate; they do not establish real-query recall,
serving latency or linear build-time scaling.

## Full-corpus builder

`scripts/build_hnsw_index.py` builds a new graph from all audited embedding
shards; it does not extend the pilot graph. It retains source shard/row
order and uses 10,000-row add calls. Default M=32 and efConstruction=200
match the pilot. The default eight build threads may change graph topology
relative to a four-thread run; preserve the actual saved artifact checksum.
No claim of bitwise reproducibility is made.

The builder verifies the coverage status, ordered source-ID alignment,
shard set/counts, configuration identity, each shard manifest and both shard
files. Shape, dtype, finite values and unit norms are checked before adding.
Cross-shard ID uniqueness is inherited from the unchanged coverage audit
and its source-row alignment, as in the exact index builder. Inputs must
remain immutable. All source checksums are rechecked before publication.

After saving, the construction graph is released and the index reloaded.
The builder verifies graph type, inner-product metric, dimensions, total
count and graph settings. The first vector of each shard is reconstructed
by external passage ID and compared exactly. Search probes check returned
IDs and direct inner-product scores. There is no full exact reference index
in memory during this build and no measured neighbor recall in this phase.
The saved efSearch=64 is a probe setting, not a selected production setting.

Output: `artifacts/indexes/e5-hnsw-m32-efc200-v1/`. The builder publishes the
index and manifest together after all checks; existing output is rejected.
Graph-add time is recorded separately from complete build/verification time.
The latter includes input checks, file writes, reload, probes and checksums.

Initial CPU Slurm budget: eight CPUs, 96 GB host memory, eight hours. These
are allocation limits, not predicted consumption. The graph retains full
FP32 vectors plus links and ID mappings; construction can temporarily need
additional capacity. The pilot does not establish linear full-build scaling.
No GPU is requested. A forced termination can leave unpublished temporary
files and stale locks; inspect them before cleanup. Graph construction has
no resume checkpoint and must restart after interruption. Sharded vector
encoding remains independently resumable and does not need to be repeated.

## Real validation query retrieval

`scripts/retrieve_hnsw.py` evaluates the fixed full-corpus M=32,
efConstruction=200 graph using the previously encoded validation queries.
It checks graph, query and split identities and binds the saved exact
ranking to its query and index manifest checksums. Coverage, vector count,
dimension and embedding configuration must agree with the exact baseline.
The graph is loaded once and efSearch is set explicitly.

The first configuration is efSearch=1024, top 1,000 passages, batches of
32 and four CPU search threads. efSearch must be at least 1,000 for this
runner's candidate depth. This is an initial controlled configuration,
not a parameter optimum. Larger efSearch settings can be evaluated without
rebuilding the graph. Outputs use a directory name containing efSearch,
batch size and thread count; an existing directory is never overwritten.

Each run publishes ranking.tsv, evaluation.json, neighbors.json and a
manifest. Judged relevance uses the same evaluator as the exact baseline.
Neighbor recall at 10, 100 and 1,000 is the macro top-K ID-set intersection
divided by K, with per-query values retained. It is a different measure
from relevance recall. It is not tie-aware: alternative equal-score exact
neighbors can reduce ID overlap. Both rankings must cover the same query
population and have sufficient depth without duplicate passage IDs.

The search timer covers a single pass of FAISS search calls, excluding
index loading, ranking writes, encoding, scoring and neighbor comparison.
There is no warmup or repetition in this correctness/quality run. Do not
compare it with the earlier exact job as a matched speedup measurement;
a repeated same-hardware benchmark remains a separate gate. Final queries
remain reserved. Inputs are rehashed before output publication.

## Initial search observation

The efSearch=1024 full validation run passed (job 23729240). Exact-neighbor
recall@1000 was 0.9153245 and judged Recall@1000 was 0.951, versus 0.9685
for exact search. This illustrates why both metrics are needed. The next
quality setting is efSearch=2048 on the same graph, with batch32/four CPU
threads unchanged. More exploration may recover missed neighbors, but
per-query and aggregate gains must be measured rather than assumed.

## Search exploration comparison

Increasing efSearch from 1024 to 2048 improved all reported quality
aggregates on the frozen validation population. Neighbor recall@1000 rose
from 0.9153245 to 0.9573705; judged Recall@1000 rose from 0.951 to 0.96025.
The observed search-call duration increased, but these were single passes
on different nodes. Do not infer a precise time scaling law. The next
quality setting is ef4096 before matched benchmark design.

## Quality sweep completed

Measured efSearch values 1024, 2048 and 4096 on the same fixed graph and
validation population. All relevance and neighbor aggregate metrics rose
across these settings. The latest judged Recall@1000 was 0.9625 versus
0.9685 for exact search; MRR@10 was 0.315047 versus 0.317547. Keep all three
points for the subsequent speed/quality curve. Do not choose a final
configuration from single-pass timings. The next gate uses exact and HNSW
on the same node, fixed threads/batch size, shared query permutations,
warmup and repeated timings. Search-only single-query percentiles remain
distinct from encoding, queueing and end-to-end service latency.
