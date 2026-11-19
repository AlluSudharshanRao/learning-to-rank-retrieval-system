# Measured results

## BM25 initial validation baseline

Run: `bm25-validation-default`. Full MS MARCO Passage v1 corpus: 8,841,823 passages. Evaluation population: 2,000 project validation queries from the training pool, selected by split version 1. This is not the official development-small benchmark or a leaderboard result.

Backend: Pyserini 2.4.0 / Lucene. Index analysis: English, Porter stemming, stopword removal. Search parameters: k1=0.9, b=0.4; retrieval depth 1,000. These are initial baseline settings, not selected optima.

| Metric | Value |
| --- | ---: |
| Recall@20 | 0.436250 |
| Recall@50 | 0.560583 |
| Recall@100 | 0.643917 |
| Recall@1000 | 0.849333 |
| MRR@10 | 0.161909 |
| Binary NDCG@10 | 0.205845 |
| Evaluated queries | 2,000 |
| Missing rankings | 0 |
| Empty rankings | 0 |

Scores are macro averages over the fixed query population. Binary NDCG uses gain 1 for known positives and log2(rank+1) discount. Unjudged results receive zero credit. Values shown are rounded to six decimals from the HPC evaluator output; retain full precision in the original report. Independent reference-evaluator parity has not yet been established.

Recall@1000 means that, on average per query, 84.9333% of known judged positives were retrieved. It does not mean that exactly that percentage of queries had a hit, nor does it measure recall of all potentially useful unjudged passages. A reranker restricted to these candidates cannot recover the missed known positives. MRR is the average reciprocal first-positive rank; it is not a success percentage and its reciprocal is not average rank.

The complete Slurm retrieval/scoring job took 00:02:02 and exited 0:0, with two allocated CPU cores and 16 GB requested RAM. This elapsed time includes startup, input checks, retrieval, serialization and evaluation; it is not serving p50/p95 latency, isolated search time, or a controlled throughput benchmark. The retrieval-loop timing is separately recorded in the run manifest and has not been transcribed here.

Artifacts reside under `artifacts/runs/bm25-validation-default/`: `ranking.tsv`, `evaluation.json`, and `manifest.json`. The run manifest links index and split identities and output hashes. Runtime artifacts remain excluded from Git. Final evaluation queries and judgments were not read. Supplementary TREC overlap exclusion, graded evaluation, confidence intervals, and reference-tool comparison remain pending.

Next comparisons must use the same corpus, validation query membership, judgment policy, and evaluator. Pretrained dense retrieval will provide a second baseline before training or reranking improvements are claimed.

## E5 passage encoding pilot

Pinned intfloat/e5-small-v2 revision ffb93f3bd4047442299a41ebb6fa998a38507c52 encoded the first 10,000 passages of the first verified JSONL shard on NVIDIA L40S. FP32 without autocast, batch size64, maximum512tokens, passage prefix, masked mean pooling and L2 normalization; two warmup batches excluded.

| Measurement | Observed value |
| --- | ---: |
| Timed encoding | 4.754380 seconds |
| Encoding throughput | 2,103.323537 passages/second |
| Peak PyTorch allocated GPU memory | 463,351,808 bytes |
| Peak PyTorch reserved GPU memory | 788,529,152 bytes |
| Output shape | 10,000 x 384 |
| Complete Slurm job | 00:01:27, exit 0:0 |

Timing includes tokenization, transfers, forward inference, pooling and CPU vector collection; excludes model loading, warmup and artifact saving. PyTorch peaks do not measure total GPU device consumption or utilization. Vector shape, finite values and unit norms were checked before publishing vectors.npy, passage_ids.json and manifest.json under artifacts/encoding-pilots/e5-n10000-b64/. Values are transcribed from the provided manifest; no independent artifact copy was inspected locally.

This single corpus-prefix run is not a representative throughput benchmark or retrieval-quality measurement. A constant-throughput extrapolation suggests roughly70minutes of encoding for the full corpus, but ignores passage-length variation, sharding/checkpoint I/O and allocation variability. Characterize sequence lengths and representative shards before selecting the full encoding budget. No comparative speedup is claimed.

## Corpus-wide sampled token lengths

A uniform sample without replacement of10,000 corpus rows, seed2026, was profiled using the pinned E5 tokenizer without truncation, including passage prefix and special tokens. Observed mean79.3199, median70, p90127, p95142, p99176, minimum14, maximum289tokens. Zero sampled passages exceeded512tokens; this is not proof that the full corpus has no longer passages. CPU profiling completed in00:00:52 with exit0:0. Saved sample SHA-256:63c22058ed2992c964b713438057a470b2690d13263790e5e0babc0fc397cdc2. Sample encoding/batch-size comparisons remain pending.

## Sampled E5 batch-size comparison

Job23193970 completed with exit0:0 in00:02:20. Both configurations ran sequentially in the same L40S allocation on the identical saved10,000-passage corpus-wide sample. Pinned E5, FP32,512token cap, masked mean pooling and normalization were held fixed. Each process performed two warmup batches and three measured repetitions.

| Measurement | Batch64 | Batch256 |
| --- | ---: | ---: |
| Repetition seconds | 5.900945,5.929125,5.904316 | 6.642215,6.686149,6.695123 |
| Median seconds | 5.904316 | 6.686149 |
| Passages/second | 1693.676367 | 1495.629267 |
| Peak allocated MiB | 439.943848 | 1351.600586 |
| Peak reserved MiB | 1822 | 5352 |

Batch64 provided approximately13.2% higher throughput in this comparison. These are sequential offline encoding observations, not a controlled serving benchmark or a global batch-size optimum. Increased padding or memory work may contribute to batch256's result, but no causal profiling was performed. Fixed run order and only three repetitions limit generalization.

Post-run comparison verified matching ordered passage IDs, finite vectors and shape10000x384. Maximum absolute component difference was3.2782554626464844e-7; minimum paired cosine was0.9999998807907104. NumPy allclose passed with atol1e-5 and rtol1e-4. Numerical agreement does not prove identical downstream nearest-neighbor rankings, especially for ties.

Decision: use batch64 FP32 as the initial L40S full-corpus configuration. At the sampled throughput, a constant-rate extrapolation is roughly87minutes of encoding for8,841,823passages, excluding checkpoint/I/O/startup costs and workload variability. Full encoding requires verified resumable shards and remains pending.

## Exact E5 validation baseline

HPC job 23620628 completed with exit 0:0 in 14 minutes 29 seconds.
Retrieval used the full 8,841,823-passage corpus, the pinned pretrained
`intfloat/e5-small-v2` encoder and exact inner-product search over
L2-normalized vectors. All 2,000 frozen validation queries had nonempty
rankings. The same binary relevance evaluator and query population were
used for BM25. Values below are the reported six-decimal aggregates.

| Metric | BM25 (k1=0.9, b=0.4) | E5 exact |
| --- | ---: | ---: |
| MRR@10 | 0.161909 | 0.317547 |
| NDCG@10 (binary) | 0.205845 | 0.381328 |
| Recall@20 | 0.436250 | 0.707167 |
| Recall@50 | 0.560583 | 0.828917 |
| Recall@100 | 0.643917 | 0.879083 |
| Recall@1000 | 0.849333 | 0.968500 |

E5 scored higher on each aggregate metric in this validation comparison.
MRR@10 increased by 0.155638 (approximately 96.1% relative), while
Recall@1000 increased by 0.119167. These compare an existing pretrained
encoder with the current BM25 configuration; they are not improvements
from project-specific training. This project validation split is not the
official MS MARCO development benchmark, and the checkpoint may have prior
MS MARCO exposure. Statistical significance and independent evaluator
parity have not yet been established. The final evaluation set remains
reserved. Slurm elapsed time includes the whole job and is not query
latency or isolated search time. Run manifest inspection is the next gate.

### Independent evaluation check

Both baseline reports passed a per-query and aggregate comparison against
pytrec-eval-terrier 0.5.11 (HPC job 23625746). All six metrics matched with
maximum per-query absolute error 0.0 across both 2,000-query runs. The
comparison preserved saved rank order and used top-ten truncation for
reciprocal rank. Metric correctness for these saved runs is independently
corroborated; paired confidence intervals remain pending.

## HNSW one-shard pilot result

HPC job 23655213 completed with exit 0:0 in 23 seconds using four CPU
threads. The reloaded index contains 100,000 vectors at 384 dimensions,
with M=32, efConstruction=200 and probe efSearch=64. HNSW add calls took
12.004085 seconds; this excludes exact-reference additions, input checks,
saving and reload verification. The serialized index is 181,620,879 bytes
(approximately 173.21 MiB). Reported Slurm MaxRSS was 735,396 K; this is
whole-job host memory, not the graph's isolated resident memory.

Across 64 stored-vector probes, mean top-ten ID-set overlap with exact
same-shard neighbors was 0.98125. This is a mechanics check on an easy,
non-held-out corpus prefix and is not comparable to full-corpus judged
Recall@10 or the 2,000-query validation metrics. Equal-score ties can
also affect ID-set overlap. No search-speed improvement was measured.

Index SHA-256:
`1e5e08674e7f3f59a650321f68f783fa779390289cb1942c47537866c074d94e`.
The next gate is full-corpus construction followed by real-query neighbor
recall and relevance evaluation. Construction cost need not scale linearly
from this pilot.

## Full HNSW construction result

HPC job 23657686 completed with exit 0:0 in 22 minutes 16 seconds. The
manifest confirms 8,841,823 vectors, 384 dimensions and all 89 audited
shards. The graph uses M=32 and efConstruction=200, built with eight CPU
threads. It passed reload metadata checks, 89 source-vector reconstruction
checks and direct score checks on stored-vector probes.

Graph-add calls took 1,246.782375 seconds. Full build and verification took
1,329.843163 seconds, including input checks, writes, reload, probes and
final checksums. The index is 16,057,891,835 bytes (approximately 14.96 GiB).
Slurm reported MaxRSS 45,422,652 K, approximately 43.3 GiB for the whole job,
not isolated index memory. The saved efSearch=64 was used for structural
probes and is not a validated query-serving configuration.

Index SHA-256:
`d57192396295fbb01b4dfd12893b46ce53978fc6d99fdb7d415fe4b83bd37419`.
Real-query neighbor recall, judged quality and matched search performance
remain pending. Build success is not evidence of retrieval speedup.

## First full-corpus HNSW validation result

HPC job 23729240 completed with exit 0:0 in 59 seconds. It searched the
verified M=32, efConstruction=200 graph with efSearch=1024, batch size 32,
four CPU threads and top 1,000 candidates for all 2,000 validation queries.
The run manifest records 2,000,000 candidates and 2.199586 seconds inside
FAISS search calls. This is a single offline pass without warmup or
repetition; it excludes loading, writes, checksums, scoring and comparison.
The exact baseline ran on another node, so no matched speedup or serving
latency claim follows from this timing.

| Metric | Exact E5 | HNSW efSearch=1024 |
| --- | ---: | ---: |
| MRR@10 | 0.317547 | 0.310898 |
| Binary NDCG@10 | 0.381328 | 0.373000 |
| Judged Recall@20 | 0.707167 | 0.692917 |
| Judged Recall@50 | 0.828917 | 0.813667 |
| Judged Recall@100 | 0.879083 | 0.863333 |
| Judged Recall@1000 | 0.968500 | 0.951000 |

Mean exact-neighbor ID-set recall was 0.9755 at 10, 0.9687 at 100 and
0.9153245 at 1,000. This metric measures agreement with saved exact
neighbors, not relevance. Equal-score alternatives may reduce ID overlap.
Final evaluation queries were not read. The observed relevance loss
motivates a higher-efSearch run; optimal search settings and confidence
intervals remain unestablished.

Evaluation SHA-256:
`3cfb655f925626184464ee912e0c59197576c67818b0dba2bab06c240f492476`.

## HNSW efSearch=2048 validation result

Job 23729437 completed with exit 0:0 in 56 seconds. It reused the same
full graph, frozen queries, batch size 32, four CPU threads and top-1,000
depth. Graph, query and exact-run manifest hashes match the ef1024 run.

| Metric | efSearch=1024 | efSearch=2048 | Exact E5 |
| --- | ---: | ---: | ---: |
| MRR@10 | 0.310898 | 0.314272 | 0.317547 |
| Binary NDCG@10 | 0.373000 | 0.377113 | 0.381328 |
| Judged Recall@20 | 0.692917 | 0.700417 | 0.707167 |
| Judged Recall@50 | 0.813667 | 0.823167 | 0.828917 |
| Judged Recall@100 | 0.863333 | 0.872333 | 0.879083 |
| Judged Recall@1000 | 0.951000 | 0.960250 | 0.968500 |
| Neighbor recall@10 | 0.975500 | 0.986000 | 1.0 by definition |
| Neighbor recall@100 | 0.968700 | 0.980865 | 1.0 by definition |
| Neighbor recall@1000 | 0.915325 | 0.957371 | 1.0 by definition |

All reported relevance and neighbor aggregates improved at ef2048. The
observed search-call time was 4.781392 seconds, compared with 2.199586 at
ef1024. These single passes ran on different nodes (cs623 versus cs624);
they are observations, not a controlled timing ratio. Job duration was
shorter despite more search-call time, illustrating why complete Slurm
runtime must not substitute for isolated search timing. Final queries were
not read. A third quality setting, ef4096, precedes matched benchmarking.

## HNSW efSearch=4096 validation result

Job 23729642 completed with exit 0:0 in 66 seconds, using the same graph,
query vectors, batch size 32 and four CPU threads as the earlier settings.
All input manifest identities remained consistent. Final queries were not
read. The single-pass FAISS search-call total was 17.768442 seconds.

| Metric | ef1024 | ef2048 | ef4096 | Exact E5 |
| --- | ---: | ---: | ---: | ---: |
| MRR@10 | 0.310898 | 0.314272 | 0.315047 | 0.317547 |
| Binary NDCG@10 | 0.373000 | 0.377113 | 0.378162 | 0.381328 |
| Judged Recall@20 | 0.692917 | 0.700417 | 0.702167 | 0.707167 |
| Judged Recall@50 | 0.813667 | 0.823167 | 0.824417 | 0.828917 |
| Judged Recall@100 | 0.863333 | 0.872333 | 0.874083 | 0.879083 |
| Judged Recall@1000 | 0.951000 | 0.960250 | 0.962500 | 0.968500 |
| Neighbor recall@10 | 0.975500 | 0.986000 | 0.990150 | 1.0 by definition |
| Neighbor recall@100 | 0.968700 | 0.980865 | 0.986745 | 1.0 by definition |
| Neighbor recall@1000 | 0.915325 | 0.957371 | 0.976986 | 1.0 by definition |

Increasing efSearch from 2048 to 4096 improved judged Recall@1000 by
0.00225 and MRR@10 by approximately 0.000776. Neighbor recall@1000
increased by 0.019615. These are observed validation aggregates rather
than statistically established gains. Quality improvements were smaller
than the preceding ef1024-to-2048 step.

The ef2048 and ef4096 runs report the same hostname but ran in separate
allocations without warmup or repetitions; the timing comparison is still
uncontrolled. Do not derive serving percentiles by dividing batch time by
query count. The next systems gate is a same-node benchmark covering both
exact and HNSW indexes, common query orders, explicit warmup and repeated
trials. Single-query search-call latency must be labeled separately from
batch throughput and from complete serving latency. No final efSearch
selection or matched speedup claim has been made.


## Matched CPU search benchmark completed

HPC job 23730406 completed with exit 0:0 in 42 minutes 50 seconds on
cs661.hpc.nyu.edu. Exact and HNSW ran in the same process on the same node
with four CPU threads, top 1,000 candidates, explicit warmup and four
rotating-order repetitions. Batch size was 32 over 2,000 queries. The
single-query workload used the same fixed 200-query sample across four
repetitions (800 repeated observations per algorithm). Quality columns
below come from the separate frozen-validation relevance runs over the
same graph/query artifacts; this benchmark does not read judgments.

| Configuration | Median batch QPS | Search p50 ms | Search p95 ms | MRR@10 | Judged Recall@1000 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact | 4.234 | 492.753 | 547.978 | 0.317547 | 0.968500 |
| HNSW ef1024 | 863.083 | 4.439 | 6.384 | 0.310898 | 0.951000 |
| HNSW ef2048 | 415.199 | 9.225 | 12.643 | 0.314272 | 0.960250 |
| HNSW ef4096 | 177.205 | 20.304 | 27.224 | 0.315047 | 0.962500 |

Observed ratios relative to exact search in this benchmark:

- HNSW ef1024: 203.87x median batch throughput; 85.84x lower search p95.
- HNSW ef2048: 98.07x median batch throughput; 43.34x lower search p95.
- HNSW ef4096: 41.86x median batch throughput; 20.13x lower search p95.

These are warm FAISS search-call measurements, not full-service throughput
or latency. Query preparation, encoding, network, queueing, document
fetching and reranking are excluded. Both indexes remain resident. Process
peak RSS/Slurm MaxRSS is whole-job memory, not an isolated comparison of
index residency. The node was not reserved exclusively. Pooled latency
observations repeat the same query sample and are not independent users.
Four repetitions describe this allocation; no cross-node or multi-seed
performance generalization is established.

Exact batch search seconds were 519.285333, 467.573178, 477.353463 and
460.203777. HNSW ef1024 seconds were 2.385249, 2.253065, 3.105179 and
2.067402; ef2048 seconds were 5.214184, 4.961618, 4.680519 and 4.485352;
ef4096 seconds were 12.305466, 11.829346, 10.791075 and 10.525997.
The full raw report is needed for per-repetition percentile inspection
and reproducible graphs.

Working candidate: ef2048 offers a useful measured quality/cost balance.
Its judged Recall@1000 is 0.00825 below exact (0.825 percentage points),
retaining approximately 99.15% of the exact recall value. This is a relative
comparison of macro metrics, not a per-query success guarantee. The final
configuration decision remains subject to reviewing raw results and the
downstream reranking candidate budget. No final-set result is reported.
