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
