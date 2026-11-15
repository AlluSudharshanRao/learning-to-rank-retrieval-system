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
