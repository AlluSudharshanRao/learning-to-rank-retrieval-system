# Evaluation protocol

Status: data validation and initial query splitting completed. Binary metric functions and fixed-query aggregation are implemented and tested locally. File-based binary evaluation is implemented and locally tested. Graded metrics and dataset-scale quality and performance measurements remain pending.

## Dataset protocol

Proposed primary data: MS MARCO Passage v1. Keep development subset results separate from full-corpus results. Freeze corpus construction and query splits before comparisons. Never mine training negatives using final evaluation queries or select a policy on final judgments. Record model pretraining exposure and data licenses before redistribution.

Use the [data protocol](data.md) for query roles and deterministic split construction. Official development-small queries are reserved for final project evaluation; validation comes from the training pool. Initial split manifests have been generated and checked on HPC. Supplementary TREC overlap exclusion remains a pre-training gate. Optional BEIR SciFact transfer evaluation follows core work.

## Metrics

See [retrieval quality metrics](retrieval_metrics.md) for definitions, a worked example, interpretation, and implementation requirements.

| Layer | Planned metrics |
| --- | --- |
| Candidate relevance | Recall at 20, 50, 100, 1000 |
| Final ordering | MRR at 10 and NDCG at 10 |
| ANN approximation | Top-K overlap against exact vector search |
| Serving | End-to-end p50 and p95, completed requests per second, timeouts and rejections |
| Resources | GPU utilization, peak GPU memory, process RAM and index size |
| Offline work | Training duration, corpus encoding throughput and index build time |

Sparse judgments limit relevance-recall interpretation. Explicitly specify graded gains, cutoffs, binary relevance thresholds, and treatment of unjudged items when implementing metrics.

## Baselines

BM25; pretrained dense exact search; pretrained dense ANN; trained dense ANN; hybrid fusion; fixed-candidate pretrained reranking; fixed-candidate trained reranking; adaptive reranking compared with the best fixed-depth configuration at matched quality or latency.

## Reproducibility requirements

Every run records run ID, code commit and worktree status, configuration, seed, dependency versions, model revision, data/split identity, index identity, hardware, allocation details, timings, raw metric outputs, and failures. Final quality comparisons use paired query-level confidence intervals.

Separate batch-one latency from batched throughput and startup from warm steady state. Include tokenization, passage lookup, queue wait, retrieval, and reranking in end-to-end latency. Use controlled offered load, report failed requests, and record load-generator placement. Repeat important measurements.

## Reporting results

Reports must include baselines, raw outputs, corpus scope, hardware, workload, and absolute and relative changes. Retain regressions and limitations alongside improvements.

## BM25 validation retrieval

scripts/retrieve_bm25.py retrieves only the frozen validation queries and evaluates their binary judgments. Initial parameters are k1=0.9, b=0.4 and depth 1000; these are not tuned optima. The runner checks index-file hashes, split-file identities and document count. It rejects duplicate passages, invalid/nonfinite scores and increasing score order. Lucene returned tie order is preserved; tied membership can depend on index construction.

Output artifacts/runs/bm25-validation-default contains ranking.tsv, evaluation.json and manifest.json. The manifest identifies index/split inputs, configuration, ranking/report hashes, Pyserini version, host/job and retrieval-loop duration. Existing output is refused; results are staged before publication. Final queries and judgments are not read. Keep inputs immutable and use one writer per run. Code commit and the complete environment snapshot remain required for experiment reporting.

The sequential loop duration includes searches, Python result handling and TSV writes, and excludes JVM loading and evaluation. It is not a p50/p95 serving latency or controlled-load throughput benchmark. Official retrieval execution, graded TREC evaluation and independent reference parity remain pending.
