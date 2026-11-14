# Evaluation protocol

Status: planned. The current validation covers smoke tests; dataset-scale quality and performance measurements are pending.

## Dataset protocol

Proposed primary data: MS MARCO Passage v1. Keep development subset results separate from full-corpus results. Freeze corpus construction and query splits before comparisons. Never mine training negatives using final evaluation queries or select a policy on final judgments. Record model pretraining exposure and data licenses before redistribution.

Use MS MARCO held-out queries for retrieval and ranking evaluation; TREC DL graded judgments for NDCG; optional BEIR SciFact transfer evaluation after core work. Exact split manifests and tuning/evaluation roles remain to be finalized.

## Metrics

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
