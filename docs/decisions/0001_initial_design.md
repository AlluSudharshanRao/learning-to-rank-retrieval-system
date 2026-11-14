# ADR 0001 Initial architecture and experiment scope

**Status:** proposed for the full pipeline. E5 encoding and exact FAISS search have been implemented and validated; dataset and training decisions remain open.

## Context

The project must demonstrate model learning and ML systems optimization with realistic Slurm GPU access. It must support staged implementation, reproducible comparisons, and defensible results.

## Proposed decision

Use MS MARCO Passage v1; E5-small-v2 dense retrieval; BM25 lexical retrieval; FAISS Flat and HNSW; MiniLM cross-encoder reranking; FastAPI serving; MLflow tracking; resumable Slurm jobs. Investigate adaptive reranking depth after strong fixed-depth baselines.

## Alternatives and consequences

Exact-only search simplifies correctness but does not study ANN tradeoffs. ANN requires a separate exact reference. Reranking the entire corpus is too expensive for the intended request path. A large generative reranker increases the initial compute and deployment burden. Distributed training is deferred until a measured bottleneck warrants its complexity. Quantization and IVF-PQ are extensions.

Small encoders permit more controlled experiments but do not guarantee competitive quality. Full-corpus vector and graph storage requires adequate CPU RAM. Existing retrieval checkpoints may already have MS MARCO exposure, limiting claims about new learning.

## Acceptance gate

Before accepting the full design, finalize dataset access, corpus construction, split roles, and compute budget. Model revisions are pinned in package configuration. Record later acceptance or supersession in this decision record.
