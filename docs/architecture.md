# Architecture

The implemented path covers pinned checkpoint loading, GPU encoding, and exact CPU vector search. Training, corpus ingestion, ANN, reranking, hybrid retrieval, and serving remain planned. Current correctness checks are recorded in [Validation](validation.md).

The [system design diagrams](design.md) show the implemented data path, planned online pipeline, training flow, and execution boundaries.

## Problem

Return relevant passages from a large collection, rank direct answers above merely related text, and control response time and compute cost. Retrieval quality and serving efficiency must be measured separately.

## Offline work

Prepare data and query splits; train the retriever; encode passages in resumable shards; build vector and lexical indexes; mine negatives using training queries; train the reranker; evaluate frozen artifacts. After passage encoder weights change, regenerate passage embeddings and rebuild the corresponding index. Track model and index compatibility explicitly.

## Request path

Encode a query, retrieve candidates using dense ANN search and optionally BM25 fusion, select a reranking depth, score query-passage pairs, sort scores, and return passage IDs with timing information. An adaptive depth policy is a hypothesis to test after fixed-depth baselines.

## Retriever

Encoder checkpoint: intfloat/e5-small-v2, using model-prescribed query and passage prefixes. A shared encoder processes each input separately and generates a 384-dimensional vector. Contrastive training encourages positives to score higher than negative passages. Start with in-batch negatives, then compare BM25 negatives and refreshed dense negatives. Other passages in a batch can be false negatives, so duplicates and known positives need handling.

Normalize vectors for cosine-equivalent inner-product search. FAISS Flat is the exact reference; HNSW provides approximate search with tunable memory, accuracy, and latency. ANN neighbor overlap is distinct from relevance recall.

## Reranker

A cross-encoder processes the query and one passage jointly. Attention permits interaction between their tokens, and a head produces a scalar relevance score. The score is not automatically a calibrated probability. Batch pairs for efficient inference; each pair still receives an independent score.

Initial proposed training objective: pairwise logistic loss, log(1 + exp(s_negative - s_positive)). Compute two pair scores and compare them in the loss; the positive and negative passages do not need to enter one model input together. Implement with a numerically stable softplus operation when coding begins.

Compare a trained MiniLM reranker with both its starting checkpoint and an untouched pretrained ranking checkpoint. Record prior training exposure. Use the same saved candidates for controlled reranker comparisons.

## Main limitations and tradeoffs

- A reranker cannot recover a passage missing from its candidate set.
- Separate passage encoding enables precomputation but compresses relevance information into a fixed vector.
- Cross-encoding adds query-specific interactions but repeats passage computation per request.
- Reranking more candidates can help quality but increases work; any gain must be measured.
- Batching can increase throughput while queue waiting increases request latency.
- CPU indexing needs appropriate RAM; a large GPU does not replace CPU memory.

## References

- https://huggingface.co/intfloat/e5-small-v2
- https://sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html
- https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
- https://sbert.net/docs/package_reference/sentence_transformer/losses.html

Checkpoint and package versions are recorded in configuration and dependency files.

## Current source boundaries

`project.py` resolves runtime storage, validates checkpoint configuration, and writes JSON metadata atomically. `encoding.py` provides masked mean pooling without importing model code at module load. `vectors.py` validates normalized vectors and checks exact rankings including tied scores. `commands.py` orchestrates explicit CLI actions with lazy optional imports. `scripts/` contains source-checkout entry points; no script performs work merely because it is imported.

Configuration and illustrative texts are versioned package data. Runtime manifests, embeddings, and indexes remain under ignored artifacts. The current examples are smoke fixtures, not a benchmark dataset. CLI output preserves readable checks; provenance metadata provides machine-readable context. Failure handling uses explicit exceptions and nonzero exits.
