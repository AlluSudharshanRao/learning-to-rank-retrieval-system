# Data protocol

Status: official source archive downloaded and validated on HPC. Extraction and record validation implemented and tested locally; HPC record validation completed successfully; split generation remains pending.

## Dataset and scope

Use MS MARCO Passage v1. The official collection contains 8,841,823 passages. Preserve source passage and query identifiers as strings in canonical records; assign explicit integer mappings only when building vector indexes. Do not substitute the document-ranking collection or passage v2.

The initial ingestion uses the official collection, query texts, and relevance judgments. Training triples and precomputed candidate lists are unnecessary for this milestone: later hard negatives will come from recorded retrieval runs.

## Query roles

| Project role | Source | Permitted use |
| --- | --- | --- |
| Training | Official training queries and judgments, excluding project validation groups | Model fitting and hard-negative mining |
| Validation | 2,000 eligible query groups reserved from the official training pool | Hyperparameter, index, and reranking-policy selection |
| Final evaluation | Official development-small queries and judgments | Frozen final comparisons |
| Supplementary evaluation | TREC DL 2019 and 2020 passage queries and graded judgments | Final graded-relevance comparisons, after core ingestion |

The official development set serves as a held-out set for this project; it is not the private MS MARCO leaderboard test set. Public pretrained checkpoints may already have seen MS MARCO supervision. Record that exposure and compare trained models with their unchanged starting checkpoint.

Before selecting training or validation queries, group query texts using Unicode NFKC normalization, case folding, and whitespace collapsing. Exclude training-pool groups matching final evaluation query text. This overlap check uses query texts, not final judgments. Keep final evaluation queries unchanged.

Order eligible groups by SHA-256 of `ltr-v1-split-2026` followed by a newline and the normalized text, with normalized text as a tie breaker. Reserve the first 2,000 groups for validation. Begin training with 50,000 queries from the remaining groups in that order, retaining complete groups at the boundary. Record the actual query count. Expansion must preserve validation membership. Fail explicitly if the source cannot supply the requested split sizes.

## Corpus policy

Full-corpus comparisons use the complete passage collection. An ingestion fixture is for parser and integrity checks only. It is not a relevance benchmark.

A larger reduced corpus requires a separate versioned construction policy before use. Never build it by selecting passages using final evaluation judgments. Report reduced-corpus results separately; they are not comparable to full-corpus leaderboard scores.

## Ingestion contract

Keep downloaded source files immutable under `data/raw/`. Write derived records and split manifests under `data/processed/`. Neither directory belongs in Git.

Record source URL, retrieval time, byte count, SHA-256, archive member names, parser version, and output counts. Local hashes identify downloaded bytes; they do not independently authenticate the source. Verify publisher checksums where available. Extract only explicitly permitted regular-file members, rejecting path traversal and links.

Validate unique identifiers, valid UTF-8, expected field counts, nonempty texts, parseable relevance labels, and query/passage references. Produce explicit counts for duplicates, overlaps, missing references, and rejected records. Do not silently drop malformed rows or rewrite source identifiers. Stream corpus processing rather than loading passage text into one Python dictionary.

Freeze manifests before encoding or training. Runs must identify the corpus and split manifests they used. Review source terms before downloading or redistributing; publish preparation code and manifests rather than raw passage text.

## First acceptance gate

- Download and validate the required official source files on HPC scratch storage.
- Produce a source manifest and an integrity report.
- Test parsing and failure handling with small synthetic fixtures locally.
- Generate deterministic, disjoint query manifests and record their counts.
- Keep final judgments out of training, mining, and tuning inputs.

No dataset-scale quality or throughput results have been measured yet.

## References

- [Microsoft dataset catalog](https://github.com/microsoft/msmarco/blob/master/Datasets.md)
- [MS MARCO passage ranking task](https://github.com/microsoft/MSMARCO-Passage-Ranking)
- [TREC Deep Learning 2020 passage resources](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2020.html)

## Download command

Transfer `scripts/download_dataset.py` and the current `src/ltr_system/` tree together. Run the source wrapper with the project Python environment:

```bash
python -I scripts/download_dataset.py --project-root "$SCRATCH/learning-to-rank"
```

Use a CPU Slurm job for archive validation; no GPU is needed. The download uses the combined archive linked in the Microsoft dataset catalog. Source terms limit the data to non-commercial research; raw data remains excluded from Git.

The command writes `data/raw/msmarco-passage-v1/collectionandqueries.tar.gz` and `source.json`. It checks HTTP length when supplied, validates gzip integrity, rejects unsafe/nonregular/duplicate members, and requires the corpus plus training and development-small query/judgment files. It inventories additional regular files without extracting them. Unexpected upstream packaging fails explicitly.

Existing archives require a matching source manifest and are verified by size and SHA-256. Failed transfers are removed; retrying starts from zero. Archive and manifest publication are separate operations: a crash between them leaves an archive without a manifest and requires inspection. Run one writer per dataset directory. Local checksums identify bytes and detect later changes; no independent publisher SHA-256 is configured.

The command does not extract files, validate TSV records, count corpus passages, or generate query splits. Those checks are the next ingestion stage.

## Record preparation

Run `python -I scripts/prepare_dataset.py --project-root "$SCRATCH/learning-to-rank"` in a CPU Slurm allocation after source download. Transfer the wrapper and `src/ltr_system/data_prepare.py` first. Use 2 CPU cores, 4 GB RAM, and a two-hour initial time limit; actual preparation resource requirements remain unmeasured. Keep at least 10 GB free for extracted records and the temporary SQLite ID registry.

Preparation rechecks source bytes and archive inventory before extracting only the five required regular files into a staging directory. It preserves TSV bytes and uses SQLite primary keys for ID and judgment uniqueness. Strict UTF-8, field counts, nonempty text, numeric identifiers, binary labels, and judgment references are checked. Parser version 2 counts identical repeated judgments and retains one logical pair in its validation database; conflicting labels fail validation. Extracted TSV bytes remain unchanged, so downstream readers must also deduplicate logical query-passage judgments. The integrity report separates source rows, unique judgments, and identical duplicate rows per judgment file. The official collection must contain exactly 8,841,823 records. Validation fails at the first malformed record rather than producing a success report with discarded rows.

On success, `data/processed/msmarco-passage-v1/` contains the five TSVs and `integrity.json`, with record counts and per-file hashes. The temporary ID database is removed. Existing output is refused. Failed runs do not publish processed output. One writer per destination is required; forceful termination can leave an unpublished `.prepare-*` directory for inspection. Query ID overlap is reported; normalized text overlap is deferred to split generation. No query split is generated here.

The downloaded source archive has SHA-256 `decac356eb8cc5b9cea2e30b8738dc6f367e4147aaefe8fc7526ddda382fd2fc`, size 1,057,717,952 bytes, and eight regular-file members. Source download completed in 00:09:34 with Slurm exit code 0:0. These observations do not establish parser correctness on the official records.

## Query split implementation

`python -I scripts/split_dataset.py --project-root "$SCRATCH/learning-to-rank"` writes `data/processed/msmarco-passage-v1-splits/`. Defaults are 2,000 validation groups and at least 50,000 training queries. Complete normalized-text groups can make actual query counts larger than these targets. The command refuses an existing destination and stages output before publication.

Only training queries with judgments are eligible. Training inputs and final query text are verified against the processed integrity report before selection. Final judgments are not read. The command excludes training queries matching final query IDs or normalized text, orders eligible groups using the documented SHA-256 seed, reserves validation first, then takes training groups. Output order is deterministic for the same inputs, settings, and Unicode normalization behavior; the manifest records the Unicode database version.

Outputs are `train.queries.tsv`, `validation.queries.tsv`, their deduplicated qrels files, `final.query_ids.json`, and `manifest.json`. The final ID list preserves final-set membership; final query text and judgments remain in the validated source directory. The manifest records source identity, selection counts, exclusion counts, unique judgment counts, and output hashes. Full corpus identity is inherited from the integrity report; splitting does not rehash or modify the passage collection.

Queries and normalized group metadata are held in RAM, unlike corpus text. Memory use on official data has not yet been measured. The first split version checks development-small overlap only. TREC supplementary query exclusion must be added before any training checkpoint is claimed to follow the supplementary held-out protocol; otherwise disclose that limitation. Normalization detects formatting and exact-text duplicates, not all semantic paraphrases. Shared passages across query splits are permitted for this retrieval task.

Local synthetic tests passed for normalization, group integrity, input-order independence, stable validation under training expansion, insufficient data, modified-source rejection, output hashes, duplicate-judgment removal, and repeatable artifacts. The official split command has not yet run on HPC.

### Official split result

The first official run produced exactly 50,000 training queries and 2,000 validation queries, with 52,867 and 2,113 unique relevance judgments respectively. Final membership remains 6,980 queries. The manifest reports zero selected-split ID and normalized-text overlap; final judgments were not read. Supplementary TREC exclusion remains pending. Preserve these artifacts and their manifest identity for subsequent experiments.

## Lexical corpus conversion

`scripts/prepare_lexical_corpus.py` converts the validated collection TSV into `data/processed/msmarco-passage-v1-jsonl/shards/part-*.jsonl`. Each line contains `id` and `contents` for Pyserini JsonCollection. Source order, string IDs, and passage text are preserved; JSON escaping is reversible and no linguistic normalization is performed. Lucene's analyzer is a separate indexing choice.

The default `--shard-size 100000` should produce 89 files for 8,841,823 passages, with 41,823 records in the last file. Shards let the Java indexer process multiple input files concurrently and avoid creating millions of filesystem entries. This shard size is an initial engineering choice, not a measured optimum; indexing thread count is configured separately.

The converter verifies the collection against integrity.json, streams records with bounded passage-text memory, checks output count, and rehashes the source and integrity report to detect changes during conversion. Global source-ID uniqueness is inherited from the validated source identity rather than recomputed. It writes per-shard counts, first/last IDs, bytes and SHA-256 to manifest.json outside the shards directory. Only the shards directory should be passed to JsonCollection. Existing output is refused; staging and final rename prevent publication of ordinary failed runs. Concurrent writers and hard-kill cleanup require separate handling.

Four local tests cover text/ID round trips, shard boundaries, deterministic output, changed-source rejection, failed-run cleanup and existing-output refusal. All 50 local tests and Ruff passed. Official conversion and full Lucene indexing remain pending.

## Lucene index build

Official conversion completed in 00:01:18 with exit 0:0: 8,841,823 passages in 89 shards, last shard 41,823 records. Counts and source collection hash match the validated input. Full Lucene indexing remains pending.

`scripts/build_lexical_index.py` builds `artifacts/indexes/bm25-v1/index/` from the JSONL shards. It verifies shard membership, counts, sizes, and SHA-256 before indexing and again before publication. Indexing and count verification run in separate child processes; both JVMs exit before the staged directory is renamed. Failed stages are retained under `.bm25-build-*` for diagnosis. Existing published output is refused. Use a single writer and immutable source data.

Initial settings: four threads, English analyzer, Porter stemming, default stopword removal, and 16 GB Java heap. Request 32 GB total Slurm memory to leave native/Python overhead; this budget is not a measured requirement. The builder overrides command-local JAVA_TOOL_OPTIONS with the selected heap size. JAVA_HOME and JVM_PATH must identify the separate BM25 runtime. Build timeout is three hours and count-verification timeout five minutes; use a four-hour Slurm allocation.

The index stores term frequencies without positions, document vectors, or raw passage text. This supports the lexical BM25 baseline; passage lookup for later reranking uses a separate corpus lookup path. BM25 k1 and b are search-time settings, not chosen by this index build. Analyzer behavior is recorded because it changes token matching independently of k1 and b.

Each build retains build-config.json, index.log, verification.json, and manifest.json. The final manifest records source identities, indexed document count, Pyserini/Java versions, analyzer/storage settings, threads, heap, host/job, indexing subprocess duration, and index-file hashes excluding write.lock. Duration includes child startup and is not query latency. Matching document count is a necessary integrity check, not proof of semantic relevance or a complete per-ID audit. Independent quality evaluation follows retrieval. Code revision and complete environment snapshots must accompany reported experiments.

Three additional local tests verify unchanged shards, tamper detection, unlisted files, duplicate names, traversal, and inconsistent counts. The local suite now passes 53 tests; actual full-corpus JVM execution remains pending on HPC.

## Dense encoding pilot

scripts/encoding_pilot.py encodes a verified prefix of the first JSONL shard, initially 10,000 passages with batch size 64. It uses the revision-pinned E5-small-v2 snapshot offline, passage prefixes, FP32 without autocast, a 512-token cap, attention-masked mean pooling and L2 normalization. Two warmup batches precede timing. Timing includes tokenization, CPU/GPU transfers, forward inference, pooling and CPU vector collection; it excludes model loading, warmup and saving. Peak PyTorch allocated/reserved GPU bytes are recorded and do not represent total device memory or GPU utilization.

Artifacts under artifacts/encoding-pilots/e5-n10000-b64 contain vectors.npy, ordered string passage IDs and a hashed manifest linking model/source settings and GPU/job. Existing output is refused. Prefix selection is deterministic but is not a representative randomized workload or relevance benchmark. Two local source-selection tests cover prefix/ID preservation and duplicate/insufficient-ID failures. GPU encoding execution remains pending.

The full 8,841,823 x 384 FP32 embedding matrix alone requires 13,581,040,128 bytes, excluding IDs, ANN structures and temporary copies. Full encoding needs resumable shards and a measured compute budget; this bounded pilot does not implement the final corpus encoding pipeline. Truncation frequency and sequence-length distributions remain to be characterized before extrapolating throughput.

## Full-corpus sampled token-length profile

scripts/profile_passage_lengths.py selects 10,000 distinct corpus-row positions uniformly without replacement using seed2026, scans verified JSONL shards, and retains sampled records in corpus order. It tokenizes with the pinned offline E5 tokenizer on CPU, adding the passage prefix and special tokens, with padding/truncation disabled. It reports minimum/mean/maximum, nearest-rank p50/p90/p95/p99, and the number/fraction strictly above512tokens. Equality to512 does not require truncation.

Artifacts/profiles/e5-n10000-seed2026 contains sample.jsonl for reuse in a GPU pilot, aligned token_lengths.json and manifest.json. The manifest records sampling/Python/model identity, hashes and length statistics. This estimates the sampled passage-length distribution, not retrieval quality or full corpus truncation counts. Uniform sampling reduces prefix bias but still has sampling error. It does not measure padding inflation; that also depends on batch ordering. Keep source shards immutable during profiling. No model forward pass, GPU or training is needed. Official profiling is pending.

## Saved-sample GPU pilot

encoding_pilot.py now accepts --sample-profile (directory containing manifest.json and sample.jsonl) and --repeats. It verifies sample checksum, full sample count and pinned tokenizer identity. Default prefix mode remains available. Pilot version2 output names include sample-hash prefix, passage count, batch size and repetition count; prior prefix artifacts remain unchanged.

For a controlled first comparison, use the saved 10,000-passage sample with batch sizes64 and256 and three repetitions each, sequentially in one GPU allocation. Keep FP32, model revision,512token cap and source order fixed. Record all repetition durations and median; passages_per_second divides sample size by median duration. Warmup uses two batches, and allocator peaks span the measured repetitions. Persisted vectors come from the final repetition. No cross-batch vector agreement is asserted automatically; compare matching IDs, norms and vector differences before deciding that configurations are interchangeable. Timings do not include artifact saving and do not establish serving latency or full corpus throughput. Run order, padding distribution and GPU load can affect these initial measurements.

Local tests verify saved sample mapping and detect modified content, mismatched model identity and partial-sample requests. All62 local tests and Ruff passed; actual sample GPU comparisons remain pending.

## Resumable dense corpus encoding

scripts/encode_corpus.py processes a half-open shard range: --start-shard0 --stop-shard1 encodes only the first100,000-passage shard. Defaults are batch64, FP32, pinned offline E5, masked mean pooling, normalized384-dimensional output and max512tokens. Output is artifacts/embeddings/e5-small-v2-fp32/part-00000/ containing vectors.npy, ordered passage_ids.json and manifest.json. No query encoding or retrieval is performed here.

Each input is verified against its shard identity. Output vectors are written batchwise to a NumPy memory-mapped file, checked for shape, FP32 dtype, finite values and unit norms, then flushed and explicitly unmapped before publication. The source is rehashed to detect changes. ID uniqueness and record count are checked. A staged directory is renamed into place only after all files and their checksums are complete. Normal exceptions remove the incomplete stage; forceful termination can leave an unpublished .encoding-* directory.

Reusing a completed shard requires matching model/corpus/encoding identity and verified file sizes, checksums, ID uniqueness/count, vector shape and unit norms. The encoder callback is skipped only after verification succeeds. Incompatible or corrupted output fails explicitly rather than silently overwriting. The configuration includes GPU model and library versions, so a different GPU class or environment requires an explicit compatibility decision, not automatic mixing.

A single-writer lock protects the embedding directory. A hard-killed process can leave .writer.lock; inspect its host/job/process before removing it. Resume occurs at shard boundaries, not within the interrupted shard. Successful publication does not provide storage backup or guarantee survival of a filesystem/power failure. Full-corpus completion will require a separate all-shard coverage check before indexing.

Four synthetic callback tests cover aligned IDs/output, completed-shard skip, interrupted shard cleanup and retry, corrupted/incompatible completed output, and writer exclusion. All66 local tests and Ruff passed. The tests exercise restart behavior without a GPU; first-shard CUDA execution and HPC restart verification remain pending.

## Complete dense embedding audit

scripts/verify_corpus_embeddings.py audits all expected shards in a CPU Slurm allocation. It checks the published shard set, source hashes, common encoding configuration, artifact sizes/checksums, FP32 shape, finite vectors and unit norms, plus ordered saved passage IDs against every source JSONL row. Global source-ID uniqueness is inherited from the validated corpus and exact row alignment, rather than an additional in-memory set over all8.8million IDs. It uses the embedding writer lock to exclude concurrent project encoding.

Successful audit writes coverage.json containing total records, shard count, dimensions, vector file bytes, source/config hashes and per-shard manifest hashes. This is a snapshot audit; later modifications require revalidation. Its existence alone is not proof that a new audit succeeded. Failed audits do not publish a replacement. Source and artifact files must remain immutable during audit. It performs no model inference, retrieval or semantic validation of vector content.

Three synthetic tests check complete/repeatable audits, missing-shard failure and reordered IDs despite updated checksums. All69 local tests and Ruff passed. The full GPU encoding job has completed; the independent CPU coverage audit remains pending before dense indexing.
