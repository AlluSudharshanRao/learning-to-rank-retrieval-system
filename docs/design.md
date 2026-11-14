# System design

The diagrams distinguish implemented correctness checks from the planned retrieval system. Solid arrows show data flow; dashed arrows show a dependency or feedback path.

## Implemented encoder and index checks

This path uses the fixture in `src/ltr_system/defaults/smoke_inputs.json`: one query and three passages. It does not use a benchmark corpus or serve requests.

```mermaid
flowchart TD
    Config["Pinned encoder configuration"] --> Download["Checkpoint download"]
    Download --> Cache["Local model snapshot"]
    Fixture["Query and passage fixture"] --> Tokens["Token IDs and attention mask"]
    Cache --> Encoder["E5 encoder on allocated GPU"]
    Tokens --> Encoder
    Encoder --> Pool["Masked mean pooling and L2 normalization"]
    Pool --> Vectors["Query vectors, passage vectors, and IDs"]
    Pool --> Provenance["Model and runtime provenance"]
    Vectors --> Flat["CPU FAISS Flat with ID map"]
    Vectors --> Reference["Direct dot products"]
    Flat --> Check["Compare scores and passage IDs"]
    Reference --> Check
    Flat --> Disk["Saved FAISS index"]
    Disk --> Reload["Reload and repeat search"]
    Reload --> Check
```

`IndexFlatIP` compares each query with every stored passage vector. The check accepts either ordering for tied scores, but each returned ID must match its reference score. Normalization makes the inner product equivalent to cosine similarity.

## Planned retrieval and reranking

All serving components below are planned. The encoder and exact-index checks establish primitives for this design, not an implemented online service.

```mermaid
flowchart TD
    Query["Search query"] --> API["Search API"]
    API --> Dense["GPU query encoder"]
    API --> Lexical["BM25 retrieval"]
    Dense --> ANN["CPU ANN index"]
    ANN --> Candidates["Candidate IDs and retrieval scores"]
    Lexical --> Candidates
    Candidates --> Fusion["Optional reciprocal rank fusion"]
    Fusion --> Budget["Fixed reranking depth"]
    Budget --> Lookup["Fetch candidate passage text"]
    Store["Passage store"] --> Lookup
    Lookup --> Queue["Bounded pair queue and batching"]
    Queue --> Reranker["GPU cross encoder"]
    Reranker --> Results["Sorted passage IDs and scores"]
    API -.-> Metrics["Request latency, throughput, and errors"]
    Queue -.-> Metrics
    Reranker -.-> Metrics
```

BM25 and fusion are comparison paths rather than mandatory dependencies of dense-only retrieval. Start with a fixed reranking depth; adaptive budgeting is a later experiment. The passage store resolves stable IDs to text. A relevant passage omitted from the candidate set cannot be recovered by the reranker.

The bounded queue limits pending inference work. End-to-end latency must include queue wait, tokenization, retrieval, passage lookup, and reranking. The initial serving target is a single allocated host; the diagram does not imply a distributed deployment.

## Planned training and evaluation

Training and validation queries drive model selection. Final evaluation queries remain outside mining, fitting, and parameter selection. The passage collection may be shared across query splits; judgments must retain their split roles.

```mermaid
flowchart TD
    Collection["Passage collection and query judgments"] --> Split["Freeze query splits and data manifest"]
    Split --> Train["Training queries and known positives"]
    Split --> Dev["Validation queries"]
    Split --> Test["Held out evaluation queries"]
    Train --> Mine["Mine and filter training negatives"]
    Mine --> Fit["Train dense retriever"]
    Fit -.-> Mine
    Fit --> Encode["Reencode corpus and build compatible index"]
    Encode --> Candidates["Retrieve candidates for training queries"]
    Train --> Candidates
    Candidates --> RankFit["Train reranker"]
    Encode --> Select["Select checkpoints and index settings"]
    Dev --> Select
    RankFit --> Select
    Select --> Freeze["Freeze model and index configuration"]
    Freeze --> Evaluate["Final quality and systems evaluation"]
    Test --> Evaluate
    Evaluate --> Report["Raw results, confidence intervals, and report"]
```

The mining feedback path is restricted to training queries. Exclude known positives and filter likely false negatives. Validation candidates are used only for selection. Reranker comparisons use identical saved candidates to isolate changes in ranking. A retriever update requires compatible corpus embeddings and an index.

## Execution and storage

The local source tree is versioned with Git. HPC storage holds checkpoints, caches, environments, vectors, and indexes; these runtime directories are excluded from source control.

```mermaid
flowchart TD
    Checkout["Local source checkout"] --> Checks["Unit tests and lint"]
    Checks --> Transfer["Transfer reviewed source revision"]
    Transfer --> Source["HPC source checkout"]
    Source --> Submit["Slurm submission"]
    Submit --> GPU["Allocated GPU job"]
    Submit --> CPU["CPU indexing or evaluation job"]
    Cache["Model snapshot and data cache"] --> GPU
    GPU --> Artifacts["Embeddings, checkpoints, and provenance"]
    Artifacts --> CPU
    CPU --> Results["Indexes and evaluation outputs"]
    GPU --> Logs["Job logs"]
    CPU --> Logs
    Results --> Retain["Retain selected outputs durably"]
    Artifacts --> Retain
```

GPU smoke jobs and the local source layout are implemented. Corpus indexing, training, and full evaluation jobs remain planned. Downloads and package installation happen before GPU allocation. Scratch is working storage; source and expensive outputs require a durable copy.

## Artifact compatibility

| Artifact | Identity required for reuse |
| --- | --- |
| Encoder checkpoint | Model repository and immutable revision |
| Passage embeddings | Encoder identity, pooling, normalization, truncation, corpus manifest, and passage IDs |
| Vector index | Embedding identity, similarity metric, index parameters, and ID mapping |
| Reranker checkpoint | Base checkpoint, training split, negative/candidate generation, and objective |
| Evaluation run | Code revision, frozen configuration, query split, judgments, hardware, and load profile |

The checkpoint manifest and smoke-test provenance are implemented. Corpus-scale manifests, reranker metadata, and experiment records will be added with their pipelines.
