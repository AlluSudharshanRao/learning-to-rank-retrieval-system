"""Validation HNSW retrieval with judged metrics and exact-neighbor agreement."""

from __future__ import annotations

import logging
import math
import os
import socket
import tempfile
from pathlib import Path

from ltr_system.data_download import sha256_file
from ltr_system.dense_search import retrieve
from ltr_system.evaluation import evaluate_files, load_rankings
from ltr_system.project import initialize, parser, read_json, write_json
from ltr_system.query_encode import validation_inputs

LOGGER = logging.getLogger(__name__)


def neighbor_agreement(query_ids, approximate, exact, cutoffs=(10, 100, 1000)) -> dict:
    if not query_ids or len(set(query_ids)) != len(query_ids):
        raise ValueError("Expected unique nonempty query population")
    if set(approximate) != set(query_ids) or set(exact) != set(query_ids):
        raise ValueError("Neighbor rankings have different query populations")
    if not cutoffs or any(type(k) is not int or k <= 0 for k in cutoffs):
        raise ValueError("Neighbor cutoffs must be positive integers")
    if len(set(cutoffs)) != len(cutoffs):
        raise ValueError("Duplicate neighbor cutoffs")
    per_query = {}
    for qid in query_ids:
        a, b = approximate[qid], exact[qid]
        if len(set(a)) != len(a) or len(set(b)) != len(b) or min(len(a), len(b)) < max(cutoffs):
            raise ValueError("Neighbor rankings contain duplicates or insufficient depth")
        per_query[qid] = {f"neighbor_recall@{k}": len(set(a[:k]) & set(b[:k])) / k for k in cutoffs}
    return {
        "queries": len(query_ids),
        "cutoffs": list(cutoffs),
        "per_query": per_query,
        "metrics": {
            name: math.fsum(values[name] for values in per_query.values()) / len(query_ids)
            for name in next(iter(per_query.values()))
        },
        "definition": "macro ID-set overlap between approximate top-K and saved exact top-K; not judged relevance",
        "tie_policy": "not tie-aware; equal-score alternatives can reduce ID overlap",
    }


def run(root: Path, ef_search: int, batch_size: int, threads: int) -> dict:
    import faiss
    import numpy as np

    k = 1000
    if ef_search < k or min(batch_size, threads) <= 0:
        raise ValueError("Use efSearch >= 1000 and positive batch size/threads")
    run_name = f"e5-hnsw-validation-ef{ef_search}-b{batch_size}-t{threads}"
    destination = root / "artifacts/runs" / run_name
    if destination.exists():
        raise FileExistsError(destination)
    graph_root = root / "artifacts/indexes/e5-hnsw-m32-efc200-v1"
    exact_root = root / "artifacts/indexes/e5-flat-ip-v1"
    exact_run = root / "artifacts/runs/e5-exact-validation-default"
    query_root = root / "artifacts/queries/e5-validation-fp32"
    split_root = root / "data/processed/msmarco-passage-v1-splits"
    manifest_paths = [
        p / "manifest.json" for p in (graph_root, exact_root, exact_run, query_root, split_root)
    ]
    expected = {path: sha256_file(path) for path in manifest_paths}
    graph, flat, baseline, queries, split = [read_json(path) for path in manifest_paths]
    ids, _, source = validation_inputs(split_root)
    if (
        graph.get("status") != "full HNSW index built and reload checked"
        or graph["index_type"] != "IndexIDMap2(IndexHNSWFlat)"
        or graph["metric"] != "inner product"
        or flat.get("status") != "exact index built and reload checked"
        or baseline.get("status") != "validation retrieval and scoring completed"
        or baseline["source"] != source
        or baseline["k"] < k
        or baseline["query_manifest_sha256"] != expected[query_root / "manifest.json"]
        or baseline["index_manifest_sha256"] != expected[exact_root / "manifest.json"]
        or graph["coverage_sha256"] != flat["coverage_sha256"]
        or graph["embedding_configuration_sha256"] != flat["embedding_configuration_sha256"]
        or queries.get("status") != "validation queries encoded"
        or queries["source"] != source
        or queries["passage_configuration_sha256"] != graph["embedding_configuration_sha256"]
        or graph["vectors"] != flat["vectors"]
        or graph["dimension"] != flat["dimension"]
        or queries["encoder"]["dimension"] != graph["dimension"]
        or queries["queries"] != len(ids)
        or (queries["prefix"], queries["normalization"], queries["pooling"], queries["precision"])
        != ("query: ", "L2", "masked mean", "float32")
        or (graph["hnsw"]["M"], graph["hnsw"]["efConstruction"]) != (32, 200)
    ):
        raise ValueError("Incompatible HNSW, exact baseline, query artifacts or validation split")
    expected.update(
        {
            graph_root / "hnsw.faiss": graph["files"]["hnsw.faiss"]["sha256"],
            exact_run / "ranking.tsv": baseline["ranking_sha256"],
            **{
                query_root / name: queries["files"][name]["sha256"]
                for name in ("vectors.npy", "query_ids.json")
            },
            split_root / "validation.queries.tsv": source["query_file_sha256"],
            split_root / "validation.qrels.tsv": split["outputs"]["validation.qrels.tsv"]["sha256"],
        }
    )

    def verify():
        for path, digest in expected.items():
            if sha256_file(path) != digest:
                raise ValueError(f"Input checksum mismatch: {path}")

    LOGGER.info("Verifying graph, saved baseline and validation query identities")
    verify()
    if read_json(query_root / "query_ids.json")["query_ids"] != ids:
        raise ValueError("Query vector ID order mismatch")
    vectors = np.load(query_root / "vectors.npy", allow_pickle=False)
    faiss.omp_set_num_threads(threads)
    index = faiss.read_index(str(graph_root / "hnsw.faiss"))
    if not isinstance(index, faiss.IndexIDMap2):
        raise ValueError("Expected external passage ID mapping")
    backend = faiss.downcast_index(index.index)
    if (
        not isinstance(backend, faiss.IndexHNSWFlat)
        or index.ntotal != graph["vectors"]
        or index.d != graph["dimension"]
        or index.metric_type != faiss.METRIC_INNER_PRODUCT
        or backend.hnsw.nb_neighbors(0) != 64
        or backend.hnsw.efConstruction != 200
    ):
        raise ValueError("Loaded graph configuration mismatch")
    backend.hnsw.efSearch = ef_search
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".hnsw-run-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "run"
        stage.mkdir()
        counts = retrieve(index, vectors, ids, stage / "ranking.tsv", k, batch_size)
        result = evaluate_files(
            split_root / "validation.queries.tsv",
            split_root / "validation.qrels.tsv",
            stage / "ranking.tsv",
            stage / "evaluation.json",
            run_name,
        )
        neighbors = neighbor_agreement(
            ids, load_rankings(stage / "ranking.tsv"), load_rankings(exact_run / "ranking.tsv")
        )
        write_json(stage / "neighbors.json", neighbors)
        verify()
        write_json(
            stage / "manifest.json",
            {
                "schema_version": 1,
                "status": "HNSW validation retrieval and comparison completed",
                "run_name": run_name,
                "split": "validation",
                "final_queries_read": False,
                "index_manifest_sha256": expected[graph_root / "manifest.json"],
                "query_manifest_sha256": expected[query_root / "manifest.json"],
                "exact_run_manifest_sha256": expected[exact_run / "manifest.json"],
                "source": source,
                "efSearch": ef_search,
                "k": k,
                "batch_size": batch_size,
                "threads": threads,
                "counts": counts,
                "faiss": faiss.__version__,
                "timing_scope": "one offline pass of search calls; no warmup or repetition; excludes loading, writes, scoring and comparison; not serving latency or matched speedup",
                "neighbor_metrics": neighbors["metrics"],
                "relevance_metrics": result["metrics"],
                "files": {
                    name: {
                        "sha256": sha256_file(stage / name),
                        "bytes": (stage / name).stat().st_size,
                    }
                    for name in ("ranking.tsv", "evaluation.json", "neighbors.json")
                },
                "host": socket.gethostname(),
                "job_id": os.environ.get("SLURM_JOB_ID"),
            },
        )
        stage.rename(destination)
    LOGGER.info("Judged relevance: %s", result["metrics"])
    LOGGER.info("Exact-neighbor agreement: %s", neighbors["metrics"])
    return result


def main() -> None:
    command = parser("Evaluate full-corpus HNSW against judgments and saved exact neighbors.")
    command.add_argument("--ef-search", type=int, default=1024)
    command.add_argument("--batch-size", type=int, default=32)
    command.add_argument("--threads", type=int, default=4)
    args = command.parse_args()
    root = initialize(args)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run full HNSW retrieval in a CPU Slurm allocation")
    if args.threads > int(os.environ.get("SLURM_CPUS_PER_TASK", "1")):
        raise ValueError("Threads exceed allocated CPUs")
    run(root, args.ef_search, args.batch_size, args.threads)
