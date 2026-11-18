"""Exact validation retrieval over the audited full-corpus FAISS index."""

from __future__ import annotations

import logging
import os
import re
import socket
import tempfile
import time
from pathlib import Path

from ltr_system.data_download import sha256_file
from ltr_system.dense_encode import check_vectors
from ltr_system.evaluation import evaluate_files
from ltr_system.project import initialize, parser, read_json, write_json
from ltr_system.query_encode import validation_inputs

LOGGER = logging.getLogger(__name__)


def retrieve(index, vectors, ids: list[str], output: Path, k: int, batch_size: int) -> dict:
    import numpy as np

    if min(k, batch_size) <= 0 or k > index.ntotal:
        raise ValueError("Positive batch size and retrieval depth <= index size required")
    if len(ids) != len(vectors) or not ids or len(set(ids)) != len(ids):
        raise ValueError("Query ID alignment or uniqueness mismatch")
    check_vectors(vectors, (len(ids), index.d))
    search_seconds = 0.0
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for start in range(0, len(ids), batch_size):
            batch = np.ascontiguousarray(vectors[start : start + batch_size])
            began = time.perf_counter()
            scores, found = index.search(batch, k)
            search_seconds += time.perf_counter() - began
            if scores.shape != (len(batch), k) or found.shape != scores.shape:
                raise ValueError("Unexpected FAISS result shape")
            if (
                not np.isfinite(scores).all()
                or (found < 0).any()
                or (np.diff(scores, axis=1) > 0).any()
            ):
                raise ValueError("Invalid FAISS scores or passage IDs")
            for offset, hits in enumerate(found):
                if len(np.unique(hits)) != k:
                    raise ValueError("Duplicate retrieved passage ID")
                for rank, pid in enumerate(hits, 1):
                    stream.write(f"{ids[start + offset]}\t{int(pid)}\t{rank}\n")
            LOGGER.info(
                "Retrieved %s/%s validation queries", min(start + batch_size, len(ids)), len(ids)
            )
    return {
        "queries": len(ids),
        "returned_passages": len(ids) * k,
        "search_call_seconds": search_seconds,
    }


def run(root: Path, run_name: str, k: int, batch_size: int, threads: int) -> dict:
    import faiss
    import numpy as np

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run_name):
        raise ValueError("Run name must be a simple directory name")
    if min(k, batch_size, threads) <= 0:
        raise ValueError("Depth, batch size and threads must be positive")
    destination = root / "artifacts/runs" / run_name
    if destination.exists():
        raise FileExistsError(destination)
    index_root = root / "artifacts/indexes/e5-flat-ip-v1"
    query_root = root / "artifacts/queries/e5-validation-fp32"
    split_root = root / "data/processed/msmarco-passage-v1-splits"
    index_report = read_json(index_root / "manifest.json")
    query_report = read_json(query_root / "manifest.json")
    ids, _, source = validation_inputs(split_root)
    if (
        index_report.get("status") != "exact index built and reload checked"
        or index_report.get("index_type") != "IndexIDMap2(IndexFlatIP)"
        or index_report.get("metric") != "inner product"
        or query_report.get("status") != "validation queries encoded"
        or query_report["source"] != source
        or query_report["passage_configuration_sha256"]
        != index_report["embedding_configuration_sha256"]
        or query_report["encoder"]["dimension"] != index_report["dimension"]
        or query_report["queries"] != len(ids)
        or (
            query_report["prefix"],
            query_report["precision"],
            query_report["pooling"],
            query_report["normalization"],
        )
        != ("query: ", "float32", "masked mean", "L2")
    ):
        raise ValueError("Incompatible index, query encoding or split identity")
    split = read_json(split_root / "manifest.json")
    expected = {
        index_root / "exact.faiss": index_report["files"]["exact.faiss"]["sha256"],
        **{
            query_root / name: query_report["files"][name]["sha256"]
            for name in ("vectors.npy", "query_ids.json")
        },
        split_root / "validation.qrels.tsv": split["outputs"]["validation.qrels.tsv"]["sha256"],
        split_root / "validation.queries.tsv": source["query_file_sha256"],
        split_root / "manifest.json": source["split_manifest_sha256"],
    }
    for directory in (index_root, query_root):
        path = directory / "manifest.json"
        expected[path] = sha256_file(path)

    def verify():
        for path, digest in expected.items():
            if sha256_file(path) != digest:
                raise ValueError(f"Input checksum mismatch: {path.name}")

    LOGGER.info("Verifying exact index and query artifact checksums")
    verify()
    if read_json(query_root / "query_ids.json")["query_ids"] != ids:
        raise ValueError("Query vector ID order differs from frozen validation queries")
    vectors = np.load(query_root / "vectors.npy", allow_pickle=False)
    faiss.omp_set_num_threads(threads)
    LOGGER.info("Loading full-corpus exact index")
    index = faiss.read_index(str(index_root / "exact.faiss"))
    if (
        not isinstance(index, faiss.IndexIDMap2)
        or not isinstance(faiss.downcast_index(index.index), faiss.IndexFlatIP)
        or index.metric_type != faiss.METRIC_INNER_PRODUCT
        or index.ntotal != index_report["vectors"]
        or index.d != index_report["dimension"]
    ):
        raise ValueError("Loaded index type, metric or dimensions differ from manifest")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dense-run-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "run"
        stage.mkdir()
        started = time.perf_counter()
        counts = retrieve(index, vectors, ids, stage / "ranking.tsv", k, batch_size)
        loop_seconds = time.perf_counter() - started
        result = evaluate_files(
            split_root / "validation.queries.tsv",
            split_root / "validation.qrels.tsv",
            stage / "ranking.tsv",
            stage / "evaluation.json",
            run_name,
        )
        verify()
        write_json(
            stage / "manifest.json",
            {
                "schema_version": 1,
                "status": "validation retrieval and scoring completed",
                "run_name": run_name,
                "split": "validation",
                "final_queries_read": False,
                "index_manifest_sha256": expected[index_root / "manifest.json"],
                "query_manifest_sha256": expected[query_root / "manifest.json"],
                "source": source,
                "faiss": faiss.__version__,
                "k": k,
                "batch_size": batch_size,
                "threads": threads,
                "counts": counts,
                "retrieval_loop_seconds": loop_seconds,
                "timing_scope": "CPU search calls and ranking writes; excludes loading, checksums, encoding and scoring; not serving latency",
                "tie_policy": "preserve FAISS order; equal-score membership/order may vary with numerical kernels",
                "ranking_sha256": sha256_file(stage / "ranking.tsv"),
                "evaluation_sha256": sha256_file(stage / "evaluation.json"),
                "host": socket.gethostname(),
                "job_id": os.environ.get("SLURM_JOB_ID"),
            },
        )
        stage.rename(destination)
    LOGGER.info("Validation metrics: %s", result["metrics"])
    return result


def main() -> None:
    command = parser("Retrieve and evaluate validation queries with exact full-corpus E5 search.")
    command.add_argument("--run-name", default="e5-exact-validation-default")
    command.add_argument("--k", type=int, default=1000)
    command.add_argument("--batch-size", type=int, default=32)
    command.add_argument("--threads", type=int, default=4)
    args = command.parse_args()
    root = initialize(args)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run full-corpus search in a CPU Slurm allocation")
    if args.threads > int(os.environ.get("SLURM_CPUS_PER_TASK", "1")):
        raise ValueError("Threads exceed allocated CPUs")
    if args.k < 1000:
        raise ValueError("Retrieve at least 1000 passages for default recall cutoffs")
    run(root, args.run_name, args.k, args.batch_size, args.threads)
