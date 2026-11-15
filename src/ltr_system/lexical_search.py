"""Validation-only BM25 retrieval with verified input identities and binary scoring."""

from __future__ import annotations

import logging
import math
import os
import re
import socket
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

from ltr_system.data_download import sha256_file
from ltr_system.data_prepare import rows
from ltr_system.evaluation import evaluate_files
from ltr_system.project import initialize, parser, read_json, write_json

LOGGER = logging.getLogger(__name__)


def retrieve(searcher, queries: Path, output: Path, k: int) -> dict:
    if type(k) is not int or k <= 0:
        raise ValueError("Retrieval depth must be a positive integer")
    seen_queries = set()
    empty = 0
    total = 0
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for _, (qid, text) in rows(queries, 2):
            if qid in seen_queries:
                raise ValueError(f"Duplicate query ID: {qid}")
            seen_queries.add(qid)
            hits = searcher.search(text, k=k)
            if len(hits) > k:
                raise ValueError("Backend returned more hits than requested")
            seen = set()
            previous = math.inf
            for rank, hit in enumerate(hits, 1):
                pid = hit.docid
                score = float(hit.score)
                if not isinstance(pid, str) or not pid or any(c.isspace() for c in pid):
                    raise ValueError("Invalid external passage ID")
                if pid in seen or not math.isfinite(score) or score > previous:
                    raise ValueError("Duplicate passage or invalid score ordering")
                seen.add(pid)
                previous = score
                stream.write(f"{qid}\t{pid}\t{rank}\n")
            empty += len(hits) == 0
            total += len(hits)
            if len(seen_queries) % 100 == 0:
                LOGGER.info("Retrieved %s validation queries", len(seen_queries))
    if not seen_queries:
        raise ValueError("Empty validation query file")
    return {"queries": len(seen_queries), "returned_passages": total, "empty_queries": empty}


def main() -> None:
    command = parser("Retrieve and evaluate validation queries with full-corpus BM25.")
    command.add_argument("--run-name", default="bm25-validation-default")
    command.add_argument("--k", type=int, default=1000)
    command.add_argument("--k1", type=float, default=0.9)
    command.add_argument("--b", type=float, default=0.4)
    args = command.parse_args()
    root = initialize(args)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", args.run_name):
        raise ValueError("Run name must be a simple directory name")
    if args.k < 1000:
        raise ValueError("Retrieve at least 1000 candidates for the default recall cutoffs")
    if not math.isfinite(args.k1) or args.k1 <= 0 or not 0 <= args.b <= 1:
        raise ValueError("Expected finite k1 > 0 and 0 <= b <= 1")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run full-corpus retrieval in a Slurm allocation")
    if version("pyserini") != "2.4.0":
        raise ValueError("Expected Pyserini 2.4.0")
    index_root = root / "artifacts/indexes/bm25-v1"
    split_root = root / "data/processed/msmarco-passage-v1-splits"
    destination = root / "artifacts/runs" / args.run_name
    if destination.exists():
        raise FileExistsError(f"Run exists; choose a different run name: {destination}")
    index_hash = sha256_file(index_root / "manifest.json")
    split_hash = sha256_file(split_root / "manifest.json")
    index_manifest = read_json(index_root / "manifest.json")
    split_manifest = read_json(split_root / "manifest.json")
    if index_manifest.get("status") != "index built and count verified":
        raise ValueError("Expected verified index manifest")
    expected_files = set(index_manifest["files"])
    actual_files = {p.name for p in (index_root / "index").iterdir() if p.name != "write.lock"}
    if actual_files != expected_files:
        raise ValueError("Index directory differs from its manifest")
    LOGGER.info("Verifying index checksums")
    for name, identity in index_manifest["files"].items():
        path = index_root / "index" / name
        if Path(name).name != name or sha256_file(path) != identity["sha256"]:
            raise ValueError("Index file identity mismatch")
    for name in ("validation.queries.tsv", "validation.qrels.tsv"):
        if sha256_file(split_root / name) != split_manifest["outputs"][name]["sha256"]:
            raise ValueError(f"Validation split identity mismatch: {name}")
    LOGGER.info("Loading LuceneSearcher")
    from pyserini.search.lucene import LuceneSearcher

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".retrieval-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "run"
        stage.mkdir()
        searcher = LuceneSearcher(str(index_root / "index"))
        try:
            if searcher.num_docs != index_manifest["documents"]:
                raise ValueError("Index document count mismatch")
            searcher.set_bm25(k1=args.k1, b=args.b)
            started = time.perf_counter()
            counts = retrieve(
                searcher, split_root / "validation.queries.tsv", stage / "ranking.tsv", args.k
            )
            retrieval_seconds = time.perf_counter() - started
        finally:
            searcher.close()
        if counts["queries"] != split_manifest["counts"]["actual_validation_queries"]:
            raise ValueError("Validation query count mismatch")
        if sha256_file(index_root / "manifest.json") != index_hash:
            raise ValueError("Index manifest changed")
        if sha256_file(split_root / "manifest.json") != split_hash:
            raise ValueError("Split manifest changed")
        for name in ("validation.queries.tsv", "validation.qrels.tsv"):
            if sha256_file(split_root / name) != split_manifest["outputs"][name]["sha256"]:
                raise ValueError(f"Validation input changed during retrieval: {name}")
        result = evaluate_files(
            split_root / "validation.queries.tsv",
            split_root / "validation.qrels.tsv",
            stage / "ranking.tsv",
            stage / "evaluation.json",
            args.run_name,
        )
        write_json(
            stage / "manifest.json",
            {
                "schema_version": 1,
                "status": "validation retrieval and scoring completed",
                "run_name": args.run_name,
                "split": "validation",
                "final_queries_read": False,
                "index_manifest_sha256": index_hash,
                "split_manifest_sha256": split_hash,
                "pyserini": version("pyserini"),
                "bm25": {"k1": args.k1, "b": args.b},
                "k": args.k,
                "counts": counts,
                "retrieval_loop_seconds": retrieval_seconds,
                "timing_scope": "sequential search, Python processing and ranking TSV writes; no load test",
                "tie_policy": "preserve Lucene returned order; tied membership may depend on index build",
                "ranking_sha256": sha256_file(stage / "ranking.tsv"),
                "evaluation_sha256": sha256_file(stage / "evaluation.json"),
                "job_id": os.environ["SLURM_JOB_ID"],
                "host": socket.gethostname(),
            },
        )
        stage.rename(destination)
    LOGGER.info("Validation metrics: %s", result["metrics"])
    LOGGER.info("Run saved: %s", destination)


if __name__ == "__main__":
    main()
