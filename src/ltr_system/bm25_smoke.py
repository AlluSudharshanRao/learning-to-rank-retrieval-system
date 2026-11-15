"""Synthetic Lucene indexing and retrieval check; not a relevance benchmark."""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

from ltr_system.project import initialize, parser, write_json

LOGGER = logging.getLogger(__name__)
DOCUMENTS = (
    {"id": "101", "contents": "quasar telescope astronomy"},
    {"id": "102", "contents": "bread flour kitchen"},
    {"id": "103", "contents": "database transaction storage"},
)


def main() -> None:
    args = parser("Build and search a three-document synthetic BM25 index.").parse_args()
    root = initialize(args)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run this Java integration check inside a Slurm allocation")
    if version("pyserini") != "2.4.0":
        raise RuntimeError("This check targets pyserini 2.4.0")
    if not Path(os.environ.get("JVM_PATH", "__missing__")).is_file():
        raise RuntimeError("Set JVM_PATH to the BM25 environment's libjvm.so")
    parent = root / "artifacts/smoke"
    parent.mkdir(parents=True, exist_ok=True)
    # Retain the tiny index for inspection; Java file release can outlive searcher.close().
    stage = Path(tempfile.mkdtemp(prefix="bm25-", dir=parent))
    LOGGER.info("Retaining smoke artifacts: %s", stage)
    collection = stage / "collection"
    collection.mkdir()
    with (collection / "documents.jsonl").open("w", encoding="utf-8") as stream:
        for document in DOCUMENTS:
            stream.write(json.dumps(document) + "\n")
    index = stage / "index"
    command = [
        sys.executable,
        "-I",
        "-u",
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        str(collection),
        "--index",
        str(index),
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        "1",
    ]
    LOGGER.info("Building synthetic index; subprocess timeout is 240 seconds")
    subprocess.run(command, check=True, timeout=240)
    LOGGER.info("Loading LuceneSearcher")
    from pyserini.search.lucene import LuceneSearcher

    results = {}
    for attempt in range(2):
        searcher = LuceneSearcher(str(index))
        try:
            searcher.set_bm25(k1=0.9, b=0.4)
            if searcher.num_docs != len(DOCUMENTS):
                raise ValueError(f"Unexpected indexed document count: {searcher.num_docs}")
            observed = {}
            for query, expected in (("quasar", "101"), ("bread", "102")):
                hits = searcher.search(query, k=3)
                if len(hits) != 1 or hits[0].docid != expected:
                    raise ValueError(f"Unexpected result IDs for synthetic query {query}")
                score = float(hits[0].score)
                if not math.isfinite(score) or score <= 0:
                    raise ValueError("Expected finite positive synthetic BM25 score")
                observed[query] = {"passage_id": hits[0].docid, "score": score}
            if searcher.search("zzzzunmatchedtoken", k=3):
                raise ValueError("Unexpected result for absent synthetic term")
            if attempt and observed != results:
                raise ValueError("Search results changed after reopening the index")
            results = observed
        finally:
            searcher.close()
    write_json(
        stage / "check.json",
        {
            "status": "PASS",
            "job_id": os.environ["SLURM_JOB_ID"],
            "artifact_directory": str(stage),
            "scope": "synthetic correctness only",
            "pyserini": version("pyserini"),
            "documents": list(DOCUMENTS),
            "bm25": {"k1": 0.9, "b": 0.4},
            "results": results,
            "reopen_agreement": True,
        },
    )
    LOGGER.info("BM25 indexing/search/reopen check: PASS; report: %s", stage / "check.json")


if __name__ == "__main__":
    main()
