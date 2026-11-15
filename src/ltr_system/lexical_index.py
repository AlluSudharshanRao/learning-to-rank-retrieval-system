"""Verified full-corpus Lucene indexing with process-scoped Java file ownership."""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

from ltr_system.data_download import sha256_file
from ltr_system.project import initialize, parser, read_json, write_json

LOGGER = logging.getLogger(__name__)


def verify_corpus(source: Path) -> dict:
    manifest = read_json(source / "manifest.json")
    if manifest.get("status") != "corpus converted; Lucene index not built":
        raise ValueError("Expected converted corpus manifest")
    entries = manifest["shards"]
    names = [entry["name"] for entry in entries]
    if not names or len(set(names)) != len(names):
        raise ValueError("Missing or duplicate shard names")
    if any(Path(name).name != name or not name.endswith(".jsonl") for name in names):
        raise ValueError("Invalid shard filename")
    if set(names) != {p.name for p in (source / "shards").iterdir()}:
        raise ValueError("Shard directory differs from manifest")
    if any(entry["records"] <= 0 for entry in entries):
        raise ValueError("Empty shard in corpus manifest")
    if sum(entry["records"] for entry in entries) != manifest["records"]:
        raise ValueError("Shard counts differ from corpus total")
    for entry in entries:
        path = source / "shards" / entry["name"]
        if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"Shard identity mismatch: {entry['name']}")
    return manifest


def build(root: Path, threads: int, heap_gb: int) -> None:
    if threads <= 0 or heap_gb <= 0:
        raise ValueError("Threads and heap size must be positive")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run full indexing in a Slurm allocation")
    if threads > int(os.environ.get("SLURM_CPUS_PER_TASK", "1")):
        raise ValueError("Index threads exceed allocated CPUs")
    if version("pyserini") != "2.4.0":
        raise ValueError("Expected Pyserini 2.4.0")
    source = root / "data/processed/msmarco-passage-v1-jsonl"
    destination = root / "artifacts/indexes/bm25-v1"
    if destination.exists():
        raise FileExistsError(f"Index output exists: {destination}")
    manifest_hash = sha256_file(source / "manifest.json")
    LOGGER.info("Verifying corpus shards")
    corpus = verify_corpus(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Retain failed stages for inspection; never delete live Java index files.
    stage = Path(tempfile.mkdtemp(prefix=".bm25-build-", dir=destination.parent))
    LOGGER.info("Build stage (retained on failure): %s", stage)
    index = stage / "index"
    environment = os.environ.copy()
    environment["JAVA_TOOL_OPTIONS"] = f"-Xmx{heap_gb}g"
    command = [
        sys.executable,
        "-I",
        "-u",
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        str(source / "shards"),
        "--index",
        str(index),
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        str(threads),
        "--language",
        "en",
        "--stemmer",
        "porter",
    ]
    write_json(
        stage / "build-config.json",
        {
            "command": command,
            "threads": threads,
            "java_heap_gb": heap_gb,
            "source_manifest_sha256": manifest_hash,
            "pyserini": version("pyserini"),
            "job_id": os.environ["SLURM_JOB_ID"],
            "host": socket.gethostname(),
        },
    )
    started = time.perf_counter()
    LOGGER.info("Starting Lucene indexing; detailed log: %s", stage / "index.log")
    with (stage / "index.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=3 * 60 * 60,
        )
    index_seconds = time.perf_counter() - started
    # Verify in a child JVM; wait for its exit before renaming the index directory.
    verification = (
        "import json,sys; from pyserini.search.lucene import LuceneSearcher; "
        "from jnius import autoclass; s=LuceneSearcher(sys.argv[1]); "
        "n=s.num_docs; s.close(); "
        "json.dump({'documents':n,'java_version':str(autoclass('java.lang.System')"
        ".getProperty('java.version'))},open(sys.argv[2],'w'))"
    )
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-u",
            "-c",
            verification,
            str(index),
            str(stage / "verification.json"),
        ],
        env=environment,
        check=True,
        timeout=300,
    )
    observed = read_json(stage / "verification.json")
    if observed["documents"] != corpus["records"]:
        raise ValueError(f"Indexed count mismatch: {observed['documents']} vs {corpus['records']}")
    # Source data is immutable during the build; detect accidental changes before publication.
    verify_corpus(source)
    if sha256_file(source / "manifest.json") != manifest_hash:
        raise ValueError("Corpus manifest changed during indexing")
    files = {
        p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
        for p in sorted(index.iterdir())
        if p.is_file() and p.name != "write.lock"
    }
    write_json(
        stage / "manifest.json",
        {
            "schema_version": 1,
            "builder_version": 1,
            "status": "index built and count verified",
            "documents": observed["documents"],
            "java_version": observed["java_version"],
            "source_manifest_sha256": manifest_hash,
            "source_collection_sha256": corpus["source_collection_sha256"],
            "pyserini": version("pyserini"),
            "analyzer": {"language": "en", "stemmer": "porter", "keep_stopwords": False},
            "store_positions": False,
            "store_docvectors": False,
            "store_raw": False,
            "threads": threads,
            "java_heap_gb": heap_gb,
            "index_seconds": index_seconds,
            "files": files,
            "job_id": os.environ["SLURM_JOB_ID"],
            "host": socket.gethostname(),
        },
    )
    stage.rename(destination)
    LOGGER.info("Index built and count verified: %s documents; %s", corpus["records"], destination)


def main() -> None:
    command = parser("Build and verify a Lucene index over validated JSONL corpus shards.")
    command.add_argument("--threads", type=int, default=4)
    command.add_argument("--java-heap-gb", type=int, default=16)
    args = command.parse_args()
    build(initialize(args), args.threads, args.java_heap_gb)


if __name__ == "__main__":
    main()
