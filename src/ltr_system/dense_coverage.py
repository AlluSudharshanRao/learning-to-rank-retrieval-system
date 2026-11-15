"""Audit complete embedding coverage and row-to-passage alignment without CUDA."""

from __future__ import annotations

import itertools
import json
import logging
import os
from pathlib import Path

from ltr_system.data_download import sha256_file
from ltr_system.dense_encode import verify_completed, writer_lock
from ltr_system.lexical_index import verify_corpus
from ltr_system.project import initialize, parser, read_json, write_json

LOGGER = logging.getLogger(__name__)


def audit(source: Path, embeddings: Path) -> dict:
    if not embeddings.is_dir():
        raise FileNotFoundError(embeddings)
    with writer_lock(embeddings):
        source_hash = sha256_file(source / "manifest.json")
        configuration_hash = sha256_file(embeddings / "configuration.json")
        corpus = verify_corpus(source)
        config = read_json(embeddings / "configuration.json")
        if config["source_corpus_manifest_sha256"] != source_hash:
            raise ValueError("Embedding configuration refers to another corpus")
        expected_names = {Path(entry["name"]).stem for entry in corpus["shards"]}
        actual_names = {p.name for p in embeddings.iterdir() if p.name.startswith("part-")}
        if actual_names != expected_names:
            raise ValueError("Missing or unexpected embedding shards")
        reports = []
        total = 0
        vector_bytes = 0
        for entry in corpus["shards"]:
            name = Path(entry["name"]).stem
            directory = embeddings / name
            identity = {
                **config,
                "source_shard": entry["name"],
                "source_shard_sha256": entry["sha256"],
            }
            report = verify_completed(directory, identity, entry["records"], config["dimension"])
            if report["records"] != entry["records"]:
                raise ValueError("Encoded manifest record count differs from source")
            ids = read_json(directory / "passage_ids.json")["passage_ids"]
            with (source / "shards" / entry["name"]).open(encoding="utf-8") as stream:
                for position, (line, pid) in enumerate(itertools.zip_longest(stream, ids)):
                    if line is None or pid is None or json.loads(line)["id"] != pid:
                        raise ValueError(f"Passage ID alignment mismatch: {name}, row {position}")
            total += len(ids)
            vector_bytes += report["files"]["vectors.npy"]["bytes"]
            reports.append(
                {
                    "name": name,
                    "records": len(ids),
                    "manifest_sha256": sha256_file(directory / "manifest.json"),
                }
            )
            LOGGER.info("Verified shard %s (%s passages)", name, len(ids))
        if total != corpus["records"]:
            raise ValueError("Total embedding coverage differs from corpus")
        if sha256_file(source / "manifest.json") != source_hash:
            raise ValueError("Source manifest changed during audit")
        if sha256_file(embeddings / "configuration.json") != configuration_hash:
            raise ValueError("Embedding configuration changed during audit")
        result = {
            "schema_version": 1,
            "status": "all embedding shards verified",
            "records": total,
            "shard_count": len(reports),
            "dimension": config["dimension"],
            "vector_file_bytes_including_headers": vector_bytes,
            "source_manifest_sha256": source_hash,
            "configuration_sha256": configuration_hash,
            "ordered_source_id_alignment": True,
            "cross_shard_uniqueness": "inherited from validated corpus and exact source-row alignment",
            "shards": reports,
        }
        write_json(embeddings / "coverage.json", result)
    return result


def main() -> None:
    args = parser(
        "Verify complete corpus embedding coverage, checksums and source IDs."
    ).parse_args()
    root = initialize(args)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run full-corpus verification in a CPU Slurm allocation")
    result = audit(
        root / "data/processed/msmarco-passage-v1-jsonl",
        root / "artifacts/embeddings/e5-small-v2-fp32",
    )
    LOGGER.info("Coverage PASS: %s shards, %s passages", result["shard_count"], result["records"])


if __name__ == "__main__":
    main()
