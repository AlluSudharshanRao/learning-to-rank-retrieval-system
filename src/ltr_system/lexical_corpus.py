"""Stream validated passage TSV into deterministic Lucene JSONL input shards."""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

from ltr_system.data_download import sha256_file
from ltr_system.data_prepare import identifier, rows
from ltr_system.project import initialize, parser, read_json, write_json

LOGGER = logging.getLogger(__name__)


def convert(source: Path, destination: Path, shard_size: int = 100_000) -> dict:
    if type(shard_size) is not int or shard_size <= 0:
        raise ValueError("Shard size must be a positive integer")
    if destination.exists():
        raise FileExistsError(f"Output exists; inspect before rerunning: {destination}")
    manifest_path = source / "integrity.json"
    integrity_hash = sha256_file(manifest_path)
    integrity = read_json(manifest_path)
    if integrity.get("status") != "records validated; splits not generated":
        raise ValueError("Expected a validated source integrity report")
    collection = source / "collection.tsv"
    identity = integrity["files"]["collection.tsv"]
    source_hash = sha256_file(collection)
    if source_hash != identity["sha256"] or collection.stat().st_size != identity["bytes"]:
        raise ValueError("Collection identity differs from integrity report")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".lexical-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "corpus"
        # Keep the manifest outside the JSONL input directory passed to Lucene.
        shards = stage / "shards"
        shards.mkdir(parents=True)
        count = 0
        output = None
        records = []
        current = None
        try:
            for _, (pid, passage) in rows(collection, 2):
                identifier(pid)
                if count % shard_size == 0:
                    if output is not None:
                        output.close()
                    path = shards / f"part-{len(records):05d}.jsonl"
                    output = path.open("w", encoding="utf-8", newline="\n")
                    current = {"name": path.name, "records": 0, "first_passage_id": pid}
                    records.append(current)
                    LOGGER.info("Writing %s", path.name)
                output.write(
                    json.dumps({"id": pid, "contents": passage}, ensure_ascii=False) + "\n"
                )
                current["records"] += 1
                current["last_passage_id"] = pid
                count += 1
        finally:
            if output is not None:
                output.close()
        if not count or count != integrity["record_counts"]["collection.tsv"]:
            raise ValueError("Converted count differs from validated corpus count")
        if sha256_file(collection) != source_hash or sha256_file(manifest_path) != integrity_hash:
            raise ValueError("Source changed during conversion")
        for record in records:
            path = shards / record["name"]
            record.update(bytes=path.stat().st_size, sha256=sha256_file(path))
        report = {
            "schema_version": 1,
            "converter_version": 1,
            "source_integrity_sha256": integrity_hash,
            "source_collection_sha256": source_hash,
            "source_archive_sha256": integrity["source_sha256"],
            "records": count,
            "shard_size": shard_size,
            "shards": records,
            "format": "UTF-8 JSONL: id, contents; source order preserved",
            "status": "corpus converted; Lucene index not built",
        }
        write_json(stage / "manifest.json", report)
        stage.rename(destination)
    LOGGER.info("Converted %s passages into %s shards: %s", count, len(records), destination)
    return report


def main() -> None:
    command = parser("Convert validated passages to sharded Lucene JSONL input.")
    command.add_argument("--shard-size", type=int, default=100_000)
    args = command.parse_args()
    root = initialize(args)
    convert(
        root / "data/processed/msmarco-passage-v1",
        root / "data/processed/msmarco-passage-v1-jsonl",
        args.shard_size,
    )


if __name__ == "__main__":
    main()
