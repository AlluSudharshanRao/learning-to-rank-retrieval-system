"""Safely extract and validate passage-v1 records using a disk-backed ID registry."""

from __future__ import annotations

import logging
import sqlite3
import tarfile
import tempfile
from pathlib import Path

from ltr_system.data_download import ARCHIVE_NAME, REQUIRED, inventory, sha256_file
from ltr_system.project import initialize, parser, read_json, write_json

LOGGER = logging.getLogger(__name__)
VERSION = 2
EXPECTED_PASSAGES = 8_841_823


def rows(path: Path, fields: int):
    with path.open(encoding="utf-8", errors="strict", newline="") as stream:
        for number, line in enumerate(stream, 1):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != fields or any(not value.strip() for value in parts):
                raise ValueError(f"{path.name}:{number}: expected {fields} nonempty fields")
            yield number, parts


def identifier(value: str) -> None:
    if not value.isascii() or not value.isdecimal():
        raise ValueError(f"Invalid numeric source identifier: {value!r}")


def validate(directory: Path, database: Path) -> dict:
    """Store IDs only; preserve original TSV text and string identifiers."""
    counts = {}
    judgment_counts = {}
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA cache_size=-32768")
        connection.execute("CREATE TABLE passages (id TEXT PRIMARY KEY) WITHOUT ROWID")
        connection.execute(
            "CREATE TABLE queries (role TEXT, id TEXT, PRIMARY KEY(role,id)) WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE judgments (role TEXT, qid TEXT, pid TEXT, label TEXT NOT NULL, "
            "PRIMARY KEY(role,qid,pid)) WITHOUT ROWID"
        )
        for filename, role in (
            ("collection.tsv", "passages"),
            ("queries.train.tsv", "train"),
            ("queries.dev.small.tsv", "final"),
        ):
            count = 0
            for number, (source_id, text) in rows(directory / filename, 2):
                identifier(source_id)
                try:
                    if role == "passages":
                        connection.execute("INSERT INTO passages VALUES (?)", (source_id,))
                    else:
                        connection.execute("INSERT INTO queries VALUES (?,?)", (role, source_id))
                except sqlite3.IntegrityError as error:
                    raise ValueError(f"{filename}:{number}: duplicate ID {source_id}") from error
                count += 1
                if count % 100_000 == 0:
                    connection.commit()
                if count % 1_000_000 == 0:
                    LOGGER.info("Validated %s: %s records", filename, count)
            if not count:
                raise ValueError(f"Empty source file: {filename}")
            connection.commit()
            counts[filename] = count
            LOGGER.info("Validated %s: %s records", filename, count)
        for filename, role in (("qrels.train.tsv", "train"), ("qrels.dev.small.tsv", "final")):
            count = 0
            duplicates = 0
            for number, (qid, iteration, pid, label) in rows(directory / filename, 4):
                identifier(qid)
                identifier(pid)
                if iteration != "0":
                    raise ValueError(f"{filename}:{number}: expected iteration 0")
                previous = connection.execute(
                    "SELECT label FROM judgments WHERE role=? AND qid=? AND pid=?",
                    (role, qid, pid),
                ).fetchone()
                if previous is not None and previous[0] != label:
                    raise ValueError(
                        f"{filename}:{number}: conflicting judgment for query {qid}, "
                        f"passage {pid}: {previous[0]} versus {label}"
                    )
                if label != "1":
                    raise ValueError(f"{filename}:{number}: expected passage-v1 binary judgment")
                if previous is None:
                    connection.execute(
                        "INSERT INTO judgments VALUES (?,?,?,?)", (role, qid, pid, label)
                    )
                else:
                    duplicates += 1
                count += 1
            if not count:
                raise ValueError(f"Empty source file: {filename}")
            connection.commit()
            counts[filename] = count
            judgment_counts[filename] = {
                "source_rows": count,
                "unique_judgments": count - duplicates,
                "identical_duplicate_rows": duplicates,
                "conflicting_judgments": 0,
            }
            LOGGER.info(
                "Validated %s: %s rows, %s identical duplicates", filename, count, duplicates
            )
        missing = connection.execute(
            "SELECT COUNT(*) FROM judgments j "
            "LEFT JOIN passages p ON p.id=j.pid "
            "LEFT JOIN queries q ON q.role=j.role AND q.id=j.qid "
            "WHERE p.id IS NULL OR q.id IS NULL"
        ).fetchone()[0]
        if missing:
            raise ValueError(f"Judgments with missing query or passage references: {missing}")
        overlap = connection.execute(
            "SELECT COUNT(*) FROM queries a JOIN queries b ON a.id=b.id "
            "WHERE a.role='train' AND b.role='final'"
        ).fetchone()[0]
        return {
            "record_counts": counts,
            "duplicate_ids": 0,
            "duplicate_judgments": sum(
                item["identical_duplicate_rows"] for item in judgment_counts.values()
            ),
            "judgment_counts": judgment_counts,
            "missing_judgment_references": 0,
            "train_final_query_id_overlap": overlap,
            "normalized_query_text_overlap": "pending split generation",
        }
    finally:
        connection.close()


def prepare(raw: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Output already exists; inspect before rerunning: {destination}")
    archive = raw / ARCHIVE_NAME
    source = read_json(raw / "source.json")
    if source.get("dataset") != "msmarco-passage-v1":
        raise ValueError("Unexpected dataset identity")
    digest = sha256_file(archive)
    if archive.stat().st_size != source["bytes"] or digest != source["sha256"]:
        raise ValueError("Archive identity differs from the source manifest")
    members = inventory(archive)
    if members != source["members"]:
        raise ValueError("Archive inventory differs from the source manifest")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".prepare-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "records"
        stage.mkdir()
        with tarfile.open(archive, "r|gz") as bundle:
            for member in bundle:
                name = Path(member.name).name
                if name in REQUIRED:
                    LOGGER.info("Extracting %s", name)
                    with bundle.extractfile(member) as stream, (stage / name).open("xb") as output:
                        while chunk := stream.read(1024 * 1024):
                            output.write(chunk)
        report = validate(stage, Path(temporary) / "ids.sqlite")
        if report["record_counts"]["collection.tsv"] != EXPECTED_PASSAGES:
            raise ValueError("Unexpected official passage-v1 corpus record count")
        report.update(
            schema_version=1,
            parser_version=VERSION,
            source_sha256=digest,
            status="records validated; splits not generated",
            files={
                name: {"sha256": sha256_file(stage / name), "bytes": (stage / name).stat().st_size}
                for name in sorted(REQUIRED)
            },
        )
        write_json(stage / "integrity.json", report)
        stage.rename(destination)
    LOGGER.info("Validated records saved: %s", destination)


def main() -> None:
    args = parser("Extract and validate MS MARCO Passage v1 records.").parse_args()
    root = initialize(args)
    prepare(
        root / "data/raw/msmarco-passage-v1",
        root / "data/processed/msmarco-passage-v1",
    )


if __name__ == "__main__":
    main()
