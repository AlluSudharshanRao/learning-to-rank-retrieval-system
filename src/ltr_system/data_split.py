"""Deterministic supervised query splits with normalized-text exclusion."""

from __future__ import annotations

import hashlib
import logging
import tempfile
import unicodedata
from pathlib import Path

from ltr_system.data_download import sha256_file
from ltr_system.data_prepare import rows
from ltr_system.project import initialize, parser, read_json, write_json

SEED = "ltr-v1-split-2026"
INPUTS = ("queries.train.tsv", "queries.dev.small.tsv", "qrels.train.tsv")
LOGGER = logging.getLogger(__name__)


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def assign(queries: dict, judged: set, final: dict, validation_groups: int, train_queries: int):
    if validation_groups <= 0 or train_queries <= 0:
        raise ValueError("Split sizes must be positive")
    final_text = {normalize(text) for text in final.values()}
    groups = {}
    excluded = 0
    for qid, text in queries.items():
        if qid not in judged:
            continue
        normalized = normalize(text)
        if not normalized:
            raise ValueError(f"Query {qid} normalizes to empty text")
        if normalized in final_text or qid in final:
            excluded += 1
            continue
        groups.setdefault(normalized, []).append(qid)
    ordered = sorted(
        groups,
        key=lambda text: (hashlib.sha256(f"{SEED}\n{text}".encode()).hexdigest(), text),
    )
    if len(ordered) <= validation_groups:
        raise ValueError("Insufficient eligible validation and training groups")
    validation = sorted(qid for text in ordered[:validation_groups] for qid in groups[text])
    training = []
    for text in ordered[validation_groups:]:
        training.extend(sorted(groups[text]))
        if len(training) >= train_queries:
            break
    if len(training) < train_queries:
        raise ValueError("Insufficient eligible training queries")
    # Check independence explicitly; the corpus itself remains shared across all splits.
    train_text = {normalize(queries[qid]) for qid in training}
    val_text = {normalize(queries[qid]) for qid in validation}
    if train_text & val_text or train_text & final_text or val_text & final_text:
        raise ValueError("Normalized query overlap between splits")
    return {"train": training, "validation": validation}, {
        "eligible_groups": len(groups),
        "excluded_judged_queries_matching_final_id_or_text": excluded,
        "unjudged_training_queries": len(set(queries) - judged),
        "validation_groups": validation_groups,
        "requested_train_queries": train_queries,
        "actual_train_queries": len(training),
        "actual_validation_queries": len(validation),
        "normalized_text_overlap_between_splits": 0,
        "query_id_overlap_between_splits": 0,
    }


def split(source: Path, destination: Path, validation_groups: int, train_queries: int) -> None:
    if destination.exists():
        raise FileExistsError(f"Output exists; inspect before rerunning: {destination}")
    integrity = read_json(source / "integrity.json")
    if integrity.get("status") != "records validated; splits not generated":
        raise ValueError("Expected validated record integrity report")
    for name in INPUTS:
        if sha256_file(source / name) != integrity["files"][name]["sha256"]:
            raise ValueError(f"Input hash mismatch: {name}")
    queries = dict(parts for _, parts in rows(source / "queries.train.tsv", 2))
    final = dict(parts for _, parts in rows(source / "queries.dev.small.tsv", 2))
    judged = {parts[0] for _, parts in rows(source / "qrels.train.tsv", 4)}
    if judged - queries.keys():
        raise ValueError("Judged queries missing from training query file")
    assignments, counts = assign(queries, judged, final, validation_groups, train_queries)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".split-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "splits"
        stage.mkdir()
        roles = {}
        for role, ids in assignments.items():
            with (stage / f"{role}.queries.tsv").open("w", encoding="utf-8", newline="\n") as out:
                for qid in ids:
                    roles[qid] = role
                    out.write(f"{qid}\t{queries[qid]}\n")
        unique_counts = {}
        for role in assignments:
            judgments = set()
            for _, (qid, iteration, pid, label) in rows(source / "qrels.train.tsv", 4):
                if roles.get(qid) == role:
                    judgments.add((qid, pid, label))
            with (stage / f"{role}.qrels.tsv").open("w", encoding="utf-8", newline="\n") as out:
                for qid, pid, label in sorted(judgments):
                    out.write(f"{qid}\t0\t{pid}\t{label}\n")
            unique_counts[role] = len(judgments)
        write_json(stage / "final.query_ids.json", {"query_ids": sorted(final)})
        outputs = {
            path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(stage.iterdir())
        }
        write_json(
            stage / "manifest.json",
            {
                "schema_version": 1,
                "split_version": 1,
                "seed": SEED,
                "normalization": "NFKC, casefold, whitespace collapse",
                "unicode_version": unicodedata.unidata_version,
                "source_integrity_sha256": sha256_file(source / "integrity.json"),
                "source_archive_sha256": integrity["source_sha256"],
                "counts": counts,
                "unique_judgment_counts": unique_counts,
                "final_queries": len(final),
                "final_judgments_read": False,
                "supplementary_trec_overlap_checked": False,
                "outputs": outputs,
            },
        )
        stage.rename(destination)
    LOGGER.info("Saved query splits: %s", destination)
    LOGGER.info("Split counts: %s", counts)


def main() -> None:
    command = parser("Generate supervised training and validation query splits.")
    command.add_argument("--validation-groups", type=int, default=2000)
    command.add_argument("--train-queries", type=int, default=50000)
    args = command.parse_args()
    root = initialize(args)
    split(
        root / "data/processed/msmarco-passage-v1",
        root / "data/processed/msmarco-passage-v1-splits",
        args.validation_groups,
        args.train_queries,
    )


if __name__ == "__main__":
    main()
