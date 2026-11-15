"""Strict TSV adapters and provenance for binary relevance evaluation."""

from __future__ import annotations

import logging
from pathlib import Path

from ltr_system.data_download import sha256_file
from ltr_system.data_prepare import rows
from ltr_system.metrics import evaluate
from ltr_system.project import initialize, parser, write_json


def load_rankings(path: Path) -> dict[str, list[str]]:
    """Read headerless query_id, passage_id, rank TSV; ranks are one-based."""
    grouped = {}
    for number, (qid, pid, raw_rank) in rows(path, 3):
        if not raw_rank.isascii() or not raw_rank.isdecimal() or int(raw_rank) <= 0:
            raise ValueError(f"{path.name}:{number}: expected positive integer rank")
        rank = int(raw_rank)
        entries = grouped.setdefault(qid, {})
        if rank in entries:
            raise ValueError(f"{path.name}:{number}: duplicate rank for query {qid}")
        entries[rank] = pid
    result = {}
    for qid, entries in grouped.items():
        if sorted(entries) != list(range(1, len(entries) + 1)):
            raise ValueError(f"Noncontiguous ranks for query {qid}")
        result[qid] = [entries[rank] for rank in range(1, len(entries) + 1)]
    return result


def load_positives(path: Path) -> tuple[dict[str, set[str]], int]:
    labels = {}
    duplicates = 0
    for number, (qid, iteration, pid, label) in rows(path, 4):
        if iteration != "0" or label not in {"0", "1"}:
            raise ValueError(f"{path.name}:{number}: expected binary qrel with iteration 0")
        key = (qid, pid)
        if key in labels:
            if labels[key] != label:
                raise ValueError(f"{path.name}:{number}: conflicting judgment")
            duplicates += 1
        labels[key] = label
    positives = {}
    for (qid, pid), label in labels.items():
        positives.setdefault(qid, set())
        if label == "1":
            positives[qid].add(pid)
    return positives, duplicates


def evaluate_files(
    queries: Path,
    qrels: Path,
    run: Path,
    output: Path,
    run_id: str,
    recall_cutoffs: tuple[int, ...] = (20, 50, 100, 1000),
    rank_cutoff: int = 10,
) -> dict:
    if output.exists():
        raise FileExistsError(f"Output exists; choose a new report path: {output}")
    if not run_id.strip():
        raise ValueError("A nonempty run ID is required")
    inputs = {"queries": queries, "qrels": qrels, "run": run}
    identities = {
        name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for name, path in inputs.items()
    }
    query_ids = [parts[0] for _, parts in rows(queries, 2)]
    positives, duplicates = load_positives(qrels)
    result = evaluate(query_ids, load_rankings(run), positives, recall_cutoffs, rank_cutoff)
    # Detect accidental input changes while reading; concurrent writers are unsupported.
    for name, path in inputs.items():
        if sha256_file(path) != identities[name]["sha256"]:
            raise ValueError(f"Input changed during evaluation: {name}")
    result.update(
        schema_version=1,
        evaluator_version=1,
        run_id=run_id,
        inputs=identities,
        identical_duplicate_qrel_rows=duplicates,
        recall_cutoffs=list(recall_cutoffs),
        rank_cutoff=rank_cutoff,
        ranking_format="headerless TSV: query_id, passage_id, one-based rank",
    )
    write_json(output, result)
    return result


def main() -> None:
    command = parser("Evaluate ranked TSV results against binary judgments.")
    command.add_argument("--queries", type=Path, required=True)
    command.add_argument("--qrels", type=Path, required=True)
    command.add_argument("--run", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--run-id", required=True)
    command.add_argument("--recall-cutoffs", type=int, nargs="+", default=[20, 50, 100, 1000])
    command.add_argument("--rank-cutoff", type=int, default=10)
    args = command.parse_args()
    root = initialize(args)

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else root / path

    result = evaluate_files(
        resolve(args.queries),
        resolve(args.qrels),
        resolve(args.run),
        resolve(args.output),
        args.run_id,
        tuple(args.recall_cutoffs),
        args.rank_cutoff,
    )
    logging.getLogger(__name__).info("Metrics: %s", result["metrics"])


if __name__ == "__main__":
    main()
