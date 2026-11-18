"""Cross-check saved binary metrics using independently parsed trec_eval inputs."""

from __future__ import annotations

import logging
import math
import os
import re
import sys
from importlib.metadata import version
from pathlib import Path

from ltr_system.data_download import sha256_file
from ltr_system.project import initialize, parser, read_json, write_json

LOGGER = logging.getLogger(__name__)
TOLERANCE = 1e-12


def independent_inputs(queries: Path, qrels: Path, ranking: Path):
    """Do not reuse the project's TSV adapters or metric implementations."""
    ids = []
    with queries.open(encoding="utf-8") as stream:
        for line in stream:
            qid, text = line.rstrip("\r\n").split("\t")
            if not qid or not text or qid in ids:
                raise ValueError("Invalid or duplicate query")
            ids.append(qid)
    if not ids:
        raise ValueError("Empty query population")
    judgments = {qid: {} for qid in ids}
    with qrels.open(encoding="utf-8") as stream:
        for line in stream:
            qid, iteration, pid, label = line.rstrip("\r\n").split("\t")
            if qid not in judgments or iteration != "0" or label not in {"0", "1"} or not pid:
                raise ValueError("Invalid binary judgment")
            if pid in judgments[qid] and judgments[qid][pid] != int(label):
                raise ValueError("Conflicting judgment")
            judgments[qid][pid] = int(label)
    if any(1 not in labels.values() for labels in judgments.values()):
        raise ValueError("Every query requires a known positive")
    ranked = {qid: {} for qid in ids}
    with ranking.open(encoding="utf-8") as stream:
        for line in stream:
            qid, pid, raw_rank = line.rstrip("\r\n").split("\t")
            if qid not in ranked or not pid or not raw_rank.isascii() or not raw_rank.isdecimal():
                raise ValueError("Invalid ranking row")
            rank = int(raw_rank)
            if rank <= 0 or rank in ranked[qid]:
                raise ValueError("Invalid or duplicate rank")
            ranked[qid][rank] = pid
    runs = {}
    for qid, entries in ranked.items():
        if sorted(entries) != list(range(1, len(entries) + 1)):
            raise ValueError("Noncontiguous ranking")
        if len(set(entries.values())) != len(entries):
            raise ValueError("Duplicate passage ID")
        # Strictly decreasing synthetic scores preserve the saved rank order.
        runs[qid] = {pid: -float(rank) for rank, pid in sorted(entries.items())}
    return ids, judgments, runs


def crosscheck(queries: Path, qrels: Path, ranking: Path, saved: Path, reference) -> dict:
    report = read_json(saved)
    paths = {"queries": queries, "qrels": qrels, "run": ranking, "evaluation": saved}
    identities = {name: sha256_file(path) for name, path in paths.items()}
    for name in ("queries", "qrels", "run"):
        if identities[name] != report["inputs"][name]["sha256"]:
            raise ValueError(f"Saved evaluation input mismatch: {name}")
    ids, judgments, runs = independent_inputs(queries, qrels, ranking)
    cutoffs = report["recall_cutoffs"]
    cutoff = report["rank_cutoff"]
    if (
        type(cutoff) is not int
        or cutoff <= 0
        or not cutoffs
        or any(type(k) is not int or k <= 0 for k in cutoffs)
        or len(set(cutoffs)) != len(cutoffs)
    ):
        raise ValueError("Invalid saved metric cutoffs")
    if report["query_count"] != len(ids) or set(report["per_query"]) != set(ids):
        raise ValueError("Saved evaluation query population mismatch")
    if report["empty_rankings"] != sum(not hits for hits in runs.values()):
        raise ValueError("Saved empty ranking count mismatch")
    active = {qid: hits for qid, hits in runs.items() if hits}
    measured = reference.RelevanceEvaluator(
        judgments, {f"ndcg_cut_{cutoff}", *[f"recall_{k}" for k in cutoffs]}
    ).evaluate(active)
    truncated = {qid: dict(list(hits.items())[:cutoff]) for qid, hits in active.items()}
    reciprocal = reference.RelevanceEvaluator(judgments, {"recip_rank"}).evaluate(truncated)
    mapping = {
        f"ndcg@{cutoff}": f"ndcg_cut_{cutoff}",
        **{f"recall@{k}": f"recall_{k}" for k in cutoffs},
    }
    expected_names = {*mapping, f"rr@{cutoff}"}
    max_errors = dict.fromkeys(expected_names, 0.0)
    reference_values = {}
    for qid in ids:
        values = {name: measured[qid][key] if runs[qid] else 0.0 for name, key in mapping.items()}
        values[f"rr@{cutoff}"] = reciprocal[qid]["recip_rank"] if runs[qid] else 0.0
        if set(report["per_query"][qid]) != expected_names:
            raise ValueError("Saved per-query metric names differ")
        for name, value in values.items():
            actual = report["per_query"][qid][name]
            if not math.isfinite(actual) or not math.isclose(
                actual, value, rel_tol=0, abs_tol=TOLERANCE
            ):
                raise ValueError(
                    f"Reference mismatch: {qid}, {name}: saved={actual}, reference={value}"
                )
            max_errors[name] = max(max_errors[name], abs(actual - value))
        reference_values[qid] = values
    aggregate = {
        (f"mrr@{cutoff}" if name == f"rr@{cutoff}" else name): math.fsum(
            reference_values[qid][name] for qid in ids
        )
        / len(ids)
        for name in expected_names
    }
    if set(report["metrics"]) != set(aggregate):
        raise ValueError("Saved aggregate metric names differ")
    for name, value in aggregate.items():
        actual = report["metrics"][name]
        if not math.isfinite(actual) or not math.isclose(
            actual, value, rel_tol=0, abs_tol=TOLERANCE
        ):
            raise ValueError(f"Aggregate reference mismatch: {name}")
    for name, path in paths.items():
        if sha256_file(path) != identities[name]:
            raise ValueError("Input changed during reference evaluation")
    return {
        "status": "PASS",
        "queries": len(ids),
        "input_sha256": identities,
        "reference_metrics": aggregate,
        "max_per_query_absolute_error": max_errors,
        "absolute_tolerance": TOLERANCE,
        "score_policy": "negative saved rank; unique scores preserve existing order",
        "mrr_policy": "recip_rank on top rank_cutoff only; macro over every frozen query",
    }


def main() -> None:
    command = parser("Cross-check saved validation metrics against pytrec_eval.")
    command.add_argument("--reference-tools", type=Path)
    command.add_argument(
        "--run-names", nargs="+", default=["bm25-validation-default", "e5-exact-validation-default"]
    )
    command.add_argument(
        "--output", type=Path, default=Path("artifacts/evaluation/reference-parity-v1.json")
    )
    args = command.parse_args()
    root = initialize(args)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run full ranking cross-checks in a CPU Slurm allocation")
    output = args.output if args.output.is_absolute() else root / args.output
    if output.exists():
        raise FileExistsError(output)
    tools = args.reference_tools or root / "tools/reference-eval"
    sys.path.insert(0, str(tools.resolve()))
    import pytrec_eval

    if version("pytrec-eval-terrier") != "0.5.11":
        raise ValueError("Expected pytrec-eval-terrier 0.5.11")
    split = root / "data/processed/msmarco-passage-v1-splits"
    split_report = read_json(split / "manifest.json")
    for name in ("validation.queries.tsv", "validation.qrels.tsv"):
        if sha256_file(split / name) != split_report["outputs"][name]["sha256"]:
            raise ValueError("Frozen validation split checksum mismatch")
    results = {}
    for name in args.run_names:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) or name in results:
            raise ValueError("Run names must be unique simple directory names")
        directory = root / "artifacts/runs" / name
        LOGGER.info("Cross-checking %s", name)
        results[name] = crosscheck(
            split / "validation.queries.tsv",
            split / "validation.qrels.tsv",
            directory / "ranking.tsv",
            directory / "evaluation.json",
            pytrec_eval,
        )
        LOGGER.info("Per-query and aggregate agreement: PASS (%s)", name)
    write_json(
        output,
        {
            "schema_version": 1,
            "status": "PASS",
            "reference": "pytrec-eval-terrier",
            "version": version("pytrec-eval-terrier"),
            "module_path": pytrec_eval.__file__,
            "scope": "binary relevance on saved validation rankings; not retrieval correctness or significance",
            "final_queries_read": False,
            "job_id": os.environ["SLURM_JOB_ID"],
            "runs": results,
        },
    )
    LOGGER.info("Saved reference comparison: %s", output)
