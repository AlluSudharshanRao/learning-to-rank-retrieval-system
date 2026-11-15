"""Binary judged-relevance metrics on explicit, already ordered passage IDs."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence, Set


def _ids(values: Sequence[str], name: str) -> None:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of IDs, not a string")
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{name} must contain nonempty string IDs")
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate IDs in {name}")


def _cutoff(k: int) -> None:
    if type(k) is not int or k <= 0:
        raise ValueError("Cutoff must be a positive integer")


def query_metrics(ranking: Sequence[str], relevant: Set[str], k: int) -> dict[str, float]:
    """Score one query; require at least one known positive and unique ranked IDs.

    Rankings are ordered by the caller. Unjudged IDs receive zero credit.
    NDCG uses binary gain and log2(rank + 1) discount, with one-based ranks.
    """
    _cutoff(k)
    _ids(ranking, "ranking")
    if not isinstance(relevant, Set):
        raise ValueError("Relevant IDs must be a set of unique known positives")
    _ids(list(relevant), "relevant IDs")
    if not relevant:
        raise ValueError("Each evaluated query must have at least one known positive")
    hits = [rank for rank, pid in enumerate(ranking[:k], 1) if pid in relevant]
    dcg = math.fsum(1.0 / math.log2(rank + 1) for rank in hits)
    ideal = math.fsum(1.0 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant)) + 1))
    return {
        f"recall@{k}": len(hits) / len(relevant),
        f"rr@{k}": 1.0 / hits[0] if hits else 0.0,
        f"ndcg@{k}": dcg / ideal,
    }


def evaluate(
    query_ids: Sequence[str],
    rankings: Mapping[str, Sequence[str]],
    positives: Mapping[str, Set[str]],
    recall_cutoffs: Sequence[int] = (20, 50, 100, 1000),
    rank_cutoff: int = 10,
) -> dict:
    """Macro-average over a fixed query list, retaining per-query scores.

    Missing rankings count as empty results. Extra ranked query IDs are rejected
    to catch accidental split mixing. Positive mappings may contain extra queries.
    """
    _ids(query_ids, "evaluation query list")
    if not query_ids:
        raise ValueError("Evaluation query list cannot be empty")
    _cutoff(rank_cutoff)
    if not recall_cutoffs:
        raise ValueError("At least one recall cutoff is required")
    for k in recall_cutoffs:
        _cutoff(k)
    if len(set(recall_cutoffs)) != len(recall_cutoffs):
        raise ValueError("Duplicate recall cutoffs")
    if set(rankings) - set(query_ids):
        raise ValueError("Rankings contain queries outside the evaluation query list")
    per_query = {}
    missing = 0
    empty = 0
    for qid in query_ids:
        if qid not in positives:
            raise ValueError(f"Missing judgments for query {qid}")
        ranking = rankings.get(qid, ())
        missing += qid not in rankings
        empty += len(ranking) == 0
        ranked = query_metrics(ranking, positives[qid], rank_cutoff)
        values = {
            f"rr@{rank_cutoff}": ranked[f"rr@{rank_cutoff}"],
            f"ndcg@{rank_cutoff}": ranked[f"ndcg@{rank_cutoff}"],
        }
        for k in recall_cutoffs:
            values[f"recall@{k}"] = query_metrics(ranking, positives[qid], k)[f"recall@{k}"]
        per_query[qid] = values
    aggregate = {
        name: math.fsum(values[name] for values in per_query.values()) / len(query_ids)
        for name in next(iter(per_query.values()))
    }
    aggregate[f"mrr@{rank_cutoff}"] = aggregate.pop(f"rr@{rank_cutoff}")
    return {
        "metrics": aggregate,
        "per_query": per_query,
        "query_count": len(query_ids),
        "missing_rankings": missing,
        "empty_rankings": empty,
        "policy": {"gain": "binary", "unjudged": "zero credit", "averaging": "macro"},
    }
