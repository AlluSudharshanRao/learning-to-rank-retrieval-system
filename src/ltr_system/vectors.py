"""Validation of normalized vectors and exact search results."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def validate_vectors(
    queries: NDArray, passages: NDArray, ids: NDArray
) -> tuple[NDArray, NDArray, NDArray]:
    if queries.ndim != 2 or passages.ndim != 2:
        raise ValueError("Query and passage vectors must be two-dimensional.")
    if not queries.shape[0] or not passages.shape[0] or not passages.shape[1]:
        raise ValueError("Empty vectors are not supported.")
    if queries.shape[1] != passages.shape[1]:
        raise ValueError("Query and passage dimensions differ.")
    if ids.ndim != 1 or len(ids) != len(passages) or ids.dtype != np.int64:
        raise ValueError("Passage IDs must be aligned one-dimensional int64 values.")
    if (ids < 0).any() or len(np.unique(ids)) != len(ids):
        raise ValueError("Passage IDs must be unique and nonnegative.")
    for vectors in (queries, passages):
        if not np.issubdtype(vectors.dtype, np.floating) or not np.isfinite(vectors).all():
            raise ValueError("Vectors must contain finite floating-point values.")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5, rtol=1e-5):
            raise ValueError("Vectors must have unit L2 norm for cosine search.")
    return (
        np.ascontiguousarray(queries, dtype=np.float32),
        np.ascontiguousarray(passages, dtype=np.float32),
        np.ascontiguousarray(ids),
    )


def check_exact_results(
    queries: NDArray, passages: NDArray, ids: NDArray, scores: NDArray, found: NDArray
) -> None:
    """Check full rankings, accepting any valid order among tied scores."""
    queries, passages, ids = validate_vectors(queries, passages, ids)
    reference = queries @ passages.T
    expected_shape = (len(queries), len(passages))
    if scores.shape != expected_shape or found.shape != expected_shape:
        raise ValueError("Expected a full ranking for every query.")
    if found.dtype != np.int64:
        raise ValueError("Returned IDs must use int64 values.")
    positions = {int(pid): position for position, pid in enumerate(ids)}
    for row in range(len(queries)):
        if set(found[row].tolist()) != set(ids.tolist()) or len(np.unique(found[row])) != len(ids):
            raise ValueError("Returned IDs are not a permutation of corpus IDs.")
        matched = reference[row, [positions[int(pid)] for pid in found[row]]]
        np.testing.assert_allclose(scores[row], matched, atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(scores[row], np.sort(reference[row])[::-1], atol=1e-6, rtol=1e-5)
