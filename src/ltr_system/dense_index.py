"""Build an exact cosine-reference FAISS index from audited FP32 embeddings."""

from __future__ import annotations

import gc
import logging
import os
import socket
import tempfile
import time
from pathlib import Path

from ltr_system.data_download import sha256_file
from ltr_system.dense_encode import check_vectors, writer_lock
from ltr_system.project import initialize, parser, read_json, write_json

LOGGER = logging.getLogger(__name__)


def numeric_ids(values: list[str]):
    import numpy as np

    for value in values:
        if (
            not isinstance(value, str)
            or not value.isascii()
            or not value.isdecimal()
            or str(int(value)) != value
            or int(value) > 2**63 - 1
        ):
            raise ValueError("Passage IDs must round-trip as nonnegative int64 values")
    result = np.array(values, dtype=np.int64)
    if len(np.unique(result)) != len(result):
        raise ValueError("Duplicate passage IDs")
    return result


def build(embeddings: Path, destination: Path, threads: int) -> dict:
    import faiss
    import numpy as np

    if threads <= 0:
        raise ValueError("Thread count must be positive")
    if destination.exists():
        raise FileExistsError(f"Index already exists: {destination}")
    if not embeddings.is_dir():
        raise FileNotFoundError(embeddings)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with writer_lock(embeddings), writer_lock(destination.parent / ".exact-build-lock"):
        coverage_hash = sha256_file(embeddings / "coverage.json")
        coverage = read_json(embeddings / "coverage.json")
        config = read_json(embeddings / "configuration.json")
        if coverage.get("status") != "all embedding shards verified":
            raise ValueError("Complete embedding coverage must be verified first")
        if sha256_file(embeddings / "configuration.json") != coverage["configuration_sha256"]:
            raise ValueError("Embedding configuration differs from coverage audit")
        entries = coverage["shards"]
        names = [entry["name"] for entry in entries]
        if (
            not names
            or len(set(names)) != len(names)
            or any(Path(name).name != name for name in names)
        ):
            raise ValueError("Invalid audited shard set")
        if set(names) != {p.name for p in embeddings.iterdir() if p.name.startswith("part-")}:
            raise ValueError("Embedding shard set changed since coverage audit")
        if (
            len(entries) != coverage["shard_count"]
            or sum(e["records"] for e in entries) != coverage["records"]
        ):
            raise ValueError("Inconsistent audited coverage counts")
        dimension = coverage["dimension"]
        if dimension != config["dimension"]:
            raise ValueError("Configuration dimension mismatch")
        faiss.omp_set_num_threads(threads)
        index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
        checks = []
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix=".exact-", dir=destination.parent) as temporary:
            stage = Path(temporary) / "output"
            stage.mkdir()
            for entry in entries:
                directory = embeddings / entry["name"]
                if sha256_file(directory / "manifest.json") != entry["manifest_sha256"]:
                    raise ValueError("Shard manifest changed after coverage audit")
                report = read_json(directory / "manifest.json")
                if (
                    report["status"] != "complete"
                    or report["records"] != entry["records"]
                    or any(report["identity"][key] != value for key, value in config.items())
                ):
                    raise ValueError("Shard configuration/count mismatch")
                for name in ("vectors.npy", "passage_ids.json"):
                    path = directory / name
                    if sha256_file(path) != report["files"][name]["sha256"]:
                        raise ValueError(f"Embedding file changed: {path}")
                ids = numeric_ids(read_json(directory / "passage_ids.json")["passage_ids"])
                if len(ids) != entry["records"]:
                    raise ValueError("Passage ID count mismatch")
                vectors = np.load(directory / "vectors.npy", mmap_mode="r", allow_pickle=False)
                try:
                    check_vectors(vectors, (len(ids), dimension))
                    # Chunk additions avoid a full-corpus NumPy copy.
                    for start in range(0, len(ids), 10_000):
                        index.add_with_ids(
                            np.ascontiguousarray(vectors[start : start + 10_000]),
                            ids[start : start + 10_000],
                        )
                    checks.append((int(ids[0]), np.array(vectors[0], copy=True)))
                finally:
                    vectors._mmap.close()
                    del vectors
                LOGGER.info("Added %s; total vectors: %s", entry["name"], index.ntotal)
            if index.ntotal != coverage["records"]:
                raise ValueError("FAISS total differs from audited corpus")
            faiss.write_index(index, str(stage / "exact.faiss"))
            del index
            gc.collect()
            restored = faiss.read_index(str(stage / "exact.faiss"))
            if restored.ntotal != coverage["records"] or restored.d != dimension:
                raise ValueError("Reloaded FAISS metadata mismatch")
            for pid, expected in checks:
                np.testing.assert_allclose(restored.reconstruct(pid), expected, atol=0, rtol=0)
            # Exercise search with one stored vector; ties need not return its own ID first.
            query = checks[0][1].reshape(1, -1)
            scores, found = restored.search(query, min(10, restored.ntotal))
            for score, pid in zip(scores[0], found[0], strict=True):
                if pid < 0 or not np.isfinite(score):
                    raise ValueError("Invalid reloaded search result")
                np.testing.assert_allclose(
                    score, query[0] @ restored.reconstruct(int(pid)), atol=1e-5, rtol=1e-5
                )
            del restored
            if sha256_file(embeddings / "coverage.json") != coverage_hash:
                raise ValueError("Coverage report changed during indexing")
            result = {
                "schema_version": 1,
                "status": "exact index built and reload checked",
                "index_type": "IndexIDMap2(IndexFlatIP)",
                "metric": "inner product",
                "cosine_equivalence": "requires L2-normalized query and passage vectors",
                "vectors": coverage["records"],
                "dimension": dimension,
                "coverage_sha256": coverage_hash,
                "embedding_configuration_sha256": coverage["configuration_sha256"],
                "faiss": faiss.__version__,
                "threads": threads,
                "build_and_verification_seconds": time.perf_counter() - started,
                "reconstruction_checks": len(checks),
                "files": {
                    "exact.faiss": {
                        "bytes": (stage / "exact.faiss").stat().st_size,
                        "sha256": sha256_file(stage / "exact.faiss"),
                    }
                },
            }
            write_json(stage / "manifest.json", result)
            stage.rename(destination)
    LOGGER.info("Exact FAISS index saved: %s", destination)
    return result


def main() -> None:
    command = parser("Build an exact FAISS index over the audited full passage corpus.")
    command.add_argument("--threads", type=int, default=4)
    args = command.parse_args()
    root = initialize(args)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run full FAISS indexing in a CPU Slurm allocation")
    if args.threads > int(os.environ.get("SLURM_CPUS_PER_TASK", "1")):
        raise ValueError("Thread count exceeds allocated CPUs")
    result = build(
        root / "artifacts/embeddings/e5-small-v2-fp32",
        root / "artifacts/indexes/e5-flat-ip-v1",
        args.threads,
    )
    LOGGER.info("Index vectors: %s; host: %s", result["vectors"], socket.gethostname())


if __name__ == "__main__":
    main()
