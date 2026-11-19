"""Build a full-corpus HNSW graph from audited, immutable FP32 embedding shards."""

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
from ltr_system.dense_index import numeric_ids
from ltr_system.project import initialize, parser, read_json, write_json

LOGGER = logging.getLogger(__name__)


def build(embeddings: Path, destination: Path, m: int, ef_construction: int, threads: int) -> dict:
    import faiss
    import numpy as np

    if not 2 <= m <= 128 or ef_construction < m or threads <= 0:
        raise ValueError("Expected 2 <= M <= 128, efConstruction >= M and positive threads")
    if destination.exists():
        raise FileExistsError(destination)
    if not embeddings.is_dir():
        raise FileNotFoundError(embeddings)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with writer_lock(embeddings), writer_lock(destination.parent / ".hnsw-build-lock"):
        started = time.perf_counter()
        coverage_path = embeddings / "coverage.json"
        config_path = embeddings / "configuration.json"
        coverage_hash = sha256_file(coverage_path)
        coverage = read_json(coverage_path)
        config = read_json(config_path)
        if (
            coverage.get("status") != "all embedding shards verified"
            or coverage.get("ordered_source_id_alignment") is not True
            or sha256_file(config_path) != coverage["configuration_sha256"]
            or coverage["dimension"] != config["dimension"]
            or (config["normalization"], config["precision"]) != ("L2", "float32")
        ):
            raise ValueError("Expected audited normalized FP32 embeddings")
        entries = coverage["shards"]
        names = [entry["name"] for entry in entries]
        if (
            not names
            or len(set(names)) != len(names)
            or any(Path(name).name != name or not name.startswith("part-") for name in names)
            or set(names) != {p.name for p in embeddings.iterdir() if p.name.startswith("part-")}
            or len(entries) != coverage["shard_count"]
            or any(entry["records"] <= 0 for entry in entries)
            or sum(entry["records"] for entry in entries) != coverage["records"]
        ):
            raise ValueError("Audited shard set or counts mismatch")
        expected = {coverage_path: coverage_hash, config_path: coverage["configuration_sha256"]}
        faiss.omp_set_num_threads(threads)
        graph = faiss.IndexHNSWFlat(config["dimension"], m, faiss.METRIC_INNER_PRODUCT)
        graph.hnsw.efConstruction = ef_construction
        graph.hnsw.efSearch = 64
        index = faiss.IndexIDMap2(graph)
        probes = []
        graph_add_seconds = 0.0
        for entry in entries:
            directory = embeddings / entry["name"]
            manifest_path = directory / "manifest.json"
            if sha256_file(manifest_path) != entry["manifest_sha256"]:
                raise ValueError("Shard manifest changed after coverage audit")
            report = read_json(manifest_path)
            if (
                report["status"] != "complete"
                or report["records"] != entry["records"]
                or any(report["identity"][key] != value for key, value in config.items())
            ):
                raise ValueError("Shard encoding configuration/count mismatch")
            expected[manifest_path] = entry["manifest_sha256"]
            for name in ("vectors.npy", "passage_ids.json"):
                path = directory / name
                if sha256_file(path) != report["files"][name]["sha256"]:
                    raise ValueError(f"Embedding file checksum mismatch: {path}")
                expected[path] = report["files"][name]["sha256"]
            ids = numeric_ids(read_json(directory / "passage_ids.json")["passage_ids"])
            if len(ids) != entry["records"]:
                raise ValueError("Passage ID count mismatch")
            vectors = np.load(directory / "vectors.npy", mmap_mode="r", allow_pickle=False)
            try:
                check_vectors(vectors, (len(ids), config["dimension"]))
                probes.append((int(ids[0]), np.array(vectors[0], copy=True)))
                for start in range(0, len(ids), 10_000):
                    batch = np.ascontiguousarray(vectors[start : start + 10_000])
                    began = time.perf_counter()
                    index.add_with_ids(batch, ids[start : start + 10_000])
                    graph_add_seconds += time.perf_counter() - began
                    LOGGER.info(
                        "Added %s/%s passages (%s)",
                        index.ntotal,
                        coverage["records"],
                        entry["name"],
                    )
                del batch
            finally:
                vectors._mmap.close()
                del vectors
        if index.ntotal != coverage["records"]:
            raise ValueError("Graph total differs from audited corpus")
        # Cross-shard ID uniqueness is inherited from the checksum-bound coverage
        # audit and its exact ordered source-ID alignment, as for the flat index.
        with tempfile.TemporaryDirectory(
            prefix=".hnsw-build-", dir=destination.parent
        ) as temporary:
            stage = Path(temporary) / "output"
            stage.mkdir()
            LOGGER.info("Saving full HNSW index")
            faiss.write_index(index, str(stage / "hnsw.faiss"))
            del index, graph
            gc.collect()
            LOGGER.info("Reloading HNSW index for verification")
            restored = faiss.read_index(str(stage / "hnsw.faiss"))
            restored_graph = faiss.downcast_index(restored.index)
            if (
                not isinstance(restored_graph, faiss.IndexHNSWFlat)
                or restored.metric_type != faiss.METRIC_INNER_PRODUCT
                or restored.ntotal != coverage["records"]
                or restored.d != config["dimension"]
                or restored_graph.hnsw.nb_neighbors(0) != 2 * m
                or restored_graph.hnsw.efConstruction != ef_construction
            ):
                raise ValueError("Reloaded graph metadata mismatch")
            restored_graph.hnsw.efSearch = 64
            query_vectors = np.stack([vector for _, vector in probes])
            for pid, vector in probes:
                np.testing.assert_array_equal(restored.reconstruct(pid), vector)
            scores, hits = restored.search(query_vectors, min(10, restored.ntotal))
            for query, values, found in zip(query_vectors, scores, hits, strict=True):
                if (
                    (found < 0).any()
                    or len(np.unique(found)) != len(found)
                    or not np.isfinite(values).all()
                    or (np.diff(values) > 0).any()
                ):
                    raise ValueError("Invalid reloaded search results")
                for score, pid in zip(values, found, strict=True):
                    np.testing.assert_allclose(
                        score, query @ restored.reconstruct(int(pid)), atol=1e-5, rtol=1e-5
                    )
            del restored_graph, restored
            LOGGER.info("Rechecking all input checksums before publication")
            for path, digest in expected.items():
                if sha256_file(path) != digest:
                    raise ValueError(f"Input changed during construction: {path}")
            index_file = stage / "hnsw.faiss"
            file_identity = {"bytes": index_file.stat().st_size, "sha256": sha256_file(index_file)}
            result = {
                "schema_version": 1,
                "status": "full HNSW index built and reload checked",
                "index_type": "IndexIDMap2(IndexHNSWFlat)",
                "metric": "inner product",
                "vectors": coverage["records"],
                "dimension": config["dimension"],
                "shard_count": len(entries),
                "coverage_sha256": coverage_hash,
                "embedding_configuration_sha256": coverage["configuration_sha256"],
                "hnsw": {"M": m, "efConstruction": ef_construction, "saved_efSearch": 64},
                "graph_add_seconds": graph_add_seconds,
                "build_and_verification_seconds": time.perf_counter() - started,
                "timing_scope": "add timer covers only graph add calls; total includes input checks, saving, reload, probes and final checksums",
                "reconstruction_checks": len(probes),
                "scope": "full corpus construction and structural checks; real-query quality and latency not measured",
                "build_order": "coverage shard order; source row order; 10000-row add calls",
                "reproducibility": "parallel insertion can vary graph; file checksum identifies this build",
                "faiss": faiss.__version__,
                "threads": threads,
                "host": socket.gethostname(),
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "files": {"hnsw.faiss": file_identity},
            }
            write_json(stage / "manifest.json", result)
            stage.rename(destination)
    return result


def main() -> None:
    command = parser("Build and verify HNSW over the full audited passage corpus.")
    command.add_argument("--m", type=int, default=32)
    command.add_argument("--ef-construction", type=int, default=200)
    command.add_argument("--threads", type=int, default=8)
    args = command.parse_args()
    root = initialize(args)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run full HNSW construction in a CPU Slurm allocation")
    if args.threads > int(os.environ.get("SLURM_CPUS_PER_TASK", "1")):
        raise ValueError("Threads exceed allocated CPUs")
    result = build(
        root / "artifacts/embeddings/e5-small-v2-fp32",
        root / f"artifacts/indexes/e5-hnsw-m{args.m}-efc{args.ef_construction}-v1",
        args.m,
        args.ef_construction,
        args.threads,
    )
    LOGGER.info("Full HNSW build complete: %s vectors", result["vectors"])
