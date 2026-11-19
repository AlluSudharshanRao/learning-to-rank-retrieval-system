"""Build and inspect HNSW on one audited passage shard before full indexing."""

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


def build_pilot(
    embeddings: Path, destination: Path, m: int, ef_construction: int, threads: int
) -> dict:
    import faiss
    import numpy as np

    if type(m) is not int or not 2 <= m <= 128:
        raise ValueError("M must be an integer between 2 and 128")
    if ef_construction < m or threads <= 0:
        raise ValueError("Expected efConstruction >= M and positive thread count")
    if destination.exists():
        raise FileExistsError(destination)
    if not embeddings.is_dir():
        raise FileNotFoundError(embeddings)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with writer_lock(embeddings), writer_lock(destination.parent / ".hnsw-pilot-lock"):
        coverage_path = embeddings / "coverage.json"
        config_path = embeddings / "configuration.json"
        coverage = read_json(coverage_path)
        config = read_json(config_path)
        if (
            coverage.get("status") != "all embedding shards verified"
            or coverage.get("ordered_source_id_alignment") is not True
            or sha256_file(config_path) != coverage["configuration_sha256"]
            or coverage["dimension"] != config["dimension"]
            or config["normalization"] != "L2"
            or config["precision"] != "float32"
        ):
            raise ValueError("Expected audited normalized FP32 embedding configuration")
        entry = coverage["shards"][0]
        if entry["name"] != "part-00000" or entry["records"] <= 0:
            raise ValueError("Expected the audited first corpus shard")
        shard = embeddings / entry["name"]
        report_path = shard / "manifest.json"
        if sha256_file(report_path) != entry["manifest_sha256"]:
            raise ValueError("Shard manifest differs from coverage audit")
        shard_report = read_json(report_path)
        if (
            shard_report["status"] != "complete"
            or shard_report["records"] != entry["records"]
            or any(shard_report["identity"][key] != value for key, value in config.items())
        ):
            raise ValueError("Shard count/configuration mismatch")
        expected = {
            coverage_path: sha256_file(coverage_path),
            config_path: coverage["configuration_sha256"],
            report_path: entry["manifest_sha256"],
            **{
                shard / name: shard_report["files"][name]["sha256"]
                for name in ("vectors.npy", "passage_ids.json")
            },
        }

        def verify_inputs():
            for path, digest in expected.items():
                if sha256_file(path) != digest:
                    raise ValueError(f"Input checksum mismatch: {path.name}")

        verify_inputs()
        ids = numeric_ids(read_json(shard / "passage_ids.json")["passage_ids"])
        if len(ids) != entry["records"]:
            raise ValueError("Passage ID count mismatch")
        dimension = config["dimension"]
        faiss.omp_set_num_threads(threads)
        graph = faiss.IndexHNSWFlat(dimension, m, faiss.METRIC_INNER_PRODUCT)
        graph.hnsw.efConstruction = ef_construction
        graph.hnsw.efSearch = 64
        index = faiss.IndexIDMap2(graph)
        exact = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
        vectors = np.load(shard / "vectors.npy", mmap_mode="r", allow_pickle=False)
        try:
            check_vectors(vectors, (len(ids), dimension))
            # Fixed source positions are sanity probes, not held-out search queries.
            positions = np.unique(np.linspace(0, len(ids) - 1, min(64, len(ids)), dtype=int))
            probes = np.array(vectors[positions], copy=True)
            probe_ids = ids[positions].copy()
            add_seconds = 0.0
            for start in range(0, len(ids), 10_000):
                batch = np.ascontiguousarray(vectors[start : start + 10_000])
                batch_ids = ids[start : start + 10_000]
                began = time.perf_counter()
                index.add_with_ids(batch, batch_ids)
                add_seconds += time.perf_counter() - began
                exact.add_with_ids(batch, batch_ids)
                LOGGER.info("Added %s/%s pilot passages", index.ntotal, len(ids))
        finally:
            vectors._mmap.close()
            del vectors
        with tempfile.TemporaryDirectory(
            prefix=".hnsw-pilot-", dir=destination.parent
        ) as temporary:
            stage = Path(temporary) / "output"
            stage.mkdir()
            if index.ntotal != len(ids):
                raise ValueError("HNSW passage count mismatch")
            faiss.write_index(index, str(stage / "hnsw.faiss"))
            # Release owning wrappers together before reload; avoid retaining a second graph.
            del index, graph
            gc.collect()
            restored = faiss.read_index(str(stage / "hnsw.faiss"))
            restored_graph = faiss.downcast_index(restored.index)
            if (
                not isinstance(restored_graph, faiss.IndexHNSWFlat)
                or restored.ntotal != len(ids)
                or restored.d != dimension
                or restored.metric_type != faiss.METRIC_INNER_PRODUCT
                or restored_graph.hnsw.efConstruction != ef_construction
                or restored_graph.hnsw.nb_neighbors(0) != 2 * m
            ):
                raise ValueError("Reloaded HNSW configuration mismatch")
            restored_graph.hnsw.efSearch = 64
            for pid, vector in zip(probe_ids, probes, strict=True):
                np.testing.assert_array_equal(restored.reconstruct(int(pid)), vector)
            k = min(10, len(ids))
            scores, found = restored.search(probes, k)
            _, target = exact.search(probes, k)
            allowed = set(ids.tolist())
            overlaps = []
            for query, query_scores, hits, truth in zip(probes, scores, found, target, strict=True):
                if (
                    len(set(hits.tolist())) != k
                    or any(int(pid) not in allowed for pid in hits)
                    or not np.isfinite(query_scores).all()
                    or (np.diff(query_scores) > 0).any()
                ):
                    raise ValueError("Invalid HNSW result IDs or scores")
                for score, pid in zip(query_scores, hits, strict=True):
                    np.testing.assert_allclose(
                        score, query @ restored.reconstruct(int(pid)), atol=1e-5, rtol=1e-5
                    )
                overlaps.append(len(set(hits.tolist()) & set(truth.tolist())) / k)
            verify_inputs()
            result = {
                "schema_version": 1,
                "status": "pilot index built and reload checked",
                "scope": "first corpus shard; stored-vector sanity probes; not full-corpus relevance or latency",
                "index_type": "IndexIDMap2(IndexHNSWFlat)",
                "metric": "inner product",
                "vectors": len(ids),
                "dimension": dimension,
                "hnsw": {"M": m, "efConstruction": ef_construction, "probe_efSearch": 64},
                "source_shard": entry["name"],
                "coverage_sha256": expected[coverage_path],
                "embedding_configuration_sha256": expected[config_path],
                "shard_manifest_sha256": entry["manifest_sha256"],
                "probe_count": len(probes),
                "probe_k": k,
                "probe_neighbor_overlap": float(np.mean(overlaps)),
                "tie_policy": "ID-set overlap; equal-score exact neighbors may differ",
                "graph_add_seconds": add_seconds,
                "timing_scope": "HNSW add calls only; no exact additions, loading, saving or verification",
                "faiss": faiss.__version__,
                "threads": threads,
                "build_order": "source row order; 10000-row add calls; parallel insertion may vary graph",
                "host": socket.gethostname(),
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "files": {
                    "hnsw.faiss": {
                        "bytes": (stage / "hnsw.faiss").stat().st_size,
                        "sha256": sha256_file(stage / "hnsw.faiss"),
                    }
                },
            }
            write_json(stage / "manifest.json", result)
            stage.rename(destination)
    return result


def main() -> None:
    command = parser("Build an HNSW pilot over the first audited passage shard.")
    command.add_argument("--m", type=int, default=32)
    command.add_argument("--ef-construction", type=int, default=200)
    command.add_argument("--threads", type=int, default=4)
    args = command.parse_args()
    root = initialize(args)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run HNSW construction in a CPU Slurm allocation")
    if args.threads > int(os.environ.get("SLURM_CPUS_PER_TASK", "1")):
        raise ValueError("Threads exceed allocated CPUs")
    result = build_pilot(
        root / "artifacts/embeddings/e5-small-v2-fp32",
        root / f"artifacts/indexes/e5-hnsw-pilot-m{args.m}-efc{args.ef_construction}",
        args.m,
        args.ef_construction,
        args.threads,
    )
    LOGGER.info("HNSW pilot completed: %s", result)
