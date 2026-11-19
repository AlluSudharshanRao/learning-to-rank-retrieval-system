"""Matched, repeated CPU search benchmarks; excludes encoding and serving overhead."""

from __future__ import annotations

import logging
import os
import platform
import random
import sys
import time
from pathlib import Path

from ltr_system.data_download import sha256_file
from ltr_system.dense_encode import check_vectors
from ltr_system.project import initialize, parser, read_json, write_json
from ltr_system.query_encode import validation_inputs

LOGGER = logging.getLogger(__name__)


def measure(index, vectors, order, k, batch_size, clock=time.perf_counter):
    """Prepare each query batch before timing; inspect outputs after timing."""
    import numpy as np

    if k <= 0 or batch_size <= 0 or not len(order) or k > index.ntotal:
        raise ValueError("Invalid benchmark depth, batch size or query order")
    times = []
    for start in range(0, len(order), batch_size):
        batch = np.ascontiguousarray(vectors[order[start : start + batch_size]])
        began = clock()
        scores, ids = index.search(batch, k)
        elapsed = clock() - began
        if elapsed <= 0 or not np.isfinite(elapsed):
            raise ValueError("Invalid timer duration")
        if (
            scores.shape != (len(batch), k)
            or ids.shape != scores.shape
            or not np.isfinite(scores).all()
            or (ids < 0).any()
            or (np.diff(scores, axis=1) > 0).any()
            or any(len(np.unique(row)) != k for row in ids)
        ):
            raise ValueError("Invalid benchmark search results")
        times.append(elapsed)
    return times


def schedule(count, sample_size, repeats, labels, seed):
    """Use identical query orders across algorithms within each repetition."""
    if (
        not 0 < sample_size <= count
        or repeats <= 0
        or not labels
        or len(set(labels)) != len(labels)
    ):
        raise ValueError("Invalid benchmark population, repeats or algorithm labels")
    rng = random.Random(seed)
    sampled = rng.sample(range(count), sample_size)
    base = list(labels)
    rng.shuffle(base)
    trials = []
    for repeat in range(repeats):
        batch_order = list(range(count))
        latency_order = sampled.copy()
        rng.shuffle(batch_order)
        rng.shuffle(latency_order)
        offset = repeat % len(base)
        trials.append(
            {
                "repeat": repeat,
                "algorithms": base[offset:] + base[:offset],
                "batch_order": batch_order,
                "latency_order": latency_order,
            }
        )
    return trials


def summarize(trials, query_count):
    import numpy as np

    if not trials or query_count <= 0:
        raise ValueError("Nonempty trials and positive query count required")
    batch_totals = [sum(trial["batch_call_seconds"]) for trial in trials]
    latencies = [value for trial in trials for value in trial["single_query_seconds"]]
    if not latencies or min(batch_totals + latencies) <= 0:
        raise ValueError("Empty or nonpositive timings")
    return {
        "batch_search_seconds_repetitions": batch_totals,
        "batch_queries_per_second_repetitions": [query_count / value for value in batch_totals],
        "median_batch_queries_per_second": float(
            np.median([query_count / value for value in batch_totals])
        ),
        "single_query_p50_ms": float(np.percentile(latencies, 50, method="linear") * 1000),
        "single_query_p95_ms": float(np.percentile(latencies, 95, method="linear") * 1000),
        "single_query_observations": len(latencies),
        "single_query_percentiles_by_repetition_ms": [
            {
                "p50": float(np.percentile(t["single_query_seconds"], 50, method="linear") * 1000),
                "p95": float(np.percentile(t["single_query_seconds"], 95, method="linear") * 1000),
            }
            for t in trials
        ],
    }


def run(
    root: Path,
    repeats: int,
    latency_queries: int,
    batch_size: int,
    threads: int,
    seed: int,
    output: Path,
):
    import faiss
    import numpy as np

    if output.exists():
        raise FileExistsError(output)
    if min(repeats, latency_queries, batch_size, threads) <= 0:
        raise ValueError("Benchmark settings must be positive")
    flat_root = root / "artifacts/indexes/e5-flat-ip-v1"
    graph_root = root / "artifacts/indexes/e5-hnsw-m32-efc200-v1"
    query_root = root / "artifacts/queries/e5-validation-fp32"
    split_root = root / "data/processed/msmarco-passage-v1-splits"
    directories = (flat_root, graph_root, query_root, split_root)
    expected = {p / "manifest.json": sha256_file(p / "manifest.json") for p in directories}
    flat, graph, queries, _ = [read_json(p / "manifest.json") for p in directories]
    ids, _, source = validation_inputs(split_root)
    if (
        flat.get("status") != "exact index built and reload checked"
        or graph.get("status") != "full HNSW index built and reload checked"
        or flat["metric"] != "inner product"
        or graph["metric"] != "inner product"
        or flat["coverage_sha256"] != graph["coverage_sha256"]
        or flat["embedding_configuration_sha256"] != graph["embedding_configuration_sha256"]
        or flat["vectors"] != graph["vectors"]
        or flat["dimension"] != graph["dimension"]
        or queries.get("status") != "validation queries encoded"
        or queries["source"] != source
        or queries["queries"] != len(ids)
        or queries["encoder"]["dimension"] != flat["dimension"]
        or queries["passage_configuration_sha256"] != flat["embedding_configuration_sha256"]
        or (queries["prefix"], queries["precision"], queries["pooling"], queries["normalization"])
        != ("query: ", "float32", "masked mean", "L2")
        or (graph["hnsw"]["M"], graph["hnsw"]["efConstruction"]) != (32, 200)
    ):
        raise ValueError("Incompatible exact, HNSW, queries or validation provenance")
    expected.update(
        {
            flat_root / "exact.faiss": flat["files"]["exact.faiss"]["sha256"],
            graph_root / "hnsw.faiss": graph["files"]["hnsw.faiss"]["sha256"],
            split_root / "validation.queries.tsv": source["query_file_sha256"],
            **{
                query_root / name: queries["files"][name]["sha256"]
                for name in ("vectors.npy", "query_ids.json")
            },
        }
    )

    def verify():
        for path, digest in expected.items():
            if sha256_file(path) != digest:
                raise ValueError(f"Benchmark input checksum mismatch: {path}")

    LOGGER.info("Checking both index files and query identities")
    verify()
    if read_json(query_root / "query_ids.json")["query_ids"] != ids:
        raise ValueError("Query vector ID order mismatch")
    vectors = np.load(query_root / "vectors.npy", allow_pickle=False)
    check_vectors(vectors, (len(ids), flat["dimension"]))
    faiss.omp_set_num_threads(threads)
    LOGGER.info("Loading both indexes into the same CPU process")
    exact = faiss.read_index(str(flat_root / "exact.faiss"))
    approximate = faiss.read_index(str(graph_root / "hnsw.faiss"))
    for index in (exact, approximate):
        if (
            not isinstance(index, faiss.IndexIDMap2)
            or index.ntotal != flat["vectors"]
            or index.d != flat["dimension"]
        ):
            raise ValueError("Loaded benchmark index metadata mismatch")
        if index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError("Loaded benchmark metric mismatch")
    backend = faiss.downcast_index(approximate.index)
    if (
        not isinstance(faiss.downcast_index(exact.index), faiss.IndexFlatIP)
        or not isinstance(backend, faiss.IndexHNSWFlat)
        or backend.hnsw.nb_neighbors(0) != 64
        or backend.hnsw.efConstruction != 200
    ):
        raise ValueError("Loaded benchmark index types/settings mismatch")
    labels = ["exact", "hnsw-ef1024", "hnsw-ef2048", "hnsw-ef4096"]
    trials = schedule(len(ids), latency_queries, repeats, labels, seed)
    results = {label: [] for label in labels}
    for trial in trials:
        for label in trial["algorithms"]:
            index = exact if label == "exact" else approximate
            if label != "exact":
                backend.hnsw.efSearch = int(label.removeprefix("hnsw-ef"))
            LOGGER.info("Repetition %s/%s: %s batch warmup", trial["repeat"] + 1, repeats, label)
            # One batch, twice, before each timed batch trial; results discarded.
            warm_order = trial["batch_order"][:batch_size]
            for _ in range(2):
                measure(index, vectors, warm_order, 1000, batch_size)
            batch_times = measure(index, vectors, trial["batch_order"], 1000, batch_size)
            LOGGER.info("%s batch search seconds: %.6f", label, sum(batch_times))
            for _ in range(2):
                measure(index, vectors, trial["latency_order"][:1], 1000, 1)
            latency_times = measure(index, vectors, trial["latency_order"], 1000, 1)
            results[label].append(
                {
                    "repeat": trial["repeat"],
                    "batch_call_seconds": batch_times,
                    "single_query_seconds": latency_times,
                }
            )
            LOGGER.info("%s single-query samples: %s", label, len(latency_times))
    verify()
    metadata = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "host": platform.node(),
        "python": sys.version,
        "numpy": np.__version__,
        "faiss": faiss.__version__,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        metadata["cpu_model"] = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")
            ),
            "unknown",
        )
    try:
        import resource

        metadata["process_peak_rss_kib_linux"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        pass
    report = {
        "schema_version": 1,
        "status": "matched CPU search benchmark completed",
        "scope": "warm in-process CPU search only; no encoding, network, queueing, concurrency or reranking",
        "timing_scope": "Python FAISS search calls only; batch preparation and output checks excluded",
        "population": source,
        "final_queries_read": False,
        "judgments_read": False,
        "input_sha256": {str(path.relative_to(root)): digest for path, digest in expected.items()},
        "k": 1000,
        "threads": threads,
        "batch_size": batch_size,
        "batch_queries": len(ids),
        "repeats": repeats,
        "latency_unique_queries": latency_queries,
        "seed": seed,
        "warmup": "two batch calls and two single-query calls per algorithm per repetition",
        "schedule": trials,
        "query_ids_in_vector_order": ids,
        "percentile_policy": "NumPy linear; pooled repeated observations of the fixed sample; not independent users or serving p95",
        "algorithm_order_policy": "seed-shuffled initial order rotated each repetition; multiples of four repeats balance algorithm position",
        "metadata": metadata,
        "results": {
            label: {"trials": values, "summary": summarize(values, len(ids))}
            for label, values in results.items()
        },
    }
    write_json(output, report)
    LOGGER.info("Benchmark saved: %s", output)
    return report


def main() -> None:
    command = parser("Benchmark exact and HNSW CPU search under matched repeated workloads.")
    command.add_argument("--repeats", type=int, default=4)
    command.add_argument("--latency-queries", type=int, default=200)
    command.add_argument("--batch-size", type=int, default=32)
    command.add_argument("--threads", type=int, default=4)
    command.add_argument("--seed", type=int, default=2026)
    command.add_argument(
        "--output", type=Path, default=Path("artifacts/benchmarks/search-matched-v1.json")
    )
    args = command.parse_args()
    root = initialize(args)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run matched search benchmarking in a CPU Slurm allocation")
    if args.threads <= 0 or args.threads > int(os.environ.get("SLURM_CPUS_PER_TASK", "1")):
        raise ValueError("Thread count must be positive and fit allocated CPUs")
    # Set before NumPy/FAISS imports in run; isolated CLI does not load them earlier.
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    output = args.output if args.output.is_absolute() else root / args.output
    run(root, args.repeats, args.latency_queries, args.batch_size, args.threads, args.seed, output)
