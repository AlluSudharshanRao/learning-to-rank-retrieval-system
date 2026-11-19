"""Validate saved benchmark evidence and render portable search tradeoff figures."""

from __future__ import annotations

import csv
import logging
import math
import tempfile
from pathlib import Path

from ltr_system.data_download import sha256_file
from ltr_system.project import initialize, parser, read_json, write_json
from ltr_system.search_benchmark import summarize

LABELS = ("exact", "hnsw-ef1024", "hnsw-ef2048", "hnsw-ef4096")
RUNS = (
    "e5-exact-validation-default",
    *[f"e5-hnsw-validation-ef{ef}-b32-t4" for ef in (1024, 2048, 4096)],
)
DISPLAY = ("Exact", "HNSW\nef1024", "HNSW\nef2048", "HNSW\nef4096")
COLORS = ("#34465d", "#168b87", "#d78326", "#7462a7")


def evidence(benchmark_path: Path, quality_root: Path) -> dict:
    import numpy as np

    benchmark = read_json(benchmark_path)
    if (
        benchmark.get("status") != "matched CPU search benchmark completed"
        or set(benchmark["results"]) != set(LABELS)
        or benchmark["batch_size"] != 32
        or benchmark["k"] != 1000
        or benchmark["threads"] != 4
        or benchmark["repeats"] != 4
        or len(benchmark["schedule"]) != 4
        or benchmark["final_queries_read"] is not False
    ):
        raise ValueError("Expected the four-configuration matched benchmark protocol")
    ids = benchmark["query_ids_in_vector_order"]
    if len(ids) != benchmark["batch_queries"] or len(set(ids)) != len(ids):
        raise ValueError("Benchmark query population mismatch")
    sample = None
    for repeat, trial in enumerate(benchmark["schedule"]):
        if (
            trial["repeat"] != repeat
            or sorted(trial["batch_order"]) != list(range(len(ids)))
            or set(trial["algorithms"]) != set(LABELS)
            or len(trial["algorithms"]) != 4
        ):
            raise ValueError("Invalid matched workload schedule")
        current_sample = set(trial["latency_order"])
        if (
            len(current_sample) != benchmark["latency_unique_queries"]
            or len(trial["latency_order"]) != len(current_sample)
            or not current_sample <= set(range(len(ids)))
            or (sample is not None and current_sample != sample)
        ):
            raise ValueError("Latency query sample changed")
        sample = current_sample
    sources = {"benchmark_sha256": sha256_file(benchmark_path), "quality": {}}
    points = []
    for label, run_name in zip(LABELS, RUNS, strict=True):
        trials = benchmark["results"][label]["trials"]
        if len(trials) != 4 or [t["repeat"] for t in trials] != list(range(4)):
            raise ValueError("Missing or reordered measured trials")
        for trial in trials:
            if (
                len(trial["batch_call_seconds"]) != math.ceil(len(ids) / 32)
                or len(trial["single_query_seconds"]) != benchmark["latency_unique_queries"]
                or any(
                    not math.isfinite(v) or v <= 0
                    for key in ("batch_call_seconds", "single_query_seconds")
                    for v in trial[key]
                )
            ):
                raise ValueError("Invalid raw benchmark observations")
        recomputed = summarize(trials, len(ids))
        saved = benchmark["results"][label]["summary"]
        for key in (
            "batch_search_seconds_repetitions",
            "batch_queries_per_second_repetitions",
            "median_batch_queries_per_second",
            "single_query_p50_ms",
            "single_query_p95_ms",
        ):
            np.testing.assert_allclose(recomputed[key], saved[key], rtol=1e-10, atol=1e-10)
        if saved["single_query_observations"] != recomputed["single_query_observations"]:
            raise ValueError("Latency sample count mismatch")
        directory = quality_root / run_name
        manifest = read_json(directory / "manifest.json")
        evaluation_path = directory / "evaluation.json"
        expected_hash = (
            manifest.get("evaluation_sha256") or manifest["files"]["evaluation.json"]["sha256"]
        )
        if sha256_file(evaluation_path) != expected_hash:
            raise ValueError("Quality evaluation checksum mismatch")
        quality = read_json(evaluation_path)
        index_name = "e5-flat-ip-v1" if label == "exact" else "e5-hnsw-m32-efc200-v1"
        if (
            manifest["source"] != benchmark["population"]
            or manifest["query_manifest_sha256"]
            != benchmark["input_sha256"]["artifacts/queries/e5-validation-fp32/manifest.json"]
            or manifest["index_manifest_sha256"]
            != benchmark["input_sha256"][f"artifacts/indexes/{index_name}/manifest.json"]
            or quality["query_count"] != len(ids)
            or set(quality["per_query"]) != set(ids)
            or quality["inputs"]["queries"]["sha256"]
            != benchmark["population"]["query_file_sha256"]
            or manifest["k"] != 1000
            or manifest["batch_size"] != 32
            or manifest["threads"] != 4
            or (label != "exact" and manifest["efSearch"] != int(label.removeprefix("hnsw-ef")))
        ):
            raise ValueError("Quality and timing artifacts refer to different configurations")
        relevance = quality["metrics"]
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in relevance.values()):
            raise ValueError("Invalid relevance metric")
        points.append({"label": label, "search": recomputed, "relevance": relevance})
        sources["quality"][label] = {
            "manifest_sha256": sha256_file(directory / "manifest.json"),
            "evaluation_sha256": sha256_file(evaluation_path),
        }
    return {
        "schema_version": 1,
        "status": "source reports verified and summaries recomputed",
        "provenance": sources,
        "points": points,
        "protocol": {
            key: benchmark[key]
            for key in (
                "k",
                "threads",
                "batch_size",
                "batch_queries",
                "repeats",
                "latency_unique_queries",
                "seed",
            )
        },
        "hardware": {key: benchmark["metadata"][key] for key in ("cpu_model", "faiss", "numpy")},
        "scope": "warm CPU search only; quality from separate validation runs on the same index/query artifacts",
        "trial_dots": "observed trials, not confidence intervals",
    }


def render(benchmark: Path, quality_root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    report = evidence(benchmark, quality_root)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    points = report["points"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.hashsalt": "ltr-search-v1",
            "savefig.facecolor": "white",
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".plots-", dir=output.parent) as temporary:
        stage = Path(temporary) / "figures"
        stage.mkdir()

        def save(fig, name):
            fig.text(
                0.5,
                0.015,
                "8.84M passages | top 1,000 | 4 CPU threads | search only; excludes encoding and serving",
                ha="center",
                fontsize=8,
                color="#566477",
            )
            fig.savefig(stage / f"{name}.png", dpi=180, bbox_inches="tight")
            fig.savefig(stage / f"{name}.svg", bbox_inches="tight", metadata={"Date": None})
            svg = stage / f"{name}.svg"
            svg.write_text(
                "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())
                + "\n",
                encoding="utf-8",
                newline="\n",
            )
            plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 4.7))
        medians = [p["search"]["median_batch_queries_per_second"] for p in points]
        ax.bar(range(4), medians, color=COLORS, width=0.55, alpha=0.85)
        for i, point in enumerate(points):
            values = point["search"]["batch_queries_per_second_repetitions"]
            ax.scatter(
                i + np.linspace(-0.12, 0.12, len(values)),
                values,
                s=22,
                color="#172c43",
                zorder=4,
                label="Individual repetitions" if i == 0 else None,
            )
            ax.text(i, max(values) * 1.2, f"{medians[i]:.2f}", ha="center", fontweight="bold")
        ax.set_yscale("log")
        ax.set_ylim(1, max(medians) * 2.5)
        ax.set_xticks(range(4), DISPLAY)
        ax.set_ylabel("Queries/sec (log scale)")
        ax.set_title(
            "Batch search throughput\nMedian of four passes; 2,000 queries per pass, batch size 32",
            loc="left",
            pad=16,
        )
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="upper left")
        fig.subplots_adjust(bottom=0.20, top=0.78)
        save(fig, "throughput")

        fig, ax = plt.subplots(figsize=(9, 4.7))
        for i, point in enumerate(points):
            stats = point["search"]
            p50, p95 = stats["single_query_p50_ms"], stats["single_query_p95_ms"]
            ax.plot([p50, p95], [i, i], color=COLORS[i], lw=4, alpha=0.6)
            ax.scatter(
                p50, i, color=COLORS[i], s=55, marker="o", label="Pooled p50" if i == 0 else None
            )
            ax.scatter(
                p95, i, color=COLORS[i], s=65, marker="D", label="Pooled p95" if i == 0 else None
            )
            raw = [v["p95"] for v in stats["single_query_percentiles_by_repetition_ms"]]
            ax.scatter(
                raw,
                i + np.linspace(-0.12, 0.12, 4),
                color="#516071",
                marker="|",
                s=65,
                label="Per-repetition p95" if i == 0 else None,
            )
            ax.annotate(
                f"{p95:.2f} ms",
                (p95, i),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
            )
        ax.set_xscale("log")
        ax.set_xlim(2, 1500)
        ax.set_yticks(range(4), [v.replace("\n", " ") for v in DISPLAY])
        ax.invert_yaxis()
        ax.set_xlabel("Single-query search-call latency, ms (log scale)")
        ax.set_title(
            "Single-query search latency\n200 fixed queries repeated four times; warm, sequential calls",
            loc="left",
            pad=16,
        )
        ax.grid(axis="x", alpha=0.2)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        fig.subplots_adjust(bottom=0.22, top=0.78)
        save(fig, "latency")

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        for ax, key, title in zip(
            axes, ("recall@1000", "mrr@10"), ("Judged Recall@1000", "MRR@10"), strict=True
        ):
            for i, point in enumerate(points):
                ax.scatter(medians[i], point["relevance"][key], color=COLORS[i], s=65)
                ax.annotate(
                    DISPLAY[i].replace("\n", " "),
                    (medians[i], point["relevance"][key]),
                    xytext=(0, 9 if i != 1 else -16),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )
            ax.set_xscale("log")
            ax.set_xlim(2, 1800)
            ax.set_xlabel("Median batch queries/sec (log scale)")
            ax.set_ylabel(title)
            ax.grid(alpha=0.2)
            ax.margins(y=0.3)
        fig.suptitle("Retrieval quality versus search throughput", x=0.07, ha="left", fontsize=14)
        fig.subplots_adjust(bottom=0.22, top=0.83, wspace=0.3)
        save(fig, "quality_tradeoff")

        report["plotting_matplotlib"] = matplotlib.__version__
        write_json(stage / "summary.json", report)
        with (stage / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                [
                    "configuration",
                    "median_batch_qps",
                    "search_p50_ms",
                    "search_p95_ms",
                    "mrr_at_10",
                    "judged_recall_at_1000",
                ]
            )
            for point in points:
                stats = point["search"]
                writer.writerow(
                    [
                        point["label"],
                        stats["median_batch_queries_per_second"],
                        stats["single_query_p50_ms"],
                        stats["single_query_p95_ms"],
                        point["relevance"]["mrr@10"],
                        point["relevance"]["recall@1000"],
                    ]
                )
        stage.rename(output)
    return report


def main() -> None:
    command = parser("Verify saved search reports and generate benchmark figures.")
    command.add_argument(
        "--benchmark", type=Path, default=Path("artifacts/transfers/search-matched-v1.json")
    )
    command.add_argument(
        "--quality-root", type=Path, default=Path("artifacts/reports/hnsw-quality")
    )
    command.add_argument("--output", type=Path, default=Path("docs/figures/search-v1"))
    args = command.parse_args()
    root = initialize(args)

    def resolve(path):
        return path if path.is_absolute() else root / path

    report = render(resolve(args.benchmark), resolve(args.quality_root), resolve(args.output))
    logging.getLogger(__name__).info(
        "Generated verified plots: %s configurations", len(report["points"])
    )
