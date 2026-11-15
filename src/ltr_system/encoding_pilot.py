"""Bounded full-precision E5 encoding pilot on verified prefix or sampled passages."""

from __future__ import annotations

import itertools
import json
import logging
import os
import socket
import statistics
import tempfile
import time
from pathlib import Path

from ltr_system.commands import _cuda
from ltr_system.data_download import sha256_file
from ltr_system.encoding import masked_mean_pool
from ltr_system.project import (
    configure_hub,
    encoder_config,
    initialize,
    parser,
    read_json,
    write_json,
)

LOGGER = logging.getLogger(__name__)


def load_prefix(path: Path, count: int) -> tuple[list[str], list[str]]:
    if count <= 0:
        raise ValueError("Passage count must be positive")
    ids, texts = [], []
    with path.open(encoding="utf-8") as stream:
        for line in itertools.islice(stream, count):
            record = json.loads(line)
            pid, text = record["id"], record["contents"]
            if not isinstance(pid, str) or not pid.isascii() or not pid.isdecimal():
                raise ValueError("Expected numeric string passage ID")
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Expected nonempty passage text")
            ids.append(pid)
            texts.append("passage: " + text)
    if len(ids) != count or len(set(ids)) != count:
        raise ValueError("Insufficient or duplicate prefix passage IDs")
    return ids, texts


def load_sample(profile: Path, count: int, config: dict) -> tuple[list[str], list[str], dict]:
    metadata = read_json(profile / "manifest.json")
    if any(metadata[key] != config[key] for key in ("repo_id", "revision", "max_length")):
        raise ValueError("Sample profile differs from pinned encoder configuration")
    if metadata["statistics"]["count"] != count:
        raise ValueError("Passage count must match the complete saved sample")
    path = profile / "sample.jsonl"
    digest = sha256_file(path)
    if digest != metadata["sample_sha256"]:
        raise ValueError("Saved sample hash mismatch")
    ids, texts = load_prefix(path, count)
    return (
        ids,
        texts,
        {
            "selection": "saved corpus-wide sample in source order",
            "sample_sha256": digest,
            "profile_manifest_sha256": sha256_file(profile / "manifest.json"),
            "source_corpus_manifest_sha256": metadata["source_manifest_sha256"],
        },
    )


def main() -> None:
    command = parser("Measure E5 encoding on a verified corpus prefix or saved sample.")
    command.add_argument("--passages", type=int, default=10_000)
    command.add_argument("--batch-size", type=int, default=64)
    command.add_argument("--sample-profile", type=Path)
    command.add_argument("--repeats", type=int, default=1)
    args = command.parse_args()
    root = initialize(args)
    if not 0 < args.passages <= 100_000 or args.batch_size <= 0 or args.repeats <= 0:
        raise ValueError("Expected 1..100000 passages and a positive batch size")
    config = encoder_config()
    if args.sample_profile:
        profile = args.sample_profile
        if not profile.is_absolute():
            profile = root / profile
        ids, texts, source_identity = load_sample(profile, args.passages, config)
        tag = f"sample-{source_identity['sample_sha256'][:12]}-"
    else:
        corpus = root / "data/processed/msmarco-passage-v1-jsonl"
        first = read_json(corpus / "manifest.json")["shards"][0]
        if Path(first["name"]).name != first["name"]:
            raise ValueError("Invalid shard filename")
        path = corpus / "shards" / first["name"]
        if sha256_file(path) != first["sha256"]:
            raise ValueError("Pilot shard hash mismatch")
        ids, texts = load_prefix(path, args.passages)
        source_identity = {
            "selection": "corpus prefix",
            "source_shard_sha256": first["sha256"],
            "source_corpus_manifest_sha256": sha256_file(corpus / "manifest.json"),
        }
        tag = ""
    model_manifest = read_json(root / "artifacts/models/e5-small-v2.json")
    if any(model_manifest[key] != config[key] for key in ("repo_id", "revision")):
        raise ValueError("Model manifest differs from pinned encoder configuration")
    destination = (
        root
        / f"artifacts/encoding-pilots/e5-{tag}n{args.passages}-b{args.batch_size}-r{args.repeats}"
    )
    if destination.exists():
        raise FileExistsError(
            f"Pilot exists; preserve it and choose another configuration: {destination}"
        )
    configure_hub(root, offline=True)
    torch = _cuda()
    import numpy as np
    import torch.nn.functional as functional
    from transformers import AutoModel, AutoTokenizer

    snapshot = Path(model_manifest["snapshot_path"])
    LOGGER.info("Loading pinned offline encoder")
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False
    )
    model = (
        AutoModel.from_pretrained(
            snapshot,
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        )
        .float()
        .to("cuda:0")
        .eval()
    )

    def encode(batch_texts):
        tokens = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=config["max_length"],
            return_tensors="pt",
        ).to("cuda:0")
        with torch.inference_mode():
            hidden = model(**tokens).last_hidden_state
            output = functional.normalize(
                masked_mean_pool(hidden, tokens["attention_mask"]), p=2, dim=1
            )
        return output.cpu().numpy()

    LOGGER.info("Warming up with two batches")
    for _ in range(2):
        encode(texts[: args.batch_size])
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    vectors = np.empty((len(ids), config["dimension"]), dtype=np.float32)
    durations = []
    for repeat in range(args.repeats):
        LOGGER.info("Starting measured repetition %s/%s", repeat + 1, args.repeats)
        started = time.perf_counter()
        for start in range(0, len(ids), args.batch_size):
            end = min(start + args.batch_size, len(ids))
            encoded = encode(texts[start:end])
            if encoded.shape != (end - start, config["dimension"]):
                raise ValueError("Unexpected encoder output shape")
            vectors[start:end] = encoded
            if (start // args.batch_size) % 50 == 0:
                LOGGER.info("Encoded %s/%s passages", end, len(ids))
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)
    seconds = statistics.median(durations)
    if not np.isfinite(vectors).all():
        raise ValueError("Nonfinite embeddings")
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5, rtol=1e-5)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pilot-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "output"
        stage.mkdir()
        np.save(stage / "vectors.npy", vectors, allow_pickle=False)
        write_json(stage / "passage_ids.json", {"passage_ids": ids})
        report = {
            "schema_version": 1,
            "pilot_version": 2,
            "scope": "encoding pilot; not retrieval quality",
            "source": source_identity,
            "repeats": args.repeats,
            "encoding_seconds_repetitions": durations,
            "summary_statistic": "median repetition duration",
            "repo_id": config["repo_id"],
            "revision": config["revision"],
            "dimension": config["dimension"],
            "max_length": config["max_length"],
            "pooling": "attention-masked mean",
            "normalization": "L2",
            "prefix": "passage: ",
            "precision": "float32; no autocast",
            "batch_size": args.batch_size,
            "passages": len(ids),
            "warmup_batches": 2,
            "encoding_seconds": seconds,
            "passages_per_second": len(ids) / seconds,
            "timing_scope": "tokenization, transfers, forward, pooling, CPU vector collection; excludes loading/warmup/saving",
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "job_id": os.environ["SLURM_JOB_ID"],
            "host": socket.gethostname(),
            "cuda_build": torch.version.cuda,
            "files": {
                p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size}
                for p in stage.iterdir()
            },
        }
        write_json(stage / "manifest.json", report)
        stage.rename(destination)
    LOGGER.info("Pilot saved: %s; passages/sec %.3f", destination, len(ids) / seconds)


if __name__ == "__main__":
    main()
