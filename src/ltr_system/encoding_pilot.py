"""Bounded full-precision E5 passage-encoding pilot on a verified corpus prefix."""

from __future__ import annotations

import itertools
import json
import logging
import os
import socket
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


def main() -> None:
    command = parser("Measure E5 passage encoding on a verified corpus-prefix pilot.")
    command.add_argument("--passages", type=int, default=10_000)
    command.add_argument("--batch-size", type=int, default=64)
    args = command.parse_args()
    root = initialize(args)
    if not 0 < args.passages <= 100_000 or args.batch_size <= 0:
        raise ValueError("Expected 1..100000 passages and a positive batch size")
    corpus = root / "data/processed/msmarco-passage-v1-jsonl"
    corpus_manifest = read_json(corpus / "manifest.json")
    first = corpus_manifest["shards"][0]
    if Path(first["name"]).name != first["name"]:
        raise ValueError("Invalid shard filename")
    path = corpus / "shards" / first["name"]
    if sha256_file(path) != first["sha256"]:
        raise ValueError("Pilot shard hash mismatch")
    ids, texts = load_prefix(path, args.passages)
    config = encoder_config()
    model_manifest = read_json(root / "artifacts/models/e5-small-v2.json")
    if any(model_manifest[key] != config[key] for key in ("repo_id", "revision")):
        raise ValueError("Model manifest differs from pinned encoder configuration")
    destination = root / f"artifacts/encoding-pilots/e5-n{args.passages}-b{args.batch_size}"
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
    seconds = time.perf_counter() - started
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
            "scope": "prefix encoding pilot; not retrieval quality",
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
            "source_shard_sha256": first["sha256"],
            "source_corpus_manifest_sha256": sha256_file(corpus / "manifest.json"),
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
