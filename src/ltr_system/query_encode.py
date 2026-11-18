"""Encode only frozen validation queries with the passage-compatible E5 checkpoint."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

from ltr_system.commands import _cuda
from ltr_system.data_download import sha256_file
from ltr_system.data_prepare import rows
from ltr_system.dense_encode import check_vectors
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


def validation_inputs(directory: Path) -> tuple[list[str], list[str], dict]:
    manifest_hash = sha256_file(directory / "manifest.json")
    manifest = read_json(directory / "manifest.json")
    query_file = directory / "validation.queries.tsv"
    query_hash = sha256_file(query_file)
    if query_hash != manifest["outputs"]["validation.queries.tsv"]["sha256"]:
        raise ValueError("Validation queries differ from split manifest")
    ids, texts = [], []
    for _, (qid, text) in rows(query_file, 2):
        ids.append(qid)
        texts.append("query: " + text)
    if (
        not ids
        or len(set(ids)) != len(ids)
        or len(ids) != manifest["counts"]["actual_validation_queries"]
    ):
        raise ValueError("Validation query count or uniqueness mismatch")
    return (
        ids,
        texts,
        {
            "split_manifest_sha256": manifest_hash,
            "query_file_sha256": query_hash,
            "split": "validation",
        },
    )


def main() -> None:
    command = parser("Encode frozen validation queries with pinned offline E5.")
    command.add_argument("--batch-size", type=int, default=64)
    args = command.parse_args()
    root = initialize(args)
    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive")
    destination = root / "artifacts/queries/e5-validation-fp32"
    if destination.exists():
        raise FileExistsError(f"Query artifacts already exist: {destination}")
    split_directory = root / "data/processed/msmarco-passage-v1-splits"
    ids, texts, identity = validation_inputs(split_directory)
    config = encoder_config()
    checkpoint = read_json(root / "artifacts/models/e5-small-v2.json")
    passage_config_path = root / "artifacts/embeddings/e5-small-v2-fp32/configuration.json"
    passage_config = read_json(passage_config_path)
    for key in ("repo_id", "revision", "dimension", "max_length"):
        if passage_config[key] != config[key]:
            raise ValueError("Passage encoding settings differ from configured encoder")
    if any(checkpoint[key] != config[key] for key in ("repo_id", "revision")):
        raise ValueError("Checkpoint identity mismatch")
    if (
        passage_config["precision"],
        passage_config["pooling"],
        passage_config["normalization"],
    ) != ("float32", "masked mean", "L2"):
        raise ValueError("Unsupported passage encoding semantics")
    configure_hub(root, offline=True)
    torch = _cuda()
    import numpy as np
    import torch.nn.functional as functional
    from transformers import AutoModel, AutoTokenizer

    LOGGER.info("Loading validation query encoder on %s", torch.cuda.get_device_name(0))
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint["snapshot_path"], local_files_only=True, trust_remote_code=False
    )
    model = (
        AutoModel.from_pretrained(
            checkpoint["snapshot_path"],
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        )
        .float()
        .to("cuda:0")
        .eval()
    )
    vectors = np.empty((len(ids), config["dimension"]), np.float32)
    for start in range(0, len(ids), args.batch_size):
        end = min(start + args.batch_size, len(ids))
        tokens = tokenizer(
            texts[start:end],
            padding=True,
            truncation=True,
            max_length=config["max_length"],
            return_tensors="pt",
        ).to("cuda:0")
        with torch.inference_mode():
            hidden = model(**tokens).last_hidden_state
            encoded = functional.normalize(
                masked_mean_pool(hidden, tokens["attention_mask"]), p=2, dim=1
            )
        batch = encoded.cpu().numpy()
        check_vectors(batch, (end - start, config["dimension"]))
        vectors[start:end] = batch
    check_vectors(vectors, (len(ids), config["dimension"]))
    _, _, current_identity = validation_inputs(split_directory)
    if current_identity != identity:
        raise ValueError("Validation inputs changed during query encoding")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".queries-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "output"
        stage.mkdir()
        np.save(stage / "vectors.npy", vectors, allow_pickle=False)
        write_json(stage / "query_ids.json", {"query_ids": ids})
        report = {
            "schema_version": 1,
            "status": "validation queries encoded",
            "source": identity,
            "passage_configuration_sha256": sha256_file(passage_config_path),
            "encoder": config,
            "prefix": "query: ",
            "precision": "float32",
            "pooling": "masked mean",
            "normalization": "L2",
            "queries": len(ids),
            "batch_size": args.batch_size,
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "job_id": os.environ["SLURM_JOB_ID"],
            "final_queries_read": False,
            "judgments_read": False,
            "files": {
                p.name: {"sha256": sha256_file(p), "bytes": p.stat().st_size}
                for p in stage.iterdir()
            },
        }
        write_json(stage / "manifest.json", report)
        stage.rename(destination)
    LOGGER.info("Saved %s validation query vectors: %s", len(ids), destination)


if __name__ == "__main__":
    main()
