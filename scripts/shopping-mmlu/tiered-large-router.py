#!/usr/bin/env python3
"""Freeze large Shopping MMLU train/validation correctness labels for router training."""

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/shopping-mmlu/tiered-real-large-v1"
OUTPUT = ROOT / "data/shopping-mmlu/tiered-real-large-router-v1"
MODELS = ("efficient", "capable")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main():
    if OUTPUT.exists():
        raise ValueError("Choose a fresh router-data version; frozen artifacts are never overwritten")
    source_manifest = json.loads((SOURCE / "manifest.json").read_text())
    predictions = read_jsonl(SOURCE / "predictions-development.jsonl")
    if len(predictions) != 4000:
        raise ValueError("Expected exactly 3,500 train and 500 validation predictions")
    expected_ids = {
        row["example_id"]
        for split in ("train", "validation")
        for row in read_jsonl(SOURCE / f"{split}.jsonl")
    }
    if {row["example_id"] for row in predictions} != expected_ids:
        raise ValueError("Prediction IDs do not match frozen train and validation data")
    if any(
        row[model]["error"] not in (None, "invalid_or_truncated_answer")
        for row in predictions
        for model in MODELS
    ):
        raise ValueError("Transport/request failures cannot become router labels")

    OUTPUT.mkdir(parents=True)
    hashes = {}
    for split in ("train", "validation"):
        rows = [row for row in predictions if row["split"] == split]
        path = OUTPUT / f"{split}.csv"
        with path.open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["question", "model", "isCorrect"])
            writer.writeheader()
            for row in rows:
                for model in MODELS:
                    writer.writerow({
                        "question": row["question"],
                        "model": model,
                        "isCorrect": int(row[model]["correct"]),
                    })
        hashes[split] = digest(path)

    manifest = {
        "version": "shopping-mmlu-tiered-real-large-router-v1",
        "source_manifest_sha256": digest(SOURCE / "manifest.json"),
        "source_predictions_sha256": digest(SOURCE / "predictions-development.jsonl"),
        "frozen_test_jsonl_sha256": source_manifest["file_sha256"]["test"],
        "counts": {"train": 3500, "validation": 500, "test_reserved": 500},
        "difficulty_counts": source_manifest["difficulty_counts_by_split"],
        "csv_sha256": hashes,
        "policy": (
            "Train only on train.csv; choose lambda/tolerance only on validation.csv; "
            "evaluate the frozen policy once on the reserved source test.jsonl."
        ),
        "label_definition": (
            "Exact A/B/C/D correctness. Invalid or truncated model output is incorrect; "
            "transport failures are forbidden."
        ),
    }
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
