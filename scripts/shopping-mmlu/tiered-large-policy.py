#!/usr/bin/env python3
"""Freeze the validation-selected large Shopping MMLU cost-aware policy."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/shopping-mmlu/tiered-real-large-v1"
ROUTER = ROOT / "data/shopping-mmlu/tiered-real-large-router-v1"
SWEEP = ROOT / "data/shopping-mmlu/tiered-real-large-token-pareto-v1"
CHECKPOINT = ROOT / "checkpoints/shopping-tiered-real-large-v1/prefill_router.pt"
COSTS = ROOT / "configs/prefill-router/shopping-costs-token-proxy.json"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    output = ROUTER / "policy.json"
    if output.exists():
        raise ValueError("Large policy is already frozen")
    dataset = json.loads((DATA / "manifest.json").read_text())
    router = json.loads((ROUTER / "manifest.json").read_text())
    selected = json.loads((SWEEP / "policy.json").read_text())
    sweep = json.loads((SWEEP / "sweep.json").read_text())
    point = sweep["selected_quality_constrained_point"]
    checkpoint_sha = digest(CHECKPOINT)
    if selected["checkpoint_sha256"] != checkpoint_sha:
        raise ValueError("Selected policy does not match the frozen checkpoint")
    if router["frozen_test_jsonl_sha256"] != dataset["file_sha256"]["test"]:
        raise ValueError("Reserved test hash mismatch")
    if digest(COSTS) != selected["costs_sha256"]:
        raise ValueError("Token prices changed after validation selection")
    policy = {
        "version": "shopping-mmlu-tiered-real-large-policy-v1",
        "selection_split": "validation",
        "quality_constraint": {"max_absolute_accuracy_loss_vs_capable": 0.02},
        "lambda_accuracy": selected["lambda_accuracy"],
        "tolerance": selected["equivalent_switchyard_tolerance"],
        "validation_result": point,
        "checkpoint_sha256": checkpoint_sha,
        "train_csv_sha256": router["csv_sha256"]["train"],
        "validation_csv_sha256": router["csv_sha256"]["validation"],
        "test_jsonl_sha256_frozen_before_evaluation": dataset["file_sha256"]["test"],
        "costs_sha256": digest(COSTS),
        "rule": (
            "Score=lambda*predicted_correctness-(1-lambda)*normalized_token_cost; "
            "the equivalent two-model Switchyard policy chooses efficient when its predicted "
            "correctness is within tolerance of the best target."
        ),
        "note": "Frozen before any model or router evaluation on the 500-row test split.",
    }
    output.write_text(json.dumps(policy, indent=2) + "\n")
    print(json.dumps(policy, indent=2))


if __name__ == "__main__":
    main()
