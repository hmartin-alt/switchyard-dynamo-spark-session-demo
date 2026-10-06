#!/usr/bin/env python3
"""Validation-only token-priced lambda sweep for the frozen tiered shopping router."""
import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
MODELS = ["efficient", "capable"]
DEFAULT_EXPERIMENT = ROOT / "data/shopping-mmlu/tiered-real-large-router-v1"
DEFAULT_PREDICTIONS = ROOT / "data/shopping-mmlu/tiered-real-large-v1/predictions-development.jsonl"
DEFAULT_CHECKPOINT = ROOT / "checkpoints/shopping-tiered-real-large-v1/prefill_router.pt"
DEFAULT_COSTS = ROOT / "configs/prefill-router/shopping-costs-token-proxy.json"

LLM_ROUTER_SRC = Path.home() / "llm-router-v3" / "src"
if str(LLM_ROUTER_SRC) not in sys.path:
    sys.path.insert(0, str(LLM_ROUTER_SRC))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def finite_nonnegative(value, name):
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def fixed_token_costs(predictions, spec):
    training = [row for row in predictions if row["split"] == "train"]
    if not training:
        raise ValueError("No training rows available for token medians")
    medians, costs = {}, []
    for model in MODELS:
        medians[model] = {
            "input_tokens": statistics.median(row[model]["input_tokens"] for row in training),
            "output_tokens": statistics.median(row[model]["output_tokens"] for row in training),
        }
        price = spec["models"][model]
        input_price = finite_nonnegative(price["input_usd_per_million"], f"{model} input price")
        output_price = finite_nonnegative(price["output_usd_per_million"], f"{model} output price")
        costs.append((medians[model]["input_tokens"] * input_price +
                      medians[model]["output_tokens"] * output_price) / 1_000_000)
    costs = np.asarray(costs, dtype=np.float64)
    if not 0 < costs[0] < costs[1]:
        raise ValueError("Efficient fixed token cost must be positive and cheaper than capable")
    return costs, medians


def choose(probabilities, normalized_costs, lambda_accuracy):
    scores = lambda_accuracy * probabilities - (1-lambda_accuracy) * normalized_costs
    # np.argmax preserves efficient-first tie semantics because efficient is index 0.
    return np.argmax(scores, axis=1)


def tolerance_for_lambda(lambda_accuracy, normalized_costs):
    if lambda_accuracy == 0:
        return 1.0
    return min(1.0, (1-lambda_accuracy) / lambda_accuracy *
               (normalized_costs[1] - normalized_costs[0]))


def sweep(probabilities, outcomes, costs):
    normalized = costs / costs[1]
    cost_delta = normalized[1] - normalized[0]
    probability_gap = probabilities[:, 1] - probabilities[:, 0]
    switch_lambdas = cost_delta / (cost_delta + probability_gap)
    candidates = np.unique(np.concatenate((
        np.linspace(0, 1, 1001),
        switch_lambdas[(switch_lambdas >= 0) & (switch_lambdas <= 1)],
    )))
    capable_accuracy = float(outcomes[:, 1].mean())
    rows, seen = [], set()
    for lambda_accuracy in candidates:
        lambda_accuracy = float(lambda_accuracy)
        choices = choose(probabilities, normalized, lambda_accuracy)
        signature = choices.tobytes()
        if signature in seen:
            continue
        seen.add(signature)
        accuracy = float(outcomes[np.arange(len(choices)), choices].mean())
        request_cost = float(costs[choices].mean())
        efficient_fraction = float((choices == 0).mean())
        rows.append({
            "lambda_accuracy": lambda_accuracy,
            "equivalent_switchyard_tolerance": tolerance_for_lambda(lambda_accuracy, normalized),
            "accuracy": accuracy,
            "accuracy_loss_vs_capable": capable_accuracy - accuracy,
            "efficient_fraction": efficient_fraction,
            "capable_fraction": 1-efficient_fraction,
            "token_usd_per_request": request_cost,
            "token_usd_per_1000_requests": 1000*request_cost,
            "token_savings_vs_capable_fraction": 1-request_cost/costs[1],
        })
    return sorted(rows, key=lambda row: (row["token_usd_per_request"], -row["accuracy"]))


def pareto(rows):
    frontier, best_accuracy = [], -1.0
    for row in sorted(rows, key=lambda item: (item["token_usd_per_request"], -item["accuracy"])):
        if row["accuracy"] > best_accuracy + 1e-12:
            frontier.append(row)
            best_accuracy = row["accuracy"]
    return frontier


def render_plot(rows, frontier, model_accuracy, costs, selected, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot([row["token_usd_per_1000_requests"] for row in rows],
            [100*row["accuracy"] for row in rows], ".", color="#a7a9ac", alpha=.55,
            label="All λ policies")
    ax.plot([row["token_usd_per_1000_requests"] for row in frontier],
            [100*row["accuracy"] for row in frontier], "o-", color="#76b900", linewidth=2,
            label="Router Pareto frontier")
    for index, model in enumerate(MODELS):
        ax.scatter(1000*costs[index], 100*model_accuracy[index], marker="s", color="black", zorder=5)
        ax.annotate(f"{model}-only", (1000*costs[index], 100*model_accuracy[index]),
                    xytext=(0, 9), textcoords="offset points", ha="center", fontsize=9)
    ax.scatter(selected["token_usd_per_1000_requests"], 100*selected["accuracy"],
               marker="D", s=70, color="#1f77b4", zorder=6, label="Selected ≤2 pp loss")
    ax.set(xlabel="Token-price proxy (USD per 1,000 requests)", ylabel="Validation accuracy (%)",
           title="Switchyard shopping router: accuracy vs token-priced cost")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.text(.5, .015, "Backend token charges only; excludes encoder and always-on fleet cost",
             ha="center", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--costs", type=Path, default=DEFAULT_COSTS)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--max-accuracy-loss", type=float, default=.02)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise ValueError("Choose a fresh output directory")
    if not 0 <= args.max_accuracy_loss <= 1:
        raise ValueError("max accuracy loss must be in [0, 1]")

    manifest = json.loads((args.experiment_dir / "manifest.json").read_text())
    policy_path = args.experiment_dir / "policy.json"
    checkpoint_path = args.experiment_dir / "checkpoint.json"
    if policy_path.exists():
        frozen_checkpoint = json.loads(policy_path.read_text())
    elif checkpoint_path.exists():
        frozen_checkpoint = json.loads(checkpoint_path.read_text())
    else:
        raise ValueError("Experiment needs policy.json or checkpoint.json with a frozen checkpoint hash")
    predictions = read_jsonl(args.predictions)
    if digest(args.predictions) != manifest["source_predictions_sha256"]:
        raise ValueError("Predictions differ from the frozen experiment manifest")
    if digest(args.checkpoint) != frozen_checkpoint["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from the frozen checkpoint manifest")
    if digest(args.experiment_dir / "validation.csv") != manifest["csv_sha256"]["validation"]:
        raise ValueError("Validation CSV differs from the frozen split")
    spec = json.loads(args.costs.read_text())
    if spec.get("basis") != "api_proxy" or not spec.get("source"):
        raise ValueError("Token sweep requires a documented api_proxy cost spec")
    costs, medians = fixed_token_costs(predictions, spec)

    from model_router_toolkit.evaluate import _build_shared_features, _load_labels
    from model_router_toolkit.prefill.extract import extract_from_checkpoint
    from model_router_toolkit.prefill.trunk import predict_proba, reconstruct_trunk

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model_names = checkpoint["model_names"]
    if model_names != MODELS:
        raise ValueError(f"Expected model order {MODELS}, got {model_names}")
    questions, label_map = _load_labels(args.experiment_dir / "validation.csv")
    outcomes = np.zeros((len(questions), len(MODELS)), dtype=np.int8)
    for row_index, (_, labels) in enumerate(label_map.items()):
        for model_index, model in enumerate(MODELS):
            outcomes[row_index, model_index] = labels.get(model, 0)
    features = extract_from_checkpoint(checkpoint, questions, device=args.device,
                                       batch_size=4, cache_dir="cache/")
    shared = _build_shared_features(checkpoint, features, model_names)
    probabilities = predict_proba(reconstruct_trunk(checkpoint, device=args.device),
                                  shared, device=args.device)

    rows = sweep(probabilities, outcomes, costs)
    frontier = pareto(rows)
    model_accuracy = outcomes.mean(axis=0)
    eligible = [row for row in rows
                if row["accuracy_loss_vs_capable"] <= args.max_accuracy_loss + 1e-12]
    if not eligible:
        raise ValueError("No policy satisfies the quality constraint")
    selected = min(eligible, key=lambda row: (row["token_usd_per_request"], -row["accuracy"]))
    report = {
        "split": "validation",
        "n_examples": len(questions),
        "checkpoint_sha256": digest(args.checkpoint),
        "predictions_sha256": digest(args.predictions),
        "validation_csv_sha256": digest(args.experiment_dir / "validation.csv"),
        "costs": spec,
        "training_token_medians": medians,
        "fixed_token_usd_per_request": dict(zip(MODELS, costs.tolist())),
        "model_accuracy": dict(zip(MODELS, model_accuracy.tolist())),
        "max_accuracy_loss": args.max_accuracy_loss,
        "selected_quality_constrained_point": selected,
        "router_pareto_frontier": frontier,
        "points": rows,
        "native_policy_mapping": "With two fixed model costs, the normalized lambda score maps exactly to Switchyard efficient-first tolerance.",
        "note": "Validation-only policy selection. Held-out test outcomes were not read. Token-price proxy excludes encoder and idle fleet costs.",
    }
    args.out_dir.mkdir(parents=True)
    (args.out_dir / "sweep.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.out_dir / "sweep.csv").open("x", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "policy.json").write_text(json.dumps({
        "selection_split": "validation",
        "lambda_accuracy": selected["lambda_accuracy"],
        "equivalent_switchyard_tolerance": selected["equivalent_switchyard_tolerance"],
        "checkpoint_sha256": digest(args.checkpoint),
        "costs_sha256": digest(args.costs),
    }, indent=2) + "\n")
    render_plot(rows, frontier, model_accuracy, costs, selected, args.out_dir / "pareto.png")
    print(json.dumps({key: value for key, value in report.items() if key != "points"}, indent=2))


if __name__ == "__main__":
    main()
