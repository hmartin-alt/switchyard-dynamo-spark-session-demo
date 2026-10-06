#!/usr/bin/env python3
"""Inspect the frozen router locally; no backend calls or retraining required."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path.home() / "llm-router-v3/src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", help="Full question including answer choices")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", type=Path, help="Save full scores as JSON (new file only)")
    args = parser.parse_args()
    experiment = ROOT / "data/shopping-mmlu/tiered-real-large-router-v1"
    checkpoint_path = ROOT / "checkpoints/shopping-tiered-real-large-v1/prefill_router.pt"
    policy = json.loads((experiment / "policy.json").read_text())
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if digest != policy["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from frozen policy")
    if args.output and args.output.exists():
        raise ValueError("Choose a new output file; existing results will not be overwritten")
    if args.question:
        rows = [{"question": args.question}]
    else:
        # First two rows per tier in file order, without filtering by correctness.
        source = [json.loads(line) for line in
                  (experiment / "heldout-live-token-v1.jsonl").read_text().splitlines() if line.strip()]
        rows = [row for tier in ("easy", "hard")
                for row in [item for item in source if item["difficulty"] == tier][:2]]

    import torch
    from model_router_toolkit.evaluate import _build_shared_features
    from model_router_toolkit.prefill.extract import extract_from_checkpoint
    from model_router_toolkit.prefill.trunk import predict_proba, reconstruct_trunk

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    names = checkpoint["model_names"]
    if names != ["efficient", "capable"]:
        raise ValueError(f"Unexpected target order: {names}")
    features = extract_from_checkpoint(checkpoint, [row["question"] for row in rows],
                                       device=args.device, batch_size=4,
                                       cache_dir=str(ROOT / "cache"))
    shared = _build_shared_features(checkpoint, features, names)
    probabilities = predict_proba(reconstruct_trunk(checkpoint, device=args.device),
                                  shared, device=args.device)
    tolerance = policy["tolerance"]
    results = []
    print(f"\nFROZEN ROUTER SCORES | tolerance={tolerance:.6f}")
    print("Local checkpoint predictions; recorded live answers are labeled separately.")
    for row, values in zip(rows, probabilities):
        efficient, capable = map(float, values)
        threshold = max(efficient, capable) - tolerance
        selected = "efficient" if efficient >= threshold else "capable"
        result = {
            "question": row["question"],
            "predicted_correctness": dict(zip(names, (efficient, capable))),
            "eligibility_threshold": threshold,
            "selected_model": selected,
        }
        print(f"\n{row['question']}")
        print(f"  E4B: {efficient:.1%} | 31B: {capable:.1%}")
        print(f"  Eligibility threshold: {threshold:.1%} -> {selected}")
        if "switchyard" in row:
            live = row["switchyard"]
            result["recorded_live_result"] = live
            result["known_answer"] = row["answer"]
            result["matches_recorded_route"] = selected == live["selected_model"]
            print(f"  Recorded live: {live['selected_model']}, answer={live['prediction']}, "
                  f"correct={live['correct']}")
            if not result["matches_recorded_route"]:
                print("  NOTE: local selection differs from recorded live selection.")
        results.append(result)
    if args.output:
        with args.output.open("x") as handle:
            json.dump({"source": "local frozen-checkpoint inference",
                       "checkpoint_sha256": digest, "tolerance": tolerance,
                       "results": results}, handle, indent=2)
            handle.write("\n")
        print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
