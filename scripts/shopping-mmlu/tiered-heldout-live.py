#!/usr/bin/env python3
"""Run the frozen Shopping MMLU policy once through live Dynamo/Switchyard endpoints."""

import argparse
import csv
import hashlib
import json
import re
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "data/shopping-mmlu/tiered-real-large-v1"
ROUTER = ROOT / "data/shopping-mmlu/tiered-real-large-router-v1"
CHECKPOINT = ROOT / "checkpoints/shopping-tiered-real-large-v1/prefill_router.pt"
COSTS = ROOT / "configs/prefill-router/shopping-costs-token-proxy.json"
SYSTEM_PROMPT = (
    "Answer the shopping multiple-choice question using the question and all four choices. "
    "Return only the letter A, B, C, or D for the best answer. Do not explain."
)
DEFAULT_URLS = {
    "efficient": "http://127.0.0.1:18000/v1",
    "capable": "http://127.0.0.1:18001/v1",
    "switchyard": "http://127.0.0.1:14000/v1",
}
REQUEST_MODELS = {"efficient": "efficient", "capable": "capable", "switchyard": "shopping"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    return values[min(len(values) - 1, int(fraction * len(values)))]


def parse_letter(text):
    normalized = (text or "").strip().upper()
    return normalized if normalized in ("A", "B", "C", "D") else None


def validate_frozen_inputs(rows, input_path, router_dir, checkpoint, policy_path):
    policy = json.loads(policy_path.read_text())
    if digest(checkpoint) != policy["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint hash mismatch")
    expected_test_hash = policy.get("test_jsonl_sha256_frozen_before_evaluation")
    if expected_test_hash:
        if digest(input_path) != expected_test_hash:
            raise ValueError("Frozen test JSONL hash mismatch")
    else:
        expected_csv_hash = policy["test_csv_sha256_frozen_before_evaluation"]
        if digest(router_dir / "test.csv") != expected_csv_hash:
            raise ValueError("Frozen test CSV hash mismatch")
        with (router_dir / "test.csv").open(newline="") as handle:
            frozen_questions = {row["question"] for row in csv.DictReader(handle)}
        live_questions = {row["question"] for row in rows}
        if live_questions != frozen_questions or len(rows) != len(frozen_questions):
            raise ValueError("Live test JSONL does not exactly match the frozen test question set")
    if any(row.get("split") != "test" for row in rows):
        raise ValueError("Only the untouched test split is allowed")
    return policy


def call(client, role, row):
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=REQUEST_MODELS[role],
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["question"]},
            ],
            temperature=0,
            max_tokens=16,
            seed=42,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = response.choices[0].message.content or ""
        usage = response.usage
        prediction = parse_letter(raw)
        return {
            "raw": raw,
            "prediction": prediction,
            "correct": prediction == row["answer"],
            "selected_model": response.model,
            "input_tokens": int(usage.prompt_tokens) if usage else None,
            "output_tokens": int(usage.completion_tokens) if usage else None,
            "latency_s": time.perf_counter() - started,
            "error": None,
        }
    except Exception as error:
        return {
            "raw": None,
            "prediction": None,
            "correct": False,
            "selected_model": None,
            "input_tokens": None,
            "output_tokens": None,
            "latency_s": time.perf_counter() - started,
            "error": f"{type(error).__name__}: {error}",
        }


def run_role(role, rows, url, concurrency, warmups):
    client = OpenAI(base_url=url, api_key="unused", max_retries=0, timeout=120)
    for index in range(warmups):
        result = call(client, role, rows[index % len(rows)])
        if result["error"]:
            raise RuntimeError(f"{role} warmup failed: {result['error']}")
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda row: call(client, role, row), rows))
    elapsed = time.perf_counter() - started
    errors = [result for result in results if result["error"]]
    if errors:
        raise RuntimeError(f"{role} had {len(errors)} errors; first: {errors[0]['error']}")
    return results, elapsed


def token_cost(result, model, prices):
    spec = prices[model]
    return (
        result["input_tokens"] * spec["input_usd_per_million"]
        + result["output_tokens"] * spec["output_usd_per_million"]
    ) / 1_000_000


def summarize(role, results, elapsed, prices):
    latencies = [row["latency_s"] for row in results]
    costs = []
    for row in results:
        model = role if role != "switchyard" else row["selected_model"]
        if model not in ("efficient", "capable"):
            raise ValueError(f"Switchyard did not expose its selected target: {model!r}")
        costs.append(token_cost(row, model, prices))
    return {
        "accuracy": statistics.mean(row["correct"] for row in results),
        "mean_latency_s": statistics.mean(latencies),
        "p50_latency_s": percentile(latencies, 0.50),
        "p95_latency_s": percentile(latencies, 0.95),
        "elapsed_s": elapsed,
        "successful_rps": len(results) / elapsed,
        "mean_input_tokens": statistics.mean(row["input_tokens"] for row in results),
        "mean_output_tokens": statistics.mean(row["output_tokens"] for row in results),
        "token_usd_per_request": statistics.mean(costs),
        "token_usd_per_1000_requests": 1000 * statistics.mean(costs),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--data-dir", type=Path, default=PILOT)
    parser.add_argument("--router-dir", type=Path, default=ROUTER)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--costs", type=Path, default=COSTS)
    parser.add_argument("--output-rows", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=4)
    parser.add_argument("--efficient-url", default=DEFAULT_URLS["efficient"])
    parser.add_argument("--capable-url", default=DEFAULT_URLS["capable"])
    parser.add_argument("--switchyard-url", default=DEFAULT_URLS["switchyard"])
    args = parser.parse_args()
    if args.output_rows.exists() or args.output_report.exists():
        raise ValueError("Choose fresh output paths; held-out evidence is never overwritten")

    input_path = args.input or args.data_dir / "test.jsonl"
    policy_path = args.policy or args.router_dir / "policy.json"
    rows = read_jsonl(input_path)
    policy = validate_frozen_inputs(
        rows, input_path, args.router_dir, args.checkpoint, policy_path
    )
    cost_spec = json.loads(args.costs.read_text())
    if "costs_sha256" in policy and digest(args.costs) != policy["costs_sha256"]:
        raise ValueError("Token prices changed after policy selection")
    prices = cost_spec["models"]
    urls = {
        "efficient": args.efficient_url,
        "capable": args.capable_url,
        "switchyard": args.switchyard_url,
    }

    results = {}
    elapsed = {}
    for role in ("efficient", "capable", "switchyard"):
        print(f"Running frozen held-out pass: {role}...", flush=True)
        results[role], elapsed[role] = run_role(
            role, rows, urls[role], args.concurrency, args.warmups
        )

    combined = []
    for index, source in enumerate(rows):
        record = {
            "example_id": source["example_id"],
            "difficulty": source["difficulty"],
            "task": source["task"],
            "question": source["question"],
            "answer": source["answer"],
        }
        for role in ("efficient", "capable", "switchyard"):
            record[role] = results[role][index]
        combined.append(record)

    summaries = {
        role: summarize(role, results[role], elapsed[role], prices)
        for role in ("efficient", "capable", "switchyard")
    }
    selected = Counter(row["selected_model"] for row in results["switchyard"])
    efficient_fraction = selected["efficient"] / len(rows)
    switchyard_cost = summaries["switchyard"]["token_usd_per_request"]
    capable_cost = summaries["capable"]["token_usd_per_request"]
    by_difficulty = {}
    for difficulty in ("easy", "hard"):
        indexes = [i for i, row in enumerate(rows) if row["difficulty"] == difficulty]
        by_difficulty[difficulty] = {
            "n": len(indexes),
            "switchyard_accuracy": statistics.mean(results["switchyard"][i]["correct"] for i in indexes),
            "efficient_fraction": statistics.mean(
                results["switchyard"][i]["selected_model"] == "efficient" for i in indexes
            ),
        }

    report = {
        "version": "shopping-tiered-heldout-live-v1",
        "split": "test",
        "n": len(rows),
        "policy_selection_split": "validation",
        "frozen_tolerance": policy.get("tolerance", policy.get("selected", {}).get("tolerance")),
        "checkpoint_sha256": digest(args.checkpoint),
        "policy_sha256": digest(policy_path),
        "test_jsonl_sha256": digest(input_path),
        "settings": {
            "system_prompt": SYSTEM_PROMPT,
            "temperature": 0,
            "seed": 42,
            "max_tokens": 16,
            "thinking": False,
            "concurrency": args.concurrency,
            "warmups_per_role": args.warmups,
        },
        "results": summaries,
        "switchyard_routing": dict(selected),
        "random_same_mix_expected_accuracy": (
            efficient_fraction * summaries["efficient"]["accuracy"]
            + (1 - efficient_fraction) * summaries["capable"]["accuracy"]
        ),
        "switchyard_accuracy_difference_vs_capable": (
            summaries["switchyard"]["accuracy"] - summaries["capable"]["accuracy"]
        ),
        "switchyard_token_savings_vs_capable_fraction": 1 - switchyard_cost / capable_cost,
        "by_difficulty": by_difficulty,
        "pricing": cost_spec,
        "note": (
            "One-shot live held-out evaluation through Dynamo and Switchyard. "
            "The checkpoint, tolerance, prompts, prices, and test set were frozen before this run."
        ),
    }

    args.output_rows.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_rows.write_text("".join(json.dumps(row) + "\n" for row in combined))
    args.output_report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
