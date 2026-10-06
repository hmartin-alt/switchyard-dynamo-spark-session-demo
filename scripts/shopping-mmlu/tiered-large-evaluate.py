#!/usr/bin/env python3
"""Collect concurrent direct-model labels for large tiered train/validation data."""

import argparse
import hashlib
import json
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data/shopping-mmlu/tiered-real-large-v1"
SYSTEM_PROMPT = (
    "Answer the shopping multiple-choice question using the question and all four choices. "
    "Return only the letter A, B, C, or D for the best answer. Do not explain."
)
MODELS = {"efficient": "efficient", "capable": "capable"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def call(client, model, row):
    started = time.perf_counter()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["question"]},
            ],
            temperature=0,
            max_tokens=16,
            seed=42,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        choice = response.choices[0]
        raw = choice.message.content or ""
        prediction = raw.strip()
        error = None
        if prediction not in ("A", "B", "C", "D") or choice.finish_reason != "stop":
            error = "invalid_or_truncated_answer"
        return {
            "prediction": prediction if error is None else None,
            "raw": raw,
            "correct": error is None and prediction == row["answer"],
            "finish_reason": choice.finish_reason,
            "input_tokens": response.usage.prompt_tokens if response.usage else None,
            "output_tokens": response.usage.completion_tokens if response.usage else None,
            "latency_s": time.perf_counter() - started,
            "error": error,
        }
    except Exception as error:
        return {
            "prediction": None,
            "raw": None,
            "correct": False,
            "finish_reason": None,
            "input_tokens": None,
            "output_tokens": None,
            "latency_s": time.perf_counter() - started,
            "error": f"{type(error).__name__}: {error}",
        }


def run_model(client, model, rows, concurrency):
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(lambda row: call(client, model, row), rows))


def stats(rows, model):
    values = [row[model] for row in rows]
    return {
        "n": len(values),
        "accuracy": statistics.mean(value["correct"] for value in values),
        "correct": sum(value["correct"] for value in values),
        "errors": sum(value["error"] is not None for value in values),
        "invalid_or_truncated_answers": sum(
            value["error"] == "invalid_or_truncated_answer" for value in values
        ),
        "request_errors": sum(
            value["error"] not in (None, "invalid_or_truncated_answer") for value in values
        ),
        "mean_latency_s": statistics.mean(value["latency_s"] for value in values),
        "median_input_tokens": statistics.median(value["input_tokens"] for value in values if value["input_tokens"] is not None),
        "median_output_tokens": statistics.median(value["output_tokens"] for value in values if value["output_tokens"] is not None),
    }


def report(rows):
    output = {"n": len(rows), "splits": {}}
    for split in ("train", "validation"):
        split_rows = [row for row in rows if row["split"] == split]
        section = {
            "n": len(split_rows),
            "overall": {model: stats(split_rows, model) for model in MODELS},
            "by_difficulty": {},
        }
        section["capable_lift"] = (
            section["overall"]["capable"]["accuracy"]
            - section["overall"]["efficient"]["accuracy"]
        )
        for difficulty in ("easy", "hard"):
            subset = [row for row in split_rows if row["difficulty"] == difficulty]
            model_stats = {model: stats(subset, model) for model in MODELS}
            model_stats["capable_lift"] = (
                model_stats["capable"]["accuracy"] - model_stats["efficient"]["accuracy"]
            )
            section["by_difficulty"][difficulty] = model_stats
        output["splits"][split] = section
    output["gates"] = {
        "validation_efficient_easy_accuracy_at_least_85pct": (
            output["splits"]["validation"]["by_difficulty"]["easy"]["efficient"]["accuracy"] >= 0.85
        ),
        "validation_capable_hard_lift_at_least_25pp": (
            output["splits"]["validation"]["by_difficulty"]["hard"]["capable_lift"] >= 0.25
        ),
        "validation_overall_capable_lift_at_least_5pp": (
            output["splits"]["validation"]["capable_lift"] >= 0.05
        ),
        "no_request_errors": all(
            row[model]["error"] in (None, "invalid_or_truncated_answer")
            for row in rows
            for model in MODELS
        ),
    }
    output["passed"] = all(output["gates"].values())
    output["note"] = "Development labels only. The 500-row test split was not loaded or evaluated."
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--efficient-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--capable-url", default="http://127.0.0.1:18001/v1")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=100)
    args = parser.parse_args()

    manifest = json.loads((args.data / "manifest.json").read_text())
    for split in ("train", "validation"):
        if digest(args.data / f"{split}.jsonl") != manifest["file_sha256"][split]:
            raise ValueError(f"Frozen {split} hash mismatch")
    rows = read_jsonl(args.data / "train.jsonl") + read_jsonl(args.data / "validation.jsonl")
    if any(row["split"] == "test" for row in rows):
        raise ValueError("Test rows must not be consumed during label collection")

    previous = read_jsonl(args.output) if args.output.exists() else []
    done = {row["example_id"] for row in previous}
    pending = [row for row in rows if row["example_id"] not in done]
    if len(previous) + len(pending) != len(rows):
        raise ValueError("Output IDs do not match the frozen development set")

    clients = {
        "efficient": OpenAI(base_url=args.efficient_url, api_key="unused", max_retries=0, timeout=120),
        "capable": OpenAI(base_url=args.capable_url, api_key="unused", max_retries=0, timeout=120),
    }
    for role, client in clients.items():
        if MODELS[role] not in {model.id for model in client.models.list().data}:
            raise ValueError(f"{role} endpoint is not ready")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for offset in range(0, len(pending), args.chunk_size):
        chunk = pending[offset : offset + args.chunk_size]
        order = list(MODELS)
        random.Random(f"large-label-chunk:{offset}").shuffle(order)
        with ThreadPoolExecutor(max_workers=2) as outer:
            futures = {
                role: outer.submit(run_model, clients[role], MODELS[role], chunk, args.concurrency)
                for role in order
            }
            results = {role: futures[role].result() for role in MODELS}
        with args.output.open("a") as handle:
            for index, source in enumerate(chunk):
                record = {**source, **{role: results[role][index] for role in MODELS}}
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                previous.append(record)
        print(f"Labeled {len(previous)}/{len(rows)} train+validation rows", flush=True)

    artifact = report(previous)
    artifact.update({
        "version": "shopping-mmlu-tiered-real-large-labels-v1",
        "dataset_manifest_sha256": digest(args.data / "manifest.json"),
        "development_predictions_sha256": digest(args.output),
        "concurrency_per_model": args.concurrency,
    })
    args.report.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
