#!/usr/bin/env python3
"""Build a frozen 4,500-row real Shopping MMLU easy/hard routing dataset."""

import argparse
import ast
import csv
import hashlib
import io
import json
import random
import zipfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PRIOR_PILOT = ROOT / "data/shopping-mmlu/tiered-real-pilot-v1/inputs.jsonl"
EXPECTED_ARCHIVE_SHA256 = "899bad7b0e30d43bac50d4e110120b009ac39a3a84a4ab92f5db25edd3278cef"

# These tasks produced 92% E4B accuracy in the 500-row pilot and require direct
# attribute, category, compatibility, or commonsense lookup rather than arithmetic.
EASY_QUOTAS = {
    "applicable_attribute_selection": {"train": 350, "validation": 50, "test": 50},
    "asin_compatibility": {"train": 100, "validation": 15, "test": 15},
    "attribute_synonym": {"train": 210, "validation": 30, "test": 30},
    "attribute_value_extraction": {"train": 250, "validation": 35, "test": 35},
    "commonsense": {"train": 340, "validation": 45, "test": 45},
    "compatible_attribute_value_selection": {"train": 820, "validation": 115, "test": 115},
    "pt_selection_from_asin": {"train": 600, "validation": 85, "test": 85},
    "query_product_type_selection": {"train": 180, "validation": 25, "test": 25},
}

# These tasks produced a 47-point capable-model lift in the pilot and require
# explicit arithmetic or unit conversion.
HARD_QUOTAS = {
    "product_numeric_reasoning": {"train": 360, "validation": 55, "test": 55},
    "unit_conversion": {"train": 290, "validation": 45, "test": 45},
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalized_question(text):
    return " ".join(text.lower().split())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def source_rows(archive, task):
    with zipfile.ZipFile(archive) as source:
        name = f"multiple_choice/{task}_dataset.csv"
        return list(csv.DictReader(io.StringIO(source.read(name).decode("utf-8-sig"))))


def valid_rows(archive, task, difficulty):
    output = []
    seen = set()
    for source_index, row in enumerate(source_rows(archive, task)):
        try:
            choices = ast.literal_eval(row["choices"])
            answer_index = int(row["answer"])
        except (ValueError, SyntaxError, TypeError):
            continue
        if not isinstance(choices, list) or len(choices) != 4 or answer_index not in range(4):
            continue
        normalized = normalized_question(row["question"])
        if normalized in seen:
            continue
        seen.add(normalized)
        question = row["question"] + "\n" + "\n".join(
            f"{chr(65 + index)}. {choice}" for index, choice in enumerate(choices)
        )
        output.append({
            "example_id": hashlib.sha256(f"{task}\n{question}".encode()).hexdigest(),
            "task": task,
            "difficulty": difficulty,
            "question": question,
            "answer": chr(65 + answer_index),
            "source_question": row["question"],
            "source_choices": choices,
            "source_answer_index": answer_index,
            "source_row": source_index,
        })
    return output


def select_task(rows, task, quotas, prior_questions):
    total = sum(quotas.values())
    if len(rows) < total:
        raise ValueError(f"{task}: need {total} valid unique rows, found {len(rows)}")

    exposed = [row for row in rows if normalized_question(row["source_question"]) in prior_questions]
    unseen = [row for row in rows if normalized_question(row["source_question"]) not in prior_questions]
    random.Random(f"shopping-tiered-large-v1:{task}:exposed").shuffle(exposed)
    random.Random(f"shopping-tiered-large-v1:{task}:unseen").shuffle(unseen)

    train_n = quotas["train"]
    train = exposed[:train_n]
    train.extend(unseen[: train_n - len(train)])
    remaining_unseen = unseen[train_n - len(exposed[:train_n]) :]
    validation_n = quotas["validation"]
    test_n = quotas["test"]
    if len(remaining_unseen) < validation_n + test_n:
        raise ValueError(f"{task}: not enough never-inspected rows for validation and test")
    validation = remaining_unseen[:validation_n]
    test = remaining_unseen[validation_n : validation_n + test_n]

    selected = []
    for split, split_rows in (("train", train), ("validation", validation), ("test", test)):
        for row in split_rows:
            selected.append({
                **row,
                "split": split,
                "previously_inspected_in_pilot": (
                    normalized_question(row["source_question"]) in prior_questions
                ),
            })
    return selected, {
        "available_valid_unique": len(rows),
        "previously_inspected_available": len(exposed),
        "selected": dict(Counter(row["split"] for row in selected)),
    }


def build(archive, output):
    if output.exists():
        raise ValueError("Choose a fresh output directory; frozen datasets are never overwritten")
    if digest(archive) != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Official archive hash differs from the audited Shopping MMLU release")
    prior_rows = read_jsonl(PRIOR_PILOT)
    prior_questions = {normalized_question(row["source_question"]) for row in prior_rows}

    selected = []
    audit = {}
    for difficulty, quotas_by_task in (("easy", EASY_QUOTAS), ("hard", HARD_QUOTAS)):
        for task, quotas in quotas_by_task.items():
            task_rows, task_audit = select_task(
                valid_rows(archive, task, difficulty), task, quotas, prior_questions
            )
            selected.extend(task_rows)
            audit[task] = task_audit

    random.Random("shopping-tiered-large-v1:global").shuffle(selected)
    ids = [row["example_id"] for row in selected]
    normalized = [normalized_question(row["source_question"]) for row in selected]
    if len(ids) != len(set(ids)) or len(normalized) != len(set(normalized)):
        raise ValueError("Duplicate examples survived selection")
    if any(row["previously_inspected_in_pilot"] for row in selected if row["split"] != "train"):
        raise ValueError("Prior pilot leakage into validation or test")

    output.mkdir(parents=True)
    for name, rows in [("inputs", selected)] + [
        (split, [row for row in selected if row["split"] == split])
        for split in ("train", "validation", "test")
    ]:
        with (output / f"{name}.jsonl").open("x") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts = dict(Counter(row["split"] for row in selected))
    difficulty_counts = {
        split: dict(Counter(row["difficulty"] for row in selected if row["split"] == split))
        for split in ("train", "validation", "test")
    }
    manifest = {
        "version": "shopping-mmlu-tiered-real-large-v1",
        "source": "Unmodified questions, choices, and labels from the official public Shopping MMLU archive",
        "source_url": "https://github.com/KL4805/ShoppingMMLU",
        "archive_sha256": digest(archive),
        "prior_pilot_sha256": digest(PRIOR_PILOT),
        "difficulty_contract": {
            "easy": "Direct attribute/category/compatibility/commonsense lookup only; no arithmetic.",
            "hard": "Product arithmetic or unit conversion required.",
        },
        "selection_policy": (
            "Fixed task quotas selected deterministically without using model outcomes. "
            "Previously inspected pilot rows may appear only in training; validation and test are new."
        ),
        "counts": counts,
        "difficulty_counts_by_split": difficulty_counts,
        "task_quotas": {"easy": EASY_QUOTAS, "hard": HARD_QUOTAS},
        "task_audit": audit,
        "file_sha256": {
            name: digest(output / f"{name}.jsonl")
            for name in ("inputs", "train", "validation", "test")
        },
        "models": {
            "efficient": "google/gemma-4-E4B-it",
            "capable": "google/gemma-4-31B-it",
        },
        "system_prompt": (
            "Answer the shopping multiple-choice question using the question and all four choices. "
            "Return only the letter A, B, C, or D for the best answer. Do not explain."
        ),
        "settings": {
            "temperature": 0,
            "seed": 42,
            "max_tokens": 16,
            "thinking": False,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.archive, args.output)
