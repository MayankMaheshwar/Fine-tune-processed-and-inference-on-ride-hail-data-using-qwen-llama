#!/usr/bin/env python3
"""Prepare the downloaded Bitext support CSV as local MLX-LM chat JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data/raw/bitext/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
)
DEFAULT_SYNTHETIC = PROJECT_ROOT / "data/synthetic/ride_hailing.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "data/processed/bitext"

SYSTEM_PROMPT = (
    "You are a clear, courteous general customer-support assistant. "
    "These examples are generic commerce support and do not define the current "
    "policies of any ride-hailing company. Do not claim to take actions you "
    "cannot take. Ask for information needed to help. For immediate danger, "
    "tell the customer to contact local emergency services."
)

PLACEHOLDER = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")
WHITESPACE = re.compile(r"\s+")


def clean_text(value: str) -> str:
    def replace_placeholder(match: re.Match[str]) -> str:
        label = re.sub(r"[_-]+", " ", match.group(1)).strip().lower()
        return label

    value = PLACEHOLDER.sub(replace_placeholder, value)
    return WHITESPACE.sub(" ", value).strip()


def load_rows(input_path: Path) -> list[dict[str, str]]:
    with input_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"instruction", "response", "category", "intent"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

        rows: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for row in reader:
            user_text = clean_text(row["instruction"] or "")
            assistant_text = clean_text(row["response"] or "")
            if not user_text or not assistant_text:
                continue
            identity = (user_text.casefold(), assistant_text.casefold())
            if identity in seen:
                continue
            seen.add(identity)
            rows.append(
                {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": assistant_text},
                    ],
                    "source": "Bitext customer support dataset",
                    "category": row["category"],
                    "intent": row["intent"],
                }
            )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_synthetic(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list) or [m.get("role") for m in messages[-2:]] != ["user", "assistant"]:
                raise ValueError(f"Expected user/assistant chat at {path}:{line_number}")
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--synthetic", type=Path, default=DEFAULT_SYNTHETIC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument("--max-general-train", type=int, default=1200)
    parser.add_argument("--synthetic-repeats", type=int, default=24)
    args = parser.parse_args()

    if args.valid_fraction <= 0 or args.test_fraction <= 0:
        parser.error("Validation and test fractions must both be positive.")
    if args.valid_fraction + args.test_fraction >= 0.5:
        parser.error("Validation and test fractions must sum to less than 0.5.")

    if args.max_general_train < 1 or args.synthetic_repeats < 1:
        parser.error("Training example limits and synthetic repeats must be positive.")

    rows = load_rows(args.input)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    test_size = max(1, round(len(rows) * args.test_fraction))
    valid_size = max(1, round(len(rows) * args.valid_fraction))
    test_rows = rows[:test_size]
    valid_rows = rows[test_size : test_size + valid_size]
    general_train_rows = rows[test_size + valid_size :]
    general_train_rows = rng.sample(
        general_train_rows, min(len(general_train_rows), args.max_general_train)
    )
    synthetic_rows = load_synthetic(args.synthetic)
    train_rows = general_train_rows + synthetic_rows * args.synthetic_repeats
    rng.shuffle(train_rows)

    args.output.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output / "train.jsonl", train_rows)
    write_jsonl(args.output / "valid.jsonl", valid_rows)
    write_jsonl(args.output / "test.jsonl", test_rows)
    print(
        f"Prepared {len(rows):,} unique Bitext examples plus "
        f"{len(synthetic_rows):,} synthetic ride-hailing examples: "
        f"{len(general_train_rows):,} general + "
        f"{len(synthetic_rows) * args.synthetic_repeats:,} repeated synthetic train, "
        f"{len(valid_rows):,} validation, "
        f"{len(test_rows):,} test. Output: {args.output}"
    )


if __name__ == "__main__":
    main()
