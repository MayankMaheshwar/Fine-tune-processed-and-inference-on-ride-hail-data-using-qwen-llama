#!/usr/bin/env python3
"""Run the held-out ride-hailing scenarios against the local base and adapter."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(app.MODEL_OPTIONS),
        default=list(app.MODEL_OPTIONS),
    )
    parser.add_argument("--max-tokens", type=int, default=160)
    args = parser.parse_args()

    report_dir = app.ROOT / "data/eval/generated"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "ride_hailing_comparison.json"
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "max_tokens": args.max_tokens,
        "results": {},
    }

    for model_name in args.models:
        model_results = []
        print(f"\n=== {model_name} ===", flush=True)
        for case in app.CASES:
            answer = app.generate_answer(
                model_name,
                [{"role": "user", "content": case["prompt"]}],
                args.max_tokens,
            )
            model_results.append(
                {
                    "id": case["id"],
                    "prompt": case["prompt"],
                    "review_checks": case["checks"],
                    "response": answer,
                }
            )
            print(f"\n[{case['id']}]\n{answer}", flush=True)
        report["results"][model_name] = model_results
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(f"\nSaved comparison report: {report_path}")


if __name__ == "__main__":
    main()
