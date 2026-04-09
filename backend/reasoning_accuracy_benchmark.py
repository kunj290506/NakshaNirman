"""
Reasoning accuracy benchmark for house-plan design.

Computes parity-style accuracy against four frontier profiles present in
architect_reasoning.frontier_comparison: chatgpt, gemini, opus, deepseek.

Usage:
  backend/.venv/Scripts/python.exe backend/reasoning_accuracy_benchmark.py
  backend/.venv/Scripts/python.exe backend/reasoning_accuracy_benchmark.py --mode primary
  backend/.venv/Scripts/python.exe backend/reasoning_accuracy_benchmark.py --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from statistics import mean
from typing import Any

from models import PlanRequest
from layout_engine import generate_plan, generate_plan_emergency_local


DEFAULT_CASES: list[dict[str, Any]] = [
    {
        "plot_width": 30,
        "plot_length": 40,
        "bedrooms": 2,
        "facing": "east",
        "extras": ["pooja", "study"],
        "city": "pune",
        "family_type": "nuclear",
    },
    {
        "plot_width": 25,
        "plot_length": 35,
        "bedrooms": 2,
        "facing": "north",
        "extras": ["store"],
        "city": "delhi",
        "family_type": "couple",
    },
    {
        "plot_width": 40,
        "plot_length": 60,
        "bedrooms": 4,
        "facing": "south",
        "extras": ["pooja", "study", "utility", "store"],
        "city": "ahmedabad",
        "family_type": "joint",
        "work_from_home": True,
        "elder_friendly": True,
    },
    {
        "plot_width": 36,
        "plot_length": 52,
        "bedrooms": 3,
        "facing": "west",
        "extras": ["utility", "balcony"],
        "city": "mumbai",
        "family_type": "nuclear",
    },
    {
        "plot_width": 32,
        "plot_length": 45,
        "bedrooms": 3,
        "facing": "south",
        "extras": ["study"],
        "city": "bengaluru",
        "family_type": "nuclear",
        "work_from_home": True,
    },
    {
        "plot_width": 45,
        "plot_length": 70,
        "bedrooms": 4,
        "facing": "east",
        "extras": ["garage", "utility", "pooja", "study"],
        "city": "hyderabad",
        "family_type": "joint",
    },
    {
        "plot_width": 28,
        "plot_length": 38,
        "bedrooms": 2,
        "facing": "west",
        "extras": ["pooja"],
        "city": "jaipur",
        "family_type": "couple",
    },
    {
        "plot_width": 50,
        "plot_length": 80,
        "bedrooms": 4,
        "facing": "north",
        "extras": ["garage", "staircase", "utility", "store", "study"],
        "city": "chennai",
        "family_type": "joint",
        "elder_friendly": True,
    },
    {
        "plot_width": 34,
        "plot_length": 48,
        "bedrooms": 3,
        "facing": "north",
        "extras": ["pooja", "balcony"],
        "city": "kochi",
        "family_type": "nuclear",
    },
    {
        "plot_width": 60,
        "plot_length": 90,
        "bedrooms": 4,
        "facing": "south",
        "extras": ["garage", "study", "utility", "store", "balcony"],
        "city": "lucknow",
        "family_type": "joint",
        "work_from_home": True,
    },
]


def _r1(val: float) -> float:
    return round(float(val), 1)


async def _generate(req: PlanRequest, mode: str):
    if mode == "primary":
        return await generate_plan(req)
    return await generate_plan_emergency_local(req)


async def run_benchmark(mode: str, threshold: float) -> dict[str, Any]:
    per_model: dict[str, list[float]] = {
        "chatgpt": [],
        "gemini": [],
        "opus": [],
        "deepseek": [],
    }
    parity_scores: list[float] = []
    cases: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for idx, payload in enumerate(DEFAULT_CASES, start=1):
        req = PlanRequest(**payload)
        try:
            plan = await _generate(req, mode=mode)
        except Exception as exc:
            failed.append(
                {
                    "case": idx,
                    "error": str(exc)[:220],
                    "request": {
                        "plot": f"{payload['plot_width']}x{payload['plot_length']}",
                        "bedrooms": payload["bedrooms"],
                        "facing": payload["facing"],
                    },
                }
            )
            continue

        ar = plan.architect_reasoning or {}
        fc = ar.get("frontier_comparison", {}) or {}
        profiles = fc.get("profiles", {}) or {}
        parity = float(fc.get("parity_score", 0.0) or 0.0)
        parity_scores.append(parity)

        row = {
            "case": idx,
            "plot": f"{payload['plot_width']}x{payload['plot_length']}",
            "bedrooms": payload["bedrooms"],
            "facing": payload["facing"],
            "parity": _r1(parity),
            "status": str(fc.get("status", "")),
            "shared_gaps": list(fc.get("shared_gaps", []) or []),
        }

        for name in per_model:
            score = float((profiles.get(name, {}) or {}).get("score", 0.0) or 0.0)
            per_model[name].append(score)
            row[name] = _r1(score)

        cases.append(row)

    weak_cases = [
        {
            "case": c["case"],
            "plot": c["plot"],
            "parity": c["parity"],
            "shared_gaps": c["shared_gaps"],
        }
        for c in cases
        if float(c["parity"]) < threshold
    ]

    summary = {
        "mode": mode,
        "threshold": threshold,
        "samples": len(DEFAULT_CASES),
        "evaluated": len(cases),
        "failed": len(failed),
        "overall_accuracy": round(mean(parity_scores), 2) if parity_scores else 0.0,
        "chatgpt_accuracy": round(mean(per_model["chatgpt"]), 2) if per_model["chatgpt"] else 0.0,
        "gemini_accuracy": round(mean(per_model["gemini"]), 2) if per_model["gemini"] else 0.0,
        "opus_accuracy": round(mean(per_model["opus"]), 2) if per_model["opus"] else 0.0,
        "deepseek_accuracy": round(mean(per_model["deepseek"]), 2) if per_model["deepseek"] else 0.0,
        "min_parity": round(min(parity_scores), 2) if parity_scores else 0.0,
        "max_parity": round(max(parity_scores), 2) if parity_scores else 0.0,
        "weak_case_count": len(weak_cases),
    }

    return {
        "summary": summary,
        "cases": cases,
        "weak_cases": weak_cases,
        "failed_cases": failed,
    }


def _print_human(result: dict[str, Any]) -> None:
    summary = result["summary"]
    print("Reasoning Accuracy Benchmark")
    print("=" * 32)
    print(json.dumps(summary, indent=2))
    print("\nCase Scores")
    for row in result["cases"]:
        print(
            f"- case {row['case']:>2} | parity {row['parity']:>5} | "
            f"C {row['chatgpt']:>5} G {row['gemini']:>5} O {row['opus']:>5} D {row['deepseek']:>5} | "
            f"{row['plot']} {row['bedrooms']}BHK {row['facing']}"
        )

    if result["weak_cases"]:
        print("\nWeak Cases")
        for row in result["weak_cases"]:
            print(f"- case {row['case']} parity={row['parity']} gaps={row['shared_gaps']}")

    if result["failed_cases"]:
        print("\nFailed Cases")
        for row in result["failed_cases"]:
            print(f"- case {row['case']} error={row['error']}")


async def _amain() -> int:
    parser = argparse.ArgumentParser(description="Benchmark backend reasoning parity.")
    parser.add_argument(
        "--mode",
        choices=["emergency", "primary"],
        default="emergency",
        help="emergency=deterministic local pipeline, primary=full pipeline",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=88.0,
        help="Parity threshold used to label weak cases",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print only JSON output",
    )
    args = parser.parse_args()

    result = await run_benchmark(mode=args.mode, threshold=float(args.threshold))
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_human(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_amain()))
