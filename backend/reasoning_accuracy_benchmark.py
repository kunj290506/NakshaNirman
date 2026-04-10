"""
Reasoning + layout fidelity benchmark for house-plan generation.

This benchmark now reports:
1) Frontier-style reasoning parity (chatgpt, gemini, opus, deepseek)
2) Objective post-plot quality signals
3) Effective score that combines both and penalizes fallback-only paths

Usage:
  backend/.venv/Scripts/python.exe backend/reasoning_accuracy_benchmark.py
  backend/.venv/Scripts/python.exe backend/reasoning_accuracy_benchmark.py --mode emergency
  backend/.venv/Scripts/python.exe backend/reasoning_accuracy_benchmark.py --suite stress --json
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


STRESS_CASES: list[dict[str, Any]] = [
    {
        "plot_width": 22,
        "plot_length": 30,
        "bedrooms": 2,
        "facing": "north",
        "extras": ["study", "store"],
        "bathrooms_target": 2,
        "city": "noida",
        "family_type": "couple",
    },
    {
        "plot_width": 24,
        "plot_length": 48,
        "bedrooms": 3,
        "facing": "east",
        "extras": ["utility"],
        "bathrooms_target": 3,
        "city": "gurugram",
        "family_type": "nuclear",
        "work_from_home": True,
    },
    {
        "plot_width": 35,
        "plot_length": 35,
        "bedrooms": 2,
        "facing": "south",
        "extras": ["pooja", "balcony"],
        "city": "indore",
        "family_type": "nuclear",
        "elder_friendly": True,
    },
    {
        "plot_width": 42,
        "plot_length": 42,
        "bedrooms": 3,
        "facing": "west",
        "extras": ["study", "utility", "store"],
        "bathrooms_target": 3,
        "city": "surat",
        "family_type": "joint",
        "work_from_home": True,
    },
    {
        "plot_width": 30,
        "plot_length": 55,
        "bedrooms": 3,
        "facing": "north",
        "extras": ["staircase", "utility"],
        "floors": 2,
        "city": "nagpur",
        "family_type": "joint",
    },
    {
        "plot_width": 52,
        "plot_length": 65,
        "bedrooms": 4,
        "facing": "east",
        "extras": ["garage", "staircase", "study", "pooja", "utility"],
        "bathrooms_target": 4,
        "city": "thane",
        "family_type": "joint",
        "elder_friendly": True,
    },
    {
        "plot_width": 26,
        "plot_length": 34,
        "bedrooms": 2,
        "facing": "south",
        "extras": [],
        "city": "bhopal",
        "family_type": "nuclear",
    },
    {
        "plot_width": 38,
        "plot_length": 58,
        "bedrooms": 3,
        "facing": "west",
        "extras": ["study", "pooja", "balcony"],
        "city": "mysuru",
        "family_type": "nuclear",
        "work_from_home": True,
        "elder_friendly": True,
    },
    {
        "plot_width": 65,
        "plot_length": 95,
        "bedrooms": 4,
        "facing": "north",
        "extras": ["garage", "study", "utility", "store", "balcony", "foyer"],
        "bathrooms_target": 5,
        "city": "kolkata",
        "family_type": "joint",
        "floors": 2,
    },
]


def _r1(val: float) -> float:
    return round(float(val), 1)


def _bounded(val: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, float(val)))


def _generation_path_factor(path: str) -> float:
    key = str(path or "").strip().lower()
    if key in {"openrouter", "openrouter_backup", "openrouter_retry", "claude", "llm"}:
        return 1.0
    if key in {"public_fallback", "plan_backup"}:
        return 0.92
    if key in {"bsp_fallback", "bsp"}:
        return 0.86
    if key in {"emergency_local", "deterministic_demo"}:
        return 0.80
    return 0.90


def _composite_effective_score(
    parity: float,
    quality_overall: float,
    completion_pct: float,
    check_pass_rate: float,
    path_factor: float,
) -> float:
    base = (
        (float(parity) * 0.45)
        + (float(quality_overall) * 0.25)
        + (float(completion_pct) * 0.20)
        + (float(check_pass_rate) * 0.10)
    )
    return _bounded(base * float(path_factor))


def _selected_cases(suite: str) -> list[dict[str, Any]]:
    key = str(suite or "all").strip().lower()
    if key == "standard":
        return list(DEFAULT_CASES)
    if key == "stress":
        return list(STRESS_CASES)
    return [*DEFAULT_CASES, *STRESS_CASES]


async def _generate(req: PlanRequest, mode: str):
    if mode == "primary":
        return await generate_plan(req)
    return await generate_plan_emergency_local(req)


async def run_benchmark(mode: str, threshold: float, suite: str) -> dict[str, Any]:
    per_model: dict[str, list[float]] = {
        "chatgpt": [],
        "gemini": [],
        "opus": [],
        "deepseek": [],
    }
    parity_scores: list[float] = []
    quality_scores: list[float] = []
    completion_scores: list[float] = []
    pass_rate_scores: list[float] = []
    effective_scores: list[float] = []

    cases: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    generation_paths: dict[str, int] = {}

    case_payloads = _selected_cases(suite)

    for idx, payload in enumerate(case_payloads, start=1):
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
        q = ar.get("quality_scores", {}) or {}
        rc = ar.get("requirement_coverage", {}) or {}
        diag = ar.get("diagnostics", {}) or {}

        parity = float(fc.get("parity_score", 0.0) or 0.0)
        quality_overall = float(q.get("overall", 0.0) or 0.0)
        completion_pct = float(rc.get("completion_pct", 0.0) or 0.0)
        check_pass_rate = float((diag.get("checks", {}) or {}).get("pass_rate", 0.0) or 0.0)

        generation_path = str(
            plan.generation_method
            or ar.get("generation_path", "")
            or ((fc.get("grounded_quality", {}) or {}).get("generation_path", ""))
            or "unknown"
        ).strip().lower() or "unknown"
        path_factor = _generation_path_factor(generation_path)
        effective_score = _composite_effective_score(
            parity,
            quality_overall,
            completion_pct,
            check_pass_rate,
            path_factor,
        )

        generation_paths[generation_path] = generation_paths.get(generation_path, 0) + 1

        parity_scores.append(parity)
        quality_scores.append(quality_overall)
        completion_scores.append(completion_pct)
        pass_rate_scores.append(check_pass_rate)
        effective_scores.append(effective_score)

        row = {
            "case": idx,
            "plot": f"{payload['plot_width']}x{payload['plot_length']}",
            "bedrooms": payload["bedrooms"],
            "facing": payload["facing"],
            "generation_path": generation_path,
            "parity": _r1(parity),
            "quality_overall": _r1(quality_overall),
            "requirement_completion": _r1(completion_pct),
            "check_pass_rate": _r1(check_pass_rate),
            "effective_score": _r1(effective_score),
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
            "path": c["generation_path"],
            "effective_score": c["effective_score"],
            "parity": c["parity"],
            "quality_overall": c["quality_overall"],
            "shared_gaps": c["shared_gaps"],
        }
        for c in cases
        if float(c["effective_score"]) < threshold
    ]

    notes: list[str] = []
    if str(mode).lower() == "emergency":
        notes.append("Emergency mode uses deterministic planner and should not be used as frontier-thinking proof.")
    if str(suite).lower() != "all":
        notes.append("Suite is partial; run --suite all for broader condition coverage.")

    summary = {
        "mode": mode,
        "suite": suite,
        "threshold": threshold,
        "samples": len(case_payloads),
        "evaluated": len(cases),
        "failed": len(failed),
        "overall_accuracy": round(mean(parity_scores), 2) if parity_scores else 0.0,
        "effective_accuracy": round(mean(effective_scores), 2) if effective_scores else 0.0,
        "quality_overall_mean": round(mean(quality_scores), 2) if quality_scores else 0.0,
        "requirement_completion_mean": round(mean(completion_scores), 2) if completion_scores else 0.0,
        "check_pass_rate_mean": round(mean(pass_rate_scores), 2) if pass_rate_scores else 0.0,
        "chatgpt_accuracy": round(mean(per_model["chatgpt"]), 2) if per_model["chatgpt"] else 0.0,
        "gemini_accuracy": round(mean(per_model["gemini"]), 2) if per_model["gemini"] else 0.0,
        "opus_accuracy": round(mean(per_model["opus"]), 2) if per_model["opus"] else 0.0,
        "deepseek_accuracy": round(mean(per_model["deepseek"]), 2) if per_model["deepseek"] else 0.0,
        "min_parity": round(min(parity_scores), 2) if parity_scores else 0.0,
        "max_parity": round(max(parity_scores), 2) if parity_scores else 0.0,
        "min_effective": round(min(effective_scores), 2) if effective_scores else 0.0,
        "max_effective": round(max(effective_scores), 2) if effective_scores else 0.0,
        "weak_case_count": len(weak_cases),
        "generation_paths": generation_paths,
        "notes": notes,
    }

    return {
        "summary": summary,
        "cases": cases,
        "weak_cases": weak_cases,
        "failed_cases": failed,
    }


def _print_human(result: dict[str, Any]) -> None:
    summary = result["summary"]
    print("Reasoning + Layout Benchmark")
    print("=" * 36)
    print(json.dumps(summary, indent=2))

    print("\nCase Scores")
    for row in result["cases"]:
        print(
            f"- case {row['case']:>2} | eff {row['effective_score']:>5} | "
            f"parity {row['parity']:>5} quality {row['quality_overall']:>5} req {row['requirement_completion']:>5} | "
            f"{row['generation_path']:<16} | {row['plot']} {row['bedrooms']}BHK {row['facing']}"
        )

    if result["weak_cases"]:
        print("\nWeak Cases")
        for row in result["weak_cases"]:
            print(
                f"- case {row['case']} eff={row['effective_score']} parity={row['parity']} "
                f"quality={row['quality_overall']} path={row['path']} gaps={row['shared_gaps']}"
            )

    if result["failed_cases"]:
        print("\nFailed Cases")
        for row in result["failed_cases"]:
            print(f"- case {row['case']} error={row['error']}")


async def _amain() -> int:
    parser = argparse.ArgumentParser(description="Benchmark backend reasoning and layout fidelity.")
    parser.add_argument(
        "--mode",
        choices=["emergency", "primary"],
        default="primary",
        help="primary=full pipeline, emergency=deterministic local pipeline",
    )
    parser.add_argument(
        "--suite",
        choices=["standard", "stress", "all"],
        default="all",
        help="Case suite to run",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=84.0,
        help="Effective score threshold used to label weak cases",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print only JSON output",
    )
    args = parser.parse_args()

    result = await run_benchmark(
        mode=str(args.mode),
        threshold=float(args.threshold),
        suite=str(args.suite),
    )
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_human(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_amain()))
