"""Run house-generation benchmark scenarios and print/save summary metrics."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.multi_agent_orchestrator import generate_dynamic_layout


def _base_payload(**kwargs: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "plot_width": 30,
        "plot_length": 40,
        "bedrooms": 2,
        "bathrooms": 2,
        "facing": "east",
        "family_type": "nuclear",
        "city": "Pune",
        "state": "Maharashtra",
        "vastu": True,
        "extras": [],
    }
    payload.update(kwargs)
    return payload


def build_scenarios(seed: int, count: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    cities = ["Pune", "Chennai", "Delhi", "Kochi", "Bengaluru", "Jaipur", "Lucknow", "Mumbai", "Goa", "Shimla"]
    states = ["Maharashtra", "Tamil Nadu", "Delhi", "Kerala", "Karnataka", "Rajasthan", "Uttar Pradesh", "Maharashtra", "Goa", "Himachal Pradesh"]
    families = ["nuclear", "joint-family", "working-couple", "elderly", "rental"]
    extras_pool = ["pooja", "study", "store", "balcony", "garage"]
    facings = ["east", "west", "north", "south"]

    scenarios: List[Dict[str, Any]] = []
    for _ in range(count):
        idx = rng.randrange(len(cities))
        scenarios.append(
            _base_payload(
                plot_width=rng.randint(18, 38),
                plot_length=rng.randint(30, 58),
                bedrooms=rng.randint(1, 4),
                bathrooms=rng.randint(1, 5),
                facing=rng.choice(facings),
                family_type=rng.choice(families),
                city=cities[idx],
                state=states[idx],
                vastu=rng.choice([True, False]),
                extras=rng.sample(extras_pool, k=rng.randint(0, 3)),
            )
        )
    return scenarios


def compute_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in results if "error" not in r]
    composites = [float(r.get("design_score", {}).get("composite", 0.0)) for r in valid]
    accepted = [r for r in valid if r.get("accepted")]
    strict_accepted = [r for r in valid if r.get("strict_accepted")]

    report = {
        "total": len(results),
        "valid": len(valid),
        "errors": len(results) - len(valid),
        "accepted": len(accepted),
        "strict_accepted": len(strict_accepted),
        "avg_composite": round(statistics.mean(composites), 2) if composites else 0.0,
        "p50_composite": round(statistics.median(composites), 2) if composites else 0.0,
        "min_composite": round(min(composites), 2) if composites else 0.0,
        "max_composite": round(max(composites), 2) if composites else 0.0,
        "gates": {
            "valid_rate_ge_0_90": len(valid) / max(len(results), 1) >= 0.90,
            "accepted_rate_ge_0_60": len(accepted) / max(len(valid), 1) >= 0.60,
            "avg_score_ge_64": (statistics.mean(composites) if composites else 0.0) >= 64.0,
        },
    }
    report["all_gates_pass"] = all(report["gates"].values())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run house-generation benchmark report")
    parser.add_argument("--count", type=int, default=60, help="Number of random scenarios")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default="", help="Optional JSON output path")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if quality gates fail")
    args = parser.parse_args()

    scenarios = build_scenarios(seed=args.seed, count=args.count)
    results = [generate_dynamic_layout(s) for s in scenarios]
    report = compute_report(results)

    print(json.dumps(report, indent=2))

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.strict and not report["all_gates_pass"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
