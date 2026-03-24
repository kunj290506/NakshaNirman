"""Real-world LLM input suite for backend robustness.

This suite simulates payload styles from multiple LLMs (GPT/Claude/Gemini/
Llama/Mistral-like shapes), validates both direct orchestrator handling and API
response behavior, and emits gate-based reports.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from main import app
from services.multi_agent_orchestrator import generate_dynamic_layout


CITIES = [
    ("Pune", "Maharashtra"),
    ("Chennai", "Tamil Nadu"),
    ("Bengaluru", "Karnataka"),
    ("Delhi", "Delhi"),
    ("Kochi", "Kerala"),
    ("Jaipur", "Rajasthan"),
]


def case_gpt_style(rng: random.Random) -> Dict[str, Any]:
    city, state = rng.choice(CITIES)
    w = rng.choice([24, 30, 36])
    l = rng.choice([35, 40, 50])
    return {
        "plot_width": str(w),
        "plot_length": f"{l} ft",
        "bedrooms": rng.choice([2, 3, "3"]),
        "bathrooms": rng.choice([2, 3, "2"]),
        "family_type": rng.choice(["nuclear", "joint-family", "working-couple"]),
        "facing": rng.choice(["east", "north", "west", "south"]),
        "vastu": rng.choice([True, False, "true", "false"]),
        "extras": rng.choice([["study"], ["pooja", "store"], "study,balcony"]),
        "city": city,
        "state": state,
    }


def case_claude_style(rng: random.Random) -> Dict[str, Any]:
    city, state = rng.choice(CITIES)
    return {
        "plot_size": rng.choice(["30x40", "24 x 36 ft", "36×50"]),
        "bhk": rng.choice([2, 3, 4, "3bhk"]),
        "bath_count": rng.choice([2, 3, "3"]),
        "household": rng.choice(["elderly", "nuclear", "rental"]),
        "direction": rng.choice(["East", "North", "west"]),
        "needs_vastu": rng.choice(["yes", "no", True, False]),
        "optional_rooms": rng.choice(["pooja,study", ["store", "balcony"], {"garage": True}]),
        "location_city": city,
        "location_state": state,
    }


def case_gemini_style(rng: random.Random) -> Dict[str, Any]:
    city, state = rng.choice(CITIES)
    return {
        "site_width": rng.choice([22, 28, 32, "28ft"]),
        "site_length": rng.choice([34, 42, 48, "42 feet"]),
        "rooms": rng.choice([2, 3, 4]),
        "bathroom_count": rng.choice([2, 3, 4]),
        "family": rng.choice(["joint-family", "working-couple", "nuclear"]),
        "entry_facing": rng.choice(["north", "east", "south"]),
        "is_vastu_required": rng.choice([1, 0, "true", "false"]),
        "add_ons": rng.choice([["study"], ["pooja", "balcony"], "garage,store"]),
        "town": city,
        "region": state,
    }


def case_llama_style(rng: random.Random) -> Dict[str, Any]:
    city, state = rng.choice(CITIES)
    return {
        "totalarea": rng.choice([900, 1200, 1500, "1200 sqft"]),
        "bedroom_count": rng.choice([1, 2, 3, 4]),
        "baths": rng.choice([1, 2, 3, 4]),
        "familytype": rng.choice(["nuclear", "elderly", "rental"]),
        "front_direction": rng.choice(["east", "west"]),
        "vastu_enabled": rng.choice(["on", "off", True, False]),
        "extrarooms": rng.choice(["study", "pooja,study", ["balcony"]]),
        "city": city,
        "province": state,
    }


def case_mistral_style(rng: random.Random) -> Dict[str, Any]:
    city, state = rng.choice(CITIES)
    return {
        "site_size": rng.choice(["18x30", "20x35", "30x50"]),
        "rooms": rng.choice([1, 2, 3, 4]),
        "bathrooms": rng.choice([1, 2, 3, 4]),
        "household": rng.choice(["nuclear", "joint-family", "working-couple", "elderly"]),
        "direction": rng.choice(["north", "south", "east", "west"]),
        "needs_vastu": rng.choice(["yes", "no"]),
        "addons": rng.choice(["study", "pooja,store", "garage,balcony", []]),
        "location_city": city,
        "location_state": state,
        "retry_strategy": rng.choice([None, "cluster", "l_plan"]),
    }


def build_cases(seed: int, rounds: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    builders = [
        case_gpt_style,
        case_claude_style,
        case_gemini_style,
        case_llama_style,
        case_mistral_style,
    ]
    cases: List[Dict[str, Any]] = []
    for _ in range(rounds):
        for fn in builders:
            cases.append(fn(rng))
    return cases


def run_suite(seed: int, rounds: int) -> Dict[str, Any]:
    cases = build_cases(seed=seed, rounds=rounds)

    direct_success = 0
    direct_errors = 0
    direct_accepted = 0
    api_2xx = 0
    api_4xx = 0
    api_5xx = 0

    with TestClient(app) as client:
        for payload in cases:
            try:
                out = generate_dynamic_layout(payload)
                direct_success += 1
                if out.get("accepted"):
                    direct_accepted += 1
            except Exception:
                direct_errors += 1

            res = client.post("/api/architect/design", json=payload)
            if 200 <= res.status_code < 300:
                api_2xx += 1
            elif 400 <= res.status_code < 500:
                api_4xx += 1
            else:
                api_5xx += 1

    total = len(cases)
    report = {
        "total_cases": total,
        "direct": {
            "success": direct_success,
            "errors": direct_errors,
            "accepted": direct_accepted,
        },
        "api": {
            "2xx": api_2xx,
            "4xx": api_4xx,
            "5xx": api_5xx,
        },
        "gates": {
            "direct_no_crash": direct_errors == 0,
            "api_no_5xx": api_5xx == 0,
            "api_handled_rate_eq_1_00": ((api_2xx + api_4xx) / max(total, 1)) >= 1.0,
        },
    }
    report["all_gates_pass"] = all(report["gates"].values())
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run real-world LLM input tests")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    report = run_suite(seed=args.seed, rounds=args.rounds)
    print(json.dumps(report, indent=2))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.strict and not report["all_gates_pass"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
