"""Fuzz test backend robustness against arbitrary AI-style payloads.

This script generates malformed and varied inputs, sends them directly to the
orchestrator and through API endpoint validation, and reports failure patterns.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient

from main import app
from services.multi_agent_orchestrator import generate_dynamic_layout


def rand_value(rng: random.Random) -> Any:
    pool = [
        None,
        "",
        "  ",
        "nan",
        "inf",
        "-inf",
        "30",
        "30x40",
        "1200 sqft",
        -999,
        -1,
        0,
        1,
        2,
        3,
        4,
        999999,
        12.5,
        {},
        {"study": True},
        [],
        ["pooja", "study"],
        [1, None, "garage"],
        True,
        False,
    ]
    return rng.choice(pool)


def make_payload(rng: random.Random) -> Dict[str, Any]:
    keys = [
        "plot_width",
        "plot_length",
        "total_area",
        "bedrooms",
        "bathrooms",
        "floors",
        "facing",
        "vastu",
        "extras",
        "family_type",
        "city",
        "state",
        "previous_strategy",
        "engine_mode",
    ]
    payload: Dict[str, Any] = {}
    for k in keys:
        if rng.random() < 0.8:
            payload[k] = rand_value(rng)

    # Occasionally provide structured but odd data.
    if rng.random() < 0.3:
        payload["plot_width"] = rng.choice([18, 20, 24, 30, 36, "24", "36 ft"])
        payload["plot_length"] = rng.choice([30, 35, 40, 45, 55, "40", "55 ft"])
    if rng.random() < 0.2:
        payload["extras"] = ",".join(rng.sample(["pooja", "study", "store", "balcony", "garage"], k=rng.randint(0, 3)))

    return payload


def run(count: int, seed: int) -> Dict[str, Any]:
    rng = random.Random(seed)

    direct_errors = 0
    direct_error_types: Counter[str] = Counter()
    direct_success = 0
    direct_accepted = 0
    direct_reasons: Counter[str] = Counter()

    api_5xx = 0
    api_4xx = 0
    api_2xx = 0

    with TestClient(app) as client:
        for _ in range(count):
            payload = make_payload(rng)

            # Direct robustness test (core engine should not crash).
            try:
                out = generate_dynamic_layout(payload)
                direct_success += 1
                if out.get("accepted"):
                    direct_accepted += 1
                reason = out.get("issues", [])
                if isinstance(reason, list) and reason:
                    direct_reasons.update([str(reason[0])[:80]])
            except Exception as exc:
                direct_errors += 1
                direct_error_types.update([type(exc).__name__])

            # API robustness: malformed payloads should be handled with 4xx/2xx, not 5xx.
            res = client.post("/api/architect/design", json=payload)
            if 200 <= res.status_code < 300:
                api_2xx += 1
            elif 400 <= res.status_code < 500:
                api_4xx += 1
            else:
                api_5xx += 1

    return {
        "count": count,
        "seed": seed,
        "direct": {
            "success": direct_success,
            "errors": direct_errors,
            "accepted": direct_accepted,
            "error_types": dict(direct_error_types),
            "sample_issue": dict(direct_reasons.most_common(5)),
        },
        "api": {
            "2xx": api_2xx,
            "4xx": api_4xx,
            "5xx": api_5xx,
        },
        "gates": {
            "direct_error_rate_le_0_01": direct_errors / max(count, 1) <= 0.01,
            "api_5xx_rate_le_0_01": api_5xx / max(count, 1) <= 0.01,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Fuzz test backend with arbitrary AI-like payloads")
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    report = run(count=args.count, seed=args.seed)
    report["all_gates_pass"] = all(report["gates"].values())

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
