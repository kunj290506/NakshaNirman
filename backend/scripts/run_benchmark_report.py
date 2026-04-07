from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layout_engine import build_room_list


CASES = [
    {"bedrooms": 1, "extras": [], "usable_w": 18.0, "usable_l": 28.5, "facing": "south", "floors": 1},
    {"bedrooms": 2, "extras": ["pooja"], "usable_w": 23.0, "usable_l": 28.5, "facing": "east", "floors": 1},
    {"bedrooms": 3, "extras": ["pooja", "study"], "usable_w": 30.0, "usable_l": 40.0, "facing": "north", "floors": 1},
    {"bedrooms": 4, "extras": ["pooja", "study", "utility"], "usable_w": 36.0, "usable_l": 52.0, "facing": "west", "floors": 2},
]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((percentile / 100.0) * (len(ordered) - 1)))
    idx = max(0, min(len(ordered) - 1, idx))
    return ordered[idx]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run lightweight benchmark gates for room program generation.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when benchmark gates fail.")
    parser.add_argument("--out", required=True, help="Output path for benchmark JSON report.")
    parser.add_argument("--iterations", type=int, default=250, help="Benchmark iterations per case.")
    parser.add_argument("--max-p95-ms", type=float, default=2.5, help="Maximum allowed p95 in ms.")
    args = parser.parse_args()

    samples_ms: list[float] = []
    errors: list[str] = []
    room_total = 0

    t_suite_start = time.perf_counter()
    for _ in range(max(1, int(args.iterations))):
        for case in CASES:
            t0 = time.perf_counter()
            try:
                rooms = build_room_list(**case)
                room_total += len(rooms)
                if not rooms:
                    errors.append("build_room_list returned empty room list")
            except Exception as exc:
                errors.append(str(exc))
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            samples_ms.append(elapsed_ms)
    suite_ms = (time.perf_counter() - t_suite_start) * 1000.0

    mean_ms = statistics.mean(samples_ms) if samples_ms else 0.0
    p95_ms = _percentile(samples_ms, 95.0)
    max_ms = max(samples_ms) if samples_ms else 0.0

    gates = {
        "no_exceptions": len(errors) == 0,
        "p95_within_limit": p95_ms <= float(args.max_p95_ms),
    }
    ok = all(gates.values())

    report = {
        "ok": ok,
        "strict_mode": bool(args.strict),
        "iterations": int(args.iterations),
        "case_count": len(CASES),
        "samples": len(samples_ms),
        "timings_ms": {
            "mean": round(mean_ms, 4),
            "p95": round(p95_ms, 4),
            "max": round(max_ms, 4),
            "suite_total": round(suite_ms, 3),
        },
        "max_p95_ms": float(args.max_p95_ms),
        "generated_rooms": int(room_total),
        "gates": gates,
        "errors": errors[:25],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        "benchmark: "
        f"ok={ok} samples={len(samples_ms)} p95_ms={report['timings_ms']['p95']} "
        f"limit={args.max_p95_ms}"
    )

    if args.strict and not ok:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
