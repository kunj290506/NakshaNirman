from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from layout_engine import build_room_list


EXTRA_POOL = [
    "pooja",
    "study",
    "store",
    "balcony",
    "garage",
    "utility",
    "foyer",
    "staircase",
    "unknown_extra",
]
DIRECTIONS = ["north", "south", "east", "west"]


def _random_extras(rng: random.Random) -> list[str]:
    count = rng.randint(0, 4)
    if count == 0:
        return []
    return rng.sample(EXTRA_POOL, k=count)


def _validate_rooms(rooms: list[dict]) -> list[str]:
    issues: list[str] = []
    if not rooms:
        issues.append("room list is empty")
        return issues

    room_types = {str(room.get("type", "")).strip().lower() for room in rooms}
    for required in ("living", "kitchen", "master_bedroom"):
        if required not in room_types:
            issues.append(f"missing core room type: {required}")

    for idx, room in enumerate(rooms):
        min_w = float(room.get("min_w", 0.0) or 0.0)
        min_h = float(room.get("min_h", 0.0) or 0.0)
        zone = int(room.get("zone", 0) or 0)
        if min_w <= 0.0 or min_h <= 0.0:
            issues.append(f"invalid min dimensions at index {idx}")
        if zone not in (1, 2, 3):
            issues.append(f"invalid zone at index {idx}")

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Fuzz robustness gate for deterministic room program builder.")
    parser.add_argument("--count", type=int, default=500, help="Number of fuzz cases.")
    parser.add_argument("--seed", type=int, default=19, help="Deterministic random seed.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on any failure.")
    parser.add_argument("--out", required=True, help="Output JSON report path.")
    args = parser.parse_args()

    rng = random.Random(int(args.seed))
    failures: list[dict[str, object]] = []

    total = max(1, int(args.count))
    for i in range(total):
        bedrooms = rng.randint(1, 4)
        plot_w = rng.uniform(20.0, 80.0)
        plot_l = rng.uniform(20.0, 120.0)
        usable_w = round(max(10.0, plot_w - 7.0), 2)
        usable_l = round(max(10.0, plot_l - 11.5), 2)
        facing = rng.choice(DIRECTIONS)
        floors = rng.randint(1, 3)
        extras = _random_extras(rng)

        case = {
            "bedrooms": bedrooms,
            "extras": extras,
            "usable_w": usable_w,
            "usable_l": usable_l,
            "facing": facing,
            "floors": floors,
        }

        try:
            rooms = build_room_list(**case)
            issues = _validate_rooms(rooms)
            if issues:
                failures.append({"index": i, "case": case, "issues": issues[:6]})
        except Exception as exc:
            failures.append({"index": i, "case": case, "issues": [f"exception: {exc}"]})

    report = {
        "ok": len(failures) == 0,
        "strict_mode": bool(args.strict),
        "seed": int(args.seed),
        "count": total,
        "failure_count": len(failures),
        "failures": failures[:40],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"fuzz: ok={report['ok']} count={total} failures={len(failures)} seed={args.seed}")

    if args.strict and failures:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
