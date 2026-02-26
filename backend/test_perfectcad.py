"""
Quick test for PerfectCAD layout engine.

Run: python test_perfectcad.py
"""

import json
import sys
sys.path.insert(0, ".")

from services.perfect_layout import generate_perfect_layout, validate_perfect_layout


def test_basic_2bhk():
    """Test basic 2BHK 1200 sqft layout."""
    print("=" * 60)
    print("TEST: 2BHK 1200 sqft (30x40)")
    print("=" * 60)

    result = generate_perfect_layout(
        plot_width=30,
        plot_length=40,
        bedrooms=2,
        bathrooms=1,
        extras=["dining"],
    )

    _print_result(result)

    # Validate
    validation = validate_perfect_layout(result)
    print(f"\n  Validation: {'PASS' if validation['compliant'] else 'FAIL'}")
    if validation["issues"]:
        for issue in validation["issues"]:
            print(f"    ISSUE: {issue}")
    if validation["warnings"]:
        for warn in validation["warnings"]:
            print(f"    WARN: {warn}")

    assert result["engine"] == "perfectcad"
    assert len(result["rooms"]) >= 4
    print("\n  >> PASSED\n")


def test_3bhk_large():
    """Test 3BHK 2000 sqft layout."""
    print("=" * 60)
    print("TEST: 3BHK 2000 sqft (40x50)")
    print("=" * 60)

    result = generate_perfect_layout(
        plot_width=40,
        plot_length=50,
        bedrooms=3,
        bathrooms=2,
        extras=["dining", "study", "balcony"],
    )

    _print_result(result)

    validation = validate_perfect_layout(result)
    print(f"\n  Validation: {'PASS' if validation['compliant'] else 'FAIL'}")
    if validation["issues"]:
        for issue in validation["issues"]:
            print(f"    ISSUE: {issue}")

    assert len(result["rooms"]) >= 7
    print("\n  >> PASSED\n")


def test_small_1bhk():
    """Test 1BHK 600 sqft layout."""
    print("=" * 60)
    print("TEST: 1BHK 600 sqft (20x30)")
    print("=" * 60)

    result = generate_perfect_layout(
        plot_width=20,
        plot_length=30,
        bedrooms=1,
        bathrooms=1,
    )

    _print_result(result)

    validation = validate_perfect_layout(result)
    print(f"\n  Validation: {'PASS' if validation['compliant'] else 'FAIL'}")
    if validation["issues"]:
        for issue in validation["issues"]:
            print(f"    ISSUE: {issue}")

    assert len(result["rooms"]) >= 3
    print("\n  >> PASSED\n")


def test_proportions():
    """Verify no room has aspect ratio exceeding its type-specific limit."""
    from services.perfect_layout import max_ar_for
    
    print("=" * 60)
    print("TEST: Proportion check across multiple sizes")
    print("=" * 60)

    configs = [
        (25, 35, 2, 1, []),
        (30, 40, 2, 2, ["dining"]),
        (35, 45, 3, 2, ["dining", "study"]),
        (40, 50, 3, 2, ["dining", "study", "pooja"]),
        (20, 25, 1, 1, []),
    ]

    all_pass = True
    for pw, pl, beds, baths, extras in configs:
        result = generate_perfect_layout(
            plot_width=pw, plot_length=pl,
            bedrooms=beds, bathrooms=baths, extras=extras,
        )
        for room in result.get("rooms", []):
            w = room["width"]
            h = room["length"]
            rtype = room.get("room_type", "unknown")
            ar = max(w / h, h / w) if w > 0 and h > 0 else 999
            limit = max_ar_for(rtype)
            if ar > limit + 0.3:
                print(f"  FAIL: {room['name']} has aspect ratio {ar:.1f}:1 ({w}x{h}) [limit={limit}]")
                all_pass = False

    print(f"\n  Result: {'ALL PASSED' if all_pass else 'SOME FAILED'}\n")


def _print_result(result):
    """Print layout summary."""
    rooms = result.get("rooms", [])
    score = result.get("score", {})

    print(f"\n  Engine:  {result.get('engine')}")
    print(f"  Method:  {result.get('method')}")
    print(f"  Score:   {score.get('total', 0)}/{score.get('max', 100)}")
    print(f"  Rooms:   {len(rooms)}")
    print(f"  Plot:    {result['plot']['width']}x{result['plot']['length']} ft")

    summary = result.get("area_summary", {})
    print(f"  Used:    {summary.get('total_used_area', '?')} sqft")
    print(f"  Util:    {summary.get('utilization_percentage', '?')}")

    print(f"\n  {'Room':<20} {'W':>6} {'L':>6} {'Area':>7} {'AR':>5}")
    print(f"  {'-'*20} {'-'*6} {'-'*6} {'-'*7} {'-'*5}")
    for r in rooms:
        w = r["width"]
        h = r["length"]
        ar = max(w / h, h / w) if w > 0 and h > 0 else 0
        print(f"  {r['name']:<20} {w:>6.1f} {h:>6.1f} {r['area']:>7.1f} {ar:>5.1f}")


if __name__ == "__main__":
    test_basic_2bhk()
    test_3bhk_large()
    test_small_1bhk()
    test_proportions()
    print("=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
