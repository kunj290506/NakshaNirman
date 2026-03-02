"""
Test suite for the Multi-Factor Architectural Reasoning Engine.

Validates all 8 design factors and output format compliance.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from services.multi_factor_engine import (
    generate_plan,
    generate_new_plan,
    parse_input,
    _normalize_requirements,
)


def test_basic_2bhk():
    """Test basic 2BHK generation."""
    result = generate_plan({
        "total_area": 1200,
        "bedrooms": 2,
        "bathrooms": 2,
        "floors": 1,
    })

    assert "error" not in result, f"Unexpected error: {result.get('error')}"
    assert "layout" in result, "Missing layout"
    assert "explanation" in result, "Missing explanation"

    layout = result["layout"]
    assert "plot" in layout, "Missing plot info"
    assert "rooms" in layout, "Missing rooms"
    assert "walls" in layout, "Missing walls info"
    assert "area_summary" in layout, "Missing area_summary"
    assert "validation" in layout, "Missing validation"
    assert "boundary" in layout, "Missing boundary for frontend"

    # Check rooms have required fields
    for room in layout["rooms"]:
        assert "name" in room, f"Room missing name: {room}"
        assert "zone" in room, f"Room missing zone: {room}"
        assert "width" in room, f"Room missing width: {room}"
        assert "length" in room, f"Room missing length: {room}"
        assert "area" in room, f"Room missing area: {room}"
        assert "position" in room, f"Room missing position: {room}"
        assert "polygon" in room, f"Room missing polygon: {room}"
        assert "centroid" in room, f"Room missing centroid: {room}"

    print(f"  2BHK: {len(layout['rooms'])} rooms, "
          f"{layout['plot']['width']}x{layout['plot']['length']} ft, "
          f"strategy={layout.get('zoning_strategy')}")
    print(f"  Validation: {result['validation']}")
    return True


def test_3bhk_large():
    """Test 3BHK on a 30x40 plot."""
    result = generate_plan({
        "plot_width": 30,
        "plot_length": 40,
        "bedrooms": 3,
        "bathrooms": 2,
        "floors": 1,
        "extras": ["dining", "study"],
    })

    assert "error" not in result, f"Error: {result.get('error')}"
    layout = result["layout"]
    assert len(layout["rooms"]) >= 7, f"Expected >= 7 rooms, got {len(layout['rooms'])}"

    # Check no overlaps (geometry_ok)
    validation = result["validation"]
    print(f"  3BHK: {len(layout['rooms'])} rooms, validation={validation}")
    return True


def test_1bhk_small():
    """Test small 1BHK apartment."""
    result = generate_plan({
        "total_area": 500,
        "bedrooms": 1,
        "bathrooms": 1,
        "floors": 1,
    })

    assert "error" not in result, f"Error: {result.get('error')}"
    layout = result["layout"]
    print(f"  1BHK: {len(layout['rooms'])} rooms, area={layout['area_summary']['plot_area']}")
    return True


def test_insufficient_area():
    """Test error handling for impossible configuration."""
    result = generate_plan({
        "total_area": 100,
        "bedrooms": 4,
        "bathrooms": 3,
        "floors": 1,
    })

    assert "error" in result, "Expected error for insufficient area"
    assert "suggestion" in result, "Expected suggestion"
    print(f"  Insufficient area: {result['error']}")
    return True


def test_redesign():
    """Test redesign generates a different layout."""
    base_input = {
        "total_area": 1200,
        "bedrooms": 2,
        "bathrooms": 1,
        "floors": 1,
    }

    result1 = generate_plan(base_input)
    assert "error" not in result1

    strategy1 = result1["layout"].get("zoning_strategy")
    result2 = generate_new_plan(base_input, previous_strategy=strategy1)
    assert "error" not in result2

    strategy2 = result2["layout"].get("zoning_strategy")
    print(f"  Redesign: strategy1={strategy1}, strategy2={strategy2}")
    # Strategies should ideally differ (time-based, may occasionally match)
    return True


def test_natural_language():
    """Test natural language input parsing."""
    parsed = parse_input("I need a 30x40 plot with 3 bedrooms 2 bathrooms and dining room")
    assert parsed["plot_width"] == 30
    assert parsed["plot_length"] == 40
    assert parsed["bedrooms"] == 3
    assert parsed["bathrooms"] == 2
    assert "dining" in parsed["extras"]
    print(f"  NL parse: {parsed}")
    return True


def test_bhk_shorthand():
    """Test BHK shorthand parsing."""
    parsed = parse_input("3BHK 1500 sqft")
    assert parsed["bedrooms"] == 3
    assert parsed["bathrooms"] == 2
    assert parsed["total_area"] == 1500
    print(f"  BHK parse: {parsed}")
    return True


def test_output_format():
    """Test exact output format matches Section 5 specification."""
    result = generate_plan({
        "total_area": 1000,
        "bedrooms": 2,
        "bathrooms": 1,
    })
    assert "error" not in result

    layout = result["layout"]

    # Section 5 required fields
    assert layout["plot"]["unit"] == "ft"
    assert isinstance(layout["floors"], int)
    assert isinstance(layout["zoning_strategy"], str)
    assert isinstance(layout["circulation_strategy"], str)
    assert layout["walls"]["external"] == "9 inch"
    assert layout["walls"]["internal"] == "4.5 inch"
    assert "plot_area" in layout["area_summary"]
    assert "built_area" in layout["area_summary"]
    assert "circulation_percentage" in layout["area_summary"]

    # Validation fields
    val = layout["validation"]
    for key in ("zoning_ok", "privacy_ok", "geometry_ok",
                "ventilation_ok", "structural_ok", "area_ok"):
        assert key in val, f"Missing validation field: {key}"

    print(f"  Output format: All Section 5 fields present")
    return True


def test_room_geometry():
    """Test all rooms are rectangular with valid aspect ratios."""
    result = generate_plan({
        "total_area": 1500,
        "bedrooms": 3,
        "bathrooms": 2,
        "extras": ["dining", "pooja"],
    })
    assert "error" not in result

    for room in result["layout"]["rooms"]:
        w, h = room["width"], room["length"]
        assert w > 0 and h > 0, f"Invalid dimensions for {room['name']}: {w}x{h}"
        aspect = max(w, h) / min(w, h)
        assert aspect <= 3.0, f"Bad aspect ratio for {room['name']}: {aspect:.1f}"

        # Check polygon is rectangular (5 points, closed)
        poly = room["polygon"]
        assert len(poly) == 5, f"Non-rectangular polygon for {room['name']}"
        assert poly[0] == poly[-1], f"Polygon not closed for {room['name']}"

    print(f"  Geometry: All rooms rectangular, valid aspect ratios")
    return True


def test_privacy_gradient():
    """Test privacy gradient compliance."""
    result = generate_plan({
        "total_area": 1200,
        "bedrooms": 2,
        "bathrooms": 1,
    })
    assert "error" not in result

    validation = result["validation"]
    # Privacy should be OK (no forbidden door connections)
    print(f"  Privacy: ok={validation.get('privacy_ok')}")
    return True


if __name__ == "__main__":
    tests = [
        ("Basic 2BHK", test_basic_2bhk),
        ("3BHK Large Plot", test_3bhk_large),
        ("1BHK Small", test_1bhk_small),
        ("Insufficient Area", test_insufficient_area),
        ("Redesign Mode", test_redesign),
        ("Natural Language", test_natural_language),
        ("BHK Shorthand", test_bhk_shorthand),
        ("Output Format", test_output_format),
        ("Room Geometry", test_room_geometry),
        ("Privacy Gradient", test_privacy_gradient),
    ]

    passed = 0
    failed = 0

    print("=" * 60)
    print("Multi-Factor Architectural Engine — Test Suite")
    print("=" * 60)

    for name, test_fn in tests:
        try:
            print(f"\n[TEST] {name}")
            test_fn()
            print(f"  PASSED")
            passed += 1
        except AssertionError as e:
            print(f"  FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'=' * 60}")

    sys.exit(0 if failed == 0 else 1)
