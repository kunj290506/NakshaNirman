"""
Generate fine-tuning training data using Claude API.
Run this ONCE. After it finishes you never need the API again.
Usage: python generate_training_data.py
Output: training_data.jsonl (keep this file safe)
"""

import anthropic
import json
import time

client = anthropic.Anthropic(api_key="YOUR_CLAUDE_API_KEY_HERE")

SYSTEM_PROMPT = """You are an expert Indian residential architect AI. Generate ONLY valid JSON floor plans. No explanation. No markdown. Only JSON.

Rules:
1. No two rooms overlap (verify mathematically)
2. All rooms within: x>=0, y>=0, x+width<=UW, y+height<=UL where UW=plot_width-7, UL=plot_length-11.5
3. Minimum sizes: living 11x11, dining 8x8, kitchen 7x8, master_bedroom 10x10, bedroom 9x9, bathroom 4x5, corridor 3.5 wide
4. Always include corridor for 2BHK+
5. Output exact schema with polygon arrays"""

SCENARIOS = [
    # (width, length, bedrooms, facing, family_type, city, extras_list)
    (30, 40, 2, "east", "nuclear", "Pune", []),
    (25, 35, 1, "north", "couple", "Mumbai", []),
    (35, 50, 3, "east", "nuclear", "Bangalore", ["pooja"]),
    (40, 60, 4, "north", "joint", "Delhi", ["pooja", "study"]),
    (28, 38, 2, "south", "nuclear", "Chennai", ["balcony"]),
    (32, 45, 3, "west", "nuclear", "Hyderabad", ["study"]),
    (22, 30, 1, "east", "couple", "Mumbai", []),
    (45, 65, 4, "north", "joint", "Ahmedabad", ["pooja", "study", "utility"]),
    (30, 42, 2, "east", "nuclear", "Kochi", ["balcony"]),
    (38, 55, 3, "north", "joint", "Jaipur", ["pooja", "store"]),
    (26, 36, 2, "south", "nuclear", "Nagpur", []),
    (33, 48, 3, "east", "nuclear", "Coimbatore", ["pooja"]),
    (36, 52, 3, "north", "joint", "Lucknow", ["utility"]),
    (28, 40, 2, "west", "couple", "Bangalore", ["study"]),
    (42, 58, 4, "east", "joint", "Hyderabad", ["pooja", "study", "store"]),
    (24, 34, 2, "north", "nuclear", "Jaipur", []),
    (50, 70, 4, "north", "joint", "Chennai", ["pooja", "study", "utility", "balcony"]),
    (30, 44, 3, "east", "nuclear", "Pune", ["study"]),
    (34, 48, 3, "south", "nuclear", "Mumbai", ["pooja"]),
    (38, 54, 3, "west", "nuclear", "Delhi", ["balcony"]),
    (26, 38, 2, "east", "nuclear", "Kochi", ["pooja"]),
    (44, 62, 4, "north", "joint", "Ahmedabad", ["pooja", "study", "garage"]),
    (32, 46, 3, "east", "couple", "Bangalore", ["study"]),
    (28, 36, 2, "north", "nuclear", "Chandigarh", []),
    (36, 50, 3, "east", "joint", "Surat", ["pooja", "utility"]),
    (30, 40, 2, "north", "nuclear", "Indore", ["balcony"]),
    (40, 56, 4, "east", "joint", "Kolkata", ["pooja", "study"]),
    (25, 35, 2, "south", "couple", "Bhopal", []),
    (35, 48, 3, "north", "nuclear", "Nagpur", ["pooja"]),
    (42, 60, 4, "west", "joint", "Mysuru", ["pooja", "study", "store"]),
]

def generate_one(width, length, bedrooms, facing, family_type, city, extras):
    extras_str = ", ".join(extras) if extras else "none"
    user_msg = (
        f"{width}x{length} plot, {facing}-facing, {bedrooms}BHK, "
        f"{family_type} family, {city}. "
        f"Extra rooms: {extras_str}. "
        f"Output only JSON."
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=3500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}]
        )
        raw = response.content[0].text.strip()

        # Validate JSON
        parsed = json.loads(raw)
        assert "rooms" in parsed and len(parsed["rooms"]) >= 3
        assert "metadata" in parsed

        return {
            "messages": [
                {
                    "role": "system",
                    "content": "You are NAKSHA-MASTER, a specialized AI architect for Indian residential floor plans. Output only valid JSON floor plans following the NakshaNirman schema. Verify zero room overlaps before output."
                },
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": raw}
            ]
        }
    except json.JSONDecodeError as e:
        print(f"  SKIP - invalid JSON: {e}")
        return None
    except AssertionError as e:
        print(f"  SKIP - incomplete plan: {e}")
        return None
    except Exception as e:
        print(f"  SKIP - error: {e}")
        return None

# Generate dataset
dataset = []
for i, scenario in enumerate(SCENARIOS):
    width, length, bedrooms, facing, family_type, city, extras = scenario
    print(f"[{i+1}/{len(SCENARIOS)}] Generating {width}x{length} {bedrooms}BHK {facing} {city}...")

    example = generate_one(*scenario)
    if example:
        dataset.append(example)
        print(f"  OK - {len(dataset)} examples collected")

    time.sleep(1)  # Rate limiting

# Save
output_file = "training_data.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for item in dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"\nDone! {len(dataset)} training examples saved to {output_file}")
print("Now run: python finetune.py")
