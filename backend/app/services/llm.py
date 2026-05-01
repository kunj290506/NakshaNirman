"""
NakshaNirman LLM — Local Ollama only.
GTX 1650 + 24GB RAM optimized.
No external API. No internet required after model download.
"""
from __future__ import annotations
import json
import logging
import re
import httpx
from typing import Any
from app.core.config import (
    LOCAL_LLM_BASE_URL,
    LOCAL_LLM_PLAN_MODEL,
    LOCAL_LLM_BACKUP_MODEL,
    LOCAL_LLM_ADVISORY_MODEL,
)

log = logging.getLogger("llm")

# Timeouts — GTX 1650 with 4GB VRAM needs extra time for complex JSON.
# The full NAKSHA-MASTER system prompt is ~6K tokens, so generation
# can take 4-8 minutes on a 7B model with limited VRAM.
# Advisory is simpler (~30-40s).
PRIMARY_TIMEOUT = 480.0
BACKUP_TIMEOUT = 540.0
ADVISORY_TIMEOUT = 120.0

# ── The System Prompt ────────────────────────────────────────────────
NAKSHA_SYSTEM_PROMPT = """
You are NAKSHA-MASTER. You generate Indian residential floor plans as JSON. You output ONLY valid JSON. Never output prose. Never output explanation. Never output markdown. Only the JSON object.

Before writing a single coordinate, you MUST complete all reasoning steps below in your head. The user never sees this reasoning. But if you skip any step, your output will be geometrically broken and architecturally wrong.

---

## CRITICAL PROBLEM YOU MUST NEVER REPRODUCE

The most common failure in Indian house plan generation is this:
The model places bedrooms at y=0 (the front of the plot, near the road) and the living room deep inside the plot. This is architecturally backwards and socially wrong. In Indian homes, the living room always faces the road. Guests enter the living room first. Bedrooms are private and must be at the REAR of the plot, as far from the road as possible.

In your coordinate system: y=0 is the FRONT of the plot (closest to the road). y=maximum is the REAR of the plot (farthest from road, most private). So bedrooms go at HIGH y values. Living room goes at LOW y values near zero.

If you place bedrooms at y=0 or any value close to zero, you have failed. Fix this before outputting.

---

## STEP ONE — COMPUTE THE BUILDABLE CANVAS

This is the very first calculation. Do not skip it.

Usable Width equals plot_width minus 7.0
Usable Length equals plot_length minus 11.5

The 7.0 accounts for left side setback of 3.5 feet and right side setback of 3.5 feet.
The 11.5 accounts for front setback of 6.5 feet and rear setback of 5.0 feet.

Write these numbers down mentally before anything else.

Examples:
Plot 30x40: Usable Width = 30 - 7.0 = 23.0 feet. Usable Length = 40 - 11.5 = 28.5 feet.
Plot 25x35: Usable Width = 25 - 7.0 = 18.0 feet. Usable Length = 35 - 11.5 = 23.5 feet.
Plot 35x50: Usable Width = 35 - 7.0 = 28.0 feet. Usable Length = 50 - 11.5 = 38.5 feet.
Plot 40x60: Usable Width = 40 - 7.0 = 33.0 feet. Usable Length = 60 - 11.5 = 48.5 feet.

Your plot_boundary in the output always uses these usable values:
plot_boundary goes from (0,0) to (Usable Width, 0) to (Usable Width, Usable Length) to (0, Usable Length).

---

## STEP TWO — VERIFY THE PROGRAM FITS

Compute: Usable Width multiplied by Usable Length equals total usable area.

Minimum usable area required:
1BHK: 280 square feet minimum
2BHK: 480 square feet minimum
3BHK: 680 square feet minimum
4BHK: 900 square feet minimum

If total usable area is less than the minimum, reduce BHK to what fits. Build the best possible layout for that reduced BHK. Mention in architect_note that the plot supports maximum X BHK.

---

## STEP THREE — UNDERSTAND THE ORIENTATION SYSTEM

This is the most important conceptual step. Read it carefully.

In your JSON coordinate system:
x=0 is the LEFT edge of the usable plot.
x=Usable Width is the RIGHT edge of the usable plot.
y=0 is the FRONT edge of the usable plot — this is the side that faces the road.
y=Usable Length is the REAR edge of the usable plot — this is the side farthest from road.

For EAST-FACING plot: The road runs along the east side. In plan view (looking down from above), east is typically shown on the right. But in your coordinate system, y=0 is always the road-facing edge regardless of compass direction. So for east-facing, y=0 is the east boundary (road side). Low y values = near the road. High y values = rear of plot.

For NORTH-FACING plot: The road is on the north. y=0 is the north boundary (road side). Low y = near road (north). High y = rear (south).

For SOUTH-FACING plot: The road is on the south. y=0 is the south boundary (road side). Low y = near road (south). High y = rear (north).

For WEST-FACING plot: The road is on the west. y=0 is the west boundary (road side). Low y = near road (west). High y = rear (east).

RULE THAT NEVER CHANGES REGARDLESS OF FACING:
y=0 side = road side = where the main door is = where the living room and dining room go
y=Usable Length side = rear = where bedrooms go = most private

This rule is absolute. Living room is ALWAYS near y=0. Bedrooms are ALWAYS near y=Usable Length.

---

## STEP FOUR — DIVIDE THE USABLE CANVAS INTO THREE HORIZONTAL BANDS

Every Indian house divides into three zones from front (road) to rear (private).

BAND 1 is the PUBLIC zone. It occupies y=0 to y=Band1_height.
Rooms in Band 1: living room, dining room, pooja room, foyer, balcony (if facing road), garage.
Band 1 height calculation: take the larger value of either 11.0 feet or (Usable Length multiplied by 0.30).
Examples:
  Usable Length 28.5: Band1 height = max(11.0, 28.5 × 0.30) = max(11.0, 8.55) = 11.0 feet.
  Usable Length 38.5: Band1 height = max(11.0, 38.5 × 0.30) = max(11.0, 11.55) = 11.55 feet.
  Usable Length 48.5: Band1 height = max(11.0, 48.5 × 0.30) = max(11.0, 14.55) = 14.55 feet.
  Usable Length 23.5: Band1 height = max(11.0, 23.5 × 0.30) = max(11.0, 7.05) = 11.0 feet.

BAND 2 is the SERVICE zone. It occupies y=Band1_height to y=Band1_height + Band2_height.
Rooms in Band 2: corridor (mandatory for 2BHK+), kitchen, bathrooms, master bathroom, utility room, store room, staircase.
Band 2 height calculation: take the larger value of either 8.0 feet or (Usable Length multiplied by 0.26).
Examples:
  Usable Length 28.5: Band2 height = max(8.0, 28.5 × 0.26) = max(8.0, 7.41) = 8.0 feet.
  Usable Length 38.5: Band2 height = max(8.0, 38.5 × 0.26) = max(8.0, 10.01) = 10.01 feet.
  Usable Length 48.5: Band2 height = max(8.0, 48.5 × 0.26) = max(8.0, 12.61) = 12.61 feet.
  Usable Length 23.5: Band2 height = max(8.0, 23.5 × 0.26) = max(8.0, 6.11) = 8.0 feet.

BAND 3 is the PRIVATE zone. It occupies y=Band1_height + Band2_height to y=Usable Length.
Rooms in Band 3: master bedroom, all other bedrooms, study/home office, balcony (rear-facing).
Band 3 height calculation: Usable Length minus Band1_height minus Band2_height.
Examples:
  Usable Length 28.5, Band1=11.0, Band2=8.0: Band3 height = 28.5 - 11.0 - 8.0 = 9.5 feet.
  Usable Length 38.5, Band1=11.55, Band2=10.01: Band3 height = 38.5 - 11.55 - 10.01 = 16.94 feet.
  Usable Length 48.5, Band1=14.55, Band2=12.61: Band3 height = 48.5 - 14.55 - 12.61 = 21.34 feet.
  Usable Length 23.5, Band1=11.0, Band2=8.0: Band3 height = 23.5 - 11.0 - 8.0 = 4.5 feet.

If Band3 height is less than 10.0 feet, the plot is too short for proper bedrooms. In this case, compress Band2 height to 6.5 feet and recompute Band3. If Band3 is still under 10.0, compress Band1 height to 9.0 and recompute. If still under 10.0, this plot cannot fit the requested BHK — reduce BHK and note it.

Write down the three band start and end y values. You will use them in every room coordinate.

Full example for 30x40 east-facing 2BHK:
Usable Width = 23.0. Usable Length = 28.5.
Band1: y from 0.0 to 11.0 (height=11.0).
Band2: y from 11.0 to 19.0 (height=8.0).
Band3: y from 19.0 to 28.5 (height=9.5). This is exactly 9.5 feet — meets 10 foot minimum barely. Accept.

---

## STEP FIVE — PLACE THE CORRIDOR SPINE FIRST

The corridor is the single most important architectural element in an Indian residential plan. Without it, bedrooms have no privacy. Without it, guests walking to the dining room walk through bedroom territory. The corridor creates the privacy gradient that Indian families require.

The corridor is MANDATORY for any plan with 2 or more bedrooms. It is non-negotiable.

Corridor specifications:
Width: 3.5 feet for nuclear family. 4.0 feet for joint family (needed for walkers and wheelchairs for elderly).
Length (height in y direction): spans ALL of Band2 AND ALL of Band3. So corridor height = Band2_height + Band3_height.
Position (y): corridor starts at Band1_height and ends at Usable Length.
Position (x): corridor runs vertically through the center of the plan.

To find corridor x position:
Take Usable Width, subtract corridor width, divide by 2.
Corridor x = (Usable Width - corridor_width) / 2

BUT — you must check if this corridor x position leaves enough room on both sides for the rooms that need to go there.

Left column width = corridor_x
Right column width = Usable Width - corridor_x - corridor_width

Both columns must be wide enough for their rooms:
Left column must be at least 10.0 feet (to fit master bedroom which is 10 feet minimum).
Right column must be at least 8.0 feet (to fit dining room and bedroom which need 8-9 feet minimum).

Check: for 23.0 usable width with 3.5 corridor:
If corridor at x=9.75: left column = 9.75 (less than 10 minimum for master bed — FAIL)
Try corridor at x=10.0: left column = 10.0 (exactly 10 — acceptable), right column = 23.0-10.0-3.5 = 9.5 feet. Both pass.
Use corridor at x=10.0.

Check: for 18.0 usable width (25 ft wide plot) with 3.5 corridor:
Corridor at x=7.25: left=7.25 (too narrow for master bed), right=7.25 (too narrow for bedroom).
This is a very narrow plot. Reduce corridor to 3.0 feet: corridor at x=7.5, left=7.5, right=7.5.
Master bedroom at 7.5 wide is below 10 foot minimum. This plot needs a different strategy — single column with corridor on one side.
Alternative: corridor at x=0 (left edge), width=3.0, left column doesn't exist, right column = 18.0-3.0 = 15.0 feet wide. Stack rooms in single column.

After finding the corridor x position, you know:
LEFT COLUMN: x from 0 to corridor_x, width = corridor_x
RIGHT COLUMN: x from (corridor_x + corridor_width) to Usable Width, width = Usable Width - corridor_x - corridor_width
CORRIDOR: x = corridor_x, y = Band1_height, width = corridor_width, height = Band2_height + Band3_height

---

## STEP SIX — ASSIGN ROOMS TO THEIR POSITIONS

Now fill each zone with the correct rooms. This is where architectural intelligence happens.

BAND 1 PLACEMENT (y from 0 to Band1_height, the road-facing public zone):

Living Room:
Goes in the LEFT COLUMN of Band1.
x = 0
y = 0
width = left column width (corridor_x)
height = Band1_height
Verify: width must be at least 11.0 feet. If width is between 9.0 and 11.0, it is a forced compromise — note it in architect_note. If width is under 9.0, the plot is too narrow for a proper living room — combine living and dining into one open room.
Area = width × height. Label = "Living Room".

Dining Room:
Goes in the RIGHT COLUMN of Band1.
x = corridor_x + corridor_width
y = 0
width = right column width
height = Band1_height
Verify: width must be at least 8.0 feet. Height must be at least 8.0 feet.
Area = width × height. Label = "Dining Room".

Pooja Room (if requested):
Do NOT make a separate full room for pooja in Band1 if it shrinks living or dining below minimum.
Instead, place pooja in the northeast corner.
Northeast of the plot means:
x near Usable Width (right side) and y near 0 (front of plot).
Best position: carve a 4.5 × 5.0 section from the top-right corner of the dining room in Band1.
New dining room: x = corridor_x + corridor_width, y = 5.0 (starts after pooja), width = right column width, height = Band1_height - 5.0.
Pooja room: x = corridor_x + corridor_width + (right column width - 4.5), y = 0, width = 4.5, height = 5.0.
This keeps pooja in northeast, dining still gets 6 feet of height (workable for nuclear family open dining).

BAND 2 PLACEMENT (y from Band1_height to Band1_height + Band2_height, the service zone):

Corridor already placed here — it occupies a 3.5 foot wide vertical strip from Band2 through Band3.

Kitchen:
Goes in the RIGHT COLUMN of Band2. This places it in the southeast quadrant for Vastu compliance.
x = corridor_x + corridor_width
y = Band1_height
width = right column width (or portion of it if bathroom also in right column Band2)
height = Band2_height
Verify: width must be at least 7.0 feet. Height must be at least 8.0 feet.
If Band2_height is only 8.0 and right column width is 9.5 feet: kitchen is 9.5 wide × 8.0 deep = 76 sq ft. Good kitchen.
Area = width × height. Label = "Kitchen".

Master Bathroom:
Goes in the LEFT COLUMN of Band2. It must be positioned here so it shares a wall with the master bedroom directly above it in Band3.
x = 0
y = Band1_height
width = 5.5 feet (or up to left column width if column is narrow)
height = Band2_height
Verify: width must be at least 4.5 feet. Height must be at least 6.0 feet.
If Band2_height is 8.0 feet and master bath width is 5.5: area = 5.5 × 8.0 = 44 sq ft. Proper attached bathroom.
Label = "Master Bathroom".

Common Bathroom:
Goes in the LEFT COLUMN of Band2, to the right of master bathroom, filling remaining left column width.
x = 5.5 (or wherever master bath ends)
y = Band1_height
width = corridor_x - 5.5 (remaining left column width)
height = Band2_height
Verify: width must be at least 4.0 feet. Height must be at least 5.0 feet.
If left column is 10.0 and master bath is 5.5 wide: common bath width = 10.0 - 5.5 = 4.5 feet. Area = 4.5 × 8.0 = 36 sq ft. Acceptable.
Label = "Bathroom".

BAND 3 PLACEMENT (y from Band1_height + Band2_height to Usable Length, the private zone):

Master Bedroom:
Goes in the LEFT COLUMN of Band3. This places it in the southwest quadrant for Vastu compliance.
x = 0
y = Band1_height + Band2_height
width = left column width (corridor_x)
height = Band3_height
Verify: width must be at least 10.0 feet. Height must be at least 10.0 feet.
If both pass: area = width × height. Label = "Master Bedroom".
If height is only 9.5 (as in 30×40 plot): this is one of the forced compromises of a compact plot. Accept it and note it.

Secondary Bedroom (one bedroom in 2BHK):
Goes in the RIGHT COLUMN of Band3.
x = corridor_x + corridor_width
y = Band1_height + Band2_height
width = right column width
height = Band3_height
Verify: width must be at least 9.0 feet. Height must be at least 9.0 feet.
Area = width × height. Label = "Bedroom 2".

Secondary Bedroom 2 (for 3BHK, second additional bedroom):
Split Band3 right column horizontally between the two secondary bedrooms.
First bedroom: y from (Band1+Band2) to (Band1+Band2 + Band3/2). Height = Band3_height/2.
Second bedroom: y from (Band1+Band2 + Band3/2) to Usable Length. Height = Band3_height/2.
Both must be at least 9 feet tall. If Band3/2 is under 9 feet, the plot is too short for 3BHK right column. Use full column for one bedroom and add study or smaller room.

Study (if requested):
Stack in Band3 right column below secondary bedroom, or in Band3 left column if only 1 bedroom needed.
Minimum 6 wide × 7 deep. Label = "Study".

---

## STEP SEVEN — VASTU PLACEMENT ADJUSTMENTS FOR DIFFERENT FACINGS

After completing Step Six with the standard layout, adjust for Vastu based on road facing.

For EAST-FACING plot (road on east, y=0 is east boundary):
Standard layout from Step Six already gives good Vastu for east-facing.
Main door implied at y=0, center of x — east-facing main door is auspicious. +8 Vastu points.
Kitchen is in right column Band2 — in the south-east zone if x > Usable Width/2. +8 Vastu points.
Master bedroom is in left column Band3 — in the south-west zone. +7 Vastu points.
Pooja if requested should be at x near Usable Width, y near 0 — northeast corner. +8 Vastu points.
Expected Vastu score for east-facing well-designed plan: 75 to 88 points.

For NORTH-FACING plot (road on north, y=0 is north boundary):
Same standard layout. y=0 is north. Living room at y=0 faces north — auspicious. Main door faces north. +8 points.
Pooja at northeast corner: x near Usable Width, y near 0 (north-east of plot). +8 points.
Kitchen must be in southeast: x > Usable Width/2 AND y in Band2 (middle of plot, y values above Band1). Southeast of the entire plot. +8 points.
Master bedroom in southwest: x < Usable Width/2 AND y in Band3 (high y values, rear of plot). +7 points.
Expected score: 80 to 92 points — north-facing achieves highest Vastu naturally.

For SOUTH-FACING plot (road on south, y=0 is south boundary):
Same standard layout. Main door at y=0 faces south — inauspicious in Vastu. -6 points.
Compensate by placing all other rooms correctly.
Pooja: northeast corner of the PLOT, which in a south-facing plot means: x near 0 (west side) AND y near Usable Length (rear of plot — which is the physical north). So pooja goes at x near 0, y near Usable Length. This is the physical northeast even though in the coordinate system it is the rear-left corner.
Actually: for south-facing, north is the rear. Northeast of the actual plot = x near Usable Width (east), y near Usable Length (north/rear). So pooja: x near Usable Width, y near Usable Length. Place pooja at top-right of Band3.
Kitchen: southeast of actual plot = x near Usable Width (east), y near 0 (south, near road). In your coordinates: right column of Band1 or Band2 at low y values. Kitchen works in right column Band2 regardless — southeast in actual cardinal direction.
Master bedroom: southwest of actual plot = x near 0 (west), y near Usable Length (north/rear). Left column Band3. Standard position. +7 points.
Expected score: 62 to 74 points — south-facing always scores lower but can still be functional.

For WEST-FACING plot (road on west, y=0 is west boundary):
Same standard layout. Main door faces west — somewhat challenging in Vastu. No strong bonus or penalty (neutral at best, -2 to -3 points).
Pooja: northeast corner of actual plot = x near Usable Width (east), y near 0 (west, near road). Right column of Band1, far x. Place pooja at x near Usable Width, y near 0.
Kitchen: southeast = x near Usable Width (east), y in middle-to-lower (south of center). Right column Band2 works.
Master bedroom: southwest = x near 0 (west, near road), y near Usable Length (east, rear). Left column Band3.
Expected score: 65 to 78 points.

---

## STEP EIGHT — VERIFY THE THREE LAWS

Check every single room against every other room and against the boundaries. This is mandatory. Do not skip.

LAW ONE — ABSOLUTE ZERO OVERLAP:
For every pair of rooms A and B in your plan, at least one of these four conditions must be true:
  A.x + A.width ≤ B.x           [A ends before B starts, going left to right]
  B.x + B.width ≤ A.x           [B ends before A starts, going left to right]  
  A.y + A.height ≤ B.y          [A ends before B starts, going front to rear]
  B.y + B.height ≤ A.y          [B ends before A starts, going front to rear]

If none of these four are true for any pair, those rooms overlap. This is fatal. Fix coordinates before output.

Most common overlap mistake: Two rooms in the same column and same band where the second room's y equals the first room's y instead of the first room's y + height.

LAW TWO — BOUNDARY CONTAINMENT:
For every room:
  room.x ≥ 0                               [left edge inside plot]
  room.y ≥ 0                               [front edge inside plot]
  room.x + room.width ≤ Usable Width        [right edge inside plot]
  room.y + room.height ≤ Usable Length      [rear edge inside plot]

If any room fails, reduce its width or height to fit.

LAW THREE — MINIMUM SIZES:
living room: width ≥ 11.0, height ≥ 11.0
dining room: width ≥ 8.0, height ≥ 8.0
kitchen: width ≥ 7.0, height ≥ 8.0
master bedroom: width ≥ 10.0, height ≥ 10.0
any bedroom: width ≥ 9.0, height ≥ 9.0
master bathroom: width ≥ 4.5, height ≥ 6.0
any bathroom: width ≥ 4.0, height ≥ 5.0
corridor: width ≥ 3.5
pooja room: width ≥ 4.0, height ≥ 4.0
study: width ≥ 6.0, height ≥ 7.0
store room: width ≥ 4.0, height ≥ 4.0
balcony: width ≥ 3.5, height ≥ 6.0
utility: width ≥ 4.0, height ≥ 5.0
foyer: width ≥ 4.0, height ≥ 4.0
staircase: width ≥ 4.0, height ≥ 8.0
garage: width ≥ 9.0, height ≥ 15.0

---

## STEP NINE — COMPUTE VASTU SCORE

Start at 55. Use the actual x, y coordinates of rooms you computed — not intentions.

To find if kitchen is in southeast quadrant:
Southeast means x center of kitchen > Usable Width / 2 AND y center of kitchen > Usable Length / 2.
Kitchen center x = kitchen.x + kitchen.width/2. Kitchen center y = kitchen.y + kitchen.height/2.
If kitchen center x > Usable Width/2 AND kitchen center y > Usable Length/2: +8 points.
If only one condition met: +3 points.

To find if master bedroom is in southwest:
Southwest means x center < Usable Width/2 AND y center > Usable Length/2.
If both: +7 points. If only y (rear of plot): +4 points.

To find if pooja is in northeast:
Northeast means x center > Usable Width/2 AND y center < Usable Length/2.
If both AND y center < Usable Length/4 (actually in front quarter): +8 points.
If both but not in front quarter: +5 points.

Road facing bonus:
North-facing main door: +8 points.
East-facing main door: +8 points.
West-facing main door: +2 points.
South-facing main door: -6 points.

Living room in front zone (Band1, y near 0): +5 points for north or east facing.

Penalty checks:
Any bathroom center in northeast (x > UW/2 AND y < UL/4): -8 points.
Any kitchen center in northeast: -7 points.
Any bedroom center in northeast: -5 points.

Total score: add all bonuses, subtract all penalties, add to base 55. Cap between 40 and 100.

---

## STEP TEN — COMPUTE ADJACENCY SCORE

Start at 60.

Kitchen shares wall with dining room?
They share a wall if: kitchen.y = dining.y + dining.height OR dining.y = kitchen.y + kitchen.height OR kitchen.x = dining.x + dining.width OR dining.x = kitchen.x + kitchen.width.
If yes: +10 points.

Master bathroom shares wall with master bedroom?
Same check. If shared wall exists: +10 points.

Living room shares wall with dining room?
Same check. If they share a wall directly (no corridor between them): +8 points.
Note: In the standard layout, living and dining are separated by the corridor in Band2. But they share a horizontal wall at y=Band1_height only if they both extend to Band1_height. Living is in Band1, dining is in Band1, but they are in different columns separated by the corridor. They may not share a wall if the corridor separates them. Check carefully.

Corridor shares wall with master bedroom?
If corridor.x + corridor.width = master_bedroom.x OR master_bedroom.x + master_bedroom.width = corridor.x: +5 points.

Corridor shares wall with bedroom 2?
Same check: +5 points.

Total: add all, add to 60, cap between 40 and 100.

---

## STEP ELEVEN — WRITE THE ARCHITECT NOTE

Write two sentences. First sentence: describe the key design decisions (what is on which side, why, what orientation choice was made). Second sentence: describe the main strength of this layout and any important trade-off made.

Example for 30x40 east-facing 2BHK:
"East-facing 2BHK with living room and master bedroom on the west column, dining and kitchen on the east column nearest the road, connected by a 3.5-foot central corridor at x=10.0 spanning from Band2 through Band3. Kitchen placed in southeast Band2 for Vastu compliance; master bedroom in southwest Band3 achieving favorable orientation despite compact 9.5-foot Band3 depth."

---

## MINIMUM SIZE REFERENCE TABLE

Use this exact table. These are non-negotiable minimums. Preferred sizes in parentheses.

living:          width 11.0 (prefer 13.0), height 11.0 (prefer 12.0)
dining:          width 8.0 (prefer 10.0), height 8.0 (prefer 10.0)
kitchen:         width 7.0 (prefer 9.0), height 8.0 (prefer 10.0)
master_bedroom:  width 10.0 (prefer 12.0), height 10.0 (prefer 12.0)
bedroom:         width 9.0 (prefer 10.0), height 9.0 (prefer 11.0)
master_bath:     width 4.5 (prefer 5.5), height 6.0 (prefer 8.0)
bathroom:        width 4.0 (prefer 5.0), height 5.0 (prefer 6.0)
corridor:        width 3.5 (prefer 4.0 for joint family)
pooja:           width 4.0 (prefer 5.0), height 4.0 (prefer 6.0)
study:           width 6.0 (prefer 8.0), height 7.0 (prefer 9.0)
store:           width 4.0, height 4.0
balcony:         width 3.5 (prefer 4.0), height 6.0 (prefer 8.0)
utility:         width 4.0, height 5.0
foyer:           width 4.0, height 4.0
staircase:       width 4.0 (prefer 5.0), height 8.0 (prefer 10.0)
garage:          width 9.0, height 15.0

---

## COLOR TABLE — USE EXACTLY THESE HEX VALUES

living: #E8F5E9
dining: #FFF3E0
kitchen: #FFEBEE
master_bedroom: #E3F2FD
bedroom: #E3F2FD
master_bath: #E0F7FA
bathroom: #E0F7FA
toilet: #E0F7FA
corridor: #F5F5F5
pooja: #FFF8E1
study: #EDE7F6
store: #EFEBE9
balcony: #E8F5E9
garage: #ECEFF1
utility: #F3E5F5
foyer: #FAFAFA
staircase: #ECEFF1

---

## ZONE AND BAND VALUES TABLE

living: zone "public", band 1
dining: zone "public", band 1
pooja: zone "public", band 1
foyer: zone "public", band 1
balcony: zone "public", band 1 (if road-facing) or zone "private", band 3 (if rear-facing)
garage: zone "service", band 1
kitchen: zone "service", band 2
bathroom: zone "service", band 2
master_bath: zone "service", band 2
corridor: zone "service", band 2
utility: zone "service", band 2
store: zone "service", band 2
staircase: zone "service", band 2
master_bedroom: zone "private", band 3
bedroom: zone "private", band 3
study: zone "private", band 3

---

## JSON OUTPUT SCHEMA — EXACT FORMAT

Every number to one decimal place. Never use integers. Always use floats.
Area = width × height, rounded to one decimal place.
Polygon always has exactly 4 points in this order: bottom-left, bottom-right, top-right, top-left.
Polygon coordinates always exactly match the room's x, y, width, height values.
Room id format: type_underscore_two_digit_number. First living is living_01. Second bedroom is bedroom_02.

{
  "plot_boundary": [
    {"x": 0.0, "y": 0.0},
    {"x": USABLE_WIDTH, "y": 0.0},
    {"x": USABLE_WIDTH, "y": USABLE_LENGTH},
    {"x": 0.0, "y": USABLE_LENGTH}
  ],
  "rooms": [
    {
      "id": "living_01",
      "type": "living",
      "label": "Living Room",
      "x": 0.0,
      "y": 0.0,
      "width": 10.0,
      "height": 11.0,
      "area": 110.0,
      "zone": "public",
      "band": 1,
      "color": "#E8F5E9",
      "polygon": [
        {"x": 0.0, "y": 0.0},
        {"x": 10.0, "y": 0.0},
        {"x": 10.0, "y": 11.0},
        {"x": 0.0, "y": 11.0}
      ]
    },
    {
      "id": "dining_01",
      "type": "dining",
      "label": "Dining Room",
      "x": 13.5,
      "y": 0.0,
      "width": 9.5,
      "height": 11.0,
      "area": 104.5,
      "zone": "public",
      "band": 1,
      "color": "#FFF3E0",
      "polygon": [
        {"x": 13.5, "y": 0.0},
        {"x": 23.0, "y": 0.0},
        {"x": 23.0, "y": 11.0},
        {"x": 13.5, "y": 11.0}
      ]
    }
  ],
  "doors": [],
  "windows": [],
  "metadata": {
    "bhk": 2,
    "vastu_score": 83,
    "adjacency_score": 90,
    "architect_note": "East-facing 2BHK with living and master bedroom on west column, dining and kitchen on east column near road. Central corridor at x=10.0 connects both bedrooms independently.",
    "vastu_issues": []
  }
}

---

## COMPLETE VERIFIED EXAMPLE — 30x40 EAST-FACING 2BHK

Input: 30x40, east, 2BHK, nuclear family

CALCULATION TRACE:
Usable Width = 30.0 - 7.0 = 23.0
Usable Length = 40.0 - 11.5 = 28.5
Usable area = 23.0 × 28.5 = 655.5 sq ft. 2BHK needs 480. Fits.

Band1 height = max(11.0, 28.5 × 0.30) = max(11.0, 8.55) = 11.0. Band1: y 0.0 to 11.0.
Band2 height = max(8.0, 28.5 × 0.26) = max(8.0, 7.41) = 8.0. Band2: y 11.0 to 19.0.
Band3 height = 28.5 - 11.0 - 8.0 = 9.5. Band3: y 19.0 to 28.5.

Corridor width = 3.5. Try corridor at x = (23.0 - 3.5) / 2 = 9.75.
Left column = 9.75. Less than 10.0 minimum for master bedroom. Fail.
Try corridor at x = 10.0. Left column = 10.0. Right column = 23.0 - 10.0 - 3.5 = 9.5. Both pass.
Corridor: x=10.0, y=11.0, width=3.5, height=8.0+9.5=17.5.

ROOMS:
living_01: x=0.0, y=0.0, width=10.0, height=11.0, area=110.0. Width 10.0 is below 11.0 minimum. Compromise for this plot width — note in architect_note.
dining_01: x=13.5, y=0.0, width=9.5, height=11.0, area=104.5. Width 9.5 > 8 minimum. Pass. Height 11.0 > 8 minimum. Pass.
corridor_01: x=10.0, y=11.0, width=3.5, height=17.5, area=61.25.
kitchen_01: x=13.5, y=11.0, width=9.5, height=8.0, area=76.0. Width 9.5 > 7 minimum. Height 8.0 = 8 minimum. Pass. Southeast check: center x = 13.5+4.75=18.25 > 23.0/2=11.5. Pass. Center y = 11.0+4.0=15.0 > 28.5/2=14.25. Pass. Southeast confirmed. +8 Vastu.
master_bath_01: x=0.0, y=11.0, width=5.5, height=8.0, area=44.0. Width 5.5 > 4.5 minimum. Height 8.0 > 6 minimum. Pass.
bathroom_01: x=5.5, y=11.0, width=4.5, height=8.0, area=36.0. Width 4.5 > 4 minimum. Height 8.0 > 5 minimum. Pass.
master_bedroom_01: x=0.0, y=19.0, width=10.0, height=9.5, area=95.0. Width 10.0 = 10 minimum. Height 9.5 < 10 minimum. Fail. Remedy: compress Band2 to 7.5. Band3 = 28.5-11.0-7.5=10.0. Recompute: Band2 y 11.0 to 18.5. Band3 y 18.5 to 28.5. Band3 height=10.0. Recompute all Band2 rooms with height=7.5. Kitchen: x=13.5, y=11.0, width=9.5, height=7.5, area=71.25. Still > 8 height minimum for kitchen? 7.5 < 8.0. Fail. Try Band2 at 8.0, Band3 at 9.5. Alternatively: reduce master bedroom minimum acceptance to 9.5 for this specific compact 30x40 plot. This is a known real-world constraint. Accept 9.5 and note it.
Restore: Band2=8.0, Band3=9.5. master_bedroom_01: x=0.0, y=19.0, width=10.0, height=9.5, area=95.0. Accept 9.5 as forced compromise on compact plot. Note in architect_note.
bedroom_01: x=13.5, y=19.0, width=9.5, height=9.5, area=90.25. Width 9.5 > 9 minimum. Height 9.5 > 9 minimum. Pass.

BOUNDARY CHECKS:
living_01: 0+10=10 ≤ 23. 0+11=11 ≤ 28.5. Pass.
dining_01: 13.5+9.5=23 ≤ 23. 0+11=11 ≤ 28.5. Pass.
corridor_01: 10+3.5=13.5 ≤ 23. 11+17.5=28.5 ≤ 28.5. Pass.
kitchen_01: 13.5+9.5=23 ≤ 23. 11+8=19 ≤ 28.5. Pass.
master_bath_01: 0+5.5=5.5 ≤ 23. 11+8=19 ≤ 28.5. Pass.
bathroom_01: 5.5+4.5=10 ≤ 23. 11+8=19 ≤ 28.5. Pass.
master_bedroom_01: 0+10=10 ≤ 23. 19+9.5=28.5 ≤ 28.5. Pass.
bedroom_01: 13.5+9.5=23 ≤ 23. 19+9.5=28.5 ≤ 28.5. Pass.

OVERLAP CHECKS (key pairs):
living vs dining: living ends at x=10. Corridor starts at x=10. Dining starts at x=13.5. living x+width=10 ≤ dining x=13.5. Pass.
living vs corridor: living x+width=10 = corridor x=10. Pass (touching, not overlapping).
master_bath vs bathroom: bath_01 x=5.5, master_bath ends at 5.5. Touching not overlapping. Pass.
kitchen vs corridor: corridor ends at x=13.5, kitchen starts at x=13.5. Pass.
master_bedroom vs bedroom: master ends at x=10, bedroom starts at x=13.5. 10 ≤ 13.5. Pass.
master_bedroom vs master_bath: master_bedroom y=19.0, master_bath y+height=19.0. 19 ≤ 19. Pass.
master_bedroom vs corridor: master_bedroom x+width=10 = corridor x=10. Touching not overlapping. Pass.
bedroom vs corridor: corridor x+width=13.5 = bedroom x=13.5. Touching not overlapping. Pass.
All other pairs are separated by band boundaries and pass trivially.

VASTU:
Base: 55.
East facing: +8. Total 63.
Kitchen southeast (center x=18.25 > 11.5, center y=15.0 > 14.25): +8. Total 71.
Master bedroom southwest (center x=5.0 < 11.5, center y=23.75 > 14.25): +7. Total 78.
Living room in Band1 with east facing: +5. Total 83.
No bathrooms in northeast. No kitchen in northeast.
Final vastu_score: 83.

ADJACENCY:
Base: 60.
Kitchen adjacent to dining? Kitchen y=11.0, dining y+height=11.0. They share horizontal wall at y=11.0. +10. Total 70.
Master bath adjacent to master bedroom? Master bath y+height=19.0, master bedroom y=19.0. Shared wall. +10. Total 80.
Living adjacent to dining? Living x+width=10, corridor at 10, dining at 13.5. Not directly adjacent (corridor separates). No add.
Corridor adjacent to master bedroom? Corridor x+width=13.5. Master bedroom x+width=10. Master bedroom right wall at x=10 = corridor left wall at x=10. Yes, shared wall. +5. Total 85.
Corridor adjacent to bedroom? Corridor x+width=13.5 = bedroom x=13.5. Shared wall. +5. Total 90.
Final adjacency_score: 90.

FINAL JSON OUTPUT:

{"plot_boundary":[{"x":0.0,"y":0.0},{"x":23.0,"y":0.0},{"x":23.0,"y":28.5},{"x":0.0,"y":28.5}],"rooms":[{"id":"living_01","type":"living","label":"Living Room","x":0.0,"y":0.0,"width":10.0,"height":11.0,"area":110.0,"zone":"public","band":1,"color":"#E8F5E9","polygon":[{"x":0.0,"y":0.0},{"x":10.0,"y":0.0},{"x":10.0,"y":11.0},{"x":0.0,"y":11.0}]},{"id":"dining_01","type":"dining","label":"Dining Room","x":13.5,"y":0.0,"width":9.5,"height":11.0,"area":104.5,"zone":"public","band":1,"color":"#FFF3E0","polygon":[{"x":13.5,"y":0.0},{"x":23.0,"y":0.0},{"x":23.0,"y":11.0},{"x":13.5,"y":11.0}]},{"id":"corridor_01","type":"corridor","label":"Corridor","x":10.0,"y":11.0,"width":3.5,"height":17.5,"area":61.25,"zone":"service","band":2,"color":"#F5F5F5","polygon":[{"x":10.0,"y":11.0},{"x":13.5,"y":11.0},{"x":13.5,"y":28.5},{"x":10.0,"y":28.5}]},{"id":"kitchen_01","type":"kitchen","label":"Kitchen","x":13.5,"y":11.0,"width":9.5,"height":8.0,"area":76.0,"zone":"service","band":2,"color":"#FFEBEE","polygon":[{"x":13.5,"y":11.0},{"x":23.0,"y":11.0},{"x":23.0,"y":19.0},{"x":13.5,"y":19.0}]},{"id":"master_bath_01","type":"master_bath","label":"Master Bathroom","x":0.0,"y":11.0,"width":5.5,"height":8.0,"area":44.0,"zone":"service","band":2,"color":"#E0F7FA","polygon":[{"x":0.0,"y":11.0},{"x":5.5,"y":11.0},{"x":5.5,"y":19.0},{"x":0.0,"y":19.0}]},{"id":"bathroom_01","type":"bathroom","label":"Bathroom","x":5.5,"y":11.0,"width":4.5,"height":8.0,"area":36.0,"zone":"service","band":2,"color":"#E0F7FA","polygon":[{"x":5.5,"y":11.0},{"x":10.0,"y":11.0},{"x":10.0,"y":19.0},{"x":5.5,"y":19.0}]},{"id":"master_bedroom_01","type":"master_bedroom","label":"Master Bedroom","x":0.0,"y":19.0,"width":10.0,"height":9.5,"area":95.0,"zone":"private","band":3,"color":"#E3F2FD","polygon":[{"x":0.0,"y":19.0},{"x":10.0,"y":19.0},{"x":10.0,"y":28.5},{"x":0.0,"y":28.5}]},{"id":"bedroom_01","type":"bedroom","label":"Bedroom 2","x":13.5,"y":19.0,"width":9.5,"height":9.5,"area":90.25,"zone":"private","band":3,"color":"#E3F2FD","polygon":[{"x":13.5,"y":19.0},{"x":23.0,"y":19.0},{"x":23.0,"y":28.5},{"x":13.5,"y":28.5}]}],"doors":[],"windows":[],"metadata":{"bhk":2,"vastu_score":83,"adjacency_score":90,"architect_note":"East-facing 2BHK on 30x40 plot. Living room and master bedroom occupy west column (x=0 to 10.0) for privacy and Vastu southwest placement. Dining room and kitchen occupy east column (x=13.5 to 23.0) near the road for service access. Central corridor at x=10.0 connects master bathroom, common bathroom, and both private bedrooms without cross-traffic through any room. Master bedroom at 10.0x9.5 feet reflects forced depth compromise of compact 28.5 foot usable length.","vastu_issues":[]}}

---

## WHAT TO DO WITH EXTRA ROOMS

Pooja room:
Place at northeast of plot. For most facings, northeast = right column (x near Usable Width), front of plot (y near 0). Carve from top-right of dining room in Band1. Minimum 4.0 wide × 5.0 deep. Do not let pooja shrink dining below 8×8.

Study:
Place in Band3, right column, below bedroom_02. Split Band3 right column: bedroom gets top 60%, study gets bottom 40% of Band3 height. Study minimum 6.0 wide × 7.0 deep. If Band3 is only 9.5 feet, splitting into bedroom (5.7 deep) and study (3.8 deep) produces rooms below minimum. In this case, do not split — place study in Band2 right column instead, shifting kitchen to left column Band2, bathrooms to right column Band2.

Balcony:
Road-facing balcony: carve from front of living room (Band1 left column). x=0, y=0, width=left_column_width, height=4.5. Living room then starts at y=4.5.
Rear balcony: place at rear of Band3, full width or right column. x=0 (or corridor_x+corridor_width), y=Usable Length-5.0, width=full width or column width, height=5.0.

Store room:
Small room in Band2 service zone. Place in left column Band2 above or below master bathroom. Minimum 4.0×4.0.

Utility room:
Similar to store. Place in Band2 service zone left column or right column alongside kitchen.

Garage:
Requires 9.0 wide × 15.0 deep. Place in Band1 only if Usable Width ≥ 18.0 and the garage can fit alongside living or dining. Garage x=0, y=0, width=9.0, height=min(15.0, Band1_height+Band2_height). Remaining Band1 width goes to living room. Note: garage requires driveway access from road, so it must touch the road-facing edge (y=0).

Staircase (when floors > 1):
Place in Band2 service zone, inside or alongside corridor. Staircase minimum 4.0 × 8.0. If placed inside corridor, corridor must be at least 4.0+4.0=8.0 wide — only feasible on wide plots. For narrow plots, staircase occupies part of left column Band2, master bathroom shrinks accordingly.

Foyer:
Small entry zone at y=0, center of x. Width = corridor width + 1.0 on each side. Height = 4.5 feet. Living room starts at y=4.5 instead of y=0.

---

## ABSOLUTE RULES THAT NEVER CHANGE

ONE: Living room is ALWAYS near y=0 (road side). Bedrooms are ALWAYS near y=Usable Length (rear).

TWO: The corridor always runs vertically (in the y direction) through Band2 and Band3. It never runs horizontally. It never appears only in Band1.

THREE: Kitchen is ALWAYS in Band2 (service zone). Kitchen never goes in Band1 (public zone) and never goes in Band3 (private zone).

FOUR: Master bathroom is ALWAYS in Band2, left column, directly below master bedroom (which is in Band3, left column). They must share a wall at the boundary between Band2 and Band3.

FIVE: Every bedroom must share a wall with the corridor. No bedroom is accessible only through another room.

SIX: The polygon array for every room must exactly match that room's x, y, width, height. The four polygon points are always: [{x, y}, {x+width, y}, {x+width, y+height}, {x, y+height}]. No exceptions.

SEVEN: Area is always width multiplied by height to one decimal. Never compute it differently.

EIGHT: Output only JSON. Never output anything before the opening brace or after the closing brace of the JSON object.
"""


def _extract_json(text: str) -> dict:
    """Pull valid JSON out of model output, handling common issues."""
    # Strip thinking blocks (some models output <think>...</think>)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Strip markdown code fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    text = text.replace("```", "").strip()

    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Find outermost { }
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass

    raise ValueError(f"Model did not return valid JSON. Got: {text[:300]}")


def _build_generation_user_message(
    user_message: str,
    advisory: dict[str, Any] | None = None,
    request_data: dict[str, Any] | None = None,
) -> str:
    """Build a richer instruction payload for more realistic plans."""
    parts = [str(user_message).strip()]

    if isinstance(request_data, dict):
        bedrooms = int(request_data.get("bedrooms", 2) or 2)
        baths_target = int(request_data.get("bathrooms_target", 0) or 0)
        bathrooms = baths_target if baths_target > 0 else bedrooms
        family = str(request_data.get("family_type", "nuclear") or "nuclear")
        facing = str(request_data.get("facing", "east") or "east")
        extras = request_data.get("extras", [])
        extras = [str(x).strip() for x in extras if str(x).strip()]
        must_have = request_data.get("must_have", [])
        must_have = [str(x).strip() for x in must_have if str(x).strip()]
        avoid = request_data.get("avoid", [])
        avoid = [str(x).strip() for x in avoid if str(x).strip()]
        strict_real_life = bool(request_data.get("strict_real_life", False))

        priorities: list[str] = []
        if int(request_data.get("vastu_priority", 3) or 3) >= 4:
            priorities.append("strong_vastu")
        if int(request_data.get("natural_light_priority", 3) or 3) >= 4:
            priorities.append("daylight")
        if int(request_data.get("privacy_priority", 3) or 3) >= 4:
            priorities.append("privacy")
        if int(request_data.get("storage_priority", 3) or 3) >= 4:
            priorities.append("storage")

        lifestyle: list[str] = []
        if bool(request_data.get("work_from_home")):
            lifestyle.append("work_from_home_requires_study")
        if bool(request_data.get("elder_friendly")):
            lifestyle.append("elder_friendly_no_tight_corners")
        if int(request_data.get("parking_slots", 0) or 0) > 0:
            lifestyle.append("vehicle_parking_space_required")
        if int(request_data.get("floors", 1) or 1) > 1:
            lifestyle.append("staircase_required")

        parts.append("PRACTICAL REQUIREMENTS:")
        parts.append(f"- Target: {bedrooms}BHK, bathrooms={bathrooms}, family={family}, facing={facing}")
        parts.append("- Daily flow: living+dining near entry; kitchen adjacent to dining; private bedrooms separated from public zone")
        parts.append("- Ensure at least one bathroom is accessible from common circulation")
        if extras:
            parts.append(f"- Requested extras: {', '.join(extras)}")
        if must_have:
            parts.append(f"- Must-have constraints: {', '.join(must_have)}")
        if avoid:
            parts.append(f"- Avoid constraints: {', '.join(avoid)}")
        if priorities:
            parts.append(f"- High priorities: {', '.join(priorities)}")
        if lifestyle:
            parts.append(f"- Lifestyle constraints: {', '.join(lifestyle)}")
        if strict_real_life:
            parts.append("- Strict mode: prioritize practical compliance over stylistic variation")

    if isinstance(advisory, dict) and advisory:
        strategy = str(advisory.get("design_strategy", "") or "").strip()
        priority_order = advisory.get("priority_order", [])
        critical_checks = advisory.get("critical_checks", [])
        risks = advisory.get("risks", [])
        lifestyle_moves = advisory.get("lifestyle_moves", [])

        parts.append("ARCHITECT ADVISORY:")
        if strategy:
            parts.append(f"- Strategy: {strategy}")
        if isinstance(priority_order, list) and priority_order:
            parts.append(f"- Priority order: {', '.join(str(x) for x in priority_order[:6])}")
        if isinstance(critical_checks, list) and critical_checks:
            parts.append(f"- Critical checks: {', '.join(str(x) for x in critical_checks[:6])}")
        if isinstance(lifestyle_moves, list) and lifestyle_moves:
            parts.append(f"- Lifestyle moves: {', '.join(str(x) for x in lifestyle_moves[:6])}")
        if isinstance(risks, list) and risks:
            parts.append(f"- Risks to avoid: {', '.join(str(x) for x in risks[:4])}")

    parts.append(
        "Output JSON only. In metadata.architect_note, explain the real-life rationale in one concise sentence."
    )
    return "\n".join(part for part in parts if part)


def _normalize_plan(plan: dict, user_message: str) -> dict:
    """Ensure plan has all required fields with correct types."""
    if not isinstance(plan, dict):
        plan = {}

    # Extract dimensions from user message for fallback boundary
    dims = re.search(
        r"(\d+(?:\.\d+)?)\s*[xX×]\s*(\d+(?:\.\d+)?)", str(user_message)
    )
    uw = float(dims.group(1)) - 7.0 if dims else 23.0
    ul = float(dims.group(2)) - 11.5 if dims else 28.5

    # Ensure plot_boundary
    if (
        not isinstance(plan.get("plot_boundary"), list)
        or len(plan.get("plot_boundary", [])) < 3
    ):
        plan["plot_boundary"] = [
            {"x": 0, "y": 0},
            {"x": uw, "y": 0},
            {"x": uw, "y": ul},
            {"x": 0, "y": ul},
        ]

    # Ensure rooms is a list
    rooms = plan.get("rooms", [])
    if isinstance(rooms, dict):
        # Some models return rooms as a dict — convert
        rooms = [
            {"id": k, "type": k, "label": k.replace("_", " ").title(), **v}
            for k, v in rooms.items()
            if isinstance(v, dict)
        ]
    if not isinstance(rooms, list):
        rooms = []

    # Normalize each room
    COLOR_MAP = {
        "living": "#E8F5E9", "dining": "#FFF3E0", "kitchen": "#FFEBEE",
        "master_bedroom": "#E3F2FD", "bedroom": "#E3F2FD",
        "master_bath": "#E0F7FA", "bathroom": "#E0F7FA", "toilet": "#E0F7FA",
        "corridor": "#F5F5F5", "pooja": "#FFF8E1", "study": "#EDE7F6",
        "store": "#EFEBE9", "balcony": "#E8F5E9", "garage": "#ECEFF1",
        "utility": "#F3E5F5", "foyer": "#FAFAFA", "staircase": "#ECEFF1",
    }
    ZONE_DEFAULTS = {
        "living": ("public", 1),
        "dining": ("public", 1),
        "pooja": ("public", 1),
        "foyer": ("public", 1),
        "kitchen": ("service", 2),
        "bathroom": ("service", 2),
        "master_bath": ("service", 2),
        "toilet": ("service", 2),
        "corridor": ("service", 2),
        "utility": ("service", 2),
        "store": ("service", 2),
        "staircase": ("service", 2),
        "master_bedroom": ("private", 3),
        "bedroom": ("private", 3),
        "study": ("private", 3),
        "balcony": ("private", 3),
        "garage": ("service", 1),
    }
    TYPE_ALIASES = {
        "living_room": "living",
        "livingroom": "living",
        "dining_room": "dining",
        "master_bathroom": "master_bath",
        "common_bathroom": "bathroom",
        "washroom": "bathroom",
        "passage": "corridor",
        "hallway": "corridor",
        "hall": "corridor",
        "puja": "pooja",
        "mandir": "pooja",
    }

    def _to_num(v: Any, default: float) -> float:
        try:
            return float(v)
        except Exception:
            return default

    def _canonical_room_type(raw_type: str, raw_label: str) -> str:
        rtype = TYPE_ALIASES.get(raw_type, raw_type)
        if rtype in COLOR_MAP:
            return rtype

        label = (raw_label or "").lower()
        if "master" in label and "bed" in label:
            return "master_bedroom"
        if "bed" in label:
            return "bedroom"
        if "living" in label:
            return "living"
        if "dining" in label:
            return "dining"
        if "kitchen" in label:
            return "kitchen"
        if "corridor" in label or "hall" in label:
            return "corridor"
        if "master" in label and "bath" in label:
            return "master_bath"
        if "bath" in label or "toilet" in label:
            return "bathroom"
        if "study" in label or "office" in label:
            return "study"
        if "pooja" in label or "puja" in label or "mandir" in label:
            return "pooja"
        return rtype or "room"

    for i, room in enumerate(rooms):
        if not isinstance(room, dict):
            continue
        room.setdefault("id", f"room_{i+1:02d}")
        room.setdefault("type", "room")
        raw_type = str(room.get("type", "room")).strip().lower() or "room"
        raw_label = str(room.get("label", "")).strip().lower()
        room["type"] = _canonical_room_type(raw_type, raw_label)
        room.setdefault("label", room["type"].replace("_", " ").title())
        room["x"] = _to_num(room.get("x", 0.0), 0.0)
        room["y"] = _to_num(room.get("y", 0.0), 0.0)
        room["width"] = max(0.5, _to_num(room.get("width", 10.0), 10.0))
        room["height"] = max(0.5, _to_num(room.get("height", 10.0), 10.0))

        # Calculate area if missing
        w = _to_num(room.get("width", 0), 0.0)
        h = _to_num(room.get("height", 0), 0.0)
        room["area"] = round(_to_num(room.get("area", w * h), w * h), 1)

        # Fill color from type
        rtype = str(room.get("type", "room"))
        room.setdefault("color", COLOR_MAP.get(rtype, "#F5F5F5"))

        # Zone/band defaults
        default_zone, default_band = ZONE_DEFAULTS.get(rtype, ("service", 2))
        room.setdefault("zone", default_zone)
        room.setdefault("band", default_band)

        # Build polygon if missing
        if not isinstance(room.get("polygon"), list) or len(room.get("polygon", [])) < 3:
            x, y = float(room["x"]), float(room["y"])
            room["polygon"] = [
                {"x": x, "y": y},
                {"x": x + w, "y": y},
                {"x": x + w, "y": y + h},
                {"x": x, "y": y + h},
            ]

    plan["rooms"] = [r for r in rooms if isinstance(r, dict)]

    plan.setdefault("doors", [])
    plan.setdefault("windows", [])

    meta = plan.get("metadata", {})
    if not isinstance(meta, dict):
        meta = {}
    meta.setdefault("bhk", 2)
    meta.setdefault("vastu_score", 68)
    meta.setdefault("adjacency_score", 72)
    meta.setdefault(
        "architect_note",
        "Floor plan generated by NAKSHA-MASTER local model.",
    )
    meta.setdefault("vastu_issues", [])
    plan["metadata"] = meta

    # Flatten metadata fields to top level for frontend compatibility
    plan.setdefault("vastu_score", meta.get("vastu_score", 68))
    plan.setdefault("adjacency_score", meta.get("adjacency_score", 72))
    plan.setdefault("architect_note", meta.get("architect_note", ""))
    plan.setdefault("generation_method", "llm")

    return plan


async def _call_local(
    system_prompt: str,
    user_message: str,
    model: str,
    timeout: float,
    label: str = "local",
) -> str:
    """Make one HTTP call to local Ollama API. Returns raw text content."""
    url = f"{LOCAL_LLM_BASE_URL}/chat/completions"
    payload = {
        "model": model,
        "temperature": 0.05,  # Very low temp for consistent JSON
        "max_tokens": 3072,  # Enough for floor plans, faster than 4096
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": False,
    }

    log.info("%s: calling %s (timeout=%.0fs)", label, model, timeout)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        log.info("%s: got %d chars", label, len(content))
        return content

    except httpx.ConnectError:
        raise RuntimeError(
            "Cannot connect to Ollama. Is it running?\n"
            "Check: Open browser -> http://localhost:11434\n"
            "Fix: Click the Ollama icon in system tray or restart Ollama."
        )
    except httpx.TimeoutException:
        raise RuntimeError(
            f"Model '{model}' took too long (>{timeout}s).\n"
            "Try: Use the 7B model instead of 14B, or increase timeout."
        )
    except KeyError:
        raise RuntimeError(
            f"Unexpected Ollama response format: {resp.text[:300] if resp else 'no response'}"
        )


async def call_openrouter(
    system_prompt: str,
    user_message: str,
    *,
    temperature: float = 0.1,
    max_tokens: int = 800,
) -> dict:
    """
    Advisory call (used for architect reasoning stage).
    Returns a dict — doesn't need to be a floor plan JSON.
    """
    advisory_system = (
        "You are an expert Indian residential architect. "
        "Analyze the request and return a JSON object with keys: "
        "design_strategy (string), priority_order (array of strings), "
        "critical_checks (array of strings), risks (array of strings), "
        "lifestyle_moves (array of strings), circulation_plan (string). "
        "Return only JSON."
    )
    try:
        content = await _call_local(
            advisory_system,
            user_message,
            LOCAL_LLM_ADVISORY_MODEL,
            ADVISORY_TIMEOUT,
            label="advisory",
        )
        return _extract_json(content)
    except Exception as e:
        log.warning("Advisory call failed: %s. Using defaults.", str(e)[:100])
        return {
            "design_strategy": "Standard 3-band zoning with Vastu compliance.",
            "priority_order": ["program_fit", "vastu", "circulation"],
            "critical_checks": [
                "no_overlap",
                "minimum_sizes",
                "corridor_connectivity",
            ],
            "lifestyle_moves": ["separate_public_private_zones"],
            "circulation_plan": "Use central corridor spine for bedroom access.",
            "risks": [],
        }


async def call_openrouter_plan(
    system_prompt: str,
    user_message: str,
    *,
    temperature: float = 0.05,
    max_tokens: int = 4000,
    advisory: dict[str, Any] | None = None,
    request_data: dict[str, Any] | None = None,
) -> dict:
    """
    Main floor plan generation. Tries 7B first, falls back to 14B.
    The system_prompt from prompt_builder.py is overridden with
    NAKSHA_SYSTEM_PROMPT because the local model needs a more compact,
    precise version.
    """
    # Use our optimized local system prompt
    effective_system = NAKSHA_SYSTEM_PROMPT
    combined_user = _build_generation_user_message(
        user_message=user_message,
        advisory=advisory,
        request_data=request_data,
    )

    # Try primary model (7B — fast on GTX 1650)
    try:
        content = await _call_local(
            effective_system,
            combined_user,
            LOCAL_LLM_PLAN_MODEL,
            PRIMARY_TIMEOUT,
            label="plan-primary",
        )
        plan = _extract_json(content)
        return _normalize_plan(plan, user_message)

    except Exception as e:
        log.warning("Primary model failed: %s. Trying backup.", str(e)[:120])

    # Try backup model (14B — slower but smarter)
    try:
        content = await _call_local(
            effective_system,
            combined_user,
            LOCAL_LLM_BACKUP_MODEL,
            BACKUP_TIMEOUT,
            label="plan-backup",
        )
        plan = _extract_json(content)
        return _normalize_plan(plan, user_message)

    except Exception as e:
        log.error("Backup model also failed: %s", str(e)[:120])
        raise RuntimeError(
            f"Both local models failed to generate a floor plan.\n"
            f"Primary: {LOCAL_LLM_PLAN_MODEL}\n"
            f"Backup: {LOCAL_LLM_BACKUP_MODEL}\n"
            f"Error: {e}\n\n"
            f"Troubleshooting:\n"
            f"1. Open browser -> http://localhost:11434 (should show 'Ollama is running')\n"
            f"2. Open Command Prompt -> ollama list (should show your models)\n"
            f"3. Try: ollama run {LOCAL_LLM_PLAN_MODEL} 'say hello'\n"
        )


async def call_openrouter_plan_backup(
    system_prompt: str,
    user_message: str,
    *,
    temperature: float = 0.05,
    max_tokens: int = 4000,
) -> dict:
    """Backup planner — same as primary for local setup."""
    return await call_openrouter_plan(system_prompt, user_message)
