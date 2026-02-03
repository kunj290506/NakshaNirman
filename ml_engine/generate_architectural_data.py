
import json
import random
import os

# Architectural Standards (Real World Ratios)
# Source: Time-Saver Standards for Building Types / NBC
# Living: 12-16%
# Kitchen: 8-12%
# Master Bed: 12-15%
# Sec Bed: 10-12%
# Bath: 4-6%
# Circulation/Walls: ~25%

def generate_architectural_plan(seed):
    random.seed(seed)
    
    # 1. Realistic Plot
    total_area_sqft = random.choice([
        random.uniform(800, 1000),  # Compact
        random.uniform(1000, 1500), # Standard
        random.uniform(1500, 2500)  # Luxury
    ])
    
    aspect_ratio = random.uniform(0.8, 1.4)
    # width * height = area => w * (w/ar) = area => w^2 = area*ar
    width_ft = (total_area_sqft * aspect_ratio) ** 0.5
    height_ft = total_area_sqft / width_ft
    
    # 2. Derive Room Sizes based on Class
    features = [
        random.choice([2, 3, 4]), # Bed
        random.choice([1, 2, 3]), # Bath
        total_area_sqft,
        aspect_ratio
    ]
    bedrooms = features[0]
    bathrooms = features[1]
    
    rooms = []
    
    # Living Room (The Hub) - 15% +/- 2%
    lr_area = total_area_sqft * random.uniform(0.13, 0.17)
    # Aspect ratio 1:1 to 1.5:1
    lr_ar = random.uniform(1.0, 1.4)
    lr_w = (lr_area * lr_ar) ** 0.5
    lr_h = lr_area / lr_w
    rooms.append({"type": "living_room", "w": lr_w, "h": lr_h, "area": lr_area})
    
    # Kitchen - 10% +/- 2%
    k_area = total_area_sqft * random.uniform(0.08, 0.12)
    k_ar = random.uniform(1.0, 1.3)
    k_w = (k_area * k_ar) ** 0.5
    k_h = k_area / k_w
    rooms.append({"type": "kitchen", "w": k_w, "h": k_h, "area": k_area})
    
    # Master Bed - 14%
    mb_area = total_area_sqft * random.uniform(0.12, 0.15)
    mb_w = (mb_area * 1.2) ** 0.5
    mb_h = mb_area / mb_w
    rooms.append({"type": "master_bedroom", "w": mb_w, "h": mb_h, "area": mb_area})
    
    # Other Rooms not needed for training inputs yet, but we generate full list
    return {
        "id": seed,
        "features": features,
        "rooms": rooms
    }

dataset = []
print("Generating 10000 architectural samples...")
for i in range(10000):
    dataset.append(generate_architectural_plan(i))

os.makedirs('dataset', exist_ok=True)
with open('dataset/architectural_data.json', 'w') as f:
    json.dump(dataset, f, indent=2)

print(f"Saved dataset/architectural_data.json with {len(dataset)} samples.")
