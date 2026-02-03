
import json
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pickle
import os

# 1. Load Data
print("Loading dataset...")
with open('dataset/architectural_data.json', 'r') as f:
    data = json.load(f)

X = [] # [bedrooms, bathrooms, total_area, aspect_ratio]
y = [] # [living_w, living_h, kitchen_w, kitchen_h]

for plan in data:
    feats = plan['features'] # [bed, bath, area, aspect]
    
    # Find LR and Kitchen dimensions
    lr = next((r for r in plan['rooms'] if r['type'] == 'living_room'), None)
    k = next((r for r in plan['rooms'] if r['type'] == 'kitchen'), None)
    
    if lr and k:
        X.append(feats)
        y.append([lr['w'], lr['h'], k['w'], k['h']])

X = np.array(X)
y = np.array(y)

# 2. Scale Features
print(f"Training on {len(X)} samples...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Train Deep Neural Network
print("Training Neural Network (MLP)...")
# Hidden layers: 128->64->32
model = MLPRegressor(hidden_layer_sizes=(128, 64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
print(f"Model MSE: {mse:.2f}")

# 6. Save Model AND Scaler
os.makedirs('models', exist_ok=True)
with open('models/layout_predictor.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    
print("Model saved to ml_engine/models/layout_predictor.pkl")
print("Scaler saved to ml_engine/models/scaler.pkl")
