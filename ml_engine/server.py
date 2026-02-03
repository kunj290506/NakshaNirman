
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

MODEL_PATH = 'models/layout_predictor.pkl'
SCALER_PATH = 'models/scaler.pkl'
model = None
scaler = None

# Load model and scaler on startup
if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print("Model and Scaler loaded successfully.")
else:
    print("WARNING: Model/Scaler not found!")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.json
    try:
        bedrooms = float(data.get('bedrooms', 2))
        bathrooms = float(data.get('bathrooms', 1))
        area = float(data.get('totalAreaSqft', 1000))
        aspect = float(data.get('aspectRatio', 1.0))
        
        # Raw features
        X_raw = np.array([[bedrooms, bathrooms, area, aspect]])
        
        # Scale features
        X_scaled = scaler.transform(X_raw)
        
        # Predict
        preds = model.predict(X_scaled)[0]
        
        return jsonify({
            "success": True,
            "prediction": {
                "living_room": { "width": preds[0], "height": preds[1] },
                "kitchen": { "width": preds[2], "height": preds[3] }
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    print("Starting Deep Learning Inference Server on port 5000...")
    app.run(debug=True, port=5000)
