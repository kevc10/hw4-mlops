from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# dummy model (simple for testing)
from sklearn.ensemble import RandomForestClassifier
import numpy as np

X = np.random.rand(100, 3)
y = np.random.randint(0, 2, 100)
model = RandomForestClassifier().fit(X, y)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model": "loaded"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if "feature1" not in data:
        return jsonify({"error": "Invalid input", "details": "missing feature1"}), 400

    if not isinstance(data["feature1"], (int, float)):
        return jsonify({"error": "Invalid input", "details": "feature1 must be number"}), 400

    X = pd.DataFrame([[data["feature1"], 0, 0]])
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1])

    return jsonify({"prediction": pred, "probability": prob, "label": "positive" if pred == 1 else "negative"})

@app.route("/predict/batch", methods=["POST"])
def batch():
    data = request.get_json()

    if not isinstance(data, list):
        return jsonify({"error": "Invalid input"}), 400

    results = []
    for item in data:
        X = pd.DataFrame([[item.get("feature1", 0), 0, 0]])
        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0][1])
        results.append({"prediction": pred, "probability": prob})

    return jsonify(results)

if __name__ == "__main__":
    app.run(port=5000)

