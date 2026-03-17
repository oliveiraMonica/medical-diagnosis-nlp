from flask import Flask, request, jsonify
import os
import joblib

from src.preprocessing import clean_text

app = Flask(__name__)

# ==============================
# PATH CONFIG
# ==============================

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Load model and vectorizer
model = joblib.load(os.path.join(MODELS_DIR, "model.pkl"))
vectorizer = joblib.load(os.path.join(MODELS_DIR, "vectorizer.pkl"))


# ==============================
# ROUTES
# ==============================

@app.route("/")
def home():
    return "Medical NLP API is running 🚀"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Preprocess
    cleaned_text = clean_text(text)

    # Vectorize
    X = vectorizer.transform([cleaned_text])

    # Predict probabilities
    probs = model.predict_proba(X)[0]

    # Get top 3 predictions
    top_indices = probs.argsort()[-3:][::-1]

    top_predictions = []
    for idx in top_indices:
        label = model.classes_[idx].strip()
        score = float(probs[idx])

        top_predictions.append({
            "label": label,
            "score": round(score * 100, 2)
        })

    return jsonify({
        "prediction": top_predictions[0]["label"],
        "top_3": top_predictions
    })


# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    app.run(debug=True)