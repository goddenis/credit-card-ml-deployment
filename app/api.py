import os
from flask import Flask, request, jsonify
from app.model_handler import ModelHandler


app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

model_handler = ModelHandler(
    model_path=os.path.join(MODELS_DIR, "model_v1.pkl"),
    feature_names_path=os.path.join(MODELS_DIR, "feature_names.pkl")
)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "service": "credit-default-ml-api"
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Empty JSON body"}), 400

        result = model_handler.predict(data)

        return jsonify({
            "prediction": result["prediction"],
            "probability": result["probability"],
            "model_version": "v1"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)