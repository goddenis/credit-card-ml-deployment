import os
import random

from flasgger import Swagger
from flask import Flask, request, jsonify

try:
    from .model_handler import ModelHandler
except ImportError:
    # Works when started as a script: `python app/api.py`
    from model_handler import ModelHandler

app = Flask(__name__)
swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "Credit Card Default Prediction API",
        "description": "API для прогнозирования дефолта по кредитным картам",
        "version": "1.0.0"
    },
    "basePath": "/",
    "schemes": ["http"]
}

swagger = Swagger(app, template=swagger_template)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

model_v1 = ModelHandler(
    model_path=os.path.join(MODELS_DIR, "model_v1.pkl"),
    feature_names_path=os.path.join(MODELS_DIR, "feature_names.pkl")
)

model_v2 = ModelHandler(
    model_path=os.path.join(MODELS_DIR, "model_v2.pkl"),
    feature_names_path=os.path.join(MODELS_DIR, "feature_names.pkl")
)


def choose_model(model_version: str | None):
    """
    Выбор модели для инференса.
    """

    if model_version == "v1":
        return model_v1, "v1"

    if model_version == "v2":
        return model_v2, "v2"

    if model_version is not None:
        raise ValueError("model_version должна быть 'v1' или 'v2'")

    assigned_version = random.choice(["v1", "v2"])

    if assigned_version == "v1":
        return model_v1, "v1"

    return model_v2, "v2"


@app.route("/health", methods=["GET"])
def health():
    """
    Проверка состояния сервиса
    ---
    tags:
      - Service
    responses:
      200:
        description: Сервис работает
        schema:
          type: object
          properties:
            status:
              type: string
              example: healthy
            service:
              type: string
              example: credit-default-ml-api
            available_models:
              type: array
              items:
                type: string
              example: ["v1", "v2"]
    """
    return jsonify({
        "status": "healthy",
        "service": "credit-default-ml-api",
        "available_models": ["v1", "v2"]
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Предсказание дефолта клиента
    ---
    tags:
      - Prediction
    parameters:
      - name: model_version
        in: query
        type: string
        required: false
        enum: [v1, v2]
        description: Версия модели. Если не указана, модель выбирается случайно 50/50.
      - in: body
        name: client_features
        required: true
        schema:
          type: object
          required:
            - LIMIT_BAL
            - SEX
            - EDUCATION
            - MARRIAGE
            - AGE
            - PAY_0
            - PAY_2
            - PAY_3
            - PAY_4
            - PAY_5
            - PAY_6
            - BILL_AMT1
            - BILL_AMT2
            - BILL_AMT3
            - BILL_AMT4
            - BILL_AMT5
            - BILL_AMT6
            - PAY_AMT1
            - PAY_AMT2
            - PAY_AMT3
            - PAY_AMT4
            - PAY_AMT5
            - PAY_AMT6
          properties:
            LIMIT_BAL:
              type: number
              example: 20000
            SEX:
              type: integer
              example: 2
            EDUCATION:
              type: integer
              example: 2
            MARRIAGE:
              type: integer
              example: 1
            AGE:
              type: integer
              example: 24
            PAY_0:
              type: integer
              example: 2
            PAY_2:
              type: integer
              example: 2
            PAY_3:
              type: integer
              example: -1
            PAY_4:
              type: integer
              example: -1
            PAY_5:
              type: integer
              example: -2
            PAY_6:
              type: integer
              example: -2
            BILL_AMT1:
              type: number
              example: 3913
            BILL_AMT2:
              type: number
              example: 3102
            BILL_AMT3:
              type: number
              example: 689
            BILL_AMT4:
              type: number
              example: 0
            BILL_AMT5:
              type: number
              example: 0
            BILL_AMT6:
              type: number
              example: 0
            PAY_AMT1:
              type: number
              example: 0
            PAY_AMT2:
              type: number
              example: 689
            PAY_AMT3:
              type: number
              example: 0
            PAY_AMT4:
              type: number
              example: 0
            PAY_AMT5:
              type: number
              example: 0
            PAY_AMT6:
              type: number
              example: 0
    responses:
      200:
        description: Успешное предсказание
      400:
        description: Ошибка входных данных
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Empty JSON body"}), 400

        requested_model_version = request.args.get("model_version")
        model_handler, used_model_version = choose_model(requested_model_version)

        result = model_handler.predict(data)

        return jsonify({
            "prediction": result["prediction"],
            "probability": result["probability"],
            "model_version": used_model_version
        }), 200

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
