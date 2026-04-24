import pytest
from app.api import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200

    data = response.get_json()
    assert data["status"] == "healthy"
    assert "available_models" in data


def get_valid_payload():
    return {
        "LIMIT_BAL": 20000,
        "SEX": 2,
        "EDUCATION": 2,
        "MARRIAGE": 1,
        "AGE": 24,
        "PAY_0": 2,
        "PAY_2": 2,
        "PAY_3": -1,
        "PAY_4": -1,
        "PAY_5": -2,
        "PAY_6": -2,
        "BILL_AMT1": 3913,
        "BILL_AMT2": 3102,
        "BILL_AMT3": 689,
        "BILL_AMT4": 0,
        "BILL_AMT5": 0,
        "BILL_AMT6": 0,
        "PAY_AMT1": 0,
        "PAY_AMT2": 689,
        "PAY_AMT3": 0,
        "PAY_AMT4": 0,
        "PAY_AMT5": 0,
        "PAY_AMT6": 0
    }


def test_predict_v1(client):
    response = client.post(
        "/predict?model_version=v1",
        json=get_valid_payload()
    )

    assert response.status_code == 200
    data = response.get_json()

    assert "prediction" in data
    assert "probability" in data
    assert data["model_version"] == "v1"


def test_predict_v2(client):
    response = client.post(
        "/predict?model_version=v2",
        json=get_valid_payload()
    )

    assert response.status_code == 200
    data = response.get_json()

    assert "prediction" in data
    assert "probability" in data
    assert data["model_version"] == "v2"


def test_predict_ab(client):
    response = client.post(
        "/predict",
        json=get_valid_payload()
    )

    assert response.status_code == 200
    data = response.get_json()

    assert data["model_version"] in ["v1", "v2"]


def test_predict_empty_body(client):
    response = client.post("/predict", json=None)

    assert response.status_code == 400


def test_predict_missing_field(client):
    payload = get_valid_payload()
    payload.pop("AGE")

    response = client.post("/predict", json=payload)

    assert response.status_code == 400