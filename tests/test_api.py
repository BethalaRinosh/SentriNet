"""API tests use synthetic records and stub inference; no real models required."""
import csv
import io
import json

from fastapi.testclient import TestClient

from app.backend import main
from src.preprocessing.pipeline import CATEGORICAL_FEATURES, MODEL_FEATURE_COLUMNS

client = TestClient(main.app)


def feature_record():
    return {
        name: (
            "tcp" if name == "protocol_type"
            else "http" if name == "service"
            else "SF" if name == "flag"
            else 1
        )
        for name in MODEL_FEATURE_COLUMNS
    }


def test_health_reports_service_and_artifact_presence(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "DEFAULT_ARTIFACT_DIR", tmp_path)
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["models_ready"] is False
    assert set(body["models_present"]) == {"binary_label", "class_label"}


def test_binary_endpoint_returns_prediction(monkeypatch):
    monkeypatch.setattr(
        main,
        "predict_records",
        lambda records, task: [
            {"prediction": "Attack", "confidence": 0.8, "attack_probability": 0.8}
            for _ in records
        ],
    )
    response = client.post("/predict/binary", json={"records": [feature_record()]})
    assert response.status_code == 200
    body = response.json()
    assert body["task"] == "binary_label"
    assert body["count"] == 1
    assert body["predictions"][0]["prediction"] == "Attack"
    assert body["predictions"][0]["attack_probability"] == 0.8


def test_multiclass_endpoint_returns_five_class_prediction(monkeypatch):
    monkeypatch.setattr(
        main,
        "predict_records",
        lambda records, task: [
            {"prediction": "Probe", "confidence": 0.6} for _ in records
        ],
    )
    response = client.post(
        "/predict/multiclass", json={"records": [feature_record()]}
    )
    assert response.status_code == 200
    assert response.json()["predictions"][0]["prediction"] == "Probe"
    assert response.json()["predictions"][0]["attack_probability"] is None


def test_missing_feature_is_rejected():
    record = feature_record()
    del record["src_bytes"]
    response = client.post("/predict/binary", json={"records": [record]})
    assert response.status_code == 422
    assert "src_bytes" in response.json()["detail"]["missing_features"]


def test_non_numeric_feature_is_rejected():
    record = feature_record()
    record["src_bytes"] = "a lot"
    response = client.post("/predict/binary", json={"records": [record]})
    assert response.status_code == 422
    assert "must be numeric" in response.json()["detail"]


def test_non_string_categorical_feature_is_rejected():
    record = feature_record()
    record["protocol_type"] = 7
    response = client.post("/predict/binary", json={"records": [record]})
    assert response.status_code == 422
    assert "must be a string" in response.json()["detail"]


def test_empty_records_are_rejected_by_request_schema():
    response = client.post("/predict/binary", json={"records": []})
    assert response.status_code == 422


def test_missing_model_artifact_returns_actionable_503(monkeypatch):
    def no_model(records, task):
        raise FileNotFoundError("missing artifact")

    monkeypatch.setattr(main, "predict_records", no_model)
    response = client.post("/predict/binary", json={"records": [feature_record()]})
    assert response.status_code == 503
    assert "python -m src.training.save_models" in response.json()["detail"]


def test_local_frontend_cors_preflight_is_allowed():
    response = client.options(
        "/health",
        headers={
            "Origin": "http://127.0.0.1:5500",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5500"



def test_non_finite_numeric_feature_is_rejected():
    record = feature_record()
    record["src_bytes"] = float("nan")
    # Send raw JSON: httpx's json= encoder rejects NaN before making a request.
    response = client.post(
        "/predict/binary",
        content=json.dumps({"records": [record]}, allow_nan=True),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert "must be finite" in response.json()["detail"]


def test_infinite_numeric_feature_is_rejected():
    record = feature_record()
    record["src_bytes"] = float("inf")
    # Raw JSON lets the server parser receive Infinity so API validation is exercised.
    response = client.post(
        "/predict/binary",
        content=json.dumps({"records": [record]}, allow_nan=True),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert "must be finite" in response.json()["detail"]


def test_negative_numeric_feature_is_rejected():
    record = feature_record()
    record["src_bytes"] = -1
    response = client.post("/predict/binary", json={"records": [record]})
    assert response.status_code == 422
    assert "must be non-negative" in response.json()["detail"]


def test_blank_categorical_feature_is_rejected():
    record = feature_record()
    record["protocol_type"] = "   "
    response = client.post("/predict/binary", json={"records": [record]})
    assert response.status_code == 422
    assert "must not be empty" in response.json()["detail"]



def test_extremely_large_integer_is_rejected_as_non_finite():
    record = feature_record()
    record["src_bytes"] = 10 ** 400
    response = client.post(
        "/predict/binary",
        content=json.dumps({"records": [record]}),
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code == 422
    assert "must be finite" in response.json()["detail"]



def csv_payload(records):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=MODEL_FEATURE_COLUMNS)
    writer.writeheader()
    writer.writerows(records)
    return buffer.getvalue().encode("utf-8")


def test_csv_batch_endpoint_returns_one_prediction_per_record(monkeypatch):
    records = [feature_record(), feature_record()]
    monkeypatch.setattr(
        main,
        "predict_records",
        lambda submitted, task: [
            {"prediction": "Normal", "confidence": 0.9},
            {"prediction": "Attack", "confidence": 0.8, "attack_probability": 0.8},
        ],
    )
    response = client.post(
        "/predict/csv",
        data={"task": "binary_label"},
        files={"file": ("connections.csv", csv_payload(records), "text/csv")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert [item["prediction"] for item in body["predictions"]] == ["Normal", "Attack"]


def test_csv_batch_rejects_missing_feature_columns():
    response = client.post(
        "/predict/csv",
        data={"task": "binary_label"},
        files={"file": ("bad.csv", b"duration,protocol_type\\n0,tcp\\n", "text/csv")},
    )
    assert response.status_code == 422
    assert "missing required feature columns" in response.json()["detail"]["message"]


def test_csv_batch_rejects_invalid_numeric_value():
    record = feature_record()
    record["src_bytes"] = "not-a-number"
    response = client.post(
        "/predict/csv",
        data={"task": "binary_label"},
        files={"file": ("bad.csv", csv_payload([record]), "text/csv")},
    )
    assert response.status_code == 422
    assert "must be numeric" in response.json()["detail"]


def test_csv_batch_rejects_empty_file():
    response = client.post(
        "/predict/csv",
        data={"task": "binary_label"},
        files={"file": ("empty.csv", b"", "text/csv")},
    )
    assert response.status_code == 422
    assert "empty" in response.json()["detail"]


def test_csv_batch_rejects_unsupported_task():
    response = client.post(
        "/predict/csv",
        data={"task": "unknown"},
        files={"file": ("connections.csv", csv_payload([feature_record()]), "text/csv")},
    )
    assert response.status_code == 422
    assert "task must be" in response.json()["detail"]



def test_csv_validation_endpoint_returns_preview_without_inference(monkeypatch):
    monkeypatch.setattr(
        main,
        "predict_records",
        lambda submitted, task: (_ for _ in ()).throw(AssertionError("inference must not run")),
    )
    response = client.post(
        "/validate/csv",
        files={"file": ("connections.csv", csv_payload([feature_record()]), "text/csv")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["valid"] is True
    assert body["count"] == 1
    assert body["columns"] == 40
    assert len(body["preview"]) == 1



def test_csv_batch_rejects_duplicate_headers():
    headers = list(MODEL_FEATURE_COLUMNS)
    headers[-1] = headers[0]
    response = client.post(
        "/validate/csv",
        files={"file": ("duplicate.csv", (",".join(headers) + "\n").encode(), "text/csv")},
    )
    assert response.status_code == 422
    assert "duplicate column" in response.json()["detail"]


def test_csv_batch_enforces_record_limit(monkeypatch):
    monkeypatch.setattr(main, "MAX_CSV_RECORDS", 1)
    response = client.post(
        "/validate/csv",
        files={"file": ("large.csv", csv_payload([feature_record(), feature_record()]), "text/csv")},
    )
    assert response.status_code == 413
    assert "at most 1 records" in response.json()["detail"]


def test_csv_batch_rejects_non_csv_filename():
    response = client.post(
        "/validate/csv",
        files={"file": ("connections.txt", csv_payload([feature_record()]), "text/plain")},
    )
    assert response.status_code == 422
    assert ".csv" in response.json()["detail"]
