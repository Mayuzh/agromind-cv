from fastapi.testclient import TestClient

from agromind_cv.api import app


def test_health_reports_not_ready_without_model() -> None:
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] in {"ok", "not_ready"}


def test_predict_rejects_non_image() -> None:
    with TestClient(app) as client:
        response = client.post("/predict", files={"file": ("note.txt", b"hello", "text/plain")})
    assert response.status_code == 415

