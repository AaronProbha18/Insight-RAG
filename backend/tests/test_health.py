from fastapi.testclient import TestClient

from plumbline.api.app import app


def test_health_ok():
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
