"""Integration: API responds with /health 200 and the service allowlist (Spec §3.1)."""

from fastapi.testclient import TestClient

from apps.api.main import app

client = TestClient(app)


def test_health_responds_200():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_services_allowlist():
    response = client.get("/api/v1/services")
    assert response.status_code == 200
    names = {entry["name"] for entry in response.json()}
    assert names == {"API Gateway", "Auth Service", "Order Service", "Payment Service"}
    # Test-harness components are never monitored services (Spec §3.1, §3.3).
    assert "Fault Controller" not in names
    assert "Traffic Generator" not in names
