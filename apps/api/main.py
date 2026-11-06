"""Minimal FastAPI app (Day 1): GET /health, GET /api/v1/services."""

from fastapi import FastAPI
from pydantic import BaseModel

# Spec §3.1 core services. Fault Controller and Traffic Generator are test-harness
# components and are explicitly excluded from the allowlist (Spec §3.1, §3.3).
SERVICES: tuple[tuple[str, str], ...] = (
    ("api-gateway", "API Gateway"),
    ("auth-service", "Auth Service"),
    ("order-service", "Order Service"),
    ("payment-service", "Payment Service"),
)


class ServiceEntry(BaseModel):
    id: str
    name: str


app = FastAPI(title="Sentinel API")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/services", response_model=list[ServiceEntry])
def services() -> list[ServiceEntry]:
    return [ServiceEntry(id=sid, name=name) for sid, name in SERVICES]
