"""Day-1 config stub: connection strings only, no ORM/migrations (Spec §12 Day 1)."""

import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://sentinel:sentinel@localhost:5432/sentinel",
)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
