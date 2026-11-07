"""Session fixture: record the training and test snapshots once (deterministic, ~6 s)."""

import pytest

from evaluation.f01_runner import record_all


@pytest.fixture(scope="session")
def work(tmp_path_factory):
    path = tmp_path_factory.mktemp("f01")
    record_all(path)
    return path
