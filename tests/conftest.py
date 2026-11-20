"""Keep every synthetic data file used by tests inside the repository."""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_path():
    root = Path(__file__).resolve().parent / ".tmp"
    root.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="test-", dir=root) as directory:
        yield Path(directory)


@pytest.fixture(autouse=True)
def paths(monkeypatch, tmp_path):
    for name in ("WEAVEHR_PROJECT", "AMSTERDAM_OMOP", "MIMIC_RAW", "CLIN_JEPA_DATA", "CLIN_JEPA_OUTPUT",
                 "CLMBR_MODEL", "MIMIC_CODE", "HF_HOME"):
        monkeypatch.setenv(name, str(tmp_path / name.lower()))
