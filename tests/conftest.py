import json
from pathlib import Path
import pytest


@pytest.fixture
def sample_problem():
    p = Path(__file__).parent / "fixtures" / "sample_problem.json"
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path
