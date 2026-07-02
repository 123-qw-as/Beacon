import json
from pathlib import Path
import pytest


@pytest.fixture(autouse=True)
def _disable_rag_by_default(monkeypatch):
    """默认关掉 RAG，避免不测 RAG 的节点测试因 .env 里 RAG_ENABLED=1
    而真打 ollama embedding（502/超时会让测试 flaky）。

    需要测 RAG 的测试在自己的函数里 monkeypatch RAG_ENABLED=True 即可覆盖。
    """
    monkeypatch.setenv("MATH_AGENT_RAG_ENABLED", "0")
    # 已 import 的模块级常量也要同步
    import math_agent.config as _cfg
    monkeypatch.setattr(_cfg, "RAG_ENABLED", False)
    # 各节点模块的 from-import 也要关
    for mod_name in ("math_agent.nodes.writer", "math_agent.nodes.analyst",
                     "math_agent.nodes.modeler"):
        try:
            mod = __import__(mod_name, fromlist=["RAG_ENABLED"])
            if hasattr(mod, "RAG_ENABLED"):
                monkeypatch.setattr(mod, "RAG_ENABLED", False)
        except ImportError:
            pass


@pytest.fixture
def sample_problem():
    p = Path(__file__).parent / "fixtures" / "sample_problem.json"
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path
