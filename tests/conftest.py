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


@pytest.fixture(autouse=True)
def _disable_llm_fallback_by_default(monkeypatch):
    """单元测试不继承开发机 .env 的真实备用模型；专项测试会显式开启。"""
    monkeypatch.setenv("MATH_AGENT_LLM_FALLBACK_MODELS", "")
    import math_agent.llm as _llm
    monkeypatch.setattr(_llm, "LLM_FALLBACK_MODELS", ())
    for state in (
        _llm._MODEL_UNHEALTHY_UNTIL,
        _llm._MODEL_FAILURE_STREAK,
        _llm._MODEL_SUCCESS_STREAK,
    ):
        state.clear()
    yield
    for state in (
        _llm._MODEL_UNHEALTHY_UNTIL,
        _llm._MODEL_FAILURE_STREAK,
        _llm._MODEL_SUCCESS_STREAK,
    ):
        state.clear()


@pytest.fixture(autouse=True)
def _disable_scholar_network_by_default(monkeypatch):
    """默认禁用 Semantic Scholar 网络调用，避免 writer 等测试真打外网（慢/flaky）。

    writer_node 调用 select_references → search_references；这里默认把
    references 模块绑定的 search_references 替换为返回空列表，writer 自动回退到静态库。
    - 需要 API 行为的测试（test_references.py）用 mocker.patch 显式覆盖，
      mocker.patch 优先级高于 monkeypatch，不受影响。
    - test_scholar.py 直接测 scholar.search_references 本身（mock requests.get），
      不经过 references 模块，因此不受本 fixture 影响。
    """
    import math_agent.tools.references as _refs
    monkeypatch.setattr(_refs, "search_references", lambda *a, **k: [])


@pytest.fixture
def sample_problem():
    p = Path(__file__).parent / "fixtures" / "sample_problem.json"
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path
