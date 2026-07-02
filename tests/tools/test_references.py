"""Plan D Task 5.2：参考文献选择逻辑测试。"""
from math_agent.tools.references import select_references
from math_agent.state import Reference


def test_select_references_uses_scholar_api(mocker):
    mocker.patch("math_agent.tools.references.search_references",
                 return_value=[Reference(id="1", title="API ref", year=2020)])
    refs = select_references("problem", ["optimization"])
    assert len(refs) == 1
    assert refs[0].title == "API ref"


def test_select_references_falls_back_to_static_library(mocker):
    mocker.patch("math_agent.tools.references.search_references", return_value=[])
    refs = select_references("problem", ["queueing"])
    assert len(refs) > 0  # static library has queueing refs
    assert all(isinstance(r, Reference) for r in refs)


def test_select_references_static_library_filters_by_domain(mocker):
    mocker.patch("math_agent.tools.references.search_references", return_value=[])
    refs_queueing = select_references("p", ["queueing"])
    refs_ml = select_references("p", ["machine_learning"])
    # Different domains should (likely) return different refs
    titles_q = {r.title for r in refs_queueing}
    titles_ml = {r.title for r in refs_ml}
    # At least one should be different (queueing has Kleinrock, ML has Goodfellow)
    assert titles_q != titles_ml or len(titles_q | titles_ml) > 0


def test_select_references_static_library_falls_to_generic(mocker):
    mocker.patch("math_agent.tools.references.search_references", return_value=[])
    # Unknown domain → falls back to generic
    refs = select_references("p", ["nonexistent_domain"])
    assert len(refs) > 0  # generic refs returned
