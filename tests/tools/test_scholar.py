"""Semantic Scholar API 封装测试（Plan D Task 5.1）。"""
from math_agent.tools.scholar import search_references
from math_agent.state import Reference


def test_search_references_returns_real_references(mocker):
    fake_resp = mocker.MagicMock()
    fake_resp.status_code = 200
    fake_resp.json.return_value = {
        "total": 1,
        "data": [{
            "paperId": "abc123",
            "title": "Bike sharing demand forecasting",
            "authors": [{"name": "Smith, J."}, {"name": "Lee, K."}],
            "year": 2018,
            "venue": "Transportation Research Part B",
            "externalIds": {"DOI": "10.1016/j.trb.2018.01.001"},
        }],
    }
    mocker.patch("math_agent.tools.scholar.requests.get", return_value=fake_resp)
    refs = search_references("bike sharing demand ARIMA", limit=5)
    assert len(refs) == 1
    assert isinstance(refs[0], Reference)
    assert refs[0].title.startswith("Bike sharing")
    assert refs[0].doi.startswith("10.")
    assert refs[0].year == 2018


def test_search_references_returns_empty_on_network_error(mocker):
    mocker.patch("math_agent.tools.scholar.requests.get", side_effect=ConnectionError("net"))
    refs = search_references("anything", limit=5)
    assert refs == []


def test_search_references_returns_empty_on_rate_limit(mocker):
    fake_resp = mocker.MagicMock()
    fake_resp.status_code = 429
    mocker.patch("math_agent.tools.scholar.requests.get", return_value=fake_resp)
    refs = search_references("anything", limit=5)
    assert refs == []
