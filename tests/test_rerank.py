import json
import urllib.error

from app import rerank as rr
from app.rerank import rerank_documents


# 重排客户端的降级路径。
#
# 重排是锦上添花：它挂了应当退回原始顺序，而不是让整条检索链路失败。
# 所以这里的重点是「出错时返回 None」，而不是「出错时抛什么」。


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def patch_urlopen(monkeypatch, handler):
    monkeypatch.setattr(rr.urllib.request, "urlopen", handler)


def test_returns_pairs_sorted_by_score(monkeypatch, make_settings):
    payload = {
        "results": [
            {"index": 2, "relevance_score": 0.11},
            {"index": 0, "relevance_score": 0.98},
            {"index": 1, "relevance_score": 0.53},
        ]
    }
    patch_urlopen(monkeypatch, lambda *a, **k: FakeResponse(payload))

    ranked = rerank_documents("问题", ["甲", "乙", "丙"], settings=make_settings())

    assert ranked == [(0, 0.98), (1, 0.53), (2, 0.11)]


def test_sends_every_document_in_one_call(monkeypatch, make_settings):
    """一次送一批，不是逐条送——重排器的价值就在于从候选堆里挑。"""

    captured = {}

    def handler(request, timeout=None):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"results": [{"index": 0, "relevance_score": 0.5}]})

    patch_urlopen(monkeypatch, handler)
    rerank_documents("问题", ["甲", "乙", "丙"], settings=make_settings())

    assert len(captured["body"]["documents"]) == 3
    assert captured["body"]["query"] == "问题"


def test_skips_when_no_api_key(monkeypatch, make_settings):
    patch_urlopen(monkeypatch, lambda *a, **k: (_ for _ in ()).throw(AssertionError("不该发请求")))

    settings = make_settings(silicon_api_key="")
    assert rerank_documents("问题", ["甲"], settings=settings) is None


def test_returns_none_for_empty_inputs(make_settings):
    assert rerank_documents("", ["甲"], settings=make_settings()) is None
    assert rerank_documents("问题", [], settings=make_settings()) is None


def test_returns_none_on_http_error(monkeypatch, make_settings):
    def handler(request, timeout=None):
        raise urllib.error.HTTPError("url", 429, "too many", {}, None)

    patch_urlopen(monkeypatch, handler)
    assert rerank_documents("问题", ["甲"], settings=make_settings()) is None


def test_returns_none_on_connection_error(monkeypatch, make_settings):
    def handler(request, timeout=None):
        raise OSError("connection refused")

    patch_urlopen(monkeypatch, handler)
    assert rerank_documents("问题", ["甲"], settings=make_settings()) is None


def test_returns_none_on_empty_results(monkeypatch, make_settings):
    patch_urlopen(monkeypatch, lambda *a, **k: FakeResponse({"results": []}))
    assert rerank_documents("问题", ["甲"], settings=make_settings()) is None


def test_returns_none_on_malformed_results(monkeypatch, make_settings):
    """分数不是数字时不能崩，退回去用原始顺序。"""

    patch_urlopen(
        monkeypatch,
        lambda *a, **k: FakeResponse({"results": [{"index": 0, "relevance_score": "很高"}]}),
    )
    assert rerank_documents("问题", ["甲"], settings=make_settings()) is None


def test_drops_out_of_range_index(monkeypatch, make_settings):
    """接口返回越界下标时，下游会取到错的文档——直接剔掉。"""

    payload = {
        "results": [
            {"index": 0, "relevance_score": 0.9},
            {"index": 99, "relevance_score": 0.99},
        ]
    }
    patch_urlopen(monkeypatch, lambda *a, **k: FakeResponse(payload))

    ranked = rerank_documents("问题", ["甲"], settings=make_settings())

    assert ranked == [(0, 0.9)]


def test_truncates_long_documents(monkeypatch, make_settings):
    captured = {}

    def handler(request, timeout=None):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse({"results": [{"index": 0, "relevance_score": 0.5}]})

    patch_urlopen(monkeypatch, handler)
    rerank_documents("问题", ["甲" * 9999], settings=make_settings())

    assert len(captured["body"]["documents"][0]) == rr.DOC_MAX_CHARS
