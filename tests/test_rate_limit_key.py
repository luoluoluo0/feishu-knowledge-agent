"""限流身份键的针对性测试。

背景：问答端点切到 JWT 双轨后，若限流仍只按 X-API-Key 分桶，所有登录
用户（请求不带 API Key）会共享同一个桶，A 的高频使用会吃光 B 的额度。
"""

from __future__ import annotations

from fastapi import Request

from app.rate_limit import _client_key


class FakeRequest:
    """只带 _client_key 需要的字段，不必起 TestClient。"""

    def __init__(self, headers: dict[str, str], host: str = "127.0.0.1"):
        self.headers = headers
        self.client = type("Client", (), {"host": host})()


def _request(headers: dict[str, str]) -> Request:
    return FakeRequest(headers)  # type: ignore[arg-type]


def test_bearer_token_gets_own_bucket_per_user():
    alice = _request({"Authorization": "Bearer token-alice"})
    bob = _request({"Authorization": "Bearer token-bob"})

    assert _client_key(alice) != _client_key(bob)
    # 同一枚令牌反复请求命中同一个桶
    assert _client_key(alice) == _client_key(alice)


def test_api_key_requests_share_bucket():
    """旧脚本共用 API Key：与单 key 时代一致，共享一个桶。"""

    first = _request({"X-API-Key": "shared-key"})
    second = _request({"X-API-Key": "shared-key"})

    assert _client_key(first) == _client_key(second)


def test_bearer_takes_precedence_over_api_key():
    """双头请求按认证双轨的优先级归桶：Bearer 有效身份优先。"""

    both = _request({"Authorization": "Bearer token-alice", "X-API-Key": "shared-key"})
    bearer_only = _request({"Authorization": "Bearer token-alice"})

    assert _client_key(both) == _client_key(bearer_only)


def test_anonymous_requests_share_one_bucket():
    anonymous_a = _request({})
    anonymous_b = _request({})

    assert _client_key(anonymous_a) == _client_key(anonymous_b)
