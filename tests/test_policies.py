from manifold.router.policies import (
    Backend,
    LeastOutstanding,
    PrefixAware,
    RoundRobin,
    make_policy,
)


def backends(n: int) -> list[Backend]:
    return [Backend(url=f"http://b{i}") for i in range(n)]


def test_round_robin_cycles():
    p = RoundRobin(backends(3))
    picks = [p.select().url for _ in range(6)]
    assert picks == ["http://b0", "http://b1", "http://b2"] * 2


def test_least_outstanding_prefers_idle():
    bs = backends(3)
    bs[0].outstanding = 5
    bs[1].outstanding = 1
    bs[2].outstanding = 3
    assert LeastOutstanding(bs).select().url == "http://b1"


def test_prefix_aware_is_sticky():
    p = PrefixAware(backends(4))
    first = p.select("session-alpha").url
    for _ in range(20):
        assert p.select("session-alpha").url == first


def test_prefix_aware_spills_under_overload():
    bs = backends(2)
    p = PrefixAware(bs, overload_factor=2.0)
    preferred = p.select("sticky-key")
    preferred.outstanding = 50
    other = next(b for b in bs if b is not preferred)
    other.outstanding = 0
    assert p.select("sticky-key") is other


def test_prefix_aware_without_key_falls_back():
    bs = backends(2)
    bs[0].outstanding = 9
    assert PrefixAware(bs).select(None).url == "http://b1"


def test_make_policy_strips_trailing_slash():
    p = make_policy("least_outstanding", ["http://x:8001/", "http://y:8002"])
    assert [b.url for b in p.backends] == ["http://x:8001", "http://y:8002"]
