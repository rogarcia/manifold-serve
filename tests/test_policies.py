import pytest

from manifold.router.policies import (
    NoReplicaAvailable,
    RequestCtx,
    decide,
    inflight_tokens,
    kv_util,
    make_policy,
    prefix,
    queue,
    session,
)
from manifold.router.prefix import PrefixIndex
from manifold.router.replicas import EngineGauges, Replica, ReplicaState


def replicas(n: int) -> list[Replica]:
    return [Replica(url=f"http://r{i}", index=i) for i in range(n)]


def gauges(waiting: int = 0, kv: float = 0.0) -> EngineGauges:
    return EngineGauges(waiting=waiting, running=0, kv_usage=kv, prefix_hits_total=0,
                        prefix_queries_total=0, ts=0.0)


def ctx(blocks: list[int] | None = None, session_id: str | None = None) -> RequestCtx:
    return RequestCtx(request_id="req", session_id=session_id, prompt_tokens=[1] * 100,
                      block_hashes=blocks or [], est_output_tokens=10)


def scores(scorer, rs: list[Replica], c: RequestCtx | None = None, index=None) -> list[float]:
    index = index or PrefixIndex(len(rs), 100)
    return [scorer(c or ctx(), r, rs, index) for r in rs]


def test_queue_scores_fewer_waiting_higher():
    rs = replicas(3)
    for r, w in zip(rs, [4, 1, 2]):
        r.scraped = gauges(waiting=w)
    assert scores(queue, rs) == pytest.approx([0.0, 1.0, 2 / 3])


def test_load_scorers_give_one_when_all_equal_or_unscraped():
    rs = replicas(2)
    assert scores(queue, rs) == [1.0, 1.0]
    assert scores(kv_util, rs) == [1.0, 1.0]
    assert scores(inflight_tokens, rs) == [1.0, 1.0]


def test_kv_util_scores_lower_usage_higher():
    rs = replicas(3)
    for r, kv in zip(rs, [0.9, 0.3, 0.6]):
        r.scraped = gauges(kv=kv)
    assert scores(kv_util, rs) == pytest.approx([0.1, 0.7, 0.4])


def test_inflight_tokens_scores_lighter_replica_higher():
    rs = replicas(2)
    rs[0].in_flight_tokens, rs[1].in_flight_tokens = 500, 100
    assert scores(inflight_tokens, rs) == pytest.approx([0.0, 0.8])


def test_prefix_is_share_of_leading_blocks_held():
    rs = replicas(2)
    index = PrefixIndex(2, 100)
    index.record(0, [11, 12, 13])
    assert scores(prefix, rs, ctx([11, 12, 13, 14]), index) == [0.75, 0.0]
    assert scores(prefix, rs, ctx([]), index) == [0.0, 0.0]


def test_session_scores_exactly_one_replica_stably():
    rs = replicas(4)
    first = scores(session, rs, ctx(session_id="conv-7"))
    assert sorted(first) == [0.0, 0.0, 0.0, 1.0]
    assert scores(session, replicas(4), ctx(session_id="conv-7")) == first
    assert scores(session, rs, ctx(session_id=None)) == [0.0] * 4


def pick(policy_name: str, rs: list[Replica], c: RequestCtx | None = None, index=None):
    index = index or PrefixIndex(len(rs), 100)
    return decide(make_policy(policy_name), c or ctx(), rs, index)


def test_round_robin_cycles_and_skips_unready():
    rs = replicas(3)
    p, index = make_policy("round_robin"), PrefixIndex(3, 100)
    assert [decide(p, ctx(), rs, index).chosen.index for _ in range(4)] == [0, 1, 2, 0]
    rs[1].state = ReplicaState.DRAINING
    assert [decide(p, ctx(), rs, index).chosen.index for _ in range(3)] == [2, 0, 2]


def test_round_robin_state_is_per_router():
    rs = replicas(2)
    assert pick("round_robin", rs).chosen.index == 0
    assert pick("round_robin", rs).chosen.index == 0


def test_least_load_picks_lowest_combined_load():
    rs = replicas(3)
    rs[0].scraped, rs[1].scraped, rs[2].scraped = gauges(5, 0.8), gauges(0, 0.1), gauges(2, 0.4)
    d = pick("least_load", rs)
    assert d.chosen.index == 1
    assert d.scores["http://r1"] == pytest.approx({"queue": 1.0, "kv_util": 0.9,
                                                   "inflight_tokens": 1.0, "total": 5.8})
    assert d.spilled is False


def test_argmax_tie_breaks_on_inflight_tokens_then_index():
    rs = replicas(3)
    assert pick("least_load", rs).chosen.index == 0
    # r0 scores queue 1, kv 1, in-flight 0; r2 scores queue 0, kv 1, in-flight 1. Both total 4.
    rs[0].in_flight_tokens = 10
    rs[0].scraped = gauges(waiting=0)
    rs[1].scraped = gauges(waiting=3, kv=0.5)
    rs[2].scraped = gauges(waiting=3)
    d = pick("least_load", rs)
    assert d.scores["http://r0"]["total"] == pytest.approx(d.scores["http://r2"]["total"])
    assert d.chosen.index == 2


def test_session_hash_sticks_then_spills_when_overloaded():
    rs = replicas(2)
    c = ctx(session_id="conv-1")
    home = pick("session_hash", rs, c).chosen.index
    assert all(pick("session_hash", rs, c).chosen.index == home for _ in range(5))

    rs[home].in_flight_requests = 5  # above 2.0 x (0 + 1)
    d = pick("session_hash", rs, c)
    assert d.chosen.index == 1 - home
    assert d.spilled is True


def test_prefix_aware_prefers_warm_replica_at_equal_load():
    rs = replicas(2)
    index = PrefixIndex(2, 100)
    index.record(1, [1, 2, 3, 4])
    d = pick("prefix_aware", rs, ctx([1, 2, 3, 4, 5]), index)
    assert d.chosen.index == 1
    assert d.matched_blocks == {"http://r0": 0, "http://r1": 4}
    assert d.spilled is False


def test_prefix_aware_spills_when_warm_replica_is_loaded():
    rs = replicas(2)
    index = PrefixIndex(2, 100)
    index.record(1, [1, 2, 3, 4])
    rs[1].scraped, rs[0].scraped = gauges(waiting=6, kv=0.7), gauges(waiting=0, kv=0.1)
    d = pick("prefix_aware", rs, ctx([1, 2, 3, 4]), index)
    assert d.chosen.index == 0
    assert d.spilled is True


def test_least_load_never_reports_a_spill():
    rs = replicas(2)
    index = PrefixIndex(2, 100)
    index.record(1, [1, 2])
    rs[1].scraped = gauges(waiting=6)
    assert pick("least_load", rs, ctx([1, 2]), index).spilled is False


def test_no_ready_replica_raises():
    rs = replicas(2)
    for r in rs:
        r.state = ReplicaState.SLEEPING
    with pytest.raises(NoReplicaAvailable):
        pick("least_load", rs)


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="unknown policy"):
        make_policy("least_outstanding")
