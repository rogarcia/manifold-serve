"""bench/m2 parsing, pass rules and schedule on hand-written fixtures."""

import asyncio
import json
import os
import random
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "bench" / "m2"))

import derive_spec
import loadgen
import summarize as s

T0 = 1000.0
TRACE = json.loads(loadgen.TRACE_FILE.read_text())


def req(trace_t, cls="interactive", status=200, ttft=0.1, tpot=20.0, truncated=False, error=None):
    sent = T0 + trace_t
    return {"class": cls, "trace_t": trace_t, "t_scheduled": sent, "t_sent": sent,
            "t_first": None if ttft is None else sent + ttft, "t_done": sent + 5, "status": status,
            "n_chunks": 10, "tpot_ms": tpot, "truncated": truncated, "error": error}


def tick(trace_t, phases, serve=None, interactive=0):
    return {"t": T0 + trace_t, "phases": phases, "serve": serve or sum(p == "ready" for p in phases),
            "run": 2, "signal": {"interactive": interactive, "batch": 0, "in_flight": [0, 0]}}


def cell(requests=(), ticks=(), decisions=(), samples=(), events=(), name="c", scale=1.0, s_=170.0):
    return s.Cell(name, list(requests), list(ticks), list(decisions), list(samples), list(events),
                  {"t0": T0, "start": 0, "time_scale": scale, "lag_ms": {"p99": 1.0, "max": 2.0}},
                  {"S": str(s_)})


def test_parse_jsonl_and_samples():
    assert s.parse_jsonl('{"a": 1}\n\n{"a": 2}\n') == [{"a": 1}, {"a": 2}]
    text = "t\treplica\twaiting\trunning\tkv_usage\n10.5\t0\t0\t12\t0.1\n11.5\t1\t3\t4\t0.2\n"
    rows = s.parse_samples(text)
    assert rows[1] == {"t": 11.5, "replica": "1", "waiting": 3, "running": 4, "kv_usage": "0.2"}
    assert s.parse_samples("") == []


def test_percentile_interpolates():
    assert s.percentile([], 90) is None
    assert s.percentile([10.0], 90) == 10.0
    assert s.percentile([0.0, 10.0], 90) == 9.0


@pytest.mark.parametrize("r, good", [
    (req(0, ttft=0.5, tpot=40.0), True),
    (req(0, ttft=0.501), False),
    (req(0, tpot=40.1), False),
    (req(0, status=503, ttft=0.0, tpot=None), False),
    (req(0, truncated=True), False),
    (req(0, error="ReadError"), False),
    (req(0, ttft=None, tpot=None), False),
    (req(0, tpot=None), True),
])
def test_goodput_definition(r, good):
    assert s.is_good(r) is good


def test_hold_time_counts_against_ttft():
    assert not s.is_good(req(0, ttft=30.0))
    assert s.goodput([req(0), req(1, ttft=30.0)]) == 0.5
    assert s.goodput([]) is None


def test_gpu_seconds_counts_starting_to_off_and_clips():
    ticks = [tick(0, ["off", "off"]), tick(10, ["starting", "off"]), tick(20, ["ready", "starting"]),
             tick(30, ["ready", "stopping"]), tick(40, ["off", "off"]), tick(50, ["off", "off"])]
    assert s.gpu_seconds(ticks, T0, T0 + 50) == 10 + 20 + 20
    assert s.gpu_seconds(ticks, T0 + 25, T0 + 35) == 5 * 2 + 5 * 2


def test_gpu_seconds_holds_phases_across_a_crash_gap():
    ticks = [tick(0, ["ready", "off"]), tick(100, ["ready", "off"])]
    assert s.gpu_seconds(ticks, T0, T0 + 100) == 100


def test_split_cold_by_phase_at_arrival():
    ticks = [tick(0, ["off", "off"]), tick(10, ["starting", "off"]), tick(20, ["ready", "off"])]
    early, starting, late = req(-5), req(15), req(25)
    warm, cold = s.split_cold([early, starting, late], ticks)
    assert warm == [late]
    assert cold == [early, starting]


def test_p1_reads_the_trace_rate_at_the_request():
    ok = cell(ticks=[tick(100, ["ready", "off"], 1), tick(345, ["ready", "starting"], 2)])
    v = s.p1(ok)
    assert v.status == s.PASS and "8.00 req/s" in v.detail
    late = cell(ticks=[tick(400, ["ready", "starting"], 2)])
    assert s.p1(late).status == s.FAIL
    assert s.p1(cell(ticks=[tick(0, ["ready", "off"], 1)])).status == s.FAIL
    assert s.p1(None).status == s.NOT_EVALUATED


def test_p1_on_a_compressed_trace():
    ticks = [tick(0, ["off", "off"], 1), tick(345 / 20, ["ready", "starting"], 2)]
    assert "8.00 req/s" in s.p1(cell(ticks=ticks, scale=20.0)).detail


def test_p2_excludes_the_cold_window_and_compares_with_the_baseline():
    ticks = [tick(160, ["starting", "off"]), tick(200, ["ready", "off"])]
    cold = [req(170, ttft=30.0)] * 5
    loop = cell(requests=cold + [req(210 + i) for i in range(99)] + [req(400, ttft=2.0)],
                ticks=ticks)
    base = cell(requests=[req(200 + i) for i in range(100)])
    v = s.p2(loop, base)
    assert v.status == s.PASS, v.detail
    assert "t=200 to 585 99.0%" in v.detail and "5 cold-window" in v.detail
    bad = cell(requests=[req(210, ttft=2.0), req(211)], ticks=ticks)
    assert s.p2(bad, base).status == s.FAIL
    assert s.p2(loop, None).status == s.NOT_EVALUATED


def test_p2_starts_when_the_backlog_clears_and_still_prints_the_whole_window():
    ticks = [tick(200, ["ready", "off"], interactive=500), tick(295, ["ready", "off"], interactive=40)]
    held = [req(220, ttft=30.0)] * 10
    served = [req(300 + i) for i in range(90)]
    v = s.p2(cell(requests=held + served, ticks=ticks), cell(requests=served))
    assert v.status == s.PASS, v.detail
    assert "t=295 to 585 100.0%" in v.detail
    assert "from t=165 with the cold backlog 90.0%" in v.detail


def test_p2_gap_to_the_baseline_fails_above_the_floor():
    ticks = [tick(100, ["ready", "off"])]
    base = cell(requests=[req(200)] * 100)

    def loop(bad):
        return cell(requests=[req(200)] * (100 - bad) + [req(201, ttft=2.0)] * bad, ticks=ticks)

    assert s.p2(loop(2), base).status == s.PASS
    v = s.p2(loop(5), base)
    assert v.status == s.FAIL and "gap 5.0" in v.detail
    assert s.p2(loop(6), cell(requests=[req(200)] * 94 + [req(201, ttft=2.0)] * 6)).status == s.FAIL


def test_p3_against_the_static_fleet():
    half = [tick(0, ["ready", "off"]), tick(1080, ["off", "off"])]
    v = s.p3(cell(ticks=half))
    assert v.status == s.PASS and "0.500" in v.detail
    full = [tick(0, ["ready", "starting"]), tick(1080, ["off", "off"])]
    assert s.p3(cell(ticks=full)).status == s.FAIL


def test_p4_needs_the_gate_under_and_the_blind_cell_over():
    c2 = cell(requests=[req(600, tpot=30.0)] * 10)
    c4 = cell(requests=[req(600, tpot=60.0)] * 10)
    assert s.p4(c2, c4).status == s.PASS
    assert s.p4(c2, c2).status == s.FAIL
    assert s.p4(c4, c4).status == s.FAIL
    assert s.p4(c2, None).status == s.NOT_EVALUATED


def test_p5_every_cold_request_completes_within_s_plus_margin():
    ticks = [tick(0, ["off", "off"]), tick(100, ["ready", "off"])]
    ok = cell(requests=[req(50, ttft=180.0), req(60, "batch", ttft=400.0)], ticks=ticks)
    assert s.p5(ok).status == s.PASS
    slow = cell(requests=[req(50, ttft=186.0)], ticks=ticks)
    assert s.p5(slow).status == s.FAIL
    shed = cell(requests=[req(50, status=503, ttft=1.0)], ticks=ticks)
    assert s.p5(shed).status == s.FAIL
    assert s.p5(cell(ticks=ticks)).status == s.FAIL


def sample(t, waiting):
    return {"t": T0 + t, "replica": "0", "waiting": waiting, "running": 100}


def test_p6_waiting_stays_zero_while_tpot_is_over():
    requests = [req(100, tpot=60.0)] * 6 + [req(200, tpot=20.0)] * 6
    assert s.p6(cell(requests, samples=[sample(105, 0), sample(205, 7)])).status == s.PASS
    v = s.p6(cell(requests, samples=[sample(105, 3)]))
    assert v.status == s.FAIL and "first waiting t=105" in v.detail
    assert s.p6(cell([req(100)] * 6)).status == s.FAIL
    assert s.p6(None).status == s.NOT_EVALUATED


def reclaim_ticks():
    return [tick(50, ["ready", "ready"]), tick(61, ["ready", "stopping"]),
            tick(80, ["ready", "off"]), tick(81, ["ready", "starting"]),
            tick(250, ["ready", "ready"])]


def test_p7_counts_losses_and_times_the_slot_from_engine_exit():
    events = [{"t": T0 + 60, "event": "reclaim", "args": "1 30"}]
    ok = cell([req(1), req(2, "batch"), req(3, truncated=True)], reclaim_ticks(), events=events)
    v = s.p7(ok)
    assert v.status == s.PASS, v.detail
    assert "truncated 1" in v.detail and "170.0 s after the engine" in v.detail
    assert "190.0 s after the reclaim" in v.detail
    batch_lost = cell([req(2, "batch", status=429)], reclaim_ticks(), events=events)
    assert s.p7(batch_lost).status == s.FAIL
    unstarted = cell([req(2, status=502, ttft=0.1)], reclaim_ticks(), events=events)
    assert s.p7(unstarted).status == s.FAIL
    never = cell([req(1)], reclaim_ticks()[:4], events=events)
    assert "did not come back" in s.p7(never).detail
    assert s.p7(cell([req(1)], reclaim_ticks())).status == s.FAIL


def test_p8_share_in_the_minute_after_ready():
    ticks = [tick(100, ["ready", "starting"]), tick(300, ["ready", "ready"])]

    def decision(t, replica):
        return {"t_granted": T0 + t, "replica_index": replica}

    good = [decision(301, 1), decision(302, 0), decision(303, 1), decision(400, 0)]
    v = s.p8(cell(ticks=ticks, decisions=good))
    assert v.status == s.PASS and "66.7%" in v.detail
    assert s.p8(cell(ticks=ticks, decisions=[decision(301, 0)])).status == s.FAIL


def test_c6_agreement_is_on_direction():
    a = [s.Verdict("P1", s.PASS, ""), s.Verdict("P2", s.FAIL, ""), s.Verdict("P3", s.PASS, "")]
    b = [s.Verdict("P1", s.PASS, ""), s.Verdict("P2", s.PASS, ""),
         s.Verdict("P3", s.NOT_EVALUATED, "")]
    lines = s.agreement(a, b)
    assert lines[0].startswith("P1 AGREE")
    assert lines[1].startswith("P2 DISAGREE")
    assert lines[2].startswith("P3 NOT EVALUATED")


def test_report_with_no_cells_marks_every_rule():
    lines = s.report({})
    rules = [x for x in lines if x.startswith("P") and "needs" in x]
    assert {x.split()[0] for x in rules} == {f"P{i}" for i in range(1, 9)}


def test_backlog_clear():
    ticks = [tick(0, ["ready", "off"], interactive=500), tick(2, ["ready", "off"], interactive=97),
             tick(3, ["ready", "off"], interactive=96)]
    assert s.backlog_clear(ticks) == T0 + 3


def test_load_cell_and_find_cell_need_done(tmp_path):
    d = tmp_path / "c2-static-2-s0"
    d.mkdir()
    (d / "requests.jsonl").write_text(json.dumps(req(10)) + "\n")
    (d / "requests.meta.json").write_text(json.dumps({"t0": 1, "start": 105, "time_scale": 1}))
    (d / "env.json").write_text('{"S": "170"}')
    assert s.find_cell(tmp_path, "c2") is None
    (d / "DONE").touch()
    c = s.find_cell(tmp_path, "c2")
    assert len(c.requests) == 1 and c.ticks == [] and c.to_trace(2.0) == 106.0


def test_derive_spec_reproduces_the_spec():
    assert derive_spec.target_interactive(170) == 26
    spec = derive_spec.derive(170.0, 0, 2, derive_spec.RAMP_RPS_PER_S, {})
    assert spec.idle_to_zero_s == 170.0 and spec.target_interactive == 26
    assert derive_spec.derive(10.0, 2, 2, 0.02, {"tick_s": 0.25}).tick_s == 0.25
    with pytest.raises(SystemExit):
        derive_spec.target_interactive(170, ramp=1.0)


CONVERSATIONS = [
    {"id": f"c{i}", "messages": [
        {"role": "user" if t % 2 == 0 else "assistant", "content": "x" * (40 + t)}
        for t in range(2 * (1 + i % 6))]}
    for i in range(3000)]


def schedule(seed=0, scale=1.0):
    return loadgen.build_schedule(TRACE["main"], CONVERSATIONS, seed, 384, scale)


def test_schedule_matches_the_trace_and_ends_on_user_turns():
    full = schedule()
    interactive = [a for a in full if a.cls == "interactive"]
    assert 5900 < len(interactive) < 6600
    assert len(full) - len(interactive) == 700
    spike = [a for a in interactive if 585 <= a.t < 645]
    assert 1250 < len(spike) < 1650
    assert not [a for a in interactive if 780 <= a.t < 960]
    assert all(a.messages[-1]["role"] == "user" for a in full)
    assert {a.max_tokens for a in full} >= {10}
    assert all(len(a.messages) == 1 for a in full if a.cls == "batch")


def test_same_seed_same_requests_and_a_window_only_clips():
    a, b = schedule(), schedule()
    assert a == b
    assert schedule(seed=1) != a
    clipped = loadgen.clip(a, 105, 645)
    assert clipped == [x for x in b if 105 <= x.t < 645]
    assert {x.t for x in clipped if x.cls == "batch"} == {200, 560}


def test_turns_of_one_session_arrive_in_order():
    seen = {}
    for a in schedule():
        if a.cls == "interactive":
            assert len(a.messages) > seen.get(a.session, 0)
            seen[a.session] = len(a.messages)


def test_compression_keeps_the_rate_and_divides_batches():
    compressed = schedule(scale=20.0)
    assert sum(a.cls == "batch" for a in compressed) == 20 + 15
    assert 250 < sum(a.cls == "interactive" for a in compressed) < 370
    assert loadgen.rate_at(TRACE["main"]["phases"], 345) == pytest.approx(8.0)


def test_poisson_gaps_have_the_rate():
    phases = [{"t0": 0, "t1": 10_000, "rps0": 5, "rps1": 5}]
    times = loadgen.arrival_times(phases, random.Random(3), 1.0)
    assert len(times) == pytest.approx(50_000, rel=0.02)


def sse_client(body, status=200):
    transport = httpx.MockTransport(lambda _: httpx.Response(status, stream=httpx.ByteStream(body)))
    return httpx.AsyncClient(transport=transport, base_url="http://x")


def record(client, arrival, tmp_path, timeout=5.0):
    path = tmp_path / "out.jsonl"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT)
    asyncio.run(loadgen.one(client, arrival, 1.0, "m", timeout, fd))
    return json.loads(path.read_text())


INTERACTIVE = loadgen.Arrival(0.0, "interactive", "s1", [{"role": "user", "content": "x"}], 5)
BATCH = loadgen.Arrival(0.0, "batch", None, [{"role": "user", "content": "x"}], 5)


def test_request_shapes():
    body = loadgen.request_body(INTERACTIVE, "m")
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert loadgen.request_headers(INTERACTIVE) == {"x-session-id": "s1"}
    assert loadgen.request_body(BATCH, "m")["stream"] is False
    assert loadgen.request_headers(BATCH) == {"x-manifold-class": "batch"}


def test_stream_record(tmp_path):
    usage = b'data: {"choices": [], "usage": {"prompt_tokens": 9, "completion_tokens": 5}}\n\n'
    body = b'data: {"a": 1}\n\ndata: {"a": 2}\n\n' + usage + b"data: [DONE]\n\n"
    r = record(sse_client(body), INTERACTIVE, tmp_path)
    assert r["status"] == 200 and r["n_chunks"] == 4 and r["completion_tokens"] == 5
    assert r["truncated"] is False and r["tpot_ms"] is not None and r["t_first"] is not None


def test_truncated_event_and_a_missing_done(tmp_path):
    body = b'data: {"a": 1}\n\ndata: {"error": {"message": "lost", "type": "truncated"}}\n\n'
    assert record(sse_client(body), INTERACTIVE, tmp_path)["truncated"] is True
    (tmp_path / "out.jsonl").unlink()
    assert record(sse_client(b'data: {"a": 1}\n\n'), INTERACTIVE, tmp_path)["truncated"] is True


def test_batch_and_shed_records(tmp_path):
    r = record(sse_client(b'{"choices": []}'), BATCH, tmp_path)
    assert r["status"] == 200 and r["truncated"] is False and r["tpot_ms"] is None
    (tmp_path / "out.jsonl").unlink()
    r = record(sse_client(b'{"error": {}}', status=429), BATCH, tmp_path)
    assert r["status"] == 429 and r["t_done"] is not None


def test_connection_failure_is_recorded(tmp_path):
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(refuse), base_url="http://x")
    r = record(client, INTERACTIVE, tmp_path)
    assert r["status"] is None and "ConnectError" in r["error"] and r["truncated"] is False
