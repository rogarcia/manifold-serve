"""The scaling decision on tables, plus a fluid model of the calibration curve."""

import pytest

from manifold.autoscaler.policy import (
    Memo,
    Observation,
    Phase,
    Signal,
    Spec,
    Target,
    replicas_needed,
    slot_states,
    step,
)
from manifold.router.replicas import ReplicaState

OFF, STARTING, READY, STOPPING = Phase.OFF, Phase.STARTING, Phase.READY, Phase.STOPPING
SPEC = Spec()


def obs(now, phases, interactive=0, batch=0, in_flight=None):
    return Observation(now, tuple(phases),
                       Signal(interactive, batch, tuple(in_flight or [0] * len(phases))))


def run_steps(frames, memo, spec=SPEC):
    target = None
    for o in frames:
        target, memo = step(o, spec, memo)
    return target, memo


def test_load_adds_interactive_and_batch_work():
    assert replicas_needed(Signal(39, 0, (0, 0)), SPEC) == 1.5
    assert replicas_needed(Signal(0, 5400, (0, 0)), SPEC) == pytest.approx(1.0)


def test_first_request_starts_a_replica_on_the_same_tick():
    target, _ = step(obs(0, [OFF, OFF], batch=1), SPEC, Memo(serve=0))
    assert target == Target(1, 1)


def test_no_demand_at_zero_stays_at_zero():
    target, _ = step(obs(0, [OFF, OFF]), SPEC, Memo(serve=0))
    assert target == Target(0, 0)


def test_scale_up_waits_for_the_window():
    memo = Memo(serve=1, n_ready=1)
    frames = [obs(t, [READY, OFF], interactive=45) for t in range(9)]
    target, memo = run_steps(frames, memo)
    assert target == Target(1, 1)
    target, _ = step(obs(10, [READY, OFF], interactive=45), SPEC, memo)
    assert target == Target(2, 2)


def test_a_brief_spike_does_not_scale_up():
    frames = [obs(t, [READY, OFF], interactive=45) for t in range(9)]
    frames += [obs(9, [READY, OFF], interactive=5)]
    frames += [obs(t, [READY, OFF], interactive=45) for t in range(10, 15)]
    target, _ = run_steps(frames, Memo(serve=1))
    assert target == Target(1, 1)


def test_no_second_start_while_one_is_starting():
    frames = [obs(t, [STARTING, OFF], interactive=90) for t in range(60)]
    target, _ = run_steps(frames, Memo(serve=1))
    assert target == Target(1, 1)


def test_cold_start_backlog_does_not_start_a_replica_during_settle():
    frames = [obs(0, [STARTING, OFF], interactive=90)]
    frames += [obs(t, [READY, OFF], interactive=90) for t in range(1, 60)]
    target, memo = run_steps(frames, Memo(serve=1))
    assert target == Target(1, 1)
    target, _ = run_steps([obs(61, [READY, OFF], interactive=90)], memo)
    assert target == Target(2, 2)


def test_scale_down_needs_load_below_the_hysteresis_line_for_the_window():
    at_line = [obs(t, [READY, READY], interactive=18, in_flight=[9, 9]) for t in range(121)]
    target, _ = run_steps(at_line, Memo(serve=2))
    assert target.serve == 1, "18 / 26 is under 0.7 replicas"

    above = [obs(t, [READY, READY], interactive=19, in_flight=[10, 9]) for t in range(200)]
    target, _ = run_steps(above, Memo(serve=2))
    assert target == Target(2, 2)


def test_no_scale_down_while_a_lost_slot_restarts():
    """Removing the top slot here would drain the only engine that is serving."""
    frames = [obs(t, [STARTING, READY], interactive=5, in_flight=[0, 5]) for t in range(200)]
    target, _ = run_steps(frames, Memo(serve=2))
    assert target == Target(2, 2)


def test_a_draining_slot_runs_until_empty():
    memo = Memo(serve=1)
    target, memo = step(obs(0, [READY, READY], interactive=5, in_flight=[3, 2]), SPEC, memo)
    assert target == Target(1, 2)
    target, memo = step(obs(1, [READY, READY], interactive=3, in_flight=[3, 0]), SPEC, memo)
    assert target == Target(1, 1)


def test_a_draining_slot_is_cut_at_the_deadline():
    memo = Memo(serve=1)
    frames = [obs(t, [READY, READY], interactive=5, in_flight=[3, 2]) for t in range(30)]
    target, memo = run_steps(frames, memo)
    assert target == Target(1, 2)
    target, _ = step(obs(30, [READY, READY], interactive=5, in_flight=[3, 2]), SPEC, memo)
    assert target == Target(1, 1)


def test_demand_returning_during_a_drain_keeps_the_slot():
    memo = Memo(serve=1, n_ready=2)
    frames = [obs(t, [READY, READY], interactive=70, in_flight=[35, 35]) for t in range(11)]
    target, _ = run_steps(frames, memo)
    assert target == Target(2, 2)


def test_idle_reaches_zero_after_the_break_even_and_not_before():
    frames = [obs(t, [READY, OFF]) for t in range(170)]
    target, memo = run_steps(frames, Memo(serve=1))
    assert target == Target(1, 1)
    target, _ = step(obs(170, [READY, OFF]), SPEC, memo)
    assert target == Target(0, 0)


def test_a_floor_keeps_one_warm():
    spec = Spec(floor=1)
    frames = [obs(t, [READY, OFF]) for t in range(400)]
    target, _ = run_steps(frames, Memo(serve=1), spec)
    assert target == Target(1, 1)


def test_a_lost_slot_inside_serve_is_requested_again():
    target, _ = step(obs(0, [OFF, READY], interactive=40, in_flight=[0, 40]), SPEC, Memo(serve=2))
    assert target == Target(2, 2)


def test_recovery_after_a_crash_changes_nothing():
    o = obs(0, [READY, READY], interactive=3, in_flight=[1, 2])
    target, _ = step(o, SPEC, Memo.recover(o))
    assert target == Target(2, 2)


def test_slot_states_pair_each_phase_with_a_router_state():
    target = Target(1, 2)
    assert slot_states(target, (READY, READY)) == [ReplicaState.READY, ReplicaState.DRAINING]
    assert slot_states(Target(2, 2), (STOPPING, STARTING)) == [ReplicaState.DRAINING,
                                                               ReplicaState.OFF]
    assert slot_states(Target(0, 0), (OFF, OFF)) == [ReplicaState.OFF, ReplicaState.OFF]


def test_second_replica_is_ready_before_the_knee_on_the_trace_ramp():
    """A fluid model: Little's law on the measured curve, with a start time of 170 s."""
    in_flight_table = [16, 32, 48, 64, 96, 128]
    rps_table = [3.5, 5.9, 7.7, 9.0, 10.35, 10.6]
    start_s, ramp = 170, 10 / 360

    def in_flight_at(rps):
        for (r0, n0), (r1, n1) in zip(zip(rps_table, in_flight_table),
                                      zip(rps_table[1:], in_flight_table[1:])):
            if rps <= r1:
                return max(0.0, n0 + (n1 - n0) * (rps - r0) / (r1 - r0))
        return in_flight_table[-1]

    memo, phases, ready_in, peak = Memo(serve=1, n_ready=1), [READY, OFF], None, 0.0
    for t in range(361):
        rps = 3 + ramp * t
        per_replica = rps / sum(p is READY for p in phases)
        n = in_flight_at(per_replica)
        peak = max(peak, n)
        total = n * sum(p is READY for p in phases)
        signal = Signal(round(total), 0, (round(n), round(n) if phases[1] is READY else 0))
        target, memo = step(Observation(t, tuple(phases), signal), SPEC, memo)
        if target.run == 2 and phases[1] is OFF:
            phases[1], ready_in = STARTING, t + start_s
        if ready_in == t:
            phases[1] = READY
    assert ready_in is not None and ready_in <= 360
    assert peak < 96, f"in flight per replica reached {peak:.0f}, past the knee"
