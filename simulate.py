"""Deterministic local AI-router comparison; no network or provider calls.

Run: python3 simulate.py [--b-initial-used 112] [--b-reset-at 50]
Times in the event engine are integer microseconds. Costs are integer micro-USDC.
All provider behavior and prices are explicit synthetic assumptions.
"""

import argparse
from collections import deque
from decimal import Decimal
from heapq import heappop, heappush
from itertools import count
from random import Random
from statistics import median

US = 1_000_000
RATE, SECONDS = 5, 60
RECOVERY_US, DEADLINE_US = 30 * US, 9 * US
STREAM_US, B_429_US = 300_000, 50_000
B_LIMIT, B_WINDOW_US = 120, 60 * US
BUDGET_WINDOW_US = 10 * US
A_COST_MICRO, B_COST_MICRO, B_429_COST_MICRO = 4_000, 12_000, 100
DEFAULT_INITIAL_USED, DEFAULT_RESET_AT_S = 112, 50

# At the same microsecond all arrivals enter the denominator first. A/B responses
# then resolve before wait expiry; waits resolve before stream ends/deadlines.
# Requests at the same time and priority resolve in ascending request-ID order.
PRIORITY = {
    "arrive": 0,
    "a_error": 1, "a_token": 1, "b_token": 1, "b_429": 1,
    "timeout": 2, "stream_end": 3, "deadline": 4,
}


def to_us(value):
    """Convert explicit second values exactly at microsecond precision."""
    return int(Decimal(str(value)) * US)


def workload(seed):
    """300 arrivals at exact 200ms intervals; one sampled A/B trace per ID."""
    rng = Random(seed)
    rows = []
    for i in range(RATE * SECONDS):
        at_us = i * US // RATE
        u = rng.random()
        mode = "rate" if at_us < RECOVERY_US and u < 0.20 else (
            "slow" if at_us < RECOVERY_US else "fast")
        a_delay_us = to_us("0.15") if mode == "rate" else (
            round((8.2 + 0.4 * rng.random()) * US) if mode == "slow"
            else round((0.25 + 0.2 * rng.random()) * US))
        b_delay_us = round((0.45 + 0.2 * rng.random()) * US)
        rows.append((at_us, mode, a_delay_us, b_delay_us))
    return rows


def p95_us(values):
    values = sorted(values)
    return values[(95 * len(values) + 99) // 100 - 1] if values else None


def simulate(rows, wait_us, *, b_initial_used=DEFAULT_INITIAL_USED,
             b_reset_at_s=DEFAULT_RESET_AT_S):
    """Model first token and terminal stream completion before a 9s deadline.

    B admission is known synchronously at dispatch and cancels A immediately.
    An exhausted B responds 429 after 50ms and leaves the original A in flight.
    The synthetic stream ends 300ms after the first token. Real admission and
    stream termination timing must be tested separately against the real router.
    """
    if not 0 <= b_initial_used <= B_LIMIT:
        raise ValueError("b_initial_used must be 0..120")
    if not 1 <= b_reset_at_s <= 60:
        raise ValueError("b_reset_at_s must be 1..60")
    serial, events = count(), []

    def put(t_us, i, kind):
        heappush(events, (t_us, PRIORITY[kind], i, next(serial), kind))

    for i, (at_us, _, _, _) in enumerate(rows):
        put(at_us, i, "arrive")
    state = [dict(a_inflight=False, b_inflight=False, first_token=False,
                  completed=False, closed=False, provider=None, ttft_us=None,
                  first_token_at_us=None, completed_at_us=None, log=[])
             for _ in rows]
    arrivals, b_decisions = deque(), deque()
    b_window, b_used = 0, b_initial_used
    reset_offset_us = (60 - b_reset_at_s) * US
    attempts = b_admitted = b_429 = b_decision_count = cancelled = 0
    cost_micro_usdc = 0

    def fallback(t_us, i, from_timeout):
        nonlocal attempts, b_admitted, b_429, b_decision_count, cancelled
        nonlocal cost_micro_usdc, b_window, b_used
        s = state[i]
        cutoff = t_us - BUDGET_WINDOW_US
        while arrivals and arrivals[0] <= cutoff:
            arrivals.popleft()
        while b_decisions and b_decisions[0] <= cutoff:
            b_decisions.popleft()
        # Exact integer 10% gate; no free cold-start burst. 429s spend budget.
        if 10 * (len(b_decisions) + 1) > len(arrivals):
            s["log"].append((t_us, "B budget denied; original A continues" if from_timeout
                             else "B budget denied; A already rate-limited"))
            return
        b_decisions.append(t_us)
        b_decision_count += 1
        attempts += 1  # Every admitted routing decision dispatches B.
        window = (t_us + reset_offset_us) // B_WINDOW_US
        if window != b_window:
            b_window, b_used = window, 0
        if b_used >= B_LIMIT:
            b_429 += 1
            cost_micro_usdc += B_429_COST_MICRO
            s["b_inflight"] = True
            s["log"].append((t_us, "B dispatched; quota exhausted"))
            put(t_us + B_429_US, i, "b_429")
            return
        b_used += 1
        b_admitted += 1
        cost_micro_usdc += B_COST_MICRO
        if from_timeout:
            s["a_inflight"] = False
            cancelled += 1
            s["log"].append((t_us, "B admitted; original A cancelled"))
        else:
            s["log"].append((t_us, "B admitted after A rate limit"))
        s["b_inflight"] = True
        put(t_us + rows[i][3], i, "b_token")

    while events:
        t_us, _, i, _, kind = heappop(events)
        s = state[i]
        at_us, mode, a_delay_us, _ = rows[i]
        if kind == "arrive":
            arrivals.append(t_us)
            s["a_inflight"] = True
            s["log"].append((t_us, "A dispatched (" + mode + ")"))
            attempts += 1
            cost_micro_usdc += A_COST_MICRO  # Include canceled/failed work.
            put(t_us + a_delay_us, i, "a_error" if mode == "rate" else "a_token")
            put(t_us + wait_us, i, "timeout")
            put(t_us + DEADLINE_US, i, "deadline")
        elif kind == "a_error" and s["a_inflight"] and not s["first_token"] and not s["closed"]:
            s["a_inflight"] = False
            s["log"].append((t_us, "A rate-limited"))
            fallback(t_us, i, False)
        elif kind == "timeout" and s["a_inflight"] and not s["first_token"] and not s["closed"]:
            s["log"].append((t_us, "A wait expired; check B budget before cancel"))
            fallback(t_us, i, True)
        elif kind == "b_429":
            s["b_inflight"] = False
            if s["first_token"]:
                outcome = "B 429 after A first token"
            elif s["a_inflight"] and not s["closed"]:
                outcome = "B 429; original A continues"
            else:
                outcome = "B 429; A already failed or deadline passed"
            s["log"].append((t_us, outcome))
        elif kind in ("a_token", "b_token") and not s["first_token"] and not s["closed"]:
            provider = "A" if kind == "a_token" else "B"
            flag = "a_inflight" if provider == "A" else "b_inflight"
            if s[flag] and t_us < at_us + DEADLINE_US:
                s[flag] = False
                s["first_token"], s["provider"] = True, provider
                s["ttft_us"], s["first_token_at_us"] = t_us - at_us, t_us
                s["log"].append((t_us, provider + " first token"))
                put(t_us + STREAM_US, i, "stream_end")
        elif kind == "stream_end" and s["first_token"] and not s["completed"]:
            if t_us < at_us + DEADLINE_US:
                s["completed"], s["completed_at_us"] = True, t_us
                s["log"].append((t_us, s["provider"] + " stream completed"))
            else:
                s["log"].append((t_us, "stream ended at or after client deadline"))
        elif kind == "deadline" and not s["completed"]:
            s["closed"] = True
            s["log"].append((t_us, "client deadline; stream incomplete"))

    completed = [s for s in state if s["completed"]]
    first_a_token = min((s["first_token_at_us"] for i, s in enumerate(state)
                         if rows[i][0] >= RECOVERY_US and s["provider"] == "A"), default=None)
    first_a_dispatch = min((at_us for at_us, *_ in rows if at_us >= RECOVERY_US), default=None)
    return dict(total=len(rows), completed=len(completed),
                first_token_success=sum(s["first_token"] for s in state),
                attempts=attempts, cost_micro_usdc=cost_micro_usdc,
                b_admitted=b_admitted, b_429=b_429,
                b_decisions=b_decision_count, cancelled=cancelled,
                p95_completed_ttft_us=p95_us([s["ttft_us"] for s in completed]),
                p95_penalized_us=p95_us([s["ttft_us"] if s["completed"] else DEADLINE_US
                                       for s in state]),
                return_a_dispatch_us=(first_a_dispatch - RECOVERY_US
                                      if first_a_dispatch is not None else None),
                first_a_token_us=(first_a_token - RECOVERY_US
                                  if first_a_token is not None else None),
                state=state)


def render_event_log(log):
    return " | ".join(f"{t_us / US:.3f}s {event}" for t_us, event in log)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b-initial-used", type=int, default=DEFAULT_INITIAL_USED,
                        help="accepted B requests already used in the current 120/min window (default 112)")
    parser.add_argument("--b-reset-at", type=int, default=DEFAULT_RESET_AT_S,
                        help="seconds after t=0 when B's first fixed window resets (default 50)")
    args = parser.parse_args(argv)
    if not 0 <= args.b_initial_used <= B_LIMIT:
        parser.error("--b-initial-used must be between 0 and 120")
    if not 1 <= args.b_reset_at <= 60:
        parser.error("--b-reset-at must be between 1 and 60")
    runs = []
    print(f"scenario: B initial used={args.b_initial_used}/{B_LIMIT}; first reset at {args.b_reset_at}s; stream duration={STREAM_US / US:.3f}s")
    print("seed policy completed% first_token% completed_TTFT_p95_s penalized_p95_s attempts/completed cost/completed_USDC B_admitted B_429 A_cancel return_A_dispatch_s first_A_token_s")
    for seed in range(1, 11):
        rows = workload(seed)
        pair = {name: simulate(rows, to_us(wait), b_initial_used=args.b_initial_used,
                               b_reset_at_s=args.b_reset_at)
                for name, wait in (("current", 8), ("candidate", 2))}
        runs.append((seed, rows, pair))
        for name, r in pair.items():
            n = r["completed"]
            print(f'{seed:2} {name:9} {100*n/r["total"]:9.2f} '
                  f'{100*r["first_token_success"]/r["total"]:11.2f} '
                  f'{r["p95_completed_ttft_us"]/US:20.3f} '
                  f'{r["p95_penalized_us"]/US:15.3f} '
                  f'{r["attempts"]/n:18.3f} '
                  f'{r["cost_micro_usdc"]/(US*n):19.6f} '
                  f'{r["b_admitted"]:10} {r["b_429"]:5} '
                  f'{r["cancelled"]:8} '
                  f'{r["return_a_dispatch_us"]/US:19.3f} '
                  f'{r["first_a_token_us"]/US:15.3f}')
    for name in ("current", "candidate"):
        policy_runs = [pair[name] for _, _, pair in runs]
        total = sum(r["total"] for r in policy_runs)
        n = sum(r["completed"] for r in policy_runs)
        ttfts = [s["ttft_us"] for r in policy_runs for s in r["state"] if s["completed"]]
        penalized = [s["ttft_us"] if s["completed"] else DEADLINE_US
                     for r in policy_runs for s in r["state"]]
        print(f'pooled {name}: completed={n}/{total}; first_token='
              f'{sum(r["first_token_success"] for r in policy_runs)}/{total}; '
              f'completed_TTFT_p95={p95_us(ttfts)/US:.6f}s; '
              f'penalized_p95={p95_us(penalized)/US:.6f}s; '
              f'attempts/completed={sum(r["attempts"] for r in policy_runs)/n:.6f}; '
              f'cost/completed_USDC={sum(r["cost_micro_usdc"] for r in policy_runs)/(US*n):.9f}; '
              f'B_admitted={sum(r["b_admitted"] for r in policy_runs)}; '
              f'B_429={sum(r["b_429"] for r in policy_runs)}')
    losses = [100 * (pair["current"]["completed"] - pair["candidate"]["completed"])
              / len(rows) for _, rows, pair in runs]
    breaches = sum(100 * (pair["current"]["completed"] - pair["candidate"]["completed"])
                   > len(rows) for _, rows, pair in runs)  # Strictly >1pp, or >3 of 300.
    print(f"median seed completion loss={median(losses):.6f} percentage points; strict >1pp breaches={breaches}/10")
    seed, rows, pair = runs[0]
    representative = next((i for i, (row, s) in enumerate(zip(rows, pair["candidate"]["state"]))
                           if row[1] == "slow" and any("B admitted" in e for _, e in s["log"])), None)
    if representative is None:
        print("representative admission: none in seed 1 under this quota configuration")
    else:
        print(f"representative admission: seed={seed}, id={representative}, arrival={rows[representative][0]/US:.3f}s")
        for name, r in pair.items():
            print(name + ": " + render_event_log(r["state"][representative]["log"]))
    denied = next((i for i, s in enumerate(pair["candidate"]["state"])
                   if any("B 429; original A continues" in e for _, e in s["log"])), None)
    recovered = next(i for i, row in enumerate(rows) if row[0] >= RECOVERY_US)
    if denied is None:
        print("B mock 429: none in seed 1 under this quota configuration")
    for label, i in (("B mock 429", denied), ("first post-recovery A dispatch", recovered)):
        if i is None:
            continue
        print(f"{label}: seed={seed}, id={i}")
        for name, r in pair.items():
            print(name + ": " + render_event_log(r["state"][i]["log"]))


if __name__ == "__main__":
    main()
