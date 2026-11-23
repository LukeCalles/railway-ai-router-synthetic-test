"""Behavioral and exact-window regressions for the synthetic router model."""

from pathlib import Path
import subprocess
import sys
import unittest

import simulate


def row(at, mode="fast", a_delay="0.3", b_delay="0.5"):
    return (simulate.to_us(at), mode, simulate.to_us(a_delay), simulate.to_us(b_delay))


class RouterSimulationTests(unittest.TestCase):
    def test_replay_uses_integer_clock_and_identical_request_trace(self):
        rows = simulate.workload(7)
        self.assertEqual(rows, simulate.workload(7))
        self.assertEqual(len(rows), 300)
        self.assertTrue(all(isinstance(t, int) for r in rows for t in (r[0], r[2], r[3])))
        a = simulate.simulate(rows, simulate.to_us(2))
        b = simulate.simulate(rows, simulate.to_us(2))
        self.assertEqual((a["completed"], a["attempts"], a["cost_micro_usdc"]),
                         (b["completed"], b["attempts"], b["cost_micro_usdc"]))
        self.assertTrue(all(sum("A dispatched" in event for _, event in s["log"]) == 1
                            for s in a["state"]))
        self.assertTrue(all(isinstance(t, int) for s in a["state"] for t, _ in s["log"]))

    def test_b_429_counts_attempt_and_preserves_original_a(self):
        rows = [row("0", "slow", "8.4")] * 10
        r = simulate.simulate(rows, simulate.to_us(2), b_initial_used=120)
        self.assertEqual((r["completed"], r["first_token_success"]), (10, 10))
        self.assertEqual((r["cancelled"], r["b_429"], r["b_decisions"]), (0, 1, 1))
        self.assertEqual(r["attempts"], 11)
        self.assertEqual(r["cost_micro_usdc"], 10 * simulate.A_COST_MICRO + simulate.B_429_COST_MICRO)
        self.assertIn((simulate.to_us("2.05"), "B 429; original A continues"), r["state"][0]["log"])

    def test_admitted_b_cancels_a_and_charges_both_provider_attempts(self):
        rows = [row("0", "slow", "8.4")] * 10
        r = simulate.simulate(rows, simulate.to_us(2))
        self.assertEqual((r["completed"], r["cancelled"], r["b_admitted"]), (10, 1, 1))
        self.assertEqual(r["attempts"], 11)
        self.assertEqual(r["cost_micro_usdc"], 10 * simulate.A_COST_MICRO + simulate.B_COST_MICRO)
        self.assertEqual(r["state"][0]["provider"], "B")
        self.assertEqual(r["state"][0]["ttft_us"], simulate.to_us("2.5"))
        self.assertEqual(r["state"][0]["completed_at_us"], simulate.to_us("2.8"))

    def test_first_token_is_not_terminal_completion(self):
        r = simulate.simulate([row("0", "slow", "8.8")], simulate.to_us(8))
        self.assertEqual(r["first_token_success"], 1)
        self.assertEqual(r["completed"], 0)  # 8.8s first token + 0.3s stream > 9s deadline.
        self.assertIsNone(r["p95_completed_ttft_us"])
        self.assertEqual(r["p95_penalized_us"], simulate.DEADLINE_US)
        self.assertIn((simulate.to_us("9"), "client deadline; stream incomplete"),
                      r["state"][0]["log"])

    def test_stream_end_exactly_at_deadline_does_not_complete(self):
        at_deadline = simulate.simulate([row("0", "slow", "8.7")], simulate.to_us(8))
        before_deadline = simulate.simulate([row("0", "slow", "8.699999")], simulate.to_us(8))
        self.assertEqual((at_deadline["first_token_success"], at_deadline["completed"]), (1, 0))
        self.assertIn((simulate.DEADLINE_US, "stream ended at or after client deadline"),
                      at_deadline["state"][0]["log"])
        self.assertEqual(before_deadline["completed"], 1)

    def test_same_time_response_resolves_before_wait(self):
        r = simulate.simulate([row("0", "slow", "2")], simulate.to_us(2))
        self.assertEqual(r["first_token_success"], 1)
        self.assertFalse(any("A wait expired" in event for _, event in r["state"][0]["log"]))

    def test_b_fixed_window_resets_exactly_at_fifty_seconds(self):
        rows = [row("49", "rate", "0.15")] * 10 + [row("50", "rate", "0.15")] * 10
        r = simulate.simulate(rows, simulate.to_us(2), b_initial_used=120)
        self.assertEqual(r["b_429"], 1)  # Before reset.
        self.assertEqual(r["b_admitted"], 1)  # After reset.
        self.assertEqual(r["attempts"], 22)

    def test_arrival_at_exact_budget_cutoff_is_excluded(self):
        def case(extra_at=None):
            rows = [row("0.2")] + [row(str(t)) for t in range(1, 8)]
            rows += [row("8.2", "slow", "8.4"), row("10.2")]
            if extra_at is not None:
                rows.append(row(extra_at))
            return simulate.simulate(rows, simulate.to_us(2), b_initial_used=0)["state"][8]["log"]
        # At t=10.2, the arrival at 0.2 is exactly at the excluded left edge.
        self.assertTrue(any("B budget denied" in e for _, e in case()))
        # One microsecond inside adds the tenth counted arrival, including the
        # t=10.2 arrival, which is processed before this same-time fallback.
        self.assertTrue(any("B admitted" in e for _, e in case("0.200001")))
        self.assertTrue(any("B budget denied" in e for _, e in case("0.199999")))

    def test_decision_at_exact_budget_cutoff_is_excluded(self):
        def case(first_shift_us=0, second_shift_us=0):
            first_at = first_shift_us
            second_at = simulate.to_us("10") + second_shift_us
            rows = [(first_at, "slow", simulate.to_us("8.4"), simulate.to_us("0.5"))]
            rows += [(first_at, "fast", simulate.to_us("0.3"), simulate.to_us("0.5"))] * 9
            rows += [(second_at, "slow", simulate.to_us("8.4"), simulate.to_us("0.5"))]
            rows += [(second_at, "fast", simulate.to_us("0.3"),
                      simulate.to_us("0.5"))] * 9
            r = simulate.simulate(rows, simulate.to_us("0.2"), b_initial_used=0)
            return r["state"][10]["log"]
        # First decision at 0.2s is exactly 10s behind second at 10.2s.
        self.assertTrue(any("B admitted" in e for _, e in case()))
        # Moving the first decision one microsecond inside keeps it in budget.
        self.assertTrue(any("B budget denied" in e for _, e in case(first_shift_us=1)))
        # Moving the second decision one microsecond earlier has the same effect.
        self.assertTrue(any("B budget denied" in e for _, e in case(second_shift_us=-1)))

    def test_cli_with_all_b_slots_exhausted_never_requires_admission_trace(self):
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("simulate.py")),
             "--b-initial-used", "120", "--b-reset-at", "60"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("representative admission: none in seed 1", proc.stdout)
        self.assertIn("B mock 429: seed=1", proc.stdout)
        self.assertIn("first post-recovery A dispatch: seed=1", proc.stdout)


if __name__ == "__main__":
    unittest.main()
