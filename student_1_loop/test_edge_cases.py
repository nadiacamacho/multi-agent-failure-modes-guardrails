"""
Step 4 — Edge-case tests for the Coordinator round-cap guardrail.

Run:  python test_edge_cases.py
"""

from __future__ import annotations

from snippet import (
    MAX_ROUNDS,
    AdversarialMockLLM,
    AgentState,
    coordinator_route,
    run_guarded_graph,
)


# ---------------------------------------------------------------------------
# Test 1 helpers
# ---------------------------------------------------------------------------
class CooperativeMockLLM:
    """Returns a clean terminating decision on the first (and only) call."""

    def __init__(self) -> None:
        self.call_count = 0
        self.token_estimate = 0

    def decide_route(self, state: AgentState) -> str:
        self.call_count += 1
        self.token_estimate += 20
        print(f"  [COOPERATIVE LLM] call #{self.call_count} → 'done'")
        return "done"


def _fresh_state(**kwargs) -> AgentState:
    base = dict(
        task_domain="financial_trading_bot",
        raw_input="SIGNAL: momentum spike on AAPL — propose market buy 100 shares @ mid",
        analysis_payload={"notes": []},
    )
    base.update(kwargs)
    return AgentState(**base)


# ---------------------------------------------------------------------------
# Test 1 — Early termination is not dragged to round 5
# ---------------------------------------------------------------------------
def test_1_early_termination() -> None:
    print("\n" + "#" * 72)
    print("# TEST 1 — Early termination (cooperative LLM on round 1)")
    print("#" * 72)

    llm = CooperativeMockLLM()
    final = run_guarded_graph(_fresh_state(), llm)

    assert final.round_number == 1, f"expected round_number==1, got {final.round_number}"
    assert final.error_log is None, f"error_log must stay None, got {final.error_log!r}"
    assert final.analysis_payload.get("termination") == "success", (
        f"expected success terminal, got {final.analysis_payload}"
    )
    assert final.analysis_payload.get("termination") != "partial_output"
    assert llm.call_count == 1

    print("\nTEST 1 RESULT: PASS")
    print(
        f"  round_number={final.round_number} termination="
        f"{final.analysis_payload.get('termination')!r} error_log={final.error_log!r}"
    )


# ---------------------------------------------------------------------------
# Test 2 — Exact boundary both directions (>= not >)
# ---------------------------------------------------------------------------
def test_2_exact_boundary() -> None:
    print("\n" + "#" * 72)
    print("# TEST 2 — Exact boundary: incoming 3→4 calls LLM; 4→5 skips LLM")
    print("#" * 72)

    llm = AdversarialMockLLM()

    # Direction A: incoming round_number == 3 → bump to 4 → LLM MUST be called
    state_a = _fresh_state(round_number=3)
    calls_before_a = llm.call_count
    state_a, route_a = coordinator_route(state_a, llm, max_rounds=MAX_ROUNDS)
    print(
        f"  A: incoming=3 → round={state_a.round_number} route={route_a!r} "
        f"llm_calls_delta={llm.call_count - calls_before_a}"
    )
    assert state_a.round_number == 4
    assert llm.call_count - calls_before_a == 1, "LLM must be called when bumping to 4"
    assert route_a == "worker_market"
    assert state_a.error_log is None

    # Direction B: incoming round_number == 4 → bump to 5 → LLM MUST NOT be called
    state_b = _fresh_state(round_number=4)
    calls_before_b = llm.call_count
    state_b, route_b = coordinator_route(state_b, llm, max_rounds=MAX_ROUNDS)
    print(
        f"  B: incoming=4 → round={state_b.round_number} route={route_b!r} "
        f"llm_calls_delta={llm.call_count - calls_before_b}"
    )
    assert state_b.round_number == 5
    assert llm.call_count - calls_before_b == 0, (
        "LLM must NOT be called when bumping to 5 (cap is >=, not >)"
    )
    assert route_b == "partial_output"
    assert state_b.error_log is not None
    assert "round_cap_exceeded" in state_b.error_log

    # If someone "simplifies" to `> 5`, bump-to-5 would still call the LLM
    # and this assertion above would fail loudly.
    print("\nTEST 2 RESULT: PASS")
    print("  Locked: round 4 calls LLM; round 5 skips LLM → partial_output (>= not >)")


# ---------------------------------------------------------------------------
# Test 3 — Worker throws mid-cycle
# ---------------------------------------------------------------------------
def test_3_worker_throws() -> None:
    print("\n" + "#" * 72)
    print("# TEST 3 — Worker throws after Coordinator already bumped round_number")
    print("#" * 72)
    print(
        "POLICY: worker exception = SEPARATE failure path (not extra round-cap\n"
        "        burn). Fail-fast to partial_output; no Coordinator re-entry\n"
        "        (avoids double-increment / hang)."
    )

    class OnceThenThrowWorker:
        """Raises on first market-analysis invocation."""

        def __init__(self) -> None:
            self.invocations = 0

        def __call__(self, state: AgentState) -> AgentState:
            self.invocations += 1
            print(
                f"  [THROWING WORKER] invocation #{self.invocations} — "
                "raising RuntimeError (MOCK, no real infra)"
            )
            raise RuntimeError("simulated market analysis backend failure")

    # Adversarial LLM that sends us to worker on first decision
    llm = AdversarialMockLLM()
    worker = OnceThenThrowWorker()
    pre_round = 0
    state = _fresh_state(round_number=pre_round)

    final = run_guarded_graph(state, llm, worker_fn=worker)

    # Coordinator bumped once (0→1) before dispatching worker; must not stick at 0
    assert final.round_number == 1, (
        f"round_number should be 1 after the dispatching pass, got {final.round_number}"
    )
    # Must not double-increment (would be 2 if we re-entered Coordinator to retry)
    assert final.round_number != 2, "must not double-increment via retry path"
    assert worker.invocations == 1, "worker must run exactly once (no silent retry loop)"
    assert llm.call_count == 1, "only one Coordinator LLM call before the crash path"
    assert final.error_log is not None
    assert "worker_exception" in final.error_log
    assert final.analysis_payload.get("termination") == "partial_output"

    print("\nTEST 3 RESULT: PASS")
    print(
        f"  round_number={final.round_number} (bumped once, not stuck at 0, not doubled) "
        f"worker_invocations={worker.invocations} "
        f"error_log={final.error_log!r} "
        f"termination={final.analysis_payload.get('termination')!r}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Cap is parameterized (MAX_ROUNDS / max_rounds), not magic 5s
# ---------------------------------------------------------------------------
def test_4_parameterized_cap() -> None:
    print("\n" + "#" * 72)
    print("# TEST 4 — Cap parameterized: max_rounds=2 under adversarial LLM")
    print("#" * 72)

    assert MAX_ROUNDS == 5, "default named constant should still be 5"
    print(f"  MAX_ROUNDS named constant = {MAX_ROUNDS}")

    custom_cap = 2
    llm = AdversarialMockLLM()
    final = run_guarded_graph(_fresh_state(), llm, max_rounds=custom_cap)

    assert final.round_number == custom_cap, (
        f"expected round_number=={custom_cap}, got {final.round_number}"
    )
    assert llm.call_count == custom_cap - 1, (
        f"expected {custom_cap - 1} LLM calls, got {llm.call_count}"
    )
    assert final.error_log is not None
    assert f"round {custom_cap}" in final.error_log, (
        f"error_log should mention the parameterized cap, got {final.error_log!r}"
    )
    assert final.analysis_payload.get("termination") == "partial_output"
    # Must NOT have run all the way to the default of 5
    assert final.round_number != MAX_ROUNDS or custom_cap == MAX_ROUNDS

    print("\nTEST 4 RESULT: PASS")
    print(
        f"  custom max_rounds={custom_cap} → final.round_number={final.round_number}, "
        f"llm_calls={llm.call_count}, error_log={final.error_log!r}"
    )


def main() -> None:
    test_1_early_termination()
    test_2_exact_boundary()
    test_3_worker_throws()
    test_4_parameterized_cap()
    print("\n" + "=" * 72)
    print("ALL STEP 4 EDGE-CASE TESTS PASSED")
    print("=" * 72)


if __name__ == "__main__":
    main()
