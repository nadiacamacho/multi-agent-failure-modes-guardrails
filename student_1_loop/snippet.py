"""
Coordinator routing + round-cap guardrail (Node 0).

Domain: Financial Trading Bot (task_domain=financial_trading_bot)

Guardrail contract:
  1. Increment state.round_number once per Coordinator pass, before routing.
  2. If round_number >= 5, do NOT call the LLM; route to terminal "partial_output".
  3. Cap is a plain if in code — never a prompt instruction.

NOTE: LangGraph/LangChain are not installed here. This module exposes the same
routing contract a LangGraph Coordinator node would enforce, with an explicit
graph runner that treats "partial_output" as a real terminal node (no edge
back to the Coordinator).
"""

from __future__ import annotations

from typing import Any, Tuple

from contract import AgentState


MAX_ROUNDS = 5
TERMINATING_DECISIONS = {"done", "validate", "partial_output", "END"}

# Same adversarial phrases as Step 1 — never a clean terminating decision
ADVERSARIAL_RESPONSES = [
    "need more analysis on the market signal",
    "unclear — maybe re-run market signal analysis?",
    "continue evaluating order book indicators",
    "not sure if risk/compliance cleared yet — check again",
    "route back to market analysis worker",
]


class AdversarialMockLLM:
    """Deterministic stub: every call returns a non-terminating routing hint."""

    def __init__(self) -> None:
        self.call_count = 0
        self.token_estimate = 0

    def decide_route(self, state: AgentState) -> str:
        self.call_count += 1
        self.token_estimate += 40
        idx = (self.call_count - 1) % len(ADVERSARIAL_RESPONSES)
        response = ADVERSARIAL_RESPONSES[idx]
        print(
            f"  [MOCK LLM] call #{self.call_count} → {response!r} "
            f"(not a terminating decision)"
        )
        return response


def partial_output_node(state: AgentState) -> AgentState:
    """
    Real terminal node: ends the graph. No edge back to Coordinator.
    Hands back whatever analysis_payload / sanitized_tool_calls exist so far
    (including empty). Does not raise; does not block.
    """
    payload = state.analysis_payload if state.analysis_payload is not None else {}
    tools = state.sanitized_tool_calls if state.sanitized_tool_calls is not None else []
    state.analysis_payload = dict(payload)
    state.analysis_payload["termination"] = "partial_output"
    state.analysis_payload["partial"] = True
    state.sanitized_tool_calls = list(tools)
    print(
        "  [PARTIAL_OUTPUT terminal] ending graph — "
        f"notes={len(state.analysis_payload.get('notes', []))} "
        f"tool_calls={len(state.sanitized_tool_calls)} "
        f"error_log={state.error_log!r}"
    )
    return state


def route_to_partial_output(state: AgentState) -> Tuple[AgentState, str]:
    """Mark forced partial termination; caller must invoke partial_output_node next."""
    return state, "partial_output"


def success_node(state: AgentState) -> AgentState:
    """Terminal success path (clean LLM terminate). No edge back to Coordinator."""
    state.analysis_payload = dict(state.analysis_payload or {})
    state.analysis_payload["termination"] = "success"
    print(
        "  [SUCCESS terminal] ending graph — "
        f"round_number={state.round_number} error_log={state.error_log!r}"
    )
    return state


def coordinator_route(
    state: AgentState,
    llm: Any,
    *,
    max_rounds: int = MAX_ROUNDS,
) -> Tuple[AgentState, str]:
    """
    One Coordinator pass with hard round-cap guardrail.

    Order is load-bearing:
      1) bump round_number (always, even if LLM would be ambiguous)
      2) if round_number >= max_rounds → short-circuit, no LLM, → partial_output
      3) otherwise ask LLM and route

    Cap threshold is the single parameter max_rounds (default: MAX_ROUNDS).
    """
    # 1. Deterministic increment — once per Coordinator pass, before routing
    state.round_number += 1
    print(f"  [COORDINATOR] round_number → {state.round_number}")

    # 2. Cap check BEFORE any LLM call — adversarial LLM cannot bypass this
    if state.round_number >= max_rounds:
        state.error_log = (
            f"round_cap_exceeded: forced termination at round {max_rounds}"
        )
        print(
            f"  [GUARDRAIL] round_number={state.round_number} >= {max_rounds} "
            "— skipping LLM, routing to partial_output"
        )
        return route_to_partial_output(state)

    # 3. LLM only runs when under the cap
    decision = llm.decide_route(state)
    normalized = decision.strip().lower()

    if normalized == "done" or normalized.startswith("done"):
        return state, "success"
    if normalized in TERMINATING_DECISIONS:
        return state, "success"
    if "validate" in normalized and "not sure" not in normalized:
        return state, "validator"

    # Ambiguous / non-terminating → re-route upstream (same failure mode as Step 1)
    # Isolated harness uses "worker_market"; main_system maps this to "analyzer".
    return state, "worker_market"


def worker_market(state: AgentState) -> AgentState:
    """Stub worker: appends a partial note; never sets is_validated."""
    print("  [WORKER market] MOCK EXECUTION BLOCKED — no real infra / API calls")
    notes = list(state.analysis_payload.get("notes", []))
    notes.append(f"market_signal_stub_{len(notes) + 1}")
    state.analysis_payload["notes"] = notes
    state.sanitized_tool_calls = list(state.sanitized_tool_calls) + [
        "market_signal_lookup:MOCK EXECUTION BLOCKED"
    ]
    return state


def run_guarded_graph(
    state: AgentState,
    llm: Any,
    *,
    max_rounds: int = MAX_ROUNDS,
    max_steps: int = 50,
    worker_fn=None,
) -> AgentState:
    """
    Explicit graph edges:
      coordinator → worker_market | partial_output | validator | success
      worker_market → coordinator   (on success)
      worker exception → partial_output  (SEPARATE failure path; see comment below)
      partial_output → END   (terminal; no edge back to coordinator)
      success → END
      validator → END

    Worker-exception policy (explicit):
      A worker raise is a SEPARATE failure path — it does NOT consume additional
      round-cap budget and we do NOT re-enter the Coordinator to retry.
      Justification: round_number counts Coordinator decision cycles. The pass
      that dispatched the worker already incremented. Retrying via Coordinator
      would either double-count that cycle or burn remaining cap on infra errors.
      Fail-fast to partial_output preserves the bumped round_number, sets
      error_log, and always reaches a terminal node (no deadlock / hang).
    """
    if worker_fn is None:
        worker_fn = worker_market

    print("=" * 72)
    print(
        f"GUARDED COORDINATOR (round_number += 1, cap at >= {max_rounds} before LLM)"
    )
    print(f"Domain: {state.task_domain}")
    print("=" * 72)

    next_node = "coordinator"
    steps = 0

    while next_node not in ("END",):
        steps += 1
        if steps > max_steps:
            raise RuntimeError(
                f"Graph exceeded max_steps={max_steps} — possible deadlock"
            )

        if next_node == "coordinator":
            print(f"\n--- Coordinator pass (incoming round_number={state.round_number}) ---")
            state, next_node = coordinator_route(
                state, llm, max_rounds=max_rounds
            )
            print(f"  [COORDINATOR] routed → {next_node!r}")

        elif next_node == "worker_market":
            try:
                state = worker_fn(state)
                next_node = "coordinator"
            except Exception as exc:
                # Separate failure path — do not bump round_number again, do not retry
                print(
                    f"  [WORKER market] EXCEPTION: {exc!r} — "
                    "routing to partial_output (no Coordinator re-entry)"
                )
                state.error_log = f"worker_exception: {exc}"
                next_node = "partial_output"

        elif next_node == "partial_output":
            state = partial_output_node(state)
            next_node = "END"  # terminal — no edge back to Coordinator

        elif next_node == "success":
            state = success_node(state)
            next_node = "END"

        elif next_node == "validator":
            print("  [VALIDATOR] reached")
            state.is_validated = True
            state.analysis_payload = dict(state.analysis_payload or {})
            state.analysis_payload["termination"] = "validated"
            next_node = "END"

        else:
            raise RuntimeError(f"Unknown node: {next_node!r}")

    print("\n=== FINAL STATE ===")
    print(f"round_number:     {state.round_number}")
    print(f"error_log:        {state.error_log!r}")
    print(f"is_validated:     {state.is_validated}")
    print(f"analysis_payload: {state.analysis_payload}")
    print(f"sanitized_tool_calls ({len(state.sanitized_tool_calls)}): "
          f"{state.sanitized_tool_calls}")
    print(f"Mock LLM calls:   {llm.call_count}")
    print(f"Mock tokens:      {getattr(llm, 'token_estimate', 'n/a')}")
    print("Graph ended cleanly (no exception, no deadlock).")
    return state


def main() -> None:
    """Step 3 proof: same adversarial LLM as Step 1, guardrail active."""
    initial = AgentState(
        task_domain="financial_trading_bot",
        raw_input="SIGNAL: momentum spike on AAPL — propose market buy 100 shares @ mid",
        analysis_payload={"notes": []},
    )
    llm = AdversarialMockLLM()
    final = run_guarded_graph(initial, llm)

    assert final.error_log is not None, "error_log must be set on forced termination"
    assert final.round_number == MAX_ROUNDS, (
        f"expected round_number=={MAX_ROUNDS}, got {final.round_number}"
    )
    assert llm.call_count == MAX_ROUNDS - 1, (
        f"LLM must not run on the capped pass; expected {MAX_ROUNDS - 1} calls, "
        f"got {llm.call_count}"
    )
    assert final.analysis_payload.get("termination") == "partial_output"
    assert "notes" in final.analysis_payload
    assert "round_cap_exceeded" in (final.error_log or "")
    print("\nASSERTIONS PASSED: cap fired at round 5, LLM skipped, terminal clean.")


if __name__ == "__main__":
    main()
