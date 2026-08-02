"""
Step 1 — Reproduce the unbounded Coordinator loop FAILURE.

Domain: Financial Trading Bot (task_domain="financial_trading_bot")

This script models a minimal Coordinator → Worker → Coordinator cycle where
routing depends ONLY on LLM output. There is NO round_number increment and
NO hard cap. The mock LLM never returns a clean terminating decision, so the
graph loops until a safety abort (so the process does not hang forever).

NOTE: LangGraph/LangChain are not installed in this environment. Rather than
guess StateGraph method signatures, this uses an explicit deterministic loop
that implements the same routing contract the Coordinator would enforce inside
a LangGraph node. No live API or infra calls are made.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Frozen team contract (mirrored locally — do not diverge without flagging)
# ---------------------------------------------------------------------------
class AgentState(BaseModel):
    task_domain: str
    raw_input: str
    round_number: int = 0
    is_validated: bool = False
    error_log: Optional[str] = None
    analysis_payload: Dict[str, Any] = Field(default_factory=dict)
    sanitized_tool_calls: list = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Adversarial mock LLM — never emits a clean terminating decision
# ---------------------------------------------------------------------------
TERMINATING_DECISIONS = {"done", "validate", "partial_output", "END"}

# Ambiguous / non-terminating phrases the Coordinator treats as "keep working"
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
        self.token_estimate = 0  # rough mock: ~40 tokens per call

    def decide_route(self, state: AgentState) -> str:
        self.call_count += 1
        self.token_estimate += 40
        # Always ambiguous — never in TERMINATING_DECISIONS
        idx = (self.call_count - 1) % len(ADVERSARIAL_RESPONSES)
        response = ADVERSARIAL_RESPONSES[idx]
        print(
            f"  [MOCK LLM] call #{self.call_count} → {response!r} "
            f"(not a terminating decision)"
        )
        return response


# ---------------------------------------------------------------------------
# Broken Coordinator — LLM-only routing, no round counter, no cap
# ---------------------------------------------------------------------------
def coordinator_route_broken(state: AgentState, llm: AdversarialMockLLM) -> str:
    """
    Routes solely from LLM text. No round_number bump. No >= 5 guardrail.
    Any non-terminating LLM string sends work back to an upstream worker.
    """
    decision = llm.decide_route(state)
    normalized = decision.strip().lower()

    if normalized in TERMINATING_DECISIONS or normalized.startswith("done"):
        return "END"
    if "validate" in normalized and "not sure" not in normalized:
        return "validator"

    # Default: re-route upstream — THIS IS THE FAILURE MODE
    return "worker_market"


def worker_market(state: AgentState) -> AgentState:
    """Stub worker: appends a partial note; never sets is_validated."""
    print("  [WORKER market] MOCK EXECUTION BLOCKED — no real infra / API calls")
    notes = state.analysis_payload.get("notes", [])
    notes = list(notes) + [f"market_signal_stub_{len(notes) + 1}"]
    state.analysis_payload["notes"] = notes
    state.sanitized_tool_calls = list(state.sanitized_tool_calls) + [
        {"tool": "market_signal_lookup", "status": "MOCK EXECUTION BLOCKED"}
    ]
    return state


# ---------------------------------------------------------------------------
# Graph runner (uncapped) — safety abort only so the demo can finish
# ---------------------------------------------------------------------------
# Safety abort is NOT a guardrail inside the Coordinator. It is a harness
# limit so this script exits. The Coordinator itself would keep looping.
SAFETY_ABORT_AFTER = 100


def run_uncapped_graph(state: AgentState, llm: AdversarialMockLLM) -> None:
    print("=" * 72)
    print("STEP 1: UNBOUNDED COORDINATOR LOOP (no round_number, no cap)")
    print(f"Domain: {state.task_domain}")
    print(f"Safety harness abort after {SAFETY_ABORT_AFTER} Coordinator passes")
    print("=" * 72)

    coordinator_passes = 0
    next_node = "coordinator"

    while True:
        if next_node == "coordinator":
            coordinator_passes += 1
            print(
                f"\n--- Coordinator pass #{coordinator_passes} "
                f"(state.round_number stays {state.round_number}) ---"
            )

            if coordinator_passes > SAFETY_ABORT_AFTER:
                print("\n" + "!" * 72)
                print(
                    f"SAFETY HARNESS ABORT after {SAFETY_ABORT_AFTER} Coordinator "
                    f"passes (and still no terminating LLM decision)."
                )
                print(
                    "Without this harness abort, the Coordinator would continue "
                    "indefinitely — round_number was never incremented and no "
                    "hard cap exists in coordinator_route_broken()."
                )
                print("!" * 72)
                print("\n=== FAILURE METRICS ===")
                print(f"Coordinator passes before abort: {coordinator_passes - 1}")
                print(f"Mock LLM calls:                  {llm.call_count}")
                print(f"Mock token estimate:             {llm.token_estimate}")
                print(f"state.round_number (unused):     {state.round_number}")
                print(f"state.is_validated:              {state.is_validated}")
                print(f"Partial notes collected:         {len(state.analysis_payload.get('notes', []))}")
                print(
                    "Verdict: LOOP IS UNBOUNDED under adversarial LLM output "
                    f"(would exceed {SAFETY_ABORT_AFTER} without harness)."
                )
                return

            next_node = coordinator_route_broken(state, llm)
            print(f"  [COORDINATOR] routed → {next_node!r}")

        elif next_node == "worker_market":
            state = worker_market(state)
            next_node = "coordinator"  # always returns control to Coordinator

        elif next_node in ("END", "validator", "partial_output"):
            print(f"\nReached terminal node {next_node!r} — unexpected for this failure demo.")
            return

        else:
            raise RuntimeError(f"Unknown node: {next_node}")


def main() -> None:
    initial = AgentState(
        task_domain="financial_trading_bot",
        raw_input="SIGNAL: momentum spike on AAPL — propose market buy 100 shares @ mid",
        analysis_payload={"notes": []},
    )
    llm = AdversarialMockLLM()
    run_uncapped_graph(initial, llm)


if __name__ == "__main__":
    main()
