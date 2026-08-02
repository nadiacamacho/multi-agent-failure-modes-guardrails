# Coordinator round cap (Student 1)

I built Node 0, the Coordinator, for the Financial Trading Bot (`task_domain=financial_trading_bot`). Without a hard stop in the node itself, routing was driven only by LLM text. An adversarial mock that never returns a clean terminate can keep bouncing Coordinator to Worker to Coordinator forever. LangGraph’s default recursion limit is 25 graph steps, so the guardrail needs to fire well before that external limit becomes the only thing that ends the run.

The fix is simple and code based. Each Coordinator pass increments `round_number` first. If `round_number >= MAX_ROUNDS` (default **5**), we skip the LLM and route to a real terminal `partial_output` node, set `error_log`, and keep whatever partial state already exists. Mock token cost is a fixed **40 tokens per `decide_route` call**, so **100** harness calls cost **4000** and **4** post guardrail calls cost **160**. That is a deterministic stub, not a live API bill.

## Before / after numbers

| Metric | Before (no guardrail) | After (guardrail) |
|--------|----------------------|-------------------|
| Loop / Coordinator passes | No in Coordinator stop; demo harness aborted at **100** passes (that **100** is a harness limit, not a measured infinity) | Hard cap at **`MAX_ROUNDS = 5`** (`>= 5`); removes the unbounded loop risk |
| `round_number` at end | **0** (never incremented) | **5** (adversarial) / **1** (cooperative early exit) |
| Mock LLM calls | **100** (at harness abort) | **4** (adversarial, default cap) |
| Mock token estimate | **4000** | **160** |
| Termination | External harness abort only | Terminal `partial_output` (cap) or `success` (clean `done`) |
| `error_log` on forced stop | `None` | `round_cap_exceeded: forced termination at round 5` |
| Partial payload preserved | N/A (run aborted externally) | **4** market signal notes + **4** sanitized tool stubs |
| Parameterized cap (`max_rounds=2`) | N/A | Fires at round **2**, **1** LLM call, **40** mock tokens |

## Edge cases

| Test | Result |
|------|--------|
| Early terminate (cooperative `done`) | Ends at round **1**, `termination=success`, `error_log=None` |
| Boundary `>=` vs `>` | Incoming 3 to 4: LLM called; incoming 4 to 5: LLM skipped, then `partial_output` |
| Worker throw mid cycle | `round_number=1` (not stuck at 0, not doubled); separate failure to `partial_output` |
| `max_rounds=2` | Cap fires at **2**; not hardcoded to literal `5` in the check path |

## Interview story

In our Financial Trading Bot stack (Market Analysis, Trade Execution, Risk/Compliance, Audit Logging), the Coordinator used to route only on LLM text. With an adversarial mock that never returned a clean terminate, nothing inside the Coordinator stopped the Coordinator to Worker to Coordinator cycle. I showed that risk by running until a safety harness cut off at **100** passes (**100** LLM calls, **4000** mock tokens at **40** tokens per call). That **100** is our abort limit, not a proof of infinity. The real failure is the missing in graph stop.

The guardrail removes that unbounded loop risk. Each Coordinator pass increments `round_number` first, then if `round_number >= MAX_ROUNDS` (default **5**) we skip the LLM and route to terminal `partial_output` with `error_log` set and whatever partial state exists. On the same adversarial LLM afterward, the run stops at round **5** with **4** LLM calls, **160** mock tokens, four usable market signal stubs, and a clean exit. Cooperative `done` ends at round **1**. The boundary is `>=`, not `>`. Worker exceptions fail fast without double counting rounds. `max_rounds=2` fires correctly when parameterized.
