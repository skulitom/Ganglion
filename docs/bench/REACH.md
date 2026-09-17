# Reach and browser transfer scorecard

Measured 2026-09-17. The same deterministic `reach` primitive and public MCP tools completed
the moving-target task in pygame and Microsoft Edge, on the console and in Anode session 3.
This delivers the reach/transfer portion of Gate C. Drag-until-condition and Solitaire remain.

## Retained measurements

Each row contains four paired trajectory seeds per policy. Success requires an application-recorded
hit. The reach runs additionally require observed cursor completion, no false actions, no ledger
gaps, a completed halt, and no pending output.

| Environment | Periodic successes | Periodic false clicks | Reach successes | Reach false clicks | Mean reach completion |
|---|---:|---:|---:|---:|---:|
| [Arena, console](results/reach-console.json) | 2/4 | 20 | 4/4 | 0 | 0.364 s |
| [Arena, Anode seat](results/reach-seat.json) | 1/4 | 23 | 4/4 | 0 | 0.376 s |
| [Edge, console](results/reach-browser-console.json) | 0/4 | 24 | 4/4 | 0 | 0.419 s |
| [Edge, Anode seat](results/reach-browser-seat.json) | 1/4 | 23 | 4/4 | 0 | 0.437 s |
| [Synthetic integration](results/reach-synthetic.json) | 3/4 | 15 | 4/4 | 0 | 0.335 s |

All five reports have zero ledger gaps and an empty pending-command list after halt. The final
console Arena run skipped two captured frames; the other four skipped none. The runtime processes
the newest frame and reports skips. Each report includes raw events, fixture truth, individual
trial outcomes, parameters, package versions, and measurement time. Edge was 153.0.4234.32;
Playwright was 1.63.0. The seat viewer remained hidden at 1280×720.

## What the comparison means

Both policies use the same taught RGB-component detector and the same trajectory seeds. The
periodic baseline reads a target position, waits **250 ms** to model agent decision time, submits
a discrete click at that earlier point, and waits another 250 ms before observing again. It gets
up to six attempts. These are explicit experimental assumptions, not measured LLM inference
latency. A faster agent, a larger target, or a more stationary target changes the comparison.

Reach receives a three-second timeout, a 1,200 px/s speed limit, six-pixel tolerance, and 30 ms
settling interval. It corrects from actual cursor samples at up to 100 Hz with proportional gain
35/s. The agent starts the intent, waits for its terminal event, and reads the outcome; those three
policy calls replace 9–12 look/input calls on average in these periodic runs. Fixture setup,
lease renewal, cursor reset, and final evaluator reads are excluded from both policy call counts.

Only the evaluator resets trial state and reads ground truth. The policy obtains detection and
state through the real MCP stdio bridge. Playwright launches an isolated browser and collects
the browser's pointer-event log; it supplies no coordinates or DOM state to the controller and
does not perform the live scored input. Browser input receipts were trusted native events.
Application success is separate from `intent_completed`, which confirms cursor arrival and
optional click release rather than arbitrary task success.

Completion times include transport and completion observation. They are not capture-to-input
latency, physical scanout, or steady-state tracking lag. Synthetic timing excludes Windows input
and DXGI. Four seeds per environment are a smoke test, not a statistical generalization result.
The browser task is a controlled moving-button fixture, not an unfamiliar website or navigation
benchmark. No learned model or connectome controller participated.

## Reproduce

```powershell
uv sync --locked --extra dev
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ganglion.cli reach-demo --environment synthetic --trials 4 --json runs/reach-synthetic.json
.venv/Scripts/python.exe -m ganglion.cli reach-demo --environment arena --trials 4 --json runs/reach-arena.json
```

The `arena` command opens and operates its own window in the current session. Use Anode
`seat_exec` or `seat_run` to execute that same command in the seat. The harness restores the
prior cursor and foreground window after closing its fixtures.

With an existing Microsoft Edge installation:

```powershell
uv sync --locked --extra dev --extra browser
.venv/Scripts/python.exe -m ganglion.cli reach-demo --environment browser --trials 4 --json runs/reach-browser.json
$env:GANGLION_BROWSER_TESTS='1'
.venv/Scripts/python.exe -m pytest -q tests/test_browser_fixture.py
```

The optional test runs headless and checks the fixture's hit/miss accounting and visible
completion. The live comparison must be headed to exercise DXGI and SendInput. The `browser`
extra does not install or replace Edge. Its launch uses Playwright's documented
[installed-browser channel](https://playwright.dev/python/docs/browsers#google-chrome--microsoft-edge).

## Remaining boundary

- These retained reach runs predate static refresh. The subsequent [drag milestone](DRAG.md)
  adds fresh GDI acquisitions during quiet DXGI periods while preserving the 50 ms input deadline
  and 250 ms stale-observation failure. Cached pixels are not stamped fresh.
- Color-component selection is not semantic recognition or persistent object identity.
- Cancellation is acknowledged by an output barrier. One previously queued command can execute
  before it; another pointer program is rejected until the old output drains.
- This scorecard does not establish drag, keyboard holds, learned control, sub-tick tracking, or
  Solitaire gameplay. Subsequent bounded drag results are in [DRAG.md](DRAG.md); see
  [the runtime contract](../RUNTIME.md) for current inputs and outcomes.

The first browser-seat startup failed before input because a cross-thread foreground request
had not completed when checked. The fixture launcher now waits for activation and temporarily
attaches the relevant input threads when needed, then detaches them. The general runtime still
never activates its target. The completed seat report above is from the corrected launcher.
