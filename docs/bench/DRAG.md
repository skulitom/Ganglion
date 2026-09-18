# Quiet-screen capture and bounded drag

Measured 2026-09-17 through the public MCP tools. One deterministic controller transfers between
pygame and an isolated Edge fixture, on the console (session 1) and seat (session 3).
There are no application rules in the core. No Solitaire gameplay or learned controller is
claimed by these results.

## Results

Each environment runs two seeds of five cases. Numbers are correctly handled cases, including
the expected failure of rejected drops and the expected cancellation of held input.

| Environment | Quiet reach | Early release | Accepted drop | Rejected drop | Cancel held | DXGI / GDI samples | Dropped observations |
|---|---:|---:|---:|---:|---:|---:|---:|
| [pygame console](results/drag-console.json) | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 872 / 255 | 2 |
| [pygame seat](results/drag-seat.json) | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 1061 / 294 | 6 |
| [Edge console](results/drag-browser-console.json) | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 942 / 289 | 83 |
| [Edge seat](results/drag-browser-seat.json) | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | 1010 / 336 | 161 |
| [Synthetic](results/drag-synthetic.json) | 2/2 | 2/2 | 2/2 | 2/2 | 2/2 | Provided frames | 0 |

All runs ended halted, with no pending commands, false actions, input failures, or ledger gaps.
Every trial recorded exactly one button-down and one button-up at the application. Capture
counts are processed observations over the whole run, not an independent frame-rate estimate.
Dropped observations remain in the scorecard; browser runs incurred more of them. This is a
short functional check, not a latency benchmark or a statistical reliability estimate.

The live runs used Python 3.13.2, MCP 2.2.0, NumPy 2.5.3, OpenCV 5.0.0.93, pygame 2.6.1,
DXCAM 0.3.0, and Playwright 1.63.0; Edge was 153.0.4234.32. Each JSON includes package metadata,
full events, independent fixture truth, per-case outcomes, and acquisition provenance.

## What each case establishes

1. **Quiet-screen reach:** the fixture stays unchanged for 500 ms before starting. There is no
   animation or redraw heartbeat. The core must still reach, settle, click, and release.
2. **Early condition release:** a slider indicator changes before the destination. The actual
   release must precede the endpoint by at least 50 pixels, and the condition must persist after
   release. This exercises `until`, not just drag-to-point.
3. **Accepted drop:** the indicator changes only after dropping inside the destination. The
   core must reach and release before fresh evidence can verify completion.
4. **Rejected drop:** the indicator appears while held, then disappears when the object snaps
   back. The correct result is `condition_not_observed`, not success from the pre-release image.
5. **Cancel while held:** cancel after `drag_started`, await `output_halted`, and confirm release
   with no pending output. The fixture records a rejected drop outside the destination.

Setup and truth telemetry belong to the evaluator. The policy teaches RGB components, obtains
destination coordinates from the public watch result, and calls `ganglion_intent`, `wait`,
`cancel`, and `look`. Playwright only launches/resets the browser fixture and retrieves its
event records. Actual browser control uses captured pixels and Windows input.

## Capture and hold bounds

DXGI's unchanged-desktop behavior is documented by Microsoft's
[AcquireNextFrame contract](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_2/nf-dxgi1_2-idxgioutputduplication-acquirenextframe).
Rather than treating absent DXGI updates as fresh evidence, the live runtime makes a new GDI
acquisition during quiet periods, with a target 20 ms interval. A reusable bitmap is flushed and
copied before publication. Acquisition start and availability are separate timestamps; the
50 ms input deadline starts at acquisition. Raw capture benchmarks remain DXGI-only.

The controller reserves 10 ms of that budget for transport and validation. A helper-confirmed
expiry before a new button action permits replanning from a new observation. It does not permit
replaying uncertain actions. The input helper independently releases a drag without timely
renewals: a renewal cannot extend beyond lease/intent expiry, 100 ms after its evidence, or
100 ms after receipt. Target/focus checks continue while held. OS scheduling can delay these
checks; they are not hard real-time guarantees.

Post-release verification requires distinct fresh samples acquired after release, matching for
the requested settling interval. A sample gap over 100 ms resets stability. An output release
failure retains ownership, retries release, and blocks new leases until core restart.

## Retained development failure

The [initial pygame console run](results/diagnostics/drag-console-initial.json) passed only 6/10
cases. Two accepted drops failed verification because scheduler polls reset the settling timer
between samples. Two other cases stopped after commands expired near the helper's admission
deadline. No input remained held.

The fixes schedule refresh relative to acquisition start, reserve command transport time,
replan only explicitly rejected pre-submission commands, and measure condition stability across
distinct samples. The input expiry bound was preserved. The corrected live runs are listed above.

## Reproduce

```powershell
uv sync --locked --extra dev
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ganglion.cli drag-demo --environment synthetic --trials 2 --json runs/drag-synthetic.json
.venv/Scripts/python.exe -m ganglion.cli drag-demo --environment arena --trials 2 --json runs/drag-arena.json
```

Execute the `arena` command through the seat's `seat_exec` or `seat_run` to check the seat. The harness
closes only its own fixtures and restores the prior foreground window and cursor.

For an existing Edge installation:

```powershell
uv sync --locked --extra dev --extra browser
.venv/Scripts/python.exe -m ganglion.cli drag-demo --environment browser --trials 2 --json runs/drag-browser.json
$env:GANGLION_BROWSER_TESTS='1'
.venv/Scripts/python.exe -m pytest -q
```

The full suite passes 81 tests with optional browser checks enabled; the clean deterministic
environment passes 78 with three browser checks skipped. Fake-clock tests cover moving/lost
destinations, disappearance predicates, stale/pre-release evidence, independent hold expiry,
release failures, and arbitration. A real helper process with a fake actuator verifies release
after its producer calls `os._exit`, for both click and drag. CI includes a synthetic drag run.

## Remaining boundary

The subsequent [Solitaire transfer](SOLITAIRE.md) demonstrates one taught legal move and one
rejected drop. The current detector selects the largest matching
colour component; it provides neither card recognition nor persistent object identity.
`condition_verified` proves only the taught predicate, while `task_success_verified` remains
false. More complex application success needs an independently taught or evaluated condition.
The core still supports only the primary display and left-button pointer programs. Keyboard,
gamepad, learned perception, and connectome controllers remain later work.
