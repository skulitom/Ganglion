# Status

*A clean checkout should explain its own state. Update this file at every gate.*

## Where we are (2026-09-17)

**Gate C demonstrated; Gate D connectome shadow integration started.** The resident core
captures pixels, detects taught colour components, runs bounded click/notification reflexes,
and serves a timestamped ledger through an MCP stdio bridge. A deterministic reach program now
corrects from measured cursor feedback, completes from observed arrival, and optionally clicks.
Drag reaches and holds a source, moves toward a point or watched destination, releases on arrival
or a taught condition, and verifies that condition from fresh samples after release. Quiet screens
use actual GDI acquisitions alongside DXGI, with acquisition source and timestamps retained.
The same public tools control the synthetic Arena, real pygame window, isolated Edge fixture,
and an explicitly taught Sawayama Solitaire board inside Anode.
The runtime contains no application-specific target rules.

The implementation includes exclusive leases, explicit renewal, independent reflex TTLs,
once-per-observation triggers, snapshot/window bindings, pointer arbitration, separate bounded
action/perception rings, observer capabilities, request deduplication, and an independent input
process that releases clicks and bounded drag holds when the producer stalls or exits.

The actual Haltere ConnectomeRNN receives generic cursor/goal observations during reach and
drag through an optional worker with a one-item mailbox. Its results enter the ledger, and an
intent may now grant it **supervised authority**: a proposal drives the pointer only when its
bounded step brings the cursor closer to the goal, the deterministic controller acts otherwise,
and completion still needs measured arrival. Exceptions and stalls leave the existing controller
running. The recorded sensor inputs can be replayed without a desktop.

The repository is public on [GitHub](https://github.com/skulitom/Ganglion) under MIT, and the
Windows CI workflow has passed remotely. Generic cursor training now runs through the actual
connectome, with headless latency/gain variation, held-out trajectories, deterministic and MLP
comparisons, and a GPU temperature guard. See [training results](docs/bench/TRAINING.md).
The deterministic suite passes **103 tests**. Three browser checks and one trained-CUDA-model
check are optional; the trained-model check was also run separately and passed.
Readout DAgger settled 3/16 static targets on one held-out set and 4/16 on a fresh set.
Two 200-update encoder/readout runs did not beat it on validation. Checkpoints remain local
and experimental; no model has been released to Hugging Face. Training peaked at 46°C.

The suite now passes **116 tests** (four optional checks skipped). With the DAgger cursor
checkpoint, the connectome proposed 90.6% of the pointer commands accepted during 33 live
Solitaire drags, and 73–77% in the synthetic reach/drag demos, all completing under the
envelope. Real-time Solitaire play runs as an evaluation harness: 108 verified steps with the
deterministic controller and 40 with the connectome, no wins. Card reading, rules and planning
live in `ganglion/evaluation/solitaire`, outside the core.

Next: make the model earn its share unsupervised (settling, long-horizon stability), measure and
reduce live inference jitter, and add the fly visual front-end so goals stop coming from taught
colour components. No checkpoint has been released to Hugging Face; the candidate does not yet
control a cursor on its own. See [the connectome experiment](docs/bench/SHADOW.md).

## First-person control in Half-Life

- The core gained bounded key holds, relative look deltas and button holds, motion and
  template-track watches, an `align` program (view to target, then bursts) that the connectome
  can drive under the supervising envelope, a concurrent `move` program the runtime renews at
  100 Hz, key and align/track reflex responses, and a whole-view change sense. 129 tests pass.
- Live in Half-Life inside Anode: the motion → track → connectome align → fire chain engaged
  console-spawned grunts on its own. Session totals: 58 reflex firings, 15 aligned bursts,
  connectome share 80.4% of view commands, first shot 0.1–3.8 s after acquisition, four grunts
  dead with no damage taken. Alarm lights and doors caused false alarms until the motion percept
  became illumination-invariant; colour HUD reflexes misfired under red lighting and were disarmed.
- No level was completed: traversal is still agent-directed, with a bump-and-turn explore
  behaviour as the only autonomy. [Scorecard and limits](docs/bench/HALFLIFE.md).

## Real-time Solitaire with the connectome in the loop

- A live loop reads the Sawayama board from captured frames, picks a move, executes it through
  the public MCP tools as a taught drag or stock click, and verifies by reading again.
  Deterministic runs: 93 drags started, 89 accepted, 14/15 deals, mean 2.32 s per step.
  Connectome runs: 33 drags, 22 accepted (eight lost the grab target before pickup and were
  retried, one refused), 7/7 deals, 3,778 connectome commands against 394 overrides.
- Inference during live drags: p50 3.4 ms, p95 8.3 ms, p99 19.9 ms, 76.6% within 5 ms, mean
  disagreement 1.82 px from the reference. The envelope, not the model, guarantees completion.
- The reader uses per-phase glyph templates taught from five labelled frames; unreadable frames
  stop the run. The game auto-plays aces and twos; higher ranks are assumed safe-rule only.
- Both deals ended in positions proven dead from the visible cards. Winning was not the aim,
  and the card logic is a harness, not a product feature.
- [Scorecard, images, reports and reproduction](docs/bench/SOLITAIRE.md).

## Solitaire transfer validation

- **88 tests pass** with the three optional browser checks; the clean deterministic environment
  passes **85 tests**, with those three skipped. Locked resolution and wheel build pass.
- In Sawayama, the core rejected a nine of clubs dropped on a six of spades (snapback, failed
  condition verification), then accepted it on a ten of hearts (vacant source, persistent
  destination, fresh post-release verification). Completion took about **0.995 seconds** for
  the accepted move and **1.928 seconds** for the expected rejection/verification timeout.
- Both attempts used actual MCP tools, DXGI/GDI capture, and Windows input in **Anode session 3**.
  Each recorded one press and release, no input failures, no pending output, and no ledger gaps.
  Saved before/after images were visually checked. No game rules or recognition were added to the core.
- An opt-in `ganglion.evaluation.transfer` runner validates a taught profile and saves its
  evidence. A subsequent attempt with the now-changed board refused input on failed preconditions.
  These predicates do not identify arbitrary card ranks or prove a full game is solvable.
- The direct executable launch redirected through Steam and replaced the main-session Steam UI
  with a seat client. After testing, the seat game/client were closed and main Steam was restored.
  A documented direct-launch experiment with a temporary app-ID file avoided the redirect but
  failed Steam initialization across sessions. The temporary file was removed. Concurrent
  main-Steam/seat-game operation remains unresolved for this build.
- [Scorecard, images, profile, and launch details](docs/bench/SOLITAIRE.md).

## Gate C drag validation

- The drag milestone passed **81 tests** with installed-Edge fixture checks and **78 tests** in
  its locked environment without torch, Haltere, ViGEm, or Playwright, with three browser checks skipped.
- **40/40 live cases pass** across pygame and Edge, each on the console and inside Anode, plus
  **10/10 synthetic cases** through the actual MCP bridge. Cases cover quiet-screen reach, early
  condition release, accepted drops, rejected drops, and cancellation while held. A rejected drop
  passes the evaluation only when the controller reports failed verification and releases input.
- Every live run ended halted, with no pending output, false actions, input failures, or ledger
  gaps. Old observations were dropped: 2/6 in pygame console/seat and 83/161 in Edge console/seat.
  These short runs establish fixture behavior, not a throughput or reliability guarantee.
- Regressions cover new-pixel capture provenance, slow acquisition, destination movement/loss,
  disappearance conditions, evidence acquired before release, pointer arbitration, lease/intent
  expiry, focus loss, cancellation, failed release, and producer exit. The forced-exit helper
  checks use a fake actuator for both click and drag.
- [Scorecard, retained failures, and reproduction commands](docs/bench/DRAG.md). The initial
  console run exposed sampling gaps, deadline rejection, and verification settling errors; its
  failing trace is retained alongside the corrected results.
- Locked dependency resolution and the wheel build pass; both HTML fixtures are packaged. CI
  includes synthetic drag and has subsequently passed on GitHub.

## Gate C reach validation

- The reach milestone passed **53 tests** with the optional installed-Edge fixture check and
  **52 tests** in its clean environment, with one browser check skipped. Those regressions remain
  in the current suite; the synthetic reach comparison also passes after the shared harness update.
- New regressions cover moving targets, a cursor that ignores submitted moves, target loss,
  stale evidence, independent intent timeout, cancellation barriers, pointer arbitration,
  ownership, request retries, and snapshot/point validation.
- The same reach program scored **4/4 hits with zero false clicks in each of four live runs**:
  Arena and Edge, each on the console and inside Anode. Mean completion was 0.364–0.437 seconds.
  Every run ended halted with no pending output or ledger gaps.
- The periodic baseline uses an explicit 250 ms decision delay / 500 ms minimum cadence. It
  completed 0–2 of four trials per live environment and made 20–24 false clicks. This models a
  delayed agent; it is not a measured LLM comparison or a broad web-navigation claim.
- [Scorecard, traces, commands, and limitations](docs/bench/REACH.md). Raw `reach-*.json` reports
  contain both policies and independent application truth. Browser control uses pixels and
  Windows input; Playwright is confined to fixture setup and evaluator truth.
- The wheel builds and includes its browser fixture. CI also runs synthetic reach and has
  subsequently passed on GitHub.

## Gate B validation

- The original Gate B milestone passed 38 tests, including a clean install without torch,
  Haltere, or ViGEm. Its regression checks remain part of the current suite.
- Regression cases cover expiry during wait, other-client renewal, request retries, competing
  reflexes, repeated/stale frames, changed layouts, ledger gaps/readers, rejected input, and a
  real helper process releasing its fake actuator after its parent calls `os._exit`.
- MCP discovery, structured results, errors, and native JPEG content are tested. Demo runs
  launch the actual stdio bridge as a subprocess, with the agent idle during the scored interval.
- Console and Anode seat demonstrations pass with no scored misses or false actions.
  Retained traces: [console](docs/bench/results/reflex-console.json),
  [seat](docs/bench/results/reflex-seat.json), [synthetic](docs/bench/results/reflex-synthetic.json).
- The seat run scored **18 hits over eight seconds**, no ledger gaps, and a successful halt.
  Capture-available to input-submitted median: **2.52 ms**, observed maximum **14.38 ms** (18 samples).
  Event-scheduled to application-received median: **31.29 ms**, observed maximum **49.65 ms**
  (17 newly appearing targets). These are short smoke measurements, not guaranteed bounds.
- A Windows GitHub Actions workflow runs tests and the synthetic demos; it has subsequently
  passed on GitHub.

Actual presentation/physical scanout is not measured. The Arena records rendering submission;
reports distinguish it from capture availability and application receipt. Targets already visible
before arming are excluded from reaction timing. Generic `effect_observed` means disappearance;
evaluator-only Arena truth verifies hits independently.

## Reproduce

```bash
uv sync --locked --extra dev
.venv/Scripts/python.exe -m pytest -q                       # deterministic tests, no desktop or GPU needed
.venv/Scripts/python.exe -m ganglion.cli demo --synthetic --seconds 5 --json runs/synthetic.json
.venv/Scripts/python.exe -m ganglion.cli demo --seconds 8 --json runs/console-reflex.json
.venv/Scripts/python.exe -m ganglion.cli reach-demo --environment synthetic --trials 4 --json runs/reach.json
.venv/Scripts/python.exe -m ganglion.cli drag-demo --environment synthetic --trials 2 --json runs/drag.json
```

Inside an Anode seat (the process must run in the seat's session, e.g. through `seat_run` or
`seat_exec` from the Anode MCP server, or `anode run`):

```bash
C:\DEV\Ganglion\.venv\Scripts\python.exe -m ganglion.cli demo --seconds 8 --json runs/seat-reflex.json
```

See [runtime setup and MCP contract](docs/RUNTIME.md) and the
[agent guide](skills/ganglion-agent/SKILL.md). The Phase 0 `doctor`, `bench`, and timing flasher
remain available. The brain-step benchmark needs torch and Haltere installed separately.

`GANGLION_HALTERE` points the bench at a Haltere checkout other than `C:\DEV\Haltere` (it needs
`artifacts/ftPath2_best.pt` and `data/built/flight.*` for the brain-step timing; without them the
bench reports the brain as unavailable and everything else still runs).

## Known failures and quirks

- The runtime supports primary-display colour components, bounded clicks/notifications,
  discrete pointer input, and cursor-feedback reach/drag. Semantic recognition, keyboard/gamepad
  programs, and learned controllers remain later work.
- GDI fallback adds acquisition/copy cost during quiet DXGI periods. It does not make cached
  evidence fresh. Acquisition age controls the input budget; slow observations are dropped.
- Solitaire's taught profile is tied to one observed tableau and requires visual comparison
  before reuse. Its successful drag does not establish a general card recognizer or solver.
- A game launched directly in the seat may relaunch through Steam. Check the resulting game
  and client sessions; a surviving `steam.exe` process alone does not prove a usable parent UI.
- Reach uses a speed-limited proportional baseline, not the planned minimum-jerk/PID or neural
  controller. Completion confirms cursor arrival and optional click release, not task success.
- Browser-seat startup required waiting for cross-thread foreground activation in the fixture
  launcher. The general runtime still never changes foreground focus.
- The control loop targets 100 Hz. Detection older than 50 ms cannot initiate input; skipped
  observations and ledger overflow are reported. Ledger rings are bounded and not durable.
- The helper's forced-parent-exit test uses a fake actuator; ordinary click/drag submission and
  release were tested live on console and seat. OS scheduling can delay a watchdog; these are
  not hard real-time guarantees. General keyboard/gamepad holds are not implemented.
- The Arena must acquire focus before testing. The general core never steals target focus and
  rejects mismatched foreground or occlusion. Window/layout changes invalidate old bindings.
- A cold OpenCV initialization caused an early stale click during development. Initialization
  now happens before control is offered; dropped old evidence does not consume an appearance edge.
- Endpoints contain private local capabilities and belong under ignored `runs/`. Request
  deduplication is bounded to 4,096 successful mutations per core instance; then halt/restart.
- An original Phase 0 `console.json` was mentioned in earlier status text but was absent when
  this work began. The retained Phase 0 seat report is `docs/bench/results/seat.json`.
- The flasher window (`ganglion.arena.flasher`) initialises video only; a full `pygame.init()`
  took eight seconds inside a child session (audio device probing), which made the bench's
  ten-second ready timeout marginal.
- Two benches must not run at the same time on one machine: the flasher's UDP ports (9700/9701 on
  loopback) are shared across sessions.
- `doctor` reports `enhance pointer precision on` on this machine: relative mouse moves are
  accelerated on the desktop (games reading raw input are unaffected).
- The seat's frame cap is `DWMFRAMEINTERVAL = 15` until `scripts/set-seat-fps.ps1` has been run as
  administrator and the machine rebooted.

## Provenance

- Connectome census: `docs/census/` (scripts and outputs; male-CNS v1.0 flat release from
  Haltere's `data/raw`, traced-only weights).
- Scratch capture benchmarks that preceded `ganglion bench`: `docs/bench/*.py`.
- Gate B traces and package metadata: `docs/bench/results/reflex-*.json`.
- Reach/periodic comparisons and browser transfer: `docs/bench/results/reach-*.json`.
- Drag/quiet-screen checks and retained diagnostic failure: `docs/bench/DRAG.md` and linked traces.
- Solitaire transfer, saved images, and changed-board refusal: `docs/bench/SOLITAIRE.md`.
- External review that shaped §4.1, §9.5, §11.1 of the plan: `suggestions/`.
