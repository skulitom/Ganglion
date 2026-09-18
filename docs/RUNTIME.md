# Using the deterministic runtime

Ganglion now runs one complete reactive path: an agent teaches a colour component, arms a
bounded click or notification, or starts a cursor-feedback reach/drag, and later reads what happened.
Vision processes fresh frames; the control/lease loop targets 100 Hz. Neural models,
keyboard/gamepad control, and further motor programs are later work.

## Install and reproduce

For a clean environment, from the repository root:

```powershell
uv sync --locked --extra dev
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m ganglion.cli demo --synthetic --seconds 5 --json runs/synthetic.json
```

This needs no Haltere data, torch, CUDA, ViGEm, or live desktop input. The synthetic demo still
launches the real MCP stdio bridge and talks to the resident core over its normal protocol.
Windows CI runs this path; live capture/input checks are opt-in commands.

On Windows, this opens and operates a temporary Arena window in the current session:

```powershell
.venv/Scripts/python.exe -m ganglion.cli demo --seconds 8 --json runs/console-reflex.json
```

Run that same command inside an Anode seat for a separate desktop. The demo verifies its Arena
has focus, scopes each click to that window, then closes its fixtures and restores the prior
cursor and foreground window. It does not launch a game or modify system configuration.

The JSON contains the full core ledger, evaluator-only Arena truth, scores, package versions,
and monotonic stage timestamps. `capture_to_submit_ms` begins when pixels are available;
`scheduled_to_receive_ms` also includes capture/display and application processing delays.
Targets already on screen before arming are excluded from the latter metric.

`render_submitted` is the timestamp after pygame's flip call, **not a measured presentation or
physical scanout timestamp**. Short demo percentiles are smoke measurements, not latency guarantees.
Misses and false actions are retained; a passing smoke run requires at least two hits, no misses
or false actions during the armed evaluation interval, no ledger loss, and a completed halt.

## Start a resident core

Headless exploration:

```powershell
.venv/Scripts/python.exe -m ganglion.cli core --synthetic --endpoint runs/core.endpoint.json
```

Live use requires the numeric HWND of the intended window on the primary display. Launch the
core inside the application's Windows session:

```powershell
.venv/Scripts/python.exe -m ganglion.cli core --window 123456 --endpoint runs/core.endpoint.json
```

`--pid 12345` or `--title "Solitaire"` selects the process or title instead; the core refuses
to guess when several visible top-level windows match. `--log PATH` appends stdout/stderr to a
file for windowless launches (for example `pythonw.exe` through a seat launcher).

The core does not activate arbitrary target applications. Focus the intended target before
claiming control. Window identity, client bounds, foreground, and click-point occlusion are
rechecked. Layout/focus/capture failures halt control; rebind after recovery or restart the core
if its worker stopped. The HWND above is an example value.

Endpoint files contain local control capabilities. Keep them under ignored `runs/` and pass a
file path to the bridge; never commit their contents. A core refuses to overwrite an existing
endpoint. Remove a stale file only after confirming its process has stopped. On normal shutdown
the core removes its own endpoint. A new core gets new capabilities and a new ledger epoch.

The endpoint contains separate controller and observer tokens. An observer configuration can
contain only `version`, `host`, `port`, `runtime_id`, and `observer_token`. The transport binds only
to 127.0.0.1. Capabilities coordinate trusted local processes; the Windows session is not a
security sandbox against software running as the same user.

## Connect an MCP client

Launch the bridge with:

```powershell
.venv/Scripts/python.exe -m ganglion.mcp --endpoint C:/DEV/Ganglion/runs/core.endpoint.json
```

Add `--observer` for read-only inspection. Each bridge receives its own client identity; use
`--client-id` to resume the same identity after a bridge reconnect. Starting the bridge does not
create a core, start capture, claim a lease, or inject input. The stdio transport uses the
installed MCP SDK v2 and keeps tool results separate from diagnostic output.

| Tool | Purpose |
|---|---|
| `ganglion_status` | State, cause, lease expiry, watches, reflexes, frame age, dropped observations |
| `ganglion_look` | Paginated ledger plus snapshot metadata and optional native JPEG image |
| `ganglion_claim` | Acquire an exclusive 0.1–60 second control lease |
| `ganglion_renew` | Explicitly renew the controlling client's lease |
| `ganglion_watch` | Teach a connected colour component in a snapshot-bound region |
| `ganglion_arm` | Arm a bounded click or notification on appearance/presence |
| `ganglion_intent` | Reach/click a watched target, or drag and verify a taught condition after release |
| `ganglion_cancel` | Cancel the current program; wait for queued output to drain |
| `ganglion_input` | One agent-chosen absolute move or bounded left click |
| `ganglion_wait` | Wait for events, a gap, shutdown, or impending lease expiry |
| `ganglion_disarm` | Remove a reflex |
| `ganglion_unwatch` | Remove a watch and its reflexes |
| `ganglion_halt` | Stop intents/reflexes, clear watches, end the lease, and request input release |

Read-only calls never renew leases. A second client cannot claim, renew, or change the active
owner's watches. Any holder of a controller capability can halt; observer capabilities cannot.
Each reflex has its own TTL that lease renewal does not extend.
Each intent also has an independent timeout; renewal never extends that timeout.

### Example sequence

1. Call `ganglion_look`. Save `snapshot.id`, `snapshot.target.rect`, and `next_cursor`.
2. Call `ganglion_claim` with `seconds: 30`.
3. Call `ganglion_watch` with a `spec` object:

```json
{
  "name": "green target",
  "snapshot_id": "COPY snapshot.id FROM look",
  "region": [200, 180, 640, 360],
  "color_rgb": [40, 220, 120],
  "tolerance": 15,
  "min_pixels": 100
}
```

4. Call `ganglion_arm` with `spec: {watch_id: "...", response: "click",
   trigger: "appear", cooldown_ms: 250, max_fires: 20, hold_ms: 20, ttl_seconds: 30}`.
5. Call `ganglion_wait` with the saved cursor. Keep the returned cursor, inspect outcomes, and
   renew deliberately when `reason` is `lease_expiring`.
6. Call `ganglion_halt` when finished.

Regions and detector outputs use **physical screen pixels**, with `[x, y, width, height]`
rectangles. The optional image defaults to JPEG at at most 640 pixels wide. `ganglion_look`
also accepts `max_width` (64–1920) and `image_format` (`jpeg` or `png`). PNG preserves small
visual details. Images above two million pixels or one MB encoded return `image_too_large`;
request a smaller width or JPEG. Use its
`image_transform.image_to_screen` scales before teaching from image coordinates.
The runtime supports the primary display only. A watch must fit within the selected window's client
area; moving/resizing/replacing that window invalidates its old bindings.

### Optional connectome shadow

`core --shadow-checkpoint PATH` loads the actual Haltere ConnectomeRNN on CUDA before exposing
the service. The ordinary deterministic install does not import torch or require Haltere.
`reach-demo --shadow-checkpoint PATH` measures the model alongside the existing controller.
`--shadow-process` on `core` and `reach-demo` runs the model in its own process behind the same
one-item mailbox, which is what keeps live inference under the evidence budget.

The model receives each eligible reach/drag correction's cursor, goal, rectangle, speed,
observation timestamp and reference point. A separate worker retains at most one pending
sample; replacing it increments a visible counter. The worker never receives an actuator.
Results older than 50 ms, past expiry, or invalidated by halt/cancellation are discarded.
An inference exception disables shadow inference without changing input ownership.

`status.shadow` reports model identity, checkpoint hash, counters and errors. The ledger records
`shadow_prediction`, `shadow_discarded` and `shadow_failed`; predictions include complete sensor
inputs, raw neural outputs, inference time and disagreement with the reference controller.
One fixed 10 ms neural step runs per consumed sample, with state reset at intent/stage/layout
boundaries and gaps above 50 ms. This does not yet align neural time with variable wall time.

The current adapter maps target error and cursor velocity to existing sensory populations.
Unimplemented visual-flow and other channels are zero. Interpreting the flight readout's first
two axes as hypothetical cursor velocity is an untrained transfer experiment. Shadow results
have `actuation_authority: false`; neither a successful Arena trial nor a completed motor
program establishes neural control or a game win. See [measurements](bench/SHADOW.md).

An intent may set `controller: "connectome"` on a core started with a checkpoint. Each control
tick then applies the newest fresh proposal (at most 50 ms old, same intent, stage and layout)
only when its step, clamped to the speed limit and client area, brings the cursor closer to the
goal or holds position within tolerance; otherwise the deterministic step is used and counted as
an override. `pointer_feedback` records `controller` per command; the intent reports
`neural_commands`, `overridden_commands` and `actuation_authority: "supervised_connectome"`.
Completion still needs measured arrival. Without a loaded model the request fails with
`controller_unavailable`.

`appear` fires only on absent-to-present evidence after arming. `present` may fire again on a
fresh frame after cooldown; use it when the target is already visible. Reusing a frame never
counts as new evidence. A small exit tolerance band reduces colour-threshold chatter.

### Closed-loop reach

After teaching a watch, call `ganglion_intent` with:

```json
{
  "spec": {
    "program": "reach",
    "watch_id": "COPY watch_id FROM watch",
    "click": true,
    "timeout_seconds": 5,
    "tolerance_px": 6,
    "settle_ms": 30,
    "speed_px_s": 1200
  }
}
```

It returns `intent_id`. The core corrects the pointer from `GetCursorPos` feedback at up to
100 Hz. A proportional controller limits each correction by speed and integrates at most 20 ms
after a scheduler stall. A pointer outside the client area enters at its nearest edge; that entry
can exceed the speed limit. All submitted destinations remain inside the bound client area.

Completion needs distinct, recent cursor samples within tolerance for the settling interval.
The optional click uses the observed cursor location and completes only after release. A submitted
move alone cannot complete an intent. The result explicitly keeps `task_success_verified: false`:
inspect the application's resulting state separately.

Use `ganglion_wait` with `kinds: ["intent_completed", "intent_failed", "intent_cancelled"]`, or
read `status.intent`. Its phase is `running`, `clicking`, `completed`, `failed`, or `cancelled`.
The latest terminal intent is retained until another starts. Failure reasons include
`target_lost`, `observation_stale`, `timeout`, `watch_removed`, and rejected output. Lease loss,
halt, or window changes cancel the intent. A color watch selects the largest matching component;
it does not guarantee object identity when several similar components compete.

Reach and drag admit new commands only with evidence at most 40 ms old, reserving 10 ms for
transport and validation. The helper rejects input after 50 ms from acquisition start. If a
command expires before submission, the controller re-observes and replans; it never replays an
uncertain button action. Programs fail after 250 ms without fresh evidence.

Live capture uses DXGI and, during quiet periods, a fresh GDI `BitBlt` acquisition at a target
20 ms interval. It flushes GDI and copies pixels before publishing them. Cached frames never get
new timestamps. Snapshots and action provenance distinguish `dxgi` from `gdi_refresh`, acquisition
start (`sample_started_mono`) from pixel availability (`captured_mono`), and processed source
counts (`capture_sources`). Slow acquisition consumes the evidence budget. Raw `bench` capture
remains DXGI-only, so fallback sampling does not inflate its frame-rate measurements.

`ganglion_cancel(intent_id=...)` ends the program and queues an output halt barrier. A command
already in flight may execute first; its deadline is at most 50 ms from its observation. New
pointer programs are rejected while `pending_commands` remain. Wait for `output_halted` or an
empty pending list before starting another program. Cancelling an intent preserves watches and
separately armed reflexes. `ganglion_halt` stops everything.

For discrete agent actions, `ganglion_input` accepts a `spec` containing `action: "move"` or
`"click"`, `point: [x, y]`, and `snapshot_id`. It checks the current layout, focus, bounds, and
occlusion. It does not re-detect an object at that point. Older snapshots from the same layout
are allowed, which makes the distinction between an agent-chosen point and a watched target
explicit. Discrete input and click reflexes cannot take the pointer during an active intent.

### First-person programs

`ganglion_input` also accepts `key` (tap), `hold` (up to four keys for `hold_ms`, re-issue to
extend), `look` (relative `delta` in mouse counts, optionally spread over `spread_ms`) and
`button` (hold a mouse button without moving the pointer). Movement keys, looks and button
holds mark a self-motion window during which motion watches report nothing.

`ganglion_watch` takes `kind`: `color` (default), `motion` (largest changed blob against a frame
`lag_ms` older on a brightness-normalised image, after `persist` comparisons), `track` (a crop
at `template_region` of the newest frame, followed by correlation within `search_px`) or `flow`
(the largest region whose optic flow disagrees with the view's own motion). A flow watch is not
blinded by own turns and walks: a phase correlation finds the translation that moves most of the
picture, dense flow on the aligned pair and one global affine fit explain the rest, and what
still disagrees by more than `flow_threshold` px over the lag is reported. What track and motion
watches followed on the last frame is a known mover: its box, grown a little, is kept out of the
alignment and the fit, so a target filling the view cannot pass for the view's own motion (the
summary's `excluded_fraction` says how much of the field that was). Its ego-motion summary
(translation in px/s, expansion and roll rates) rides along in the detection and in every
snapshot as `flow`, with `credible` false when the alignment was outside the percept's range
(about a third of the field over the lag) or fewer than half the sampled vectors agreed with
the fitted motion; `ganglion core --lptc-from-flow` hands a credible summary to the model's
lptc channel and zeros otherwise. Only adapter version 4 checkpoints take it (they were trained
with the slip of turning views in that channel); older versions ignore the flow, so the flag is
safe but idle with them.

`ganglion_intent` with `program: "align"` turns the view until the watched target sits at
`point` (client centre by default) for `settle_ms`, then holds `fire.button` for `fire.hold_ms`,
`fire.repeat` times; `controller: "connectome"` lets the model propose the view velocity under
the envelope. `program: "move"` holds `keys` until `until`, `timeout_seconds` or cancellation,
renewed by the runtime every 100 ms, and runs beside a pointer program. `ganglion_arm` responses
`key` (a bounded tap), `align` and `track` (an align intent on the watch, or on a tracker cut
around the detected blob) complete the reflex table.

### Drag and verify

Teach a source watch and a completion-condition watch, then call `ganglion_intent`:

```json
{
  "spec": {
    "program": "drag",
    "watch_id": "COPY source watch_id",
    "destination": [520, 180],
    "until": {"watch_id": "COPY condition watch_id", "present": true},
    "timeout_seconds": 5,
    "verification_seconds": 1,
    "settle_ms": 80,
    "speed_px_s": 600
  }
}
```

The destination above is an example in physical screen pixels. Supply exactly one of
`destination` or `destination_watch_id`; a watched destination follows the detected component.
The core reaches the source, confirms arrival, presses left, then moves toward the destination.
It releases when the predicate matches or arrival at the destination is confirmed. A predicate
already satisfied before pickup fails with `condition_already_satisfied`, without pressing.

With `until`, completion requires distinct samples acquired **after release** that match the
predicate for `settle_ms`. A gap over 100 ms resets that stability interval. `present: false`
can verify disappearance. A transient match while held is insufficient: a rejected drop that
snaps back fails `condition_not_observed` when the verification budget expires. Without `until`,
completion confirms cursor arrival and release only. In either case `task_success_verified`
remains false; `condition_verified` means only that the taught pixel predicate passed.

Drag phases are `running`, `completed`, `failed`, or `cancelled`. The `stage` field traces
`approaching`, `pressing`, `dragging`, `releasing`, and `verifying`; `button_held` records helper
acknowledgement. The source may disappear after pickup, but a lost watched destination fails
the move. Removing any referenced watch ends the intent. Cancellation and failure request
release; wait for `output_halted` before reusing the pointer.

## Delivery, actions, and bounded resources

- Cursors encode core epoch and sequence. Different readers keep independent cursors. A
  restarted core rejects old cursors. `lost_events` reports records overwritten before the
  delivered frontier; `has_more` means continue paging.
- Perception and action/fault events have separate 512-entry rings. Neither is durable.
  Complete historical coverage is not promised; skipped/old frames and ring overflow are visible.
- At most 16 watches and 32 reflexes are admitted. Detection older than 50 ms cannot start an
  action. Dropped old evidence does not consume an appearance edge.
- One intent or discrete command owns the pointer at a time. Competing reflexes produce `reflex_rejected`, never an
  interleaved button sequence. Each click holds for 5–100 ms, further capped by its admission
  deadline. The independent input helper releases on completion, halt, parent pipe closure, or
  error. Drag renewals extend a hold only to the earliest of lease/intent expiry, 100 ms after
  evidence acquisition, or 100 ms from helper receipt. Missing renewals release the button even
  when the producer is stalled. The helper checks target/focus while held and never reacquires
  an expired hold. These are scheduling bounds, not hard real-time OS guarantees.
- If release fails, the helper retains ownership and retries. The core reports a fault and
  refuses new leases until restarted; it does not acknowledge a successful cancellation barrier.
  No general keyboard/gamepad hold interface is implemented.
- Mutating tools accept `request_id`. Reuse it only for retries of the same operation. A changed
  payload with the same ID is rejected. The core retains 4,096 successful mutation results and
  then requires a halt/restart rather than evicting retry protection silently.
- `reflex_fired` means a command was admitted; `input_submitted` means SendInput accepted it;
  `input_released` records release. `effect_observed` means the watched target disappeared.
  That last event does not by itself prove task success. The Arena evaluator independently
  distinguishes hits, misses, and false actions.
- `pointer_feedback` records the observed cursor, submitted destination, target evidence and
  command ID in the perception ring; lifecycle, click and fault events use the action ring.
  Continuous motion can fill the perception ring within several seconds. Drain it with cursors.

## Next boundary

Gate C reach, drag-until-condition, quiet-screen capture, and browser-fixture transfer are
implemented. The [reach comparison](bench/REACH.md) and [drag scorecard](bench/DRAG.md) give
reproduction commands and bounded claims. A [Solitaire application check](bench/SOLITAIRE.md)
now demonstrates one accepted move and one rejected drop inside Anode. Application teaching
lives in an external profile; the core has no card rules. Broader board recognition remains open.
Templates, meters, OCR, gamepad input, recording, skill packs, learned vision, and connectome
controllers remain on the plan. No per-game rules were introduced in the core.
