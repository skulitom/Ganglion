# Ganglion — a fly-brain reflex layer for LLM agents

*Plan v0.1, 2026-09-16. Project directory `C:\DEV\Ganglion`. Builds on
[Haltere](../Haltere) (the male-CNS fly brain that flies Liftoff) and the seat tool (the
background Windows seat for agents).*

## 0. In one paragraph

LLM agents perceive and act at seconds per step; screens and games change at milliseconds.
Ganglion is a resident sub-brain that sits between an agent (Claude Code or Codex, over MCP)
and the computer. It watches the screen at the display's frame rate through a fly visual system,
keeps a timestamped account of everything that happened between the agent's turns, executes the
agent's intents as closed-loop motor programs (reach, track, dodge, hold, steer, fly) at 100 Hz,
and fires pre-armed reflexes within one frame. The agent stays the intelligence: it decides what
matters, teaches the sub-brain what to watch and how to react, and receives compressed perception
instead of stale screenshots. The name: in a fly the thoracic ganglia run the fast reflexes and
motor patterns while the brain sends descending commands. That is exactly the split.

Two things were measured today that make this concrete rather than hopeful (details in §2.3):

| fact | number |
|---|---|
| Haltere's 30,000-neuron connectome brain, one step, batch 1, RTX 4090 | **1.82 ms** (550 Hz) |
| DXGI desktop duplication inside the seat (child session 3, 1280x720) | **66 new frames/s, 0.06 ms per poll** |
| DXGI desktop duplication on the console (1920x1080, 75 Hz monitor) | 76 new frames/s, 0.07 ms per poll, 240/s in video mode |
| GDI capture (what the seat tool's screenshot uses), any region size | 13.4 ms per frame |

So a 100 Hz control loop with a frame-rate visual front-end fits on this machine, on the desktop
and inside the seat, with most of the tick budget to spare.

## 1. The gap Ganglion fills

Today a computer-use agent runs open loop: screenshot, think for seconds, one click, screenshot.
Anything that needs continuous or timely control is out of reach: dragging until a value reads
right, following a moving thing, dodging, aiming, rhythm, catching a dialog the instant it
appears, reacting to a health bar. And the agent misses everything that happened while it was
thinking. Two capabilities close the gap, and both are things a fly does with a brain of a few
hundred thousand neurons:

1. **Ultra-fast perception.** Not "a faster screenshot", but a running, always-on account of the
   screen: what moved where, what appeared, what is looming, what changed, where the tracked
   things are now, all stamped in milliseconds and compressed into a few hundred tokens when the
   agent asks. The agent never misses a fast event because the sub-brain saw it.
2. **Snap judgements and reactions.** Pre-armed reflexes and closed-loop motor programs that act
   within a frame of the trigger, without a round trip to the model. The agent arms them, tunes
   them and audits every firing in the ledger. The sub-brain only ever judges what the agent
   armed or taught; anything ambiguous wakes the agent instead of acting.

The requirement the user set: **any computer use and any game**, not one application. So the
built-in perception is pixel-based and application-agnostic (motion, looming, change, appearance,
small objects, wide-field flow, cursor and window state), and everything application-specific is
taught at runtime by the agent (point at things in a snapshot) or distilled offline from the
agent's own labels. Game telemetry (Liftoff's UDP stream) is an optional extra sense, never a
requirement.

## 2. What we already have

### 2.1 Haltere: a trained connectome brain with a 100 Hz loop

Everything below is from the current code (`C:\DEV\Haltere`, HEAD `22745c0`, 124 tests green).

- **The brain** is `ConnectomeRNN` (`haltere/brain/model.py:91`): 30,000 neurons of the male-CNS
  v1.0 connectome, 2,767,698 edges carrying 37.5 M synapses, rate `4 * sigmoid(v)`, forward Euler
  with learnable per-neuron time constants, one step = 10 ms (`BrainConfig.dt = 0.01`). State is
  just `v [N, B]` and a low-passed action `[B, 4]`. Sparse recurrence is `torch.sparse.mm` with a
  hand-written chunked backward (`haltere/brain/sparse.py`); no CUDA-only code, it runs on CPU
  too (but at 72 ms per step, so CPU is not a real-time option for the full graph).
- **Sensory channels** are written into named neuron populations by `PopulationEncoder`
  (`haltere/brain/encoders.py:12`): each neuron has a preferred direction in channel space,
  neurons of one cell type and body side share a tuning (`tuning_groups`, `model.py:83`), and the
  current is a soft-rectified projection. The shipped map (`configs/train.yaml:63`) writes
  gyro into haltere afferents (439 neurons), attitude into ocellar neurons (22), optic flow into
  lobula-plate tangential cells HS/VS (48), airflow into Johnston's organ (475), wing load into
  campaniform sensilla (218), heading into EPG compass cells (50) and the goal vector into PFL3/FC2
  (116). 24 input numbers in total, all built by one function shared between simulator and game,
  `observe_from_sensors` (`haltere/sim/tasks.py:46`), which has an explicit seam for injecting a
  goal that does not come from telemetry (`rel_b`): that seam is the "descending command".
- **Motor readout**: a linear map from whitened rates of the 83 wing motor neurons plus the 3,830
  premotor neurons to four axes, `[throttle, roll, pitch, yaw]`, then a muscle-like low-pass.
  Whitening uses slowly tracked running statistics, so batch-1 inference is exact.
- **Learnable**: per-edge gains (2.77 M), per-neuron gain/bias/tau, unknown-sign edges, encoder
  tunings, readout: 2.88 M parameters; "reservoir mode" freezes all but encoders and readout.
- **The live loop** (`haltere/liftoff/commands.py:686`, `TelemetryPilot.step`
  `haltere/liftoff/pilot.py:567`) is telemetry-driven at ~100 Hz: drain the UDP socket to the
  newest frame, build senses, one brain step, map sticks through the measured radial deadzone
  model, send to a virtual Xbox pad (ViGEm) or over UDP to the pad bridge. It has the operational
  scar tissue a real loop needs: dead-man neutral sticks after 500 ms of silence, focus watchdog,
  crash and stuck detection with a reset key, pad-dropped watchdog that re-plugs the pad, an
  exception-proof goal function that holds the last carrot rather than raise inside the loop.
- **Lessons that transfer** (README "what worked" and the code): rates `rate_max 4` and encoder
  gain 12 or the signal never reaches the motor neurons; read out of premotor neurons, not only the
  last motor stage (R² 0.7-0.8 vs 0.4-0.5); imitation of a working controller first, task cost
  second; bake the measured latency into training (`delay_steps`) instead of filtering at
  runtime (an output low-pass made Liftoff worse); domain randomisation (35-40%) is what
  survived the sim-to-game gap; the game decides which brain ships, sim rankings do not transfer
  at fine granularity; the whole pipeline is paced by a GPU thermal guard
  (`haltere/train/thermal.py`) because this PC has shut down from heat once.
- **The training harness** (`haltere/train/bptt.py`, `imitate.py`) only touches a small
  interface: a vehicle with `step(state, action)` and a task with `channels`, `reset_*`,
  `observe`, `cost`, `tick`, `metrics`. A new simulator plugs in there (§10.4).
- **Measured today**: `load_checkpoint('artifacts/ftPath2_best.pt', 'cuda')` takes 0.86 s after
  a 2.4 s import; params 11.5 MB, buffers 178 MB (edge indices); one `no_grad` step at batch 1 is
  1.817 ms on CUDA (1.890 ms if the weight matrix is rebuilt every step) and 71.98 ms on CPU.
  Training runs at ~3.2 s per iteration at B=256, T=64.

### 2.2 The seat, and why the fast loop must not go through it

From the seat tool's current source and docs:

- A **child session** (loopback RDP session of the same user) is a real second seat: own desktop,
  pointer, focus, processes; same files, same Steam library, same network ports. Roles: daemon in
  the user's session, seat host inside the seat (input, capture, launching, gamepad), thin CLI and
  MCP clients on a named pipe, NDJSON, one request in flight per connection.
- **Capture is GDI** `Graphics.CopyFromScreen` → PNG/JPEG → base64 → two pipe hops, one frame
  per request (its screen capture source). There is no streaming path, no
  DXGI, no Windows.Graphics.Capture anywhere in the repo. Fine for an agent's screenshot, not a
  perception loop.
- **Input is the right primitive with the wrong transport**: `SendInput` with scan codes
  (`Core/Input/InputInjector.cs`), ViGEm Xbox 360 pads with explicit report submission
  (`Core/Gamepad/GamepadManager.cs`, `AutoSubmitReport = false`). But every op serialises through
  two semaphores and two JSON hops, and the common ops embed sleeps (click hold 20 ms, key hold
  40 ms, drag 24 steps × 8 ms, pad tap 80 ms). Also `InputBlockReason()` does a `user32` foreground
  check on every send.
- **The seat is a normal session with two documented ways in**: `run`/`seat_run` starts a
  fire-and-forget process that survives the seat host (`Seat/SeatHost.cs:342`), and
  `SeatLauncher.LaunchInSession` (`Core/Launch/SeatLauncher.cs:25`) places any exe into a session
  via Task Scheduler without elevation. `seat_exec` jobs are the wrong shape for a control loop
  (30-minute hard cap, kill-on-job-close). Loopback TCP/UDP is shared across sessions (Haltere's
  telemetry and pad bridge already cross sessions daily); named pipes work cross-session with
  `CurrentUserOnly`.
- **Seat display facts**: RDP composition is capped by `DWMFRAMEINTERVAL = 15` (about 66 Hz,
  registry, set by the seat tool's fps setup); its GPU option sets `bEnumerateHWBeforeSW` so the seat renders
  on the RTX 4090 (Liftoff's D3D11 loading screen in session 5 was the evidence); exclusive
  fullscreen and protected video do not capture (use borderless); a hidden or minimised viewer can
  suspend RDP rendering, mitigated by `RemoteDesktop_SuppressWhenMinimized = 2`; a foreground
  `GameInputServiceWindow` blocks all synthetic input (the seat tool detects it; a repair script exists).
- The seat tool's own docs are honest that "competitive twitch play is not what this is". Today's
  measurement narrows that: the seat delivers 66 fresh frames a second to a DXGI reader with the
  viewer hidden, which is a 60 Hz game's full output.

### 2.3 Measurements taken today (2026-09-16)

Scratch benchmarks (dxcam 0.0.5 for DXGI duplication, windows-capture 2.0.1 for Windows Graphics
Capture, mss for GDI), an animated window providing fresh frames:

| capture path | console session (1920x1080, 75 Hz monitor) | seat (session 3, 1280x720, viewer hidden) |
|---|---|---|
| GDI `BitBlt` (mss) | 13.4 ms per frame regardless of size (vsync-bound) | not measured; the seat tool's screenshot op uses it |
| **DXGI desktop duplication** (dxcam) | **76 changed frames/s; 0.07 ms per poll; 241/s in video mode** | **66 changed frames/s; 0.06 ms per poll; frames 720x1280x3** |
| Windows Graphics Capture (windows-capture) | 38 fps, 26.6 ms between frames | 34 fps, 30.3 ms between frames |

| compute path | measurement |
|---|---|
| Haltere brain, 30k neurons, 2.77 M edges, batch 1, CUDA, `no_grad` | 1.817 ms per step (550 Hz) |
| same, CPU (16-core 7950X3D) | 71.98 ms per step (14 Hz) |
| Liftoff loop latency measured in Haltere (pad step → game input → gyro response) | 22 ms → 33 ms |
| GPU temperature at idle during these tests | 41-45 °C |

Conclusions: DXGI duplication is the capture path everywhere (it works in the child session,
which the seat tool's repository had never tested); the seat is capped at the RDP frame interval, the console
at the monitor's refresh; the brain step is 18% of a 10 ms tick on the GPU, so the full stack
(retina sampling, optic lobe, core, readouts) has a realistic budget of ≤ 5 ms per tick.

## 3. Design principles

1. **Two timescales, one ledger.** The fast loop (100 Hz motor, frame-rate vision) never waits
   for the model. The slow loop (the agent) reads a compressed, timestamped ledger of what the
   fast loop saw and did. Nothing the fast loop does is invisible to the agent afterwards.
2. **Descending commands, not remote control.** The agent sends intents (what to achieve, what to
   watch, what to do if) rather than mouse coordinates. It can still send raw input for discrete
   actions, executed by the core without pipe sleeps.
3. **Fly-native first, taught second.** Built-in percepts come from the fly visual system and are
   application-agnostic. Application semantics ("this is the enemy", "this bar is health") are
   taught per session by pointing, or distilled offline from the agent's labels. Generalization
   is the user's core priority; §9.5 turns it into rules and measurements.
4. **Prove the loop with boring controllers, then let the fly brain earn its place.** Haltere
   needed an MLP baseline to show the loop worked before the connectome learned anything. Every
   motor program ships first as a deterministic controller behind the same interface; the
   connectome version replaces it when it wins on the real bench.
5. **Own sandbox first.** A synthetic Arena (headless for training, windowed for the real
   capture-to-input chain with ground truth) before real apps; the console bench before the seat;
   the seat for anything the user wants to keep their desktop during.
6. **Measure before choosing.** Tick rates, capture paths, latency compensation and population
   choices are set by `ganglion bench`, not by assumption. This plan's numbers are replaced by
   measurements as they arrive.
7. **Safe by construction.** Dead-man renewal (reflexes and intents expire unless the agent keeps
   talking), rate budgets, an allow-list of keys and regions per session, a kill switch, and no
   anti-cheat games at all.
8. **Heat is a first-class constraint.** Training paced (temperature guard 65-70 °C, idle time per
   iteration, one GPU job at a time); inference kept to a small GPU duty cycle; the user is told
   before any long job.

## 4. Architecture

```
                 agent turn (seconds)                               fast loop (10 ms tick, frame-rate vision)
  ┌──────────────────────────┐   MCP stdio   ┌───────────────┐  NDJSON/TCP  ┌───────────────────────────────────────┐
  │ Claude Code / Codex      │◄─────────────►│ ganglion mcp  │◄────────────►│ ganglion core   (inside the target      │
  │  look / watch / teach /  │               │ (thin client, │  loopback,   │  session: console bench or seat)  │
  │  intent / arm / wait /   │               │ user session) │  cross-      │                                         │
  │  input / halt / skills   │               └───────────────┘  session OK  │  capture ── retina ── optic lobe ─┐     │
  └──────────────────────────┘                                              │  (DXGI)    (hex eyes)  (fly vision)│     │
                                                                            │  sensors ─────────────────────────┤     │
                                                                            │  (cursor, windows, change tiles,  │     │
                                                                            │   templates, meters, telemetry)   ▼     │
                                                                            │                            central core │
                                                                            │                            (Haltere RNN)│
                                                                            │  ledger ◄── reflex table ◄──┬─ readouts │
                                                                            │  (events, tracks, fires)    │           │
                                                                            │                              ▼          │
                                                                            │                    actuators: SendInput │
                                                                            │                    (mouse/keys), ViGEm  │
                                                                            └───────────────────────────────────────┘
                                                                                             │ (Windows input queue of THAT session)
                                                                                             ▼
                                                                                  the application / the game
```

**Processes.**

- `ganglion core` — one process in the session where the application runs. Dedicated threads:
  capture (DXGI duplication, newest frame only, like Haltere's telemetry drain), vision (retina
  sampling + optic lobe on the GPU), tick (100 Hz: senses → core brain → readouts → reflex table →
  motor programs → actuators), ledger writer, and a protocol server (NDJSON over loopback TCP,
  the same shape as the seat tool's pipe protocol so tooling is familiar). Python 3.13 + torch cu128 like
  Haltere, with the capture and sampling stages in native code (dxcam is a thin ctypes wrapper;
  hex sampling is a precomputed gather on the GPU). Rust replaces stages only if `ganglion bench`
  shows jitter the Python loop cannot hold (Haltere's Python pilot holds 100 Hz today).
- `ganglion mcp` — stdio MCP server the agent launches (Claude Code: `claude mcp add`; Codex:
  `[mcp_servers.ganglion]` in `~/.codex/config.toml`, both `command`/`args` stdio). Thin client of
  the core; owns nothing. Tool results are small JSON plus at most one image.
- `ganglion arena` — the project's own sandbox (§9.1): a batched headless simulator for training
  and a real window with a ground-truth telemetry stream for end-to-end tests.
- Lifecycle in the seat is the seat tool's job: `seat_start`, `seat_run ganglion core ...`, `seat_stop`;
  its screenshot/UIA tools remain available to the agent for slow, exact inspection.

**Per-tick data flow.** Frame (BGRA, GPU-resident when possible) → luminance → per-eye gaze
window → hexagonal receptor sampling → optic-lobe network → feature maps (motion, looming, small
object, flicker, wide-field flow) and central channels → core brain step with the agent's goal
channels → motor readouts → motor programs (which may override or blend readouts) → actuators.
In parallel the deterministic sensors update (cursor position and velocity from the OS, held
buttons, foreground window and its rectangle, change tiles, template matches, meters, telemetry).
Both streams feed the ledger and the reflex table.

**Timing model.** Vision runs at the capture rate (60-75 Hz here, 66 in the seat); the tick
runs at 100 Hz using the latest frame and the exact OS sensors; actuators are called at tick
rate; reflex evaluation happens at every vision frame and every tick (whichever brings new
evidence). Latency from a screen change to injected input is then capture (≤ 1 frame) +
vision (≈ 2-3 ms measured target) + one tick (≤ 10 ms) → **≤ 1 frame + 13 ms worst case**, and
the application's own frame time on top. That is the fly's regime (its looming escape circuit,
LC4/LPLC2 → giant fiber, reacts in tens of milliseconds). Treat that figure as the initial
**screen-change-to-input-submission target**, to be published as a percentile plus a deadline-miss
rate under named conditions (idle GPU, and a game rendering), never as a guaranteed bound;
application receipt and visible effect get their own numbers (§2.3, `ganglion bench`).

### 4.1 Runtime contracts (settled before Phase 1 code, from the 2026-09-17 review)

- **One owner per actuator.** An actuation arbiter hands out exclusive leases for the pointer, each
  held key or button, and each gamepad axis group. Programs and reflexes declare the resources they
  need; a conflict is a rejection or an explicit pre-emption result, never a silent blend (a drag
  holding the left button and a reflex clicking it is the canonical bug). Priority order is fixed:
  halt and fault recovery first. Blending only when a task needs it and the rule is written down.
- **Triggers evaluate once per observation.** Every capture gets an observation ID; a tick that
  reuses the same observation is not new evidence. Triggers have edge/level semantics, hysteresis,
  cooldown and a maximum action duration, so a persistent detection fires the configured number of
  times whether vision runs at 60 or 100 Hz.
- **Leases are deliberate.** A controlling client holds a bounded control lease with an absolute
  monotonic expiry returned on every call; read-only calls (`status`, `look` without side effects)
  never renew it; `wait` is capped at the remaining lease and returns `lease_expiring` so renewal is
  an explicit act; each action is bounded on its own so renewal cannot turn a key tap into a hold.
  Expiry releases every input the lease owned. The loopback protocol carries a session-scoped
  capability token, one controlling client by default, observer permissions separate, message-size
  limits, unknown operations rejected.
- **Faults invalidate commands explicitly.** Runtime states `ready`, `running`, `degraded`, `halted`
  with a cause. Lost target identity, capture access loss (distinct from a merely static desktop:
  DXGI reports the two differently), controller failure or expiry stop the affected programs and
  release their inputs; resuming needs fresh observations and valid ownership. A separate watchdog
  process owns the "release everything" path so a crashed core cannot leave a key held; a failed
  release stays a visible fault.
- **Targets are bound to the snapshot that defined them.** Snapshots carry an ID, session and
  window identity, layout revision, capture size and the image-to-screen transform; watches and
  intents name their coordinate space; targets are revalidated at actuation and must be rebound
  after a window, display or session change. Cursor coordinates and relative camera deltas are
  different types. Every input records whether it was submitted and whether its effect was
  observed.
- **The ledger has delivery semantics.** Monotonic sequence numbers, per-client cursors,
  `next_cursor`, the available range and explicit overflow counts; action lifecycle records kept
  apart from the perception summary with reserved capacity for failures and releases; observation,
  intent, reflex and command IDs on every event; idempotency keys on mutating requests so a retried
  click is never a second click.

## 5. Perception

### 5.1 Retina: the screen through fly eyes

The fly samples the world through ~750 ommatidia per eye on a hexagonal lattice. Ganglion samples
the screen the same way, with several independently pointable eyes (a gaze window = centre + zoom
over the captured frame):

| eye | default gaze | purpose |
|---|---|---|
| E0 wide | whole screen, fit to lattice | global change, wide-field flow, looming anywhere |
| E1 fovea | follows the cursor, the current intent target, or the crosshair; 4-6x zoom | precise tracking and reach |
| E2..En watch | regions the agent asks to watch (`watch`) | dialogs, meters, spawn points |

Sampling follows the flyvis convention so pretrained visual models apply unchanged: a hex lattice
of extent 15 = 721 receptors (31 columns across), each a 13x13-pixel box kernel at 1x zoom
(`flyvis.datasets.rendering.BoxEye(extent=15, kernel_size=13)`), luminance only in v1. For the
male-CNS route the lattice is the eye's own: the connectome assigns 892 hexagonal column
coordinates to the right optic lobe's columnar neurons (§10.2), so each screen sample lands in a
real column's lamina cells. Sampling
is a precomputed gather on the GPU (µs). Colour is not thrown away: the deterministic sensors
keep RGB for meters and templates; a colour-opponent receptor variant (the fly has R7/R8) is a
later option if a task needs it.

Gaze is an output as well as an input: the fly moves its head with neck motor neurons, and
Ganglion's fovea position is driven the same way (§10.3), so "look at that" is a learnable
reflex, not only a parameter. A gaze change must never look like a threat: panning the fovea over
a static desktop moves every receptor and zooming resembles expansion, so every gaze transform is
versioned and time-stamped, motion is compensated in a common coordinate space or the affected
temporal state is reset with a measured settling period during which looming and motion outputs
are suppressed, track identities survive gaze moves when the evidence allows, and reassigning a
watch resets its state instead of inheriting another region's history. Sampling itself is
calibrated, not assumed: luminance conversion, contrast normalisation, border behaviour and warm-up
are specified and probed with bars, contrast reversals, small targets and expansion stimuli, with
accuracy published against object size, speed, contrast and zoom; at 31 columns across a wide-field
eye a 1920-pixel screen is sampled every ~62 px, so tiny UI elements belong to the fovea and to
the native-resolution deterministic sensors.

### 5.2 The fly visual system as the front-end

Two routes to a connectome-constrained optic lobe, used in sequence:

- **Route A, bootstrap: flyvis.** Lappalainen et al. 2024 (Nature) ship a connectome-constrained
  model of the motion pathways of the optic lobe: 64 cell types, 721 columns, 45,669 cells,
  integration step `dt = 1/100` s, trained on optic-flow prediction, 50 pretrained networks
  (`pip install flyvis`, `flyvis download-pretrained`; tested on Python 3.9-3.12 on Linux, so it
  may need its own 3.12 venv on this machine or a re-implementation of its simple per-type
  dynamics inside Ganglion, which also removes the dependency). It gives working T4/T5
  direction-selective motion detectors, ON/OFF channels and lobula-plate tangential-cell-like
  outputs on day one. Latency to measure: one 721-receptor frame through 45,669 cells on the
  4090 (expected low single-digit ms).
- **Route B, target: the male-CNS optic lobe through Haltere's pipeline.** Haltere already selects
  populations declaratively from the male-CNS v1.0 annotations (`haltere populations`,
  `haltere build`, `configs/flight.yaml`). The optic lobe, the visual projection neurons (LC and
  LPLC types), the looming-to-escape pathway (LC4 and LPLC2 onto the giant fiber DNp01), the neck
  motor neurons and the descending neurons are all in that dataset; §10.2 gives the census of
  what is present locally. One connectome, one code path, one brain: vision, central circuits and
  motor readouts wired as in the animal. Route A serves as a teacher (match its cell-type
  responses on Arena videos) and as the reference for the built-in percepts.

### 5.3 Fly-native percepts (built-in, application-agnostic)

Each is a named channel the reflex table can trigger on and the ledger can report, with a
per-frame scalar and a per-column map:

| percept | fly origin | Ganglion computation (v1 deterministic, v2 from the network) |
|---|---|---|
| motion field | T4/T5 | per-column direction and speed; v1: block optical flow on the hex lattice |
| looming | LPLC2, LC4 → giant fiber | expansion rate and angular size of a growing object; v1: divergence of the flow field + blob growth |
| small object | LC11, LC10 | small dark/bright things moving against the background |
| flicker / ON-OFF change | L1/L2 | per-column luminance change with sign and persistence |
| wide-field flow | HS/VS | is the whole view moving (scroll, camera pan, own motion) and how |
| novelty | change persistence | appeared / vanished regions after a debounce |

The v1 deterministic versions exist so the product works before any network is trained; v2 reads
the same channels off the connectome network (route A then B) and the two are compared on the
Arena's ground truth.

### 5.4 Deterministic sensors (cheap and exact)

- Cursor position and velocity (`GetCursorPos`), held buttons and keys (ours), foreground window
  and rectangle, window under the cursor.
- Change tiles: 64x36 tiles of the frame, hashed every frame → a change map and per-region
  "quiet since" clocks (this is what makes `wait` cheap and exact).
- Template matchers for taught crops (OpenCV normalised cross-correlation at low resolution,
  GPU when available) → position and confidence streams.
- Meters: agent-defined bars/regions read as a fill ratio or dominant colour (health, progress).
- UI Automation focus and element under cursor, rate-limited (the seat tool's UIA is process-spawn per
  query and 4 s walks; Ganglion keeps a cached, throttled reader for the fast path and leaves
  deep trees to the seat tool's `seat_observe`).
- Optional OCR of changed regions at low rate (Windows OCR, local), off by default.
- Telemetry adapters: Liftoff UDP first (reusing `haltere/liftoff/telemetry.py`), a small plugin
  interface for others (a game that exposes state gets an exact extra sense; none is required).

### 5.5 Taught percepts (application semantics, per session or distilled)

- `watch`/`teach`: the agent names things by pointing at a snapshot: a region, a crop, a colour,
  a meter, "the thing that just moved". Ganglion builds a matcher (template plus a tiny embedding
  head) that runs every frame and reports position and confidence; the agent verifies with
  `look`.
- Offline distillation: with recording on, episodes (frames at 10-20 Hz, events, actions,
  outcomes) are stored locally; the agent (or Claude Opus 5 through the API, only when the user
  enables it) labels frames in batch; small heads are trained in minutes (paced) and saved as a
  skill pack for that application. This is the main lever for "leverage the main model's
  intelligence": the model's judgement becomes a 1 ms classifier the reflex table can use.

### 5.6 The percept ledger (what the agent actually receives)

A ring buffer of events `{t_ms, kind, where, what, confidence, data}` with kinds `change`,
`appear`, `vanish`, `motion`, `loom`, `track`, `meter`, `reflex_fired`, `intent_progress`,
`intent_done`, `intent_failed`, `anomaly`, `notify`. `look` returns the delta since the agent's
last call, compressed: runs merged, ranked by salience, capped (default 40 events, ≤ ~1.5k
tokens), plus current track positions, cursor, foreground window, and **one** composite image
(current frame at 640 px wide with optional overlays: tracks, change heat, cursor path since last
look, regions). One image with trails replaces ten screenshots.

## 6. Snap judgements and reflexes

A reflex is `{trigger, condition, response, budget, ttl}`:

- **Triggers**: any percept threshold in a region (loom > x toward the crosshair; motion entering
  region R; object appears in R; change in R after quiet; track lost; meter below y; taught thing
  detected with confidence > c; telemetry predicate; cursor arrived; timer).
- **Responses**: motor programs (tap key, hold key for N ms, click at target, reach then click,
  drag, stick burst or pattern, `dodge` = move opposite to the loom direction, `align` = steer to
  keep a target at the crosshair, `stop`), gaze moves (look at it), `notify` (wake the agent with
  the event), `cancel_intent`.
- **Budgets**: max firings per second, cooldown, allowed keys/buttons, allowed region, and
  an independent reflex `ttl` (default 30 s), plus the explicitly renewed control lease (§4.1).
  Read-only calls never renew control. When either bound expires, affected actions stop and
  owned inputs are released.
- **Judgement gate**: a reflex may require confidence above a threshold; below it the sub-brain
  notifies instead of acting. Every firing is a ledger event with the evidence that caused it, so
  the agent audits and tunes thresholds like a coach.

Evaluation is a table scan in the core at every frame and tick (microseconds), so the reaction
latency is the capture-to-input path of §4. Learned reflexes come from the Arena: dodge looming
(the giant-fiber pathway is the natural readout), intercept, pursue, keep-at-crosshair, with the
same imitation-then-cost recipe as Haltere (§10.4).

## 7. Motor programs and actuation

| program | closed-loop on | v1 controller | v2 (fly brain) |
|---|---|---|---|
| `reach(target, click?)` | OS cursor position; fovea confirms when the cursor is hidden | minimum-jerk profile + PID with measured latency compensation | core brain readout (goal channel = target vector) |
| `track(target, offset)` | tracked position (template / motion / telemetry) | PID on target error | core brain, moving-target training as in Haltere's path brains |
| `drag(from, to \| until)` | cursor + percept condition | reach + hold + reach | same |
| `scroll(dir, until)` | wide-field flow + change | paced wheel notches with flow feedback | same |
| `hold/press(pattern)` | timing | scheduler at tick resolution (1 ms timer period) | n/a |
| `steer(target)` (first-person, vehicles) | motion field, loom, taught target | yaw/pitch PID via relative mouse or right stick | core brain: compass/goal channels, DN readouts |
| `fly(waypoints)` | telemetry | Haltere's `TelemetryPilot` as-is | Haltere's brains as-is |
| `pad(pattern)` | none | stick/button sequences | n/a |

Actuation: `SendInput` (absolute moves for UI, relative deltas for games that read raw input,
scan-code keys, unicode text) and ViGEm pads through `vgamepad` (Haltere) or the same
`Nefarius.ViGEm.Client` semantics (explicit report submission, one HID report per tick). All
sleeps live in the core's scheduler, never in the transport. Pointer ballistics: the Windows
"enhance pointer precision" curve is either turned off for the bench or modelled in the Arena so
relative moves land where predicted; the reach controller closes the loop on the observed cursor
anyway. Every program has the interface `start(params) → id`, `tick(senses) → command`, `status`,
`cancel`, so deterministic and connectome versions are hot-swappable per program
(`brain: pid | mlp | connectome`).

## 8. Agent interface (MCP)

Stdio MCP server in the same repo (`python -m ganglion.mcp`), registered with

```bash
claude mcp add ganglion -- C:\DEV\Ganglion\.venv\Scripts\python.exe -m ganglion.mcp
```

and for Codex in `~/.codex/config.toml`:

```toml
[mcp_servers.ganglion]
command = "C:\\DEV\\Ganglion\\.venv\\Scripts\\python.exe"
args = ["-m", "ganglion.mcp"]
tool_timeout_sec = 90
```

Tools (v1; names are `ganglion_*` so they never collide with the seat tool's `seat_*`):

| tool | what it does | returns |
|---|---|---|
| `ganglion_status` | core alive, session id, screen size, capture fps, tick jitter, GPU ms per tick, armed reflexes, active intents, budgets | JSON |
| `ganglion_look` | ledger delta since last call + tracks + cursor + foreground window + one composite image (`region`, `zoom`, `overlays`, `image=false` to skip) | JSON + image |
| `ganglion_watch` | declare things to watch: `{name, region \| crop \| colour \| meter \| telemetry}`; assign an eye; returns watch ids | JSON |
| `ganglion_teach` / `ganglion_forget` | attach example crops/labels to a name (few-shot); remove | JSON |
| `ganglion_intent` | start a motor program (§7) with parameters and completion conditions; non-blocking | intent id |
| `ganglion_arm` / `ganglion_disarm` | add or remove a reflex (§6) | reflex id |
| `ganglion_wait` | block ≤ N s (default 20, max 60 to fit client tool timeouts) until an intent completes/fails, a reflex fires, a watched percept changes, or a condition holds; returns the ledger delta | JSON + optional image |
| `ganglion_input` | discrete input executed by the core: click/move/key/type/pad with exact timing, no transport sleeps | JSON |
| `ganglion_cancel` / `ganglion_halt` | cancel intents or reflexes; all-stop, release every held input, neutral sticks | JSON |
| `ganglion_record` | start/stop episode recording (frames + events + actions), local only | JSON |
| `ganglion_skills` | list/load/save skill packs: watches + reflexes + taught matchers + tuned parameters for an application | JSON |
| `ganglion_guide` | embedded guidance for agents (mirrors `skills/ganglion-agent/SKILL.md`) | text |

Control leases are claimed and renewed explicitly (§4.1); read-only calls never renew them.
Image results use MCP image content (JPEG, ≤ 640 px unless
asked). `wait` is the only long call; it is how the sub-brain "wakes" the agent, since MCP has no
push channel to the model. A later **body mode** inverts control for tighter coupling: Ganglion
hosts the agent loop through the Claude Agent SDK (or `claude -p` streaming input) and injects
events as turns; MCP stays the primary integration because it works unchanged with both Claude
Code and Codex.

### 8.1 What a session looks like

```
agent  ganglion_look                        -> "foreground: Notepad; cursor (812,411); 0 tracks; quiet 4.2 s" + image
agent  ganglion_watch {name:"save_dialog", region:[600,300,720,200]}
agent  ganglion_arm  {trigger:{appear:"save_dialog"}, response:{key:"Escape"}, ttl:60, notify:true}
agent  ganglion_intent {reach:{target:[1180,32], click:"left"}}
agent  ganglion_wait {until:"intent_done", timeout:5}
       -> events: intent_done reach 0.31 s, 2 corrections; reflex "save_dialog" fired at +0.9 s (Escape); image with trail
```

And in a game, with a taught enemy and a looming reflex:

```
agent  ganglion_teach {name:"enemy", crops:[...]}                # crops cut from the last look
agent  ganglion_arm {trigger:{loom:{region:"centre", rate:">1.5"}}, response:{dodge:{keys:"a/d", ms:250}}, budget:{per_s:2}}
agent  ganglion_arm {trigger:{detect:"enemy", conf:">0.8"}, response:{align:{target:"enemy"}, then:{key:"mouse1", hold:80}}, budget:{per_s:4}}
agent  ganglion_intent {steer:{target:"waypoint_marker"}}
agent  ganglion_wait {timeout:20}                                 # reads the ledger: fires, misses, health meter trend
```

## 9. Sandboxes

The user's instinct that this needs its own sandbox at first is right, for two reasons: training
needs a batched, ground-truthed world, and end-to-end tests need a real window whose truth we
know. The seat remains the deployment target for anything the user wants to keep their desktop
during.

### 9.1 Arena (own sandbox)

`ganglion.arena`, two faces of one world:

- **Headless, batched, on the GPU** (like Haltere's quad simulator): a 2D "screen" with cursor
  dynamics (pointer ballistics model, configurable latency and gain jitter), sprites that are
  static, moving, appearing, looming, flickering; windows and dialogs that pop up; scrolling
  lists; meters; simple game modes (dodge, pursue, intercept, aim, drive). Rendered as luminance
  videos onto the hex lattice for the optic lobe and differentiable where it matters (sprite
  positions with respect to actions, soft-splat rendering) so BPTT works as in Haltere.
- **As a real window** (SDL/pygame at vsync) with a UDP telemetry stream of the true state, so
  the entire chain DXGI → retina → brain → SendInput is tested against known truth on the console
  or in the seat. This is Haltere's `liftoff fake` pattern and it is how latency compensation
  and controllers are validated before any real application.

### 9.2 Console bench

The user's own desktop (a second monitor, or Cathode's virtual monitor) while they are away:
lowest latency (monitor refresh, 75 Hz measured), exclusive-fullscreen games capture fine, no
frame cap. Used for measurements and for anything the seat's 66 Hz cap would distort.

### 9.3 seat

`seat_start` → `seat_run` the core inside the seat (the process survives the seat host; not
`seat_exec`) → the core captures with DXGI (works, measured 66 fresh frames/s with the viewer
hidden) and injects with `SendInput` inside the seat → the MCP server in the user's session talks
to it over loopback TCP. ViGEm pads are machine-wide, so a game in the seat and one on the desktop
must not both listen to the same pad (Haltere's pad bridge convention: one pad process, one
game). Borderless window mode for games; Sandboxie-boxed Steam as Haltere does for Liftoff. The
seat's frame cap is `DWMFRAMEINTERVAL` (registry, admin, reboot): trying 10 (≈100 Hz) is a cheap
experiment for the user to decide on.

### 9.4 The target set (chosen by the user, 2026-09-17)

Generalization is the core priority, so the targets were picked to span input styles, visual
styles and cadences, and every one of them enters Ganglion through skill packs (data authored
with the public tools), never through core code. All are installed in the Steam libraries
(`D:\STEAM`, `C:\SteamLibrary`). The order builds skills incrementally:

| # | target | input | what it exercises | capture / input notes | policy |
|---|---|---|---|---|---|
| 1 | Desktop, browser, tabs (Explorer, Settings, Chrome/Edge, editors) | absolute mouse, keys | the base tools: reach/click, drag (sliders, files), scroll-until, typed text with timing, window switching, dialog and toast reflexes, "wait until loaded" for free | UIA and the seat tool's `seat_observe` as the exact slow sense; Claude in Chrome covers DOM work, Ganglion covers pixels in any app | none |
| 2 | The Zachtronics Solitaire Collection | absolute mouse, drag-and-drop | `drag`-until-condition on rule-constrained card moves with visual completion (the card lands or snaps back), taught card percepts (rank, suit, face-down), stack tracking; the Gate C test bed for drag | 2D, static, windowed | none |
| 3 | Warhammer 40,000: Dawn of War (Definitive Edition first; GOTY/Winter Assault/Dark Crusade/Soulstorm installed too) | absolute mouse, hotkeys | RTS micro at a strategic cadence that suits the agent: drag-select, minimap clicks, edge-scroll camera, build-queue hotkey sequences; change-detection reflexes (minimap alert → notify), meter reflexes (unit health → retreat) | Definitive Edition has modern windowed support; skirmish vs AI | no anti-cheat in skirmish |
| 4 | Diablo II: Resurrected (Infernal Edition) | absolute mouse click-to-move, hotkeys | meters (health/mana globes) driving the classic reflex "health < 40% → potion key within a frame"; enemies as taught crops; loot labels via OCR; inventory drag-and-drop | windowed mode in options | **offline single-player only**: Blizzard's EULA forbids automation and online play would risk the account; never Battle.net |
| 5 | Liftoff | ViGEm gamepad + UDP telemetry | the telemetry adapter and gamepad path; Haltere's brains as a `fly` program; the "any game with telemetry" case | Sandboxie-boxed Steam in the seat, borderless (Haltere convention) | none |
| 6 | Getting Over It with Bennett Foddy | mouse only, continuous | pure closed-loop continuous control of a nonlinear physics plant (the hammer) from visual feedback alone; the benchmark for the fly-brain control core and for learning a plant from human traces | Unity; windowed; cursor hidden and confined, so the loop closes on the tracked hammer, not the OS cursor | none |
| 7 | Half-Life (Black Mesa, HL2 and Opposing Force available for variety) | relative mouse (raw input), WASD | FPS aim and steer with relative deltas, `align` + fire, dodge on looming (barnacles, grenades, headcrab leaps), HUD digits as meters | `-windowed -noborder -w 1280 -h 720`; GoldSrc renders fine in the seat's D3D path (to verify in Phase 0) | single-player only (VAC-secured servers off-limits) |
| 8 | STAR WARS Battlefront (Classic, 2004) (Classic Collection installed too) | relative mouse + keys, vehicles | crowded scenes: many units and vehicles, radar, taught-percept scaling, vehicle steering | Steam `/win` launch option gives a window that locks the cursor and cannot be minimised, harmless inside a seat; the Classic Collection has native windowed modes | instant action vs bots only |
| 9 | Abiotic Factor | mouse + WASD, inventory UI | modern UE5 visuals plus UI inside a game: crafting menus, inventory drag-and-drop, survival meters, melee/ranged combat | UE5 borderless; host solo | no anti-cheat (verified via Steam discussions); solo worlds only |

What each unlocks: 1 proves the tools, 2 proves drag-and-drop with visual completion, 3 and 4
prove UI programs and reflexes at two cadences, 5 proves telemetry and pads, 6 is the
continuous-control benchmark, 7-9 prove relative-mouse 3D play with rising visual complexity.
A tenth target that is not on the list but is installed, "A Difficult Game About Climbing", is a
free second plant for the Getting Over It skill. Generalization is proven through small transfers
between two of these before all nine become release requirements (§11.1, Gate C).

### 9.5 Generalization discipline

Rules that keep "any computer use and any game" true rather than claimed:

1. **No application code in the core.** Anything specific to a target is data in a skill pack
   (watches, taught crops, reflex parameters, key maps), authored through the same MCP tools any
   agent could use on a new game. A target that needs a core change fails the discipline and the
   change must be generalised first.
2. **One percept vocabulary.** A new built-in percept must show value on at least three targets
   before it is merged.
3. **Joint training with a held-out target.** Every learned model (percepts, controllers) is
   trained on the Arena plus several targets and evaluated on one target it never saw; the
   held-out score is the headline number.
4. **Replay regression.** Recorded episodes from every target replay through the perception
   stack in CI; per-target precision/recall and reflex latency must not regress.
5. **The decathlon.** The same fixed task list runs on every target and is scored per target:
   find and click a named UI element; drag until a condition holds; track a moving thing for 10 s;
   react to an event within a latency bound; keep a meter above a threshold for a minute; complete
   a scripted objective. Score cards live in `docs/decathlon/`.
6. **Time-to-first-useful-play.** For each new target, measure the agent's teaching time (watches,
   crops, reflexes) until baseline play; the goal is under five minutes of agent time for a game
   Ganglion has never seen.
7. **Three claims, three comparisons.** "Continuous observation helps" = agent with periodic
   screenshots vs agent with the ledger, same discrete actions. "Local feedback helps" = ledger
   plus discrete actions vs ledger plus deterministic reflexes and controllers. "The learned brain
   helps" = identical runtime and percepts with deterministic, compact learned, and connectome
   controllers. Report success rate, reaction latency, false actions, recoveries, agent calls and
   tokens, teaching time and GPU cost, on matched seeds with repeated trials.
8. **Zero-shot, taught and fine-tuned are different results.** A target taught during an evaluation
   is not zero-shot; held-out tasks and recordings are split before thresholds or prompts are tuned,
   and a final untouched evaluation set is preserved. Where a primitive does not apply to a game,
   use a different task rather than forcing every game through one checklist.
9. **Freshness and completeness are different promises.** Control uses the newest frame only; the
   ledger therefore reports dropped-frame counts and coverage gaps, event recall is qualified by
   event duration versus capture cadence, and full-rate clips around triggers (not the 10-20 Hz
   recording) are what diagnose a missed one-frame event.

## 10. The fly-brain substrate

### 10.1 Staged use of the connectome

- **S0 (Phase 3): reuse Haltere's brain unchanged** as the central + motor core, imported from
  `C:\DEV\Haltere` (`pip install -e`), graph files shipped next to checkpoints (9.6 MB). The
  channels map onto computer control better than expected: goal ← target vector (screen or game
  space, `tanh(rel / scale)` as today), compass ← heading/scroll direction, haltere ← cursor or
  stick rates, lptc ← the wide-field flow that now comes from a real optic lobe instead of
  telemetry, jo / wing_cs / ocelli ← velocity, load and orientation analogues (or zero); motor ←
  four axes: for a pad the sticks, for a mouse (dx, dy, button pressure, scroll). Trained per
  program in the Arena with the Haltere recipe.
- **S1 (Phase 4): add the optic lobe and the reflex pathway.** Route A then B (§5.2). New readouts
  from the descending neurons (the giant fiber DNp01 and the DN population Haltere already
  selects, 1,314 neurons) give a "reflex vector" (escape/dodge, freeze, approach); neck motor
  neurons drive gaze (E1 fovea position).
- **S2: actuator-specific motor populations** chosen by decodability probes (Haltere's method: a
  linear readout on whitened rates against a teacher's commands) rather than by assumption: wing
  motor neurons for sticks, leg motor neurons as the candidate for the mouse, neck for gaze.

### 10.2 Census of the local connectome (computed 2026-09-16 from `C:\DEV\Haltere\data\raw`)

The male-CNS v1.0 flat release is complete on disk (annotations, neurotransmitters, weights;
211,577 bodies, 165,122 traced, 25.56 M edges carrying 124 M synapses across the whole CNS).
Everything route B needs is there:

| what | count | notes |
|---|---:|---|
| optic lobe, both sides (`ol_intrinsic` + photoreceptors + `visual_projection` + `visual_centrifugal`) | 105,265 | right side 52,060; right optic lobe alone 4.64 M edges / 19.3 M synapses |
| **columnar neurons with hexagonal column coordinates** (`assignedOlHex1/2`) | **23,720** | 15 types (L1, L2, L3, L5, C2, C3, T1, Mi1, Mi4, Mi9, Tm1, Tm2, Tm4, Tm9, Tm20); **892 distinct columns on the right eye**: a ready-made pixel-to-column map |
| photoreceptors `R1-R6`, `R7*`, `R8*` | 6,098 | only 1,394 `R1-R6` traced and none with column coordinates, so pixels are injected at the lamina (`L1`/`L2`/`L3`, 892 hex-addressed columns), the fly's first processing stage |
| motion detectors `T4a-d`, `T5a-d` | 6,865 + 6,720 | all present; ACh |
| lobula-plate tangential cells (wide-field flow) | HSN/HSE/HSS/HST 8, VS 18 (+VSm/VST 16), H2 2, VCH/DCH 4; H1 2 (tagged `ol_intrinsic`) | HS input is 85-90% T4a/T5a, VS input 55% T4d/T5d, as in the literature |
| looming / escape projection neurons | LC4 126, LPLC2 185, LPLC1 134, LPLC4 97, LC6 124, LC16 182 | 48 `LC*` types, 4,253 cells in total; small-object LC11 143, LC10 960, LC12 498 |
| **giant fiber `DNp01`** | 2 | input: `LC4` 17.3% (all 126 cells) + `LPLC2` 13.2% (all 185 cells); partners `DNp02`/`DNp03`/`DNp04` (55% LC4)/`DNp06`/`DNp11`; outputs to `TTMn`, `GFC2-4`, `DNp11` |
| steering / backward descending neurons | `DNa01`, `DNa02` (PFL3 input), `MDN` | present, 2-4 cells each |
| descending neurons / ascending neurons | 1,314 / 1,846 | 480 / 567 types; DN → VNC motor 229,565 synapses |
| **neck motor neurons** (`subclass == 'nm'`) | 44 | 20 in the brain exiting via the cervical nerve, 24 in T1; **direct HS/VS → neck-MN edges exist** (VS → `CvN6` 216 synapses) — the gaze pathway is wired |
| leg motor neurons `fl`/`ml`/`hl` | 135 / 116 / 130 | muscle-named types (tibia extensor/flexor, ...) |
| wing / haltere motor neurons | 67 / 16 | Haltere's 83 |
| ocellar interneurons `OCG`/`OCC` | 46 | there are **no ocellar photoreceptors** in the release |

Candidate real-time graphs (edges at synapse weight ≥ 3, Haltere's threshold):

| set | neurons | edges (w ≥ 3) | edges (w ≥ 1) |
|---|---:|---:|---:|
| Haltere flight graph (today) | 30,000 | 2.77 M | 6.20 M |
| C: right optic lobe + right VPNs + centrifugal + all DNs + neck MNs | 53,418 | 2.72 M | 6.51 M |
| **D: right optic lobe + right VPNs + flight graph + DNs + neck MNs** | **80,988** | **5.46 M** | 12.64 M |
| F: both optic lobes + flight graph + DNs + neck MNs | 133,561 | 8.01 M | 18.90 M |
| G: every traced neuron | 165,122 | 10.51 M | 25.56 M |

Set D is the route-B target: one connectome from photons to sticks. Its sparse step moves about
44 MB of CSR data (2 × 5.46 M edges) — a fraction of the 4090's bandwidth even at 1 kHz.
Extrapolating Haltere's measured 1.82 ms (2.77 M edges) gives roughly 3-4 ms per step for set D,
inside the 5 ms budget; Phase 0's bench replaces the extrapolation. Several eyes are **several
recurrent states, not one brain**: the optic-lobe network runs with shared weights and one state
per eye (a batch dimension of the visual network only; its state and intermediate tensors scale
with the eye count and are benchmarked at one, two and four eyes), and their percepts are fused
explicitly — with eye identity, gaze transform, observation age and confidence — into the single
central controller. If the goal is ultimately one integrated connectome, the plan has to say how
several visual samples enter that one recurrent graph; batching does not achieve it.

**Neural time is independent of frame arrival.** The networks integrate with a fixed 10 ms step
(Haltere and flyvis both use `dt = 0.01`); frames arrive at 60-75 Hz. Advancing one step per
frame would run the brain at 0.6 s of model time per second and make every temporal tuning depend
on the capture rate, so the contract is fixed sub-steps on the 100 Hz tick with sample-and-hold
of the latest time-stamped observation: a held frame advances the dynamics but never counts as a
new observation (no re-triggered appearance events); catch-up after stalls is bounded and the
state is reset beyond that bound; the sub-step cost is part of the compute budget. Acceptance:
equivalent motion replayed at 30/60/75 Hz with jitter and drops gives the same speed estimates,
direction tuning and trigger times in seconds.

Two findings for Haltere while doing this: `configs/flight.yaml` line 31 `{entry_nerve: [ON]}` is
read by YAML as the boolean `true` (`flight.meta.json` shows `"entryNerve": [true]`), so
`pop__ocelli` holds only the 22 OCG cells; the ocellar nerve carries no photoreceptors in this
dataset anyway, and quoting `'ON'` would add 39 mechanosensory cells and change the graph (which
invalidates checkpoints), so leave it unless a rebuild is planned. And
`haltere.connectome.sources.load_annotations()` keeps only 18 columns and drops
`assignedOlHex1/2` and `somaLocation`; Ganglion reads the feather directly
(`pyarrow.feather.read_feather`). Soma positions are 8 nm voxel coordinates (x separates the
hemispheres, right at small x; z runs brain → VNC); for retinotopic drawings the hex column
coordinates are the right key. Census scripts and full outputs are kept in `docs/census/`
(`01_schema.py` … `04_followups.py` with `*.out.txt`; run with Haltere's venv Python); the exact
pandas selectors are recorded there and become `configs/vision.yaml` populations in Phase 4. The
capture benchmarks of §2.3 are in `docs/bench/` (`bench.py`, `seatbench.py`, `wgc.py`,
`animate.py`) until `ganglion bench` replaces them.

### 10.3 Sizes and compute budget

| component | neurons | notes |
|---|---:|---|
| Haltere core (flight graph) | 30,000 | 2.77 M edges; 1.82 ms per step measured |
| flyvis optic lobe, one eye (route A) | 45,669 | 721 columns × 64 types; `dt` 1/100 |
| male-CNS right optic lobe + projection neurons (route B) | 52,060 | 892 hex-addressed columns; 2.7 M edges at w ≥ 3 |
| set D = the above + flight graph + DNs + neck MNs | 80,988 | 5.46 M edges at w ≥ 3; est. 3-4 ms per step |
| descending (1,314) + neck (44) + leg (381) motor readouts | 1,739 | linear readouts, negligible |

Target: ≤ 5 ms GPU per tick for the whole stack at batch 1 (two eyes + core), measured by
`ganglion bench`; if the male-CNS optic lobe is too large for that, it is pruned by Haltere's
coupling-rank rule (`haltere/connectome/graph.py:214`) or run at the vision rate (60 Hz) while the
core stays at 100 Hz.

### 10.4 Training recipe (Haltere's, generalised)

1. **Baselines first**: deterministic controllers (§7) and an MLP policy on the Arena task prove
   the task, the observation and the cost are sane.
2. **Imitation with DAgger** (`haltere/train/imitate.py`): the connectome brain reproduces the
   teacher's commands on the teacher's own episodes, with a growing share flown by the student.
3. **Task-cost fine-tune** through the differentiable Arena (truncated BPTT, `brain_detach_every
   8`), with latency baked in (`delay_steps` = the measured capture-to-input latency, 2-6 steps),
   smoothness costs (`w_dact`), and domain randomisation of latency (0-60 ms), pointer gain (±40%),
   noise and sprite appearance so the result survives real applications.
4. **Bench decides**: rank brains on the Arena window and real applications, not the headless sim.
5. **Thermal pacing** everywhere: `max_gpu_temp` 65-70, `iter_sleep`/`batch_sleep`, one GPU job at
   a time, never during a live session, and the user informed before any run over a minute.

Non-negotiable parameters inherited from Haltere: `rate_max 4.0`, `encoder_gain 12.0`,
per-(type, side) tuning groups, readout whitening with slow running statistics, readout from
premotor populations as well as motor neurons.

### 10.5 Datasets

- Arena episodes (unlimited, labelled by construction).
- Human traces: the user's own screen + input recordings at 100 Hz (`ganglion record`), for
  imitation of natural cursor motion and game play; local only, opt-in.
- Agent-labelled frames from recorded sessions (§5.5) for taught percepts.
- Haltere's Liftoff datasets and brains, reused as the first real-game case.

## 11. Phases, milestones, acceptance tests

Effort is in focused working days; Claude can do most of it autonomously in workflows, with the
user needed for installs that need admin rights, game choices, and anything that touches their
desktop while they are using it.

| phase | scope | acceptance test | needs the user |
|---|---|---|---|
| **0. Measure and scaffold** (1-2 days) | repo skeleton, venv (3.13 + torch cu128 + dxcam + mcp), `ganglion doctor`, `ganglion bench` (capture, SendInput→cursor latency, brain step, Arena window change→capture latency, tick jitter), CI with pytest | `ganglion bench` prints the latency table for console and seat; numbers in this plan replaced | nothing |
| **1. Core loop v0, deterministic** (3-5 days) | DXGI capture thread, sensors (§5.4), change tiles, ledger, actuators, motor programs v1 (reach/drag/scroll/hold/input), reflex table with deterministic triggers, MCP server with look/watch/intent/arm/wait/input/halt/status, Arena window with truth telemetry | Claude Code drives the Arena window through MCP: clicks a target within 1 frame of its appearance via an armed reflex; reaches a moving target with < 1 tick of steady-state lag; same inside the seat; `look` ≤ 100 ms; the desktop/browser decathlon tasks (§9.5) pass; Dawn of War Definitive Edition skirmish UI driven end-to-end (drag-select, minimap, build queue) | run a session in the seat |
| **2. Fly vision front-end** (4-6 days) | hex retina with gaze windows on the GPU; flyvis optic lobe running at capture rate; fly-native percepts (§5.3) exposed in the ledger and as triggers; Arena dodge/pursue modes; percept latency measured | looming-triggered dodge fires within 1 frame + 13 ms; motion-based tracking of an untaught moving sprite; percept maps agree with Arena truth (precision/recall targets set in Phase 0); Diablo II offline: potion reflex from the health globe within a frame, loot labels read; Half-Life: dodge reflex on a looming barnacle/grenade | nothing |
| **3. Connectome control core** (5-8 days) | headless Arena as a Haltere task; MLP baseline; imitation + fine-tune of the Haltere brain for reach/track/steer; hot-swap per program; Liftoff via the existing telemetry adapter as the first real game | fly-brain `reach`/`track` matches or beats the PID on the Arena window (error, settle time, overshoot); Liftoff hover/lap flown from a Ganglion intent in the seat; Getting Over It: closed-loop hammer control from visual feedback clears the first tree from the pot, deterministic and fly-brain versions compared | approve GPU training runs (paced) |
| **4. Male-CNS optic lobe, reflex readouts, gaze** (6-10 days) | route B populations through Haltere's build; flyvis as teacher; DN/giant-fiber reflex vector; neck-MN gaze; taught percepts + offline distillation loop; skill packs; body-mode prototype | one connectome (male CNS) runs vision + core + reflexes ≤ 5 ms per tick; Half-Life, Battlefront 2004 and Abiotic Factor played by agent + reflexes (agent chooses targets, sub-brain aims/dodges) with a measured win over agent-only; the full decathlon scored on all eight targets with one held-out game (§9.5) | enable cloud labelling or not |
| **ongoing** | safety review, `SKILL.md` for agents, README with videos (as Haltere/the seat tool), publication | | publish decisions |

Each phase ends with the memory file updated (`ganglion-project`) and `STATUS.md` in the repo
brought current (phase, known failures, exact reproduction commands), so a fresh session or a
clean checkout explains its own state.

### 11.1 Delivery gates inside Phases 1-3 (from the 2026-09-17 review)

Calendar estimates above are guesses; these gates are the reviewable outcomes, in order:

- **Gate A, reproducible foundation** (= Phase 0): installable package, locked environment with
  optional extras for GPU models, `doctor` that separates detection from any active probe, data
  locations parameterised (`GANGLION_HALTERE`), versions and provenance recorded next to every
  benchmark, wire and skill-pack schemas versioned before recordings depend on them. Exit: a fresh
  environment installs, validates and runs the deterministic tests without a desktop or GPU.
- **Gate B, one observable reaction**: an Arena mode where a target appears at random places and
  times, accepts clicks and logs ground truth; only `look`, `watch`, `arm`, `wait`, `halt`,
  `status`, one deterministic detector and one bounded click response, on the final runtime
  interfaces and with the full timing chain instrumented (event scheduled, presented, captured,
  percept ready, input submitted, input received, effect observed). Exit: the agent arms the
  reaction, leaves for several seconds and returns to a ledger of successes, misses, false actions
  and timings; nothing fires outside a valid lease; halt and target loss verified; console first,
  then the seat. This is the project's central promise, demonstrated without training.
- **Gate C, one closed-loop program and one transfer**: `reach` with a deterministic controller,
  explicit target-loss outcome and completion by observed state, on a moving Arena target and on
  a browser task, then `drag`-until-condition (the Solitaire Collection is its test bed). Exit: a
  scorecard comparing periodic agent input with the local controller, and the same primitive
  working in two environments through the public interface only.
  *2026-09-17 implementation:* deterministic reach and bounded drag now transfer through the same
  MCP tools between pygame and Edge, on the console and seat. Quiet-screen capture and
  post-release condition checks are covered by the [drag scorecard](docs/bench/DRAG.md).
  The subsequent [Solitaire check](docs/bench/SOLITAIRE.md) demonstrates one taught legal card
  move and a rejected drop inside the seat; general card recognition/board tracking remain open.
- **Gate D, connectome experiments on a working system**: the visual model and later the
  controllers run in shadow mode first (same observations, logged predictions, no authority), are
  compared on frozen replay sets and live Arena runs, and are promoted per percept or program only
  after agreed criteria. Hot-swaps have a lifecycle: state initialisation, shadow warm-up,
  ownership handover, rollback; a swap never injects a command discontinuity into an active drag.
  *2026-09-17 implementation:* the actual Haltere core now runs optionally in a separate shadow
  worker on the same generic motor observations, with bounded queuing, inference timing,
  prediction disagreement and frozen replay. Its flight checkpoint remains untrained for
  desktop control; no neural promotion or fly visual front-end is claimed. The next work is
  generic perception/control adaptation and latency, with application tests measuring transfer.
  *Later the same day:* intents may grant the connectome supervised authority (a proposal acts
  only when it brings the cursor closer; the deterministic controller acts otherwise), and the
  DAgger cursor readout produced 90.6% of accepted pointer commands during live Sawayama drags.
  Solitaire runs as an evaluation harness outside the core; see `docs/bench/SOLITAIRE.md`.

First regression cases to keep green from Gate B on: lease expiry during a wait and renewal by
another client; lost response then retry of the same mutating request; simultaneous drag and
click with cancellation and pre-emption; repeated observation IDs and sustained triggers; capture
access loss, inference stall and input-submission failure; a target window moved, resized or
replaced after teaching, including scaling; ledger overflow with two readers; gaze pan and zoom
over a static scene and dropped frames. Protocol and state tests run on a fake clock and fake
actuator; perception on replay; capture and input on explicitly selected Windows runs.

## 12. Risks and non-goals

| risk | mitigation |
|---|---|
| Seat frame cap (66 Hz) and RDP rendering suspension when hidden | measured fine today with the viewer hidden; console bench for higher rates; `DWMFRAMEINTERVAL 10` experiment; `SuppressWhenMinimized` already set by the seat tool |
| Python jitter at 100 Hz | dedicated threads, torch releases the GIL, 1 ms timer period, GC tuning, process priority; Haltere holds 100 Hz today; Rust stage-by-stage if the bench says so |
| GPU contention with the game | tiny kernels on a separate CUDA stream; vision at 60 Hz if needed; ≤ 5 ms budget enforced by the bench |
| The connectome brain fails to learn a program (Haltere took days of diagnosis) | deterministic path keeps the product useful; decodability probes before training; imitation first; the interface hides which controller runs |
| Perception floods the agent with tokens | ledger compression, caps, one composite image; `look` budgets tunable |
| `GameInputServiceWindow` blocks `SendInput` in the seat | the seat tool's repair script; detection reported in `ganglion_status` |
| Heat | paced training, low temperature guard, one job at a time, never during flights |
| Privacy | everything local by default; recordings opt-in; API labelling only when enabled by the user |
| Anti-cheat and terms of service | non-goal: no online competitive or anti-cheat-protected games; `SendInput`/ViGEm only, no driver-level or hardware spoofing. Per target: Diablo II: Resurrected offline single-player only (Blizzard's EULA forbids automation; online play would risk the account); Half-Life and Battlefront single-player / vs bots only (VAC-secured servers off-limits); Abiotic Factor solo worlds |
| Old games in the seat (2004 Battlefront, original Dawn of War, GoldSrc) | windowed modes are quirky (`/win` locks the cursor, no borderless without a helper); Phase 0 adds a `ganglion doctor --game` check (launches in the seat, verifies capture and input) and the modern editions (Definitive Edition, Classic Collection) are the first choice |
| flyvis on Python 3.13 / Windows | separate 3.12 venv or re-implement its per-type dynamics in Ganglion (small), keeping their connectome and parameters |

Non-goals: a security sandbox (the seat is desktop isolation, not security; same here), automating
purchases or credentials (Claude's rules apply regardless of the sub-brain), replacing the agent's
judgement.

## 13. Repo layout, tooling, conventions

```
Ganglion/
  PLAN.md  README.md  pyproject.toml  configs/  skills/ganglion-agent/SKILL.md  tests/  docs/
  ganglion/
    cli.py           ganglion doctor | bench | core | mcp | arena | record | train | ...
    core/            loop.py (tick scheduler), capture.py (DXGI), sensors.py, ledger.py,
                     actuators.py (SendInput, ViGEm), intents.py (motor programs), reflexes.py,
                     protocol.py (NDJSON over loopback TCP)
    retina/          hexeye.py (gaze windows, hex sampling on the GPU), eyes.py
    brain/           optic.py (flyvis wrapper, later male-CNS optic lobe), core.py (Haltere
                     ConnectomeRNN adapters), readouts.py, controllers.py (minjerk/PID/MLP)
    percepts/        motion.py, loom.py, change.py, templates.py, meters.py, ocr.py, telemetry/
    arena/           world.py (batched sim), window.py (SDL app + truth UDP), tasks.py
    mcp/             server.py (FastMCP), tools.py, guide.md
    train/           datasets.py, label_with_llm.py; imitate/bptt/thermal imported from haltere
```

- Python 3.13 venv with torch 2.11 cu128 (same as Haltere), `pip install -e C:\DEV\Haltere` for the
  brain, `dxcam`, `mcp`, `vgamepad`, `opencv-python`, `pygame`; `uv` for everything.
- Conventions carried over: one CLI, YAML configs, checkpoints self-describing with their config,
  slim exports, tests with `pytest -q`, GIFs/MP4s in `docs/` for the README, thermal guard in every
  training entry point, memory file kept current for resumption.
- Fast-loop rules (from Haltere's scar tissue): newest-frame-only drains; no exception may escape
  a tick (degrade to the last safe command); all sleeps in the scheduler; dead-man on every
  actuator; every watchdog logs a ledger event.

**How to resume** (for a fresh session): read this plan, `ganglion doctor`, `ganglion bench`, then
the phase's acceptance test; the memory file `ganglion-project` holds the current phase and the
last measurements.

## 14. Decisions taken here, and open questions for the user

Decided (routine calls, easy to reverse):

- Name and directory: **Ganglion**, `C:\DEV\Ganglion`.
- Integration format: **MCP stdio** first (works with Claude Code and Codex unchanged), body mode
  via the Claude Agent SDK later.
- Capture: **DXGI desktop duplication** on both the console and in the seat; WGC and GDI only as
  fallbacks.
- Language: Python + torch for everything at first (Haltere precedent holds 100 Hz), native stages
  only where the bench demands.
- Sandbox order: Arena → console bench → seat; the fast loop runs inside the target session
  as its own process and never through the seat tool's pipe.
- Fly vision: flyvis to bootstrap, male-CNS optic lobe as the target substrate.

Decided by the user (2026-09-17):

- The name stays Ganglion (the cyst connotation was raised and accepted).
- Generalization is the core priority (§9.5).
- Target set: desktop/browser/tab navigation, the Zachtronics Solitaire Collection (drag and drop),
  Dawn of War, Diablo II: Resurrected (offline), Liftoff, Getting Over It, Half-Life, Star Wars
  Battlefront (2004), Abiotic Factor (§9.4).
- The seat's frame cap may be raised: `scripts/set-seat-fps.ps1` writes `DWMFRAMEINTERVAL = 10`
  (administrator, then a reboot at the user's convenience); the console may be used meanwhile.
- Everything stays local for now: no cloud labelling; distillation runs on local models or the
  agent's own session labels.
- Public repository on GitHub under the MIT licence from the start.
- The external review in `suggestions/` (2026-09-17) was adopted where it sharpened the plan:
  runtime contracts (§4.1), the latency budget as a measured target (§4), gaze compensation and
  sampling calibration (§5.1), explicit eye fusion and the neural-time contract (§10.2), the
  three-claim evaluation and the zero-shot/taught/fine-tuned split (§9.5), and the delivery gates
  (§11.1). Deferred: a durable audit spool for the ledger and shuffled-topology controls, both
  scheduled for when Gate D has a working system to apply them to.

Open: none that block Gates A-C. The order of the real-game work after Gate C follows §9.4 unless
a game's windowed mode fails the `doctor --game` check.

## Appendix A. References

- Male CNS connectome v1.0: Janelia FlyEM, Cambridge Connectomics, Google Connectomics; Berg et
  al., Cell, September 2026; neuPrint `male-cns:v1.0`; bulk files at
  `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/` (Haltere README).
- Lappalainen et al., "Connectome-constrained networks predict neural activity across the fly
  visual system", Nature 2024; code and pretrained models: <https://github.com/TuragaLab/flyvis>,
  docs <https://turagalab.github.io/flyvis/> (BoxEye extent 15 → 721 receptors, kernel 13;
  45,669 cells; `dt = 1/100`; 50 pretrained networks).
- Looming escape pathway: Ache et al., "Neural Basis for Looming Size and Velocity Encoding in the
  Drosophila Giant Fiber Escape Pathway", Current Biology 2019 (LC4 and LPLC2 synapse directly
  onto the giant fiber; size and velocity components).
- The seat tool's protocol, architecture, capture, input injection and in-session launch: its own docs and source.
- Haltere brain and loop: `haltere/brain/model.py`, `encoders.py`, `sparse.py`;
  `haltere/sim/tasks.py::observe_from_sensors`; `haltere/liftoff/pilot.py`, `commands.py`;
  `haltere/train/bptt.py`, `imitate.py`, `thermal.py`.
- Capture APIs in RDP sessions: DXGI duplication is commonly reported unavailable or black in RDP;
  today's measurement shows it working in the seat's child session on this machine (RTX 4090,
  `bEnumerateHWBeforeSW = 1`), which is the fact this plan relies on and Phase 0 re-verifies.
- Codex MCP configuration: `~/.codex/config.toml` `[mcp_servers.<name>]` with `command`, `args`,
  `env`, `tool_timeout_sec`; Claude Code: `claude mcp add`.
