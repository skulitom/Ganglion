# Connectome shadow integration

Measured 2026-09-17. Haltere's actual `ConnectomeRNN` now receives the same generic motor
observations used by Ganglion's reach/drag controller. The loaded flight checkpoint contains
30,000 neurons and 2,767,698 edges. Its identity and SHA-256 are recorded in every report.

This starts Gate D. The model has no input authority. All successful reaches below were
executed by the existing proportional controller. These initial measurements used a flight
readout. Subsequent [generic cursor training](TRAINING.md) has produced experimental readouts,
but no candidate has earned input authority. There is no fly visual front-end or Solitaire win.

## Live results and frozen replay

| Measurement | Recorded live Arena, Anode session 3 | Live Arena after packing sensor transfers | Frozen replay |
|---|---:|---:|---:|
| Reference reaches accepted / attempted | 3 / 3 | 3 / 3 | No input |
| Reference false actions | 0 | 0 | No input |
| Completed neural inferences | 86 | 93 | 86 |
| Inference p50 | 5.584 ms | 5.276 ms | 1.712 ms |
| Inference p95 | 19.877 ms | 10.752 ms | 2.448 ms |
| Inference p99 | 31.701 ms | 32.266 ms | 2.715 ms |
| Inferences within 5 ms | 38.4% | 40.9% | — |
| Mean proposal/reference point disagreement | 22.02 px | 21.21 px | 21.96 px |

Sources: [recorded live run](results/shadow-connectome-seat-recorded.json),
[packed-transfer run](results/shadow-connectome-seat-packed.json),
[frozen replay](results/shadow-connectome-replay.json). An earlier, less fully instrumented
[live run](results/shadow-connectome-seat.json) is retained too.

The packed adapter uses one host-to-GPU sensor transfer per step instead of seven. The replay
reproduced all 86 recorded cursor proposals; maximum raw neural-output difference was
3.39e-7. Both live runs ended halted with no pending input or ledger gaps. The recorded run
replaced seven queued samples and discarded one result; the packed run replaced four and
discarded one. Capture dropped three and 21 observations respectively. These are small,
sequential measurements under ordinary machine load, not a controlled performance comparison.
Replay excludes live capture, control-thread scheduling and application rendering.

The live 5 ms target is not met. A flight-trained readout does not become a cursor controller
just because inference is fast. Proposal disagreement measures a single correction against
the reference, not closed-loop task error or completion quality. Report-level `passed` refers
to the deterministic reference checks; `shadow_score.promoted` remains false.

## Boundary and model adapter

- The runtime copies cursor, goal, speed, client bounds, observation age and reference point
  into an immutable sample. A separate worker holds at most one pending sample, replacing old
  work rather than allowing a backlog. It has no reference to the output worker or runtime.
- Predictions, raw neural outputs, inference time, input samples and reset boundaries enter
  the bounded ledger. Halt/cancellation invalidates in-flight observations. Stale and expired
  results are discarded. An inference exception disables only the shadow worker.
- Target error maps to the existing goal population, and cursor velocity maps to the haltere
  and Johnston's-organ channels. Other sensory channels, including optic flow, are zero.
  Flight outputs 0 and 1 are interpreted as hypothetical cursor velocities for comparison.
  Cursor-trained checkpoints declare adapter v2, scaling goal error by intent speed; old
  checkpoints and recordings retain adapter v1. Training provenance is included in the ledger.
  There is no deterministic correction added to those neural proposals.
- Each consumed sample advances the neural state by one 10 ms step. New intents/stages/layouts
  and observation gaps over 50 ms reset it. This is an explicit experimental time contract;
  matching neural time to variable live sampling remains unresolved.

## Reproduce

The optional environment requires torch with CUDA, the local Haltere package, and its graph
and checkpoint assets. The measured environment used Python 3.13.2 and torch 2.11.0+cu128.
The deterministic install and tests continue to work without them.

Run the live command **inside Anode**, using `seat_exec` or `seat_run`:

```powershell
.venv/Scripts/python.exe -m ganglion.cli reach-demo --environment arena --trials 3 --shadow-checkpoint C:/DEV/Haltere/artifacts/ftPath2_best.pt --json runs/shadow-seat.json
```

The evaluator launches and closes its owned Arena window. MCP transports only the agent's
commands and observations; the neural worker and deterministic controller run in the resident
process. A separately launched `core` accepts the same `--shadow-checkpoint` option.

Replay requires no desktop or input:

```powershell
.venv/Scripts/python.exe -m ganglion.brain.replay docs/bench/results/shadow-connectome-seat-recorded.json --checkpoint C:/DEV/Haltere/artifacts/ftPath2_best.pt --out runs/shadow-replay.json
```

## Supervised authority

Later on 2026-09-17 the shadow gained a bounded way to act. An intent may name
`controller: "connectome"`; the core then consults the newest valid proposal for that intent,
stage and layout (at most 50 ms old) on every control tick and applies its velocity **only if**
the resulting step, clamped to the intent speed limit and the client area, brings the cursor
closer to the goal (or holds position within tolerance). Otherwise the deterministic controller
acts for that tick and the override is counted. Completion still requires measured cursor
arrival; every pointer command records which controller produced it; a core without a loaded
model refuses the request. The regression tests drive the supervision with fake proposers that
help, sabotage, or are ignored, and a stalled model still cannot change actual commands.

| Measurement | Synthetic reach, 4 trials | Synthetic drag, 10 cases | Live Solitaire, 33 drags |
|---|---:|---:|---:|
| Task outcome | 4/4 completed, 0 false actions | 10/10 cases passed | 22 accepted by the game |
| Connectome commands / overridden | 108 / 33 (76.6%) | 451 / 169 (72.7%) | 3,778 / 394 (90.6%) |
| Inference p50 / p95 / p99 | 4.21 / 8.50 / 10.28 ms | — | 3.40 / 8.31 / 19.90 ms |
| Within 5 ms | 58.6% | — | 76.6% |
| Mean proposal/reference disagreement | 3.87 px | — | 1.82 px |

Sources: [reach](results/solitaire-live/neural-reach-synthetic.json),
[drag](results/solitaire-live/neural-drag-synthetic.json), and the
[Solitaire runs](SOLITAIRE.md#real-time-play-through-the-reactive-loop-second-pass-2026-09-17).
The model is the DAgger round-3 cursor readout from [training](TRAINING.md), which settles only
3–4 of 16 static targets on its own; the envelope, not the model, guarantees completion. The
share therefore measures how often the fly model's proposal was acceptable, not that it could
have finished the reach unaided. The live runs used the deterministic taught colour perception
for goals; there is still no fly visual front-end. `promoted` remains false everywhere.

Reproduce (the live command inside Anode):

```powershell
.venv/Scripts/python.exe -m ganglion.cli reach-demo --environment synthetic --trials 4 --shadow-checkpoint runs/cursor-dagger-v1/round-03/cursor-readout.pt --controller connectome --json runs/neural-reach.json
.venv/Scripts/python.exe -m ganglion.cli drag-demo --environment synthetic --trials 2 --shadow-checkpoint runs/cursor-dagger-v1/round-03/cursor-readout.pt --controller connectome --json runs/neural-drag.json
```

## Evidence age and neural time

Two timing faults came out of the live sessions. A proposal stayed usable for 50 ms counted
from the end of inference, so one computed from an 80 ms-old observation could still drive a
step; reuse is now bounded by the age of the observation the proposal was computed from, and
steps with no proposal fresh enough are counted apart from steps the envelope rejected
(`stale_commands`, controller `deterministic_stale`, in intent status and ledger shares). The
adapter also stepped the network once per consumed sample whatever the wall time between
samples, so neural time ran at the sampling rate; it now advances the network by
round(elapsed / 10 ms) steps, at most five, holding the observation, and the ledger records
`neural_steps`. Regression tests pin both behaviours.

## First person

The same envelope now drives a first-person view: [Half-Life](HALFLIFE.md) drags the align
program's virtual cursor and goal through the connectome each tick. Across the recorded pilot
session the model produced 80.4% of the view commands (5,071 against 1,233 overrides), fired
15 aligned bursts and killed four grunts without damage taken; its share fell against targets
that strafed fast, where the envelope overrode most proposals.

## Controlled transfer in the seat

The Arena reach demo (a target moving on a sinusoid, eight paired trial seeds, 1,200 px/s,
6 px tolerance, click on arrival) ran inside the Anode seat with the deterministic
controller, with the DAgger v1 checkpoint under supervised authority, and then with the
all-motor-neuron readout (DAgger v1b) under the same authority, on the same machine and
Windows session. `ganglion.arena.compare` scores both from the ledger: time to
complete, time until the measured cursor first came within tolerance, mean error over the
whole reach (approach included), interventions (steps the envelope handed to the reference),
stale steps (no proposal fresh enough) and the accepted share.

| Controller | Completed | Hits | Mean time (s) | First within 6 px (median s) | Tracking error (mean px) | Interventions | Stale | Accepted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deterministic | 8/8 | 8/8 | 0.42 | 0.29 | 139.5 | n/a | n/a | n/a |
| connectome | 8/8 | 8/8 | 0.52 | 0.41 | 130.7 | 15.6% | 13.1% | 71.3% |
| connectome-all-motor | 8/8 | 8/8 | 0.42 | 0.29 | 151.4 | 16.4% | 29.5% | 54.1% |

Sources: [deterministic](results/transfer-seat-deterministic.json),
[connectome](results/transfer-seat-connectome.json),
[connectome, all motor neurons](results/transfer-seat-connectome-all-motor.json),
[comparison](results/transfer-seat.json).
Inference during the connectome run: p50 5.1 ms, p95 15.7 ms, p99 44.2 ms, with 13% of the
steps stale under the corrected freshness rule; lost ledger events 0 and 0.
The live picture matches the suite: the envelope completes every reach either way, the
model's share of steps says how often it was allowed to act, and the time-to-tolerance and
tracking-error columns say what that cost. The all-motor readout reached tolerance at a median
0.29 s against 0.41 s for the
512-neuron readout and 0.29 s for the reference, with
16.4% interventions and 29.5% stale steps: its inference ran at p50 4.9 ms but p95 30 ms,
so the corrected freshness rule handed nearly a third of the steps to the reference, and the
reference's pace is what the table shows.

## Validation and next work

The full suite passes 97 tests including the three optional browser checks. New regressions
verify that even a stalled model proposing out-of-bounds points cannot change actual commands,
that cancellation discards an in-flight result, that inference failures remain isolated, that
the mailbox stays bounded, and that frozen replay preserves reset boundaries. PNG observation
round trips and encoded-response bounds are also tested. Locked resolution and wheel build pass.

Next, measure and reduce live scheduling/inference jitter, train generic cursor/track readouts
against the existing controller with held-out trajectories, and compare actual closed-loop
performance in Arena before promotion. Fly visual perception remains a separate unfinished
component. Application transfer, including Solitaire, should evaluate these reusable abilities.

The owned Solitaire, Arena and core processes were closed after testing. Steam and its UI
helpers were restored to main session 1. All live input in this work stayed in Anode.
