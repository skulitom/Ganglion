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
