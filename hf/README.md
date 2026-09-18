---
license: mit
library_name: ganglion
tags:
  - connectome
  - drosophila
  - reflex
  - cursor-control
  - haltere
pipeline_tag: reinforcement-learning
---

# Ganglion cursor readouts on the Haltere connectome

Two checkpoints of the [Haltere](https://github.com/skulitom/Haltere) fly-brain connectome
(a 30,000-neuron recurrent network with the connectome's structure and signs, flight-trained)
with a linear motor readout that turns the network's motor-neuron rates into a cursor or view
velocity, trained by imitation of a proportional controller in [Ganglion](https://github.com/skulitom/Ganglion),
the reflex layer that runs them against live applications at 100 Hz.

| File | What it is | Where it stands |
|---|---|---|
| `cursor-readout-v6.pt` | Adapter version 5 (the goal as a unit direction plus a tanh distance, goal scale 0.1), all 3,913 motor neurons, three DAgger rounds | The live configuration. Under the reflex layer's supervising envelope it matches the reference controller on the fixed suite (settle 32/32 in 335 ms, jump 256/256, pursuit 31/32 at 5.1 px, camera 32/32 at 3.7 px) and, live in Half-Life, equals it on time to the first shot and tracking while firing on fewer of its intents. Alone it cannot stop: it holds speed through the goal and oscillates (0/32 settled). |
| `cursor-readout-v3b.pt` | Adapter version 3 (goal error over 0.3 s of intent speed), all motor neurons, one DAgger round | The checkpoint that stands on its own: alone it settles 19/32 static targets at 14.8 px; under supervision settle 32/32 in 375 ms with the envelope intervening on 1.9% of steps. Slower to the goal than v6. |

The `report-*.json` files beside them are the training and suite reports they were selected
from. Everything here was measured on one machine (RTX 4090, Windows 11); the numbers are the
suite's and the live harness's, with their caveats, and are documented in full in the
repository's [TRAINING.md](https://github.com/skulitom/Ganglion/blob/main/docs/bench/TRAINING.md)
and [HALFLIFE.md](https://github.com/skulitom/Ganglion/blob/main/docs/bench/HALFLIFE.md).

## What the model does and does not do

The network is frozen. The readout is a ridge regression from the motor neurons' rates to a
two-dimensional velocity in units of the intent's speed, fitted on synthetic cursor episodes
(random plants with input delay, gain and speed) to the commands of a memoryless proportional
controller, then refined with DAgger on the student's own rollouts. The runtime feeds the
network the goal error and, for version 6 checkpoints, nothing else; the readout's proposal
is accepted by a supervising envelope only if it moves the cursor toward the goal within the
intent's speed limit, otherwise the deterministic reference acts for that tick.

Measured, not claimed:

- Under supervision, v6 reaches its first shot in a live Half-Life fight as fast as the
  reference controller at the same step limit and tracks the target as tightly (thirty clean
  trials each, one grunt: first shot 1.00 against 0.95 s median, p 0.61; tracking error 134
  against 140 px, p 0.76), but fires on fewer of its align intents (58% against 75%,
  p 0.003). It does not make the chain faster than the reference.
- Alone, no checkpoint here beats the reference. v3b settles 19/32; v6 settles none. A linear
  readout of this network can hold speed or stop, not both: the network answers a reversal of
  the goal direction in 60 to 70 ms and its state after a fast approach is not linearly
  distinguishable from its state during it.
- Every live trial before 15:00 on 2026-09-18 ran with a stray virtual controller holding a
  move-backward input in the game; the comparisons between conditions stand, the absolute
  figures describe a fight fought against a wall.

## Loading

The checkpoints are standard `torch.save` files of the Haltere `ConnectomeRNN` with the
readout installed and a `ganglion_cursor` metadata record (adapter version, goal scale, the
training options and the seeds). They need the `haltere` package to load and CUDA to run in
real time:

```python
from ganglion.brain.haltere_cursor import HaltereCursor   # from the Ganglion repository
model = HaltereCursor("cursor-readout-v6.pt")              # reads the adapter version and goal scale from the file
```

`ganglion core --shadow-checkpoint cursor-readout-v6.pt --shadow-process` runs it as the
reflex layer's shadow predictor; `ganglion.train.suite` scores it on the fixed suite.

## Provenance

Base checkpoint: Haltere `ftPath2_best.pt` (the flight-trained connectome). Training scripts:
`ganglion.train.cursor_readout` and `ganglion.train.cursor_dagger` at the commits recorded in
the reports. License: MIT, as the repositories.
