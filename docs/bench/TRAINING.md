# Generic cursor training

Measured 2026-09-17. These experiments train Haltere's actual 30,000-neuron,
2,767,698-edge ConnectomeRNN for cursor corrections. They use no card recognition,
game rules, application coordinates, screenshots, or direct sensory-to-output bypass.
All candidates remain in shadow mode; training does not send Windows input.

The controller still gets a goal and measured cursor position from deterministic
perception. This is motor adaptation, not a trained fly visual front-end. A checkpoint's
`desktop_trained` flag records that cursor training happened, not that it passed a
quality gate. No Ganglion model has been released on Hugging Face yet.

## Task and measurements

`ganglion.train.cursor_world` generates independent seeded episodes with varied screen
dimensions, intent speed (400–1,800 px/s), actuator gain (0.6–1.4), input delay
(0–60 ms), initial reach distance (2–400 px), and static or moving targets. Near-target
episodes teach settling. Moving speeds are limited by the sampled plant's capability.
The plant queues absolute coordinates computed at issue time. Gain is a deliberate
synthetic disturbance, not a measured model of Windows absolute input.

Adapter v2 normalises goal error by 0.3 seconds of intent speed and observed cursor
velocity by intent speed. Training and runtime use the same channel values; tests
check their agreement. Fly populations receive these values through Haltere's existing
encoders. Motor-neuron rates feed the learned readout. Recurrence, signs, neuron gains,
biases and time constants stay fixed in all experiments below.

The existing speed-limited proportional controller supplies imitation labels. An
independent 24→64→64→2 MLP provides a non-connectome comparison using the same sensor
inputs and labels. Feature selection and normalisation use training data only.
Validation chooses candidates; the final candidate is then evaluated on test episodes.
Each evaluation runs for 2,000 ticks (20 simulated seconds). Static success requires
error at most 6 px throughout the last ten ticks. Terminal error averages the last
30 ticks over all 32 test episodes. This generous horizon does not establish fast
reaching; settling time, target jumps, occlusion and variable sampling remain untested.

## Readout results

| Experiment / test seeds | Controller | Static targets settled | Mean terminal error |
|---|---|---:|---:|
| Readout v2 / 3000–3031 | Proportional teacher | 16 / 16 | 1.97 px |
| Readout v2 / 3000–3031 | MLP | 12 / 16 | 7.00 px |
| Readout v2 / 3000–3031 | Connectome | 0 / 16 | 510.48 px |
| DAgger v1 / 4000–4031 | Proportional teacher | 16 / 16 | 2.41 px |
| DAgger v1 / 4000–4031 | MLP | 16 / 16 | 3.37 px |
| DAgger v1 / 4000–4031 | Connectome | 3 / 16 | 260.63 px |

Sources: [readout v2](results/cursor-readout-v2.json),
[DAgger v1](results/cursor-dagger-v1.json). Different rows from different experiments
use different test seeds; compare controllers within an experiment. These small test
sets do not establish generalisation or a connectome advantage.

The first fit uses 12,800 teacher samples, selects 512 responsive motor neurons, and
fits a regularised readout. Its held-out imitation MSE is 0.01890, but its own cursor
trajectories diverge. Recorded-action agreement is therefore insufficient. DAgger adds
states reached by the current model, labelled by the teacher, growing the cache to
51,200 samples over three rounds. On the same validation split, static success rises
from 0/8 to 2/8 and terminal error falls from 376.73 to 289.22 px across those rounds.
The resulting checkpoint still fails most test episodes. Peak GPU temperatures were
42°C and 43°C respectively.

An earlier local `cursor-readout-v1` run is invalid for performance comparison. Its
simulator applied gain relative to the arrival-time cursor, inventing an extra feedback
loop, and used a shorter horizon for long reaches. The corrected plant applies gain
once at issue time; a regression checks delayed coordinates and teacher convergence.
The invalid local checkpoint is not a candidate or a published model.

## Encoder training

`cursor_finetune` backpropagates through the real recurrent network into sensory
encoders and the motor readout. It uses 16 parallel episodes, 32-step updates, state
detachment every eight steps, fixed motor whitening, gradient clipping, and a gradual
increase in model-driven trajectories. Model selection uses validation static success,
then terminal error. The current trainer includes the unchanged source checkpoint as
a candidate, preventing a worse training result from silently replacing it.

The first 200-update run reset the readout and used encoder/readout learning rates
1e-4/1e-3. Loss rose as model-driven trajectories increased, and none of its four
validation checkpoints settled a static target. Its selected checkpoint settled 0/16
test targets, with 606.36 px mean terminal error and a 45°C peak. The
[failed experiment](results/cursor-finetune-v1.json) is retained. Its report predates
configurable learning rates and baseline candidate selection; reproduce its update
settings with `--reset-readout --encoder-lr 1e-4 --readout-lr 1e-3` from the readout-v2
checkpoint.

The second run retained the DAgger head and used learning rates 1e-6/1e-5 for another
200 updates (102,400 labelled observations). None of its four checkpoints beat the
unchanged source on validation, so selection correctly retained DAgger round 3.
On fresh test seeds 6000–6031 that same source settled **4/16** static targets, with
272.39 px mean terminal error; the teacher settled **16/16** with 3.07 px error.
This 4/16 result uses a different test set from the earlier 3/16 result and is not an
improvement from fine-tuning. The run took 231.42 seconds and peaked at 46°C.
[Full report](results/cursor-finetune-v2.json).

A [checkpoint comparison](results/cursor-parameter-verification.json) confirms that
only 15 encoder/readout tensors changed in the final fine-tuned candidate; all other
stored tensors, including recurrent parameters and normalisation statistics, stayed
identical. Every tensor is finite. The real CUDA checkpoint also passes the optional
training/runtime adapter parity and reset test. The CPU suite passes 103 tests, with
four optional checks skipped; locked resolution and wheel construction pass.

The retained local candidate is `runs/cursor-dagger-v1/round-03/cursor-readout.pt`,
SHA-256 `c455ec5387de8dbce0ab1039581d7462e6c26f107c9dec964172b59755551f22`.
It is experimental and has no input authority. The next useful experiments should
address long-horizon feedback stability and measure settling time, then assess whether
constrained recurrent adaptation helps. Fine-tuning currently resets training episodes
after 192 ticks, much shorter than evaluation; longer on-policy trajectories are a
specific distribution gap to investigate. More imitation samples alone have not
established stable control or an advantage over the small MLP.

## Fixed suite: settling, jumps, pursuit and camera tracking

`ganglion.train.suite` runs four controllers over identical seeded episodes (32 per task,
2,000 ticks of 10 ms): the proportional reference (`teacher`), a 24→64→64→2 MLP trained on
the same 51,200 DAgger samples as the connectome readout, the connectome readout on its own,
and the same connectome under the runtime's supervising envelope with the reference acting
whenever a proposal is rejected (`supervised`). *Settle* starts 2–400 px from a static target
and succeeds when the error stays within 6 px for the final half second. *Jump* moves the
target every 2.5 s by 40 px up to a plant-dependent cap and scores each of the 256 reaches.
*Pursuit* follows a bouncing target and succeeds when the mean error after the first second is
within 12 px. *Camera* is pursuit in an unbounded view with two extra ticks of capture latency,
where a target more than half a frame away is lost. Settling time is the first tick after which
the error never leaves tolerance; tracking error is the mean after the first second;
interventions are ticks on which the envelope rejected the proposal and the reference acted.

| Task | Controller | Success | Settling / acquisition (median ms) | Tracking error (mean px) | Interventions | Accepted |
|---|---|---:|---:|---:|---:|---:|
| settle | teacher | 32/32 | 285 | 2.2 | n/a | n/a |
| settle | mlp | 32/32 | 375 | 3.3 | n/a | n/a |
| settle | connectome | 3/32 | 1815 | 170.6 | n/a | n/a |
| settle | supervised | 32/32 | 685 | 3.4 | 20.4% | 79.6% |
| jump | teacher | 256/256 | 540 | 20.5 | n/a | n/a |
| jump | mlp | 256/256 | 600 | 22.1 | n/a | n/a |
| jump | connectome | 28/256 | 855 | 166.4 | n/a | n/a |
| jump | supervised | 228/256 | 990 | 29.8 | 24.5% | 75.5% |
| pursuit | teacher | 31/32 | 160 | 5.5 | n/a | n/a |
| pursuit | mlp | 29/32 | 190 | 6.6 | n/a | n/a |
| pursuit | connectome | 2/32 | 670 | 192.4 | n/a | n/a |
| pursuit | supervised | 23/32 | 300 | 11.9 | 31.0% | 69.0% |
| camera | teacher | 30/32 (lost 0) | 490 | 6.2 | n/a | n/a |
| camera | mlp | 28/32 (lost 0) | 550 | 7.3 | n/a | n/a |
| camera | connectome | 0/32 (lost 30) | 1060 | 137.6 | n/a | n/a |
| camera | supervised | 21/32 (lost 0) | 820 | 11.5 | 40.0% | 60.0% |

Source: [cursor-suite-v1](results/cursor-suite-v1.json) with the DAgger v1 round 3 checkpoint;
185 s on the RTX 4090, peak 44°C. Jump tracking error includes the reaches themselves.

Read across the rows: the connectome alone settles 3 of 32 static targets and loses 30 of 32
camera targets. Under supervision it completes every settle episode, but more slowly than the
reference on its own (685 ms against 285 ms median), with 20–40% of its proposals rejected, and
it tracks moving targets with about twice the reference's error. The MLP on the same data stays
within a few percent of the reference everywhere. So the connectome's share of accepted commands
(60–80%) measures how often the envelope lets it act, not a contribution: on these tasks the
model currently costs time and accuracy, and the envelope caps the cost. That is the number to
move before any more application work.

## Longer, kicked, model-driven trajectories

The fine-tuner's episodes were 192 ticks against 2,000-tick evaluations and its gradients
spanned 32 ticks; DAgger harvested 400-tick episodes. `cursor_world` now takes `jump_every`:
the target jumps 40 px up to the plant's reach cap, so a trajectory holds fresh reaches from
whatever state the model reached. `cursor_dagger` takes `--episode-steps`, `--kick-every`,
`--student-max` and `--seed-base`; `cursor_finetune` takes `--episode-ticks`, `--window`,
`--truncate`, `--kick-every` and `--student-max`.

The first run of the changed experiment (DAgger v2) added three rounds of 32 episodes of
1,000 ticks with a kick every 250 ticks and 40–70% model-driven episodes to the v1 cache
(147,200 samples). Validation static success went 0/8, 0/8, 1/8 across rounds against v1's
2/8, and on the same fresh test seeds 4000–4031 the selected round settled **1/16** static
targets with 233.1 px mean terminal error against v1's 3/16 and 260.6 px. The teacher and MLP
settled 16/16 (2.4 and 2.9 px). Longer, kicked, more model-driven data did not improve a frozen
ridge readout; the v1 round 3 checkpoint stays the candidate. 176 s, peak 43°C.
[Full report](results/cursor-dagger-v2.json).

On the fixed suite, with the same MLP baseline and episodes as above, the v2 round 3
checkpoint scores ([cursor-suite-v2](results/cursor-suite-v2.json)):

| Task | Controller | Success | Settling / acquisition (median ms) | Tracking error (mean px) | Interventions | Accepted |
|---|---|---:|---:|---:|---:|---:|
| settle | connectome | 1/32 | 10230 | 104.9 | n/a | n/a |
| settle | supervised | 32/32 | 965 | 3.7 | 17.2% | 82.8% |
| jump | connectome | 9/256 | 930 | 116.1 | n/a | n/a |
| jump | supervised | 225/256 | 1000 | 29.6 | 20.4% | 79.6% |
| pursuit | connectome | 2/32 | 710 | 111.7 | n/a | n/a |
| pursuit | supervised | 20/32 | 360 | 11.2 | 31.1% | 68.9% |
| camera | connectome | 1/32 (lost 28) | 1390 | 113.0 | n/a | n/a |
| camera | supervised | 16/32 (lost 0) | 755 | 14.8 | 37.1% | 62.9% |

## Reproduce

Use a CUDA-enabled Python environment with Haltere installed and its graph/checkpoint
assets available. These experiments used Python 3.13.2, torch 2.11.0+cu128 and an RTX
4090. The slim checkpoints still depend on Haltere's graph files; they are not portable
standalone downloads. The base flight checkpoint SHA-256 appears in each report.

```powershell
.venv/Scripts/python.exe -m ganglion.train.cursor_readout --checkpoint C:/DEV/Haltere/artifacts/ftPath2_best.pt --out runs/cursor-readout-v2 --episodes 64 --steps 200 --features 512 --seconds 600 --max-gpu-temp 65
.venv/Scripts/python.exe -m ganglion.train.cursor_dagger --checkpoint runs/cursor-readout-v2/cursor-readout.pt --features runs/cursor-readout-v2/features.pt --out runs/cursor-dagger-v1 --rounds 3 --neurons 512 --seconds 600 --max-gpu-temp 65
.venv/Scripts/python.exe -m ganglion.train.cursor_finetune --checkpoint runs/cursor-dagger-v1/round-03/cursor-readout.pt --out runs/cursor-finetune-v2 --iterations 200 --encoder-lr 1e-6 --readout-lr 1e-5 --test-seed-start 6000 --seconds 900 --max-gpu-temp 65
.venv/Scripts/python.exe -m ganglion.train.cursor_dagger --checkpoint runs/cursor-dagger-v1/round-03/cursor-readout.pt --features runs/cursor-dagger-v1/features.pt --out runs/cursor-dagger-v2 --rounds 3 --episode-steps 1000 --kick-every 250 --student-max 0.9 --seed-base 30000 --seconds 1700 --max-gpu-temp 65
.venv/Scripts/python.exe -m ganglion.train.suite --checkpoint runs/cursor-dagger-v1/round-03/cursor-readout.pt --mlp-features runs/cursor-dagger-v1/features.pt --out runs/suite-v1 --seconds 1500
```

Output directories must not already exist. Run one GPU job at a time. Each command has
a wall-time budget, pauses above the selected temperature until it drops by 8°C, and
stops if temperature telemetry is missing. A stopped run preserves earlier checkpoints;
there is no automatic scheduler or unattended continuation. Optimizer snapshots are
saved during fine-tuning, but the current command resumes model weights with a fresh
optimizer. `runs/`, weights, optimizer state and feature caches stay outside Git.

The optional `tests/test_cursor_checkpoint.py` check compares real checkpoint outputs
through the training path and resident adapter, including a state reset. Set
`GANGLION_CURSOR_CHECKPOINT` to a cursor checkpoint and run it in the CUDA environment.
The ordinary regression suite remains independent of torch, Haltere and model assets.

Before a Hugging Face release, a candidate needs stable held-out control, live shadow
validation in Anode, measured latency, a portable graph dependency, and a model card
documenting data provenance, terms and limitations. Source code is MIT; external assets
retain their own terms. The deterministic runtime remains the only input controller.
