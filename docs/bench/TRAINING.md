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

Adapter v2 normalises goal error by 0.3 seconds of intent speed (the checkpoint's `goal_scale`
since v5) and observed cursor velocity by intent speed. Training and runtime use the same channel values; tests
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

## Gradient fine-tuning selected on the suite

`cursor_finetune` now selects candidates on the fixed suite (settling and pursuit on validation
episodes 40000–40015 and 41000–41015, the connectome acting alone, 1,000 ticks) instead of the
old terminal-error score, keeps the unchanged source as a candidate, and scores the selected
checkpoint on the suite's test episodes alone and under supervision. `--train` chooses the
parameter groups: `encoders` and `readout` as before, `edges` for the 2,767,698 per-edge
log-gains under the connectome prior (structure and signs stay fixed), `neurons` for the
per-neuron gains, biases and time constants. Learning rates are per group.

Two runs from the DAgger v1 checkpoint used 1,000-tick episodes with kicks every 250 ticks,
96-tick updates truncated every 32 ticks, up to 90% model-driven episodes and 300 updates.
Encoders and readout at 1e-5 and 1e-4 ([v3](results/cursor-finetune-v3.json)): the imitation
loss rose from 0.042 to 0.43 with gradient norms near 12 before clipping, validation static
success fell from 3/16 to 0/16, and selection kept the source (465 s, peak 49°C). Adding the
edge gains at 1e-4 ([v4](results/cursor-finetune-v4.json)) did the same: loss 0.042 to 0.36,
0–1/16, source kept (494 s, peak 62°C). The scales explain it: the ridge readout's weights
average 0.02, so an Adam rate of 1e-4 rewrites them within a few hundred steps whatever the
gradient says, and the gradients through 32 ticks of the 30,000-neuron recurrence explode, so
the rewrite is noise. Smaller rates (the earlier 1e-6/1e-5 run) leave the least-squares optimum
where it is. Gradient training through this network, as set up, is not the lever; the
representation the readout sees is.

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

## What the readout sees: dropping the own-velocity input

The reference controller is memoryless: its command is a function of the goal error alone.
The own-velocity channels (haltere and Johnston's organ analogues) that adapter v2 feeds the
network are therefore not evidence about the goal; in teacher-driven imitation data they are a
lagged copy of the teacher's own action, a shortcut a readout can lean on that fails in closed
loop. Adapter **v3** is v2 without those channels; the training world, the runtime adapter and
the suite all speak it, and a checkpoint records which version it was trained on.

Trained the same way as the v2 lineage (readout on 12,800 teacher samples, then three DAgger
rounds), the v3 readout settled 6/16 held-out static targets straight after the first fit
where v2 settled 0/16, and its DAgger rounds validated at 5/8, 4/8 and 4/8 against v1's 0/8,
1/8 and 2/8 ([readout v3](results/cursor-readout-v3.json), [DAgger v3](results/cursor-dagger-v3.json)).
On the fixed suite, with an MLP trained on the same v3 samples
([cursor-suite-v3](results/cursor-suite-v3.json)):

| Task | Controller | Success | Settling / acquisition (median ms) | Tracking error (mean px) | Interventions | Accepted |
|---|---|---:|---:|---:|---:|---:|
| settle | mlp | 32/32 | 320 | 3.0 | n/a | n/a |
| settle | connectome | 12/32 | 555 | 50.4 | n/a | n/a |
| settle | supervised | 32/32 | 1070 | 5.6 | 0.4% | 99.6% |
| jump | connectome | 44/256 | 400 | 141.3 | n/a | n/a |
| jump | supervised | 197/256 | 1030 | 28.5 | 6.1% | 93.9% |
| pursuit | mlp | 31/32 | 180 | 5.8 | n/a | n/a |
| pursuit | connectome | 1/32 | 390 | 178.1 | n/a | n/a |
| pursuit | supervised | 10/32 | 325 | 19.1 | 17.5% | 82.5% |
| camera | mlp | 31/32 (lost 0) | 495 | 6.4 | n/a | n/a |
| camera | connectome | 0/32 (lost 30) | 1200 | 115.0 | n/a | n/a |
| camera | supervised | 8/32 (lost 0) | 1525 | 22.3 | 24.8% | 75.2% |

Against the v1 rows above, the velocity-free readout settles four times as many static targets
on its own (12/32 against 3/32) with a third of the error, and the envelope almost never has
to reject it (0.4% against 20%); but it is slow, so the supervised settling median rises from
685 to 1,070 ms, and on moving targets it is worse both alone and supervised (pursuit 10/32
against 23/32, camera 8/32 against 21/32). The MLP on the same velocity-free inputs matches the
reference on every task. So the shortcut was real and removing it helps settling, while the
readout still lacks the gain and the anticipation that tracking needs. Neither lineage is a
candidate for promotion; both are recorded.

The third velocity-free DAgger round, which the old selection passed over for round 1 despite
its lower terminal error, scores on the suite ([cursor-suite-v3r3](results/cursor-suite-v3r3.json)):

| Task | Controller | Success | Settling / acquisition (median ms) | Tracking error (mean px) | Interventions | Accepted |
|---|---|---:|---:|---:|---:|---:|
| settle | connectome | 18/32 | 395 | 54.5 | n/a | n/a |
| settle | supervised | 32/32 | 380 | 3.7 | 4.5% | 95.5% |
| jump | connectome | 33/256 | 690 | 147.8 | n/a | n/a |
| jump | supervised | 226/256 | 665 | 23.0 | 8.5% | 91.5% |
| pursuit | connectome | 4/32 | 280 | 135.4 | n/a | n/a |
| pursuit | supervised | 20/32 | 225 | 11.5 | 16.8% | 83.2% |
| camera | connectome | 2/32 (lost 28) | 575 | 85.6 | n/a | n/a |
| camera | supervised | 18/32 (lost 0) | 545 | 12.3 | 21.2% | 78.8% |

## Reading out all motor neurons

The v2 lineage's readout uses 512 motor neurons chosen by label correlation. Refitting the
DAgger v1 cache plus one fresh round on all 3,913 motor neurons with the lightest ridge
(1e-5) improves the imitation fit (validation MSE 0.0141 against 0.0244) and, on its own,
controls worse: validation static success 0/8 and fresh held-out 0/16 with 160 px terminal
error against 3/16 for the 512-neuron readout ([DAgger v1b](results/cursor-dagger-v1b.json)).
On the suite ([cursor-suite-v1b](results/cursor-suite-v1b.json)):

| Task | Controller | Success | Settling / acquisition (median ms) | Tracking error (mean px) | Interventions | Accepted |
|---|---|---:|---:|---:|---:|---:|
| settle | connectome | 0/32 | 19945 | 80.5 | n/a | n/a |
| settle | supervised | 32/32 | 460 | 2.9 | 7.0% | 93.0% |
| jump | connectome | 8/256 | 1650 | 100.1 | n/a | n/a |
| jump | supervised | 255/256 | 660 | 23.3 | 18.1% | 81.9% |
| pursuit | connectome | 4/32 | 225 | 106.8 | n/a | n/a |
| pursuit | supervised | 28/32 | 210 | 7.6 | 34.9% | 65.1% |
| camera | connectome | 1/32 (lost 26) | 580 | 98.5 | n/a | n/a |
| camera | supervised | 27/32 (lost 0) | 570 | 8.0 | 42.0% | 58.0% |

Alone, this readout is fast and imprecise: it settles nothing but tracks with half the error of
the 512-neuron readout and acquires moving targets sooner. Under the envelope that is the best
supervised controller measured so far: settling 460 ms median against 685 ms (reference 285),
every jump but one, pursuit 28/32 with 7.6 px against 23/32 with 11.9 px (reference 5.5), camera
27/32 with 8.0 px against 21/32 with 11.5 px. The envelope's share of the work is visible in the
interventions column (7% on settle, 35–42% on moving targets), so this is a supervised result,
not autonomy; but it is the first change that moved the number the runtime actually uses.

## Both together: velocity-free inputs, all motor neurons

Refitting the velocity-free DAgger cache plus one fresh round on all 3,913 motor neurons
([DAgger v3b](results/cursor-dagger-v3b.json)) settles **10/16** fresh held-out static targets
with 73 px terminal error, against 3/16 for either change alone and 0/16 for the all-motor
readout with velocity inputs. On the suite ([cursor-suite-v3b](results/cursor-suite-v3b.json)):

| Task | Controller | Success | Settling / acquisition (median ms) | Tracking error (mean px) | Interventions | Accepted |
|---|---|---:|---:|---:|---:|---:|
| settle | connectome | 19/32 | 780 | 14.8 | n/a | n/a |
| settle | supervised | 32/32 | 375 | 3.7 | 1.9% | 98.1% |
| jump | connectome | 106/256 | 875 | 52.5 | n/a | n/a |
| jump | supervised | 241/256 | 650 | 22.8 | 6.1% | 93.9% |
| pursuit | connectome | 10/32 | 280 | 52.6 | n/a | n/a |
| pursuit | supervised | 24/32 | 215 | 10.1 | 6.1% | 93.9% |
| camera | connectome | 1/32 (lost 26) | 560 | 77.4 | n/a | n/a |
| camera | supervised | 18/32 (lost 0) | 550 | 11.4 | 19.0% | 81.0% |

Alone, this is the first readout that settles more than half the static targets (19/32) with a
tracking error near the tolerance (14.8 px), reaches 106 of 256 jumps and 10 of 32 pursuits.
Under the envelope it settles fastest of all candidates (375 ms median against the reference's
285) with 2% interventions, and it trails the velocity-fed all-motor readout only on moving
targets (pursuit 24/32 at 10.1 px against 28/32 at 7.6 px; camera 18/32 against 27/32), where a
velocity input evidently helps. Two candidates remain: this one for settling and for acting on
its own, the velocity-fed all-motor readout for supervised tracking of moving targets.

Two more DAgger rounds on that readout with 1,000-tick episodes, kicks every 250 ticks and up
to 90% model-driven episodes ([DAgger v3c](results/cursor-dagger-v3c.json); validation round 1 5/8 static, 171 px terminal; round 2 5/8 static, 303 px terminal;
round-01 selected) settle **7/16** fresh held-out static
targets with 149 px terminal error. On the suite
([cursor-suite-v3c](results/cursor-suite-v3c.json)):

| Task | Controller | Success | Settling / acquisition (median ms) | Tracking error (mean px) | Interventions | Accepted |
|---|---|---:|---:|---:|---:|---:|
| settle | connectome | 21/32 | 300 | 80.6 | n/a | n/a |
| settle | supervised | 32/32 | 345 | 4.0 | 2.5% | 97.5% |
| jump | connectome | 64/256 | 530 | 109.1 | n/a | n/a |
| jump | supervised | 228/256 | 690 | 23.0 | 6.5% | 93.5% |
| pursuit | connectome | 1/32 | 230 | 120.1 | n/a | n/a |
| pursuit | supervised | 22/32 | 230 | 10.6 | 15.0% | 85.0% |
| camera | connectome | 2/32 (lost 26) | 550 | 91.6 | n/a | n/a |
| camera | supervised | 18/32 (lost 0) | 440 | 12.1 | 23.4% | 76.6% |

More on-policy data with the stronger readout did not help either: settling alone rose to 21/32 but with 81 px error, moving targets got worse alone (jump 64/256, pursuit 1/32), and under the envelope every task moved by a few percent either way. The single refit (v3b) stays the candidate, and the pattern across v2, v3c and the gradient runs is consistent: with this frozen network, what the readout sees decides more than how much on-policy data it gets.

## Feeding the fly's motion channel: adapter v4

The lptc channel is the connectome's wide-field motion input, and until now training fed it
zeros. Adapter **v4** is v3 (goal error only) plus, for a *view* episode, what the flow percept
reports when the view turns: the image translation over the last 60 ms seen two frames late,
which is minus the view's own motion, in units of intent speed, plus the expansion and roll
rates (zero for a turn). The training world runs view episodes (unbounded, two frames of
capture latency, target starting inside the frame) beside cursor episodes; the harvest mixes
them half and half; the suite's camera task is a view episode; and `ganglion core
--lptc-from-flow` hands a v4 checkpoint the live flow summary, which older versions ignore.

Readout on 12,800 samples from all 3,913 motor neurons ([readout v4](results/cursor-readout-v4.json)):
6/16 held-out static targets with
213 px terminal error (MLP 16/16).
Three DAgger rounds ([DAgger v4](results/cursor-dagger-v4.json); validation 4/8; 4/8; 4/8; round-02 selected):
8/16 with 159 px.
On the suite, with an MLP trained on the same v4 samples ([cursor-suite-v4](results/cursor-suite-v4.json)):

| Task | Controller | Success | Settling / acquisition (median ms) | Tracking error (mean px) | Interventions | Accepted |
|---|---|---:|---:|---:|---:|---:|
| settle | connectome | 18/32 | 390 | 60.7 | n/a | n/a |
| settle | supervised | 32/32 | 490 | 4.2 | 3.4% | 96.6% |
| jump | connectome | 51/256 | 580 | 159.9 | n/a | n/a |
| jump | supervised | 225/256 | 750 | 24.1 | 7.2% | 92.8% |
| pursuit | connectome | 3/32 | 230 | 201.4 | n/a | n/a |
| pursuit | supervised | 20/32 | 225 | 12.4 | 17.1% | 82.9% |
| camera | connectome | 3/32 (lost 29) | 575 | 87.9 | n/a | n/a |
| camera | supervised | 30/32 (lost 0) | 505 | 7.5 | 25.0% | 75.0% |

A second question the channel lets us ask: does the flight-trained network already do
something with visual slip? The v3b checkpoint, whose readout never saw a non-zero lptc, scored
on the camera task with the slip fed in ([cursor-suite-v3b-slip](results/cursor-suite-v3b-slip.json))
against its own rows above:

| Task | Controller | Success | Settling / acquisition (median ms) | Tracking error (mean px) | Interventions | Accepted |
|---|---|---:|---:|---:|---:|---:|
| camera | connectome | 0/32 (lost 32) | 600 | 247.6 | n/a | n/a |
| camera | supervised | 31/32 (lost 0) | 490 | 4.4 | 96.5% | 3.5% |

Training with the slip gives the best supervised camera tracking so far (30/32 at 7.5 px, with the model acting on three quarters of the steps) at a cost on the cursor tasks alone, where half the harvest is now view episodes (settle 18/32 at 61 px against v3b's 19/32 at 15 px). The untrained network has no usable innate response to slip through the readout: fed to v3b, the slip makes it lose every camera target alone and the envelope rejects 96% of its proposals, so the near-reference row there is the reference's work. Both of the fly's self-motion channels now exist in training and at runtime; which one a task should get, and whether the live flow matches the simulated slip, is the next measurement, on a turning view in the seat.

## Slip robustness: adapter v4b

The live channel is not the simulated slip: it is absent when the percept finds it not credible
and it reports a scene-dependent share of the true motion, and in forty Half-Life engagements
the v4 readout was overridden four times as often with the channel fed than with zeros
([HALFLIFE.md](HALFLIFE.md)). The training world therefore gained three options: a share of
view episodes without the slip (`--slip-dropout`), single ticks where it blanks
(`--slip-blank`), and a per-episode gain drawn from a range (`--slip-gain`). **v4b** is v4
trained with dropout 0.3, blanking 0.05 and gain
0.7 to 1.0, the same harvest and DAgger schedule otherwise.

Readout ([readout v4b](results/cursor-readout-v4b.json)): 5/16
held-out static targets with 203 px terminal error (v4 at the same stage: 6/16 at 213 px).
DAgger ([DAgger v4b](results/cursor-dagger-v4b.json); validation 4/8; 3/8; 3/8; round-01 selected):
3/16 with 376 px (v4 after DAgger: 8/16 at 159 px).
The suite with the slip fed as trained ([cursor-suite-v4b](results/cursor-suite-v4b.json)), and the
same checkpoint scored with the channel absent (`--sense-version 3`,
[cursor-suite-v4b-noslip](results/cursor-suite-v4b-noslip.json)), against v4 with the slip:

| Task | Controller | v4b, slip fed | v4b, channel absent | v4, slip fed |
|---|---|---:|---:|---:|
| settle | connectome | 13/32 at 211.9 px | 13/32 at 211.3 px | 18/32 at 60.7 px |
| settle | supervised | 32/32 at 3.7 px | 32/32 at 3.7 px | 32/32 at 4.2 px |
| jump | connectome | 26/256 at 416.7 px | 26/256 at 410.3 px | 51/256 at 159.9 px |
| jump | supervised | 238/256 at 23.8 px | 239/256 at 23.8 px | 225/256 at 24.1 px |
| pursuit | connectome | 3/32 at 405.6 px | 3/32 at 405.6 px | 3/32 at 201.4 px |
| pursuit | supervised | 23/32 at 9.8 px | 25/32 at 9.8 px | 20/32 at 12.4 px |
| camera | connectome | 1/32 at 131.8 px | 0/32 at 131.5 px | 3/32 at 87.9 px |
| camera | supervised | 30/32 at 6.6 px | 28/32 at 8.1 px | 30/32 at 7.5 px |

Full rows for v4b with the slip:

| Task | Controller | Success | Settling / acquisition (median ms) | Tracking error (mean px) | Interventions | Accepted |
|---|---|---:|---:|---:|---:|---:|
| settle | connectome | 13/32 | 255 | 211.9 | n/a | n/a |
| settle | supervised | 32/32 | 465 | 3.7 | 11.6% | 88.4% |
| jump | connectome | 26/256 | 425 | 416.7 | n/a | n/a |
| jump | supervised | 238/256 | 750 | 23.8 | 14.5% | 85.5% |
| pursuit | connectome | 3/32 | 220 | 405.6 | n/a | n/a |
| pursuit | supervised | 23/32 | 205 | 9.8 | 28.4% | 71.6% |
| camera | connectome | 1/32 (lost 31) | 510 | 131.8 | n/a | n/a |
| camera | supervised | 30/32 (lost 0) | 455 | 6.6 | 37.5% | 62.5% |

The robustness options made the readout indifferent to the channel, not better at using it: under supervision the camera task scores 30/32 at 6.6 px with the slip and 28/32 at 8.1 px without, jump and pursuit match or edge past v4, but the model alone is weaker than v4 everywhere (settle 13/32 at 212 px against 18/32 at 61 px) and the envelope intervenes 1.5 to 3.4 times as often (settle 11.6% against 3.4%, jump 14.5% against 7.2%, pursuit 28.4% against 17.1%, camera 37.5% against 25.0%), so the outcomes are the envelope's more than the model's. A readout trained on a channel it cannot trust learns to discount it; making the live channel worth having is a matter of what the training world shows in it (independent movers, the percept's own failure modes), not of noise on the simulated slip. v4 remains the tracking candidate and v4b is not run live.

## A stronger input near the goal: goal scale 0.1

The adapter normalises the goal error by 0.3 s of intent speed, so at 50 px from a target the
network's goal input is 0.14 of its range and the readout's proposals fade as the cursor
closes, which is where the supervised model loses time to the reference (settling 465 to
490 ms against 285 ms on the suite; acquisition 2.13 s against 1.64 s live). The scale is now a
checkpoint parameter (`--goal-scale`, recorded and honoured by the runtime, DAgger and the
suite). **v5** is v3 (goal error only) with all motor neurons and a goal scale of 0.1, so the
input saturates beyond about 120 px and is three times stronger near the goal.

Readout ([readout v5](results/cursor-readout-v5.json)): 3/16 held-out static targets with
550 px terminal error (v3b after DAgger: 10/16 at 73 px). DAgger ([DAgger v5](results/cursor-dagger-v5.json);
validation 3/8; 2/8; 0/8; round-01 selected): 3/16 with 383 px.
The suite ([cursor-suite-v5](results/cursor-suite-v5.json)) against v3b and v4 (success at mean
tracking error, median settling or acquisition):

| Task | Controller | v5 (goal scale 0.1) | v5b (0.2) | v3b (0.3) | v4 (0.3, slip) |
|---|---|---:|---:|---:|---:|
| settle | teacher | 32/32 at 2.2 px, 285 ms | 32/32 at 2.2 px, 285 ms | 32/32 at 2.2 px, 285 ms | 32/32 at 2.2 px, 285 ms |
| settle | connectome | 5/32 at 234.1 px, 260 ms | 22/32 at 56.2 px, 450 ms | 19/32 at 14.8 px, 780 ms | 18/32 at 60.7 px, 390 ms |
| settle | supervised | 32/32 at 3.3 px, 340 ms | 32/32 at 3.6 px, 345 ms | 32/32 at 3.7 px, 375 ms | 32/32 at 4.2 px, 490 ms |
| jump | teacher | 256/256 at 20.3 px, 510 ms | 256/256 at 20.3 px, 510 ms | 256/256 at 20.3 px, 510 ms | 256/256 at 20.3 px, 510 ms |
| jump | connectome | 2/256 at 403.7 px, 650 ms | 86/256 at 93.5 px, 630 ms | 106/256 at 52.5 px, 875 ms | 51/256 at 159.9 px, 580 ms |
| jump | supervised | 241/256 at 22.8 px, 680 ms | 227/256 at 23.2 px, 660 ms | 241/256 at 22.8 px, 650 ms | 225/256 at 24.1 px, 750 ms |
| pursuit | teacher | 31/32 at 5.5 px, 160 ms | 31/32 at 5.5 px, 160 ms | 31/32 at 5.5 px, 160 ms | 31/32 at 5.5 px, 160 ms |
| pursuit | connectome | 0/32 at 339.9 px, 220 ms | 6/32 at 110.6 px, 225 ms | 10/32 at 52.6 px, 280 ms | 3/32 at 201.4 px, 230 ms |
| pursuit | supervised | 28/32 at 8.9 px, 170 ms | 22/32 at 11.3 px, 190 ms | 24/32 at 10.1 px, 215 ms | 20/32 at 12.4 px, 225 ms |
| camera | teacher | 32/32 at 4.0 px, 445 ms | 32/32 at 4.0 px, 445 ms | 30/32 at 6.2 px, 490 ms | 32/32 at 4.0 px, 445 ms |
| camera | connectome | 0/32 at 135.8 px, 525 ms | 0/32 at 114.6 px, 465 ms | 1/32 at 77.4 px, 560 ms | 3/32 at 87.9 px, 575 ms |
| camera | supervised | 31/32 at 6.6 px, 480 ms | 24/32 at 9.4 px, 450 ms | 18/32 at 11.4 px, 550 ms | 30/32 at 7.5 px, 505 ms |

A goal scale of 0.1 gives the fastest supervised settling on the suite so far (340 ms against 375 ms for v3b and 490 ms for v4, the reference at 285 ms), the fastest supervised pursuit acquisition (28/32 at 8.9 px in 170 ms; v1b matches the 28/32 at 7.6 px in 210 ms) and the best supervised camera row that is the model's own work (31/32 at 6.6 px; v3b fed the slip reached 31/32 at 4.4 px with the envelope on 96.5% of steps), but the model alone falls apart (settle 5/32 at 234 px against v3b's 19/32 at 14.8 px, jump and pursuit near zero) and the envelope intervenes on 13 to 35% of steps against v3b's 2 to 19%. The stronger input makes the readout propose larger steps near the goal, which the envelope keeps in bounds and the model alone cannot. v3b stays the checkpoint that stands on its own and v5 the one that settles fastest under supervision.

**v5b** is the same recipe at a goal scale of 0.2 ([readout v5b](results/cursor-readout-v5b.json):
7/16 held-out at 433 px; [DAgger v5b](results/cursor-dagger-v5b.json), validation 2/8; 3/8; 4/8,
round-03 selected: 7/16 at 137 px; [cursor-suite-v5b](results/cursor-suite-v5b.json)). At 0.2 the model keeps its feet: alone it settles 22/32 at 450 ms (v3b 19/32 at 780 ms, the most settled of any checkpoint), and under supervision it settles in 345 ms with the envelope intervening on 3.4% of steps (v5: 340 ms at 12.7%; v3b: 375 ms at 1.9%), so the settling gain of the stronger input survives at a scale where the readout still stands on its own. Camera sits between v3b and v5 (24/32 at 9.4 px under supervision); pursuit does not: 22/32 at 11.3 px is below both v3b (24/32 at 10.1 px) and v5 (28/32 at 8.9 px), though it acquires faster than v3b (190 ms against 215 ms). v5b is the settling candidate, v5 the tracking candidate under supervision, v3b the most autonomous on the moving tasks.

## The goal as a direction: adapter version 5

Live at matched speed ([HALFLIFE.md](HALFLIFE.md)) the supervised v4 reaches its first shot as
fast as the reference but fires on fewer of its intents: its unfired intents time out hovering
about 40 px from a moving target. Every readout's proposal magnitude shrinks with the
distance, from the intent speed far away to about half at 15 to 30 px and a fifth inside 15 px
(v4 0.20, v1b 0.48, v5b 0.36 in units of the intent speed), because the goal input, the tanh
of the error over 0.3 s of intent speed, is 0.05 there and a ridge readout cannot make a full
step from it. Adapter **version 5** encodes the goal as a unit direction at every distance
and puts the tanh distance (over `goal_scale` seconds of speed) in the fourth slot, so the
readout can hold speed to the goal and use the distance to stop. **v6** is version 5 with all
motor neurons and a goal scale of 0.1, the same harvest and DAgger schedule as before.

Readout ([readout v6](results/cursor-readout-v6.json)): 1/16 held-out static targets with
460 px terminal error. DAgger ([DAgger v6](results/cursor-dagger-v6.json); validation 0/8; 0/8; 0/8; round-03
selected): 0/16 with 75 px. The suite ([cursor-suite-v6](results/cursor-suite-v6.json)) against v5b and v3b:

| Task | Controller | v6 (direction, adapter 5) | v5b (goal scale 0.2) | v3b |
|---|---|---:|---:|---:|
| settle | teacher | 32/32 at 2.2 px, 285 ms | 32/32 at 2.2 px, 285 ms | 32/32 at 2.2 px, 285 ms |
| settle | connectome | 0/32 at 41.7 px, 19970 ms | 22/32 at 56.2 px, 450 ms | 19/32 at 14.8 px, 780 ms |
| settle | supervised | 32/32 at 2.4 px, 335 ms, 6% overridden | 32/32 at 3.6 px, 345 ms, 3% overridden | 32/32 at 3.7 px, 375 ms, 2% overridden |
| jump | teacher | 256/256 at 20.3 px, 510 ms | 256/256 at 20.3 px, 510 ms | 256/256 at 20.3 px, 510 ms |
| jump | connectome | 2/256 at 76.9 px, 2195 ms | 86/256 at 93.5 px, 630 ms | 106/256 at 52.5 px, 875 ms |
| jump | supervised | 256/256 at 22.3 px, 665 ms, 20% overridden | 227/256 at 23.2 px, 660 ms, 10% overridden | 241/256 at 22.8 px, 650 ms, 6% overridden |
| pursuit | teacher | 31/32 at 5.5 px, 160 ms | 31/32 at 5.5 px, 160 ms | 31/32 at 5.5 px, 160 ms |
| pursuit | connectome | 2/32 at 65.7 px, 240 ms | 6/32 at 110.6 px, 225 ms | 10/32 at 52.6 px, 280 ms |
| pursuit | supervised | 31/32 at 5.1 px, 200 ms, 49% overridden | 22/32 at 11.3 px, 190 ms, 21% overridden | 24/32 at 10.1 px, 215 ms, 6% overridden |
| camera | teacher | 32/32 at 4.0 px, 445 ms | 32/32 at 4.0 px, 445 ms | 30/32 at 6.2 px, 490 ms |
| camera | connectome | 2/32 at 98.4 px, 540 ms | 0/32 at 114.6 px, 465 ms | 1/32 at 77.4 px, 560 ms |
| camera | supervised | 32/32 at 3.7 px, 530 ms, 58% overridden | 24/32 at 9.4 px, 450 ms, 31% overridden | 18/32 at 11.4 px, 550 ms, 19% overridden |

The direction encoding removes the fading but not the stopping: alone the model never settles (0/32; it holds speed through the goal and oscillates around it, 42 px mean error), while under supervision it matches the teacher on every task, settle 32/32 in 335 ms at 2.4 px, jump 256/256 at 22.3 px, pursuit 31/32 at 5.1 px against the teacher's 5.5, camera 32/32 at 3.7 px against 4.0, with the envelope intervening on half the steps of the moving tasks, where it is now the stopping rule. Under supervision the model's contribution is the full-speed direction and the deceleration is the envelope's; a readout that stops on its own needs the distance to act on the magnitude, which a linear readout of this network does not give it.

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
.venv/Scripts/python.exe -m ganglion.train.cursor_finetune --checkpoint runs/cursor-dagger-v1/round-03/cursor-readout.pt --out runs/cursor-finetune-v4 --iterations 300 --train encoders,readout,edges --encoder-lr 1e-5 --readout-lr 1e-4 --edge-lr 1e-4 --episode-ticks 1000 --window 96 --truncate 32 --kick-every 250 --student-max 0.9 --seconds 1700
.venv/Scripts/python.exe -m ganglion.train.cursor_readout --checkpoint C:/DEV/Haltere/artifacts/ftPath2_best.pt --out runs/cursor-readout-v3 --episodes 64 --steps 200 --features 512 --seconds 900 --adapter-version 3
.venv/Scripts/python.exe -m ganglion.train.cursor_dagger --checkpoint runs/cursor-readout-v3/cursor-readout.pt --features runs/cursor-readout-v3/features.pt --out runs/cursor-dagger-v3 --rounds 3 --neurons 512 --seconds 900
.venv/Scripts/python.exe -m ganglion.train.suite --checkpoint runs/cursor-dagger-v3/round-01/cursor-readout.pt --mlp-features runs/cursor-dagger-v3/features.pt --out runs/suite-v3 --seconds 1500
.venv/Scripts/python.exe -m ganglion.train.cursor_readout --checkpoint C:/DEV/Haltere/artifacts/ftPath2_best.pt --out runs/cursor-readout-v4 --episodes 64 --steps 200 --features 4096 --seconds 900 --adapter-version 4 --view-fraction 0.5
.venv/Scripts/python.exe -m ganglion.train.cursor_dagger --checkpoint runs/cursor-readout-v4/cursor-readout.pt --features runs/cursor-readout-v4/features.pt --out runs/cursor-dagger-v4 --rounds 3 --neurons 4096 --seconds 900
.venv/Scripts/python.exe -m ganglion.train.suite --checkpoint runs/cursor-dagger-v3b/round-01/cursor-readout.pt --sense-version 4 --mlp runs/suite-v3/mlp-baseline.pt --out runs/suite-v3b-slip --seconds 1500
.venv/Scripts/python.exe -m ganglion.train.cursor_readout --checkpoint C:/DEV/Haltere/artifacts/ftPath2_best.pt --out runs/cursor-readout-v4b --episodes 64 --steps 200 --features 4096 --seconds 900 --adapter-version 4 --view-fraction 0.5 --slip-dropout 0.3 --slip-blank 0.05 --slip-gain 0.7 1.0
.venv/Scripts/python.exe -m ganglion.train.cursor_dagger --checkpoint runs/cursor-readout-v4b/cursor-readout.pt --features runs/cursor-readout-v4b/features.pt --out runs/cursor-dagger-v4b --rounds 3 --neurons 4096 --seconds 900
.venv/Scripts/python.exe -m ganglion.train.suite --checkpoint runs/cursor-dagger-v4b/round-01/cursor-readout.pt --mlp runs/suite-v4/mlp-baseline.pt --out runs/suite-v4b --seconds 1500
.venv/Scripts/python.exe -m ganglion.train.suite --checkpoint runs/cursor-dagger-v4b/round-01/cursor-readout.pt --sense-version 3 --mlp runs/suite-v3/mlp-baseline.pt --out runs/suite-v4b-noslip --seconds 1500
.venv/Scripts/python.exe -m ganglion.train.cursor_readout --checkpoint C:/DEV/Haltere/artifacts/ftPath2_best.pt --out runs/cursor-readout-v5 --episodes 64 --steps 200 --features 4096 --seconds 900 --adapter-version 3 --goal-scale 0.1
.venv/Scripts/python.exe -m ganglion.train.cursor_dagger --checkpoint runs/cursor-readout-v5/cursor-readout.pt --features runs/cursor-readout-v5/features.pt --out runs/cursor-dagger-v5 --rounds 3 --neurons 4096 --seconds 900
.venv/Scripts/python.exe -m ganglion.train.suite --checkpoint runs/cursor-dagger-v5/round-01/cursor-readout.pt --mlp-features runs/cursor-dagger-v5/features.pt --out runs/suite-v5 --seconds 1500
.venv/Scripts/python.exe -m ganglion.train.cursor_readout --checkpoint C:/DEV/Haltere/artifacts/ftPath2_best.pt --out runs/cursor-readout-v5b --episodes 64 --steps 200 --features 4096 --seconds 900 --adapter-version 3 --goal-scale 0.2
.venv/Scripts/python.exe -m ganglion.train.cursor_dagger --checkpoint runs/cursor-readout-v5b/cursor-readout.pt --features runs/cursor-readout-v5b/features.pt --out runs/cursor-dagger-v5b --rounds 3 --neurons 4096 --seconds 900
.venv/Scripts/python.exe -m ganglion.train.suite --checkpoint runs/cursor-dagger-v5b/round-03/cursor-readout.pt --mlp-features runs/cursor-dagger-v5b/features.pt --out runs/suite-v5b --seconds 1500
.venv/Scripts/python.exe -m ganglion.train.cursor_readout --checkpoint C:/DEV/Haltere/artifacts/ftPath2_best.pt --out runs/cursor-readout-v6 --episodes 64 --steps 200 --features 4096 --seconds 900 --adapter-version 5 --goal-scale 0.1
.venv/Scripts/python.exe -m ganglion.train.cursor_dagger --checkpoint runs/cursor-readout-v6/cursor-readout.pt --features runs/cursor-readout-v6/features.pt --out runs/cursor-dagger-v6 --rounds 3 --neurons 4096 --seconds 900
.venv/Scripts/python.exe -m ganglion.train.suite --checkpoint runs/cursor-dagger-v6/round-03/cursor-readout.pt --mlp-features runs/cursor-dagger-v6/features.pt --out runs/suite-v6 --seconds 1500
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
