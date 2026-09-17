# Progress review: make the fly model's contribution measurable

Reviewed 2026-09-17 through commit `6a82aa54cd52b180d1f8df0409f51db97da4163d`.
This is an assessment and proposed next work, not an implementation or promotion.

## Assessment

Ganglion now has a credible reactive runtime: bounded input, observation freshness,
leases, independent release, reusable reach/drag/align/move programs, and retained
evidence from several environments. The transition from cursor dragging to relative
camera control is meaningful transfer of the architecture. Keeping the agent's slow
decisions outside the resident loop is working as a design.

The central research claim still needs evidence: does the fly connectome improve
real-time perception and control? Supervised deployment shows that the model can
participate in a working loop. It does not yet establish independent competence or
an advantage over the deterministic controller or small learned baselines.

I would keep the fly model central and temporarily cap application-specific expansion.
The next milestone should demonstrate one reliable visual feedback skill, with a
measured contribution from the fly model, across a small number of different plants.

## Evidence reviewed

| Observation | Interpretation |
|---|---|
| Current CPU suite: 129 passed, 4 optional checks skipped in 16.56 s | Stronger regression coverage; this review independently ran it. |
| Connectome alone: 3/16 static targets on one test split, 4/16 on another | Cursor stability remains unresolved; these are different splits, not an improvement between checkpoints. |
| MLP: 16/16 static targets, 3.37 px mean terminal error on the DAgger test split | The benchmark is tractable; the biological model has not established an advantage. |
| Live supervised Solitaire: 90.6% of pointer commands from the model | Integration evidence; the fallback changes the trajectory and can make critical corrections. |
| Half-Life report: 80.4% model view commands; 15 aligned-and-fired intents among 58 | Useful transfer, alongside substantial target loss and timeouts. The report records 68 firing bursts; intents and bursts are different counts. |
| Live Solitaire inference p50/p95/p99: 3.40/8.31/19.90 ms | Median speed is encouraging; the latency tail and complete perception-to-effect delay matter more for dependable reactions. |

Sources: [training](../docs/bench/TRAINING.md),
[supervised control](../docs/bench/SHADOW.md),
[Half-Life](../docs/bench/HALFLIFE.md),
[engagement summary](../docs/bench/results/halflife/grunt-fights-2026-09-17.json).
Live results were inspected as retained evidence, not rerun in this review. No desktop
input, GPU training, game launch, or model publication was performed.

## 1. Fix and unify time semantics first

**Reproduced defect:** `ShadowWorker.proposal()` checks freshness from inference
completion, while the worker initially admits results based on source observation age.
A fake-clock run submitted a sample at 10.000 s, completed inference at 10.049 s, then
requested a proposal at 10.080 s. The proposal was returned: its source was 80 ms old,
although the declared source-age budget is 50 ms. A current target observation does
not make the model's earlier input current.

Store source acquisition time, observation ID and expiry with every proposal. Check
them again at use time, including intent/layout binding. Carry the proposal's source
ID into the action ledger so reused outputs and their actual ages are measurable.
Add regressions for delayed inference, reuse, expiry, cancellation and a target jump.
See [proposal reuse](../ganglion/brain/shadow.py) and
[runtime selection](../ganglion/core/runtime.py).

Separately, neural time still advances one fixed 10 ms step per consumed observation.
Latest-only delivery under a variable capture cadence changes the relationship between
neural time and wall time. Choose bounded fixed substeps or a validated variable-step
contract, and train with that same contract. Replay the same trajectory at 30/60/75 Hz
with jitter and dropped frames. Extra neural steps must not manufacture fresh evidence.

Measure acquisition → percept → model → input → observed effect as paired events.
Report p50/p95/p99, missed deadlines and source age. An inference benchmark or a 100 Hz
scheduler alone does not establish a 100 Hz visual feedback loop.

## 2. Replace command share with causal comparisons

Keep the supervisor for live experiments, but compare deterministic-only, MLP-only,
connectome-only, supervised MLP and supervised connectome on identical seeded episodes.
Run unassisted learned policies in the headless plant first. A random-proposal control
under the same supervisor would also reveal how much competence the wrapper supplies.

Report completion, settling time, integrated tracking error, overshoot, intervention
frequency and longest uninterrupted model-controlled interval. Count distinct model
inferences as well as commands; a proposal can be reused. Use paired comparisons and
multiple training seeds before claiming an advantage.

Acceptance share is a diagnostic, not a learning objective. Many harmless small steps
can inflate it while a few deterministic interventions rescue the task. The progress
check is also not a general completion or stability guarantee: it checks a proposed
coordinate, while actual motion may be delayed, ignored or disturbed, and targets move.

## 3. Change the training experiment before increasing its duration

The current fine-tuner resets episodes after 192 ticks (1.92 s), detaches recurrent
state every eight ticks (80 ms), and evaluates over 20 s. It optimises per-action
imitation error. The NumPy plant receives detached actions, so gradients do not flow
through the cursor trajectory. These are specific limitations, not evidence that the
connectome cannot learn the task.

Start with longer model-driven episodes, deliberately including drift and recovery,
target changes, delayed observations, and near-target settling. Compare 8/32/64-step
gradient windows at a controlled compute budget. Continue labelling states reached by
the model; this is the distribution problem addressed by
[DAgger](https://proceedings.mlr.press/v15/ross11a.html).

For a trajectory loss, use a differentiable generic plant or a method that can optimise
rollout outcomes; adding a distance term computed from the existing detached NumPy
state will not supply the intended gradient. Measure motor-feature sensitivity, encoder
gradient norms, saturation and state drift. Then test restricted adaptation of neuron
gains/biases/time constants before expanding to edge gains. Preserve topology and known
signs, and compare against the frozen graph and small recurrent/MLP baselines.

Calibrate actuator dynamics separately for absolute pointer motion and relative camera
motion. `align` currently estimates its virtual cursor from submitted mouse deltas and
a configured gain; this is not a measurement of actual camera displacement. Train and
evaluate both plant types, with measured delay and gain ranges.

## 4. Move fly visual perception onto the critical path

The cursor adapter still zeros the optic-flow channel. Meanwhile the deterministic
motion detector suppresses detection while the runtime moves the view/player.
`move` continually extends that suppression window. Existing template tracks can
continue, but discovery of new moving objects goes blind during locomotion.

Build a small camera-motion-compensated motion task: independently moving targets
during pan/translation, brightness changes, occlusion and distractors. Evaluate target
velocity, false alarms, identity retention, reacquisition and latency. Estimate global
image motion, then test whether a fly visual pathway improves local motion evidence.
Do not treat commanded mouse movement as ground-truth optic flow.

An existing reference is [Flyvis](https://github.com/TuragaLab/flyvis), the authors'
implementation of a connectome-constrained fly visual model, with optic-flow training,
moving-edge and custom-stimulus examples. Use it as a measured reference or explicit
integration prototype; it does not automatically validate the current male-CNS graph
or provide application semantics. A biological motion front-end must earn its latency
and accuracy on Ganglion's observations.

## 5. Keep transfer work bounded and release evidence coherent

Use Solitaire and Half-Life as regression environments. Add application-specific work
only when it exposes a reusable failure: motion under camera movement, lighting,
occlusion, action latency, target reacquisition or input cleanup. Winning more games
is not the next research gate.

Extend the real helper-process producer-death regression to the new key holds, button
holds and queued relative movement. Existing forced-death coverage exercises click and
drag; in-memory halt tests cover the new paths but do not replace process-level checks.

Update current authority descriptions: `TRAINING.md` and the cursor-adapter docstring
still say shadow-only, while intents now permit supervised model actions. Keep historical
results clearly dated. Reports should include commit, dirty-tree status, checkpoint and
graph hashes, adapter/time contract, seed splits and policy/fallback mode.

For Hugging Face, prepare a portable inference package and model card after a repeatable
benchmark demonstrates useful behaviour. A research checkpoint can be published earlier
if explicitly presented with its failures and dependencies; publication itself is not
model promotion. The current local graph-path dependency needs a reproducible asset
setup in either case.

## Proposed next milestone

Freeze a paired suite of at least 200 held-out episodes covering pointer settling,
moving-target pursuit and relative-camera tracking. Establish the teacher's feasible
timeout and latency envelope first. A proposed first gate is at least 95% success on
teacher-solvable episodes with the learned controller alone, plus reported settling
time and tracking error relative to the baselines. This is a proposed engineering gate,
not a biological conclusion or a result already achieved.

Then transfer the same trained skill to one controlled live fixture in Anode, keeping
the supervisor and measuring interventions. In parallel with the motor experiment,
validate a visual motion front-end on recorded clips. This would advance both halves
of the project's stated purpose: fast perception and fast response using the fly model.
