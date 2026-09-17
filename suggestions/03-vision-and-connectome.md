# Experiments that make the fly-brain claim convincing

## 1. Define neural time independently of frame arrival

**Evidence:** PLAN.md:245-249 runs vision at roughly 60-75 Hz; line 287 gives the bootstrap network a 10 ms integration step. [Haltere's model](../../Haltere/haltere/brain/model.py):198-232 likewise uses a configured timestep for neural and action updates.

If a model advances one fixed 10 ms step per frame at 60 Hz, it advances only 0.6 seconds of model time per wall-clock second. Temporal tuning and controller behavior then depend on capture rate. This is a conditional design problem, not an observed Ganglion implementation bug.

**Suggestion:** explicitly choose a sampling/integration contract: fixed neural substeps with timestamped sample-and-hold input, or a validated variable-step scheme. Specify what happens to elapsed time when frames drop, how catch-up is bounded, and when state is reset. Advancing neural state on held input must not fabricate a new visual observation or retrigger an appearance event.

Flyvis exposes the integration timestep and initial state explicitly; its reference also warns about large integration steps. Matching receptor geometry alone does not settle timing and initialization. [Flyvis network reference](https://turagalab.github.io/flyvis/reference/network/).

**Acceptance:** replay equivalent motion at 30/60/75 Hz with jitter and dropped frames. Compare speed estimates, direction tuning, and trigger timing in seconds. Include the cost of any extra substeps in the compute budget.

## 2. Several batch entries are several recurrent states

**Evidence:** PLAN.md:604-609 proposes making each gaze window a batch entry of the whole network. [Haltere](../../Haltere/haltere/brain/model.py):184-186 creates state per batch entry, and lines 213-234 return actions per entry. Batching does not by itself connect those entries into one shared brain.

**Suggestion:** choose between two explicit architectures:

- Shared visual weights with separate eye states; transform and fuse their percepts into one central controller.
- Separate complete brains per eye, with an explicit downstream action selector.

I would start with the first because it gives one controller a coherent target and one action output. Include eye identity, gaze transform, observation age, and confidence in fusion. This is an engineering preference to test, not a claim of biological fidelity. If the goal is ultimately one integrated connectome, specify how multiple visual samples enter that single recurrent graph rather than relying on batching to achieve it.

Benchmark one, two, and four eyes directly. Shared weights may save work, but state and intermediate tensors still scale with batch size. Reconcile the two-eye budget with the phrase "batch 1" at PLAN.md:636.

## 3. Gaze movement must not look like a threat

**Evidence:** PLAN.md:261-279 moves and zooms gaze windows; lines 308-315 derive motion and looming from their changing inputs.

**Failure example:** moving the fovea over a static desktop changes many receptors. Zooming into a static object resembles expansion. A motion or looming reflex could react to the controller's own gaze change.

**Suggestion:** timestamp and version every gaze transform. Compensate motion in a common coordinate space or reset/reinitialize the affected temporal state and suppress unreliable outputs during a measured settling period. Track identities should survive gaze moves when there is enough evidence. Make watch reassignment reset the appropriate state instead of inheriting another region's history.

**Acceptance:** pan and zoom across a static scene without false looming actions, then confirm that genuine motion remains detectable after the documented recovery period.

## 4. Validate stimulus calibration and spatial limits

**Evidence:** PLAN.md:267-275 assumes the Flyvis sampling convention enables unchanged pretrained models. Its official tutorial describes 721 samples across a 31-column lattice and spatial averaging. [Flyvis custom-stimulus tutorial](https://turagalab.github.io/flyvis/examples/07_flyvision_providing_custom_stimuli/).

**Suggestion:** specify luminance conversion, range, contrast normalization, field of view, border behavior, and warm-up. Preserve native-resolution crops for precise UI work. On a 1920-pixel-wide screen spread over approximately 31 sample columns, the average horizontal spacing is about 62 pixels; a tiny icon can disappear between or within coarse samples. Foveation and deterministic sensors can address this, but that needs measurement.

For the male-CNS route, the census demonstrates that column coordinates and candidate pathways exist. It does not demonstrate that luminance injected into L1/L2/L3 reproduces their responses, or that the selected network already detects looming. Validate directional bars, contrast reversals, small targets, and expansion stimuli before closed-loop training. Check coordinate orientation and per-type column coverage after graph selection and pruning.

**Acceptance:** publish accuracy versus object size, speed, contrast, and zoom. Compare calibration probes before and after pruning; record which pathways and columns survive.

## 5. Test the value of topology and transferred assumptions

The plan already says that a connectome controller must earn its place. Extend that into a compact experiment matrix:

| Variant | What it tests |
|---|---|
| Deterministic controller | Whether the sensing/action loop and task are tractable |
| Compact learned controller with comparable resource budget | Whether learning alone accounts for gains |
| Connectome with frozen recurrence and learned interfaces | Whether the graph provides useful dynamics without extensive adaptation |
| Trained connectome | Whether optimizing the biological graph adds value |
| Degree/sign-controlled shuffled topology, where feasible | Whether gains depend on the particular wiring rather than sparsity or size |

Keep observations, teaching data, latency, tuning effort, and evaluation tasks comparable. Report multiple seeds, learning curves, and held-out results; select the simplest variant that meets the product target while retaining the biological experiments as research.

PLAN.md:655-657 calls several Haltere settings non-negotiable. They are valuable initial defaults, but computer control and a larger visual graph are new settings. Save inherited values explicitly and run bounded ablations when diagnostics suggest saturation or poor signal propagation.

Also give controller hot-swaps a lifecycle contract: state initialization, warm-up in shadow mode, ownership handover, and rollback. Swapping a recurrent controller into an active drag or flight should not inject an unexplained command discontinuity.
