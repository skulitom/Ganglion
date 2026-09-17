# Measure the whole capability

## 1. Separate capture throughput from reaction latency

**Evidence:** [bench.py](../docs/bench/bench.py):8-24 and [seatbench.py](../docs/bench/seatbench.py):19-28 count non-null grabs, then time an unconditioned sequence of grabs. Those timed calls are not separated into fresh-frame and no-frame samples. Neither script observes a corresponding injected input or application response.

The reported 0.06-0.07 ms poll cost is useful API overhead evidence. It does not establish scene-change-to-observation latency. The scripts already acknowledge that video mode repeats frames; keep that qualification in headline tables too. Upstream DXcam documents both no-new-frame returns and repeated frames in video mode. Its current documentation may describe features absent from the plan's measured 0.0.5 version, so record the installed version before adopting newer APIs. [DXcam documentation](https://github.com/ra1nty/DXcam).

**Suggestion:** make the Arena display a machine-readable event/frame ID and acknowledge received inputs. Instrument:

| Timestamp | Meaning |
|---|---|
| Event scheduled | Arena decided to change the scene |
| Present timestamp, where available | Frame containing the event was presented to the capture path |
| Capture available | Ganglion can read the corresponding frame |
| Percept ready | Detector output is available |
| Input submitted | The input API returned, including success count |
| Input received | Arena processed the corresponding input |
| Effect observed | Resulting application state is confirmed |

Use a shared monotonic timebase with verified units and clock alignment. Keep scheduled/render-submitted time distinct from actual presentation time. Software timestamps do not measure physical monitor scanout. Distinguish new desktop content, pointer-only updates, repeated frames, and overwritten frames where the backend exposes that information.

Report p50/p95/p99, observed maximum, sample count, missed deadlines, misses, and false activations. Pair stage timings by event ID; do not add independent stage percentiles and call the sum an end-to-end percentile. Include CPU conversion, upload, GPU completion, and readback. Label short runs as smoke measurements.

## 2. Treat the latency budget as a target

**Evidence:** PLAN.md:245-250 calls one frame plus 13 ms a worst case. That expression excludes unbounded scheduling and resource-contention delays. Phase 1 at line 676 also asks for reaction within one frame, which is a different promise.

**Suggestion:** choose one measurable endpoint and publish a percentile objective plus a deadline-miss rate under named conditions. For example, the existing one-frame-plus-13-ms figure can become an initial *screen-change-to-input-submission target*, subject to Phase 0 results. Application receipt and visible effect need separate numbers. Do not describe a measured maximum as a guaranteed worst-case bound.

Run console and seat tests with an idle GPU and representative rendering load. Sweep watch count, gaze count, frame size, recording, and ledger readers. Benchmarks should expose saturation, not just the best configuration. Measure what the game loses in frame rate as well as what Ganglion achieves.

A separate CUDA stream and an edge-count extrapolation are hypotheses about performance, not evidence of latency isolation. Admit new watches only when a measured budget allows them; define which optional work degrades first.

## 3. Keep control freshness and event completeness separate

**Evidence:** PLAN.md:43 says the agent never misses a fast event, while lines 222 and 728 choose newest-frame-only processing.

**Implication:** an event can appear and disappear between captures, or exist only in a frame overwritten before perception. A fresh control loop cannot promise a complete visual history.

**Suggestion:** retain newest-frame processing for control, expose dropped-frame counts and coverage gaps, and qualify event recall by event duration and capture cadence. A bounded secondary recording path can aid diagnosis, but also needs explicit overflow behavior. Recording at 10-20 Hz (PLAN.md:343) cannot reconstruct all one-frame events; use optional full-rate clips around triggers when diagnosing them.

**Acceptance:** inject short events of varying duration and random phase relative to capture. Plot detection probability versus duration and report where evidence was dropped. Keep evaluator truth separate from policy inputs in pixel-only runs.

## 4. Harden the scratch benchmarks when promoting them

These are small improvements to research tooling, not blockers to the project's concept:

- [wgc.py](../docs/bench/wgc.py):6-16 and [seatwgc.py](../docs/bench/seatwgc.py):6-16 stop only inside a frame callback. No initial frame, or a stall after one frame, can leave the run waiting indefinitely. Use an external deadline and an explicit empty/insufficient-sample result.
- [bench.py](../docs/bench/bench.py):26-27 and 46-47, and [seatbench.py](../docs/bench/seatbench.py):12-13 and 30-31, print exceptions without a failing process status. The eventual CLI should emit structured failure/unsupported results and a documented exit code.
- Preserve successful-frame cost separately from empty polling cost, release capture resources in cleanup paths, and divide counts by actual measured elapsed time.
- Store machine/session conditions, exact package versions, configuration, raw samples, and the command used alongside each summary. Add a retained brain-step benchmark: its reported timing currently has no corresponding script in this directory.

## 5. Evaluate three different claims separately

The plan already requires baselines and a held-out target (PLAN.md:529-550). Make the comparison more diagnostic:

| Claim | Comparison |
|---|---|
| Continuous observation helps | Agent with periodic screenshots versus agent with ledger, using the same discrete action interface |
| Local feedback helps | Ledger plus discrete actions versus ledger plus deterministic reflexes/controllers |
| The learned brain helps | Identical runtime/percepts with deterministic, compact learned, and connectome controllers |

Report success rate, reaction latency, false actions, recovery frequency, agent calls/tokens, teaching time, and GPU cost. Use matched task seeds and repeated trials.

Separate **zero-shot transfer**, **bounded teaching through MCP**, and **weight fine-tuning**. A target taught during evaluation is not zero-shot. Split held-out tasks and recordings before tuning thresholds or prompts, and preserve a final untouched evaluation set. Use different tasks where a primitive does not apply instead of forcing every game into an identical checklist.
