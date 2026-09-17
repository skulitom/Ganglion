# A first milestone that proves the project

The proposed Phase 1 in [PLAN](../PLAN.md):675-679 combines infrastructure, many motor programs, desktop tasks, and an RTS integration. I would retain those destinations while splitting delivery into evidence gates. These gates replace calendar guesses with reviewable outcomes.

## Gate A: reproducible foundation

- Add a minimal installable package and locked environment, with optional extras for GPU models and training. The deterministic runtime should be testable without loading connectome assets.
- Add `doctor` checks for target session, display/capture availability, dependency versions, input capability, and model/data paths. Distinguish detection from an explicitly requested active input probe.
- Parameterize census data locations instead of relying only on `C:\DEV\Haltere`. Retain data release identifiers, graph selectors, file hashes, dependency versions, and source revisions alongside derived counts.
- Put phase status, known failures, and exact reproduction commands in the repository. The external `ganglion-project` memory mentioned at PLAN.md:682 and 732-734 is useful context, but a clean checkout should explain its own state.
- Version the wire schema, skill-pack schema, observation transforms, and model metadata before saved recordings depend on them. Pin sibling dependencies to known revisions for reproducible runs.

**Exit:** a fresh environment can install the minimal package, validate configuration, and run deterministic unit tests without a live desktop or GPU. Hardware-specific checks clearly report what they could not test.

## Gate B: one observable reaction

Build an Arena mode that makes a target appear at randomized locations and times, accepts clicks, and logs ground truth. Implement only the path needed for `look`, `watch`, `arm`, `wait`, `halt`, and status, with a deterministic detector and one bounded click response. Use the same runtime interfaces intended for later controllers.

Instrument the timing chain described in [the measurement notes](02-measurement-and-evaluation.md). Keep ground-truth telemetry available to the evaluator, with an explicit switch controlling whether a policy can use it.

**Exit:** the agent can arm the reaction, leave the tools for several seconds, then retrieve a ledger showing success, misses, false actions, and timing. No action occurs outside the valid lease. Halt and target loss have verified behavior. Start on the console, then repeat in the seat.

## Gate C: one closed-loop program and one transfer

Add `reach` with a deterministic controller, an explicit target-loss outcome, and completion based on observed state. Test both a moving Arena target and a simple browser task with a stable target condition. Then add drag-until-condition if the shared interfaces hold up.

Allow application differences through watch definitions and parameters. When a core change is necessary, describe the general capability it adds and rerun both tasks. Do not use a single success as evidence of broad transfer.

**Exit:** a concise scorecard compares periodic agent input with the local controller, including failures, teaching time, and task success. The same primitive works in two different environments through the public interface.

## Gate D: connectome experiments on a working system

Integrate the visual model in shadow mode: it receives the same observations and logs predictions, while the established path controls actions. Compare it on frozen replay sets and live Arena runs, then promote it for one percept or motor program after it passes the agreed criteria.

This creates useful artifacts even if a particular network underperforms: a working runtime, recordings, a benchmark, and a precise result about what the model did or did not improve.

## First regression cases

Use a fake clock and fake actuator for protocol/state tests, replay for perception, and explicitly selected Windows integration runs for capture/input behavior. The most valuable cases are:

1. Lease expiry during a wait and attempted renewal by another client.
2. Lost response followed by retry of the same mutating request.
3. Simultaneous drag and click, including cancellation and preemption.
4. Repeated observation IDs and sustained trigger conditions.
5. Capture access loss, inference stall, and input-submission failure.
6. Target window moved/replaced after teaching, including image scaling.
7. Ledger overflow and two independent readers.
8. Gaze pan/zoom over a static scene and dropped visual frames.

Keep performance results outside ordinary unit-test pass/fail unless the runner and workload are controlled. Publish a short demonstration together with its failure cases and the command needed to reproduce its scorecard.
