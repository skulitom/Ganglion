# Ganglion: project assessment and suggestions

For the implemented runtime, training and first-person experiments, see the
[current progress review](05-progress-review.md). The assessment below records the
earlier planning-stage review.

Review date: 2026-09-17. These are proposals, not changes to the agreed project plan.

## My assessment

Ganglion is worth building. The strongest idea is the combination of a continuously updated percept ledger and bounded, local motor programs: the agent can reason slowly while its instructions remain responsive to the world. The separation between the MCP client and the in-session runtime is sensible. Reusing Anode and Haltere gives the project a credible starting point.

The fly connectome is an interesting research direction. Its contribution to computer control still needs to be demonstrated. Keep it central to the research agenda while letting the deterministic runtime deliver useful results independently. The plan already embraces this principle; protect it when implementation gets difficult.

The biggest risk is coupling too many uncertain projects into the first release: a runtime, a visual system, a simulator, a teaching interface, several game integrations, and a learned controller. Generalization should remain the objective. Prove it through small transfers between different applications before making eight applications release requirements.

## What is already strong

- **Measurements before implementation.** Capture probes and retained connectome census outputs make the plan unusually inspectable.
- **Deterministic controllers behind a shared interface.** This keeps experiments comparable and provides a working fallback.
- **A real window paired with a headless Arena.** This can reveal failures that simulation-only evaluation misses.
- **Application-specific skill data outside the core.** A good foundation for testing transfer.
- **Explicit attention to input release, recording consent, and thermal limits.** These belong in the architecture from the start.

## Review ratings

These rate design readiness, not the quality of a deployed system. The repository currently contains planning documents and research scripts; there is no Ganglion runtime or test suite here.

| Dimension | Assessment | Main reason |
|---|---|---|
| Correctness | Needs contracts before implementation | Ownership, lease expiry, observation time, and multi-eye fusion are underspecified. |
| Security and isolation | Good intent; enforcement unspecified | Local transport still needs client identity and scoped control; session and input ownership must be explicit. |
| Performance | Promising component evidence; end-to-end unproven | Poll cost and isolated inference speed do not establish reaction latency under game load. |
| Maintainability | Strong proposed boundaries; reproducibility incomplete | Useful source/output pairs exist, but dependencies, data locations, and benchmark provenance are not packaged. |

## Recommended priorities

P1 means resolve before building the relevant subsystem. P2 means resolve before claiming generalization or promoting a research model. These are design findings, not assertions of existing runtime bugs.

| Priority | Recommendation | Detail |
|---|---|---|
| P1 | Specify actuator ownership, leases, cancellation, and recovery | [Runtime contracts](01-runtime-contracts.md) |
| P1 | Measure scene change through application response, with tail latency | [Measurement and evaluation](02-measurement-and-evaluation.md) |
| P1 | Define neural time, gaze changes, and fusion across eyes | [Vision and connectome experiments](03-vision-and-connectome.md) |
| P1 | Build one complete, measurable Arena demonstration | [First milestone](04-first-milestone.md) |
| P2 | Separate zero-shot transfer from teaching and fine-tuning | [Measurement and evaluation](02-measurement-and-evaluation.md) |
| P2 | Test whether biological topology adds value over matched baselines | [Vision and connectome experiments](03-vision-and-connectome.md) |
| P2 | Make evidence reproducible from a clean checkout | [First milestone](04-first-milestone.md) |

My preferred first demonstration: an agent teaches a target, arms a click reflex, spends several seconds away from the tools, and returns to an accurate account of successful reactions and failures. Repeat inside the seat, then transfer the same primitive to a small browser task. That demonstrates the project's central promise without waiting for training.

## Review scope

Inspected [README](../README.md), [PLAN](../PLAN.md), capture benchmark scripts, census scripts and selected retained outputs. Also inspected the referenced [Haltere brain implementation](../../Haltere/haltere/brain/model.py) to check timestep and batch semantics, and consulted upstream documentation linked in the detailed notes.

No capture sessions, desktop input, GPU training, or large census jobs were run. Existing performance numbers were reviewed as reported evidence, not independently reproduced. This directory was not a Git repository when reviewed. Suggestions only were added; the existing plan and scripts were left intact.
