---
name: ganglion-agent
description: Use a running Ganglion core for timestamped observations, bounded colour reflexes, and cursor-feedback reach/drag.
---

# Ganglion agent guide

Use the configured `ganglion_*` tools. The resident core must already run inside the target
application's session. See `docs/RUNTIME.md` in this repository for setup and the exact contract.

1. Read `ganglion_status` and `ganglion_look`. Check the target, state, observation age, and
   snapshot ID. Images may be scaled; convert image points with `image_transform.image_to_screen`.
2. Claim a bounded lease explicitly. Inspection never renews it. Keep your bridge's client ID
   stable if reconnecting; another client cannot renew your lease.
3. Teach `ganglion_watch(spec=...)` with snapshot ID, screen-pixel region `[x,y,w,h]`, RGB colour,
   tolerance, and minimum component area. This detects colour components, not semantic objects.
4. Arm `ganglion_arm(spec=...)` with the returned watch ID, `click` or `notify`, an `appear` or
   `present` trigger, cooldown, max firings, click hold time, and reflex TTL. Alternatively start
   `ganglion_intent(spec={program: "reach", watch_id: "...", click: true, timeout_seconds: 5})`.
   Reach owns the pointer until completion/failure/cancellation and confirms arrival from cursor feedback.
5. Keep `next_cursor` from look/wait. Page while `has_more`; inspect `lost_events` and action
   failures. `lease_expiring` is a request to make an explicit renewal decision. Renewing a
   lease does not extend a reflex's TTL or an intent's timeout. Wait for `intent_completed`,
   `intent_failed`, or `intent_cancelled`, then inspect `status.intent` and the application's state.
6. Distinguish admitted commands, submitted input, release, and observed effects. A target
   disappearing is not sufficient evidence of arbitrary task success.
7. Halt when the task ends. After a target/layout failure, take a new snapshot and rebind watches.

Reuse a mutation's `request_id` only when retrying exactly that mutation after an uncertain
response. Do not replay an action with a fresh ID merely because its response was lost.
Use `ganglion_cancel` to stop one program; pending commands drain before another starts. An in-flight
command can execute before the helper acknowledges cancellation. Halt stops intents and reflexes.
`ganglion_input` accepts a snapshot-bound point and one `move` or `click`; it does not follow targets.
Reach and drag require fresh evidence (40 ms command admission, 50 ms helper expiry, 250 ms stale
failure). Quiet desktops use fresh GDI acquisitions, labelled separately from DXGI. Inspect source
and sample-start timestamps; a slow acquisition still consumes the evidence budget.

For drag, teach a source and optionally destination/condition watches, then use
`ganglion_intent(spec={program: "drag", watch_id: "...", destination: [x,y],
until: {watch_id: "...", present: true}, timeout_seconds: 5, verification_seconds: 1})`.
Supply exactly one destination point or `destination_watch_id`. The core releases on reaching
the destination or matching the condition, then verifies that condition from fresh samples after
release. Without `until`, completion establishes only arrival and release. A pre-satisfied condition
fails before pickup. `condition_verified` does not prove arbitrary application success.

On a core started with `--shadow-checkpoint` (add `--shadow-process` for live runs: it keeps
inference under the evidence budget), add `controller: "connectome"` to a reach, drag or align
spec to let the fly model propose the velocity; the core applies a proposal only when it brings
the cursor closer to the goal and reports `neural_commands`, `overridden_commands` and
`stale_commands` (steps with no proposal fresh enough).

First person: `ganglion_input` holds keys, sends relative look deltas and holds buttons;
`ganglion_watch` kinds `motion` and `track` see what moves and follow it, `flow` sees what moves
on its own while the view itself moves; `ganglion_intent`
`align` turns to a target and fires, `move` holds keys continuously; `ganglion_arm` responses
`key`, `align` and `track` react at frame rate. Arm motion reflexes only where motion means a
threat: doors and flashing lights move too.

The helper releases a drag without continuing renewals within at most 100 ms, subject to OS
scheduling. Wait for `output_halted` after cancellation; a release fault requires core restart.
Keyboard/gamepad programs and learned vision remain unavailable.
