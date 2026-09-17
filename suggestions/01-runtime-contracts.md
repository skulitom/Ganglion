# Runtime contracts to settle first

These proposals refine the architecture in [PLAN](../PLAN.md); they do not require implementing the learned brain first.

## 1. Give each actuator one owner

**Evidence:** PLAN.md:237-249 and 377-403 allow reflexes, motor programs, and brain readouts to produce actions, with evaluation on both vision frames and ticks. Arbitration is not defined.

**Failure example:** a drag holds the left mouse button while a reflex clicks with it. The reflex releases the button and silently ends the drag. A steer intent and an align reflex can similarly issue incompatible mouse deltas.

**Suggestion:** introduce one actuation arbiter. Start with exclusive leases for the pointer, each held key/button, and each gamepad axis group. Programs declare required resources; conflicts return a rejection or an explicit preemption result. Define a stable priority order with halt and fault recovery first. Avoid blending until there is a task that needs it and a precise blending rule.

Evaluate visual triggers once per observation ID. Give triggers edge/level semantics, hysteresis, cooldown, and a maximum action duration. A new tick using the same observation must not count as new evidence for a second appearance event.

**Acceptance:** overlapping drag/click and steer/align requests produce deterministic ownership transitions; a persistent detection generates the configured number of firings at both 60 and 100 Hz.

## 2. Make lease renewal deliberate and reconcile it with wait

**Evidence:** PLAN.md:370-372 defaults to a 30-second TTL renewed by every agent call. Lines 432 and 439 permit a 60-second wait and also renew on every call.

**Failure examples:** an intent expires halfway through a legitimate wait; a status-polling observer unintentionally keeps another client's reflexes armed.

**Suggestion:** bind a bounded lease to a controlling client and return its absolute monotonic expiry. Read-only inspection should not renew control implicitly. An owner can explicitly renew selected operations. For v1, cap each wait at the remaining lease duration and return a distinct `lease_expiring` result so the agent can renew intentionally. Bound each action separately so renewal never turns a short key press into an indefinite hold.

Use a session-scoped capability for the loopback protocol, one controlling client by default, and separate observer permissions. Specify message-size limits and reject unknown operations. Loopback binding alone does not express which local client owns the controls.

**Acceptance:** an observer cannot prolong a lease; a long wait returns before expiry with an explicit reason; expiry releases all inputs owned by that lease.

## 3. Faults should invalidate commands explicitly

**Evidence:** PLAN.md:728-730 says to degrade to the last safe command after exceptions. A previously appropriate command can become wrong when capture or focus changes.

**Suggestion:** use explicit states such as `ready`, `running`, `degraded`, and `halted`, with a cause and recovery requirement. On lost target identity, invalid capture, controller failure, or expiry, stop affected programs and attempt to release their owned inputs. Resuming should require fresh observations and valid ownership. Define any permitted command holdover by actuator and maximum duration.

Put lease enforcement where it can still operate if vision or inference stalls. An in-process exception handler cannot handle termination of the whole process. If crash recovery is a claimed property, test a separate input-owning helper or watchdog and document its limits. A failed release must remain a visible fault.

Distinguish a healthy static desktop from a broken capture source: a timeout waiting for a new DXGI frame is different from access loss, which requires reinitializing duplication. [Microsoft documents these separate outcomes](https://learn.microsoft.com/en-us/windows/win32/api/dxgi1_2/nf-dxgi1_2-idxgioutputduplication-acquirenextframe).

**Acceptance:** inject controller errors, capture access loss, focus changes, process termination, and failed input submission. Verify release behavior and explicit failure records. A static scene alone must not be mistaken for a dead capture thread.

## 4. Bind targets to the snapshot that defined them

**Evidence:** PLAN.md:339-342 teaches by pointing at a snapshot; lines 355-357 return a scaled composite; lines 395-400 support both absolute UI and relative game input. Coordinate conversion and target identity are not specified.

**Suggestion:** return a snapshot ID, session identity, target window identity, layout revision, capture dimensions, and image-to-client/screen transforms. Watches and intents name their coordinate space. Revalidate the target and transform at actuation; require rebinding after incompatible window, display, or session changes. Treat relative camera deltas separately from cursor coordinates.

Record whether input was submitted and whether its intended effect was observed. `SendInput` reports inserted events, is constrained by integrity levels, and can be affected by existing keyboard state; it does not acknowledge task completion. [Microsoft SendInput documentation](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-sendinput).

**Acceptance:** test a scaled screenshot, moved/resized window, mixed display scaling, and target replacement. Old coordinates must not silently become valid actions in a different window.

## 5. Give the ledger explicit delivery and loss semantics

**Evidence:** PLAN.md:351-357 defines a capped ring buffer and a delta since the last call. The plan also promises that nothing the fast loop does becomes invisible (lines 166-168).

**Suggestion:** assign monotonic event sequence numbers and per-client cursors. Return `next_cursor`, the available sequence range, and explicit overflow/omission counts. Separate a compact perception summary from retained action lifecycle records. Reserve capacity for failures and input release events; if complete audit history is required, give it a bounded durable spool and a defined behavior when that fills.

Attach observation, intent, reflex, and command IDs to events. Use idempotency keys for mutating requests so reconnect/retry cannot cause a second click or a second intent. Distinguish request acceptance, input submission, and observed completion.

**Acceptance:** overflow the perception ring, reconnect after losing a response, and use two readers. Readers detect gaps, do not consume each other's cursors, and a retried action is not executed twice.
