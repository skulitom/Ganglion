# Recorded results

Every file here is the raw output of one measured run, kept so the numbers quoted in the
bench documents can be checked. Nothing in this folder is a promotion or a release; all
connectome runs are shadow or supervised. Dates are in each file's `measured_at` field.

## Reflex, reach and drag through the public tools

| File | What it records | Quoted in |
|---|---|---|
| `reflex-synthetic.json`, `reflex-console.json`, `reflex-seat.json` | Pixel-to-click reflex latency in the synthetic Arena, on the console session and inside the Anode seat | [REACH.md](../REACH.md) |
| `reach-synthetic.json`, `reach-console.json`, `reach-seat.json`, `reach-browser-console.json`, `reach-browser-seat.json` | Cursor-feedback reach against a periodic-input baseline, same environments plus the Edge fixture | [REACH.md](../REACH.md) |
| `drag-synthetic.json`, `drag-console.json`, `drag-seat.json`, `drag-browser-console.json`, `drag-browser-seat.json` | Bounded drag with verified release, same environments | [REACH.md](../REACH.md) |
| `seat.json` | Capture rate, input-to-pixel latency and tick jitter inside the seat | [REACH.md](../REACH.md) |
| `diagnostics/` | Capture and timing probes behind the numbers above | [REACH.md](../REACH.md) |

## Connectome in the loop

| File | What it records | Quoted in |
|---|---|---|
| `shadow-connectome-seat.json`, `shadow-connectome-seat-recorded.json`, `shadow-connectome-seat-packed.json` | The flight checkpoint proposing alongside the deterministic reach in the seat, with recorded sensor inputs | [SHADOW.md](../SHADOW.md) |
| `shadow-connectome-replay.json` | Frozen replay of those inputs through the model, off the desktop | [SHADOW.md](../SHADOW.md) |
| `transfer-seat-deterministic.json`, `transfer-seat-connectome.json`, `transfer-seat-connectome-all-motor.json`, `transfer-seat-connectome-all-motor-cap2.json`, `transfer-seat-connectome-v3b.json`, `transfer-seat-connectome-all-motor-process.json`, `transfer-seat-connectome-all-motor-process-spin.json`, `transfer-seat.json` | The Arena reach demo in the seat with the reference and each connectome readout under supervised authority (catch-up cap, own process, polling wait), and their ledger comparison | [SHADOW.md](../SHADOW.md) |
| `shadow-latency-console.json`, `shadow-latency-seat.json`, `shadow-latency-seat-capture.json`, `shadow-latency-seat-flasher.json` | The adapter paced like the reach loop, alone and beside a busy thread, the desktop capture or a rendering window | [SHADOW.md](../SHADOW.md) |
| `solitaire-seat/`, `solitaire-live/` | Taught Sawayama board runs and the live play harness, deterministic and connectome | [SOLITAIRE.md](../SOLITAIRE.md) |
| `halflife/` | The Half-Life pilot sessions: the first fights' ledger summary and screenshots, the ledger summary of the engagement with the motion channel fed live, the flow percept's calibration against known turns, and the repeated-engagement comparison of the motion channel fed against zeros, the v1b readout and the deterministic reference (`channel-trials/` holds the per-trial files) | [HALFLIFE.md](../HALFLIFE.md) |

## Cursor training

| File | What it records | Quoted in |
|---|---|---|
| `cursor-readout-v2.json` | Ridge readout on the frozen connectome, adapter v2, with the MLP comparison | [TRAINING.md](../TRAINING.md) |
| `cursor-dagger-v1.json` | Three DAgger rounds on that readout; its round 3 is the v2 candidate | [TRAINING.md](../TRAINING.md) |
| `cursor-dagger-v2.json` | DAgger with 1,000-tick kicked episodes; did not improve | [TRAINING.md](../TRAINING.md) |
| `cursor-dagger-v1b.json` | Refit on all 3,913 motor neurons; the best supervised controller so far | [TRAINING.md](../TRAINING.md) |
| `cursor-readout-v5.json`, `cursor-dagger-v5.json`, `cursor-suite-v5.json` | Goal scale 0.1 (v5) readout, DAgger and the suite | [TRAINING.md](../TRAINING.md) |
| `cursor-readout-v7.json`, `cursor-dagger-v7.json`, `cursor-suite-v7.json` | v6 with the stopping samples weighted ten times in the readout fit (v7) | [TRAINING.md](../TRAINING.md) |
| `cursor-readout-v6.json`, `cursor-dagger-v6.json`, `cursor-suite-v6.json` | The direction-encoded goal (adapter version 5, v6) readout, DAgger and the suite | [TRAINING.md](../TRAINING.md) |
| `cursor-readout-v5b.json`, `cursor-dagger-v5b.json`, `cursor-suite-v5b.json` | Goal scale 0.2 (v5b) readout, DAgger and the suite | [TRAINING.md](../TRAINING.md) |
| `cursor-readout-v4b.json`, `cursor-dagger-v4b.json`, `cursor-suite-v4b.json`, `cursor-suite-v4b-noslip.json` | Adapter v4b (slip dropout, blanking and gain) readout, DAgger, and the suite with the slip and with the channel absent | [TRAINING.md](../TRAINING.md) |
| `cursor-readout-v4.json`, `cursor-dagger-v4.json` | Adapter v4 (visual slip in lptc) readout and DAgger rounds | [TRAINING.md](../TRAINING.md) |
| `cursor-suite-v4.json`, `cursor-suite-v3b-slip.json` | The suite for the v4 candidate, and the v3b checkpoint scored with the slip fed in | [TRAINING.md](../TRAINING.md) |
| `cursor-readout-v3.json`, `cursor-dagger-v3.json`, `cursor-dagger-v3b.json`, `cursor-dagger-v3c.json` | Adapter v3 (no own-velocity input) readout, its DAgger rounds, the refit on all motor neurons and its on-policy continuation | [TRAINING.md](../TRAINING.md) |
| `cursor-finetune-v1.json` to `cursor-finetune-v4.json` | Gradient fine-tuning of encoders, readout and edge gains; none beat their source | [TRAINING.md](../TRAINING.md) |
| `cursor-parameter-verification.json` | Which tensors a fine-tuned checkpoint changed | [TRAINING.md](../TRAINING.md) |
| `cursor-suite-v1.json`, `cursor-suite-v1b.json`, `cursor-suite-v2.json`, `cursor-suite-v3.json`, `cursor-suite-v3r3.json`, `cursor-suite-v3b.json`, `cursor-suite-v3c.json` | The fixed suite (settle, jump, pursuit, camera) for each checkpoint, four controllers on identical episodes | [TRAINING.md](../TRAINING.md) |

Checkpoints, feature caches and optimiser states stay under the ignored `runs/` folder; the
reports above name them by path and SHA-256.
