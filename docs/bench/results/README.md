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
| `transfer-seat-deterministic.json`, `transfer-seat-connectome.json`, `transfer-seat-connectome-all-motor.json`, `transfer-seat.json` | The Arena reach demo in the seat with the reference, the 512-neuron readout and the all-motor readout under supervised authority, and their ledger comparison | [SHADOW.md](../SHADOW.md) |
| `solitaire-seat/`, `solitaire-live/` | Taught Sawayama board runs and the live play harness, deterministic and connectome | [SOLITAIRE.md](../SOLITAIRE.md) |
| `halflife/` | The Half-Life pilot session ledger summary and screenshots | [HALFLIFE.md](../HALFLIFE.md) |

## Cursor training

| File | What it records | Quoted in |
|---|---|---|
| `cursor-readout-v2.json` | Ridge readout on the frozen connectome, adapter v2, with the MLP comparison | [TRAINING.md](../TRAINING.md) |
| `cursor-dagger-v1.json` | Three DAgger rounds on that readout; its round 3 is the v2 candidate | [TRAINING.md](../TRAINING.md) |
| `cursor-dagger-v2.json` | DAgger with 1,000-tick kicked episodes; did not improve | [TRAINING.md](../TRAINING.md) |
| `cursor-dagger-v1b.json` | Refit on all 3,913 motor neurons; the best supervised controller so far | [TRAINING.md](../TRAINING.md) |
| `cursor-readout-v3.json`, `cursor-dagger-v3.json` | Adapter v3 (no own-velocity input) readout and its DAgger rounds | [TRAINING.md](../TRAINING.md) |
| `cursor-finetune-v1.json` to `cursor-finetune-v4.json` | Gradient fine-tuning of encoders, readout and edge gains; none beat their source | [TRAINING.md](../TRAINING.md) |
| `cursor-parameter-verification.json` | Which tensors a fine-tuned checkpoint changed | [TRAINING.md](../TRAINING.md) |
| `cursor-suite-v1.json`, `cursor-suite-v1b.json`, `cursor-suite-v2.json`, `cursor-suite-v3.json`, `cursor-suite-v3r3.json` | The fixed suite (settle, jump, pursuit, camera) for each checkpoint, four controllers on identical episodes | [TRAINING.md](../TRAINING.md) |

Checkpoints, feature caches and optimiser states stay under the ignored `runs/` folder; the
reports above name them by path and SHA-256.
