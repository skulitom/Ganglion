"""DAgger: fit the real connectome readout on states visited by its own policy.

Connectome recurrence and sensory encoders remain frozen in this first experiment.
Candidate selection uses validation episodes; a fresh test split is scored once.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from .cursor_readout import Guard, harvest, fit, evaluate, save_checkpoint, train_mlp


def run(args):
    import torch
    from haltere.train.bptt import load_checkpoint
    args.out.mkdir(parents=True, exist_ok=False)
    guard = Guard(args.seconds, args.max_gpu_temp)
    started = time.perf_counter()
    base = args.checkpoint.resolve()
    brain, cfg, _ = load_checkpoint(base, "cuda")
    if brain.device.type != "cuda" or brain.__class__.__name__ != "ConnectomeRNN":
        raise ValueError("Requires the actual CUDA connectome")
    original = torch.load(base, map_location="cpu", weights_only=True).get("ganglion_cursor", {})
    if original.get("adapter_version") != 2:
        raise ValueError("DAgger requires a cursor adapter-v2 checkpoint")
    brain.eval()
    for parameter in brain.parameters():
        parameter.requires_grad_(False)
    cache = torch.load(args.features, map_location="cpu", weights_only=True)
    train, validation = cache["train"], cache["validation"]
    seeds = list(cache["splits"]["train"])
    validation_seeds = list(cache["splits"]["validation"])
    report = {"experiment": "connectome cursor readout DAgger", "base_checkpoint": str(base),
              "base_sha256": hashlib.sha256(base.read_bytes()).hexdigest(), "adapter_version": 2,
              "plant_version": 2, "rounds": [], "promoted": False, "test_seeds": list(range(4000, 4032)),
              "neurons": brain.N, "edges": int(brain.edge_index.shape[1])}
    best = None
    for round_index in range(1, args.rounds+1):
        new_seeds = list(range(10000+round_index*100, 10000+round_index*100+32))
        fraction = min(.75, .25+round_index*.15)
        new = harvest(torch, brain, new_seeds, guard, steps=400, batch=16, student_fraction=fraction)
        train = tuple(torch.cat((old, added)) for old, added in zip(train, new))
        del new
        seeds.extend(new_seeds)
        fitted = fit(torch, brain, train, validation, guard, args.neurons)
        score = evaluate(torch, brain, validation_seeds, guard, policy="connectome")
        folder = args.out/f"round-{round_index:02d}"
        folder.mkdir()
        metadata = {"adapter_version": 2, "trained": True, "training_domain": "synthetic cursor episodes",
                    "method": "frozen connectome, DAgger ridge motor readout", "control_authority": False,
                    "training_seeds": seeds.copy(), "validation_seeds": validation_seeds,
                    "readout_fit": fitted, "parent_sha256": report["base_sha256"]}
        checkpoint = save_checkpoint(torch, brain, cfg, folder, metadata)
        row = {"round": round_index, "student_fraction": fraction, "training_samples": len(train[0]),
               "fit": fitted, "validation_closed_loop": score, "checkpoint": str(checkpoint)}
        report["rounds"].append(row)
        key = (score["static_episodes"]-score["static_settled_within_6px"], score["mean_terminal_error_px"])
        if best is None or key < best[0]:
            best = key, checkpoint
        print(json.dumps(row), flush=True)
        (args.out/"progress.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Keep raw motor-learning features local under the ignored output directory.
    torch.save({"train": train, "validation": validation,
                "splits": {"train": seeds, "validation": validation_seeds}}, args.out/"features.pt")
    selected, _, _ = load_checkpoint(best[1], "cuda")
    mlp, report["mlp_fit"] = train_mlp(torch, train, validation, guard, selected.device)
    report["held_out"] = []
    for policy in ("teacher", "mlp", "connectome"):
        score = evaluate(torch, selected, report["test_seeds"], guard, policy=policy, mlp=mlp)
        report["held_out"].append(score)
        print(json.dumps({"stage": "fresh_test", **score}), flush=True)
    report.update(selected_checkpoint=str(best[1]), peak_gpu_c=guard.peak,
                  elapsed_seconds=time.perf_counter()-started, selection="validation static success, then terminal error")
    (args.out/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"finished": True, "report": str(args.out/"report.json"), "peak_gpu_c": guard.peak}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--neurons", type=int, default=512)
    p.add_argument("--seconds", type=float, default=600)
    p.add_argument("--max-gpu-temp", type=float, default=65)
    args = p.parse_args()
    if not 1 <= args.rounds <= 5 or not 32 <= args.neurons <= 2048 or not 30 <= args.seconds <= 1800 or not 50 <= args.max_gpu_temp <= 70:
        p.error("Use rounds 1–5, neurons 32–2048, seconds 30–1800 and temperature 50–70")
    run(args)


if __name__ == "__main__":
    main()
