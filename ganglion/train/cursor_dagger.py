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
    version = original.get("adapter_version")
    if version not in (2, 3, 4, 5):
        raise ValueError("DAgger requires a cursor adapter version 2, 3 or 4 checkpoint")
    view_fraction = args.view_fraction if args.view_fraction is not None else float(original.get("view_fraction", 0.0))
    slip = dict(original.get("slip") or {})
    slip["slip_gain"] = tuple(slip.get("slip_gain", (1.0, 1.0)))
    goal_scale = float(original.get("goal_scale", .3))
    near_goal_weight = float(args.near_goal_weight if args.near_goal_weight is not None else original.get("near_goal_weight", 1.0))
    max_lag_steps = args.max_lag_steps if args.max_lag_steps is not None else original.get("max_lag_steps")
    brain.eval()
    for parameter in brain.parameters():
        parameter.requires_grad_(False)
    cache = torch.load(args.features, map_location="cpu", weights_only=True)
    check_provenance(cache.get("provenance"), version=version, goal_scale=goal_scale, view_fraction=view_fraction, slip=slip)
    train, validation = cache["train"], cache["validation"]
    seeds = list(cache["splits"]["train"])
    validation_seeds = list(cache["splits"]["validation"])
    report = {"experiment": "connectome cursor readout DAgger", "base_checkpoint": str(base),
              "base_sha256": hashlib.sha256(base.read_bytes()).hexdigest(), "adapter_version": version,
              "plant_version": 2, "rounds": [], "promoted": False, "test_seeds": list(range(4000, 4032)),
              "neurons": brain.N, "edges": int(brain.edge_index.shape[1]),
              "episode_steps": args.episode_steps, "kick_every": args.kick_every or None,
              "student_max": args.student_max, "seed_base": args.seed_base, "view_fraction": view_fraction,
              "slip": slip, "goal_scale": goal_scale, "near_goal_weight": near_goal_weight, "max_lag_steps": max_lag_steps}
    best = None
    for round_index in range(1, args.rounds+1):
        new_seeds = list(range(args.seed_base+round_index*100, args.seed_base+round_index*100+32))
        fraction = min(args.student_max, .25+round_index*.15)
        new = harvest(torch, brain, new_seeds, guard, steps=args.episode_steps, batch=16,
                      student_fraction=fraction, jump_every=args.kick_every or None, sense_version=version,
                      view_fraction=view_fraction, slip=slip, goal_scale=goal_scale)
        train = tuple(torch.cat((old, added)) for old, added in zip(train, new))
        del new
        seeds.extend(new_seeds)
        fitted = fit(torch, brain, train, validation, guard, args.neurons, near_goal_weight, max_lag_steps)
        score = evaluate(torch, brain, validation_seeds, guard, policy="connectome", sense_version=version,
                         goal_scale=goal_scale)
        folder = args.out/f"round-{round_index:02d}"
        folder.mkdir()
        metadata = {"adapter_version": version, "trained": True, "training_domain": "synthetic cursor episodes",
                    "view_fraction": view_fraction, "slip": slip, "goal_scale": goal_scale,
                    "near_goal_weight": near_goal_weight, "max_lag_steps": max_lag_steps,
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
                "splits": {"train": seeds, "validation": validation_seeds},
                "provenance": {"adapter_version": version, "goal_scale": goal_scale,
                               "view_fraction": view_fraction, "slip": slip}}, args.out/"features.pt")
    selected, _, _ = load_checkpoint(best[1], "cuda")
    mlp, report["mlp_fit"] = train_mlp(torch, train, validation, guard, selected.device)
    report["held_out"] = []
    for policy in ("teacher", "mlp", "connectome"):
        score = evaluate(torch, selected, report["test_seeds"], guard, policy=policy, mlp=mlp, sense_version=version,
                         goal_scale=goal_scale)
        report["held_out"].append(score)
        print(json.dumps({"stage": "fresh_test", **score}), flush=True)
    report.update(selected_checkpoint=str(best[1]), peak_gpu_c=guard.peak,
                  elapsed_seconds=time.perf_counter()-started, selection="validation static success, then terminal error")
    (args.out/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"finished": True, "report": str(args.out/"report.json"), "peak_gpu_c": guard.peak}), flush=True)


def check_provenance(provenance, *, version, goal_scale, view_fraction=None, slip=None):
    """A feature cache harvested under another sensory contract than the checkpoint's would
    train a readout on features the runtime never produces; refuse it. Caches from before
    provenance was recorded pass with a warning."""
    if provenance is None:
        print(json.dumps({"stage": "features", "provenance": "unrecorded"}), flush=True)
        return
    if provenance.get("adapter_version") != version:
        raise ValueError(f"The feature cache was harvested with adapter version {provenance.get('adapter_version')}, not {version}")
    if abs(float(provenance.get("goal_scale", .3)) - goal_scale) > 1e-9:
        raise ValueError(f"The feature cache was harvested with goal scale {provenance.get('goal_scale')}, not {goal_scale}")
    if view_fraction is not None and abs(float(provenance.get("view_fraction", 0.0)) - view_fraction) > 1e-9:
        raise ValueError("The feature cache was harvested with another view fraction than the checkpoint's")
    if slip is not None:
        recorded = dict(provenance.get("slip") or {})
        recorded["slip_gain"] = tuple(recorded.get("slip_gain", (1.0, 1.0)))
        if recorded != slip:
            raise ValueError("The feature cache was harvested with other slip options than the checkpoint's")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--neurons", type=int, default=512)
    p.add_argument("--near-goal-weight", type=float, help="ridge weight of the decelerating samples (default: the checkpoint's)")
    p.add_argument("--max-lag-steps", type=int, help="lag limit on the motor neurons the readout may use (default: the checkpoint's)")
    p.add_argument("--seconds", type=float, default=600)
    p.add_argument("--max-gpu-temp", type=float, default=65)
    p.add_argument("--episode-steps", type=int, default=400, help="ticks per harvested episode")
    p.add_argument("--kick-every", type=int, default=0, help="jump the target every N ticks (0: never)")
    p.add_argument("--student-max", type=float, default=.75, help="largest model-driven share of a round")
    p.add_argument("--seed-base", type=int, default=10000, help="first harvest seed; keep rounds of different runs apart")
    p.add_argument("--view-fraction", type=float, help="share of view episodes per round (default: the checkpoint's)")
    args = p.parse_args()
    if args.view_fraction is not None and not 0 <= args.view_fraction <= 1:
        p.error("Use a view fraction between 0 and 1")
    if not 1 <= args.rounds <= 5 or not 32 <= args.neurons <= 4096 or not 30 <= args.seconds <= 1800 or not 50 <= args.max_gpu_temp <= 70:
        p.error("Use rounds 1–5, neurons 32–4096, seconds 30–1800 and temperature 50–70")
    if not 80 <= args.episode_steps <= 2000 or not (args.kick_every == 0 or 50 <= args.kick_every <= 1000):
        p.error("Use episode steps 80–2000 and kick every 0 or 50–1000 ticks")
    if not .25 <= args.student_max <= 1 or not 10000 <= args.seed_base <= 90000:
        p.error("Use student max 0.25–1 and seed base 10000–90000")
    run(args)


if __name__ == "__main__":
    main()
