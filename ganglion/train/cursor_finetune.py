"""Train sensory encoders and motor readout through the real fly connectome.

The recurrent connectome, signs, neuron gains, biases and time constants remain
fixed. Truncated BPTT and mixed teacher/student trajectories adapt its interfaces.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .cursor_readout import Guard, observe, evaluate, save_checkpoint
from .cursor_world import CursorWorld


def run(args):
    import torch
    from haltere.train.bptt import load_checkpoint
    args.out.mkdir(parents=True, exist_ok=False)
    guard = Guard(args.seconds, args.max_gpu_temp, pace=.005)
    started = time.perf_counter()
    base = args.checkpoint.resolve()
    brain, cfg, _ = load_checkpoint(base, "cuda")
    source = torch.load(base, map_location="cpu", weights_only=True).get("ganglion_cursor", {})
    if brain.device.type != "cuda" or brain.__class__.__name__ != "ConnectomeRNN" or source.get("adapter_version") != 2:
        raise ValueError("Requires an adapter-v2 cursor ConnectomeRNN on CUDA")
    # Keep whitening fixed while learning, so batch composition never subtracts
    # the signal. Recurrence stays biologically constrained and unchanged.
    brain.eval()
    encoders, readout = [], []
    for name, parameter in brain.named_parameters():
        parameter.requires_grad_(name.startswith(("encoders.", "readout.")))
        if name.startswith("encoders."):
            encoders.append(parameter)
        elif name.startswith("readout."):
            readout.append(parameter)
    if args.reset_readout:
        with torch.no_grad():
            brain.readout.weight.zero_()
            brain.readout.bias.zero_()
    optimizer = torch.optim.Adam([{"params": encoders, "lr": args.encoder_lr},
                                  {"params": readout, "lr": args.readout_lr}])
    weights = brain.weight_matrix().detach()
    report = {"experiment": "connectome encoder/readout DAgger fine-tuning",
              "base_checkpoint": str(base), "base_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
              "adapter_version": 2, "plant_version": 2, "neurons": brain.N,
              "edges": int(brain.edge_index.shape[1]), "trainable_parameters": sum(p.numel() for p in encoders+readout),
              "recurrent_parameters_frozen": True, "reset_readout": args.reset_readout,
              "encoder_lr": args.encoder_lr, "readout_lr": args.readout_lr,
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(brain.device),
              "batch": 16, "window_steps": args.window, "truncate_steps": args.truncate,
              "episode_ticks": args.episode_ticks, "kick_every": args.kick_every or None,
              "student_max": args.student_max,
              "validation_seeds": list(range(2000, 2016)),
              "test_seeds": list(range(args.test_seed_start, args.test_seed_start+32)),
              "history": [], "validation": [], "promoted": False}
    (args.out/"config.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    episode = 0
    world, state, previous = None, None, None
    # Include the unchanged source in selection, so extra training cannot silently
    # replace it with a worse validation candidate.
    baseline, _, _ = load_checkpoint(base, "cuda")
    baseline_score = evaluate(torch, baseline, report["validation_seeds"], guard, policy="connectome")
    report["validation"].append({"iteration": 0, "score": baseline_score, "checkpoint": str(base)})
    best = ((baseline_score["static_episodes"]-baseline_score["static_settled_within_6px"],
             baseline_score["mean_terminal_error_px"]), base)
    del baseline
    print(json.dumps({"stage": "baseline_validation", "score": baseline_score}), flush=True)
    for iteration in range(1, args.iterations+1):
        guard.check(force=True)
        if world is None or world.tick >= args.episode_ticks:
            episode += 1
            world = CursorWorld(range(100000+episode*16, 100000+(episode+1)*16),
                                jump_every=args.kick_every or None)
            state, previous = brain.init_state(16), None
        student_fraction = min(args.student_max, max(0, (iteration-20)/(args.iterations-20))*args.student_max)
        students = int(16*student_fraction)
        loss = torch.zeros((), device=brain.device)
        for tick in range(args.window):
            if tick and tick % args.truncate == 0:
                state = brain.detach_state(state)
            obs, _ = observe(torch, brain, world.senses(previous))
            action, state, _ = brain(obs, state, weights)
            target = world.teacher()
            labels = torch.from_numpy(((target-world.cursor)/(world.speed[:, None]*.01)).astype(np.float32)).to(brain.device)
            loss = loss + (action[:, :2]-labels.clamp(-1, 1)).square().mean()
            driven = target.copy()
            if students:
                driven[:students] = np.rint(world.cursor[:students] + action[:students, :2].detach().cpu().numpy()*world.speed[:students, None]*.01)
                driven = np.clip(driven, world.rect[:, :2], world.rect[:, :2]+world.rect[:, 2:]-1)
            previous = world.cursor.copy()
            world.step(driven)
        loss = loss / args.window
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss; stop without replacing saved checkpoints")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(encoders+readout, 1, error_if_nonfinite=True))
        optimizer.step()
        state = brain.detach_state(state)
        row = {"iteration": iteration, "mse": float(loss.detach()), "gradient_norm": grad,
               "student_fraction": student_fraction, "peak_gpu_c": guard.peak,
               "elapsed_seconds": time.perf_counter()-started}
        report["history"].append(row)
        if iteration % 10 == 0:
            print(json.dumps(row), flush=True)
        if iteration % 50 == 0 or iteration == args.iterations:
            score = evaluate(torch, brain, report["validation_seeds"], guard, policy="connectome")
            folder = args.out/f"iteration-{iteration:04d}"
            folder.mkdir()
            metadata = {"adapter_version": 2, "trained": True, "training_domain": "synthetic cursor episodes",
                        "method": "encoder/readout truncated-BPTT DAgger", "control_authority": False,
                        "recurrent_parameters_frozen": True, "parent_sha256": report["base_sha256"],
                        "encoder_lr": args.encoder_lr, "readout_lr": args.readout_lr,
                        "iteration": iteration, "training_seed_start": 100016,
                        "training_seed_end": 100000+(episode+1)*16-1,
                        "validation_seeds": report["validation_seeds"]}
            checkpoint = save_checkpoint(torch, brain, cfg, folder, metadata)
            torch.save({"optimizer": optimizer.state_dict(), "iteration": iteration}, folder/"optimizer.pt")
            record = {"iteration": iteration, "score": score, "checkpoint": str(checkpoint)}
            report["validation"].append(record)
            key = (score["static_episodes"]-score["static_settled_within_6px"], score["mean_terminal_error_px"])
            if best is None or key < best[0]:
                best = key, checkpoint
            print(json.dumps({"stage": "validation", **record}), flush=True)
            (args.out/"progress.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        time.sleep(.02)
    selected, _, _ = load_checkpoint(best[1], "cuda")
    report["held_out"] = []
    for policy in ("teacher", "connectome"):
        score = evaluate(torch, selected, report["test_seeds"], guard, policy=policy)
        report["held_out"].append(score)
        print(json.dumps({"stage": "fresh_test", **score}), flush=True)
    report.update(selected_checkpoint=str(best[1]),
                  selected_checkpoint_sha256=hashlib.sha256(best[1].read_bytes()).hexdigest(), peak_gpu_c=guard.peak,
                  training_samples=args.iterations*args.window*16,
                  elapsed_seconds=time.perf_counter()-started)
    (args.out/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"finished": True, "report": str(args.out/"report.json"), "peak_gpu_c": guard.peak}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--seconds", type=float, default=900)
    p.add_argument("--max-gpu-temp", type=float, default=65)
    p.add_argument("--reset-readout", action="store_true")
    p.add_argument("--encoder-lr", type=float, default=1e-6)
    p.add_argument("--readout-lr", type=float, default=1e-5)
    p.add_argument("--test-seed-start", type=int, default=5000)
    p.add_argument("--episode-ticks", type=int, default=192, help="ticks before a fresh batch of episodes")
    p.add_argument("--window", type=int, default=32, help="ticks per gradient update")
    p.add_argument("--truncate", type=int, default=8, help="ticks between state detachments inside a window")
    p.add_argument("--kick-every", type=int, default=0, help="jump the targets every N ticks (0: never)")
    p.add_argument("--student-max", type=float, default=.75, help="largest model-driven share of the batch")
    args = p.parse_args()
    if not 96 <= args.episode_ticks <= 4000 or not 8 <= args.window <= 256 or not 1 <= args.truncate <= args.window:
        p.error("Use episode ticks 96–4000, window 8–256 and truncation 1–window")
    if not (args.kick_every == 0 or 50 <= args.kick_every <= 2000) or not 0 <= args.student_max <= 1:
        p.error("Use kick every 0 or 50–2000 ticks and student max 0–1")
    if not 50 <= args.iterations <= 500 or not 30 <= args.seconds <= 1800 or not 50 <= args.max_gpu_temp <= 70:
        p.error("Use iterations 50–500, seconds 30–1800 and temperature 50–70")
    if not 1e-7 <= args.encoder_lr <= 1e-3 or not 1e-7 <= args.readout_lr <= 1e-3:
        p.error("Use learning rates between 1e-7 and 1e-3")
    if not 5000 <= args.test_seed_start <= 9000:
        p.error("Use test seed start 5000–9000, disjoint from training and validation")
    run(args)


if __name__ == "__main__":
    main()
