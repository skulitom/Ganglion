"""Train the connectome's interfaces, and optionally its constrained recurrence, for cursor control.

The recurrent structure and signs stay those of the connectome. By default the sensory encoders
and the motor readout adapt through truncated BPTT on mixed teacher/student trajectories;
``--train edges`` also adapts the per-edge log-gains under the connectome prior, and
``--train neurons`` the per-neuron gains, biases and time constants. Candidates are selected on
the fixed suite (settling and pursuit on validation episodes, the connectome acting alone), the
unchanged source stays a candidate, and the selected checkpoint is then scored on the suite's
test episodes alone and under the supervising envelope.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .cursor_readout import Guard, observe, save_checkpoint
from .cursor_world import CursorWorld
from .suite import SEEDS, TASKS, connectome_proposer, run_task

GROUPS = {"encoders": ("encoders.",), "readout": ("readout.",), "edges": ("log_edge_gain",),
          "neurons": ("log_gain", "bias", "log_tau")}
VALIDATION = {"settle": list(range(40000, 40016)), "pursuit": list(range(41000, 41016))}


def parameter_groups(brain, train):
    """Freeze everything but the requested groups; return the trainable parameters per group."""
    unknown = set(train) - set(GROUPS)
    if unknown:
        raise ValueError(f"Unknown parameter groups {sorted(unknown)}")
    groups = {name: [] for name in train}
    for name, parameter in brain.named_parameters():
        owner = next((g for g in train if name.startswith(GROUPS[g])), None)
        parameter.requires_grad_(owner is not None)
        if owner is not None:
            groups[owner].append(parameter)
    return groups


def suite_scores(torch, brain, guard, seeds_by_task, steps, controllers=("connectome",), sense_version=2, goal_scale=.3):
    """The fixed suite's tasks with this brain proposing, one fresh network state per run."""
    brain.eval()
    scores = {}
    for task, seeds in seeds_by_task.items():
        for controller in controllers:
            scores[f"{task}/{controller}"] = run_task(task, seeds, controller, steps=steps, guard=guard,
                                                     propose=connectome_proposer(torch, brain, len(seeds)),
                                                     sense_version=sense_version, goal_scale=goal_scale)
    return scores


def selection_key(scores):
    """Fewest unsettled static targets, then the lowest pursuit tracking error: the connectome alone."""
    settle, pursuit = scores["settle/connectome"], scores["pursuit/connectome"]
    error = pursuit["tracking_error_px"]["mean"]
    return settle["episodes"] - settle["success"], float("inf") if error is None else float(error)


def brief(scores):
    return {name: {"success": s["success"], "episodes": s.get("reaches", s["episodes"]),
                   "tracking_error_px": s["tracking_error_px"]["mean"],
                   **({"lost": s["lost"]} if "lost" in s else {}),
                   **({"interventions": s["interventions"]["fraction"]} if "interventions" in s else {})}
            for name, s in scores.items()}


def run(args):
    import torch
    from haltere.train.bptt import load_checkpoint
    args.out.mkdir(parents=True, exist_ok=False)
    guard = Guard(args.seconds, args.max_gpu_temp, pace=.002)
    started = time.perf_counter()
    base = args.checkpoint.resolve()
    brain, cfg, _ = load_checkpoint(base, "cuda")
    source = torch.load(base, map_location="cpu", weights_only=True).get("ganglion_cursor", {})
    version = source.get("adapter_version")
    goal_scale = float(source.get("goal_scale", .3))
    view_fraction = float(source.get("view_fraction", 0.0))
    slip = dict(source.get("slip") or {})
    slip["slip_gain"] = tuple(slip.get("slip_gain", (1.0, 1.0)))
    if brain.device.type != "cuda" or brain.__class__.__name__ != "ConnectomeRNN" or version not in (2, 3, 4, 5):
        raise ValueError("Requires an adapter version 2, 3 or 4 cursor ConnectomeRNN on CUDA")
    # Keep whitening fixed while learning, so batch composition never subtracts
    # the signal. Structure and signs stay those of the connectome.
    brain.eval()
    train = [g.strip() for g in args.train.split(",") if g.strip()]
    groups = parameter_groups(brain, train)
    rates = {"encoders": args.encoder_lr, "readout": args.readout_lr, "edges": args.edge_lr, "neurons": args.neuron_lr}
    if args.reset_readout:
        with torch.no_grad():
            brain.readout.weight.zero_()
            brain.readout.bias.zero_()
    if args.edge_prior is not None:
        brain.cfg.edge_prior = args.edge_prior
    optimizer = torch.optim.Adam([{"params": groups[g], "lr": rates[g]} for g in train if groups[g]])
    trainable = [p for g in train for p in groups[g]]
    recurrent = "edges" in train or "neurons" in train
    weights = None if recurrent else brain.weight_matrix().detach()   # recomputed each step when it learns
    report = {"experiment": "connectome cursor fine-tuning selected on the fixed suite",
              "base_checkpoint": str(base), "base_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
              "adapter_version": version, "goal_scale": goal_scale, "view_fraction": view_fraction, "slip": slip,
              "plant_version": 2, "neurons": brain.N,
              "edges": int(brain.edge_index.shape[1]), "trained_groups": train,
              "trainable_parameters": {g: int(sum(p.numel() for p in groups[g])) for g in train},
              "recurrent_structure_and_signs_frozen": True, "recurrent_magnitudes_trained": "edges" in train,
              "neuron_parameters_trained": "neurons" in train, "edge_prior": brain.cfg.edge_prior,
              "reset_readout": args.reset_readout, "learning_rates": {g: rates[g] for g in train},
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(brain.device),
              "batch": 16, "window_steps": args.window, "truncate_steps": args.truncate,
              "episode_ticks": args.episode_ticks, "kick_every": args.kick_every or None,
              "student_max": args.student_max, "validation_seeds": VALIDATION, "validation_steps": args.validation_steps,
              "test_seeds": SEEDS, "history": [], "validation": [], "promoted": False}
    (args.out/"config.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # The unchanged source is a candidate, so training cannot silently replace it with worse.
    baseline_scores = suite_scores(torch, brain, guard, VALIDATION, args.validation_steps, sense_version=version,
                                   goal_scale=goal_scale)
    best = (selection_key(baseline_scores), base, baseline_scores)
    report["validation"].append({"iteration": 0, "checkpoint": str(base), "scores": brief(baseline_scores),
                                 "key": list(best[0])})
    print(json.dumps({"stage": "baseline_validation", "scores": brief(baseline_scores)}), flush=True)
    episode = 0
    world, state, previous = None, None, None
    for iteration in range(1, args.iterations+1):
        guard.check(force=True)
        if world is None or world.tick >= args.episode_ticks:
            episode += 1
            world = CursorWorld(range(100000+episode*16, 100000+(episode+1)*16),
                                jump_every=args.kick_every or None, sense_version=version, goal_scale=goal_scale,
                                view=np.arange(16) < int(round(16 * view_fraction)), **slip)
            state, previous = brain.init_state(16), None
        student_fraction = min(args.student_max, max(0, (iteration-20)/max(1, args.iterations-20))*args.student_max)
        students = int(16*student_fraction)
        loss = torch.zeros((), device=brain.device)
        regulariser = torch.zeros((), device=brain.device)
        for tick in range(args.window):
            if tick and tick % args.truncate == 0:
                state = brain.detach_state(state)
            obs, _ = observe(torch, brain, world.senses(previous))
            action, state, aux = brain(obs, state, weights)
            target = world.teacher()
            labels = torch.from_numpy(((target-world.cursor)/(world.speed[:, None]*.01)).astype(np.float32)).to(brain.device)
            loss = loss + (action[:, :2]-labels.clamp(-1, 1)).square().mean()
            if recurrent:
                regulariser = regulariser + brain.regularization(aux)
            driven = target.copy()
            if students:
                driven[:students] = np.rint(world.cursor[:students] + action[:students, :2].detach().cpu().numpy()*world.speed[:students, None]*.01)
                driven = np.clip(driven, world.rect[:, :2], world.rect[:, :2]+world.rect[:, 2:]-1)
            previous = world.cursor.copy()
            world.step(driven)
        loss = loss / args.window
        total = loss + regulariser / args.window
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite training loss; stop without replacing saved checkpoints")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        grad = float(torch.nn.utils.clip_grad_norm_(trainable, 1, error_if_nonfinite=True))
        optimizer.step()
        state = brain.detach_state(state)
        row = {"iteration": iteration, "mse": float(loss.detach()), "regulariser": float(regulariser.detach()) / args.window,
               "gradient_norm": grad, "student_fraction": student_fraction, "peak_gpu_c": guard.peak,
               "elapsed_seconds": time.perf_counter()-started}
        report["history"].append(row)
        if iteration % 10 == 0:
            print(json.dumps(row), flush=True)
        if iteration % args.validate_every == 0 or iteration == args.iterations:
            scores = suite_scores(torch, brain, guard, VALIDATION, args.validation_steps, sense_version=version,
                                  goal_scale=goal_scale)
            folder = args.out/f"iteration-{iteration:04d}"
            folder.mkdir()
            metadata = {"adapter_version": version, "trained": True, "training_domain": "synthetic cursor episodes",
                        "goal_scale": goal_scale, "view_fraction": view_fraction, "slip": slip,
                        "method": "truncated-BPTT DAgger on " + "+".join(train) + ", selected on the fixed suite",
                        "control_authority": False, "recurrent_structure_and_signs_frozen": True,
                        "recurrent_magnitudes_trained": "edges" in train, "neuron_parameters_trained": "neurons" in train,
                        "parent_sha256": report["base_sha256"], "learning_rates": report["learning_rates"],
                        "iteration": iteration, "training_seed_start": 100016,
                        "training_seed_end": 100000+(episode+1)*16-1, "validation_seeds": VALIDATION}
            checkpoint = save_checkpoint(torch, brain, cfg, folder, metadata)
            torch.save({"optimizer": optimizer.state_dict(), "iteration": iteration}, folder/"optimizer.pt")
            key = selection_key(scores)
            record = {"iteration": iteration, "checkpoint": str(checkpoint), "scores": brief(scores), "key": list(key)}
            report["validation"].append(record)
            if key < best[0]:
                best = (key, checkpoint, scores)
            print(json.dumps({"stage": "validation", **record}), flush=True)
            (args.out/"progress.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        time.sleep(.01)
    selected, _, _ = load_checkpoint(best[1], "cuda")
    selected.eval()
    report["suite"] = {}
    for task in TASKS:
        report["suite"][task] = {}
        for controller in ("connectome", "supervised"):
            score = run_task(task, SEEDS[task], controller, steps=args.test_steps, guard=guard,
                             propose=connectome_proposer(torch, selected, len(SEEDS[task])), sense_version=version,
                             goal_scale=goal_scale)
            report["suite"][task][controller] = score
            print(json.dumps({"stage": "suite", **{k: v for k, v in score.items() if k not in ("per_episode", "seeds")}}), flush=True)
    report.update(selected_checkpoint=str(best[1]), selected_key=list(best[0]),
                  selected_checkpoint_sha256=hashlib.sha256(Path(best[1]).read_bytes()).hexdigest(),
                  peak_gpu_c=guard.peak, training_samples=args.iterations*args.window*16,
                  elapsed_seconds=time.perf_counter()-started)
    (args.out/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"finished": True, "report": str(args.out/"report.json"), "selected": str(best[1]),
                      "peak_gpu_c": guard.peak}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--iterations", type=int, default=200)
    p.add_argument("--seconds", type=float, default=900)
    p.add_argument("--max-gpu-temp", type=float, default=65)
    p.add_argument("--reset-readout", action="store_true")
    p.add_argument("--train", default="encoders,readout", help="comma list of encoders, readout, edges, neurons")
    p.add_argument("--encoder-lr", type=float, default=1e-6)
    p.add_argument("--readout-lr", type=float, default=1e-5)
    p.add_argument("--edge-lr", type=float, default=1e-4, help="per-edge log-gain rate when edges are trained")
    p.add_argument("--neuron-lr", type=float, default=1e-5, help="per-neuron gain, bias and time-constant rate")
    p.add_argument("--edge-prior", type=float, help="override the L2 pull of edge log-gains towards the connectome")
    p.add_argument("--episode-ticks", type=int, default=192, help="ticks before a fresh batch of episodes")
    p.add_argument("--window", type=int, default=32, help="ticks per gradient update")
    p.add_argument("--truncate", type=int, default=8, help="ticks between state detachments inside a window")
    p.add_argument("--kick-every", type=int, default=0, help="jump the targets every N ticks (0: never)")
    p.add_argument("--student-max", type=float, default=.75, help="largest model-driven share of the batch")
    p.add_argument("--validate-every", type=int, default=50)
    p.add_argument("--validation-steps", type=int, default=1000)
    p.add_argument("--test-steps", type=int, default=2000)
    args = p.parse_args()
    if not 50 <= args.iterations <= 1000 or not 30 <= args.seconds <= 1800 or not 50 <= args.max_gpu_temp <= 70:
        p.error("Use iterations 50–1000, seconds 30–1800 and temperature 50–70")
    for rate in (args.encoder_lr, args.readout_lr, args.edge_lr, args.neuron_lr):
        if not 1e-7 <= rate <= 1e-2:
            p.error("Use learning rates between 1e-7 and 1e-2")
    if not 96 <= args.episode_ticks <= 4000 or not 8 <= args.window <= 256 or not 1 <= args.truncate <= args.window:
        p.error("Use episode ticks 96–4000, window 8–256 and truncation 1–window")
    if not (args.kick_every == 0 or 50 <= args.kick_every <= 2000) or not 0 <= args.student_max <= 1:
        p.error("Use kick every 0 or 50–2000 ticks and student max 0–1")
    if not 10 <= args.validate_every <= 500 or not 300 <= args.validation_steps <= 2000 or not 500 <= args.test_steps <= 4000:
        p.error("Use validate every 10–500, validation steps 300–2000 and test steps 500–4000")
    if args.edge_prior is not None and not 0 <= args.edge_prior <= 1:
        p.error("Use an edge prior between 0 and 1")
    run(args)


if __name__ == "__main__":
    main()
