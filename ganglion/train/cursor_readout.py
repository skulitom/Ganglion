"""Fit a cursor readout on the actual frozen Haltere connectome, then test closed loop.

No application-specific data or direct sensory-to-output bypass is used by the
connectome candidate. This is a reservoir/readout experiment, not full BPTT.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from math import isfinite
from pathlib import Path
import time

import numpy as np

from .cursor_world import CursorWorld


class Guard:
    def __init__(self, seconds, temperature=65, pace=.003, *, clock=time.perf_counter,
                 sleep=time.sleep, temperature_reader=None):
        self.clock, self.sleep = clock, sleep
        if temperature_reader is None:
            from haltere.train.thermal import gpu_temperature
            temperature_reader = gpu_temperature
        self.temperature_reader = temperature_reader
        self.deadline = clock() + seconds
        self.limit, self.pace, self.peak = temperature, pace, 0
        self.ticks = 0
        self.check(force=True)

    def check(self, *, force=False):
        if self.clock() >= self.deadline:
            raise TimeoutError("Training wall-time budget exhausted")
        self.ticks += 1
        if force or self.ticks % 20 == 0:
            temp = self.temperature_reader()
            if temp is None or not isfinite(temp):
                raise RuntimeError("GPU temperature unavailable; stop training")
            self.peak = max(self.peak, temp)
            if temp > self.limit:
                print(f"Pausing at {temp:.0f} C; resume below {self.limit-8:.0f} C", flush=True)
                while temp > self.limit-8:
                    if self.clock() >= self.deadline:
                        raise TimeoutError("Temperature pause exhausted training budget")
                    self.sleep(1)
                    temp = self.temperature_reader()
                    if temp is None or not isfinite(temp):
                        raise RuntimeError("GPU temperature unavailable while paused")
                    self.peak = max(self.peak, temp)
        self.sleep(self.pace)


def observe(torch, brain, senses):
    keys = tuple(brain.channel_dims)
    packed = torch.from_numpy(np.concatenate([senses[k] for k in keys], axis=1)).to(brain.device)
    obs, offset = {}, 0
    for key in keys:
        obs[key] = packed[:, offset:offset+brain.channel_dims[key]]
        offset += brain.channel_dims[key]
    return obs, packed


def slip_options(args):
    """The world's slip robustness options from the command line (or a checkpoint's record)."""
    return {"slip_dropout": float(args.slip_dropout), "slip_blank": float(args.slip_blank),
            "slip_gain": tuple(float(g) for g in args.slip_gain)}


def harvest(torch, brain, seeds, guard, *, steps, batch, student_fraction=0, jump_every=None, sense_version=2,
            view_fraction=0.0, slip=None, goal_scale=.3):
    features, inputs, labels = [], [], []
    weights = brain.weight_matrix().detach()
    for offset in range(0, len(seeds), batch):
        chunk = seeds[offset:offset+batch]
        view = np.arange(len(chunk)) < int(round(len(chunk) * view_fraction))   # the first ones are views
        world = CursorWorld(chunk, steps=steps, jump_every=jump_every, sense_version=sense_version, view=view,
                            goal_scale=goal_scale, **(slip or {}))
        state, previous = brain.init_state(world.B), None
        with torch.inference_mode():
            for _ in range(steps):
                guard.check()
                obs, packed = observe(torch, brain, world.senses(previous))
                action, state, _ = brain(obs, state, weights)
                rates = brain.cfg.rate_max * torch.sigmoid(state["v"][brain.motor_idx]).T
                target = world.teacher()
                command = (target-world.cursor)/(world.speed[:, None]*.01)
                features.append(rates.cpu())
                inputs.append(packed.cpu())
                labels.append(torch.from_numpy(np.clip(command, -1, 1).astype(np.float32)))
                previous = world.cursor.copy()
                driven = target.copy()
                students = int(world.B*student_fraction)
                if students:
                    driven[:students] = np.rint(world.cursor[:students] + action[:students, :2].cpu().numpy()*world.speed[:students, None]*.01)
                    bounds = world.bounds
                    driven = np.clip(driven, bounds[:, :2], bounds[:, :2]+bounds[:, 2:]-1)
                world.step(driven)
        print(json.dumps({"stage": "harvest", "episodes": min(offset+batch, len(seeds)),
                          "total_episodes": len(seeds), "peak_gpu_c": guard.peak}), flush=True)
    return tuple(torch.cat(values).clone() for values in (features, inputs, labels))


def install_head(torch, brain, mean, variance, selected, weights, bias):
    with torch.no_grad():
        brain.motor_norm.running_mean.copy_(mean.to(brain.device))
        brain.motor_norm.running_var.copy_(variance.to(brain.device))
        brain.readout.weight.zero_()
        brain.readout.bias.zero_()
        brain.readout.weight[:2, selected.to(brain.device)] = weights.T.to(brain.device)
        brain.readout.bias[:2].copy_(bias.to(brain.device))
    brain.cfg.action_tau = brain.cfg.dt


def probe_lags(torch, brain, steps=30):
    """Each motor neuron's response lag to a reversal of the goal direction, in network steps:
    the network is driven with a unit goal direction along one axis until it settles, the
    direction is reversed, and the lag is the first step at which the neuron has covered half of
    its eventual change. The worst of the four axis reversals is kept; a neuron that does not
    respond gets the probe length. The goal is fed as adapter version 5 would (a unit direction
    with the distance slot at 0.5), which every version's network accepts."""
    weights = brain.weight_matrix().detach()
    dims = dict(brain.channel_dims)

    def observation(goal):
        o = {k: torch.zeros(1, d, device=brain.device) for k, d in dims.items()}
        o["goal"] = torch.tensor([goal], device=brain.device, dtype=torch.float32)
        return o

    def rates(state):
        return (brain.cfg.rate_max * torch.sigmoid(state["v"][brain.motor_idx])).reshape(-1)

    worst = None
    with torch.inference_mode():
        for axis in (0, 1):
            for sign in (1.0, -1.0):
                state = brain.init_state(1)
                for _ in range(steps):
                    _, state, _ = brain(observation([0.0, 0.0, 0.0, 0.0]), state, weights)
                goal = [0.0, 0.0, 0.0, 0.5]
                goal[axis] = sign
                for _ in range(steps):
                    _, state, _ = brain(observation(goal), state, weights)
                before = rates(state).clone()
                goal[axis] = -sign
                trace = []
                for _ in range(steps):
                    _, state, _ = brain(observation(goal), state, weights)
                    trace.append(rates(state).clone())
                trace = torch.stack(trace)                                   # (steps, motor)
                change = trace[-1] - before
                covered = (trace - before).abs() >= .5 * change.abs()
                lag = torch.where(covered.any(0), covered.float().argmax(0), torch.full_like(change, steps, dtype=torch.long))
                lag = torch.where(change.abs() < 1e-3 * brain.cfg.rate_max, torch.full_like(lag, steps), lag)
                worst = lag if worst is None else torch.maximum(worst, lag)
    return worst.cpu()


def near_goal_weights(torch, target, near_goal_weight):
    """Sample weights for the ridge fit: `near_goal_weight` on the samples where the teacher is
    already decelerating (its command shorter than a full step), 1 elsewhere."""
    weights = torch.ones(len(target), dtype=torch.float64)
    if near_goal_weight != 1:
        weights[target.norm(dim=1) < .98] = float(near_goal_weight)
    return weights


def fit(torch, brain, train, validation, guard, feature_count, near_goal_weight=1.0, max_lag_steps=None):
    raw, _, target = train
    val_raw, _, val_target = validation
    mean, variance = raw.mean(0), raw.var(0, unbiased=False)
    std = (variance+brain.motor_norm.eps).sqrt()
    x = (raw-mean)/std
    centered = target-target.mean(0)
    # Select responsive motor neurons using training labels only. Validation and
    # test samples never participate in feature selection or normalisation.
    correlation = (x.T @ centered).abs() / centered.square().sum(0).sqrt().clamp_min(1e-6)
    score = correlation.amax(1)
    fast = None
    if max_lag_steps is not None:
        # Only neurons that answer a reversal of the goal direction within `max_lag_steps`
        # network steps: the readout must change sign in time to stop.
        lags = probe_lags(torch, brain)
        fast = lags <= max_lag_steps
        if int(fast.sum()) < 8:
            raise ValueError(f"Only {int(fast.sum())} motor neurons answer within {max_lag_steps} steps; nothing to fit")
        score = torch.where(fast, score, torch.full_like(score, -1.0))
    selected = score.topk(min(feature_count, int((score >= 0).sum()))).indices.sort().values
    x = x[:, selected].to(brain.device, dtype=torch.float64)
    val_x = ((val_raw-mean)/std)[:, selected].to(brain.device)
    y = torch.atanh(target.clamp(-.995, .995)).to(brain.device, dtype=torch.float64)
    w = near_goal_weights(torch, target, near_goal_weight).to(brain.device)
    bias = (w[:, None] * y).sum(0) / w.sum()
    gram, rhs = x.T @ (w[:, None] * x) / w.sum(), x.T @ (w[:, None] * (y-bias)) / w.sum()
    trials, best = [], None
    for penalty in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        guard.check(force=True)
        weights = torch.linalg.solve(gram + penalty*torch.eye(len(selected), device=brain.device, dtype=torch.float64), rhs).float()
        prediction = torch.tanh(val_x @ weights + bias.float()).cpu()
        mse = float((prediction-val_target).square().mean())
        trials.append({"ridge": penalty, "validation_mse": mse})
        if best is None or mse < best[0]:
            best = mse, penalty, weights.cpu(), bias.float().cpu()
    install_head(torch, brain, mean, variance, selected, best[2], best[3])
    return {"selected_motor_neurons": len(selected), "ridge": best[1], "near_goal_weight": float(near_goal_weight),
            "max_lag_steps": max_lag_steps, "fast_motor_neurons": None if fast is None else int(fast.sum()),
            "validation_mse": best[0], "candidates": trials}


def build_mlp(torch):
    """The non-connectome comparison: the same 24 sensor values to a cursor velocity."""
    return torch.nn.Sequential(torch.nn.Linear(24, 64), torch.nn.Tanh(),
                               torch.nn.Linear(64, 64), torch.nn.Tanh(),
                               torch.nn.Linear(64, 2), torch.nn.Tanh())


def train_mlp(torch, train, validation, guard, device):
    torch.manual_seed(734)
    model = build_mlp(torch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=.002)
    _, x, y = train
    x, y = x.to(device), y.to(device)
    vx, vy = validation[1].to(device), validation[2].to(device)
    best, saved = float("inf"), None
    for i in range(600):
        guard.check()
        indices = torch.randint(len(x), (min(512, len(x)),), device=device)
        loss = (model(x[indices])-y[indices]).square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (i+1) % 100 == 0:
            with torch.no_grad():
                mse = float((model(vx)-vy).square().mean())
            if mse < best:
                best = mse
                saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(saved)
    model.eval()
    return model, {"validation_mse": best, "iterations": 600}


def evaluate(torch, brain, seeds, guard, *, policy, mlp=None, steps=2000, sense_version=2, goal_scale=.3):
    world = CursorWorld(seeds, steps=steps, sense_version=sense_version, goal_scale=goal_scale)
    state, previous = brain.init_state(world.B), None
    weights = brain.weight_matrix().detach()
    errors = []
    with torch.inference_mode():
        for _ in range(steps):
            guard.check()
            if policy == "teacher":
                point = world.teacher()
            else:
                obs, packed = observe(torch, brain, world.senses(previous))
                if policy == "connectome":
                    action, state, _ = brain(obs, state, weights)
                    action = action[:, :2]
                else:
                    action = mlp(packed)
                point = np.rint(world.cursor + action.cpu().numpy()*world.speed[:, None]*.01)
                point = np.clip(point, world.rect[:, :2], world.rect[:, :2]+world.rect[:, 2:]-1)
            previous = world.cursor.copy()
            world.step(point)
            errors.append(np.linalg.norm(world.goal-world.cursor, axis=1))
    errors = np.array(errors)
    terminal = errors[-30:].mean(0)
    static = np.linalg.norm(world.velocity, axis=1) == 0
    return {"policy": policy, "episodes": world.B, "seeds": list(seeds), "steps": steps,
            "static_settled_within_6px": int((np.max(errors[-10:, static], axis=0) <= 6).sum()),
            "static_episodes": int(static.sum()), "mean_terminal_error_px": float(terminal.mean()),
            "p95_terminal_error_px": float(np.percentile(terminal, 95)),
            "moving_mean_last_100_error_px": float(errors[-100:, ~static].mean()),
            "per_episode_terminal_error_px": terminal.tolist()}


def save_checkpoint(torch, brain, cfg, output, metadata):
    from haltere.train.export import export_slim
    # Retain Haltere's compatible checkpoint shape, with Ganglion's sensory contract.
    full = output / "cursor-full.tmp.pt"
    cfg.brain.action_tau = brain.cfg.action_tau
    torch.save({"model": brain.state_dict(), "config": cfg.to_dict(), "iter": metadata.get("iteration", 1),
                "graph": str(Path(cfg.train.graph).resolve()), "channels": brain.channel_dims}, full)
    destination = output / "cursor-readout.pt"
    export_slim(full, destination)
    ck = torch.load(destination, map_location="cpu", weights_only=True)
    ck["ganglion_cursor"] = metadata
    torch.save(ck, destination)
    full.unlink()  # This function's own temporary file only.
    return destination


def run(args):
    import torch
    from haltere.train.bptt import load_checkpoint
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full-connectome training experiment")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    guard = Guard(args.seconds, temperature=args.max_gpu_temp)
    base = args.checkpoint.resolve()
    brain, cfg, _ = load_checkpoint(base, "cuda")
    if brain.__class__.__name__ != "ConnectomeRNN" or brain.motor_norm is None:
        raise ValueError("Requires a Haltere connectome with motor whitening")
    brain.eval()
    for parameter in brain.parameters():
        parameter.requires_grad_(False)
    splits = {"train": list(range(1000, 1000+args.episodes)), "validation": list(range(2000, 2016)),
              "test": list(range(3000, 3032))}
    version = args.adapter_version
    report = {"experiment": "frozen-connectome cursor readout", "base_checkpoint": str(base),
              "base_sha256": hashlib.sha256(base.read_bytes()).hexdigest(), "neurons": brain.N,
              "edges": int(brain.edge_index.shape[1]), "torch": torch.__version__,
              "adapter_version": version, "view_fraction": args.view_fraction, "slip": slip_options(args), "goal_scale": args.goal_scale, "splits": splits,
              "training_steps": args.steps,
              "plant_version": 2, "reach_radius_px": [2, 400],
              "motion_limit": "min(150, speed*gain/(delay_ticks+1)*0.2) per axis",
              "application_win_verified": False, "promoted": False}
    (args.out/"config.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    train = harvest(torch, brain, splits["train"], guard, steps=args.steps, batch=16, sense_version=version,
                    view_fraction=args.view_fraction, slip=slip_options(args), goal_scale=args.goal_scale)
    validation = harvest(torch, brain, splits["validation"], guard, steps=args.steps, batch=16, sense_version=version,
                         view_fraction=args.view_fraction, slip=slip_options(args), goal_scale=args.goal_scale)
    torch.save({"train": train, "validation": validation, "splits": splits,
                "provenance": {"adapter_version": version, "goal_scale": args.goal_scale,
                               "view_fraction": args.view_fraction, "slip": slip_options(args)}}, args.out/"features.pt")
    print(json.dumps({"stage": "fit", "training_samples": len(train[0])}), flush=True)
    report["readout_fit"] = fit(torch, brain, train, validation, guard, args.features, args.near_goal_weight,
                                args.max_lag_steps)
    metadata = {"adapter_version": version, "trained": True, "training_domain": "synthetic cursor episodes",
                "view_fraction": args.view_fraction, "slip": slip_options(args), "goal_scale": args.goal_scale,
                "near_goal_weight": args.near_goal_weight, "max_lag_steps": args.max_lag_steps,
                "method": "frozen connectome, ridge motor readout", "base_sha256": report["base_sha256"],
                "control_authority": False, "training_seeds": splits["train"],
                "validation_seeds": splits["validation"], "readout_fit": report["readout_fit"]}
    checkpoint = save_checkpoint(torch, brain, cfg, args.out, metadata)
    print(json.dumps({"stage": "checkpoint", "path": str(checkpoint), "fit": report["readout_fit"]}), flush=True)
    mlp, report["mlp_fit"] = train_mlp(torch, train, validation, guard, brain.device)
    torch.save({"model": mlp.state_dict(), "channels": brain.channel_dims, "adapter_version": version,
                "goal_scale": args.goal_scale}, args.out/"mlp-baseline.pt")
    report["held_out"] = []
    for policy in ("teacher", "mlp", "connectome"):
        score = evaluate(torch, brain, splits["test"], guard, policy=policy, mlp=mlp, sense_version=version,
                         goal_scale=args.goal_scale)
        report["held_out"].append(score)
        print(json.dumps({"stage": "evaluate", **score}), flush=True)
    report.update(elapsed_seconds=time.perf_counter()-started, peak_gpu_c=guard.peak,
                  checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    (args.out/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"finished": True, "report": str(args.out/"report.json"),
                      "elapsed_seconds": report["elapsed_seconds"], "peak_gpu_c": guard.peak}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--features", type=int, default=512)
    p.add_argument("--seconds", type=float, default=600)
    p.add_argument("--max-gpu-temp", type=float, default=65)
    p.add_argument("--adapter-version", type=int, default=2, choices=(2, 3, 4, 5, 6),
                   help="2: goal error and own velocity; 3: goal error only; 4: goal error and the visual slip of views; "
                        "5: goal direction at full strength plus the distance; 6: 5 plus the own velocity")
    p.add_argument("--view-fraction", type=float, default=0.0,
                   help="share of harvested episodes that are views (unbounded, capture latency, slip in lptc)")
    p.add_argument("--max-lag-steps", type=int, default=None,
                   help="keep only motor neurons that answer a goal reversal within this many network steps")
    p.add_argument("--near-goal-weight", type=float, default=1.0,
                   help="weight of the samples where the teacher is already decelerating in the ridge fit")
    p.add_argument("--goal-scale", type=float, default=0.3,
                   help="seconds of intent speed the goal error is normalised by (smaller: stronger input near the goal)")
    p.add_argument("--slip-dropout", type=float, default=0.0, help="share of view episodes trained without the slip")
    p.add_argument("--slip-blank", type=float, default=0.0, help="share of ticks where the slip blanks")
    p.add_argument("--slip-gain", type=float, nargs=2, default=(1.0, 1.0), metavar=("LO", "HI"),
                   help="per-episode gain range on the slip")
    args = p.parse_args()
    if not 0 <= args.view_fraction <= 1:
        p.error("Use a view fraction between 0 and 1")
    if not (0 <= args.slip_dropout <= 1 and 0 <= args.slip_blank <= 1 and 0 < args.slip_gain[0] <= args.slip_gain[1] <= 2):
        p.error("Use slip dropout and blank between 0 and 1 and a slip gain range inside (0, 2]")
    if not .02 <= args.goal_scale <= 2:
        p.error("Use a goal scale between 0.02 and 2 seconds of intent speed")
    if not 1 <= args.near_goal_weight <= 100:
        p.error("Use a near-goal weight between 1 and 100")
    if args.max_lag_steps is not None and not 1 <= args.max_lag_steps <= 30:
        p.error("Use a lag limit between 1 and 30 network steps")
    if not (16 <= args.episodes <= 256 and 80 <= args.steps <= 500 and 32 <= args.features <= 4096
            and 10 <= args.seconds <= 1800 and 50 <= args.max_gpu_temp <= 70):
        p.error("Use bounded episodes 16–256, steps 80–500, features 32–4096, seconds 10–1800, temperature 50–70")
    run(args)


if __name__ == "__main__":
    main()
