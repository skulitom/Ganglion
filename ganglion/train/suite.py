"""Fixed closed-loop suite: settling, target jumps, pursuit and camera tracking on identical episodes.

Every controller sees exactly the same seeded plants, start positions and target motion, and
differs only in who chooses each tick's point: the deterministic reference (``teacher``), an MLP
on the same sensor channels (``mlp``), the connectome on its own (``connectome``), or the
connectome under the runtime envelope with the reference acting whenever a proposal is rejected
(``supervised``). Task measures are success, settling time and tracking error; interventions count
how often the envelope had to act. The accepted share of commands is a diagnostic, never a score.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .cursor_world import VIEW_LATENCY_TICKS, CursorWorld

TICK = .01
TOLERANCE_PX = 6.0
CAMERA_LATENCY_TICKS = VIEW_LATENCY_TICKS
JUMP_EVERY = 250                  # ticks between target jumps in the jump task (2.5 s)
TASKS = ("settle", "jump", "pursuit", "camera")
CONTROLLERS = ("teacher", "mlp", "connectome", "supervised")
SEEDS = {"settle": list(range(20000, 20032)), "jump": list(range(21000, 21032)),
         "pursuit": list(range(22000, 22032)), "camera": list(range(23000, 23032))}
CHANNEL_DIMS = {"goal": 4, "haltere": 3, "jo": 3, "lptc": 6, "ocelli": 3, "wing_cs": 3, "compass": 2}
CHANNEL_ORDER = tuple(CHANNEL_DIMS)      # the MLP takes the checkpoint's own channel order


class SuiteWorld(CursorWorld):
    """CursorWorld with the suite's task variations. The episode depends on the seeds and the
    task only, so every controller meets the same plant, start, target and motion."""

    def __init__(self, seeds, task, *, steps=2000, sense_version=2, goal_scale=.3):
        if task not in TASKS:
            raise ValueError(f"Unknown suite task {task!r}")
        super().__init__(seeds, steps=steps, jump_every=JUMP_EVERY if task == "jump" else None,
                         sense_version=sense_version, view=(task == "camera"), goal_scale=goal_scale)
        self.task = task
        rngs = [np.random.default_rng(seed + 1_000_003) for seed in self.seeds]
        if task in ("settle", "jump"):
            self.velocity[:] = 0
        else:
            limits = np.minimum(150, self.speed * self.gain / (self.delay + 1) * .2)
            self.velocity = np.array([r.uniform(-limit, limit, 2) for r, limit in zip(rngs, limits)])
            self.velocity[np.linalg.norm(self.velocity, axis=1) < 1] = 20   # every pursuit target moves
        if task == "jump":
            self._cap_start(self.reach_cap)      # the first reach is sized like the jumps
        self.lost = np.zeros(self.B, dtype=bool)
        self.lost_at = np.full(self.B, steps)     # tick index at which the target left the view

    def step(self, point):
        point = np.asarray(point, dtype=float)
        if self.task == "camera":
            point = np.where(self.lost[:, None], self.cursor, point)   # a lost target gets no commands
        super().step(point)
        if self.task == "camera":
            error = self.goal - self.cursor
            gone = (np.abs(error[:, 0]) > self.rect[:, 2] / 2) | (np.abs(error[:, 1]) > self.rect[:, 3] / 2)
            self.lost_at[gone & ~self.lost] = self.tick - 1
            self.lost |= gone

    def _cap_start(self, cap):
        offset = self.cursor - self.goal
        norm = np.maximum(np.linalg.norm(offset, axis=1), 1e-9)
        offset *= np.minimum(1, cap / norm)[:, None]
        self.cursor = self.goal + offset
        self.initial = self.cursor.copy()


def pack(senses, order=CHANNEL_ORDER):
    return np.concatenate([senses[key] for key in order], axis=1).astype(np.float32)


def proposed_points(cursor, velocity, dt, speed, rect):
    """Where a velocity proposal (in units of the intent speed) puts the cursor this tick, under the
    same speed limit and area clamp as the runtime envelope, but without its acceptance test."""
    velocity = np.asarray(velocity, dtype=float)
    finite = np.isfinite(velocity).all(axis=1)
    velocity = np.where(finite[:, None], velocity, 0)
    dt = min(.02, max(0, dt))
    limit = np.asarray(speed, dtype=float) * dt
    step = velocity * limit[:, None]
    length = np.hypot(step[:, 0], step[:, 1])
    over = length > limit
    step[over] *= (limit[over] / length[over])[:, None]
    low, high = rect[:, :2], rect[:, :2] + rect[:, 2:] - 1
    clamped = np.clip(cursor, low, high)
    return np.clip(np.rint(clamped + step), low, high), clamped, finite


def supervise_batch(cursor, goal, velocity, dt, speed, rect, tolerance=TOLERANCE_PX):
    """Vectorised twin of ganglion.core.reach.supervise: candidate points and a rejected mask."""
    candidate, clamped, finite = proposed_points(cursor, velocity, dt, speed, rect)
    before = np.hypot(goal[:, 0] - clamped[:, 0], goal[:, 1] - clamped[:, 1])
    after = np.hypot(goal[:, 0] - candidate[:, 0], goal[:, 1] - candidate[:, 1])
    rejected = ~finite | ((before > tolerance) & (after >= before)) | (after > before + .5)
    return candidate, rejected


def settle_tick(error, tolerance=TOLERANCE_PX):
    """First tick from which the error stays within tolerance to the end, or None."""
    bad = np.flatnonzero(np.asarray(error) > tolerance)
    if len(bad) == 0:
        return 0
    return None if bad[-1] + 1 >= len(error) else int(bad[-1] + 1)


def run_task(task, seeds, controller, *, propose=None, steps=2000, guard=None, sense_version=2, goal_scale=.3):
    """Run one controller over the task's episodes. ``propose(senses) -> (B, 2) velocities`` is
    required for every controller but the teacher and is called once per tick with the same
    sensor channels the runtime adapter builds. A proposer with a ``bind(world)`` method is handed
    the episode state first: for oracles and diagnostics, never for a policy under test."""
    if controller != "teacher" and propose is None:
        raise ValueError(f"{controller} needs a proposer")
    world = SuiteWorld(seeds, task, steps=steps, sense_version=sense_version, goal_scale=goal_scale)
    if hasattr(propose, "bind"):
        propose.bind(world)
    rect = world.bounds
    errors = np.zeros((steps, world.B))
    rejected = np.zeros((steps, world.B), dtype=bool)
    previous = None
    for tick in range(steps):
        if guard is not None:
            guard.check()
        reference = world.teacher()
        if controller == "teacher":
            point = reference
        else:
            velocity = propose(world.senses(previous))
            if controller == "supervised":
                candidate, reject = supervise_batch(world.cursor, world.goal, velocity, TICK, world.speed, rect)
                point = np.where(reject[:, None], reference, candidate)
                rejected[tick] = reject & ~world.lost
            else:
                point = proposed_points(world.cursor, velocity, TICK, world.speed, rect)[0]
        previous = world.cursor.copy()
        world.step(point)
        errors[tick] = np.hypot(*(world.goal - world.cursor).T)
    return summarize(task, controller, world, errors, rejected)


def summarize(task, controller, world, errors, rejected):
    steps, B = errors.shape
    late = slice(100, None)                                  # after the first second
    lost_at = world.lost_at if task == "camera" else np.full(B, steps)
    valid = np.arange(steps)[:, None] < lost_at[None, :]     # ticks before the target left the view
    tracking = np.array([errors[late, i][valid[late, i]].mean() if valid[late, i].any() else float("inf")
                         for i in range(B)])
    worst_late = np.array([errors[late, i][valid[late, i]].max() if valid[late, i].any() else float("inf")
                           for i in range(B)])
    interventions = rejected.sum(0)
    considered = valid.sum(0)
    result = {"task": task, "controller": controller, "episodes": B, "steps": steps,
              "tolerance_px": TOLERANCE_PX, "seeds": list(world.seeds)}
    if task == "settle":
        settled = [settle_tick(errors[:, i]) for i in range(B)]
        success = [t is not None and steps - t >= 50 for t in settled]
        settling = [t * TICK * 1000 for t in settled if t is not None]
        result.update(success=int(sum(success)),
                      settling_ms=_spread(settling), settled_episodes=len(settling),
                      per_episode={"settling_ms": [None if t is None else t * TICK * 1000 for t in settled]})
    elif task == "jump":
        times, success, total = [], 0, 0
        for i in range(B):
            for start in range(0, steps, JUMP_EVERY):
                total += 1
                segment = errors[start:start + JUMP_EVERY, i]
                t = settle_tick(segment)
                if t is not None and len(segment) - t >= 20:
                    success += 1
                    times.append(t * TICK * 1000)
        result.update(reaches=total, success=success, settling_ms=_spread(times), settled_reaches=len(times),
                      per_episode={"tracking_error_px": tracking.tolist()})
    else:
        within = np.array([(errors[late, i][valid[late, i]] <= 2 * TOLERANCE_PX).mean() if valid[late, i].any() else 0.0
                           for i in range(B)])
        acquired = [settle_tick(np.minimum.accumulate(errors[:, i]), 2 * TOLERANCE_PX) for i in range(B)]
        success = (tracking <= 2 * TOLERANCE_PX) & (lost_at == steps)
        result.update(success=int(success.sum()), within_12px_fraction=float(within.mean()),
                      acquisition_ms=_spread([t * TICK * 1000 for t in acquired if t is not None]),
                      per_episode={"tracking_error_px": tracking.tolist(), "within_12px_fraction": within.tolist()})
        if task == "camera":
            result.update(lost=int((lost_at < steps).sum()),
                          lost_after_ms=_spread([t * TICK * 1000 for t in lost_at if t < steps]))
    result["tracking_error_px"] = {"mean": _finite(tracking, np.mean), "p95": _finite(tracking, lambda v: np.percentile(v, 95)),
                                   "worst_late": _finite(worst_late, np.max)}
    if controller == "supervised":
        share = 1 - interventions.sum() / max(1, considered.sum())
        result["interventions"] = {"ticks": int(interventions.sum()), "fraction": float(interventions.sum() / max(1, considered.sum())),
                                   "per_episode": interventions.tolist()}
        result["accepted_share"] = float(share)
        result["note"] = "interventions: ticks on which the envelope rejected the proposal and the reference acted"
    return result


def _spread(values):
    values = [v for v in values if v is not None]
    if not values:
        return {"median": None, "p90": None, "max": None}
    return {"median": float(np.median(values)), "p90": float(np.percentile(values, 90)), "max": float(max(values))}


def _finite(values, reduce):
    """Reduce the finite entries of an array, or None when an episode never produced one."""
    finite = values[np.isfinite(values)]
    return float(reduce(finite)) if len(finite) else None


def table(report):
    lines = ["| Task | Controller | Success | Settling / acquisition (median ms) | Tracking error (mean px) | Interventions | Accepted |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for task in report["tasks"]:
        for controller, r in report["tasks"][task]["controllers"].items():
            total = r.get("reaches", r["episodes"])
            timing = r.get("settling_ms") or r.get("acquisition_ms") or {}
            median = timing.get("median")
            err = r["tracking_error_px"]["mean"]
            success = f"{r['success']}/{total}" + (f" (lost {r['lost']})" if "lost" in r else "")
            interventions = f"{r['interventions']['fraction']:.1%}" if "interventions" in r else "n/a"
            accepted = f"{r['accepted_share']:.1%}" if "accepted_share" in r else "n/a"
            lines.append(f"| {task} | {controller} | {success} | {'n/a' if median is None else f'{median:.0f}'} | "
                         f"{'n/a' if err is None else f'{err:.1f}'} | {interventions} | {accepted} |")
    return "\n".join(lines)


def connectome_proposer(torch, brain, batch):
    from .cursor_readout import observe
    state = {"value": brain.init_state(batch)}
    weights = brain.weight_matrix().detach()

    def propose(senses):
        obs, _ = observe(torch, brain, senses)
        with torch.inference_mode():
            action, state["value"], _ = brain(obs, state["value"], weights)
        return action[:, :2].float().cpu().numpy()
    return propose


def mlp_proposer(torch, mlp, device, order=CHANNEL_ORDER):
    def propose(senses):
        with torch.inference_mode():
            return mlp(torch.from_numpy(pack(senses, order)).to(device)).float().cpu().numpy()
    return propose


def run(args):
    import torch
    from haltere.train.bptt import load_checkpoint
    from .cursor_readout import Guard, build_mlp, train_mlp
    if not torch.cuda.is_available():
        raise RuntimeError("The suite runs the actual connectome on CUDA")
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    guard = Guard(args.seconds, args.max_gpu_temp, pace=.001)
    checkpoint = args.checkpoint.resolve()
    brain, _, _ = load_checkpoint(checkpoint, "cuda")
    if brain.__class__.__name__ != "ConnectomeRNN" or dict(brain.channel_dims) != CHANNEL_DIMS:
        raise ValueError("Requires a Haltere cursor ConnectomeRNN with the cursor sensory contract")
    order = tuple(brain.channel_dims)        # the order the MLP features were packed in
    brain.eval()
    for parameter in brain.parameters():
        parameter.requires_grad_(False)
    provenance = torch.load(checkpoint, map_location="cpu", weights_only=True).get("ganglion_cursor", {})
    version = args.sense_version or provenance.get("adapter_version", 2)
    if version not in (2, 3, 4):
        raise ValueError("The suite speaks sensory adapter versions 2, 3 and 4")
    goal_scale = float(provenance.get("goal_scale", .3))
    mlp = build_mlp(torch).to(brain.device)
    if args.mlp:
        saved = torch.load(args.mlp, map_location="cpu", weights_only=True)
        if saved.get("adapter_version", 2) != version and not args.sense_version:
            raise ValueError("The MLP baseline was trained on another sensory adapter version than the checkpoint")
        if abs(float(saved.get("goal_scale", .3)) - goal_scale) > 1e-9:
            raise ValueError("The MLP baseline was trained with another goal scale than the checkpoint")
        mlp.load_state_dict(saved["model"])
        mlp_source = {"path": str(args.mlp), "sha256": hashlib.sha256(Path(args.mlp).read_bytes()).hexdigest(),
                      "adapter_version": saved.get("adapter_version", 2)}
    else:
        cache = torch.load(args.mlp_features, map_location="cpu", weights_only=True)
        from .cursor_dagger import check_provenance
        check_provenance(cache.get("provenance"), version=version, goal_scale=goal_scale)
        mlp, fit = train_mlp(torch, cache["train"], cache["validation"], guard, brain.device)
        path = args.out / "mlp-baseline.pt"
        torch.save({"model": mlp.state_dict(), "channels": brain.channel_dims, "adapter_version": version,
                    "goal_scale": goal_scale, "features": str(args.mlp_features)}, path)
        mlp_source = {"path": str(path), "trained_from": str(args.mlp_features), "fit": fit,
                      "training_samples": int(len(cache["train"][0]))}
    mlp.eval()
    report = {"suite": "cursor-suite-v1", "measured_at": datetime.now(timezone.utc).isoformat(),
              "checkpoint": {"path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                             "training": provenance}, "mlp": mlp_source,
              "adapter_version": version, "checkpoint_adapter_version": provenance.get("adapter_version", 2),
              "goal_scale": goal_scale,
              "steps": args.steps, "tick_seconds": TICK, "tolerance_px": TOLERANCE_PX,
              "camera_latency_ticks": CAMERA_LATENCY_TICKS, "jump_every_ticks": JUMP_EVERY,
              "torch": torch.__version__, "gpu": torch.cuda.get_device_name(brain.device),
              "controllers": list(CONTROLLERS), "tasks": {}, "promoted": False, "actuation_authority": False}
    for task in TASKS:
        seeds = SEEDS[task]
        report["tasks"][task] = {"seeds": seeds, "controllers": {}}
        for controller in CONTROLLERS:
            propose = None
            if controller == "mlp":
                propose = mlp_proposer(torch, mlp, brain.device, order)
            elif controller in ("connectome", "supervised"):
                propose = connectome_proposer(torch, brain, len(seeds))
            result = run_task(task, seeds, controller, propose=propose, steps=args.steps, guard=guard,
                              sense_version=version, goal_scale=goal_scale)
            report["tasks"][task]["controllers"][controller] = result
            print(json.dumps({k: v for k, v in result.items() if k not in ("per_episode", "seeds")}), flush=True)
            (args.out / "progress.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    report.update(elapsed_seconds=time.perf_counter() - started, peak_gpu_c=guard.peak)
    (args.out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out / "table.md").write_text(table(report) + "\n", encoding="utf-8")
    print(table(report), flush=True)
    print(json.dumps({"finished": True, "report": str(args.out / "report.json"),
                      "elapsed_seconds": report["elapsed_seconds"], "peak_gpu_c": guard.peak}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--mlp", type=Path, help="MLP baseline state to load (else trained from --mlp-features)")
    p.add_argument("--mlp-features", type=Path, help="feature cache to train the MLP baseline from")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--sense-version", type=int, choices=(2, 3, 4),
                   help="feed the world's channels of this adapter version instead of the checkpoint's own")
    p.add_argument("--seconds", type=float, default=900)
    p.add_argument("--max-gpu-temp", type=float, default=65)
    args = p.parse_args()
    if not (args.mlp or args.mlp_features):
        p.error("Give --mlp or --mlp-features")
    if not (500 <= args.steps <= 4000 and 30 <= args.seconds <= 1800 and 50 <= args.max_gpu_temp <= 70):
        p.error("Use steps 500–4000, seconds 30–1800 and temperature 50–70")
    run(args)


if __name__ == "__main__":
    main()
