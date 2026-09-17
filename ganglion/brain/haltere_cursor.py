"""Experimental screen-coordinate adapter for Haltere's actual ConnectomeRNN.

The flight checkpoint has NOT been trained for desktop control. Its first two
outputs are interpreted as candidate cursor velocities only for shadow comparison.
No direct sensory bypass or deterministic correction is added to these proposals.
"""
import hashlib
from math import hypot, tanh
from pathlib import Path


def channels(sample, previous=None):
    """Versioned, application-independent analogues of Haltere's sensory channels.

Goal is screen error / 400 px; cursor velocity is normalised by the intent speed.
Flow, attitude, compass, and load channels remain zero: no claim of fly vision.
"""
    dx, dy = (sample.goal[i] - sample.cursor[i] for i in range(2))
    vx = vy = 0.0
    if previous is not None and .001 <= sample.submitted - previous.submitted <= .05:
        dt = sample.submitted - previous.submitted
        vx, vy = ((sample.cursor[i] - previous.cursor[i]) / dt / sample.speed for i in range(2))
    velocity = [tanh(vx), tanh(vy), 0.0]
    return {"goal": [tanh(dx/400), tanh(dy/400), 0.0, tanh(hypot(dx, dy)/400)],
            "haltere": velocity, "jo": velocity, "lptc": [0.0]*6,
            "ocelli": [0.0]*3, "wing_cs": [0.0]*3, "compass": [0.0]*2}


class HaltereCursor:
    def __init__(self, checkpoint):
        import torch
        from haltere.train.bptt import load_checkpoint
        if not torch.cuda.is_available():
            raise RuntimeError("The full connectome shadow requires CUDA; CPU cannot meet its tick budget")
        self.torch = torch
        checkpoint = Path(checkpoint).resolve()
        self.brain, _, _ = load_checkpoint(checkpoint, "cuda")
        if self.brain.__class__.__name__ != "ConnectomeRNN":
            raise ValueError("Expected the actual Haltere ConnectomeRNN, not a baseline checkpoint")
        if abs(self.brain.cfg.dt - .01) > 1e-8:
            raise ValueError("Expected a 10 ms neural step")
        expected = {"goal": 4, "haltere": 3, "jo": 3, "lptc": 6, "ocelli": 3, "wing_cs": 3, "compass": 2}
        if self.brain.channel_dims != expected or self.brain.cfg.n_actions != 4:
            raise ValueError("Checkpoint sensory/action contract does not match adapter v1")
        self.brain.eval()
        self.weights = self.brain.weight_matrix().detach()
        # One host-to-device copy instead of seven tiny transfers per observation.
        self.order = tuple(expected)
        self.input = torch.empty((1, sum(expected.values())), device=self.brain.device)
        self.host_input = torch.empty_like(self.input, device="cpu", pin_memory=True)
        self.observations = {}
        offset = 0
        for key in self.order:
            self.observations[key] = self.input[:, offset:offset+expected[key]]
            offset += expected[key]
        self.metadata = {"type": "Haltere ConnectomeRNN", "checkpoint": str(checkpoint),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(), "adapter_version": 1,
            "neurons": self.brain.N, "edges": int(self.brain.edge_index.shape[1]),
            "device": str(self.brain.device), "neural_dt_seconds": self.brain.cfg.dt,
            "desktop_trained": False, "visual_frontend": "deterministic taught colour components",
            "readout_interpretation": "flight outputs 0 and 1 as hypothetical cursor velocity"}
        self.reset()
        with torch.inference_mode():
            zeros = {k: torch.zeros(1, d, device=self.brain.device) for k, d in expected.items()}
            for _ in range(30):
                _, self.state, _ = self.brain(zeros, self.state, self.weights)
            torch.cuda.synchronize()
        self.reset()

    def reset(self):
        self.state = self.brain.init_state(1)
        self.previous = None

    def predict(self, sample):
        torch = self.torch
        with torch.inference_mode():
            values = channels(sample, self.previous)
            packed = [v for key in self.order for v in values[key]]
            self.host_input.copy_(torch.tensor([packed]))
            self.input.copy_(self.host_input, non_blocking=True)
            action, self.state, _ = self.brain(self.observations, self.state, self.weights)
            values = action[0].cpu().tolist()  # Includes CUDA completion in measured latency.
        self.previous = sample
        x, y, w, h = sample.rect
        dt = min(.02, max(0, sample.dt))
        point = [min(x+w-1, max(x, round(sample.cursor[0] + values[0]*sample.speed*dt))),
                 min(y+h-1, max(y, round(sample.cursor[1] + values[1]*sample.speed*dt)))]
        return {"point": point, "raw_actions": values, "desktop_trained": False}
