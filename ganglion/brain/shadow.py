"""Latest-only model observations with no reference to any actuator or runtime.

The control loop only copies an immutable sample into a one-item mailbox. Model
work, resets, exceptions, and timing run on this worker's thread, outside the
runtime lock. Predictions are ledger records, never commands.
"""
from dataclasses import asdict, dataclass
from math import hypot, isfinite
import threading
import time


@dataclass(frozen=True)
class MotorSample:
    intent_id: str
    stage: str
    layout_revision: int
    observation_id: int
    sampled: float
    submitted: float
    expires: float
    cursor: tuple
    goal: tuple
    reference: tuple
    rect: tuple
    speed: float
    dt: float
    flow: tuple | None = None    # (tx px/s, ty px/s, divergence 1/s, curl 1/s) wide-field flow, if fed

    @property
    def binding(self):
        return self.intent_id, self.stage, self.layout_revision


class ShadowWorker:
    def __init__(self, predictor, ledger, *, clock=time.perf_counter):
        self.predictor, self.ledger, self.clock = predictor, ledger, clock
        self.condition = threading.Condition()
        self.pending = None
        self.latest = None      # (binding, sampled_mono, raw_actions, finished_mono) of the newest valid prediction
        self.generation = 0
        self.closed = False
        self.failed = None
        self.submitted = self.replaced = self.predictions = self.discarded = 0
        self.thread = threading.Thread(target=self._run, name="ganglion-shadow", daemon=True)
        self.thread.start()

    def submit(self, sample):
        with self.condition:
            if self.closed or self.failed:
                return
            self.submitted += 1
            self.replaced += self.pending is not None
            self.pending = self.generation, sample
            self.condition.notify()

    def invalidate(self):
        with self.condition:
            self.generation += 1
            self.pending = None
            self.latest = None

    def proposal(self, binding, now, max_age=.05):
        """Velocity (in units of intent speed) proposed by the newest valid prediction for this
        intent, stage and layout, or None when there is none fresh enough.

        Freshness is the age of the observation the proposal was computed from, not the time
        since inference finished: the 50 ms evidence budget applies to what the model saw.
        """
        with self.condition:
            latest = self.latest
        if latest is None or latest[0] != binding or now - latest[1] > max_age:
            return None
        return latest[2]

    def status(self):
        with self.condition:
            return {"mode": "shadow", "actuation_authority": False,
                    "state": "closed" if self.closed else "failed" if self.failed else "ready",
                    "error": self.failed, "submitted": self.submitted, "replaced": self.replaced,
                    "predictions": self.predictions, "discarded": self.discarded,
                    "model": self.predictor.metadata}

    def close(self):
        with self.condition:
            self.closed = True
            self.pending = None
            self.generation += 1
            self.condition.notify()
        self.thread.join(.2)  # A stalled model cannot stall actuator cleanup.

    def _run(self):
        previous = None
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.pending is not None or self.closed)
                if self.closed:
                    return
                generation, sample = self.pending
                self.pending = None
            now = self.clock()
            if now > sample.expires or now - sample.sampled > .05:
                self.discarded += 1
                continue
            reset = (previous is None or previous[0] != generation or previous[1].binding != sample.binding
                     or sample.submitted - previous[1].submitted > .05)
            try:
                started = self.clock()
                if reset:
                    self.predictor.reset()
                prediction = self.predictor.predict(sample)
                elapsed_ms = (self.clock() - started) * 1000
                point = tuple(prediction["point"])
                if len(point) != 2 or not all(isfinite(x) for x in point):
                    raise ValueError("Non-finite or malformed neural point")
                finished = self.clock()
                with self.condition:
                    valid = (not self.closed and generation == self.generation
                             and finished <= sample.expires and finished - sample.sampled <= .05)
                    if valid:
                        self.predictions += 1
                        raw = list(prediction.get("raw_actions", []))[:2]
                        self.latest = (sample.binding, sample.sampled, raw, finished) if len(raw) == 2 else None
                    else:
                        self.discarded += 1
                steps = int(prediction.get("neural_steps", 1))   # how far neural time advanced
                record = {k: v for k, v in prediction.items() if k != "neural_steps"}
                self.ledger.append("shadow_prediction" if valid else "shadow_discarded", finished,
                    intent_id=sample.intent_id, stage=sample.stage, observation_id=sample.observation_id,
                    sample_started_mono=sample.sampled, submitted_mono=sample.submitted,
                    model_started_mono=started, inference_ms=elapsed_ms,
                    within_5ms=elapsed_ms <= 5, neural_steps=steps, reset=reset,
                    sensor_input=asdict(sample),
                    actuation_authority=False, reference_point=list(sample.reference),
                    disagreement_px=hypot(point[0]-sample.reference[0], point[1]-sample.reference[1]),
                    **record)
                previous = generation, sample
            except Exception as exc:
                with self.condition:
                    self.failed = f"{type(exc).__name__}: {exc}"
                    self.pending = None
                    self.latest = None
                self.ledger.append("shadow_failed", self.clock(), critical=True,
                                   error=self.failed, actuation_authority=False)
                return
