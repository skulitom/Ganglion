"""The shadow worker in its own process: the same one-item mailbox, without the runtime's interpreter.

Measured live, the model steps in 2 ms alone but its p95 reaches 25-30 ms inside the runtime
process, where the worker thread shares the interpreter with capture, detection, the tick and
the service. This variant keeps the ShadowWorker contract (submit, proposal, invalidate, status,
close; predictions and discards as ledger records; no actuation authority) and runs inference in
a child process that loads the checkpoint itself. Samples and results cross a pipe; the child
keeps only the newest sample, resets the network on generation, binding or gap changes as the
thread version does, and timestamps with the same monotonic clock, which is system-wide on
Windows, so freshness is judged the same way on both sides.
"""
from __future__ import annotations

from dataclasses import asdict
from math import hypot, isfinite
import multiprocessing as mp
import threading
import time

MAX_AGE = .05


def fine_timer():
    """Ask Windows for 1 ms timer resolution in this process; sleeps and waits otherwise round to 15.6 ms."""
    try:
        import ctypes
        ctypes.windll.winmm.timeBeginPeriod(1)
    except (AttributeError, OSError):
        pass


def serve(factory, connection, clock_name):
    """Child process: load the predictor, then answer the newest sample until told to stop."""
    fine_timer()
    clock = getattr(time, clock_name)
    try:
        predictor = factory()
        connection.send(("ready", predictor.metadata))
    except Exception as exc:
        connection.send(("failed", f"{type(exc).__name__}: {exc}"))
        return
    previous = None
    replaced = discarded = 0
    while True:
        message = connection.recv()
        while connection.poll():                       # keep only the newest sample
            newer = connection.recv()
            if newer is None or message is None:
                message = None
                break
            replaced += 1
            message = newer
        if message is None:
            return
        generation, sample = message
        now = clock()
        if now > sample.expires or now - sample.sampled > MAX_AGE:
            discarded += 1
            connection.send(("counts", replaced, discarded))
            continue
        reset = (previous is None or previous[0] != generation or previous[1].binding != sample.binding
                 or sample.submitted - previous[1].submitted > MAX_AGE)
        try:
            started = clock()
            if reset:
                predictor.reset()
            prediction = predictor.predict(sample)
            elapsed_ms = (clock() - started) * 1000
            point = tuple(prediction["point"])
            if len(point) != 2 or not all(isfinite(x) for x in point):
                raise ValueError("Non-finite or malformed neural point")
            finished = clock()
            steps = int(prediction.get("neural_steps", 1))
            record = {k: v for k, v in prediction.items() if k != "neural_steps"}
            record.update(intent_id=sample.intent_id, stage=sample.stage, observation_id=sample.observation_id,
                          sample_started_mono=sample.sampled, submitted_mono=sample.submitted,
                          model_started_mono=started, inference_ms=elapsed_ms, within_5ms=elapsed_ms <= 5,
                          neural_steps=steps, reset=reset, sensor_input=asdict(sample),
                          actuation_authority=False, reference_point=list(sample.reference),
                          disagreement_px=hypot(point[0] - sample.reference[0], point[1] - sample.reference[1]))
            raw = list(prediction.get("raw_actions", []))[:2]
            connection.send(("prediction", generation, sample.binding, sample.sampled, finished, sample.expires,
                             raw if len(raw) == 2 else None, record, replaced, discarded))
            previous = generation, sample
        except Exception as exc:
            connection.send(("failed", f"{type(exc).__name__}: {exc}"))
            return


class ShadowProcess:
    """Drop-in for ShadowWorker whose model lives in a child process."""

    def __init__(self, factory, ledger, *, clock=time.perf_counter, start_timeout=120):
        self.ledger, self.clock = ledger, clock
        self.condition = threading.Condition()
        self.latest = None
        self.generation = 0
        self.closed = False
        self.failed = None
        self.submitted = self.replaced = self.predictions = self.discarded = 0
        self.metadata = {"type": "starting"}
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        clock_name = getattr(clock, "__name__", "perf_counter")
        self.process = context.Process(target=serve, args=(factory, child, clock_name), name="ganglion-shadow", daemon=True)
        self.process.start()
        child.close()
        if not self.connection.poll(start_timeout):
            self.close()
            raise RuntimeError("The shadow process did not report ready in time")
        kind, payload = self.connection.recv()
        if kind != "ready":
            self.close()
            raise RuntimeError(f"The shadow process failed to start: {payload}")
        self.metadata = payload
        self.thread = threading.Thread(target=self._collect, name="ganglion-shadow-reader", daemon=True)
        self.thread.start()

    # ------------------------------------------------------------ the ShadowWorker contract
    @property
    def predictor(self):
        return self

    def submit(self, sample):
        with self.condition:
            if self.closed or self.failed:
                return
            self.submitted += 1
            generation = self.generation
        try:
            self.connection.send((generation, sample))
        except (OSError, ValueError):
            pass

    def invalidate(self):
        with self.condition:
            self.generation += 1
            self.latest = None

    def proposal(self, binding, now, max_age=MAX_AGE):
        with self.condition:
            latest = self.latest
        if latest is None or latest[0] != binding or now - latest[1] > max_age:
            return None
        return latest[2]

    def status(self):
        with self.condition:
            return {"mode": "shadow", "actuation_authority": False, "process": True,
                    "state": "closed" if self.closed else "failed" if self.failed else "ready",
                    "error": self.failed, "submitted": self.submitted, "replaced": self.replaced,
                    "predictions": self.predictions, "discarded": self.discarded, "model": self.metadata}

    def close(self):
        with self.condition:
            self.closed = True
            self.generation += 1
        try:
            self.connection.send(None)
        except (OSError, ValueError):
            pass
        self.process.join(1)
        if self.process.is_alive():
            self.process.terminate()
        try:
            self.connection.close()
        except OSError:
            pass

    # ------------------------------------------------------------ results
    def _collect(self):
        while True:
            try:
                message = self.connection.recv()
            except (EOFError, OSError):
                return
            kind = message[0]
            if kind == "counts":
                with self.condition:
                    self.replaced, self.discarded = message[1], message[2]
                continue
            if kind == "failed":
                with self.condition:
                    self.failed = message[1]
                    self.latest = None
                self.ledger.append("shadow_failed", self.clock(), critical=True, error=self.failed,
                                   actuation_authority=False)
                return
            _, generation, binding, sampled, finished, expires, raw, record, replaced, discarded = message
            with self.condition:
                valid = (not self.closed and generation == self.generation
                         and finished <= expires and finished - sampled <= MAX_AGE)
                self.replaced, self.discarded = replaced, discarded
                if valid:
                    self.predictions += 1
                    self.latest = (binding, sampled, raw, finished) if raw is not None else None
                else:
                    self.discarded += 1
            self.ledger.append("shadow_prediction" if valid else "shadow_discarded", finished, **record)
