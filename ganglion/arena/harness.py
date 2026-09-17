"""Shared lifecycle for isolated, evaluator-controlled MCP experiments."""
from contextlib import contextmanager, ExitStack
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

from ganglion.core.output import MemoryOutput, ProcessOutput, SimulatedOutput
from ganglion.core.protocol import Service, Server
from ganglion.core.runner import Runner
from ganglion.core.runtime import Runtime
from .target import SyntheticArena


@contextmanager
def experiment(environment, world_factory, *, scenario="reach", shadow_predictor=None):
    if environment not in ("synthetic", "arena", "browser"):
        raise ValueError("Choose synthetic, arena, or browser")
    with tempfile.TemporaryDirectory(prefix="ganglion-experiment-") as directory, ExitStack() as stack:
        temp, window = Path(directory), None
        if environment == "synthetic":
            capture = SyntheticArena(world=world_factory())
            if scenario == "manipulation":
                from .manipulation import SimulatedPointer
                output = SimulatedOutput(time.perf_counter, SimulatedPointer(capture))
            else:
                output = MemoryOutput(time.perf_counter, capture.click)
            target, session_id = capture.target, 0

            def reset(trial):
                with capture.lock:
                    capture.world.reset(trial)
        else:
            from ganglion.core.actuators import Actuators
            from ganglion.core.capture import Capture
            from ganglion.core.session import current
            from ganglion.core.window import api, describe
            session_id = current().session_id
            restore = Actuators()
            cursor, foreground = restore.cursor_pos(), api().GetForegroundWindow()

            def restore_desktop():
                restore.move_abs(*cursor)
                if foreground:
                    api().SetForegroundWindow(foreground)
            stack.callback(restore_desktop)
            ready, truth_file, stop, trial_file = (temp / s for s in ("ready.json", "truth.json", "stop", "trial.json"))
            log = stack.enter_context((temp / "fixture.log").open("w"))
            module = "ganglion.arena.browser" if environment == "browser" else "ganglion.arena.target"
            process = subprocess.Popen([sys.executable, "-m", module, "--ready", str(ready),
                "--truth", str(truth_file), "--stop-file", str(stop), "--trial-file", str(trial_file),
                "--scenario", scenario, "--seconds", "150"], stdout=log, stderr=log,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

            def stop_fixture():
                stop.touch()
                process.wait(timeout=10)
            stack.callback(stop_fixture)
            until = time.perf_counter() + 20
            while not ready.exists() and process.poll() is None and time.perf_counter() < until:
                time.sleep(.02)
            if not ready.exists():
                raise RuntimeError("Fixture failed to start: " + (temp / "fixture.log").read_text())
            window = json.loads(ready.read_text())
            target = lambda: describe(window["hwnd"])
            capture, output = Capture(refresh_static=True), ProcessOutput()

            def reset(trial):
                staging = trial_file.with_suffix(".tmp")
                staging.write_text(json.dumps(trial))
                staging.replace(trial_file)
        stack.callback(output.close)
        capture.start()
        stack.callback(capture.stop)
        runtime = Runtime(time.perf_counter, output, session_id=session_id, shadow_predictor=shadow_predictor)
        runner = Runner(runtime, capture, target).start()
        stack.callback(runner.close)
        service = Service(runtime)
        server = Server(service)
        stack.callback(server.server_close)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        stack.callback(server.shutdown)
        stack.callback(service.closed.set)
        endpoint = temp / "endpoint.json"
        endpoint.write_text(json.dumps(server.endpoint()))

        def finish():
            runner.close()
            if environment == "synthetic":
                with capture.lock:
                    return list(capture.world.events)
            stop_fixture()
            return json.loads(truth_file.read_text())

        yield SimpleNamespace(endpoint=endpoint, reset=reset, finish=finish, session_id=session_id,
                              shadow_status=lambda: runtime.shadow.status() if runtime.shadow else None,
                              browser_version=window.get("browser_version") if window else None)
