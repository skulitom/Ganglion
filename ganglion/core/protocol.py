"""Authenticated, bounded NDJSON transport shared by MCP and the Arena evaluator."""
from __future__ import annotations

import base64
import json
import secrets
import socket
import socketserver
import threading
import time
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from .runtime import RuntimeErrorWithCode
from .schema import OPERATIONS, READ_ONLY, Request

MAX_REQUEST = 65536
MAX_RESPONSE = 2 * 1024 * 1024


class Service:
    def __init__(self, runtime, *, controller_token=None, observer_token=None):
        self.runtime = runtime
        self.controller_token = controller_token or secrets.token_urlsafe(32)
        self.observer_token = observer_token or secrets.token_urlsafe(32)
        self.cache = {}
        self.mutations = threading.Lock()
        self.closed = threading.Event()

    def handle(self, raw):
        try:
            request = Request.model_validate(raw)
            controller = secrets.compare_digest(request.token, self.controller_token)
            observer = secrets.compare_digest(request.token, self.observer_token)
            if not (controller or observer):
                raise RuntimeErrorWithCode("unauthorized", "Invalid core capability.")
            if request.op not in OPERATIONS:
                raise RuntimeErrorWithCode("unknown_operation", f"Supported operations: {', '.join(OPERATIONS)}")
            if request.op not in READ_ONLY and not controller:
                raise RuntimeErrorWithCode("observer_only", "Observer capabilities cannot control the runtime.")
            params = OPERATIONS[request.op].model_validate(request.args)
            if request.op in READ_ONLY:
                result = self._execute(request, params)
            else:
                key = request.client_id, request.request_id
                fingerprint = json.dumps([request.op, params.model_dump()], sort_keys=True, allow_nan=False)
                with self.mutations:
                    if key in self.cache:
                        old, result = self.cache[key]
                        if old != fingerprint:
                            raise RuntimeErrorWithCode("request_id_conflict", "Use a new request ID for changed arguments.")
                    else:
                        if len(self.cache) >= 4096 and request.op != "halt":
                            raise RuntimeErrorWithCode("request_budget", "Core request journal is full; halt and restart.")
                        result = self._execute(request, params)
                        if len(self.cache) < 4096:
                            self.cache[key] = fingerprint, result
            return {"ok": True, "version": 1, "request_id": request.request_id,
                    "runtime_id": self.runtime.ledger.epoch, "result": result}
        except ValidationError as exc:
            return {"ok": False, "error": {"code": "invalid_arguments",
                    "message": exc.json(include_input=False, include_url=False)}}
        except (ValueError, RuntimeErrorWithCode) as exc:
            return {"ok": False, "error": {"code": getattr(exc, "code", "invalid_arguments"), "message": str(exc)}}
        except Exception as exc:
            return {"ok": False, "error": {"code": "internal_error", "message": f"{type(exc).__name__}: {exc}"}}

    def _execute(self, request, params):
        r, op, client = self.runtime, request.op, request.client_id
        if op == "status":
            return r.status()
        if op == "claim":
            return r.claim(client, params.seconds)
        if op == "renew":
            return r.renew(client, params.seconds)
        if op == "watch":
            return r.add_watch(client, params)
        if op == "arm":
            return r.arm(client, params)
        if op == "intent":
            return r.start_intent(client, params)
        if op == "cancel":
            return r.cancel(client, params.id)
        if op == "input":
            return r.input(client, params)
        if op in ("unwatch", "disarm"):
            return r.remove(client, params.id, watch=op == "unwatch")
        if op == "halt":
            # Any authenticated controller can stop; observer credentials cannot.
            return r.halt()
        if op == "wait":
            return self._wait(client, params)
        if op == "look":
            with r.lock:
                result = r.status() | r.ledger.read(params.cursor, params.limit)
                frame = r.frame
            if params.image and frame is not None:
                import cv2
                h, w = frame.shape[:2]
                width = min(params.max_width, w)
                height = max(1, round(h * width / w))
                if width * height > 2_000_000:
                    raise RuntimeErrorWithCode("image_too_large", "Request a smaller max_width (image exceeds 2 million pixels).")
                small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                extension = ".png" if params.image_format == "png" else ".jpg"
                options = [cv2.IMWRITE_PNG_COMPRESSION, 3] if extension == ".png" else [cv2.IMWRITE_JPEG_QUALITY, 80]
                ok, data = cv2.imencode(extension, small, options)
                if not ok:
                    raise ValueError("image encoding failed")
                if data.nbytes > 1_000_000:
                    raise RuntimeErrorWithCode("image_too_large", "Request JPEG or a smaller max_width (encoded image exceeds 1 MB).")
                result["image"] = {"mime_type": "image/" + params.image_format, "base64": base64.b64encode(data).decode(),
                                   "width": width, "height": height,
                                   "image_to_screen": {"scale_x": w / width, "scale_y": h / height,
                                                       "offset_x": 0, "offset_y": 0}}
            return result
        raise ValueError("unsupported operation")

    def _wait(self, client, params):
        r = self.runtime
        with r.lock:
            start_cursor = params.cursor or r.ledger.cursor()
            r.ledger.decode(start_cursor)
            lease_id = r.lease_id if r.owner == client else None
        scan_cursor = start_cursor
        deadline = time.perf_counter() + params.timeout
        reason = "timeout"
        while not self.closed.is_set():
            with r.lock:
                r.tick()
                events = r.ledger.read(scan_cursor, 200)
                scan_cursor = events["next_cursor"]
                if events["lost_events"]:
                    reason = "ledger_gap"
                    break
                if lease_id is not None and r.lease_id != lease_id:
                    reason = "lease_ended"
                    break
                if lease_id is not None and r.expires - r.clock() <= 0.05:
                    reason = "lease_expiring"
                    break
                if any(not params.kinds or e["kind"] in params.kinds for e in events["events"]):
                    reason = "event"
                    break
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            if not events["has_more"]:
                self.closed.wait(min(0.01, remaining))
        if self.closed.is_set():
            reason = "core_closed"
        return {"reason": reason, **r.ledger.read(start_cursor), "lease": r.lease()}


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        self.connection.settimeout(5)
        try:
            raw = self.rfile.readline(MAX_REQUEST + 1)
            if len(raw) > MAX_REQUEST or not raw.endswith(b"\n"):
                response = {"ok": False, "error": {"code": "message_too_large", "message": "Send one bounded JSON line."}}
            else:
                try:
                    response = self.server.service.handle(json.loads(raw))
                except (ValueError, UnicodeError):
                    response = {"ok": False, "error": {"code": "invalid_json", "message": "Expected a JSON object."}}
            self.wfile.write(json.dumps(response, allow_nan=False).encode() + b"\n")
        except (OSError, ValueError):
            return


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = False
    daemon_threads = True

    def __init__(self, service, port=0):
        self.service = service
        self.slots = threading.BoundedSemaphore(32)
        super().__init__(("127.0.0.1", port), Handler)

    def process_request(self, request, client_address):
        if not self.slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except BaseException:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()

    def endpoint(self):
        return {"version": 1, "host": "127.0.0.1", "port": self.server_address[1],
                "runtime_id": self.service.runtime.ledger.epoch,
                "controller_token": self.service.controller_token,
                "observer_token": self.service.observer_token}


class Client:
    def __init__(self, endpoint, *, client_id=None, observer=False):
        self.endpoint = json.loads(Path(endpoint).read_text()) if isinstance(endpoint, (str, Path)) else endpoint
        if self.endpoint.get("host") != "127.0.0.1" or self.endpoint.get("version") != 1:
            raise ValueError("expected a version 1 loopback endpoint")
        self.client_id = client_id or uuid4().hex
        self.token = self.endpoint["observer_token" if observer else "controller_token"]

    def call(self, op, args=None, *, request_id=None):
        request = {"version": 1, "token": self.token, "client_id": self.client_id,
                   "request_id": request_id or uuid4().hex, "op": op, "args": args or {}}
        data = json.dumps(request, allow_nan=False).encode() + b"\n"
        if len(data) > MAX_REQUEST:
            raise ValueError("request exceeds 64 KiB")
        with socket.create_connection(("127.0.0.1", self.endpoint["port"]), timeout=5) as sock:
            sock.settimeout(65)
            sock.sendall(data)
            with sock.makefile("rb") as reader:
                raw = reader.readline(MAX_RESPONSE + 1)
        if len(raw) > MAX_RESPONSE or not raw.endswith(b"\n"):
            raise ValueError("incomplete or oversized core response")
        response = json.loads(raw)
        if not response["ok"]:
            error = response["error"]
            raise RuntimeErrorWithCode(error["code"], error["message"])
        if response["runtime_id"] != self.endpoint["runtime_id"]:
            raise ValueError("core restarted; reload its endpoint")
        return response["result"]
