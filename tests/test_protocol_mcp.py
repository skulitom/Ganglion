import asyncio
import json
import socket
import threading
import time

import pytest

from ganglion.core.output import MemoryOutput
from ganglion.core.protocol import Client, Server, Service
from ganglion.core.runtime import Runtime, RuntimeErrorWithCode
from ganglion.mcp.server import build


@pytest.fixture
def service():
    runtime = Runtime(time.perf_counter, MemoryOutput(time.perf_counter))
    return Service(runtime)


def request(service, op, args=None, *, client="a", ident="one", observer=False):
    return {"token": service.observer_token if observer else service.controller_token,
            "client_id": client, "request_id": ident, "op": op, "args": args or {}}


def test_auth_observer_and_unknown_operations(service):
    raw = request(service, "claim")
    raw["token"] = "x" * 32
    assert service.handle(raw)["error"]["code"] == "unauthorized"
    assert service.handle(request(service, "claim", observer=True))["error"]["code"] == "observer_only"
    assert service.handle(request(service, "status", observer=True))["ok"]
    assert service.handle(request(service, "shell"))["error"]["code"] == "unknown_operation"


def test_idempotent_retry_and_conflict(service):
    req = request(service, "claim", {"seconds": 1})
    first = service.handle(req)
    time.sleep(0.01)
    assert service.handle(req) == first
    assert len(service.runtime.ledger.actions) == 2
    req["args"] = {"seconds": 2}
    assert service.handle(req)["error"]["code"] == "request_id_conflict"


def test_intent_and_point_input_retry_authority_and_snapshot_contract():
    from test_reach import setup
    c, output, runtime, _, wid = setup()
    runtime.cancel("agent", runtime.intent.id)
    runtime.tick()
    service = Service(runtime)
    req = request(service, "intent", {"watch_id": wid}, client="agent", ident="intent")
    first = service.handle(req)
    assert first["ok"]
    assert service.handle(req) == first
    for op, args in (("intent", {"watch_id": wid}), ("cancel", {"id": runtime.intent.id}),
                     ("input", {"action": "move", "point": [2, 3], "snapshot_id": runtime.snapshot()["id"]})):
        assert service.handle(request(service, op, args, observer=True))["error"]["code"] == "observer_only"
    runtime.cancel("agent", runtime.intent.id)
    runtime.tick()
    raw = request(service, "input", {"action": "click", "point": [20, 30],
                                    "snapshot_id": runtime.snapshot()["id"]}, client="agent", ident="point")
    first = service.handle(raw)
    assert first["ok"]
    assert service.handle(raw) == first
    assert len(output.commands) == 1
    runtime.tick()
    raw["request_id"] = "outside"
    raw["args"]["point"] = [640, 10]
    assert service.handle(raw)["error"]["code"] == "point_outside_target"
    raw["args"]["snapshot_id"] = "old:1:1"
    assert service.handle(raw)["error"]["code"] == "stale_snapshot"


@pytest.mark.parametrize("args", [{"seconds": -1}, {"seconds": float("nan")}, {"seconds": 61}, {"extra": True}])
def test_invalid_arguments_have_no_effect(service, args):
    assert not service.handle(request(service, "claim", args))["ok"]
    assert service.runtime.owner is None


def test_wait_returns_before_lease_expires_without_renewal(service):
    service.handle(request(service, "claim", {"seconds": 0.15}))
    expiry = service.runtime.expires
    start = time.perf_counter()
    result = service.handle(request(service, "wait", {"timeout": 60}))
    assert result["result"]["reason"] == "lease_expiring"
    assert time.perf_counter() - start < 0.5
    assert service.runtime.expires == expiry


@pytest.fixture
def running_server(service):
    with Server(service) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        yield server
        service.closed.set()
        server.shutdown()
        thread.join()


def test_tcp_json_bounds_and_client(running_server):
    client = Client(running_server.endpoint())
    assert client.call("status")["state"] == "ready"
    with socket.create_connection(running_server.server_address) as connection:
        connection.sendall(b"not json\n")
        response = connection.makefile("rb").readline()
        assert json.loads(response)["error"]["code"] == "invalid_json"
    with pytest.raises(RuntimeErrorWithCode):
        Client(running_server.endpoint(), observer=True).call("claim")


def test_wait_does_not_block_halt(running_server):
    client = Client(running_server.endpoint())
    client.call("claim", {"seconds": 5})
    results = []
    waiting = threading.Thread(target=lambda: results.append(client.call("wait", {"timeout": 5})))
    waiting.start()
    time.sleep(0.05)
    client.call("halt")
    waiting.join(1)
    assert not waiting.is_alive()
    assert results[0]["reason"] in ("lease_ended", "event")


def test_mcp_discovery_structured_content_and_native_image(running_server):
    import numpy as np
    runtime = running_server.service.runtime
    runtime.observe(np.zeros((100, 120, 3), dtype=np.uint8), time.perf_counter(), 1,
                    {"hwnd": 0, "pid": 0, "rect": [0, 0, 120, 100]})
    async def exercise():
        from mcp import Client as MCPClient
        async with MCPClient(build(running_server.endpoint())) as client:
            tools = await client.list_tools()
            names = {tool.name for tool in tools.tools}
            assert {"ganglion_look", "ganglion_arm", "ganglion_halt", "ganglion_wait",
                    "ganglion_intent", "ganglion_cancel", "ganglion_input"} <= names
            result = await client.call_tool("ganglion_look")
            assert not result.is_error
            assert result.structured_content["snapshot"]["width"] == 120
            assert any(item.type == "image" for item in result.content)
            bad = await client.call_tool("ganglion_claim", {"seconds": -2})
            assert bad.is_error
    asyncio.run(exercise())


def test_lossless_observation_round_trip_and_coordinate_transform(running_server):
    import base64
    import cv2
    import numpy as np
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[100:130, 800:820] = [20, 140, 250]
    runtime = running_server.service.runtime
    runtime.observe(frame, time.perf_counter(), 1,
                    {"hwnd": 0, "pid": 0, "rect": [0, 0, 1280, 720]})

    async def exercise():
        from mcp import Client as MCPClient
        async with MCPClient(build(running_server.endpoint())) as client:
            default = await client.call_tool("ganglion_look")
            assert default.structured_content["image_transform"]["width"] == 640
            result = await client.call_tool("ganglion_look", {"max_width": 1280, "image_format": "png"})
            assert not result.is_error
            transform = result.structured_content["image_transform"]
            assert (transform["width"], transform["height"]) == (1280, 720)
            assert transform["image_to_screen"] == {"scale_x": 1, "scale_y": 1, "offset_x": 0, "offset_y": 0}
            picture = next(item for item in result.content if item.type == "image")
            assert picture.mime_type == "image/png"
            decoded = cv2.imdecode(np.frombuffer(base64.b64decode(picture.data), dtype=np.uint8), cv2.IMREAD_COLOR)
            np.testing.assert_array_equal(decoded, frame)
            for args in ({"max_width": 4096}, {"image_format": "gif"}):
                assert (await client.call_tool("ganglion_look", args)).is_error
    asyncio.run(exercise())


def test_observation_rejects_oversized_encoding_without_truncated_transport(running_server):
    import numpy as np
    runtime = running_server.service.runtime
    frame = np.random.default_rng(0).integers(0, 256, (720, 1280, 3), dtype=np.uint8)
    runtime.observe(frame, time.perf_counter(), 1,
                    {"hwnd": 0, "pid": 0, "rect": [0, 0, 1280, 720]})
    client = Client(running_server.endpoint(), observer=True)
    with pytest.raises(RuntimeErrorWithCode) as caught:
        client.call("look", {"image": True, "max_width": 1280, "image_format": "png"})
    assert caught.value.code == "image_too_large"
    assert client.call("look", {"image": True})["image"]["width"] == 640
