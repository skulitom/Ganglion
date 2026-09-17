"""MCP SDK v2 tools with native images and structured results."""
from __future__ import annotations

import asyncio
import json
from typing import Literal

from mcp.server import MCPServer
from mcp.types import CallToolResult, ImageContent, TextContent, ToolAnnotations

from ganglion.core.protocol import Client
from ganglion.core.schema import ArmSpec, WatchSpec, IntentSpec, InputSpec


def build(endpoint, *, observer=False, client_id=None):
    client = Client(endpoint, observer=observer, client_id=client_id)
    server = MCPServer("ganglion_mcp", instructions=(
        "Use status and look, claim a bounded lease, teach color watches in physical screen pixels, "
        "then arm reflexes or start a reach/drag intent. Read-only calls never renew a lease. Explicitly renew before expiry. "
        "Pass next_cursor to look/wait to consume events without gaps. Halt when finished. "
        "Supports color components, cursor-feedback reach/drag, and bounded left clicks; no semantic recognition. "
        "Drag can stop at a destination or taught condition, then verify that condition after release. "
        "Align turns a first-person view with relative mouse deltas until a watched target sits at a point, "
        "then optionally fires; watches can detect motion while the runtime is not moving the view; "
        "input can hold keys, send relative look deltas and hold buttons for bounded times. "
        "Intent completion reports observed state, not arbitrary application success."))
    read = ToolAnnotations(read_only_hint=True, destructive_hint=False, open_world_hint=False)
    write = ToolAnnotations(read_only_hint=False, destructive_hint=False, open_world_hint=False)

    async def call(op, args=None, request_id=None):
        try:
            result = await asyncio.to_thread(client.call, op, args, request_id=request_id)
            picture = result.pop("image", None)
            content = [TextContent(type="text", text=json.dumps(result))]
            if picture:
                result["image_transform"] = {k: v for k, v in picture.items() if k != "base64"}
                content[0] = TextContent(type="text", text=json.dumps(result))
                content.append(ImageContent(type="image", data=picture["base64"], mime_type=picture["mime_type"]))
            return CallToolResult(content=content, structured_content=result)
        except Exception as exc:
            return CallToolResult(content=[TextContent(type="text", text=f"{type(exc).__name__}: {exc}")],
                                  is_error=True)

    @server.tool(annotations=read)
    async def ganglion_status() -> CallToolResult:
        """Read runtime state, lease expiry, watches and reflexes; does not renew control."""
        return await call("status")

    @server.tool(annotations=read)
    async def ganglion_look(cursor: str | None = None, image: bool = True, limit: int = 40,
                            max_width: int = 640, image_format: Literal["jpeg", "png"] = "jpeg") -> CallToolResult:
        """Read events and a snapshot. PNG preserves small details; width is bounded 64–1920.
        Use image_to_screen before teaching; keep next_cursor. Oversized images require a smaller width.
        """
        return await call("look", {"cursor": cursor, "image": image, "limit": limit,
                                   "max_width": max_width, "image_format": image_format})

    @server.tool(annotations=write)
    async def ganglion_claim(seconds: float = 30, request_id: str | None = None) -> CallToolResult:
        """Acquire exclusive control for 0.1–60 seconds. A retry can reuse request_id."""
        return await call("claim", {"seconds": seconds}, request_id)

    @server.tool(annotations=write)
    async def ganglion_renew(seconds: float = 30, request_id: str | None = None) -> CallToolResult:
        """Explicitly renew your live control lease. Reflex TTLs remain independently bounded."""
        return await call("renew", {"seconds": seconds}, request_id)

    @server.tool(annotations=write)
    async def ganglion_watch(spec: WatchSpec, request_id: str | None = None) -> CallToolResult:
        """Teach a target in a screen-pixel region bound to look's snapshot_id: a connected RGB
        colour component (kind color), or the largest blob that changed against a frame lag_ms
        older (kind motion), reported only while no own command is moving the view or player.
        """
        return await call("watch", spec.model_dump(), request_id)

    @server.tool(annotations=write)
    async def ganglion_arm(spec: ArmSpec, request_id: str | None = None) -> CallToolResult:
        """Arm an appearance/presence reflex with cooldown and firing budget: bounded left click,
        notify, a bounded key tap (key), or an align intent on the same watch (align options)."""
        return await call("arm", spec.model_dump(), request_id)

    @server.tool(annotations=read)
    async def ganglion_wait(cursor: str | None = None, timeout: float = 20,
                            kinds: list[str] | None = None) -> CallToolResult:
        """Wait for new events or lease expiry; returns early to permit deliberate renewal."""
        return await call("wait", {"cursor": cursor, "timeout": timeout, "kinds": kinds or []})

    @server.tool(annotations=write)
    async def ganglion_intent(spec: IntentSpec, request_id: str | None = None) -> CallToolResult:
        """Reach, drag, or align on a watched target.

        Reach can click after arrival. Drag requires a destination point or destination_watch_id;
        optional until={watch_id,present} releases early on that condition and verifies it after
        release. Arrival also releases, then verifies. A condition already satisfied prevents pickup.
        Align (first person) sends relative mouse deltas of gain counts per pixel of error until the
        target sits at point (client centre by default) for settle_ms, then holds fire.button for
        fire.hold_ms, repeat times with interval_ms between; it ends when the target is lost.

        Owns the pointer until completion, failure, or cancellation. Read intent state in status
        or wait for intent_completed/intent_failed/intent_cancelled. Timeout never renews a lease.
        """
        return await call("intent", spec.model_dump(), request_id)

    @server.tool(annotations=write)
    async def ganglion_cancel(intent_id: str, request_id: str | None = None) -> CallToolResult:
        """Cancel a reach/drag and release its input. Pending output drains before another may start."""
        return await call("cancel", {"id": intent_id}, request_id)

    @server.tool(annotations=write)
    async def ganglion_input(spec: InputSpec, request_id: str | None = None) -> CallToolResult:
        """Submit one bounded agent-chosen input: move/click at a point, key tap, hold of up to
        four keys for hold_ms (re-issue to extend), look (relative mouse delta in counts, optionally
        spread over spread_ms) or button hold without moving the pointer.

        Coordinates are physical screen pixels. Checks current layout/focus/occlusion; does not
        re-detect what is at the chosen point. Use an intent for a target that can move.
        """
        return await call("input", spec.model_dump(), request_id)

    @server.tool(annotations=write)
    async def ganglion_disarm(reflex_id: str, request_id: str | None = None) -> CallToolResult:
        """Remove a reflex; an already submitted bounded click still releases normally."""
        return await call("disarm", {"id": reflex_id}, request_id)

    @server.tool(annotations=write)
    async def ganglion_unwatch(watch_id: str, request_id: str | None = None) -> CallToolResult:
        """Remove a watch and all its reflexes."""
        return await call("unwatch", {"id": watch_id}, request_id)

    @server.tool(annotations=write)
    async def ganglion_halt(request_id: str | None = None) -> CallToolResult:
        """Stop all reflexes, release input, clear watches, and end the control lease."""
        return await call("halt", {}, request_id)

    return server
