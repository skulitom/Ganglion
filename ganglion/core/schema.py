"""Version 1 runtime inputs. Coordinates are physical pixels on the primary display."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Pixel = Annotated[int, Field(strict=True, ge=0, le=65535)]
Channel = Annotated[int, Field(strict=True, ge=0, le=255)]
Identifier = Annotated[str, Field(min_length=1, max_length=128)]
KeyName = Annotated[str, Field(min_length=1, max_length=16, pattern=r"^[a-z0-9]+$")]
Delta = Annotated[int, Field(strict=True, ge=-4000, le=4000)]
Button = Literal["left", "right", "middle"]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Empty(Model):
    pass


class LeaseSpec(Model):
    seconds: float = Field(default=30, ge=0.1, le=60)


class ReadSpec(Model):
    cursor: str | None = Field(default=None, max_length=128)
    limit: int = Field(default=40, ge=1, le=200)
    image: bool = False
    max_width: int = Field(default=640, ge=64, le=1920)
    image_format: Literal["jpeg", "png"] = "jpeg"


class WaitSpec(Model):
    cursor: str | None = Field(default=None, max_length=128)
    timeout: float = Field(default=20, ge=0, le=60)
    kinds: list[str] = Field(default_factory=list, max_length=16)


class WatchSpec(Model):
    name: str = Field(min_length=1, max_length=80)
    snapshot_id: Identifier
    region: tuple[Pixel, Pixel, Annotated[int, Field(gt=0, le=8192)],
                  Annotated[int, Field(gt=0, le=8192)]]
    kind: Literal["color", "motion", "track", "flow"] = "color"
    color_rgb: tuple[Channel, Channel, Channel] | None = None
    tolerance: int = Field(default=20, ge=0, le=100)
    min_pixels: int = Field(default=25, ge=1, le=1000000)
    max_pixels: int | None = Field(default=None, ge=1, le=10000000)
    threshold: int = Field(default=30, ge=1, le=255)     # motion: per-pixel grey difference
    lag_ms: int = Field(default=60, ge=10, le=1000)       # motion: age of the comparison frame
    persist: int = Field(default=2, ge=1, le=20)          # motion: consecutive comparisons before it counts
    template_region: tuple[Pixel, Pixel, Annotated[int, Field(gt=0, le=1024)],
                           Annotated[int, Field(gt=0, le=1024)]] | None = None   # track: crop of the newest frame
    search_px: int = Field(default=120, ge=4, le=2000)    # track: search radius around the last position
    min_score: float = Field(default=0.55, ge=0.1, le=1.0)
    update: float = Field(default=0.1, ge=0.0, le=1.0)    # track: template adaptation per match
    flow_scale: int = Field(default=4, ge=1, le=8)        # flow: reduction factor of the flow field
    flow_threshold: float = Field(default=3.0, ge=0.5, le=100)   # flow: residual px over the lag that counts as independent
    coordinate_space: Literal["screen"] = "screen"

    @model_validator(mode="after")
    def check_kind(self):
        if self.kind == "color" and self.color_rgb is None:
            raise ValueError("a color watch needs color_rgb")
        if self.kind == "track" and self.template_region is None:
            raise ValueError("a track watch needs template_region")
        return self


class FireSpec(Model):
    button: Button = "left"
    hold_ms: int = Field(default=100, ge=20, le=1000)
    repeat: int = Field(default=1, ge=1, le=20)
    interval_ms: int = Field(default=250, ge=0, le=3000)


class AlignOptions(Model):
    """Parameters a reflex uses to start an align intent on its own watch (or on the track it spawns)."""
    controller: Literal["deterministic", "connectome"] = "deterministic"
    speed_px_s: float = Field(default=1200, ge=100, le=5000)
    point: tuple[Pixel, Pixel] | None = None
    tolerance_px: float = Field(default=12, ge=1, le=200)
    settle_ms: int = Field(default=30, ge=10, le=500)
    gain: float = Field(default=1.0, ge=0.05, le=20)
    max_step: int = Field(default=60, ge=1, le=400)
    absence_ms: int = Field(default=300, ge=50, le=2000)
    timeout_seconds: float = Field(default=3, ge=0.1, le=30)
    fire: FireSpec | None = None


class ArmSpec(Model):
    watch_id: Identifier
    response: Literal["click", "notify", "key", "align", "track"] = "click"
    trigger: Literal["appear", "present"] = "appear"
    cooldown_ms: int = Field(default=250, ge=50, le=60000)
    max_fires: int = Field(default=20, ge=1, le=1000)
    hold_ms: int = Field(default=20, ge=5, le=1000)
    ttl_seconds: float = Field(default=30, ge=0.1, le=60)
    key: KeyName | None = None
    align: AlignOptions | None = None

    @model_validator(mode="after")
    def check_response(self):
        if self.response == "click" and self.hold_ms > 100:
            raise ValueError("a click holds at most 100 ms")
        if self.response == "key" and self.key is None:
            raise ValueError("a key response names the key")
        if self.response in ("align", "track") and self.align is None:
            raise ValueError("an align or track response needs align options")
        return self


class RemoveSpec(Model):
    id: Identifier


class ConditionSpec(Model):
    watch_id: Identifier
    present: bool = True


class IntentSpec(Model):
    program: Literal["reach", "drag", "align", "move"] = "reach"
    watch_id: Identifier | None = None
    keys: list[KeyName] = Field(default_factory=list, max_length=4)   # move: keys held together
    click: bool = False
    timeout_seconds: float = Field(default=5, ge=0.1, le=30)
    tolerance_px: float = Field(default=6, ge=1, le=200)
    settle_ms: int = Field(default=30, ge=10, le=500)
    speed_px_s: float = Field(default=1200, ge=100, le=5000)
    destination: tuple[Pixel, Pixel] | None = None
    destination_watch_id: Identifier | None = None
    until: ConditionSpec | None = None
    verification_seconds: float = Field(default=1, ge=0.1, le=5)
    controller: Literal["deterministic", "connectome"] = "deterministic"
    # align: turn the view with relative mouse deltas until the target sits at point (client centre
    # by default), then optionally hold a button; gain is mouse counts per pixel of error.
    point: tuple[Pixel, Pixel] | None = None
    gain: float = Field(default=1.0, ge=0.05, le=20)
    max_step: int = Field(default=60, ge=1, le=400)
    absence_ms: int = Field(default=300, ge=50, le=2000)
    fire: FireSpec | None = None

    @model_validator(mode="after")
    def check_program(self):
        if self.program == "move":
            if not self.keys:
                raise ValueError("move needs keys")
            if self.watch_id is not None or self.destination is not None or self.destination_watch_id is not None \
                    or self.click or self.point is not None or self.fire is not None:
                raise ValueError("move accepts keys, an optional until condition and timeout_seconds")
            return self
        if self.keys:
            raise ValueError("keys are a move parameter")
        if self.watch_id is None:
            raise ValueError("this program needs watch_id")
        if self.program == "align":
            if self.destination is not None or self.destination_watch_id is not None or self.until is not None or self.click:
                raise ValueError("align accepts point, gain, max_step, absence_ms and fire; not drag or click parameters")
            return self
        if self.point is not None or self.fire is not None:
            raise ValueError("point and fire are align parameters")
        if self.program == "reach":
            if self.destination is not None or self.destination_watch_id is not None or self.until is not None:
                raise ValueError("destination and until are drag parameters")
        else:
            if (self.destination is None) == (self.destination_watch_id is None):
                raise ValueError("drag requires exactly one destination point or destination_watch_id")
            if self.click:
                raise ValueError("drag owns its button sequence; do not set click")
        return self


class InputSpec(Model):
    """One bounded agent-chosen input.

    move/click: absolute pointer at point. key: tap one key. hold: hold up to four keys together
    for hold_ms (re-issue to extend). look: relative mouse delta in counts, optionally spread over
    spread_ms. button: hold a mouse button without moving the pointer.
    """
    action: Literal["move", "click", "key", "hold", "look", "button"]
    snapshot_id: Identifier
    point: tuple[Pixel, Pixel] | None = None
    button: Button = "left"
    hold_ms: int = Field(default=20, ge=5, le=1000)
    key: KeyName | None = None
    keys: list[KeyName] = Field(default_factory=list, max_length=4)
    delta: tuple[Delta, Delta] | None = None
    spread_ms: int = Field(default=0, ge=0, le=500)

    @model_validator(mode="after")
    def check_action(self):
        if self.action in ("move", "click") and self.point is None:
            raise ValueError(f"{self.action} needs a point")
        if self.action == "click" and self.hold_ms > 100:
            raise ValueError("a click holds at most 100 ms")
        if self.action == "key" and self.key is None:
            raise ValueError("key needs key")
        if self.action == "hold" and not self.keys:
            raise ValueError("hold needs keys")
        if self.action == "look" and self.delta is None:
            raise ValueError("look needs delta")
        return self


class Request(Model):
    version: Literal[1] = 1
    token: str = Field(min_length=16, max_length=256)
    client_id: str = Field(min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    request_id: str = Field(min_length=1, max_length=128)
    op: str = Field(min_length=1, max_length=32)
    args: dict = Field(default_factory=dict)


OPERATIONS = {
    "status": Empty, "look": ReadSpec, "wait": WaitSpec,
    "claim": LeaseSpec, "renew": LeaseSpec, "watch": WatchSpec,
    "arm": ArmSpec, "disarm": RemoveSpec, "unwatch": RemoveSpec, "halt": Empty,
    "intent": IntentSpec, "cancel": RemoveSpec,
    "input": InputSpec,
}
READ_ONLY = {"status", "look", "wait"}
