"""Version 1 runtime inputs. Coordinates are physical pixels on the primary display."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Pixel = Annotated[int, Field(strict=True, ge=0, le=65535)]
Channel = Annotated[int, Field(strict=True, ge=0, le=255)]
Identifier = Annotated[str, Field(min_length=1, max_length=128)]


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
    color_rgb: tuple[Channel, Channel, Channel]
    tolerance: int = Field(default=20, ge=0, le=100)
    min_pixels: int = Field(default=25, ge=1, le=1000000)
    coordinate_space: Literal["screen"] = "screen"


class ArmSpec(Model):
    watch_id: Identifier
    response: Literal["click", "notify"] = "click"
    trigger: Literal["appear", "present"] = "appear"
    cooldown_ms: int = Field(default=250, ge=50, le=60000)
    max_fires: int = Field(default=20, ge=1, le=1000)
    hold_ms: int = Field(default=20, ge=5, le=100)
    ttl_seconds: float = Field(default=30, ge=0.1, le=60)


class RemoveSpec(Model):
    id: Identifier


class ConditionSpec(Model):
    watch_id: Identifier
    present: bool = True


class IntentSpec(Model):
    program: Literal["reach", "drag"] = "reach"
    watch_id: Identifier
    click: bool = False
    timeout_seconds: float = Field(default=5, ge=0.1, le=30)
    tolerance_px: float = Field(default=6, ge=1, le=64)
    settle_ms: int = Field(default=30, ge=10, le=500)
    speed_px_s: float = Field(default=1200, ge=100, le=5000)
    destination: tuple[Pixel, Pixel] | None = None
    destination_watch_id: Identifier | None = None
    until: ConditionSpec | None = None
    verification_seconds: float = Field(default=1, ge=0.1, le=5)
    controller: Literal["deterministic", "connectome"] = "deterministic"

    @model_validator(mode="after")
    def check_program(self):
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
    action: Literal["move", "click"]
    snapshot_id: Identifier
    point: tuple[Pixel, Pixel]


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
