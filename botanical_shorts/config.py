"""Configuration loading.

Settings live in ``config/settings.yaml`` so the design decisions Brian signed
off on (scope, letterbox treatment, duration) are editable without touching
code. Secrets never live here -- they come from the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "settings.yaml"


class ConfigError(RuntimeError):
    """Raised when configuration or required secrets are missing/invalid."""


@dataclass(frozen=True)
class SourceConfig:
    # Subject tags queried against BHL. Botanical-only for v1; adding
    # zoological/entomological subjects here is the one-line widening.
    subjects: list[str]
    # Only these page types are considered plates worth publishing.
    page_types: list[str]
    # Publication year window. Classic plate-embedded lettering era.
    year_min: int
    year_max: int
    # How many titles to pull per subject, and pages to consider per item.
    titles_per_subject: int
    max_items_per_title: int
    max_pages_per_item: int
    # Hard cap on candidates evaluated in one run, to bound API/vision spend.
    max_candidates: int
    # How many recently published videos a work stays locked for. Guards
    # against near-identical plates from one long-running serial landing close
    # together in the feed. Bounded by the number of distinct usable titles --
    # a cooldown approaching that count leaves the walk nothing to publish.
    title_cooldown: int = 120
    # How many plates one work may contribute to a single batch. Replaces the
    # useful half of the cooldown: near-identical plates from one serial are
    # most jarring when they land in the same week's uploads. 0 = unlimited.
    max_plates_per_title_per_batch: int = 3


@dataclass(frozen=True)
class LicenseConfig:
    # Substring matches (case-insensitive) against BHL rights/license fields.
    allowed_rights: list[str]
    allowed_licenses: list[str]
    # When BHL reports no rights information at all, do we allow it?
    allow_unknown: bool


@dataclass(frozen=True)
class VisionConfig:
    enabled: bool
    model: str
    # "log_only"  -> record the caption verdict but never reject on it (v1)
    # "hard_gate" -> reject candidates without plate-embedded lettering
    caption_mode: str
    # Scan quality is a real gate from day one; below this the plate is
    # rejected (damage, foxing, bleed-through, cut-off plate).
    min_scan_quality: int
    max_vision_calls: int
    # How many plates per batch may reach review without the model having seen
    # them, when the call budget runs out mid-walk. Zero restores the old
    # behaviour of stopping the walk instead.
    max_uninspected_per_batch: int = 0


@dataclass(frozen=True)
class ImageConfig:
    width: int
    height: int
    # Fraction of the frame's short edge left as margin around the plate.
    margin_ratio: float
    # "sampled_paper" -> fill letterbox margins with the plate's own paper
    #                    tone, sampled from its border (signed off)
    # "fixed"         -> constant colour from fixed_fill_color
    # "black"         -> plain bars
    letterbox: str
    fixed_fill_color: str
    # Subtle inner keyline drawn around the plate edge; 0 disables.
    border_px: int
    border_color: str
    # Minimum source resolution accepted, before framing.
    min_source_width: int
    min_source_height: int
    # Plates wider than this (width/height) are skipped as unframeable.
    max_source_aspect: float
    # Minimum mean luminance of the scan's darkest edge strip. Below this the
    # scan has a black frame or dark mount, sampled_paper falls back to
    # parchment, and the plate reads as a dark rectangle pasted on a sheet.
    min_border_luminance: float = 60.0
    # Minimum fraction of the plate carrying ink, measured against its own
    # paper tone. Catches faint pencil studies, which score well on scan
    # quality -- the scan is fine -- and still frame as an empty page.
    min_ink_coverage: float = 0.05
    # Coverage measured inside the inked region instead of across the sheet.
    # Whole-sheet coverage cannot tell a small exact engraving from a faint
    # sketch: both use little of the paper, and only one reads as empty.
    min_subject_ink_coverage: float = 0.0


@dataclass(frozen=True)
class VideoConfig:
    duration_seconds: float
    fps: int
    crf: int


@dataclass(frozen=True)
class UploadConfig:
    enabled: bool
    privacy_status: str
    # Hours from now for the scheduled auto-publish (the passive veto window).
    publish_delay_hours: int
    category_id: str
    tags: list[str]
    made_for_kids: bool


@dataclass(frozen=True)
class NotifyConfig:
    enabled: bool
    # Slack incoming-webhook URL comes from SLACK_WEBHOOK_URL.
    channel_hint: str


@dataclass(frozen=True)
class Config:
    source: SourceConfig
    license: LicenseConfig
    vision: VisionConfig
    image: ImageConfig
    video: VideoConfig
    upload: UploadConfig
    notify: NotifyConfig
    history_path: Path
    output_dir: Path
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"settings.yaml is missing the '{name}' section")
    return value


def load_config(path: Path | str | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    data = yaml.safe_load(path.read_text()) or {}

    cfg = Config(
        source=SourceConfig(**_section(data, "source")),
        license=LicenseConfig(**_section(data, "license")),
        vision=VisionConfig(**_section(data, "vision")),
        image=ImageConfig(**_section(data, "image")),
        video=VideoConfig(**_section(data, "video")),
        upload=UploadConfig(**_section(data, "upload")),
        notify=NotifyConfig(**_section(data, "notify")),
        history_path=REPO_ROOT / str(data.get("history_path", "state/history.json")),
        output_dir=REPO_ROOT / str(data.get("output_dir", "build")),
        raw=data,
    )
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if cfg.vision.caption_mode not in {"log_only", "hard_gate"}:
        raise ConfigError(
            f"vision.caption_mode must be 'log_only' or 'hard_gate', "
            f"got {cfg.vision.caption_mode!r}"
        )
    if cfg.image.letterbox not in {"sampled_paper", "fixed", "black"}:
        raise ConfigError(
            f"image.letterbox must be one of sampled_paper/fixed/black, "
            f"got {cfg.image.letterbox!r}"
        )
    if not 0 <= cfg.image.margin_ratio < 0.5:
        raise ConfigError("image.margin_ratio must be in [0, 0.5)")
    if cfg.video.duration_seconds <= 0:
        raise ConfigError("video.duration_seconds must be positive")
    if not cfg.source.subjects:
        raise ConfigError("source.subjects must list at least one subject")


def require_env(name: str) -> str:
    """Fetch a required secret from the environment."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(f"required environment variable {name} is not set")
    return value


def optional_env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()
