"""Pipeline orchestration: fetch -> licence -> quality -> frame -> render -> upload -> notify.

The pass over candidates is lazy and stops at the first plate that clears every
gate, so a normal run costs a handful of BHL calls and one vision call rather
than scoring the whole pool.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from . import bhl, imaging, licensing, metadata, notify, video, youtube
from .config import Config, require_env
from .history import History
from .vision import VisionAuthError, VisionVerdict, inspect_plate, passes

log = logging.getLogger(__name__)


@dataclass
class Rejection:
    page_id: str
    stage: str
    reason: str


@dataclass
class RunResult:
    accepted: bool
    summary: dict[str, Any] = field(default_factory=dict)
    rejections: list[Rejection] = field(default_factory=list)
    video_path: Path | None = None
    image_path: Path | None = None


def _anthropic_client():
    from anthropic import Anthropic

    return Anthropic(api_key=require_env("ANTHROPIC_API_KEY"))


def select_and_build(
    cfg: Config,
    *,
    history: History,
    session: requests.Session | None = None,
    vision_client: Any = None,
    dry_run: bool = False,
) -> RunResult:
    """Find one publishable plate and produce the framed still and video."""
    session = session or requests.Session()
    client = bhl.BHLClient(require_env("BHL_API_KEY"), session=session)

    if cfg.vision.enabled and vision_client is None:
        vision_client = _anthropic_client()

    rejections: list[Rejection] = []
    vision_calls = 0

    candidates = bhl.iter_candidates(
        client,
        subjects=cfg.source.subjects,
        page_types=cfg.source.page_types,
        year_min=cfg.source.year_min,
        year_max=cfg.source.year_max,
        titles_per_subject=cfg.source.titles_per_subject,
        max_items_per_title=cfg.source.max_items_per_title,
        max_pages_per_item=cfg.source.max_pages_per_item,
        limit=cfg.source.max_candidates,
        # Published work is filtered inside the walk so it never consumes the
        # candidate budget, and the window rotates as the channel grows.
        skip_pages=history.page_ids,
        skip_items=history.item_ids,
        skip_titles=history.recent_title_ids(cfg.source.title_cooldown),
        title_offset=len(history.entries),
    )

    for candidate in candidates:
        page_id = candidate.page_id

        if history.has_page(page_id):
            rejections.append(Rejection(page_id, "history", "page already published"))
            continue
        if history.has_item(candidate.item_id):
            rejections.append(
                Rejection(page_id, "history", f"volume {candidate.item_id} already featured")
            )
            continue

        verdict = licensing.evaluate(candidate, cfg.license)
        if not verdict.allowed:
            rejections.append(Rejection(page_id, "licence", verdict.reason))
            continue

        try:
            data = bhl.download_page_image(candidate, session=session)
            img = imaging.load_image(data)
            imaging.check_source_resolution(
                img, cfg.image.min_source_width, cfg.image.min_source_height
            )
            imaging.check_aspect(img, cfg.image.max_source_aspect)
            # Cheap and local, so both run before spending a vision call.
            imaging.check_border_tone(img, cfg.image.min_border_luminance)
            imaging.check_ink_coverage(img, cfg.image.min_ink_coverage)
        except (bhl.BHLError, imaging.ImageError, requests.RequestException) as exc:
            rejections.append(Rejection(page_id, "download", str(exc)))
            continue

        vision_verdict: VisionVerdict | None = None
        if cfg.vision.enabled:
            if vision_calls >= cfg.vision.max_vision_calls:
                rejections.append(Rejection(page_id, "vision", "vision call budget exhausted"))
                break
            vision_calls += 1
            # A rejected credential is fatal, not a property of this candidate;
            # let it propagate rather than burning the budget on every plate.
            vision_verdict = inspect_plate(vision_client, img, model=cfg.vision.model)
            ok, reason = passes(
                vision_verdict,
                min_quality=cfg.vision.min_scan_quality,
                caption_mode=cfg.vision.caption_mode,
            )
            if not ok:
                rejections.append(Rejection(page_id, "vision", reason))
                continue
            if cfg.vision.caption_mode == "log_only" and not vision_verdict.caption_embedded:
                # Recorded, deliberately not enforced in v1.
                log.info(
                    "page %s has no plate-embedded lettering; accepting anyway (caption_mode=log_only)",
                    page_id,
                )

        try:
            framed = imaging.frame_vertical(
                img,
                width=cfg.image.width,
                height=cfg.image.height,
                margin_ratio=cfg.image.margin_ratio,
                letterbox=cfg.image.letterbox,
                fixed_fill_color=cfg.image.fixed_fill_color,
                border_px=cfg.image.border_px,
                border_color=cfg.image.border_color,
            )
        except imaging.ImageError as exc:
            rejections.append(Rejection(page_id, "framing", str(exc)))
            continue

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        out_dir = cfg.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        image_path = out_dir / f"{stamp}-{page_id}.png"
        video_path = out_dir / f"{stamp}-{page_id}.mp4"
        framed.image.save(image_path, format="PNG")

        if not dry_run:
            video.render_still(
                framed.image,
                video_path,
                duration_seconds=cfg.video.duration_seconds,
                fps=cfg.video.fps,
                crf=cfg.video.crf,
            )

        summary = {
            "page_id": page_id,
            "item_id": candidate.item_id,
            "title_id": candidate.title_id,
            "title": metadata.build_title(candidate, vision_verdict),
            "description": metadata.build_description(candidate, vision_verdict),
            "citation": candidate.citation(),
            "page_url": candidate.page_url,
            "thumb_url": candidate.thumb_url,
            "subject": candidate.subject,
            "rights": candidate.rights,
            "license": candidate.license_name,
            "licence_reason": verdict.reason,
            "source_size": list(framed.source_size),
            "plate_size": list(framed.plate_size),
            "fill_color": list(framed.fill_color),
            "upscaled": framed.upscaled,
            "letterbox": cfg.image.letterbox,
            "duration_seconds": cfg.video.duration_seconds,
        }
        if vision_verdict:
            summary.update(
                {
                    "scan_quality": vision_verdict.scan_quality,
                    "caption_embedded": vision_verdict.caption_embedded,
                    "species_name_visible": vision_verdict.species_name_visible,
                    "is_spread": vision_verdict.is_spread,
                    "subject_summary": vision_verdict.subject_summary,
                    "vision_issues": vision_verdict.issues,
                    "caption_mode": cfg.vision.caption_mode,
                }
            )

        return RunResult(
            accepted=True,
            summary=summary,
            rejections=rejections,
            video_path=video_path if not dry_run else None,
            image_path=image_path,
        )

    return RunResult(accepted=False, rejections=rejections)


def run(
    cfg: Config,
    *,
    dry_run: bool = False,
    skip_upload: bool = False,
    count: int = 1,
) -> list[RunResult]:
    """Produce ``count`` plates. Normally 1; more for seeding a new channel.

    History is recorded between iterations, so a batch never picks the same
    plate or volume twice.
    """
    history = History(cfg.history_path)
    results: list[RunResult] = []
    for n in range(count):
        if count > 1:
            log.info("--- plate %d of %d ---", n + 1, count)
        result = _run_once(cfg, history, dry_run=dry_run, skip_upload=skip_upload)
        results.append(result)
        if not result.accepted:
            log.error("stopping batch after %d of %d", n, count)
            break
    return results


def _run_once(
    cfg: Config,
    history: History,
    *,
    dry_run: bool = False,
    skip_upload: bool = False,
) -> RunResult:
    result = select_and_build(cfg, history=history, dry_run=dry_run)

    if not result.accepted:
        log.error(
            "no publishable plate found; %d candidates rejected", len(result.rejections)
        )
        for rej in result.rejections[:20]:
            log.error("  page %s rejected at %s: %s", rej.page_id, rej.stage, rej.reason)
        return result

    for rej in result.rejections:
        log.info("page %s rejected at %s: %s", rej.page_id, rej.stage, rej.reason)

    if dry_run or skip_upload or not cfg.upload.enabled:
        log.info("upload skipped; artefacts in %s", cfg.output_dir)
        result.summary["upload_skipped"] = True
        notify.notify(result.summary, enabled=cfg.notify.enabled)
        notify.save_summary(result.summary, cfg.output_dir / "summary.json")
        return result

    creds = youtube.build_credentials(
        require_env("YOUTUBE_CLIENT_ID"),
        require_env("YOUTUBE_CLIENT_SECRET"),
        require_env("YOUTUBE_REFRESH_TOKEN"),
    )
    publish_at = youtube.scheduled_publish_time(cfg.upload.publish_delay_hours)

    assert result.video_path is not None
    upload = youtube.upload_video(
        result.video_path,
        title=result.summary["title"],
        description=result.summary["description"],
        tags=cfg.upload.tags,
        category_id=cfg.upload.category_id,
        privacy_status=cfg.upload.privacy_status,
        publish_at=publish_at,
        made_for_kids=cfg.upload.made_for_kids,
        credentials=creds,
    )

    result.summary.update(
        {
            "video_id": upload.video_id,
            "url": upload.url,
            "studio_url": upload.studio_url,
            "publish_at": upload.publish_at,
        }
    )

    history.record(
        {
            "page_id": result.summary["page_id"],
            "item_id": result.summary["item_id"],
            # Recorded so the cooldown can see which work this came from.
            "title_id": result.summary["title_id"],
            "title": result.summary["title"],
            "video_id": upload.video_id,
            "publish_at": upload.publish_at,
        }
    )
    history.save()

    notify.notify(result.summary, enabled=cfg.notify.enabled)
    notify.save_summary(result.summary, cfg.output_dir / "summary.json")
    log.info("uploaded %s, publishes %s", upload.video_id, upload.publish_at)
    return result
