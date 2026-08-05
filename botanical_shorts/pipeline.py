"""Pipeline orchestration: fetch -> licence -> quality -> frame -> render -> upload -> notify.

The pass over candidates is lazy and stops at the first plate that clears every
gate, so a normal run costs a handful of BHL calls and one vision call rather
than scoring the whole pool.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

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


# One work contributes up to max_items_per_title x max_pages_per_item
# consecutive candidates -- eighteen, at the current settings. Scan condition
# is a property of the digitisation, not the page, so a volume shot against a
# black backdrop fails every plate in it for the same reason. Left alone, one
# such work eats eighteen full-resolution downloads and can exhaust the walk
# before a usable plate surfaces; this is what stopped the first real batch at
# zero. After this many hard rejections the rest of the work is skipped
# without paying for it.
MAX_TITLE_STRIKES = 4

# A separate, smaller allowance for vision calls that failed outright rather
# than returning a verdict. One flaky response should not end a run, but a
# sustained outage should stop quickly rather than retrying down the pool.
MAX_VISION_ERRORS = 4


def select_and_build(
    cfg: Config,
    *,
    history: History,
    session: requests.Session | None = None,
    vision_client: Any = None,
    dry_run: bool = False,
    blocked_pages: set[str] | None = None,
) -> RunResult:
    """Find one publishable plate and produce the framed still and video.

    ``blocked_pages`` is an optional caller-owned set of pages already known to
    fail a gate. Every gate here is deterministic given the scan, so a page
    rejected once will be rejected again; a batch passes the same set through
    each selection so it pays for each dud download once rather than once per
    plate. Mutated in place as new duds are found.
    """
    session = session or requests.Session()
    client = bhl.BHLClient(require_env("BHL_API_KEY"), session=session)

    if cfg.vision.enabled and vision_client is None:
        vision_client = _anthropic_client()

    rejections: list[Rejection] = []
    vision_calls = 0
    vision_errors = 0
    blocked = blocked_pages if blocked_pages is not None else set()
    title_strikes: dict[str, int] = {}

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
        skip_pages=set(history.page_ids) | blocked,
        skip_items=history.item_ids,
        skip_titles=history.recent_title_ids(cfg.source.title_cooldown),
        title_offset=len(history.entries),
    )

    def strike(candidate: bhl.PageCandidate) -> None:
        title_strikes[candidate.title_id] = title_strikes.get(candidate.title_id, 0) + 1
        blocked.add(candidate.page_id)

    for candidate in candidates:
        page_id = candidate.page_id

        if title_strikes.get(candidate.title_id, 0) >= MAX_TITLE_STRIKES:
            rejections.append(
                Rejection(
                    page_id,
                    "title",
                    f"work {candidate.title_id} already failed {MAX_TITLE_STRIKES} "
                    "plates on this walk; skipping the rest of it",
                )
            )
            continue

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
            strike(candidate)
            continue

        vision_verdict: VisionVerdict | None = None
        if cfg.vision.enabled:
            if vision_calls >= cfg.vision.max_vision_calls:
                rejections.append(Rejection(page_id, "vision", "vision call budget exhausted"))
                break
            # A rejected credential is fatal, not a property of this candidate;
            # let it propagate rather than burning the budget on every plate.
            vision_verdict = inspect_plate(vision_client, img, model=cfg.vision.model)
            if vision_verdict.error:
                # Not a verdict on the plate, so it does not spend the call
                # budget and does not retire the page: the plate goes back in
                # the pool for a later run to judge.
                vision_errors += 1
                rejections.append(
                    Rejection(page_id, "vision", f"inspection failed: {vision_verdict.error}")
                )
                if vision_errors >= MAX_VISION_ERRORS:
                    log.error(
                        "%d vision calls failed outright; stopping rather than "
                        "walking the pool against a broken API",
                        vision_errors,
                    )
                    break
                continue
            vision_calls += 1
            ok, reason = passes(
                vision_verdict,
                min_quality=cfg.vision.min_scan_quality,
                caption_mode=cfg.vision.caption_mode,
            )
            if not ok:
                rejections.append(Rejection(page_id, "vision", reason))
                strike(candidate)
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
            strike(candidate)
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
            # Enough to rebuild the candidate later. A reviewed batch is
            # published from a different, later run on a fresh machine, so
            # everything the frame depends on has to travel in the manifest.
            "candidate": _candidate_fields(candidate),
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
        log.error("no publishable plate found")
        for line in summarise_rejections(result.rejections):
            log.error("%s", line)
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


# -- batch review ------------------------------------------------------------
#
# The gates settle what is mechanically wrong. What survives can still be
# wrong in ways only a person can name -- a photograph among engravings, a
# skull diagram, a plate whose subject does not match its category. So a batch
# is built, looked at once, and only then published.
#
# The manifest stores metadata rather than rendered frames. Runners are
# ephemeral, so anything needed at publish time has to be committed, and
# committing a megabyte of PNG per plate would bloat the repository for no
# gain: the same page always yields the same frame, so it is cheaper to
# re-derive it than to carry it.


def _reason_key(reason: str) -> str:
    """Collapse a rejection reason to the kind of thing it is.

    Reasons carry the measurement that produced them -- "border luminance 6 is
    below 60" -- so counting them raw gives one bucket per candidate. Stripping
    the numbers turns a list back into a distribution.
    """
    import re

    key = re.sub(r"\d+(\.\d+)?", "N", reason)
    return key[:90]


def summarise_rejections(rejections: Sequence[Rejection]) -> list[str]:
    """Per-stage counts, then the reasons within each stage, commonest first.

    Printing the first N rejections instead -- which is what this replaced --
    reports whatever the walk happened to meet earliest. On a walk that dies at
    the vision budget, those are all early download rejections and the vision
    verdicts that actually stopped it never appear.
    """
    if not rejections:
        return ["no candidates were rejected: the pool itself was empty"]

    by_stage: dict[str, list[Rejection]] = {}
    for rej in rejections:
        by_stage.setdefault(rej.stage, []).append(rej)

    order = sorted(by_stage, key=lambda s: -len(by_stage[s]))
    lines = [
        "%d candidates rejected: %s"
        % (
            len(rejections),
            ", ".join(f"{s}={len(by_stage[s])}" for s in order),
        )
    ]
    for stage in order:
        counts: dict[str, int] = {}
        for rej in by_stage[stage]:
            key = _reason_key(rej.reason)
            counts[key] = counts.get(key, 0) + 1
        lines.append(f"  {stage}:")
        for key, n in sorted(counts.items(), key=lambda kv: -kv[1])[:8]:
            lines.append(f"    {n:4} x {key}")
    return lines


def _candidate_fields(cand: bhl.PageCandidate) -> dict[str, Any]:
    return {
        "page_id": cand.page_id,
        "item_id": cand.item_id,
        "title_id": cand.title_id,
        "title": cand.title,
        "year": cand.year,
        "publisher": cand.publisher,
        "authors": list(cand.authors),
        "page_types": list(cand.page_types),
        "rights": cand.rights,
        "license_name": cand.license_name,
        "license_url": cand.license_url,
        "source": cand.source,
        "subject": cand.subject,
    }


def build_batch(cfg: Config, *, count: int) -> list[dict[str, Any]]:
    """Select and frame ``count`` plates without uploading any of them.

    Each accepted plate is recorded into the in-memory history so the next
    iteration does not offer the same page, volume or recently-used work --
    but the history is never saved, because nothing here has been published.
    Rejections are written to history separately by :func:`apply_review`.
    """
    history = History(cfg.history_path)
    batch: list[dict[str, Any]] = []
    # Shared across the whole batch: every walk starts from the same subject
    # list, so without this each selection re-downloads the same duds.
    blocked: set[str] = set()

    for n in range(count):
        log.info("--- selecting %d of %d ---", n + 1, count)
        result = select_and_build(cfg, history=history, dry_run=True, blocked_pages=blocked)
        if not result.accepted:
            # Report *why*, the way _run_once does. A batch that stops short is
            # indistinguishable from an empty pool without this, and the
            # rejection list is the only record of which gate did it.
            log.warning("no further plate found; batch stops at %d", len(batch))
            for line in summarise_rejections(result.rejections):
                log.warning("%s", line)
            break

        entry = dict(result.summary)
        entry["image_path"] = str(result.image_path)
        batch.append(entry)
        # In-memory only: this plate is a candidate, not a publication.
        history.record(
            {
                "page_id": entry["page_id"],
                "item_id": entry["item_id"],
                "title_id": entry["title_id"],
            }
        )
    return batch


def publish_batch(cfg: Config, batch: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Render and upload an approved batch, recording each publication.

    The frame is re-derived from the page rather than carried through from the
    build, so a rejected plate costs nothing to have considered and an approved
    one costs one extra download.
    """
    history = History(cfg.history_path)
    session = requests.Session()
    published: list[dict[str, Any]] = []

    creds = youtube.build_credentials(
        require_env("YOUTUBE_CLIENT_ID"),
        require_env("YOUTUBE_CLIENT_SECRET"),
        require_env("YOUTUBE_REFRESH_TOKEN"),
    )

    for entry in batch:
        cand = bhl.PageCandidate(**{
            k: v for k, v in entry["candidate"].items()
            if k in bhl.PageCandidate.__dataclass_fields__
        })
        img = imaging.load_image(bhl.download_page_image(cand, session=session))
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

        out_dir = cfg.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        video_path = out_dir / f"{stamp}-{cand.page_id}.mp4"
        framed.image.save(out_dir / f"{stamp}-{cand.page_id}.png", format="PNG")
        video.render_still(
            framed.image,
            video_path,
            duration_seconds=cfg.video.duration_seconds,
            fps=cfg.video.fps,
            crf=cfg.video.crf,
        )

        upload = youtube.upload_video(
            video_path,
            title=entry["title"],
            description=entry["description"],
            tags=cfg.upload.tags,
            category_id=cfg.upload.category_id,
            privacy_status=cfg.upload.privacy_status,
            publish_at=youtube.scheduled_publish_time(cfg.upload.publish_delay_hours),
            made_for_kids=cfg.upload.made_for_kids,
            credentials=creds,
        )
        history.record(
            {
                "page_id": cand.page_id,
                "item_id": cand.item_id,
                "title_id": cand.title_id,
                "title": entry["title"],
                "video_id": upload.video_id,
                "publish_at": upload.publish_at,
            }
        )
        # Saved after every upload, not at the end: a run that dies partway
        # must not leave live videos unrecorded, or the next run republishes
        # them. The seeding batch hit exactly this.
        history.save()
        published.append({**entry, "video_id": upload.video_id, "url": upload.url})
        log.info("uploaded %s, publishes %s", upload.video_id, upload.publish_at)

    return published


def record_rejections(cfg: Config, rejected: Sequence[dict[str, Any]]) -> None:
    """Mark rejected plates so they are never offered again.

    Written into the same history the selector already consults, with a flag
    rather than a video id -- the page and volume dedupe both key off ids that
    are present either way, so a rejection retires a plate exactly as a
    publication does.
    """
    history = History(cfg.history_path)
    for entry in rejected:
        history.record(
            {
                "page_id": entry["page_id"],
                "item_id": entry["item_id"],
                "title_id": entry["title_id"],
                "title": entry.get("title", ""),
                "rejected": True,
            }
        )
    history.save()
