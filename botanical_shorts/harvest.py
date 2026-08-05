"""Harvest: walk the pool once, judge everything, keep what survives.

This replaces the first-match search the pipeline was built around. That search
was right for publishing one plate a day -- walk lazily, stop at the first plate
that clears every gate, spend a handful of API calls. Used to build a batch of
fifteen it had to be run fifteen times, and almost every mechanism added over
the last day existed only to make that repetition survivable: a blocked-page
cache so the walks stopped re-downloading each other's duds, an offset rotation
so they did not re-tread the same head of the list, a per-walk vision budget
whose exhaustion could end a batch mid-flight.

None of that is needed here. One walk, no early stop, and the result is a *set*
rather than whatever the walk happened to trip over first. Selection and
ordering then become decisions taken over that set, where they can be read and
argued with, instead of exclusion rules smuggled into the walk one plate at a
time.

The cost model inverts too. The old shape spent vision calls on whatever the
walk encountered while hunting; this spends one per plate that survived the
cheap local gates, and does it once per refill rather than once per batch.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Iterable

import requests
from PIL import Image

from . import bhl, imaging, licensing, metadata, vision
from .config import Config, require_env
from .history import History
from .pipeline import MAX_TITLE_STRIKES, MAX_VISION_ERRORS, Rejection, check_image_gates

log = logging.getLogger(__name__)


# How many bits of a 256-bit perceptual hash two plates may differ by and still
# count as the same illustration. Rescans of one engraving land in the low
# teens; genuinely different plates from the same book, drawn in the same style
# by the same hand, sit far above it. Deliberately tight -- a false match hides
# a good plate, and Brian can still catch a near-repeat by eye, which is the
# failure direction that already works.
DUPLICATE_DISTANCE = 18


@dataclass
class HarvestResult:
    entries: list[dict[str, Any]] = field(default_factory=list)
    images: list[Image.Image] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)
    vision_calls: int = 0


def prepare_plate(
    img: Image.Image,
    verdict: vision.VisionVerdict | None,
    cfg: Config,
) -> Image.Image:
    """Apply the one transform a verdict can call for: cutting a spread.

    Shared by the harvest and the publisher so a plate is framed from exactly
    the same pixels in both. The publisher re-derives the image from the page
    rather than carrying it through, which only works if this decision is
    reproducible from the stored verdict.
    """
    if verdict and verdict.is_spread and verdict.illustration_side in {"left", "right"}:
        return imaging.split_spread(img, verdict.illustration_side)
    return img


def harvest(
    cfg: Config,
    *,
    limit: int,
    session: requests.Session | None = None,
    vision_client: Any = None,
    known_pages: Iterable[str] = (),
    subjects: Iterable[str] | None = None,
    category: str = "",
) -> HarvestResult:
    """Walk ``limit`` candidates, judging every one of them.

    ``known_pages`` are pages already in the queue or already published --
    filtered inside the walk so they never consume the candidate budget.

    ``subjects`` narrows the walk to some of the configured headings, and
    ``category`` is stamped on everything kept. Harvesting a category at a time
    is what stops Ornithology's 6,730 titles from crowding out Mycology's 640:
    a single walk over the flattened heading list would spend its whole budget
    on whichever category came first.
    """
    session = session or requests.Session()
    client = bhl.BHLClient(require_env("BHL_API_KEY"), session=session)
    history = History(cfg.history_path)

    if cfg.vision.enabled and cfg.vision.max_vision_calls > 0 and vision_client is None:
        from .pipeline import _anthropic_client

        vision_client = _anthropic_client()

    result = HarvestResult()
    vision_errors = 0
    title_strikes: dict[str, int] = {}

    candidates = bhl.iter_candidates(
        client,
        subjects=list(subjects) if subjects is not None else cfg.source.subjects,
        page_types=cfg.source.page_types,
        year_min=cfg.source.year_min,
        year_max=cfg.source.year_max,
        titles_per_subject=cfg.source.titles_per_subject,
        max_items_per_title=cfg.source.max_items_per_title,
        max_pages_per_item=cfg.source.max_pages_per_item,
        limit=limit,
        # Pages retire permanently once published or rejected. Volumes and
        # works do not: a volume of a plate book holds dozens of different
        # species, and refusing the rest of it after taking one was the ceiling
        # that held every batch to four. Spacing similar plates apart is the
        # job of the publish order, not of the walk.
        skip_pages=set(history.page_ids) | {str(p) for p in known_pages},
    )

    for candidate in candidates:
        page_id = candidate.page_id

        # A volume shot against a black backdrop fails every plate in it for
        # the same reason, and scan condition is a property of the digitisation
        # rather than the page. Worth keeping here: it is the difference
        # between four downloads and eighteen.
        if title_strikes.get(candidate.title_id, 0) >= MAX_TITLE_STRIKES:
            result.rejections.append(
                Rejection(page_id, "title", f"work {candidate.title_id} keeps failing")
            )
            continue

        verdict = licensing.evaluate(candidate, cfg.license)
        if not verdict.allowed:
            result.rejections.append(Rejection(page_id, "licence", verdict.reason))
            continue

        try:
            img = imaging.load_image(bhl.download_page_image(candidate, session=session))
            check_image_gates(img, cfg)
        except (bhl.BHLError, imaging.ImageError, requests.RequestException) as exc:
            result.rejections.append(Rejection(page_id, "download", str(exc)))
            title_strikes[candidate.title_id] = title_strikes.get(candidate.title_id, 0) + 1
            continue

        seen: vision.VisionVerdict | None = None
        if cfg.vision.enabled:
            if result.vision_calls >= cfg.vision.max_vision_calls:
                log.info("vision budget spent after %d calls; stopping the harvest", result.vision_calls)
                break
            seen = vision.inspect_plate(vision_client, img, model=cfg.vision.model)
            if seen.error:
                if seen.error_is_transport:
                    vision_errors += 1
                result.rejections.append(Rejection(page_id, "vision", seen.error))
                if vision_errors >= MAX_VISION_ERRORS:
                    log.error("%d transport failures; the API looks down", vision_errors)
                    break
                continue
            result.vision_calls += 1
            ok, reason = vision.passes(
                seen,
                min_quality=cfg.vision.min_scan_quality,
                caption_mode=cfg.vision.caption_mode,
                allow_spread=seen.illustration_side in {"left", "right"},
            )
            if not ok:
                result.rejections.append(Rejection(page_id, "vision", reason))
                title_strikes[candidate.title_id] = title_strikes.get(candidate.title_id, 0) + 1
                continue

        try:
            plate = prepare_plate(img, seen, cfg)
            if plate is not img:
                check_image_gates(plate, cfg)
            framed = imaging.frame_vertical(
                plate,
                width=cfg.image.width,
                height=cfg.image.height,
                margin_ratio=cfg.image.margin_ratio,
                letterbox=cfg.image.letterbox,
                fixed_fill_color=cfg.image.fixed_fill_color,
                border_px=cfg.image.border_px,
                border_color=cfg.image.border_color,
            )
        except imaging.ImageError as exc:
            result.rejections.append(Rejection(page_id, "framing", str(exc)))
            continue

        from .pipeline import _candidate_fields

        entry: dict[str, Any] = {
            "page_id": page_id,
            "item_id": candidate.item_id,
            "title_id": candidate.title_id,
            "title": metadata.build_title(candidate, seen),
            "description": metadata.build_description(candidate, seen),
            "citation": candidate.citation(),
            "page_url": candidate.page_url,
            "candidate": _candidate_fields(candidate),
            # The heading the walk found it under, and the category that
            # heading belongs to. The category is what a viewer sees and what
            # names a playlist; the heading is a cataloguing artefact.
            "subject": candidate.subject,
            "category": category or cfg.source.category_of(candidate.subject),
            "status": "pending",
        }
        if seen:
            entry.update(
                {
                    "scan_quality": seen.scan_quality,
                    "subject_summary": seen.subject_summary,
                    "is_spread": seen.is_spread,
                    "illustration_side": seen.illustration_side,
                }
            )
        # Looked at last, because it needs the framed plate. Comparing against
        # everything kept so far catches the case Brian found by eye: the same
        # engraving reissued across volumes, differing only in scanning or
        # printing hue, which page and volume ids cannot see and which is
        # genuinely hard to spot on a page of a hundred thumbnails.
        phash = imaging.perceptual_hash(framed.image)
        twin = next(
            (
                kept
                for kept in result.entries
                if imaging.hash_distance(phash, kept["phash"]) <= DUPLICATE_DISTANCE
            ),
            None,
        )
        if twin is not None:
            result.rejections.append(
                Rejection(
                    page_id,
                    "duplicate",
                    f"same illustration as page {twin['page_id']} "
                    f"({imaging.hash_distance(phash, twin['phash'])} bits apart)",
                )
            )
            continue
        entry["phash"] = phash

        result.entries.append(entry)
        result.images.append(framed.image)
        log.info("kept %s: %s", page_id, entry["title"][:70])

    return result


def harvest_all(
    cfg: Config,
    *,
    per_category: int,
    known_pages: Iterable[str] = (),
) -> HarvestResult:
    """Harvest every configured category, each with its own candidate budget.

    Sequential rather than combined, because the categories are wildly uneven:
    Ornithology carries 6,730 titles against Mycology's 640, and one walk over
    the flattened heading list would spend everything on whichever came first
    and report the rest as empty.

    The vision budget is shared and drains as the sweep runs, so a later
    category can find it spent. That is why the budget is generous and the
    per-category limit modest: the plates a run never reached are still in the
    pool, and the next refill starts where this one gave up.
    """
    import requests as _requests

    combined = HarvestResult()
    session = _requests.Session()
    vision_client = None
    if cfg.vision.enabled and cfg.vision.max_vision_calls > 0:
        from .pipeline import _anthropic_client

        vision_client = _anthropic_client()

    seen = set(str(p) for p in known_pages)

    for name, headings in cfg.source.categories.items():
        remaining = cfg.vision.max_vision_calls - combined.vision_calls
        if cfg.vision.enabled and remaining <= 0:
            log.warning("vision budget spent; %s and later categories skipped", name)
            break

        log.info("=== harvesting %s (%d headings) ===", name, len(headings))
        import dataclasses as _dc

        scoped = _dc.replace(
            cfg, vision=_dc.replace(cfg.vision, max_vision_calls=max(0, remaining))
        )
        result = harvest(
            scoped,
            limit=per_category,
            session=session,
            vision_client=vision_client,
            known_pages=seen,
            subjects=headings,
            category=name,
        )
        combined.entries.extend(result.entries)
        combined.images.extend(result.images)
        combined.rejections.extend(result.rejections)
        combined.vision_calls += result.vision_calls
        seen.update(str(e["page_id"]) for e in result.entries)
        log.info("%s: kept %d, %d vision calls", name, len(result.entries), result.vision_calls)

    return combined


def _spread(piles: list[list[dict[str, Any]]], key) -> list[dict[str, Any]]:
    """Deal from the largest pile that is not the one just used.

    Round-robin instead looks fine until the small piles run out and the
    dominant pile's remainder lands in one block at the end. Taking the largest
    first keeps the piles even, so the heaviest is spread as thinly as its
    share allows. When only one pile is left it must repeat, which is the
    pool-size effect rather than a fault.
    """
    piles = [p for p in piles if p]
    out: list[dict[str, Any]] = []
    last = None
    while piles:
        piles.sort(key=len, reverse=True)
        pick = next((p for p in piles if key(p[0]) != last), piles[0])
        entry = pick.pop(0)
        last = key(entry)
        out.append(entry)
        if not pick:
            piles.remove(pick)
    return out


def publish_order(entries: list[dict[str, Any]], *, seed: int | None = None) -> list[dict[str, Any]]:
    """Order approved plates so consecutive uploads are rarely from one work.

    Brian's requirement, in his words: no long string of very similar images in
    a row, some randomness is good, and similarity that comes from the pool
    simply being small is fine.

    So: group by work, shuffle within and between, then deal one from each work
    in rotation. Works with the most plates naturally recur soonest, which is
    the best spacing available -- when one work supplies most of the queue its
    plates must repeat, and that is the pool-size effect rather than a fault.
    Shuffling first means two runs over the same queue give different orders.
    """
    rng = random.Random(seed)

    def work_of(e):
        return str(e.get("title_id") or "")

    def cat_of(e):
        return str(e.get("category") or "")

    # Two passes over the same dealing rule. Within a category, space the works
    # apart; then across categories, space the categories apart while keeping
    # each category's internal order. Doing it in one pass on a combined key
    # would let five birds run together as long as they came from five
    # different books, which is the thing that would actually read as
    # repetitive.
    by_category: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        by_category.setdefault(cat_of(entry), []).append(entry)

    category_piles: list[list[dict[str, Any]]] = []
    for items in by_category.values():
        by_work: dict[str, list[dict[str, Any]]] = {}
        for entry in items:
            by_work.setdefault(work_of(entry), []).append(entry)
        work_piles = list(by_work.values())
        for pile in work_piles:
            rng.shuffle(pile)
        rng.shuffle(work_piles)
        category_piles.append(_spread(work_piles, work_of))

    rng.shuffle(category_piles)
    return _spread(category_piles, cat_of)


def publish_from_queue(
    cfg: Config,
    entries: list[dict[str, Any]],
    on_published=None,
) -> list[dict[str, Any]]:
    """Upload approved plates. No judgement happens here -- that already did.

    ``on_published`` is called after each upload is safely recorded, and is the
    only reliable way for a caller to learn what went live. Reading the return
    value is not: this run stopped at YouTube's daily cap after nine uploads,
    the exception propagated, and the caller's ``finally`` block saw an empty
    list because the assignment had never happened. Nine live videos were left
    marked approved, one rerun away from being published twice.

    The frame is re-derived from the page rather than carried through from the
    harvest: the same page and the same stored verdict always yield the same
    frame, so re-deriving costs one download and carrying it would cost a
    megabyte of PNG per plate in the repository.
    """
    from . import video, youtube

    session = requests.Session()
    history = History(cfg.history_path)
    published: list[dict[str, Any]] = []

    creds = youtube.build_credentials(
        require_env("YOUTUBE_CLIENT_ID"),
        require_env("YOUTUBE_CLIENT_SECRET"),
        require_env("YOUTUBE_REFRESH_TOKEN"),
    )

    for entry in entries:
        cand = bhl.PageCandidate(**{
            k: v for k, v in entry["candidate"].items()
            if k in bhl.PageCandidate.__dataclass_fields__
        })
        img = imaging.load_image(bhl.download_page_image(cand, session=session))

        # Rebuilt from the stored verdict, so the frame matches what was
        # reviewed rather than what a fresh vision call might say today.
        if entry.get("illustration_side") in {"left", "right"}:
            img = imaging.split_spread(img, entry["illustration_side"])

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
        video_path = out_dir / f"{cand.page_id}.mp4"
        framed.image.save(out_dir / f"{cand.page_id}.png", format="PNG")
        video.render_still(
            framed.image,
            video_path,
            duration_seconds=cfg.video.duration_seconds,
            fps=cfg.video.fps,
            crf=cfg.video.crf,
        )

        try:
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
        except youtube.UploadError as exc:
            # The per-channel daily cap is a normal end to a run, not a fault:
            # the remaining plates stay approved and go out tomorrow. Treating
            # it as a crash is what let nine live videos go unrecorded.
            if "uploadLimitExceeded" in str(exc) or "exceeded the number of videos" in str(exc):
                log.warning(
                    "YouTube's daily upload cap reached after %d uploads; "
                    "the rest of the queue keeps its place",
                    len(published),
                )
                break
            raise
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
        # Saved after every upload: a run that dies partway must not leave live
        # videos unrecorded, or the next run republishes them.
        history.save()
        record = {**entry, "video_id": upload.video_id, "url": upload.url}
        published.append(record)
        if on_published is not None:
            on_published(record)
        log.info("uploaded %s (%s)", upload.video_id, entry["title"][:60])

    return published


def retire(cfg: Config, entries: list[dict[str, Any]]) -> None:
    """Record rejected plates in history so they are never harvested again."""
    history = History(cfg.history_path)
    for entry in entries:
        history.record(
            {
                "page_id": entry["page_id"],
                "item_id": entry.get("item_id", ""),
                "title_id": entry.get("title_id", ""),
                "title": entry.get("title", ""),
                "rejected": True,
            }
        )
    history.save()
