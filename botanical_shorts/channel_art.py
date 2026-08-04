"""Channel banner and profile picture, composited from real licence-passed plates.

Same rule as the video pipeline: **no text is ever composited**. The channel
name lives in YouTube's own chrome, which sits over the banner anyway; adding
modern typography to engraved plates is the one thing this channel does not do.

Two very different framing problems, which is why they do not share the video
pipeline's :func:`imaging.frame_vertical`:

**Banner** (2560x1440, safe area 1546x423 centred)
    YouTube crops this differently on every surface: phones see roughly the
    safe area alone, desktop a wider band, TV the whole 2560x1440. So the art
    is built as one continuous row of plates centred on the safe band --
    readable when cropped to the middle, and still a complete-looking sheet
    when shown in full. Plates that overflow the canvas edge are faded into
    the paper so the crop reads as deliberate rather than sliced.

**Avatar** (800x800, displayed as a circle)
    A whole plate scaled into 800px is a speck surrounded by margin, and the
    circular mask eats the corners where a plate's caption usually sits. So
    this crops *into* one plate, choosing the window with the densest
    engraving via :func:`densest_square` -- tight on the drawing, no whitespace.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterator, Sequence

from PIL import Image, ImageColor, ImageDraw, ImageStat

from . import bhl, imaging, licensing
from .config import Config, require_env

log = logging.getLogger(__name__)

BANNER_SIZE = (2560, 1440)
# YouTube's "safe area": the only region guaranteed visible on every device.
SAFE_AREA = (1546, 423)
AVATAR_SIZE = (800, 800)


@dataclass
class Plate:
    """A licence-cleared plate image plus the credit needed to attribute it."""

    image: Image.Image = field(repr=False)
    candidate: bhl.PageCandidate

    @property
    def citation(self) -> str:
        return self.candidate.citation()


# Below this mean border luminance, the scan's own edge is dark -- a black
# scan frame, a dark mount board, or a photograph shot against cloth. Such a
# plate cannot be made to sit on a sheet of paper: whatever tone fills the
# margins, the plate reads as a rectangle pasted on top. The video pipeline
# never had to care, because there the plate fills the frame.
MIN_BORDER_LUMINANCE = 140


def blends_with_paper(img: Image.Image) -> tuple[bool, str]:
    """Whether this scan can sit on a paper field without looking pasted on."""
    w, h = img.size
    bw, bh = max(1, int(w * 0.04)), max(1, int(h * 0.04))
    strips = [
        img.crop((0, 0, w, bh)),
        img.crop((0, h - bh, w, h)),
        img.crop((0, 0, bw, h)),
        img.crop((w - bw, 0, w, h)),
    ]
    # The darkest edge decides: one black scan frame is enough to spoil it,
    # even if the other three edges are clean paper.
    darkest = min(sum(ImageStat.Stat(s).median[:3]) / 3 for s in strips)
    if darkest < MIN_BORDER_LUMINANCE:
        return False, f"dark border (luminance {darkest:.0f} < {MIN_BORDER_LUMINANCE})"
    return True, "blends"


# A plate carrying less ink than this over its surface is a faint pencil study
# or a barely-inked outline. At full height in a video it is a delicate
# drawing; shrunk into a collage tile it is an empty white rectangle. Measured
# as the fraction of the plate at least MIN_INK_DEPTH below its own paper tone.
MIN_INK_COVERAGE = 0.05
MIN_INK_DEPTH = 40


def ink_coverage(img: Image.Image) -> float:
    """Fraction of the plate carrying meaningful ink, against its own paper."""
    cells, gw, gh, _ = _ink_grid(img)
    inked = sum(1 for row in cells for v in row if v >= MIN_INK_DEPTH)
    return inked / max(1, gw * gh)


ART_PROMPT = """You are choosing plates for the channel art of a YouTube channel \
that posts historical botanical illustrations. This is a scanned page from a \
natural history book.

Answer whether this page is suitable as one tile in a collage of botanical plates.

Suitable: a drawn, engraved or lithographed illustration whose subject is a PLANT \
-- flowers, foliage, fruit, seeds, roots, fungi, or a botanical dissection.

NOT suitable, answer false for any of these:
- a title page, index, contents page, dedication, or any page that is mostly text
- a photograph rather than a drawing, engraving or lithograph
- a fanciful or allegorical scene: human or animal figures, fairies, costumed \
characters, personified flowers, landscapes with people
- a map, diagram, portrait, or an illustration of an animal rather than a plant
- anything showing scanning furniture: a ruler, a measuring scale, a colour \
calibration bar or chart, a library stamp, or a colour/greyscale target
- a sketchbook or notebook page: rough pencil studies, handwritten notes or \
annotations in the margin rather than a finished plate
- a plate so faint, damaged, stained or skewed that it would look poor at small size

Respond with ONLY a JSON object: {"suitable": true/false, "reason": "<8 words>"}"""


def _art_client():
    from anthropic import Anthropic

    return Anthropic(api_key=require_env("ANTHROPIC_API_KEY"))


def suitable_for_art(client, img: Image.Image, *, model: str) -> tuple[bool, str]:
    """Ask Claude whether this plate belongs in the collage.

    A separate question from the video pipeline's, and deliberately stricter.
    An episode shows one plate at full height, where a title page or a
    figurative print would simply be the wrong video; here a dozen plates sit
    side by side, and a page of letterpress among them does not read as an odd
    choice, it reads as a bug.
    """
    import json

    from .vision import _encode, _parse

    encoded, media_type = _encode(img)
    try:
        message = client.messages.create(
            model=model,
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": encoded,
                            },
                        },
                        {"type": "text", "text": ART_PROMPT},
                    ],
                }
            ],
        )
        raw = "".join(b.text for b in message.content if b.type == "text")
        data = _parse(raw)
    except Exception as exc:
        if getattr(exc, "status_code", None) in (401, 403):
            from .vision import VisionAuthError

            raise VisionAuthError(
                "Anthropic rejected the API key; no amount of retrying will help."
            ) from exc
        log.warning("art suitability check failed: %s", exc)
        # Fail closed: an unchecked plate is not worth the risk of a title
        # page on the channel banner.
        return False, f"check errored: {exc}"

    del json
    return bool(data.get("suitable")), str(data.get("reason") or "").strip()


def collect_plates(
    cfg: Config,
    *,
    count: int,
    session=None,
    vision_client=None,
    max_aspect: float = 1.6,
) -> list[Plate]:
    """Walk BHL for ``count`` distinct licence-passed plates fit for a collage.

    Deliberately does *not* consult the published history: channel art is not
    an episode, and reusing a plate here neither burns it nor repeats it in the
    feed.

    ``max_aspect`` is looser than the video pipeline's, because a row of varied
    plate shapes is the point here rather than a defect.
    """
    import requests

    session = session or requests.Session()
    client = bhl.BHLClient(require_env("BHL_API_KEY"), session=session)
    if cfg.vision.enabled and vision_client is None:
        vision_client = _art_client()

    plates: list[Plate] = []
    seen_items: set[str] = set()
    # Channel art is built rarely, so this is generous -- but unbounded vision
    # calls against a pool of 400 candidates is not a bill worth risking for a
    # banner.
    checks_left = count * 4

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
    )

    for candidate in candidates:
        if len(plates) >= count:
            break
        # One plate per volume: six plates from the same book looks like a
        # mistake rather than a collection.
        if candidate.item_id in seen_items:
            continue

        verdict = licensing.evaluate(candidate, cfg.license)
        if not verdict.allowed:
            log.debug("page %s rejected at licence: %s", candidate.page_id, verdict.reason)
            continue

        try:
            data = bhl.download_page_image(candidate, session=session)
            img = imaging.load_image(data)
            imaging.check_source_resolution(
                img, cfg.image.min_source_width, cfg.image.min_source_height
            )
            imaging.check_aspect(img, max_aspect)
        except Exception as exc:  # any unusable scan is just the next candidate
            log.debug("page %s unusable: %s", candidate.page_id, exc)
            continue

        # Cheap and local, so these run before spending a vision call.
        blends, why = blends_with_paper(img)
        if not blends:
            log.debug("page %s rejected: %s", candidate.page_id, why)
            continue

        coverage = ink_coverage(img)
        if coverage < MIN_INK_COVERAGE:
            log.debug(
                "page %s rejected: only %.1f%% ink, reads as blank at tile size",
                candidate.page_id, coverage * 100,
            )
            continue

        if vision_client is not None:
            if checks_left <= 0:
                log.warning("suitability budget exhausted after %d plates", len(plates))
                break
            checks_left -= 1
            ok, reason = suitable_for_art(vision_client, img, model=cfg.vision.model)
            if not ok:
                log.info("page %s not collage material: %s", candidate.page_id, reason)
                continue

        seen_items.add(candidate.item_id)
        plates.append(Plate(image=img, candidate=candidate))
        log.info("plate %d/%d: page %s (%s)", len(plates), count, candidate.page_id, verdict.reason)

    if not plates:
        raise RuntimeError("no suitable plates found; nothing to build channel art from")
    return plates


def average_paper_color(plates: Sequence[Plate]) -> tuple[int, int, int]:
    """The tone the whole sheet is built on: median of each plate's own paper.

    Median rather than mean so one heavily browned or one bleached scan does
    not drag the background away from where most of the plates actually sit.
    """
    samples = [imaging.sample_paper_color(p.image) for p in plates]
    channels = [sorted(s[i] for s in samples) for i in range(3)]

    def mid(values: list[int]) -> int:
        n = len(values)
        return values[n // 2] if n % 2 else round((values[n // 2 - 1] + values[n // 2]) / 2)

    return (mid(channels[0]), mid(channels[1]), mid(channels[2]))


# --------------------------------------------------------------------------
# Avatar
# --------------------------------------------------------------------------


def _ink_grid(img: Image.Image, grid: int = 160) -> tuple[list[list[float]], int, int, float]:
    """Downscale to a coarse map of how much *ink* each cell holds.

    Ink is measured against the plate's own paper tone rather than against
    white: an 1820s scan can have a paper baseline down in the 180s, and
    measuring darkness absolutely would score the blank margin of a browned
    plate the same as light engraving on a clean one.
    """
    paper = imaging.sample_paper_color(img)
    paper_lum = sum(paper) / 3

    w, h = img.size
    scale = grid / max(w, h)
    gw, gh = max(1, round(w * scale)), max(1, round(h * scale))
    small = img.convert("L").resize((gw, gh), Image.BILINEAR)
    px = small.load()

    cells = [[max(0.0, paper_lum - px[x, y]) for x in range(gw)] for y in range(gh)]
    return cells, gw, gh, w / gw


def _integral(cells: list[list[float]], gw: int, gh: int) -> list[list[float]]:
    """Summed-area table, so any window's ink total is four lookups."""
    table = [[0.0] * (gw + 1) for _ in range(gh + 1)]
    for y in range(gh):
        row_sum = 0.0
        for x in range(gw):
            row_sum += cells[y][x]
            table[y + 1][x + 1] = table[y][x + 1] + row_sum
    return table


def _window_mean(table: list[list[float]], x0: int, y0: int, x1: int, y1: int) -> float:
    area = (x1 - x0) * (y1 - y0)
    if area <= 0:
        return 0.0
    total = table[y1][x1] - table[y0][x1] - table[y1][x0] + table[y0][x0]
    return total / area


def densest_square(
    img: Image.Image,
    *,
    min_side_px: int = 800,
    ratios: Sequence[float] = (0.45, 0.6, 0.75, 0.9),
) -> tuple[int, int, int]:
    """Find the square window holding the most engraving. Returns (x, y, side).

    Scored as a blend of the whole window and its central 60%. The centre term
    matters because the result is displayed as a circle: a square whose ink all
    sits in one corner scores well on raw density and then loses that corner to
    the mask. Weighting the middle picks windows whose subject is *in* the
    visible disc.
    """
    cells, gw, gh, cell_px = _ink_grid(img)
    table = _integral(cells, gw, gh)

    short_px = min(img.width, img.height)
    best: tuple[float, int, int, int] | None = None

    for ratio in ratios:
        side_px = int(short_px * ratio)
        # Never crop below the output resolution if the scan can do better;
        # upscaling a 400px crop to 800 undoes the point of a tight crop.
        if side_px < min_side_px and short_px >= min_side_px:
            continue
        side = max(2, int(round(side_px / cell_px)))
        if side > gw or side > gh:
            continue

        stride = max(1, side // 8)
        inset = max(1, int(side * 0.2))  # central 60%

        for y in range(0, gh - side + 1, stride):
            for x in range(0, gw - side + 1, stride):
                whole = _window_mean(table, x, y, x + side, y + side)
                centre = _window_mean(
                    table, x + inset, y + inset, x + side - inset, y + side - inset
                )
                score = 0.6 * whole + 0.4 * centre
                if best is None or score > best[0]:
                    best = (score, x, y, side)

    if best is None:
        # Every ratio was ruled out by min_side_px: fall back to the largest
        # centred square the scan can give.
        side = short_px
        return ((img.width - side) // 2, (img.height - side) // 2, side)

    _, gx, gy, gside = best
    x = int(round(gx * cell_px))
    y = int(round(gy * cell_px))
    side = int(round(gside * cell_px))
    # Clamp back inside the scan; rounding can push the window a pixel over.
    side = min(side, img.width, img.height)
    x = max(0, min(x, img.width - side))
    y = max(0, min(y, img.height - side))
    return x, y, side


def build_avatar(
    plate: Plate,
    *,
    size: tuple[int, int] = AVATAR_SIZE,
    border_px: int = 0,
    border_color: str = "#00000026",
) -> Image.Image:
    """Crop into the plate's densest detail and square it off for the circle."""
    img = plate.image
    x, y, side = densest_square(img, min_side_px=size[0])
    crop = img.crop((x, y, x + side, y + side)).resize(size, Image.LANCZOS)
    log.info(
        "avatar: cropped %dx%d at (%d,%d) from a %dx%d scan of page %s",
        side, side, x, y, img.width, img.height, plate.candidate.page_id,
    )

    if border_px > 0:
        # Sits just inside the circle's edge, so it reads as a rim rather than
        # a square frame with clipped corners.
        overlay = Image.new("RGBA", crop.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.ellipse(
            [0, 0, size[0] - 1, size[1] - 1],
            outline=ImageColor.getrgb(border_color),
            width=border_px,
        )
        crop = Image.alpha_composite(crop.convert("RGBA"), overlay).convert("RGB")
    return crop


def colorfulness(img: Image.Image) -> float:
    """Mean separation between the colour channels.

    Near zero for anything neutral -- pencil, plain engraving, and, crucially,
    the black-and-white measuring scales and greyscale targets that scanning
    operators lay beside a page.
    """
    from PIL import ImageChops

    r, g, b = img.convert("RGB").split()
    pairs = (
        ImageChops.difference(r, g),
        ImageChops.difference(g, b),
        ImageChops.difference(r, b),
    )
    return sum(ImageStat.Stat(d).mean[0] for d in pairs) / 3


def best_avatar_plate(plates: Sequence[Plate]) -> Plate:
    """Pick the plate whose densest crop makes the strongest 800px circle.

    Contrast alone is the wrong measure, and picking it cost a build: a
    scanner's black-and-white ruler has more standard deviation than any
    drawing on the page, so the avatar came out as a measuring scale and a
    line of handwriting. Weighting by colour separation as well pushes the
    choice towards chromolithographs -- which is what actually reads at the
    size a profile picture is seen.
    """

    def score(plate: Plate) -> float:
        x, y, side = densest_square(plate.image, min_side_px=AVATAR_SIZE[0])
        crop = plate.image.crop((x, y, x + side, y + side))
        contrast = ImageStat.Stat(crop.convert("L")).stddev[0]
        # The constant keeps a superb monochrome engraving in the running
        # rather than letting any colour at all beat it outright.
        return contrast * (colorfulness(crop) + 6.0)

    return max(plates, key=score)


def circular_preview(avatar: Image.Image) -> Image.Image:
    """The avatar as YouTube will actually show it -- for eyeballing, not upload.

    Upload the square; YouTube applies its own mask. This exists so the crop
    can be judged against the shape it will really appear in.
    """
    mask = Image.new("L", avatar.size, 0)
    ImageDraw.Draw(mask).ellipse([0, 0, avatar.width - 1, avatar.height - 1], fill=255)
    out = Image.new("RGBA", avatar.size, (0, 0, 0, 0))
    out.paste(avatar, (0, 0), mask)
    return out


# --------------------------------------------------------------------------
# Banner
# --------------------------------------------------------------------------


def _fade_edges(canvas: Image.Image, fill: tuple[int, int, int], span: int) -> Image.Image:
    """Fade the far left and right into the paper tone.

    The plate row is built wider than the canvas on purpose, so the TV crop
    never shows an empty end. Without this the outermost plate is simply
    guillotined by the canvas edge; with it, the row reads as continuing off
    the sheet.
    """
    if span <= 0:
        return canvas
    w, h = canvas.size
    veil = Image.new("RGBA", (w, h), fill + (0,))
    px = veil.load()
    for x in range(span):
        # Opaque at the very edge, clear by `span` in.
        alpha = int(255 * (1 - x / span) ** 1.5)
        for edge_x in (x, w - 1 - x):
            for y in range(h):
                px[edge_x, y] = fill + (alpha,)
    return Image.alpha_composite(canvas.convert("RGBA"), veil).convert("RGB")


def _scale_to_height(images: Sequence[Image.Image], height: int) -> list[Image.Image]:
    out = []
    for img in images:
        w = max(1, int(round(img.width * height / img.height)))
        out.append(img.resize((w, height), Image.LANCZOS))
    return out


def _lay_row(
    canvas: Image.Image,
    scaled: Sequence[Image.Image],
    *,
    y: int,
    gap: int,
    start_index: int = 0,
    x_offset: int = 0,
    border_px: int = 0,
    border_color: str = "#00000026",
) -> int:
    """Paste a left-to-right row that overruns both canvas edges.

    Returns how many plates landed fully inside the horizontal safe area.
    """
    width = canvas.width
    safe_x0, safe_x1 = (width - SAFE_AREA[0]) // 2, (width + SAFE_AREA[0]) // 2

    # Start far enough left that the row is already mid-plate at x=0.
    x = x_offset - max(img.width for img in scaled)
    i, in_safe = start_index, 0
    draw = ImageDraw.Draw(canvas)

    while x < width:
        img = scaled[i % len(scaled)]
        canvas.paste(img, (x, y))
        if x >= safe_x0 and x + img.width <= safe_x1:
            in_safe += 1
        if border_px > 0:
            draw.rectangle(
                [x, y, x + img.width - 1, y + img.height - 1],
                outline=ImageColor.getrgb(border_color),
                width=border_px,
            )
        x += img.width + gap
        i += 1
    return in_safe


def build_banner(
    plates: Sequence[Plate],
    *,
    size: tuple[int, int] = BANNER_SIZE,
    safe_area: tuple[int, int] = SAFE_AREA,
    margin_ratio: float = 0.07,
    gap_ratio: float = 0.035,
    border_px: int = 0,
    border_color: str = "#00000026",
    veil_alpha: int = 168,
    fill: tuple[int, int, int] | None = None,
) -> Image.Image:
    """A full-bleed collage with a crisp hero row across the safe band.

    Two layers, because the safe area and the canvas want opposite things.

    The **hero row** is sized to the safe area: 423px is the only vertical
    space guaranteed to survive YouTube's cropping, so anything meant to be
    seen must fit inside it. That alone would leave three quarters of the
    2560x1440 canvas as bare paper -- fine on a phone, a thin strip adrift in
    an empty sheet on desktop and TV.

    So beneath it sits a **background field** of larger plates tiled over the
    whole canvas and veiled back towards the paper tone. It fills the frame at
    every crop, reads as a wall of specimen sheets rather than as competing
    subject matter, and stays quiet enough for YouTube's own channel name and
    avatar, which are drawn over the middle of this image.
    """
    width, height = size
    safe_w, safe_h = safe_area
    fill = fill or average_paper_color(plates)
    sources = [p.image for p in plates]

    canvas = Image.new("RGB", (width, height), fill)

    # --- background field ---------------------------------------------------
    bg_h = int(height * 0.42)
    bg_gap = int(bg_h * 0.05)
    bg = _scale_to_height(sources, bg_h)
    row_index = 0
    y = -bg_h // 3
    while y < height:
        # Stagger each row and start on a different plate, so the tiling does
        # not line up into visible columns.
        _lay_row(
            canvas,
            bg,
            y=y,
            gap=bg_gap,
            start_index=row_index * 3 + 1,
            x_offset=int(bg_h * 0.37 * row_index),
        )
        y += bg_h + bg_gap
        row_index += 1

    veil = Image.new("RGBA", (width, height), fill + (veil_alpha,))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), veil).convert("RGB")

    # --- hero row -----------------------------------------------------------
    plate_h = max(1, int(safe_h * (1 - 2 * margin_ratio)))
    gap = int(safe_h * gap_ratio)
    hero = _scale_to_height(sources, plate_h)
    in_safe = _lay_row(
        canvas,
        hero,
        y=height // 2 - plate_h // 2,
        gap=gap,
        border_px=border_px,
        border_color=border_color,
    )

    log.info(
        "banner: %d background rows, %d hero plates fully inside the %dx%d safe area",
        row_index, in_safe, safe_w, safe_h,
    )
    return _fade_edges(canvas, fill, span=int(width * 0.05))


def safe_area_preview(banner: Image.Image, safe_area: tuple[int, int] = SAFE_AREA) -> Image.Image:
    """Exactly what a phone shows: the banner cropped to its safe area."""
    w, h = banner.size
    sw, sh = safe_area
    return banner.crop(((w - sw) // 2, (h - sh) // 2, (w + sw) // 2, (h + sh) // 2))


def attribution_block(plates: Sequence[Plate]) -> str:
    """Credit lines for the channel's About section.

    The plates are public domain, so this is not legally required -- but the
    channel's whole proposition is that these are real historical works from a
    named library, and unattributed art quietly undercuts that.
    """
    lines = ["Channel art composited from public-domain plates:", ""]
    seen: set[str] = set()
    for plate in plates:
        cite = plate.citation
        if cite in seen:
            continue
        seen.add(cite)
        lines.append(f"- {cite} ({plate.candidate.page_url})")
    lines += ["", "Digitised by the Biodiversity Heritage Library."]
    return "\n".join(lines)
