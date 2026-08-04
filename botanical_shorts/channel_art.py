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


def collect_plates(
    cfg: Config,
    *,
    count: int,
    session=None,
    max_aspect: float = 1.6,
) -> list[Plate]:
    """Walk BHL for ``count`` distinct licence-passed plates.

    Deliberately does *not* consult the published history: channel art is not
    an episode, and reusing a plate here neither burns it nor repeats it in the
    feed. It also skips the vision call -- these get looked at by a human
    before they go anywhere near the channel, which is a stronger gate than
    scan_quality, and a banner needs a dozen plates where an episode needs one.

    ``max_aspect`` is looser than the video pipeline's, because a row of
    varied plate shapes is the point here rather than a defect.
    """
    import requests

    session = session or requests.Session()
    client = bhl.BHLClient(require_env("BHL_API_KEY"), session=session)

    plates: list[Plate] = []
    seen_items: set[str] = set()

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
        # One plate per volume: a row of six plates from the same book looks
        # like a mistake rather than a collection.
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

        seen_items.add(candidate.item_id)
        plates.append(Plate(image=img, candidate=candidate))
        log.info("plate %d/%d: page %s (%s)", len(plates), count, candidate.page_id, verdict.reason)

    if not plates:
        raise RuntimeError("no licence-passed plates found; nothing to build channel art from")
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


def best_avatar_plate(plates: Sequence[Plate]) -> Plate:
    """Pick the plate whose densest crop carries the most contrast.

    Standard deviation stands in for "visual interest": a crop of flat paper or
    of a uniform wash scores low, while one holding engraved line work against
    paper scores high. At 800px shown as a circle, contrast is what survives.
    """

    def score(plate: Plate) -> float:
        x, y, side = densest_square(plate.image, min_side_px=AVATAR_SIZE[0])
        crop = plate.image.crop((x, y, x + side, y + side)).convert("L")
        return ImageStat.Stat(crop).stddev[0]

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


def build_banner(
    plates: Sequence[Plate],
    *,
    size: tuple[int, int] = BANNER_SIZE,
    safe_area: tuple[int, int] = SAFE_AREA,
    margin_ratio: float = 0.08,
    gap_ratio: float = 0.035,
    border_px: int = 0,
    border_color: str = "#00000026",
    fill: tuple[int, int, int] | None = None,
) -> Image.Image:
    """Tile plates into a banner, keeping every plate inside the safe band.

    Plate height is set by the *safe area*, not the canvas: 423px is the only
    vertical space guaranteed to survive YouTube's cropping, so a plate taller
    than that would be decapitated on a phone. The row then extends past the
    canvas horizontally, which is free -- horizontal overflow is cropped
    towards the centre, so the plates that survive are the ones already placed
    centrally.
    """
    width, height = size
    safe_w, safe_h = safe_area
    fill = fill or average_paper_color(plates)

    canvas = Image.new("RGB", (width, height), fill)

    plate_h = max(1, int(safe_h * (1 - 2 * margin_ratio)))
    gap = int(safe_h * gap_ratio)
    centre_y = height // 2

    # Scale every plate to a common height so the row reads as one shelf;
    # widths vary with each plate's own proportions, which is the variety.
    scaled: list[Image.Image] = []
    for plate in plates:
        img = plate.image
        scale = plate_h / img.height
        w = max(1, int(round(img.width * scale)))
        scaled.append(img.resize((w, plate_h), Image.LANCZOS))

    # Repeat the set until the row overruns the canvas, so the TV crop is full
    # edge to edge. With enough distinct plates this never actually repeats
    # inside the visible band.
    row: list[Image.Image] = []
    row_w = 0
    i = 0
    while row_w < width + 2 * plate_h:
        img = scaled[i % len(scaled)]
        row.append(img)
        row_w += img.width + gap
        i += 1
        if i > 200:  # pathological guard; never reached with real plate sizes
            break
    row_w -= gap

    x = (width - row_w) // 2
    placed_in_safe = 0
    safe_x0, safe_x1 = (width - safe_w) // 2, (width + safe_w) // 2

    for img in row:
        y = centre_y - plate_h // 2
        canvas.paste(img, (x, y))
        if x >= safe_x0 and x + img.width <= safe_x1:
            placed_in_safe += 1
        if border_px > 0:
            overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            ImageDraw.Draw(overlay).rectangle(
                [x, y, x + img.width - 1, y + plate_h - 1],
                outline=ImageColor.getrgb(border_color),
                width=border_px,
            )
            canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")
        x += img.width + gap

    log.info(
        "banner: %d plates in the row, %d fully inside the %dx%d safe area",
        len(row), placed_in_safe, safe_w, safe_h,
    )
    return _fade_edges(canvas, fill, span=int(width * 0.06))


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
