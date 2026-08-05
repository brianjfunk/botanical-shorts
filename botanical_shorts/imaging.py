"""Framing a scanned plate into a vertical 9:16 frame.

No text is ever composited here -- the plate's own engraved/lettered caption is
the only typography in the frame, which is the channel's defining constraint.
The only decisions are: how much of the frame the plate occupies, and what
fills the margins.

Letterbox treatments (config ``image.letterbox``):

``sampled_paper``
    Sample the scan's own border pixels, take a robust median, and fill the
    margins with that tone so the frame reads as one continuous sheet of aged
    paper. Adapts per plate, which matters because BHL scans range from cream
    to grey to heavily browned.
``fixed``
    One constant colour across every episode.
``black``
    Plain bars.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

from PIL import Image, ImageColor, ImageStat

log = logging.getLogger(__name__)

# Never upscale a plate by more than this; beyond it the scan turns to mush.
MAX_UPSCALE = 1.6

# How far below its own paper tone a pixel must fall to count as ink.
MIN_INK_DEPTH = 40


class ImageError(RuntimeError):
    """The source scan is unusable for framing."""


@dataclass(frozen=True)
class FramedImage:
    image: Image.Image
    fill_color: tuple[int, int, int]
    source_size: tuple[int, int]
    plate_size: tuple[int, int]
    upscaled: bool


def load_image(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:  # Pillow raises a wide variety here
        raise ImageError(f"could not decode scan: {exc}") from exc
    return img.convert("RGB")


def sample_paper_color(img: Image.Image, border_ratio: float = 0.04) -> tuple[int, int, int]:
    """Estimate the plate's paper tone from its outer border.

    Uses the median of the four edge strips rather than the mean so a dark
    engraved element or a scanner shadow intruding into one strip does not drag
    the fill colour away from the true paper tone.
    """
    w, h = img.size
    bw = max(1, int(w * border_ratio))
    bh = max(1, int(h * border_ratio))

    strips = [
        img.crop((0, 0, w, bh)),           # top
        img.crop((0, h - bh, w, h)),       # bottom
        img.crop((0, 0, bw, h)),           # left
        img.crop((w - bw, 0, w, h)),       # right
    ]

    channels: list[list[float]] = [[], [], []]
    for strip in strips:
        median = ImageStat.Stat(strip).median
        for i in range(3):
            channels[i].append(median[i])

    def mid(values: list[float]) -> int:
        values = sorted(values)
        n = len(values)
        middle = values[n // 2] if n % 2 else (values[n // 2 - 1] + values[n // 2]) / 2
        return int(round(middle))

    color = tuple(max(0, min(255, mid(c))) for c in channels)

    # A near-black sample means the scan is bordered by scanner void, not
    # paper. Filling with that would produce the stark bars we ruled out, so
    # fall back to a warm parchment.
    if sum(color) / 3 < 40:
        log.info("border sample was near-black (%s); using parchment fallback", color)
        return (232, 222, 201)
    return color  # type: ignore[return-value]


def _resolve_fill(img: Image.Image, letterbox: str, fixed_color: str) -> tuple[int, int, int]:
    if letterbox == "black":
        return (0, 0, 0)
    if letterbox == "fixed":
        return ImageColor.getrgb(fixed_color)
    return sample_paper_color(img)


def frame_vertical(
    img: Image.Image,
    *,
    width: int,
    height: int,
    margin_ratio: float,
    letterbox: str,
    fixed_fill_color: str,
    border_px: int = 0,
    border_color: str = "#00000022",
) -> FramedImage:
    """Fit the whole plate inside a ``width`` x ``height`` frame.

    The plate is always scaled to *fit*, never cropped to fill: cutting a plate
    would risk clipping the engraved caption at the frame edge, which the spec
    forbids.
    """
    if img.width < 1 or img.height < 1:
        raise ImageError("source image has zero dimension")

    source_size = img.size
    fill = _resolve_fill(img, letterbox, fixed_fill_color)

    avail_w = max(1, int(width * (1 - 2 * margin_ratio)))
    avail_h = max(1, int(height * (1 - 2 * margin_ratio)))

    scale = min(avail_w / img.width, avail_h / img.height)
    upscaled = scale > 1.0
    if scale > MAX_UPSCALE:
        scale = MAX_UPSCALE

    plate_w = max(1, int(round(img.width * scale)))
    plate_h = max(1, int(round(img.height * scale)))
    plate = img.resize((plate_w, plate_h), Image.LANCZOS)

    canvas = Image.new("RGB", (width, height), fill)
    offset = ((width - plate_w) // 2, (height - plate_h) // 2)
    canvas.paste(plate, offset)

    if border_px > 0:
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        from PIL import ImageDraw  # local import keeps the hot path light

        draw = ImageDraw.Draw(overlay)
        x0, y0 = offset
        # ImageColor handles both #rrggbb and #rrggbbaa, so a translucent
        # keyline is expressible directly in config.
        draw.rectangle(
            [x0, y0, x0 + plate_w - 1, y0 + plate_h - 1],
            outline=ImageColor.getrgb(border_color),
            width=border_px,
        )
        canvas = Image.alpha_composite(canvas.convert("RGBA"), overlay).convert("RGB")

    return FramedImage(
        image=canvas,
        fill_color=fill,
        source_size=source_size,
        plate_size=(plate_w, plate_h),
        upscaled=upscaled,
    )


def perceptual_hash(img: Image.Image, side: int = 16) -> int:
    """A hash of what the plate *looks like*, not of its bytes.

    Difference hash: shrink to a small grey grid and record, for each pixel,
    whether it is darker than the one to its right. Only those comparisons
    survive, which is what makes it useful here -- the same engraving rescanned
    or reprinted differs in hue, brightness and paper tone while its light-dark
    structure is almost unchanged, so the two hashes stay within a few bits of
    each other. Byte equality or a colour histogram would miss it entirely.

    That is precisely the case Brian found by eye and called a memory game: two
    plates of the same illustration differing only in scanning or printing hue.
    """
    grid = img.convert("L").resize((side + 1, side), Image.LANCZOS)
    px = grid.load()
    bits = 0
    for y in range(side):
        for x in range(side):
            bits = (bits << 1) | int(px[x, y] > px[x + 1, y])
    return bits


def hash_distance(a: int, b: int) -> int:
    """How many bits two perceptual hashes differ by."""
    return bin(a ^ b).count("1")


def find_gutter(img: Image.Image, search_ratio: float = 0.16) -> int:
    """Locate the fold between two facing pages, as an x coordinate.

    A bound volume photographed open has a shadow in the fold, so the gutter is
    a dark column near the middle -- but so is the engraving, and on a plate
    that runs close to the fold the engraving is darker. What separates them is
    shape, not depth: a fold is a *narrow* dark line with paper on both sides,
    while an engraving is a broad mass with paper on one side only. So each
    column is scored against the lighter of its two neighbourhoods, and a wide
    dark region scores nothing however dark it is.

    Falls back to the exact centre when no column stands out, which is right
    for a flatbed capture with no shadow: the fold is still near the middle,
    and half of a symmetric spread is the correct crop either way.
    """
    w, h = img.size
    mid = w // 2
    span = max(1, int(w * search_ratio))
    lo, hi = max(1, mid - span), min(w - 1, mid + span)

    # One row of column means, cheaply: squash the whole image to a single
    # pixel high, so a column is summarised by its average darkness.
    row = img.convert("L").resize((w, 1), Image.BILINEAR)
    px = row.load()
    columns = [px[x, 0] for x in range(w)]

    # Far enough out to clear the shadow itself, near enough to still be the
    # same page rather than the opposite margin.
    reach = max(4, int(w * 0.02))

    best_x, best_contrast = mid, 0.0
    for x in range(lo, hi):
        left = columns[max(0, x - 2 * reach) : max(1, x - reach)]
        right = columns[min(w - 1, x + reach) : min(w, x + 2 * reach)]
        if not left or not right:
            continue
        # The lighter side is the weaker evidence, so it decides: a plate edge
        # has paper on one side and ink on the other and scores near zero.
        contrast = min(sum(left) / len(left), sum(right) / len(right)) - columns[x]
        if contrast > best_contrast:
            best_x, best_contrast = x, contrast

    # A real fold is decisively darker than the paper on both sides of it.
    # Anything shallower is page texture, and the centre is the better guess.
    return best_x if best_contrast >= 12 else mid


def split_spread(img: Image.Image, side: str) -> Image.Image:
    """Return one half of a two-page capture, cut at the fold.

    This is the one place a source scan is cut, and it is not the cropping the
    spec forbids: that rule protects the plate's engraved caption from being
    clipped to fill a frame. Here the cut falls in the gutter *between* two
    pages, so the plate on the chosen side survives whole, with its own caption
    intact -- what is discarded is the facing leaf of letterpress.
    """
    if side not in {"left", "right"}:
        raise ImageError(f"cannot split a spread toward {side!r}")

    w, h = img.size
    gutter = find_gutter(img)
    # Pull in slightly past the fold so the shadow and the opposite page's
    # inner margin do not survive at the edge of the crop.
    inset = max(1, int(w * 0.01))
    box = (0, 0, max(1, gutter - inset), h) if side == "left" else (min(w - 1, gutter + inset), 0, w, h)
    half = img.crop(box)
    if half.width < 1 or half.height < 1:
        raise ImageError("splitting the spread produced an empty half")
    return half


def _ink_grid(img: Image.Image, grid: int = 160) -> tuple[list[list[float]], int, int, float]:
    """Downscale to a coarse map of how much *ink* each cell holds.

    Ink is measured against the plate's own paper tone rather than against
    white: an 1820s scan can have a paper baseline down in the 180s, and
    measuring darkness absolutely would score the blank margin of a browned
    plate the same as light engraving on a clean one.
    """
    paper = sample_paper_color(img)
    paper_lum = sum(paper) / 3

    w, h = img.size
    scale = grid / max(w, h)
    gw, gh = max(1, round(w * scale)), max(1, round(h * scale))
    small = img.convert("L").resize((gw, gh), Image.BILINEAR)
    px = small.load()

    cells = [[max(0.0, paper_lum - px[x, y]) for x in range(gw)] for y in range(gh)]
    return cells, gw, gh, w / gw


def ink_coverage(img: Image.Image) -> float:
    """Fraction of the plate carrying meaningful ink, against its own paper."""
    cells, gw, gh, _ = _ink_grid(img)
    inked = sum(1 for row in cells for v in row if v >= MIN_INK_DEPTH)
    return inked / max(1, gw * gh)


def border_luminance(img: Image.Image) -> float:
    """Mean luminance of the scan's *darkest* edge strip.

    The darkest edge decides rather than the average: one black scan frame is
    enough to spoil the effect even when the other three edges are clean paper.
    """
    w, h = img.size
    bw, bh = max(1, int(w * 0.04)), max(1, int(h * 0.04))
    strips = [
        img.crop((0, 0, w, bh)),
        img.crop((0, h - bh, w, h)),
        img.crop((0, 0, bw, h)),
        img.crop((w - bw, 0, w, h)),
    ]
    return min(sum(ImageStat.Stat(s).median[:3]) / 3 for s in strips)


def check_border_tone(img: Image.Image, min_luminance: float) -> None:
    """Reject scans whose own edge is dark rather than paper.

    A black scan frame, a dark mount board, or a photograph shot against cloth
    cannot be letterboxed onto paper. ``sample_paper_color`` samples that edge,
    reads it as scanner void, and falls back to parchment -- so the plate ends
    up as a hard dark rectangle sitting on a parchment field, which is the one
    result the letterbox decision was meant to avoid.
    """
    lum = border_luminance(img)
    if lum < min_luminance:
        raise ImageError(
            f"border luminance {lum:.0f} is below {min_luminance:.0f}: the scan "
            "has a dark frame or mount and cannot sit on a paper field"
        )


def subject_ink_coverage(img: Image.Image) -> float:
    """Ink coverage measured inside the inked region rather than across the sheet.

    Whole-sheet coverage conflates *sparse* with *faint*. A single delicate
    specimen engraved in the middle of a large sheet covers very little of it
    and reads beautifully; a pencil study covering the same fraction reads as
    blank. The difference is not how much of the paper is used but how densely
    the used part is worked, so the region is found first and measured second.

    The bounding box is taken between the 2nd and 98th percentile of inked
    cells on each axis, so a speck of foxing in a corner cannot stretch the box
    back out to the whole sheet.
    """
    cells, gw, gh, _ = _ink_grid(img)
    inked = [(x, y) for y in range(gh) for x in range(gw) if cells[y][x] >= MIN_INK_DEPTH]
    if len(inked) < 4:
        return 0.0

    def bounds(values: list[int]) -> tuple[int, int]:
        values = sorted(values)
        lo = values[int(len(values) * 0.02)]
        hi = values[min(len(values) - 1, int(len(values) * 0.98))]
        return lo, hi

    x0, x1 = bounds([x for x, _ in inked])
    y0, y1 = bounds([y for _, y in inked])
    area = max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
    within = sum(1 for x, y in inked if x0 <= x <= x1 and y0 <= y <= y1)
    return within / area


def check_ink_coverage(img: Image.Image, min_coverage: float, min_subject: float = 0.0) -> None:
    """Reject plates too faintly inked to read as a picture.

    Distinct from scan quality, which scores the *scan*: a pristine capture of
    a faint pencil study scores highly there and still gives a frame that looks
    empty.

    Two measurements, because one could not separate the cases seen in the pool
    audit. ``min_coverage`` is a low floor against a genuinely empty sheet;
    ``min_subject`` asks whether the worked part of the sheet is densely worked,
    which is what distinguishes a small exact engraving -- rejected wrongly by
    the single whole-sheet measure -- from a faint sketchbook page.
    """
    coverage = ink_coverage(img)
    if coverage < min_coverage:
        raise ImageError(
            f"only {coverage * 100:.1f}% of the plate carries ink "
            f"(minimum {min_coverage * 100:.1f}%): it would read as blank"
        )
    subject = subject_ink_coverage(img)
    if subject < min_subject:
        raise ImageError(
            f"the inked area is only {subject * 100:.1f}% worked "
            f"(minimum {min_subject * 100:.1f}%): too faint to read as a picture"
        )


def check_source_resolution(img: Image.Image, min_width: int, min_height: int) -> None:
    if img.width < min_width or img.height < min_height:
        raise ImageError(
            f"scan is {img.width}x{img.height}, below the {min_width}x{min_height} minimum"
        )


def check_aspect(img: Image.Image, max_source_aspect: float) -> None:
    """Reject plates too landscape to fill a vertical frame.

    Fitting a wide plate into 9:16 without cropping leaves it occupying a small
    band across the middle -- on a phone it reads as a tiny picture adrift in a
    field of paper. Cropping to fill is not an option, because it would risk
    clipping the plate's engraved caption, so the answer is to skip these and
    take the next candidate. Most bound plates are portrait, so this discards
    little of the pool.
    """
    aspect = img.width / img.height
    if aspect > max_source_aspect:
        raise ImageError(
            f"plate aspect {aspect:.2f} exceeds {max_source_aspect:.2f}; "
            "too landscape to frame vertically without cropping"
        )
