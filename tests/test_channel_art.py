"""Geometry and crop-selection tests for channel art. No network.

The thing worth testing here is not "does it produce a PNG" but the two
constraints that are invisible until the art is already live on the channel:
that nothing essential falls outside YouTube's safe area, and that the avatar
crop lands on the drawing rather than on blank paper.
"""

from __future__ import annotations

from PIL import Image, ImageDraw

from botanical_shorts import channel_art
from botanical_shorts.channel_art import AVATAR_SIZE, BANNER_SIZE, SAFE_AREA, Plate

from .test_pipeline import make_candidate


def plate(width: int = 800, height: int = 1200, *, paper=(232, 222, 201), ink_box=None) -> Plate:
    """A synthetic plate: aged paper, with optional engraving in one region."""
    img = Image.new("RGB", (width, height), paper)
    if ink_box:
        draw = ImageDraw.Draw(img)
        x0, y0, x1, y1 = ink_box
        # Hatching rather than a solid block, so ink density and contrast both
        # register the way real engraving does.
        for y in range(y0, y1, 4):
            draw.line([(x0, y), (x1, y)], fill=(30, 25, 20), width=2)
    return Plate(image=img, candidate=make_candidate(page_id=f"p{width}x{height}"))


def plates(n: int = 6) -> list[Plate]:
    # Varied proportions, which is what the banner row is meant to show off.
    # Each carries ink across its middle, so tests can tell plate from paper --
    # a blank plate is indistinguishable from the background it sits on.
    shapes = [(800, 1200), (900, 1100), (750, 1300), (1000, 1150), (820, 1250), (880, 1050)]
    out = []
    for i, (w, h) in enumerate(shapes[:n]):
        ink = (int(w * 0.1), int(h * 0.2), int(w * 0.9), int(h * 0.8))
        out.append(
            Plate(
                image=plate(w, h, ink_box=ink).image,
                candidate=make_candidate(page_id=str(i), item_id=str(i)),
            )
        )
    return out


# -- banner -----------------------------------------------------------------

def test_banner_is_exactly_the_size_youtube_expects():
    banner = channel_art.build_banner(plates())
    assert banner.size == BANNER_SIZE == (2560, 1440)


def test_every_plate_stays_inside_the_safe_band_vertically():
    """The constraint that matters: a phone crops to 1546x423 and nothing else.

    A plate taller than the safe area is not merely tight -- it is beheaded on
    the most common surface the channel is viewed on.
    """
    banner = channel_art.build_banner(plates())
    safe_top = (BANNER_SIZE[1] - SAFE_AREA[1]) // 2
    safe_bottom = safe_top + SAFE_AREA[1]

    # Anything not background must fall between safe_top and safe_bottom.
    fill = banner.getpixel((5, 5))
    for y in (safe_top - 12, safe_bottom + 12, 40, BANNER_SIZE[1] - 40):
        row = [banner.getpixel((x, y)) for x in range(0, BANNER_SIZE[0], 40)]
        assert all(px == fill for px in row), f"content found outside the safe band at y={y}"


def test_banner_row_spans_the_full_canvas_width():
    """The TV crop shows all 2560px; an empty end would read as unfinished.

    Scans a region rather than one column: a single x can legitimately land in
    a plate's own blank margin, which says nothing about whether the row
    reaches that far.
    """
    banner = channel_art.build_banner(plates())
    mid_y = BANNER_SIZE[1] // 2
    fill = banner.getpixel((5, 5))
    fade = int(BANNER_SIZE[0] * 0.06)

    # Just inside the fade, at both ends.
    regions = {
        "left": range(fade + 10, fade + 250, 5),
        "right": range(BANNER_SIZE[0] - fade - 250, BANNER_SIZE[0] - fade - 10, 5),
    }
    for name, xs in regions.items():
        found = any(
            banner.getpixel((x, y)) != fill
            for x in xs
            for y in range(mid_y - 120, mid_y + 120, 8)
        )
        assert found, f"no plate content in the {name} end of the row"


def test_safe_area_preview_matches_youtubes_crop():
    banner = channel_art.build_banner(plates())
    assert channel_art.safe_area_preview(banner).size == SAFE_AREA


def test_banner_survives_fewer_plates_than_fill_the_row():
    """One plate must still produce a full sheet, by repeating along the row."""
    banner = channel_art.build_banner(plates(1))
    assert banner.size == BANNER_SIZE


def test_paper_tone_is_the_median_not_the_mean():
    """One bleached outlier must not lift the whole background."""
    warm = [Plate(image=plate(paper=(230, 218, 195)).image, candidate=make_candidate())] * 3
    outlier = [Plate(image=plate(paper=(255, 255, 255)).image, candidate=make_candidate())]
    tone = channel_art.average_paper_color(warm + outlier)
    assert tone == (230, 218, 195)


# -- avatar -----------------------------------------------------------------

def test_avatar_is_square_and_the_size_youtube_expects():
    avatar = channel_art.build_avatar(plate(1600, 2000, ink_box=(600, 700, 1100, 1300)))
    assert avatar.size == AVATAR_SIZE == (800, 800)


def test_crop_lands_on_the_engraving_not_the_margin():
    """The whole point of cropping in: no wide empty margin at small sizes."""
    img = plate(1600, 2000, ink_box=(900, 1300, 1400, 1800)).image
    x, y, side = channel_art.densest_square(img)
    cx, cy = x + side // 2, y + side // 2
    # The chosen window's centre should sit within the inked region.
    assert 900 <= cx <= 1400, f"crop centre x={cx} missed the ink"
    assert 1300 <= cy <= 1800, f"crop centre y={cy} missed the ink"


def test_crop_does_not_fall_below_the_output_resolution():
    """Cropping tighter than 800px and upscaling would undo the tight crop."""
    img = plate(1600, 2000, ink_box=(700, 900, 1200, 1500)).image
    _, _, side = channel_art.densest_square(img, min_side_px=800)
    assert side >= 800


def test_small_scan_still_yields_a_crop():
    """A scan smaller than 800px cannot satisfy min_side_px; it must not crash."""
    img = plate(700, 900, ink_box=(200, 300, 500, 700)).image
    x, y, side = channel_art.densest_square(img, min_side_px=800)
    assert side > 0 and x + side <= img.width and y + side <= img.height


def test_crop_stays_inside_the_source():
    img = plate(1600, 2000, ink_box=(1200, 1600, 1590, 1990)).image
    x, y, side = channel_art.densest_square(img)
    assert x >= 0 and y >= 0
    assert x + side <= img.width and y + side <= img.height


def test_avatar_plate_choice_prefers_contrast_over_blank_paper():
    blank = Plate(image=plate(1600, 2000).image, candidate=make_candidate(page_id="blank"))
    drawn = Plate(
        image=plate(1600, 2000, ink_box=(500, 600, 1100, 1400)).image,
        candidate=make_candidate(page_id="drawn"),
    )
    assert channel_art.best_avatar_plate([blank, drawn]).candidate.page_id == "drawn"


def test_circular_preview_clears_the_corners():
    avatar = channel_art.build_avatar(plate(1600, 2000, ink_box=(600, 700, 1100, 1300)))
    preview = channel_art.circular_preview(avatar)
    assert preview.mode == "RGBA"
    assert preview.getpixel((2, 2))[3] == 0          # corner masked away
    assert preview.getpixel((400, 400))[3] == 255    # centre kept


# -- attribution ------------------------------------------------------------

def test_attribution_lists_each_source_once():
    same = make_candidate(page_id="1")
    items = [Plate(image=plate().image, candidate=same) for _ in range(3)]
    block = channel_art.attribution_block(items)
    assert block.count("Curtis's Botanical Magazine") == 1
    assert "Biodiversity Heritage Library" in block
