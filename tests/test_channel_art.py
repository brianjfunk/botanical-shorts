"""Geometry and crop-selection tests for channel art. No network.

The thing worth testing here is not "does it produce a PNG" but the two
constraints that are invisible until the art is already live on the channel:
that nothing essential falls outside YouTube's safe area, and that the avatar
crop lands on the drawing rather than on blank paper.
"""

from __future__ import annotations

import pytest
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


def _contrast(img: Image.Image) -> float:
    from PIL import ImageStat

    return ImageStat.Stat(img.convert("L")).stddev[0]


def test_hero_row_fits_inside_the_safe_band():
    """The hero row must survive a phone's crop to 1546x423 intact.

    Sized from the safe area, not the canvas: a hero plate taller than 423px
    is not merely tight, it is beheaded on the most common surface the channel
    is viewed on.
    """
    banner = channel_art.build_banner(plates(), veil_alpha=255)  # hide the field
    fill = banner.getpixel((5, 5))
    safe_top = (BANNER_SIZE[1] - SAFE_AREA[1]) // 2
    safe_bottom = safe_top + SAFE_AREA[1]

    for y in (safe_top - 12, safe_bottom + 12, 40, BANNER_SIZE[1] - 40):
        row = [banner.getpixel((x, y)) for x in range(0, BANNER_SIZE[0], 40)]
        assert all(px == fill for px in row), f"hero content outside the safe band at y={y}"


def test_background_field_fills_the_whole_canvas():
    """No bare corner: the TV crop shows all 2560x1440, not just the band."""
    banner = channel_art.build_banner(plates())
    # Corners and mid-edges, well away from the hero row.
    probes = [(300, 120), (2260, 120), (300, 1320), (2260, 1320), (1280, 100)]
    for x, y in probes:
        patch = banner.crop((x - 60, y - 60, x + 60, y + 60))
        assert _contrast(patch) > 1.0, f"canvas is bare at ({x},{y})"


def test_the_hero_row_reads_stronger_than_the_field_behind_it():
    """The veil has to actually recede, or the banner is visual noise."""
    banner = channel_art.build_banner(plates())
    mid_y = BANNER_SIZE[1] // 2
    hero = banner.crop((900, mid_y - 150, 1660, mid_y + 150))
    field = banner.crop((900, 60, 1660, 360))
    assert _contrast(hero) > _contrast(field) * 1.3


def test_banner_row_spans_the_full_canvas_width():
    """An empty end of the hero row would read as unfinished on desktop."""
    banner = channel_art.build_banner(plates(), veil_alpha=255)
    mid_y = BANNER_SIZE[1] // 2
    fill = banner.getpixel((5, 5))
    fade = int(BANNER_SIZE[0] * 0.05)

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
        assert found, f"no plate content in the {name} end of the hero row"


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


# -- plate suitability ------------------------------------------------------

def test_plate_on_paper_is_accepted():
    ok, _ = channel_art.blends_with_paper(plate(1200, 1500, ink_box=(300, 400, 900, 1100)).image)
    assert ok


@pytest.mark.parametrize("border", [(0, 0, 0), (60, 40, 25)])
def test_dark_bordered_scan_is_rejected(border):
    """A black scan frame or dark mount can never sit on a paper sheet.

    This is what let a photograph shot against brown cloth, and two scans with
    heavy black frames, into the first banner: whatever tone fills the margins,
    they read as rectangles pasted on top rather than as sheets on a desk.
    """
    img = Image.new("RGB", (1200, 1500), border)
    img.paste(Image.new("RGB", (1000, 1300), (232, 222, 201)), (100, 100))
    ok, reason = channel_art.blends_with_paper(img)
    assert not ok and "dark border" in reason


def test_faint_sketch_is_rejected_as_effectively_blank():
    """Two pencil studies reached the first banner as empty white rectangles.

    At full height in a video they are delicate drawings; shrunk to a collage
    tile there is nothing there to see.
    """
    faint = Image.new("RGB", (1200, 1500), (250, 250, 248))
    draw = ImageDraw.Draw(faint)
    for y in range(500, 560, 12):  # a few pale strokes
        draw.line([(500, y), (700, y)], fill=(205, 205, 203), width=1)
    assert channel_art.ink_coverage(faint) < channel_art.MIN_INK_COVERAGE


def test_a_properly_engraved_plate_clears_the_ink_floor():
    inked = plate(1200, 1500, ink_box=(200, 300, 1000, 1200))
    assert channel_art.ink_coverage(inked.image) >= channel_art.MIN_INK_COVERAGE


def test_one_dark_edge_is_enough_to_reject():
    """Three clean edges do not rescue a plate guillotined by a black bar."""
    img = Image.new("RGB", (1200, 1500), (232, 222, 201))
    img.paste(Image.new("RGB", (1200, 90), (5, 5, 5)), (0, 0))
    ok, _ = channel_art.blends_with_paper(img)
    assert not ok


# -- attribution ------------------------------------------------------------

def test_attribution_lists_each_source_once():
    same = make_candidate(page_id="1")
    items = [Plate(image=plate().image, candidate=same) for _ in range(3)]
    block = channel_art.attribution_block(items)
    assert block.count("Curtis's Botanical Magazine") == 1
    assert "Biodiversity Heritage Library" in block
