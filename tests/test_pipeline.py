from __future__ import annotations

import json

import pytest
from PIL import Image

from botanical_shorts import bhl, imaging, licensing, metadata
from botanical_shorts.config import LicenseConfig, load_config
from botanical_shorts.history import History
from botanical_shorts.vision import VisionVerdict, passes


def make_candidate(**overrides) -> bhl.PageCandidate:
    base = dict(
        page_id="123",
        item_id="456",
        title_id="789",
        title="Curtis's Botanical Magazine",
        year="1805",
        publisher="London",
        authors=["William Curtis"],
        page_types=["Illustration"],
        rights="Public domain",
        license_name="",
        license_url="",
        source="Missouri Botanical Garden",
    )
    base.update(overrides)
    return bhl.PageCandidate(**base)


# -- config -----------------------------------------------------------------

def test_shipped_config_loads_and_matches_signed_off_decisions():
    cfg = load_config()
    assert cfg.image.letterbox == "sampled_paper"
    assert cfg.vision.caption_mode == "log_only"
    assert 1.0 <= cfg.video.duration_seconds <= 2.0
    assert cfg.upload.privacy_status == "private"
    assert cfg.upload.publish_delay_hours == 24


# -- BHL field tolerance -----------------------------------------------------

def test_pick_falls_back_through_aliases_and_case():
    assert bhl.pick({"TitleID": "7"}, "title_id") == "7"
    assert bhl.pick({"TitleId": "7"}, "title_id") == "7"
    assert bhl.pick({"titleid": "7"}, "title_id") == "7"
    assert bhl.pick({"TitleID": ""}, "title_id", "fallback") == "fallback"
    assert bhl.pick({}, "title_id") is None


def test_page_types_handles_string_list_and_nested_dict():
    assert bhl._page_types({"PageTypes": "Illustration"}) == ["Illustration"]
    assert bhl._page_types({"PageTypes": ["Text", "Illustration"]}) == ["Text", "Illustration"]
    assert bhl._page_types({"PageTypes": [{"PageTypeName": "Foldout"}]}) == ["Foldout"]
    assert bhl._page_types({}) == []


@pytest.mark.parametrize(
    "raw,expected",
    [("1805", "1805"), ("[1805]", "1805"), ("1805-1810", "1805"), ("n.d.", ""), (None, "")],
)
def test_year_normalisation(raw, expected):
    assert bhl._year(raw) == expected


def test_citation_includes_title_author_year():
    text = make_candidate().citation()
    assert "Curtis's Botanical Magazine" in text
    assert "William Curtis" in text
    assert "1805" in text
    assert "Biodiversity Heritage Library" in text


# -- licensing ---------------------------------------------------------------

@pytest.fixture
def license_cfg() -> LicenseConfig:
    return LicenseConfig(
        allowed_rights=["public domain", "cc0"],
        allowed_licenses=["cc0", "cc by"],
        allow_unknown=False,
    )


def test_public_domain_allowed(license_cfg):
    assert licensing.evaluate(make_candidate(), license_cfg).allowed


def test_cc0_licence_allowed(license_cfg):
    cand = make_candidate(rights="", license_name="CC0 1.0 Universal")
    assert licensing.evaluate(cand, license_cfg).allowed


def test_noncommercial_rejected_even_when_allowlist_would_match(license_cfg):
    cand = make_candidate(rights="", license_name="CC BY-NC 4.0")
    verdict = licensing.evaluate(cand, license_cfg)
    assert not verdict.allowed
    assert "restrictive" in verdict.reason


def test_no_derivatives_rejected(license_cfg):
    cand = make_candidate(rights="", license_name="CC BY-ND 4.0")
    assert not licensing.evaluate(cand, license_cfg).allowed


def test_in_copyright_rejected(license_cfg):
    assert not licensing.evaluate(make_candidate(rights="In Copyright"), license_cfg).allowed


def test_unknown_rights_fail_closed(license_cfg):
    cand = make_candidate(rights="", license_name="", license_url="")
    verdict = licensing.evaluate(cand, license_cfg)
    assert not verdict.allowed


def test_unknown_rights_allowed_when_configured():
    cfg = LicenseConfig(allowed_rights=["public domain"], allowed_licenses=[], allow_unknown=True)
    cand = make_candidate(rights="", license_name="", license_url="")
    assert licensing.evaluate(cand, cfg).allowed


def test_unrecognised_rights_string_rejected(license_cfg):
    cand = make_candidate(rights="Some Novel Rights Statement")
    assert not licensing.evaluate(cand, license_cfg).allowed


def test_nd_substring_in_word_does_not_falsely_block(license_cfg):
    # "nd" as a bare token is a No-Derivatives marker, but it must not match
    # inside ordinary words like "Foundation".
    cand = make_candidate(rights="Public domain", license_name="Wellcome Foundation grant")
    assert licensing.evaluate(cand, license_cfg).allowed


# -- imaging -----------------------------------------------------------------

def make_plate(w=800, h=1000, paper=(230, 218, 196)) -> Image.Image:
    img = Image.new("RGB", (w, h), paper)
    # A dark block in the middle, standing in for the engraving.
    for x in range(w // 4, 3 * w // 4):
        for y in range(h // 4, 3 * h // 4):
            img.putpixel((x, y), (40, 40, 40))
    return img


def test_sampled_paper_matches_the_border_tone():
    paper = (231, 219, 197)
    color = imaging.sample_paper_color(make_plate(paper=paper))
    assert all(abs(a - b) <= 4 for a, b in zip(color, paper)), color


def test_sampled_paper_falls_back_when_border_is_scanner_void():
    color = imaging.sample_paper_color(Image.new("RGB", (400, 400), (5, 5, 5)))
    assert sum(color) / 3 > 150


def test_frame_produces_exact_target_size_and_never_crops():
    img = make_plate(1600, 900)  # landscape, the hard case
    framed = imaging.frame_vertical(
        img,
        width=1080,
        height=1920,
        margin_ratio=0.06,
        letterbox="sampled_paper",
        fixed_fill_color="#E8DEC9",
    )
    assert framed.image.size == (1080, 1920)
    # Whole plate fits inside the safe area -- nothing clipped at the edge.
    assert framed.plate_size[0] <= int(1080 * 0.88) + 1
    assert framed.plate_size[1] <= int(1920 * 0.88) + 1
    # Aspect ratio preserved.
    assert abs(framed.plate_size[0] / framed.plate_size[1] - 1600 / 900) < 0.02


def test_frame_corners_are_filled_with_paper_not_black():
    framed = imaging.frame_vertical(
        make_plate(1600, 900),
        width=1080,
        height=1920,
        margin_ratio=0.06,
        letterbox="sampled_paper",
        fixed_fill_color="#E8DEC9",
        border_px=0,
    )
    corner = framed.image.getpixel((5, 5))
    assert sum(corner) / 3 > 150, corner


def test_black_letterbox_gives_black_margins():
    framed = imaging.frame_vertical(
        make_plate(1600, 900),
        width=1080,
        height=1920,
        margin_ratio=0.06,
        letterbox="black",
        fixed_fill_color="#E8DEC9",
        border_px=0,
    )
    assert framed.image.getpixel((5, 5)) == (0, 0, 0)


def test_tiny_plate_is_not_upscaled_past_the_limit():
    framed = imaging.frame_vertical(
        make_plate(200, 250),
        width=1080,
        height=1920,
        margin_ratio=0.06,
        letterbox="sampled_paper",
        fixed_fill_color="#E8DEC9",
    )
    assert framed.plate_size[0] <= int(200 * imaging.MAX_UPSCALE) + 1


def test_resolution_gate_rejects_small_scans():
    with pytest.raises(imaging.ImageError):
        imaging.check_source_resolution(make_plate(300, 300), 700, 900)


def test_load_image_rejects_garbage():
    with pytest.raises(imaging.ImageError):
        imaging.load_image(b"not an image")


# -- vision gates ------------------------------------------------------------

def verdict(**overrides) -> VisionVerdict:
    base = dict(
        scan_quality=9,
        caption_embedded=True,
        is_illustration=True,
        subject_summary="A flowering magnolia branch",
        issues=[],
    )
    base.update(overrides)
    return VisionVerdict(**base)


def test_clean_plate_accepted():
    ok, _ = passes(verdict(), min_quality=7, caption_mode="log_only")
    assert ok


def test_poor_scan_rejected():
    ok, reason = passes(verdict(scan_quality=4, issues=["bleed-through"]), min_quality=7, caption_mode="log_only")
    assert not ok and "bleed-through" in reason


def test_non_illustration_rejected():
    ok, reason = passes(verdict(is_illustration=False), min_quality=7, caption_mode="log_only")
    assert not ok and "pictorial" in reason


def test_missing_caption_accepted_in_log_only_mode():
    ok, _ = passes(verdict(caption_embedded=False), min_quality=7, caption_mode="log_only")
    assert ok, "log_only must record the caption verdict without enforcing it"


def test_missing_caption_rejected_in_hard_gate_mode():
    ok, reason = passes(verdict(caption_embedded=False), min_quality=7, caption_mode="hard_gate")
    assert not ok and "lettering" in reason


def test_vision_error_fails_closed():
    ok, _ = passes(
        VisionVerdict(0, False, False, "", [], error="timeout"),
        min_quality=7,
        caption_mode="log_only",
    )
    assert not ok


# -- metadata ----------------------------------------------------------------

def test_description_carries_attribution_and_rights():
    desc = metadata.build_description(make_candidate(), verdict())
    assert "Curtis's Botanical Magazine" in desc
    assert "Biodiversity Heritage Library" in desc
    assert "Public domain" in desc
    assert "biodiversitylibrary.org/page/123" in desc
    assert len(desc) <= metadata.YOUTUBE_DESC_LIMIT


def test_title_is_truncated_to_youtube_limit():
    title = metadata.build_title(make_candidate(), verdict(subject_summary="x" * 300))
    assert len(title) <= metadata.YOUTUBE_TITLE_LIMIT


def test_title_falls_back_to_publication_when_vision_is_absent():
    assert "Curtis" in metadata.build_title(make_candidate(), None)


# -- history -----------------------------------------------------------------

def test_history_round_trip(tmp_path):
    path = tmp_path / "history.json"
    hist = History(path)
    assert not hist.has_page("123")

    hist.record({"page_id": "123", "item_id": "456", "title": "Magnolia"})
    hist.save()

    reloaded = History(path)
    assert reloaded.has_page("123")
    assert reloaded.has_item("456")
    assert not reloaded.has_page("999")
    assert reloaded.entries[0]["published_at"]


def test_corrupt_history_does_not_crash(tmp_path):
    path = tmp_path / "history.json"
    path.write_text("{ not json")
    assert History(path).entries == []


def test_shipped_history_file_is_valid_json():
    from botanical_shorts.config import REPO_ROOT

    data = json.loads((REPO_ROOT / "state" / "history.json").read_text())
    assert isinstance(data, list)


# -- scheduling --------------------------------------------------------------

def test_publish_time_is_rfc3339_utc():
    from datetime import datetime, timezone

    from botanical_shorts.youtube import scheduled_publish_time

    now = datetime(2026, 8, 3, 12, 0, 0, tzinfo=timezone.utc)
    assert scheduled_publish_time(24, now=now) == "2026-08-04T12:00:00Z"


def test_landscape_plate_rejected_as_unframeable():
    with pytest.raises(imaging.ImageError, match="too landscape"):
        imaging.check_aspect(make_plate(1700, 1100), 1.25)


def test_portrait_plate_passes_aspect_gate():
    imaging.check_aspect(make_plate(800, 1100), 1.25)


def test_portrait_plate_fills_most_of_the_frame():
    # The common case: a portrait plate should dominate the frame, not float
    # in it. Guards against a regression in margin/fit handling.
    framed = imaging.frame_vertical(
        make_plate(1000, 1500),
        width=1080,
        height=1920,
        margin_ratio=0.06,
        letterbox="sampled_paper",
        fixed_fill_color="#E8DEC9",
    )
    coverage = (framed.plate_size[0] * framed.plate_size[1]) / (1080 * 1920)
    assert coverage > 0.55, f"plate covers only {coverage:.0%} of frame"
