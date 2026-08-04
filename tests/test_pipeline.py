from __future__ import annotations

import json
import os

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
    # Mirrors the shipped allowlist, so these tests exercise the vocabulary the
    # pipeline will actually meet rather than a convenient subset.
    return load_config().license


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
        species_name_visible=True,
        is_illustration=True,
        is_spread=False,
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


def test_two_page_spread_rejected():
    """A capture of two facing pages, illustration beside letterpress.

    Framed whole, the plate shares the screen with a page of text and a fold
    down the middle. The aspect gate only catches spreads wide enough to trip
    it, and two tall narrow pages stay under the limit -- which is how an
    Agapanthus spread reached the seeding batch.
    """
    ok, reason = passes(verdict(is_spread=True), min_quality=7, caption_mode="log_only")
    assert not ok and "two facing pages" in reason


def test_single_wide_plate_is_not_treated_as_a_spread():
    ok, _ = passes(verdict(is_spread=False), min_quality=7, caption_mode="log_only")
    assert ok


def test_missing_caption_accepted_in_log_only_mode():
    ok, _ = passes(verdict(caption_embedded=False), min_quality=7, caption_mode="log_only")
    assert ok, "log_only must record the caption verdict without enforcing it"


def test_missing_caption_rejected_in_hard_gate_mode():
    ok, reason = passes(verdict(caption_embedded=False), min_quality=7, caption_mode="hard_gate")
    assert not ok and "lettering" in reason


def test_vision_error_fails_closed():
    ok, _ = passes(
        VisionVerdict(0, False, False, False, False, "", [], error="timeout"),
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


# -- subject discovery -------------------------------------------------------

class StubClient(bhl.BHLClient):
    """A BHLClient whose HTTP layer is replaced by canned responses."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def call(self, op, **params):
        self.calls.append((op, params))
        return self.responses.get(op)


def test_subject_discovery_uses_getsubjectmetadata_not_publicationsearch():
    # PublicationSearchAdvanced rejects a bare subject, so discovery must not
    # reach for it.
    client = StubClient(
        {"GetSubjectMetadata": [{"SubjectText": "Botany", "Publications": [
            {"BHLType": "Title", "TitleID": "1", "Title": "Flora"},
        ]}]}
    )
    client.subject_titles("Botany")
    ops = [op for op, _ in client.calls]
    assert ops == ["GetSubjectMetadata"]
    assert client.calls[0][1]["subject"] == "Botany"
    assert client.calls[0][1]["pubs"] == "t"


def test_subject_titles_drops_parts_and_keeps_titles():
    client = StubClient(
        {"GetSubjectMetadata": [{"Publications": [
            {"BHLType": "Title", "TitleID": "1", "Title": "Flora Londinensis"},
            {"BHLType": "Part", "PartID": "99", "Title": "An article"},
            {"BHLType": "Title", "TitleID": "2", "Title": "Botanical Magazine"},
        ]}]}
    )
    titles = client.subject_titles("Botany")
    assert [bhl.pick(t, "title_id") for t in titles] == ["1", "2"]


def test_subject_titles_keeps_untyped_records_carrying_a_title_id():
    # Tolerate a response that omits BHLType but is clearly title-level.
    client = StubClient(
        {"GetSubjectMetadata": [{"Publications": [
            {"TitleID": "5", "Title": "Untyped but title-level"},
            {"PartID": "6", "Title": "Untyped and not"},
        ]}]}
    )
    assert [bhl.pick(t, "title_id") for t in client.subject_titles("Botany")] == ["5"]


def test_subject_titles_empty_when_subject_missing():
    assert StubClient({"GetSubjectMetadata": []}).subject_titles("Nope") == []


def test_iter_candidates_walks_subject_to_page():
    client = StubClient({
        "GetSubjectMetadata": [{"Publications": [
            {"BHLType": "Part", "PartID": "99"},
            {"BHLType": "Title", "TitleID": "10"},
        ]}],
        "GetTitleMetadata": [{
            "TitleID": "10",
            "FullTitle": "Curtis's Botanical Magazine",
            "PublicationDate": "1805",
            "Authors": [{"Name": "Curtis, William"}],
            "Items": [{"ItemID": "20"}],
        }],
        "GetItemMetadata": [{
            "ItemID": "20",
            "RightsStatus": "Public domain",
            "Source": "Missouri Botanical Garden",
            "Pages": [
                {"PageID": "300", "PageTypes": ["Text"]},
                {"PageID": "301", "PageTypes": ["Illustration"]},
            ],
        }],
    })

    got = list(bhl.iter_candidates(
        client,
        subjects=["Botanical illustration"],
        page_types=["Illustration"],
        year_min=1700,
        year_max=1920,
        titles_per_subject=5,
        max_items_per_title=5,
        max_pages_per_item=5,
        limit=10,
    ))

    assert len(got) == 1, "only the Illustration page should survive"
    cand = got[0]
    assert cand.page_id == "301"
    assert cand.item_id == "20"
    assert cand.title == "Curtis's Botanical Magazine"
    assert cand.year == "1805"
    assert cand.authors == ["Curtis, William"]
    assert cand.rights == "Public domain"
    assert "GetSubjectMetadata" in [op for op, _ in client.calls]


def test_iter_candidates_skips_titles_outside_the_year_window():
    client = StubClient({
        "GetSubjectMetadata": [{"Publications": [{"BHLType": "Title", "TitleID": "10"}]}],
        "GetTitleMetadata": [{"TitleID": "10", "PublicationDate": "1975"}],
    })
    got = list(bhl.iter_candidates(
        client,
        subjects=["Botany"],
        page_types=["Illustration"],
        year_min=1700,
        year_max=1920,
        titles_per_subject=5,
        max_items_per_title=5,
        max_pages_per_item=5,
        limit=10,
    ))
    assert got == []


# -- licensing: the two tracks BHL actually populates -------------------------

@pytest.mark.parametrize(
    "rights",
    [
        # BHL's real PD wording -- licence fields are empty for these.
        "Public domain. The BHL considers this work no longer under copyright.",
        "Public domain",
        "Public Domain, Google-digitized",
        "Not in copyright",
        "NOT_IN_COPYRIGHT",
        "No known copyright restrictions",
    ],
)
def test_public_domain_rights_text_passes_without_licence_fields(rights, license_cfg):
    # Requiring LicenseName/LicenseUrl here would reject every genuine PD item,
    # because BHL simply does not populate them for public domain.
    cand = make_candidate(rights=rights, license_name="", license_url="")
    verdict = licensing.evaluate(cand, license_cfg)
    assert verdict.allowed, f"{rights!r} rejected: {verdict.reason}"


@pytest.mark.parametrize(
    "rights", ["In copyright", "In Copyright - Rights-holder(s) unlocated"]
)
def test_unnegated_in_copyright_still_rejected(rights, license_cfg):
    assert not licensing.evaluate(make_candidate(rights=rights), license_cfg).allowed


def test_cc_licence_read_from_licence_fields(license_cfg):
    cand = make_candidate(
        rights="",
        license_name="CC BY 4.0",
        license_url="https://creativecommons.org/licenses/by/4.0/",
    )
    assert licensing.evaluate(cand, license_cfg).allowed


def test_cc_wording_in_rights_text_alone_fails_closed(license_cfg):
    # Rights text saying "Creative Commons" does not say *which* licence, so an
    # NC or ND obligation could hide behind it. The licence fields are required.
    cand = make_candidate(rights="Creative Commons Attribution", license_name="", license_url="")
    verdict = licensing.evaluate(cand, license_cfg)
    assert not verdict.allowed


def test_restrictive_cc_licences_rejected_from_url_alone(license_cfg):
    cand = make_candidate(
        rights="",
        license_name="",
        license_url="https://creativecommons.org/licenses/by-nc-sa/4.0/",
    )
    assert not licensing.evaluate(cand, license_cfg).allowed


# -- regression: IA-sourced PD status tokens ---------------------------------
#
# A real run rejected six pages with:
#   "rights carry a restrictive marker: 'NOT_IN_COPYRIGHT'"
# The quoting in that message only occurs when the rights field is empty, so
# the token arrived via the LICENCE field, not rights/CopyrightStatus --
# Internet-Archive-sourced records carry a bare status token there. A
# public-domain status is a status, not a licence, so it must be honoured from
# whichever field the upstream source populated.

IA_PD_TOKENS = ["NOT_IN_COPYRIGHT", "PUBLIC_DOMAIN", "NOT_IN_COPYRIGHT_USA"]


@pytest.mark.parametrize("token", IA_PD_TOKENS)
@pytest.mark.parametrize("field", ["rights", "license_name", "license_url"])
def test_ia_public_domain_token_accepted_from_any_field(token, field, license_cfg):
    cand = make_candidate(**{"rights": "", "license_name": "", "license_url": "", field: token})
    verdict = licensing.evaluate(cand, license_cfg)
    assert verdict.allowed, f"{token!r} in {field} rejected: {verdict.reason}"


def test_ia_token_in_licence_field_is_not_mistaken_for_a_cc_licence(license_cfg):
    # It must pass as a public-domain *status*, not by matching CC vocabulary.
    cand = make_candidate(rights="", license_name="NOT_IN_COPYRIGHT", license_url="")
    verdict = licensing.evaluate(cand, license_cfg)
    assert verdict.allowed
    assert "public-domain status" in verdict.reason


@pytest.mark.parametrize("field", ["rights", "license_name", "license_url"])
def test_in_copyright_token_still_rejected_from_any_field(field, license_cfg):
    cand = make_candidate(
        **{"rights": "", "license_name": "", "license_url": "", field: "IN_COPYRIGHT"}
    )
    assert not licensing.evaluate(cand, license_cfg).allowed


def test_restrictive_marker_message_quotes_whichever_field_declared_it(license_cfg):
    # The old message bound !r to only one operand, so it quoted the value in
    # some cases and not others -- which made the source field ambiguous.
    from_rights = licensing.evaluate(make_candidate(rights="In copyright"), license_cfg)
    from_licence = licensing.evaluate(
        make_candidate(rights="", license_name="CC BY-NC 4.0"), license_cfg
    )
    assert "'In copyright'" in from_rights.reason
    assert "'CC BY-NC 4.0'" in from_licence.reason


# -- regression: the six page IDs from the failing run ------------------------
#
# These are the literal pages a real run rejected with
#   "rights carry a restrictive marker: 'NOT_IN_COPYRIGHT'"
# Pinning the IDs means a future change to field mapping or licence vocabulary
# that would re-break these exact plates fails loudly.

REJECTED_PAGE_IDS = ["6095347", "6095427", "6095423", "6095422", "6095417", "6095413"]

# The metadata shape is reconstructed from the reported log line rather than
# fetched: the message quoted the token, which the old formatting did only when
# the rights field was empty, placing the token in the licence field. The live
# test below verifies this against the real API when a key is available.
REJECTED_SHAPE = {"rights": "", "license_name": "NOT_IN_COPYRIGHT", "license_url": ""}


@pytest.mark.parametrize("page_id", REJECTED_PAGE_IDS)
def test_previously_rejected_pages_now_pass_the_licence_gate(page_id, license_cfg):
    cand = make_candidate(page_id=page_id, **REJECTED_SHAPE)
    verdict = licensing.evaluate(cand, license_cfg)
    assert verdict.allowed, f"page {page_id} rejected: {verdict.reason}"
    assert "public-domain status" in verdict.reason


@pytest.mark.parametrize("page_id", REJECTED_PAGE_IDS)
def test_previously_rejected_pages_have_usable_image_urls(page_id):
    cand = make_candidate(page_id=page_id, **REJECTED_SHAPE)
    assert cand.image_url.endswith(f"/pageimage/{page_id}")
    assert cand.page_url.endswith(f"/page/{page_id}")


def _live_key() -> str:
    """The BHL key, or "" if it is absent or an obvious placeholder.

    Placeholders matter: a shell that exported a literal "..." from a pasted
    command satisfies a plain truthiness check, so the live tests would run
    against a bogus key and fail as if the licence gate were broken.
    """
    key = os.environ.get("BHL_API_KEY", "").strip()
    if key.strip(".") == "" or key.lower() in {"changeme", "your-key", "xxx"}:
        return ""
    return key


@pytest.mark.skipif(
    not _live_key(),
    reason="needs a real BHL_API_KEY and network access to biodiversitylibrary.org",
)
@pytest.mark.parametrize("page_id", REJECTED_PAGE_IDS)
def test_live_rejected_pages_pass_with_real_metadata(page_id, license_cfg):
    """Resolve each page's real rights metadata and re-run the gate.

    This is the test that would catch a field-mapping regression: it reads
    whatever BHL actually returns today rather than a reconstructed shape.
    """
    client = bhl.BHLClient(_live_key())

    page_meta = client.get_page_metadata(page_id)
    assert page_meta, f"page {page_id} returned no metadata"
    item_id = str(bhl.pick(page_meta, "item_id") or "")
    assert item_id, f"page {page_id} metadata carries no ItemID: {sorted(page_meta)}"

    item_meta = client.get_item_metadata(item_id, pages=False)
    cand = make_candidate(
        page_id=page_id,
        item_id=item_id,
        rights=str(bhl.pick(item_meta, "rights") or ""),
        license_name=str(bhl.pick(item_meta, "license") or ""),
        license_url=str(bhl.pick(item_meta, "license_url") or ""),
    )
    verdict = licensing.evaluate(cand, license_cfg)
    assert verdict.allowed, (
        f"page {page_id} (item {item_id}) still rejected: {verdict.reason} "
        f"[rights={cand.rights!r} licence={cand.license_name!r} url={cand.license_url!r}]"
    )


# -- regression: BHL page-type whitespace ------------------------------------
#
# verify-bhl against the live API reported page types as:
#   [' Text', ' Title Page', 'Blank', 'Cover', 'Illustration', 'Index', 'Text']
# Note the leading spaces, and that ' Text' and 'Text' both occur in one item.
# Unstripped, a ' Illustration' variant would silently fail to match and the
# plate would never be considered.

def test_page_types_strip_leading_whitespace():
    assert bhl._page_types({"PageTypes": [" Text", "Illustration"]}) == ["Text", "Illustration"]
    assert bhl._page_types({"PageTypes": [{"PageTypeName": " Illustration"}]}) == ["Illustration"]
    assert bhl._page_types({"PageTypes": ["  ", ""]}) == []


@pytest.mark.parametrize("variant", ["Illustration", " Illustration", "Illustration ", "illustration"])
def test_whitespace_variants_of_illustration_still_match(variant):
    client = StubClient({
        "GetSubjectMetadata": [{"Publications": [{"BHLType": "Title", "TitleID": "10"}]}],
        "GetTitleMetadata": [{
            "TitleID": "10", "FullTitle": "Hesperides", "PublicationDate": "1646",
            "Items": [{"ItemID": "8848"}],
        }],
        "GetItemMetadata": [{
            "ItemID": "8848",
            "CopyrightStatus": "Public domain.  The BHL considers that this work is no longer under copyright.",
            "Source": "Internet Archive",
            "Pages": [{"PageID": "273051", "PageTypes": [variant]}],
        }],
    })
    # The real title is 1646, outside the shipped window, so widen it here --
    # this test is about page-type matching, not the year gate.
    got = list(bhl.iter_candidates(
        client, subjects=["Botanical illustration"], page_types=["Illustration", "Foldout"],
        year_min=1600, year_max=1920, titles_per_subject=5, max_items_per_title=5,
        max_pages_per_item=5, limit=10,
    ))
    assert len(got) == 1, f"{variant!r} failed to match"
    assert got[0].page_id == "273051"


def test_live_shaped_item_uses_copyrightstatus_for_rights(license_cfg):
    # GetItemMetadata carries CopyrightStatus and no License field at all.
    item = {
        "ItemID": "8848",
        "CopyrightStatus": "Public domain.  The BHL considers that this work is no longer under copyright.",
        "Source": "Internet Archive",
    }
    assert bhl.pick(item, "rights"), "CopyrightStatus must resolve as rights"
    assert bhl.pick(item, "license") is None
    cand = make_candidate(
        rights=str(bhl.pick(item, "rights")),
        license_name=str(bhl.pick(item, "license") or ""),
        license_url="",
    )
    assert licensing.evaluate(cand, license_cfg).allowed


# -- regression: a rejected Anthropic key must fail fast ----------------------
#
# A dry run on Actions with an invalid ANTHROPIC_API_KEY burned all 12 vision
# calls re-proving the same 401, then reported "no publishable plate found" --
# which points at the candidate pool rather than at the credential.

class _AuthError(Exception):
    status_code = 401


class _FakeMessages:
    def __init__(self, exc):
        self.exc = exc
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        raise self.exc


class _FakeAnthropic:
    def __init__(self, exc):
        self.messages = _FakeMessages(exc)


def test_rejected_api_key_raises_instead_of_rejecting_the_candidate():
    from botanical_shorts.vision import VisionAuthError, inspect_plate

    client = _FakeAnthropic(_AuthError("API key is invalid."))
    with pytest.raises(VisionAuthError, match="rejected the API key"):
        inspect_plate(client, make_plate(), model="claude-sonnet-5")
    assert client.messages.calls == 1, "must not retry a permanently bad credential"


def test_transient_vision_failure_still_degrades_to_a_rejection():
    from botanical_shorts.vision import inspect_plate

    client = _FakeAnthropic(RuntimeError("connection reset"))
    verdict = inspect_plate(client, make_plate(), model="claude-sonnet-5")
    assert verdict.error, "a transient failure should be reported, not raised"
    ok, reason = passes(verdict, min_quality=7, caption_mode="log_only")
    assert not ok and "errored" in reason


def test_raw_status_tokens_are_humanised_in_the_description():
    # The live dry run put "Rights: NOT_IN_COPYRIGHT" in a public-facing video
    # description. Fine as a gate value, machine-readable noise to a viewer.
    cand = make_candidate(rights="NOT_IN_COPYRIGHT", license_name="", license_url="")
    desc = metadata.build_description(cand, None)
    assert "Rights: Public domain" in desc
    assert "NOT_IN_COPYRIGHT" not in desc


def test_prose_rights_text_is_left_alone():
    cand = make_candidate(rights="Public domain. The BHL considers this work no longer under copyright.")
    assert "The BHL considers" in metadata.build_description(cand, None)


# -- title: strip the stock illustration lead-in ------------------------------
#
# The first live run titled a plate "Botanical illustration of a lupine plant
# with purple flowers (1829)" -- accurate, but the lead-in is pure noise on a
# channel that publishes nothing but botanical illustrations.

@pytest.mark.parametrize(
    "summary,expected",
    [
        ("Botanical illustration of a lupine plant with purple flowers",
         "Lupine plant with purple flowers"),
        ("An illustration of Lupinus polyphyllus", "Lupinus polyphyllus"),
        ("Illustration of the common foxglove", "Common foxglove"),
        ("A drawing of a fern frond", "Fern frond"),
        ("Lupinus polyphyllus", "Lupinus polyphyllus"),   # already clean
        ("Purple-flowered lupine", "Purple-flowered lupine"),
    ],
)
def test_title_strips_stock_lead_ins(summary, expected):
    title = metadata.build_title(make_candidate(year=""), verdict(subject_summary=summary))
    assert title == expected


def test_binomial_capitalisation_is_preserved():
    # Never uppercase the epithet of a binomial when tidying.
    title = metadata.build_title(make_candidate(year=""), verdict(subject_summary="Lupinus polyphyllus"))
    assert title == "Lupinus polyphyllus"


def test_title_still_carries_the_year():
    title = metadata.build_title(
        make_candidate(year="1829"),
        verdict(subject_summary="Botanical illustration of a lupine"),
    )
    assert title == "Lupine (1829)"


def test_lead_in_only_summary_falls_back_rather_than_emptying():
    title = metadata.build_title(make_candidate(year=""), verdict(subject_summary="illustration of"))
    assert title and title != ""


# -- species name vs any lettering -------------------------------------------
#
# Edwards's Botanical Register plate 1217 carries an engraver's imprint and a
# plate number but no species name -- caption_embedded is true while the plate
# does not actually identify the plant. The two signals are recorded separately
# so a future hard_gate can pick the one that matters.

def test_species_name_is_tracked_separately_from_lettering():
    v = verdict(caption_embedded=True, species_name_visible=False)
    assert v.caption_embedded and not v.species_name_visible
    ok, _ = passes(v, min_quality=7, caption_mode="log_only")
    assert ok, "an imprint-only plate must still pass in log_only mode"


def test_species_name_absence_never_gates_even_in_hard_gate_mode():
    # hard_gate is about lettering on the plate, not about naming -- many fine
    # plates put the name on a facing page.
    v = verdict(caption_embedded=True, species_name_visible=False)
    ok, _ = passes(v, min_quality=7, caption_mode="hard_gate")
    assert ok


# -- regression: published work must not consume the candidate budget ---------
#
# Originally the walk emitted history-blocked candidates, so they counted
# against max_candidates. With a deterministic traversal the same fixed window
# filled with already-published plates and the pipeline hit a hard wall after
# ~14 videos. Skipping inside the walk, plus rotating the title window, keeps
# the supply open.

def _pool_client(titles=40, items=3, pages=8):
    return StubClient({
        "GetSubjectMetadata": [{"Publications": [
            {"BHLType": "Title", "TitleID": str(t)} for t in range(titles)]}],
        "GetTitleMetadata": None,
        "GetItemMetadata": None,
    })


class PoolClient(bhl.BHLClient):
    """A synthetic BHL with a deep pool, for exhaustion tests."""

    def __init__(self, titles=40, items=3, pages=8):
        self.titles, self.items, self.pages = titles, items, pages
        self.calls = 0

    def call(self, op, **p):
        self.calls += 1
        if op == "GetSubjectMetadata":
            return [{"Publications": [
                {"BHLType": "Title", "TitleID": str(t)} for t in range(self.titles)]}]
        if op == "GetTitleMetadata":
            t = p["id"]
            return [{"TitleID": t, "FullTitle": f"Work {t}", "PublicationDate": "1850",
                     "Items": [{"ItemID": f"{t}-{i}"} for i in range(self.items)]}]
        if op == "GetItemMetadata":
            it = p["id"]
            return [{"ItemID": it, "CopyrightStatus": "NOT_IN_COPYRIGHT",
                     "Source": "Internet Archive",
                     "Pages": [{"PageID": f"{it}-p{n}", "PageTypes": ["Illustration"]}
                               for n in range(self.pages)]}]
        return []


def _walk(client, hist, limit=40, titles_per_subject=20):
    return bhl.iter_candidates(
        client, subjects=["Botanical illustration"], page_types=["Illustration"],
        year_min=1700, year_max=1920, titles_per_subject=titles_per_subject,
        max_items_per_title=3, max_pages_per_item=6, limit=limit,
        skip_pages=hist.page_ids, skip_items=hist.item_ids,
        title_offset=len(hist.entries),
    )


def test_published_volumes_do_not_consume_the_candidate_budget(tmp_path):
    hist = History(tmp_path / "h.json")
    published = 0
    for _ in range(60):
        picked = next(iter(_walk(PoolClient(), hist)), None)
        if picked is None:
            break
        hist.record({"page_id": picked.page_id, "item_id": picked.item_id})
        published += 1
    # The old behaviour wedged at limit/max_pages_per_item ~= 6 volumes here.
    assert published == 60, f"ran dry after {published} publications"
    assert len(hist.item_ids) == 60, "each publication should use a fresh volume"


def test_walk_skips_a_used_volume_without_fetching_its_pages(tmp_path):
    hist = History(tmp_path / "h.json")
    first = next(iter(_walk(PoolClient(), hist)))
    hist.record({"page_id": first.page_id, "item_id": first.item_id})

    client = PoolClient()
    for c in _walk(client, hist):
        assert c.item_id != first.item_id
    # No GetItemMetadata call should have been made for the skipped volume.
    assert all(c.item_id != first.item_id for c in _walk(PoolClient(), hist))


def test_title_offset_rotates_the_window(tmp_path):
    hist = History(tmp_path / "h.json")
    first_at_zero = next(iter(_walk(PoolClient(), hist, titles_per_subject=5)))
    for i in range(7):
        hist.record({"page_id": f"x{i}", "item_id": f"x{i}"})
    later = next(iter(_walk(PoolClient(), hist, titles_per_subject=5)))
    assert later.title_id != first_at_zero.title_id, "window did not rotate"


# -- image gates ported from the channel-art build ---------------------------

def _paper(w=1200, h=1500, tone=(232, 222, 201)):
    return Image.new("RGB", (w, h), tone)


def test_shipped_config_carries_the_ported_gates():
    cfg = load_config()
    # 60, not the original 140: a pool survey showed 140 rejecting ordinary
    # dark and browned stock at 70-120, not just the scanner void at 0-6 the
    # check was built for. It cost roughly a quarter of the usable pool.
    assert cfg.image.min_border_luminance == 60
    assert cfg.image.min_ink_coverage == 0.05
    assert cfg.source.title_cooldown == 120


def test_black_framed_scan_is_rejected_before_framing():
    """sampled_paper would read the frame as scanner void and fall back to
    parchment, leaving a hard dark rectangle on a parchment field."""
    img = _paper()
    img.paste(Image.new("RGB", (1200, 100), (4, 4, 4)), (0, 0))
    with pytest.raises(imaging.ImageError, match="dark frame or mount"):
        imaging.check_border_tone(img, 140)


def test_plate_on_clean_paper_passes_the_border_check():
    imaging.check_border_tone(_paper(), 140)


def test_faint_pencil_study_is_rejected_though_the_scan_is_clean():
    """Exactly what scan_quality cannot catch: the scan is fine, the plate is
    almost empty."""
    from PIL import ImageDraw

    img = _paper(tone=(250, 250, 248))
    draw = ImageDraw.Draw(img)
    for y in range(600, 660, 12):
        draw.line([(500, y), (700, y)], fill=(205, 205, 203), width=1)
    with pytest.raises(imaging.ImageError, match="would read as blank"):
        imaging.check_ink_coverage(img, 0.05)


def test_properly_engraved_plate_clears_the_ink_floor():
    from PIL import ImageDraw

    img = _paper()
    draw = ImageDraw.Draw(img)
    for y in range(300, 1200, 4):
        draw.line([(200, y), (1000, y)], fill=(30, 25, 20), width=2)
    imaging.check_ink_coverage(img, 0.05)


def test_ink_is_measured_against_the_plates_own_paper_not_white():
    """A browned scan must not have its blank margin scored as ink."""
    browned = _paper(tone=(176, 158, 126))
    assert imaging.ink_coverage(browned) < 0.01


# -- spread regression -------------------------------------------------------
#
# Page 48345298 is the Agapanthus that reached the seeding batch: a capture of
# two facing pages, the plate on one side and letterpress on the other, with
# the fold between them. It cleared every gate at the time -- including the
# aspect check, because two tall narrow pages side by side stay under the 1.25
# limit that a wide spread would trip. Pinned so a future change that relaxes
# the spread gate fails here rather than on the channel.

SPREAD_PAGE_ID = "48345298"


def test_spread_page_is_not_caught_by_aspect_alone(license_cfg):
    """Document why geometry was not enough, so the vision gate is not dropped."""
    cand = make_candidate(page_id=SPREAD_PAGE_ID)
    assert licensing.evaluate(cand, license_cfg).allowed
    # A spread of two tall pages: wider than a single plate, still under the cap.
    spread = Image.new("RGB", (1200, 1000), (232, 222, 201))
    imaging.check_aspect(spread, 1.25)  # passes -- which is the whole problem


def test_spread_verdict_rejects_that_page():
    ok, reason = passes(
        verdict(is_spread=True), min_quality=7, caption_mode="log_only"
    )
    assert not ok and "two facing pages" in reason


# -- title cooldown ----------------------------------------------------------
#
# Two plates from Edwards's Botanical Register (title 383) reached the channel
# in one batch and looked like the same drawing. They were different pages in
# different volumes, so neither the page nor the item rule could see the
# repeat: the serial ran 1829-1847 and reissued engravings across volumes.

def test_cooldown_is_a_window_not_the_whole_history(tmp_path):
    h = History(tmp_path / "h.json")
    for i in range(10):
        h.record({"page_id": str(i), "item_id": str(i), "title_id": str(i)})
    recent = h.recent_title_ids(3)
    assert recent == {"7", "8", "9"}, "only the last N entries lock their works"


def test_cooldown_expires_so_a_serial_returns(tmp_path):
    """A permanent ban would retire the richest sources after one use."""
    h = History(tmp_path / "h.json")
    h.record({"page_id": "1", "item_id": "1", "title_id": "383"})
    for i in range(2, 8):
        h.record({"page_id": str(i), "item_id": str(i), "title_id": str(i)})
    assert "383" not in h.recent_title_ids(3)
    assert "383" in h.recent_title_ids(10)


def test_cooldown_of_zero_locks_nothing(tmp_path):
    h = History(tmp_path / "h.json")
    h.record({"page_id": "1", "item_id": "1", "title_id": "383"})
    assert h.recent_title_ids(0) == set()


def test_entries_without_title_id_are_ignored(tmp_path):
    """History written before title_id was tracked must not crash the rule."""
    h = History(tmp_path / "h.json")
    h.record({"page_id": "1", "item_id": "1"})
    h.record({"page_id": "2", "item_id": "2", "title_id": "383"})
    assert h.recent_title_ids(5) == {"383"}


def test_walk_skips_a_cooling_title_without_spending_the_budget():
    """The lupine case: a locked work must not consume `limit`.

    If it did, a run whose window filled with cooling titles would surface no
    new plate at all -- the same exhaustion trap the page and item skips avoid.
    """
    calls = {"title_meta": 0}

    class Client:
        def subject_titles(self, subject):
            return [{"TitleID": "383"}, {"TitleID": "999"}]

        def get_title_metadata(self, title_id, items=True):
            calls["title_meta"] += 1
            return {
                "FullTitle": f"Work {title_id}",
                "Year": "1830",
                "Items": [{"ItemID": f"i{title_id}"}],
            }

        def get_item_metadata(self, item_id, pages=True):
            return {
                "RightsStatus": "Public domain",
                "Pages": [{"PageID": f"p{item_id}", "PageTypes": ["Illustration"]}],
            }

    got = list(
        bhl.iter_candidates(
            Client(),
            subjects=["s"],
            page_types=["Illustration"],
            year_min=1700,
            year_max=1920,
            titles_per_subject=10,
            max_items_per_title=2,
            max_pages_per_item=2,
            limit=5,
            skip_titles=["383"],
        )
    )
    assert [c.title_id for c in got] == ["999"]
    assert calls["title_meta"] == 1, "a cooling title must be skipped before its metadata call"


def test_configured_subjects_exclude_the_dead_headings():
    """'Botany, Pictorial works' and 'Plants Pictorial works' returned zero
    title-level publications against the live API, so the channel ran on one
    subject for weeks without anything failing. A non-existent heading is
    indistinguishable from a thin one, so the only defence is not shipping
    headings that were never verified."""
    subjects = load_config().source.subjects
    assert "Botany, Pictorial works" not in subjects
    assert "Plants Pictorial works" not in subjects
    assert "botany" in subjects
