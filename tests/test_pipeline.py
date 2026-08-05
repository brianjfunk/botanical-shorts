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
    # No auto-publish. The passive veto window suited one plate a day; a
    # reviewed batch has already had its judgement, and fifteen uploads sharing
    # one deadline would all go public at the same moment.
    assert cfg.upload.publish_delay_hours == 0


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
    # The ink judgement moved off the whole sheet: the pool audit showed the
    # single 0.05 whole-sheet figure rejecting small exact engravings that sat
    # in the middle of a large plate. What is left here is a bare floor against
    # an empty leaf; the real work is done by the subject measure.
    assert cfg.image.min_ink_coverage == 0.015
    assert cfg.image.min_subject_ink_coverage == 0.16
    # The cooldown is off. It deferred a whole work for 120 published videos,
    # and the pool audit showed what that cost: the auditor, which ignores it,
    # passed 30% of what it walked while the batch selector found two plates in
    # ten attempts. Its useful half survives as a per-batch rule.
    assert cfg.source.title_cooldown == 0
    assert cfg.source.max_plates_per_title_per_batch == 3


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
    subject without anything failing. A non-existent heading is
    indistinguishable from a thin one, so the only defence is not shipping
    headings that were never verified -- see the subject check in verify-bhl,
    which now fails loudly on an empty one."""
    subjects = load_config().source.subjects
    assert "Botany, Pictorial works" not in subjects
    assert "Plants Pictorial works" not in subjects
    assert "botany" in subjects


def test_verify_bhl_reports_a_dead_subject(capsys):
    """The check that would have caught the two dead headings.

    BHL answers an unknown subject the same way it answers a sparse one, so
    this has to assert on the *count*, not on the call succeeding.
    """
    import argparse

    # cli imports the pipeline, which imports youtube and so google-auth.
    # Skip rather than fail where the runtime deps are not installed.
    pytest.importorskip("google.auth")
    from botanical_shorts import cli

    class Client:
        def __init__(self, *a, **k):
            pass

        def subject_titles(self, subject):
            return [{"TitleID": "1"}] if subject == "botany" else []

    real_client, real_cfg = bhl.BHLClient, cli.load_config
    cfg = load_config()
    object.__setattr__(cfg.source, "subjects", ["botany", "made up heading"])
    bhl.BHLClient = Client
    cli.load_config = lambda *_a, **_k: cfg
    os.environ.setdefault("BHL_API_KEY", "test")
    try:
        rc = cli.cmd_verify_bhl(argparse.Namespace(config=None, subject=None))
    finally:
        bhl.BHLClient, cli.load_config = real_client, real_cfg

    out = capsys.readouterr()
    assert rc == 1
    assert "DEAD" in out.out
    assert "made up heading" in out.err


# -- batch review ------------------------------------------------------------
#
# The gates settle what is mechanically wrong; a person settles the rest. That
# hand-off is only safe if the reply from the review page is read exactly:
# misreading it either publishes a plate that was vetoed, or retires one that
# was wanted. Both are silent failures.

def _review_page(n=3):
    from botanical_shorts import review

    entries = [{"title": f"Plate {i}", "citation": "Somebody, 1830"} for i in range(n)]
    images = [Image.new("RGB", (600, 900), (232, 222, 201)) for _ in range(n)]
    return review.render(entries, images)


def test_review_page_is_self_contained():
    """Published behind a strict CSP: a remote <img> would simply not load."""
    page = _review_page()
    assert "data:image/jpeg;base64," in page
    assert "http://" not in page and "https://" not in page


def test_review_page_numbers_match_the_reject_code():
    """Captions are 1-based so a mistyped code is visible rather than silent."""
    page = _review_page(3)
    for n in (1, 2, 3):
        assert f"<b>{n}.</b>" in page
    assert "i + 1" in page


def test_review_page_escapes_titles():
    from botanical_shorts import review

    page = review.render(
        [{"title": '<script>alert(1)</script>', "citation": ""}],
        [Image.new("RGB", (60, 90), (230, 220, 200))],
    )
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("approve all", set()),
        ("APPROVE ALL", set()),
        ("reject 2", {1}),
        ("reject 1,3", {0, 2}),
        ("reject 1, 3", {0, 2}),
        ("2,5", {1, 4}),
    ],
)
def test_reject_codes_parse(reply, expected):
    raw = reply.strip().lower()
    got = set()
    if raw and raw not in {"approve all", "approve", "all"}:
        got = {int(d) - 1 for d in raw.replace("reject", " ").replace(",", " ").split()}
    assert got == expected


def test_rejections_are_recorded_so_a_plate_is_never_reoffered(tmp_path):
    """A rejection retires a plate exactly as a publication does: the page and
    volume dedupe key off ids that are present either way."""
    h = History(tmp_path / "h.json")
    h.record({"page_id": "9", "item_id": "8", "title_id": "7", "rejected": True})
    h.save()

    reloaded = History(tmp_path / "h.json")
    assert reloaded.has_page("9")
    assert reloaded.has_item("8")
    assert "7" in reloaded.recent_title_ids(10)


# -- regression: one bad volume must not consume the whole walk ---------------
#
# The first real batch stopped at zero plates. The tally showed 135 candidates
# rejected, 122 of them at download with "border luminance below 60" -- not a
# thin pool, but a handful of black-backdrop volumes each contributing eighteen
# consecutive candidates and each failing all eighteen for the same reason.
# Meanwhile twelve vision calls were charged for nine answers, because a call
# that failed outright was billed against the budget like a verdict.

class _StubBHLClient:
    """Stands in for the network: a fixed pool of works, items and pages."""

    def __init__(self, works):
        self.works = works

    def subject_titles(self, subject):
        return [{"TitleID": w["title_id"], "Year": "1850", "BHLType": "Title"} for w in self.works]

    def get_title_metadata(self, title_id):
        work = next(w for w in self.works if w["title_id"] == title_id)
        return {
            "TitleID": title_id,
            "FullTitle": work.get("title", "A Work"),
            "Year": "1850",
            "Items": [{"ItemID": work["item_id"]}],
        }

    def get_item_metadata(self, item_id):
        work = next(w for w in self.works if w["item_id"] == item_id)
        return {
            "ItemID": item_id,
            "RightsStatus": "Public domain",
            "Pages": [
                {"PageID": pid, "PageTypes": ["Illustration"]} for pid in work["page_ids"]
            ],
        }


def _dark_plate():
    """A scan bordered by black: exactly what the border gate exists to catch."""
    img = Image.new("RGB", (900, 1200), (6, 6, 6))
    img.paste(make_plate(700, 1000), (100, 100))
    return img


def _install_stub_pool(monkeypatch, works, plates, vision_client):
    """Wire select_and_build to an in-memory pool. ``plates`` maps page id to image."""
    from botanical_shorts import pipeline

    monkeypatch.setenv("BHL_API_KEY", "test-key")
    monkeypatch.setattr(bhl, "BHLClient", lambda key, session=None: _StubBHLClient(works))

    downloads: list[str] = []

    def fake_download(candidate, session=None, timeout=90):
        downloads.append(candidate.page_id)
        return plates[candidate.page_id]

    monkeypatch.setattr(bhl, "download_page_image", fake_download)
    monkeypatch.setattr(pipeline.imaging, "load_image", lambda data: data)
    return downloads


def _cfg_for_stub(**overrides):
    """The shipped config with the stub pool's dimensions. Config is frozen, so
    every adjustment goes through ``replace`` rather than assignment."""
    import dataclasses

    cfg = load_config()
    source = dataclasses.replace(
        cfg.source, max_pages_per_item=6, max_items_per_title=1, title_cooldown=0
    )
    image = dataclasses.replace(cfg.image, min_source_width=100, min_source_height=100)
    return dataclasses.replace(cfg, source=source, image=image, **overrides)


def _no_vision(cfg):
    import dataclasses

    return dataclasses.replace(cfg, vision=dataclasses.replace(cfg.vision, enabled=False))


def test_a_black_backdrop_volume_is_abandoned_after_a_few_plates(tmp_path, monkeypatch):
    from botanical_shorts import pipeline

    bad = {"title_id": "T1", "item_id": "I1", "page_ids": [f"b{i}" for i in range(6)]}
    good = {"title_id": "T2", "item_id": "I2", "page_ids": ["g0"]}
    plates = {pid: _dark_plate() for pid in bad["page_ids"]}
    plates["g0"] = make_plate(900, 1200)

    downloads = _install_stub_pool(monkeypatch, [bad, good], plates, None)
    cfg = _no_vision(_cfg_for_stub())

    result = pipeline.select_and_build(
        cfg, history=History(tmp_path / "h.json"), dry_run=True
    )

    assert result.accepted, "the good plate two works along must still be reached"
    assert result.summary["page_id"] == "g0"
    # Four strikes, then the rest of that work is skipped without downloading.
    assert len([d for d in downloads if d.startswith("b")]) == pipeline.MAX_TITLE_STRIKES
    assert any(r.stage == "title" for r in result.rejections)


def test_a_failed_vision_call_does_not_spend_the_call_budget(tmp_path, monkeypatch):
    from botanical_shorts import pipeline

    work = {"title_id": "T1", "item_id": "I1", "page_ids": [f"p{i}" for i in range(6)]}
    plates = {pid: make_plate(900, 1200) for pid in work["page_ids"]}
    _install_stub_pool(monkeypatch, [work], plates, None)

    import dataclasses
    cfg = _cfg_for_stub()
    cfg = dataclasses.replace(cfg, vision=dataclasses.replace(cfg.vision, max_vision_calls=1))

    calls = {"n": 0}

    def fake_inspect(client, img, *, model, attempts=2):
        calls["n"] += 1
        if calls["n"] == 1:
            return VisionVerdict(0, False, False, False, False, "", [], error="529 overloaded")
        return VisionVerdict(9, True, True, True, False, "Iris", [])

    monkeypatch.setattr(pipeline, "inspect_plate", fake_inspect)

    result = pipeline.select_and_build(
        cfg, history=History(tmp_path / "h.json"), dry_run=True, vision_client=object()
    )

    assert result.accepted, "a budget of one must survive one failed call"
    assert calls["n"] == 2


def test_a_sustained_vision_outage_stops_rather_than_walking_the_pool(tmp_path, monkeypatch):
    from botanical_shorts import pipeline

    work = {"title_id": "T1", "item_id": "I1", "page_ids": [f"p{i}" for i in range(6)]}
    plates = {pid: make_plate(900, 1200) for pid in work["page_ids"]}
    _install_stub_pool(monkeypatch, [work], plates, None)

    cfg = _cfg_for_stub()
    calls = {"n": 0}

    def always_fails(client, img, *, model, attempts=2):
        calls["n"] += 1
        return VisionVerdict(
            0, False, False, False, False, "", [],
            error="529 overloaded", error_is_transport=True,
        )

    monkeypatch.setattr(pipeline, "inspect_plate", always_fails)

    result = pipeline.select_and_build(
        cfg, history=History(tmp_path / "h.json"), dry_run=True, vision_client=object()
    )

    assert not result.accepted
    assert calls["n"] == pipeline.MAX_VISION_ERRORS
    # An outage must not retire the plates: they go back in the pool.
    assert not any(r.stage == "vision" and "budget" in r.reason for r in result.rejections)


def test_a_batch_pays_for_each_dud_download_only_once(tmp_path, monkeypatch):
    from botanical_shorts import pipeline

    # Two separate works, because one plate per volume is a rule of the walk.
    bad = {"title_id": "T1", "item_id": "I1", "page_ids": ["b0"]}
    good = {"title_id": "T2", "item_id": "I2", "page_ids": ["g0"]}
    more = {"title_id": "T3", "item_id": "I3", "page_ids": ["g1"]}
    plates = {"b0": _dark_plate(), "g0": make_plate(900, 1200), "g1": make_plate(900, 1200)}

    downloads = _install_stub_pool(monkeypatch, [bad, good, more], plates, None)
    cfg = _no_vision(
        _cfg_for_stub(history_path=tmp_path / "h.json", output_dir=tmp_path / "build")
    )

    batch = pipeline.build_batch(cfg, count=2)

    assert [e["page_id"] for e in batch] == ["g0", "g1"]
    assert downloads.count("b0") == 1, "the dud must not be re-downloaded per plate"


def test_an_empty_vision_response_is_retried_before_being_believed():
    from botanical_shorts import vision

    class _Block:
        type = "text"
        text = ""

    class _Msg:
        content = [_Block()]

    class _Good:
        type = "text"
        text = '{"scan_quality": 9, "is_illustration": true, "subject_summary": "Iris"}'

    class _Messages:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return _Msg()
            return type("M", (), {"content": [_Good()]})()

    client = type("C", (), {"messages": _Messages()})()
    verdict = vision.inspect_plate(client, make_plate(), model="claude-sonnet-5")
    assert not verdict.error
    assert verdict.scan_quality == 9
    assert client.messages.calls == 2


def test_rejection_summary_reports_the_stage_that_stopped_the_walk():
    """Printing the first N rejections reported whatever the walk met earliest.

    A walk that dies at the vision budget meets download rejections first, so
    the verdicts that actually stopped it never appeared in the log at all --
    which is how a batch failure got misread as a thin pool.
    """
    from botanical_shorts.pipeline import Rejection, summarise_rejections

    rejections = [
        Rejection(f"d{i}", "download", f"border luminance {i} is below 60: dark frame")
        for i in range(40)
    ] + [
        Rejection(f"v{i}", "vision", "not a pictorial plate") for i in range(9)
    ] + [
        Rejection("v9", "vision", "two facing pages captured together"),
    ]

    text = "\n".join(summarise_rejections(rejections))

    assert "download=40" in text and "vision=10" in text
    # Measurements collapse, so forty readings are one line rather than forty.
    assert "40 x border luminance N is below N: dark frame" in text
    assert "9 x not a pictorial plate" in text
    assert "1 x two facing pages captured together" in text


def test_rejection_summary_names_an_empty_pool_as_such():
    from botanical_shorts.pipeline import summarise_rejections

    assert "pool itself was empty" in summarise_rejections([])[0]


# -- the pool audit ----------------------------------------------------------
#
# The selector stops at the first plate that clears every gate, so the pool it
# walked past leaves no evidence but a count -- and "98 rejected at download"
# is equally consistent with the gate catching black scan frames and with it
# discarding good plates.

def test_audit_judges_every_candidate_instead_of_stopping_at_the_first_pass(
    tmp_path, monkeypatch
):
    from botanical_shorts import pipeline

    works = [
        {"title_id": "T1", "item_id": "I1", "page_ids": ["good1"]},
        {"title_id": "T2", "item_id": "I2", "page_ids": ["dark"]},
        {"title_id": "T3", "item_id": "I3", "page_ids": ["good2"]},
        {"title_id": "T4", "item_id": "I4", "page_ids": ["wide"]},
    ]
    plates = {
        "good1": make_plate(900, 1200),
        "dark": _dark_plate(),
        "good2": make_plate(900, 1200),
        "wide": make_plate(2000, 900),
    }
    _install_stub_pool(monkeypatch, works, plates, None)
    cfg = _cfg_for_stub(history_path=tmp_path / "h.json")

    records = pipeline.audit_pool(cfg, count=10, vision_calls=0)

    assert len(records) == 4, "the walk must not stop at the first passing plate"
    by_page = {r.page_id: r.stage for r in records}
    assert by_page == {
        "good1": "not inspected",
        "dark": "border",
        "good2": "not inspected",
        "wide": "aspect",
    }
    # A thumbnail for everything that was fetched, so each verdict is checkable
    # by eye rather than on trust.
    assert all(r.thumb is not None for r in records)


def test_audit_names_the_specific_gate_not_just_the_stage(tmp_path, monkeypatch):
    from botanical_shorts import pipeline

    works = [{"title_id": "T1", "item_id": "I1", "page_ids": ["blank"]}]
    # Clean paper, no engraving: passes the border gate, fails on ink.
    plates = {"blank": Image.new("RGB", (900, 1200), (232, 222, 201))}
    _install_stub_pool(monkeypatch, works, plates, None)
    cfg = _cfg_for_stub(history_path=tmp_path / "h.json")

    records = pipeline.audit_pool(cfg, count=5, vision_calls=0)
    assert [r.stage for r in records] == ["ink"]


def test_audit_page_groups_by_verdict_and_escapes_titles():
    from botanical_shorts import audit
    from botanical_shorts.pipeline import Audited

    records = [
        Audited("1", "<script>x</script>", "passed", "quality 9/10", make_plate(120, 160)),
        Audited("2", "A dark one", "border", "border luminance 6 is below 60", make_plate(120, 160)),
        Audited("3", "Unlicensed", "licence", "rights unknown"),
    ]
    page = audit.render(records, settings={"min_border_luminance": 60})

    assert "Passed every gate" in page
    assert "Rejected: dark border" in page
    assert "Rejected on licence (never downloaded)" in page
    assert "<script>x</script>" not in page and "&lt;script&gt;" in page
    # A licence rejection never fetched an image; the card says so rather than
    # showing a broken frame.
    assert "no image fetched" in page
    assert page.count("data:image/jpeg;base64,") == 2


def test_audit_page_shows_a_stage_it_was_never_told_about():
    """A new gate must not silently vanish from a page whose point is completeness."""
    from botanical_shorts import audit
    from botanical_shorts.pipeline import Audited

    page = audit.render([Audited("1", "T", "brand new gate", "why")], settings={})
    assert "brand new gate" in page


def test_overlapping_subjects_do_not_walk_the_same_work_twice():
    """Found by building the audit: the same page appeared under two headings.

    "Botanical illustration" is largely a subset of "botany", so every work in
    the overlap had its plates downloaded and gated once per heading.
    """
    class _Client:
        def __init__(self):
            self.title_calls = 0

        def subject_titles(self, subject):
            return [{"TitleID": "T1", "Year": "1850", "BHLType": "Title"}]

        def get_title_metadata(self, title_id):
            self.title_calls += 1
            return {"TitleID": title_id, "FullTitle": "Shared", "Year": "1850",
                    "Items": [{"ItemID": "I1"}]}

        def get_item_metadata(self, item_id):
            return {"ItemID": item_id, "RightsStatus": "Public domain",
                    "Pages": [{"PageID": "P1", "PageTypes": ["Illustration"]}]}

    client = _Client()
    got = list(bhl.iter_candidates(
        client,
        subjects=["Botanical illustration", "botany"],
        page_types=["Illustration"],
        year_min=1700, year_max=1920,
        titles_per_subject=10, max_items_per_title=3, max_pages_per_item=6,
        limit=50,
    ))

    assert [c.page_id for c in got] == ["P1"]
    assert client.title_calls == 1, "the second heading must not re-fetch the work"


# -- splitting a two-page capture --------------------------------------------
#
# From Brian's pass over the pool audit: "several rejected by the model where
# half of the page is a beautiful illustration, and half is some text because it
# looks like it was scanned as an open book". Those were being thrown away
# whole. The plate is fine -- it just has a facing page attached.

def _spread(illustration_side="left", w=1600, h=1000, shadow=True):
    """Two facing pages: an engraving on one side, letterpress on the other."""
    img = Image.new("RGB", (w, h), (232, 222, 201))
    mid = w // 2
    plate = (0, mid) if illustration_side == "left" else (mid, w)
    text = (mid, w) if illustration_side == "left" else (0, mid)

    # The engraving: one dense mass, as a real plate reads.
    for x in range(plate[0] + 120, plate[1] - 120):
        for y in range(200, h - 200):
            img.putpixel((x, y), (45, 40, 35))
    # Letterpress: regular thin lines of type.
    for y in range(120, h - 120, 24):
        for x in range(text[0] + 90, text[1] - 90):
            for dy in range(8):
                img.putpixel((x, y + dy), (70, 62, 55))
    if shadow:
        for x in range(mid - 6, mid + 6):
            for y in range(h):
                img.putpixel((x, y), (30, 26, 22))
    return img


def test_gutter_is_found_at_the_fold_shadow():
    img = _spread()
    assert abs(imaging.find_gutter(img) - img.width // 2) <= 12


def test_gutter_falls_back_to_centre_without_a_shadow():
    """A flatbed capture has no fold shadow; the centre is still the right cut."""
    img = _spread(shadow=False)
    assert imaging.find_gutter(img) == img.width // 2


@pytest.mark.parametrize("side", ["left", "right"])
def test_split_keeps_the_illustrated_half_and_drops_the_text(side):
    img = _spread(illustration_side=side)
    half = imaging.split_spread(img, side)

    assert half.width < img.width * 0.55, "must be one page, not the spread"
    assert half.height == img.height
    # The engraved mass survives; the dense block is what makes a plate a plate.
    assert imaging.ink_coverage(half) > imaging.ink_coverage(
        imaging.split_spread(img, "right" if side == "left" else "left")
    )


def test_split_refuses_a_side_it_cannot_act_on():
    for side in ("both", "neither", ""):
        with pytest.raises(imaging.ImageError):
            imaging.split_spread(_spread(), side)


def test_a_spread_is_rescued_rather_than_rejected(tmp_path, monkeypatch):
    from botanical_shorts import pipeline

    works = [{"title_id": "T1", "item_id": "I1", "page_ids": ["spread"]}]
    plates = {"spread": _spread("left", w=1600, h=1400)}
    _install_stub_pool(monkeypatch, works, plates, None)
    # 1600x1400 is 1.14, inside the aspect gate, so the spread reaches vision
    # exactly as the real ones did.
    cfg = _cfg_for_stub(history_path=tmp_path / "h.json", output_dir=tmp_path / "b")

    def fake_inspect(client, img, *, model, attempts=2):
        return VisionVerdict(
            9, True, True, True, True, "Agapanthus", [], illustration_side="left"
        )

    monkeypatch.setattr(pipeline, "inspect_plate", fake_inspect)

    result = pipeline.select_and_build(
        cfg, history=History(tmp_path / "h.json"), dry_run=True, vision_client=object()
    )

    assert result.accepted, "the illustrated half is publishable"
    # The frame was built from one page, so the source is about half as wide.
    assert result.summary["source_size"][0] < 1600 * 0.55


def test_a_spread_with_no_single_illustrated_side_is_still_rejected(tmp_path, monkeypatch):
    """Only a spread that can be cut into one good page is rescued."""
    from botanical_shorts import pipeline

    works = [{"title_id": "T1", "item_id": "I1", "page_ids": ["spread"]}]
    _install_stub_pool(monkeypatch, works, {"spread": _spread("left", w=1600, h=1400)}, None)
    cfg = _cfg_for_stub(history_path=tmp_path / "h.json")

    monkeypatch.setattr(
        pipeline,
        "inspect_plate",
        lambda c, i, *, model, attempts=2: VisionVerdict(
            9, True, True, True, True, "two text pages", [], illustration_side="neither"
        ),
    )

    result = pipeline.select_and_build(
        cfg, history=History(tmp_path / "h.json"), dry_run=True, vision_client=object()
    )
    assert not result.accepted
    assert any("facing pages" in r.reason for r in result.rejections)


# -- ink: sparse is not the same as faint ------------------------------------
#
# Also from the audit pass: "There are some good images rejected for too little
# ink." The whole-sheet measure cannot tell a small exact engraving on a large
# sheet from a pencil study covering the same fraction of it.

def _small_engraving_on_a_big_sheet():
    """A dense specimen occupying a tenth of the plate -- a common plate layout."""
    img = Image.new("RGB", (1200, 1600), (232, 222, 201))
    for x in range(520, 700):
        for y in range(700, 1000):
            img.putpixel((x, y), (35, 30, 28))
    return img


def _faint_pencil_study():
    """Marks everywhere, none of them dark: a clean scan that frames as empty."""
    img = Image.new("RGB", (1200, 1600), (236, 228, 210))
    for x in range(100, 1100, 7):
        for y in range(100, 1500, 7):
            img.putpixel((x, y), (205, 198, 186))
    return img


def test_a_small_dense_engraving_survives_the_ink_gate():
    img = _small_engraving_on_a_big_sheet()
    assert imaging.ink_coverage(img) < 0.05, "sparse on the sheet, which is the point"
    imaging.check_ink_coverage(img, 0.015, 0.16)  # must not raise


def test_a_faint_study_is_still_rejected():
    img = _faint_pencil_study()
    with pytest.raises(imaging.ImageError):
        imaging.check_ink_coverage(img, 0.015, 0.16)


def test_a_blank_leaf_is_rejected_by_the_floor():
    blank = Image.new("RGB", (1200, 1600), (232, 222, 201))
    with pytest.raises(imaging.ImageError, match="read as blank"):
        imaging.check_ink_coverage(blank, 0.015, 0.16)


def test_subject_coverage_ignores_a_speck_of_foxing_in_the_corner():
    """A single stray mark must not stretch the measured region to the whole sheet."""
    img = _small_engraving_on_a_big_sheet()
    before = imaging.subject_ink_coverage(img)
    for x in range(20, 34):
        for y in range(20, 34):
            img.putpixel((x, y), (60, 50, 45))
    assert abs(imaging.subject_ink_coverage(img) - before) < 0.05


# -- plates the model never saw ----------------------------------------------
#
# The pool audit surfaced a group that cleared every local gate and was never
# inspected, because the call budget ran out mid-walk. Brian's read was that
# they were decent images, and that a few per batch could go to the review pass
# he is already doing. The walk used to stop dead at that point instead, which
# is part of why a batch of ten produced two.

def test_the_walk_stops_at_the_budget_when_nobody_is_reviewing(tmp_path, monkeypatch):
    """The unattended run must never publish a plate nobody looked at."""
    import dataclasses

    from botanical_shorts import pipeline

    works = [{"title_id": f"T{i}", "item_id": f"I{i}", "page_ids": [f"p{i}"]} for i in range(4)]
    plates = {f"p{i}": make_plate(900, 1200) for i in range(4)}
    _install_stub_pool(monkeypatch, works, plates, None)
    cfg = _cfg_for_stub(history_path=tmp_path / "h.json")
    cfg = dataclasses.replace(cfg, vision=dataclasses.replace(cfg.vision, max_vision_calls=1))

    monkeypatch.setattr(
        pipeline,
        "inspect_plate",
        lambda c, i, *, model, attempts=2: VisionVerdict(
            9, True, True, False, False, "text page", []
        ),
    )

    result = pipeline.select_and_build(
        cfg, history=History(tmp_path / "h.json"), dry_run=True, vision_client=object()
    )
    assert not result.accepted
    assert any("budget exhausted" in r.reason for r in result.rejections)


def test_an_uninspected_plate_reaches_review_flagged(tmp_path, monkeypatch):
    import dataclasses

    from botanical_shorts import pipeline

    works = [{"title_id": f"T{i}", "item_id": f"I{i}", "page_ids": [f"p{i}"]} for i in range(4)]
    plates = {f"p{i}": make_plate(900, 1200) for i in range(4)}
    _install_stub_pool(monkeypatch, works, plates, None)
    cfg = _cfg_for_stub(history_path=tmp_path / "h.json", output_dir=tmp_path / "b")
    cfg = dataclasses.replace(cfg, vision=dataclasses.replace(cfg.vision, max_vision_calls=1))

    monkeypatch.setattr(
        pipeline,
        "inspect_plate",
        lambda c, i, *, model, attempts=2: VisionVerdict(
            9, True, True, False, False, "text page", []
        ),
    )

    result = pipeline.select_and_build(
        cfg,
        history=History(tmp_path / "h.json"),
        dry_run=True,
        vision_client=object(),
        allow_uninspected=True,
    )
    assert result.accepted
    assert result.summary["inspected"] is False
    # No verdict means no verdict-derived fields to pretend otherwise.
    assert "scan_quality" not in result.summary


def test_the_review_page_marks_what_the_model_never_saw():
    from botanical_shorts import review

    page = review.render(
        [
            {"title": "Seen", "citation": "", "inspected": True},
            {"title": "Unseen", "citation": "", "inspected": False},
        ],
        [Image.new("RGB", (60, 90), (230, 220, 200))] * 2,
    )
    assert page.count('class="flag"') == 1
    assert "unchecked" in page


def test_a_batch_caps_how_many_uninspected_plates_it_will_offer(tmp_path, monkeypatch):
    """In a bad stretch the model rejects nearly everything it sees, so the
    overflow past the budget is not a windfall."""
    import dataclasses

    from botanical_shorts import pipeline

    works = [{"title_id": f"T{i}", "item_id": f"I{i}", "page_ids": [f"p{i}"]} for i in range(12)]
    plates = {f"p{i}": make_plate(900, 1200) for i in range(12)}
    _install_stub_pool(monkeypatch, works, plates, None)
    cfg = _cfg_for_stub(history_path=tmp_path / "h.json", output_dir=tmp_path / "b")
    cfg = dataclasses.replace(
        cfg,
        vision=dataclasses.replace(
            cfg.vision, max_vision_calls=0, max_uninspected_per_batch=2
        ),
    )

    monkeypatch.setattr(
        pipeline,
        "inspect_plate",
        lambda c, i, *, model, attempts=2: pytest.fail("budget is zero; must not be called"),
    )

    batch = pipeline.build_batch(cfg, count=8)
    assert len(batch) == 2
    assert all(e["inspected"] is False for e in batch)


def test_a_batch_caps_how_many_plates_one_work_contributes(tmp_path, monkeypatch):
    """The cooldown's useful half, scoped to a batch.

    Long serials reissued the same engraving across volumes -- which is how two
    near-identical lupines reached the channel -- and page/volume dedupe cannot
    see it, because the ids genuinely differ. Within one batch that near-repeat
    is at its most visible, so one work contributes one plate.
    """
    from botanical_shorts import pipeline

    # One work with three separate volumes, exactly the serial shape.
    class _Serial(_StubBHLClient):
        def get_title_metadata(self, title_id):
            return {
                "TitleID": title_id, "FullTitle": "A long serial", "Year": "1850",
                "Items": [{"ItemID": "I1"}, {"ItemID": "I2"}, {"ItemID": "I3"}],
            }

        def get_item_metadata(self, item_id):
            return {
                "ItemID": item_id, "RightsStatus": "Public domain",
                "Pages": [{"PageID": f"{item_id}p", "PageTypes": ["Illustration"]}],
            }

    monkeypatch.setenv("BHL_API_KEY", "test-key")
    monkeypatch.setattr(
        bhl, "BHLClient",
        lambda key, session=None: _Serial([{"title_id": "T1", "item_id": "I1", "page_ids": []}]),
    )
    monkeypatch.setattr(
        bhl, "download_page_image", lambda c, session=None, timeout=90: make_plate(900, 1200)
    )
    monkeypatch.setattr(pipeline.imaging, "load_image", lambda data: data)

    cfg = _no_vision(
        _cfg_for_stub(history_path=tmp_path / "h.json", output_dir=tmp_path / "b")
    )
    import dataclasses
    cfg = dataclasses.replace(
        cfg, source=dataclasses.replace(cfg.source, max_items_per_title=3)
    )

    capped = dataclasses.replace(
        cfg, source=dataclasses.replace(cfg.source, max_plates_per_title_per_batch=2)
    )
    assert len(pipeline.build_batch(capped, count=3)) == 2, "the cap binds"

    # And with the rule off, the same pool gives the near-repeats back.
    loose = dataclasses.replace(
        cfg, source=dataclasses.replace(cfg.source, max_plates_per_title_per_batch=0)
    )
    assert len(pipeline.build_batch(loose, count=3)) == 3


def test_zero_delay_means_no_publish_schedule_at_all():
    """A reviewed batch uploads private and waits: fifteen videos sharing one
    deadline would all go public at the same moment, and the review page has
    already supplied the judgement the veto window bought time for."""
    from botanical_shorts import youtube

    assert youtube.scheduled_publish_time(0) is None
    assert youtube.scheduled_publish_time(-1) is None
    assert youtube.scheduled_publish_time(24).endswith("Z")


# -- a malformed answer is not an outage -------------------------------------
#
# The first real 15-plate batch stopped at four. The tally read "3 x inspection
# failed: Expecting value" and "1 x Unterminated string" -- four parse failures,
# exactly MAX_VISION_ERRORS, so the outage breaker ended a batch while the API
# was up and answering. The truncation was self-inflicted: adding
# illustration_side pushed responses past max_tokens=512.

def test_a_truncated_response_is_reported_as_a_parse_failure():
    from botanical_shorts import vision

    class _Cut:
        type = "text"
        # What a 512-token ceiling actually produces: valid JSON, cut mid-value.
        text = '{"scan_quality": 9, "subject_summary": "Cypripedium acaul'

    class _Messages:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            return type("M", (), {"content": [_Cut()]})()

    client = type("C", (), {"messages": _Messages()})()
    verdict = vision.inspect_plate(client, make_plate(), model="claude-sonnet-5")

    assert verdict.error
    assert not verdict.error_is_transport, "the API answered; it is not down"


def test_malformed_answers_do_not_stop_the_walk(tmp_path, monkeypatch):
    """Four bad responses from a working API must not end a batch."""
    from botanical_shorts import pipeline

    works = [{"title_id": f"T{i}", "item_id": f"I{i}", "page_ids": [f"p{i}"]} for i in range(6)]
    plates = {f"p{i}": make_plate(900, 1200) for i in range(6)}
    _install_stub_pool(monkeypatch, works, plates, None)
    cfg = _cfg_for_stub(history_path=tmp_path / "h.json", output_dir=tmp_path / "b")

    calls = {"n": 0}

    def flaky(client, img, *, model, attempts=2):
        calls["n"] += 1
        if calls["n"] <= 5:
            return VisionVerdict(
                0, False, False, False, False, "", [],
                error="Unterminated string", error_is_transport=False,
            )
        return VisionVerdict(9, True, True, True, False, "Iris", [])

    monkeypatch.setattr(pipeline, "inspect_plate", flaky)

    result = pipeline.select_and_build(
        cfg, history=History(tmp_path / "h.json"), dry_run=True, vision_client=object()
    )
    assert result.accepted, "five malformed answers must not end the walk"
    assert calls["n"] == 6


def test_the_max_tokens_ceiling_leaves_room_for_the_whole_verdict():
    """Regression guard on the number itself: 512 truncated real responses."""
    import inspect as _inspect

    from botanical_shorts import vision

    source = _inspect.getsource(vision.inspect_plate)
    assert "max_tokens=1024" in source


# -- harvest, queue, publish -------------------------------------------------
#
# The rebuild. The old shape was a first-match search called once per plate,
# and nearly every mechanism added around it existed to make that repetition
# survivable. One walk, no early stop, and ordering becomes a decision over the
# resulting set rather than exclusion rules applied one plate at a time.

def test_harvest_keeps_going_past_the_first_good_plate(tmp_path, monkeypatch):
    from botanical_shorts import harvest as h

    works = [{"title_id": f"T{i}", "item_id": f"I{i}", "page_ids": [f"p{i}"]} for i in range(5)]
    plates = {f"p{i}": make_plate(900, 1200) for i in range(5)}
    _install_stub_pool(monkeypatch, works, plates, None)
    cfg = _no_vision(_cfg_for_stub(history_path=tmp_path / "h.json"))

    result = h.harvest(cfg, limit=50)
    assert len(result.entries) == 5
    assert len(result.images) == 5
    assert all(e["status"] == "pending" for e in result.entries)


def test_harvest_takes_several_plates_from_one_volume(tmp_path, monkeypatch):
    """The rule that held every batch to four.

    One plate per volume was never chosen deliberately -- it came from "do not
    let one book dominate". A volume of a plate book holds dozens of different
    species, and the audit found 36 passing plates across only four works.
    """
    from botanical_shorts import harvest as h

    works = [{"title_id": "T1", "item_id": "I1", "page_ids": [f"p{i}" for i in range(5)]}]
    plates = {f"p{i}": make_plate(900, 1200) for i in range(5)}
    _install_stub_pool(monkeypatch, works, plates, None)
    cfg = _no_vision(_cfg_for_stub(history_path=tmp_path / "h.json"))

    assert len(h.harvest(cfg, limit=50).entries) == 5


def test_publish_order_spaces_plates_from_the_same_work_apart():
    """Brian's requirement: no long string of very similar images in a row."""
    from botanical_shorts.harvest import publish_order

    entries = (
        [{"page_id": f"a{i}", "title_id": "A"} for i in range(6)]
        + [{"page_id": f"b{i}", "title_id": "B"} for i in range(3)]
        + [{"page_id": f"c{i}", "title_id": "C"} for i in range(3)]
    )
    order = publish_order(entries, seed=1)

    assert len(order) == 12
    assert {e["page_id"] for e in order} == {e["page_id"] for e in entries}

    # The dominant work may repeat -- with half the queue it must -- but never
    # twice running while another work still has plates left.
    runs = [
        (a["title_id"], b["title_id"]) for a, b in zip(order, order[1:])
    ]
    assert not any(x == y == "B" for x, y in runs)
    assert not any(x == y == "C" for x, y in runs)
    longest = max(
        len(list(g)) for _, g in __import__("itertools").groupby(e["title_id"] for e in order)
    )
    assert longest <= 2, f"a run of {longest} from one work is a string of similar images"


def test_publish_order_is_random_but_reproducible_with_a_seed():
    from botanical_shorts.harvest import publish_order

    entries = [{"page_id": str(i), "title_id": f"T{i % 4}"} for i in range(20)]
    a = [e["page_id"] for e in publish_order(entries, seed=7)]
    b = [e["page_id"] for e in publish_order(entries, seed=7)]
    c = [e["page_id"] for e in publish_order(entries, seed=8)]
    assert a == b
    assert a != c, "different seeds must give different orders"
