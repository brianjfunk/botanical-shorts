"""Command-line entry points.

    python -m botanical_shorts.cli run           # the daily pipeline
    python -m botanical_shorts.cli run --dry-run # no render, no upload
    python -m botanical_shorts.cli verify-bhl    # confirm live API field names
    python -m botanical_shorts.cli preview IMAGE # compare letterbox treatments
    python -m botanical_shorts.cli channel-art   # build banner + profile picture
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import bhl, imaging, licensing, pipeline
from .config import ConfigError, load_config, optional_env, require_env


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    results = pipeline.run(
        cfg, dry_run=args.dry_run, skip_upload=args.skip_upload, count=args.count
    )
    accepted = [r for r in results if r.accepted]
    print(json.dumps([r.summary for r in accepted], indent=2, ensure_ascii=False))

    if not accepted:
        print("No publishable plate found this run.", file=sys.stderr)
        return 1
    if len(accepted) < args.count:
        print(
            f"Produced {len(accepted)} of {args.count} requested plates.", file=sys.stderr
        )
        return 1
    return 0


def cmd_verify_bhl(args: argparse.Namespace) -> int:
    """Dump real BHL responses and check the field mapping.

    The published API schema was not reachable when this client was written, so
    field names go through :func:`bhl.pick` with a list of aliases. Run this
    where biodiversitylibrary.org is reachable to confirm which alias actually
    fires -- and to catch any logical field that resolves to nothing.
    """
    cfg = load_config(args.config)
    client = bhl.BHLClient(require_env("BHL_API_KEY"))
    subject = args.subject or cfg.source.subjects[0]

    # Every configured subject, not just the probed one. Two headings sat in
    # config returning zero title-level publications and nothing failed,
    # because BHL answers an unknown subject exactly as it answers a real but
    # thin one: HTTP 200 and an empty list. Field-name checks could never have
    # caught that, which is why this is its own gate.
    print("== Configured subjects")
    empty_subjects: list[str] = []
    for name in cfg.source.subjects:
        try:
            count = len(client.subject_titles(name))
        except Exception as exc:
            print(f"   [err ] {name!r}: {exc}")
            empty_subjects.append(name)
            continue
        if count:
            print(f"   [ok  ] {count:5} titles  {name!r}")
        else:
            print(f"   [DEAD]     0 titles  {name!r}")
            empty_subjects.append(name)

    if empty_subjects:
        print(
            f"\n   {len(empty_subjects)} subject(s) return nothing: "
            f"{', '.join(repr(s) for s in empty_subjects)}."
            "\n   These contribute no plates at all. Find real headings with:"
            "\n     python -m botanical_shorts.cli find-subjects <term>",
            file=sys.stderr,
        )
        return 1

    print(f"\n== GetSubjectMetadata(subject={subject!r}, pubs=t)")
    subject_meta = client.get_subject_metadata(subject, pubs=True)
    if not subject_meta:
        print(f"   subject {subject!r} not found in BHL", file=sys.stderr)
        return 1
    print("   keys:", sorted(subject_meta.keys()))
    publications = bhl._as_list(bhl.pick(subject_meta, "publications", []))
    print(f"   {len(publications)} publications")
    if publications:
        print("   publication record keys:", sorted(publications[0].keys()))
        types = {str(bhl.pick(p, "bhl_type") or "?") for p in publications}
        print("   BHLType values present:", sorted(types))

    titles = client.subject_titles(subject)
    print(f"   {len(titles)} title-level (Parts dropped)")
    if not titles:
        print("   no title-level publications; try a different subject", file=sys.stderr)
        return 1

    title_id = str(bhl.pick(titles[0], "title_id") or "")
    print(f"\n== GetTitleMetadata(id={title_id})")
    title_meta = client.get_title_metadata(title_id)
    print("   keys:", sorted(title_meta.keys()))

    items = bhl._as_list(bhl.pick(title_meta, "items", []))
    print(f"   {len(items)} items")
    if not items:
        return 1
    item_id = str(bhl.pick(items[0], "item_id") or "")

    print(f"\n== GetItemMetadata(id={item_id})")
    item_meta = client.get_item_metadata(item_id)
    print("   keys:", sorted(item_meta.keys()))

    pages = bhl._as_list(bhl.pick(item_meta, "pages", []))
    print(f"   {len(pages)} pages")
    if pages:
        print("   page record keys:", sorted(pages[0].keys()))
        seen: set[str] = set()
        for page in pages:
            seen.update(bhl._page_types(page))
        print("   page types present:", sorted(seen))
        print("   configured page_types:", cfg.source.page_types)
        overlap = {t.lower() for t in seen} & {t.lower() for t in cfg.source.page_types}
        print("   -> overlap:", sorted(overlap) or "NONE (adjust source.page_types)")

    print("\n== Field mapping resolution")
    # (label, record, logical, required). The licence fields are genuinely
    # optional: BHL omits them entirely for public-domain items, which declare
    # status in CopyrightStatus instead. Flagging their absence as a failure
    # would send you chasing alias names that do not exist.
    checks = [
        ("title_id", titles[0], "title_id", True),
        ("full_title", title_meta, "full_title", True),
        ("year", title_meta, "year", True),
        ("publisher", title_meta, "publisher", False),
        ("authors", title_meta, "authors", False),
        ("item_id", items[0], "item_id", True),
        ("rights", item_meta, "rights", True),
        ("license", item_meta, "license", False),
        ("license_url", item_meta, "license_url", False),
        ("source", item_meta, "source", False),
    ]
    if pages:
        checks += [
            ("page_id", pages[0], "page_id", True),
            ("page_types", pages[0], "page_types", True),
        ]

    missing_required: list[str] = []
    absent_optional: list[str] = []
    for label, record, logical, required in checks:
        value = bhl.pick(record, logical)
        present = value not in (None, "", [], {})
        if present:
            status = "ok "
        elif required:
            status = "MISS"
            missing_required.append(label)
        else:
            status = " - "
            absent_optional.append(label)
        # Show page types normalised, since that is the form the pipeline
        # actually matches against.
        if logical == "page_types" and present:
            value = bhl._page_types(record)
        print(f"   [{status}] {label:<14} = {str(value)[:70]}")

    if absent_optional:
        print(
            f"\n   Not present (optional): {', '.join(absent_optional)}."
            "\n   license/license_url are expected to be absent for public-domain"
            "\n   items -- BHL only populates them for CC-licensed material."
        )

    if missing_required:
        print(
            f"\n   {len(missing_required)} required field(s) unresolved: "
            f"{', '.join(missing_required)}."
            "\n   Add the real key names to FIELD_ALIASES in botanical_shorts/bhl.py.",
            file=sys.stderr,
        )
        return 1
    print("\n   All required fields resolved.")
    return 0


def cmd_check_youtube(args: argparse.Namespace) -> int:
    """Confirm the stored YouTube credentials still work. Uploads nothing."""
    from . import youtube

    missing = [
        name
        for name in ("YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET", "YOUTUBE_REFRESH_TOKEN")
        if not optional_env(name)
    ]
    if missing:
        print(f"missing secret(s): {', '.join(missing)}", file=sys.stderr)
        return 1

    info = youtube.check_credentials(
        require_env("YOUTUBE_CLIENT_ID"),
        require_env("YOUTUBE_CLIENT_SECRET"),
        require_env("YOUTUBE_REFRESH_TOKEN"),
    )
    print("== YouTube credentials")
    for key, value in info.items():
        print(f"   {key:<13} = {value}")

    video_id = args.video_id or optional_env("CHECK_VIDEO_ID")
    if video_id:
        creds = youtube.build_credentials(
            require_env("YOUTUBE_CLIENT_ID"),
            require_env("YOUTUBE_CLIENT_SECRET"),
            require_env("YOUTUBE_REFRESH_TOKEN"),
        )
        print(f"\n== Video {video_id}")
        try:
            for key, value in youtube.describe_video(video_id, creds).items():
                print(f"   {key:<16} = {value}")
        except Exception as exc:
            print(f"   lookup failed: {exc}")

    print("\n   Refresh token is valid. Nothing was uploaded.")
    return 0


def cmd_find_subjects(args: argparse.Namespace) -> int:
    """Search BHL for subject headings that actually carry titles.

    Two of the three configured subjects returned zero title-level
    publications, so the channel had been running on one subject without
    anything failing loudly -- a heading that does not exist looks exactly like
    one that is merely thin. This searches for real headings and reports how
    many titles each carries, so `source.subjects` can be set from evidence.
    """
    cfg = load_config(args.config)
    client = bhl.BHLClient(require_env("BHL_API_KEY"))

    # The published API reference was not reachable when this was written, so
    # the op name and its parameter are both guesses. SubjectSearch+searchterm
    # returned an empty Result for every term -- including "botany" -- which is
    # what a wrong op or wrong parameter name looks like, since a genuinely
    # empty search would still be odd for so broad a word. Try the plausible
    # spellings and report which one actually answers.
    # Confirmed against the live API: the parameter is `subject`, not
    # `searchterm`. With `searchterm` the call succeeds and returns an empty
    # Result -- indistinguishable from a search that genuinely found nothing,
    # which is why two dead subjects sat in config unnoticed. The other
    # spellings are kept as fallbacks in case the API changes.
    ops = [
        ("SubjectSearch", "subject"),
        ("SubjectSearch", "searchterm"),
        ("GetSubjects", None),
    ]

    def _probe(term: str) -> list[str]:
        for op, param in ops:
            kwargs = {param: term} if param else {}
            try:
                result = client.call(op, **kwargs)
            except Exception as exc:
                print(f"   {op}({param or ''}) -> error: {str(exc)[:70]}")
                continue
            records = bhl._as_list(result)
            if not records:
                print(f"   {op}({param or ''}) -> empty")
                continue
            print(f"   {op}({param or ''}) -> {len(records)} records")
            if args.raw:
                print(f"      record keys: {sorted(records[0].keys())}")
                print(f"      first: {str(records[0])[:200]}")
            names = []
            for rec in records:
                name = str(
                    rec.get("SubjectText")
                    or rec.get("Subject")
                    or rec.get("Name")
                    or rec.get("Title")
                    or ""
                ).strip()
                if name:
                    names.append(name)
            if names:
                return names
        return []

    found: dict[str, int] = {}
    for term in args.terms:
        print(f"\n== searching {term!r}")
        names = _probe(term)
        print(f"   {len(names)} headings returned")

        # Order matters more than it looks. BHL returns headings roughly
        # reverse-alphabetically, so slicing the first N tests long compounds
        # ("wood-boring insects", "willow-feeding insects") and never the base
        # heading that actually carries the titles. Probe the search term
        # itself first, then shortest-first, so "Insects" beats "understudied
        # insects" for a place in the budget.
        exact = [n for n in names if n.strip().lower() == term.strip().lower()]
        rest = sorted((n for n in names if n not in exact), key=lambda n: (len(n), n))
        for name in (exact + rest)[: args.per_term]:
            if name in found:
                continue
            try:
                titles = client.subject_titles(name)
            except Exception as exc:
                print(f"   [err ] {name[:60]}: {exc}")
                continue
            found[name] = len(titles)
            flag = "ok  " if titles else "EMPTY"
            print(f"   [{flag}] {len(titles):4} titles  {name[:60]}")

    usable = {k: v for k, v in found.items() if v > 0}
    print(f"\n== {len(usable)} headings carry titles, of {len(found)} probed")
    for name, count in sorted(usable.items(), key=lambda kv: -kv[1])[:25]:
        print(f"   {count:4}  {name}")

    configured = set(cfg.source.subjects)
    dead = [s for s in configured if found.get(s, 0) == 0 and s in found]
    if dead:
        print(f"\n   Configured but empty: {', '.join(dead)}")
    return 0


def cmd_backfill_history(args: argparse.Namespace) -> int:
    """Fill in title_id for history entries recorded before it was tracked.

    The cooldown works off title_id, so entries without one are invisible to
    it -- the 19 already-published plates would not lock their works. Resolves
    each page to its title and rewrites the file.
    """
    from .history import History

    cfg = load_config(args.config)
    client = bhl.BHLClient(require_env("BHL_API_KEY"))
    history = History(cfg.history_path)

    missing = [e for e in history.entries if not e.get("title_id")]
    print(f"{len(missing)} of {len(history.entries)} entries need a title_id")

    for entry in missing:
        page_id = str(entry.get("page_id") or "")
        if not page_id:
            continue
        try:
            page = client.get_page_metadata(page_id)
            item_id = str(bhl.pick(page, "item_id") or entry.get("item_id") or "")
            item = client.get_item_metadata(item_id, pages=False) if item_id else {}
            title_id = str(bhl.pick(item, "title_id") or "")
        except Exception as exc:
            print(f"   page {page_id}: lookup failed ({exc})")
            continue
        if title_id:
            entry["title_id"] = title_id
            print(f"   page {page_id} -> title {title_id}")
        else:
            print(f"   page {page_id}: no title_id resolved")

    if args.dry_run:
        print("\ndry run; history not written")
        return 0

    history.save()
    print(f"\nwrote {cfg.history_path}")
    return 0


def cmd_pool_survey(args: argparse.Namespace) -> int:
    """Measure how deep the usable plate pool is, and how many works it spans.

    The question this answers is how long a *title-level cooldown* can be. That
    is bounded by the number of distinct titles carrying at least one usable
    plate, not by the plate count: a cooldown of N videos needs N distinct
    titles to rotate through, and a serial like Edwards's Botanical Register
    contributes thousands of plates but only one title.

    Metadata and licensing are surveyed exhaustively, since they cost no
    downloads. The image gates are sampled, because settling them for the whole
    pool would mean fetching every scan.
    """
    import requests

    cfg = load_config(args.config)
    session = requests.Session()
    client = bhl.BHLClient(require_env("BHL_API_KEY"), session=session)

    titles_per_subject = args.titles or cfg.source.titles_per_subject
    # Measure a candidate category without committing it to config first.
    subjects = args.subjects or cfg.source.subjects

    print("== Titles per subject (title-level publications, Parts dropped)")
    all_title_ids: set[str] = set()
    for subject in subjects:
        try:
            titles = client.subject_titles(subject)
        except Exception as exc:
            print(f"   {subject!r}: lookup failed ({exc})")
            continue
        ids = {str(bhl.pick(t, "title_id") or "") for t in titles}
        ids.discard("")
        all_title_ids |= ids
        print(f"   {subject!r}: {len(ids)} titles")
    print(f"   union across subjects: {len(all_title_ids)} distinct titles")
    print(f"   (config reaches at most {titles_per_subject} per subject)")

    print("\n== Walking candidates")
    candidates = bhl.iter_candidates(
        client,
        subjects=subjects,
        page_types=cfg.source.page_types,
        year_min=cfg.source.year_min,
        year_max=cfg.source.year_max,
        titles_per_subject=titles_per_subject,
        max_items_per_title=cfg.source.max_items_per_title,
        max_pages_per_item=cfg.source.max_pages_per_item,
        limit=args.limit,
    )

    seen = 0
    licensed: list[bhl.PageCandidate] = []
    titles_seen: set[str] = set()
    titles_licensed: set[str] = set()
    for cand in candidates:
        seen += 1
        if cand.title_id:
            titles_seen.add(cand.title_id)
        if licensing.evaluate(cand, cfg.license).allowed:
            licensed.append(cand)
            if cand.title_id:
                titles_licensed.add(cand.title_id)
        if seen % 100 == 0:
            print(f"   {seen} candidates, {len(titles_seen)} titles so far...")

    print(f"   {seen} candidates across {len(titles_seen)} titles")
    print(f"   {len(licensed)} licence-passed across {len(titles_licensed)} titles")

    # Sample the image gates. Spread the sample across the walk rather than
    # taking the first N, so it is not dominated by one title.
    sample_size = min(args.sample, len(licensed))
    step = max(1, len(licensed) // sample_size) if sample_size else 1
    sample = licensed[::step][:sample_size]

    print(f"\n== Image gates, sampled on {len(sample)} of {len(licensed)} plates")
    passed = 0
    reasons: dict[str, int] = {}
    titles_passing: set[str] = set()
    # Aspect is recorded for every scan that downloads, including ones that
    # fail other gates. max_source_aspect was chosen against botanical plates,
    # which are overwhelmingly portrait; fish, reptile and Haeckel plates are
    # not, so the threshold needs re-deciding per category against real
    # numbers rather than carried over.
    aspects: list[float] = []
    for cand in sample:
        try:
            img = imaging.load_image(bhl.download_page_image(cand, session=session))
        except Exception as exc:
            key = str(exc).split(":")[0][:48]
            reasons[key] = reasons.get(key, 0) + 1
            continue
        aspects.append(img.width / img.height)
        try:
            imaging.check_source_resolution(
                img, cfg.image.min_source_width, cfg.image.min_source_height
            )
            imaging.check_aspect(img, cfg.image.max_source_aspect)
            imaging.check_border_tone(img, cfg.image.min_border_luminance)
            imaging.check_ink_coverage(img, cfg.image.min_ink_coverage)
        except Exception as exc:
            key = str(exc).split(":")[0][:48]
            reasons[key] = reasons.get(key, 0) + 1
            continue
        passed += 1
        if cand.title_id:
            titles_passing.add(cand.title_id)

    rate = passed / len(sample) if sample else 0.0
    print(f"   {passed}/{len(sample)} passed ({rate:.0%})")
    for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"   {n:4}  {reason}")

    if aspects:
        ordered = sorted(aspects)
        def pct(p: float) -> float:
            return ordered[min(len(ordered) - 1, int(len(ordered) * p))]

        print(f"\n== Aspect (width/height) over {len(ordered)} scans")
        print(f"   portrait (<=1.0) : {sum(1 for a in ordered if a <= 1.0) / len(ordered):.0%}")
        print(f"   median           : {pct(0.5):.2f}")
        print(f"   75th / 90th      : {pct(0.75):.2f} / {pct(0.90):.2f}")
        print(f"   widest           : {ordered[-1]:.2f}")
        print("   share kept at each candidate max_source_aspect:")
        for threshold in (1.25, 1.4, 1.6, 2.0):
            kept = sum(1 for a in ordered if a <= threshold) / len(ordered)
            mark = "  <- current" if abs(threshold - cfg.image.max_source_aspect) < 1e-9 else ""
            print(f"     {threshold:>4}  {kept:>4.0%}{mark}")

    print("\n== Implied pool")
    est_plates = int(len(licensed) * rate)
    print(f"   licence-passed plates walked : {len(licensed)}")
    print(f"   estimated past image gates   : ~{est_plates} (sampled rate, before vision)")
    print(f"   distinct titles, licence-ok  : {len(titles_licensed)}")
    print(f"   distinct titles in sample    : {len(titles_passing)} of {len(sample)} sampled")
    print()
    print("   A title-level cooldown of N videos needs N distinct usable titles.")
    for weeks, label in ((13, "3 months"), (26, "6 months")):
        need = weeks * 5
        verdict = "reachable" if len(titles_licensed) >= need else "NOT reachable"
        print(f"   {label:>9} at 5/week = {need:3} videos -> {verdict}")
    return 0


def cmd_page_info(args: argparse.Namespace) -> int:
    """Resolve page ids to their volume and parent work.

    History records page and item ids, which is all the dedupe rules need but
    not enough to answer "did these two come from the same book?" -- the
    question that decides whether a title-level rule would catch a repeat.
    Walks page -> item -> title for each id and prints them side by side.
    """
    cfg = load_config(args.config)
    del cfg  # only needed to surface a config error early
    client = bhl.BHLClient(require_env("BHL_API_KEY"))

    rows = []
    for page_id in args.page_ids:
        page = client.get_page_metadata(page_id)
        item_id = str(bhl.pick(page, "item_id") or "")
        item = client.get_item_metadata(item_id, pages=False) if item_id else {}
        title_id = str(bhl.pick(item, "title_id") or "")
        title = client.get_title_metadata(title_id, items=False) if title_id else {}
        rows.append(
            {
                "page_id": page_id,
                "item_id": item_id,
                "title_id": title_id,
                "title": str(bhl.pick(title, "full_title") or "?"),
                "year": str(bhl.pick(title, "year") or "?"),
            }
        )
        print(f"page {page_id}")
        print(f"   item_id  = {rows[-1]['item_id']}")
        print(f"   title_id = {rows[-1]['title_id']}")
        print(f"   title    = {rows[-1]['title'][:90]}")
        print(f"   year     = {rows[-1]['year']}")

    title_ids = [r["title_id"] for r in rows if r["title_id"]]
    if len(rows) > 1:
        print()
        if len(set(title_ids)) == 1 and title_ids:
            print(f"SAME TITLE ({title_ids[0]}): a title-level rule would relate these.")
        else:
            print("DIFFERENT TITLES: a title-level rule would NOT relate these.")
    return 0


def cmd_channel_art(args: argparse.Namespace) -> int:
    """Build the channel banner and profile picture from real plates.

    Uploads nothing -- YouTube's branding endpoints are a different scope, and
    channel art is a set-once decision worth looking at before it lands. The
    outputs go in ``build/channel-art/`` and get attached to the workflow run.
    """
    from . import channel_art

    cfg = load_config(args.config)
    out_dir = Path(args.out) if args.out else cfg.output_dir / "channel-art"
    out_dir.mkdir(parents=True, exist_ok=True)

    plates = channel_art.collect_plates(cfg, count=args.plates, offset=args.offset)
    print(f"collected {len(plates)} licence-passed plates (offset {args.offset})")

    banner = channel_art.build_banner(
        plates,
        border_px=cfg.image.border_px,
        border_color=cfg.image.border_color,
    )
    banner.save(out_dir / "banner.png")
    channel_art.safe_area_preview(banner).save(out_dir / "banner-safe-area.png")

    avatar_plate = channel_art.best_avatar_plate(plates)
    avatar = channel_art.build_avatar(avatar_plate)
    avatar.save(out_dir / "avatar.png")
    channel_art.circular_preview(avatar).save(out_dir / "avatar-circle.png")

    # The offset is what makes a build reproducible, so it belongs with the
    # record of which plates were used.
    (out_dir / "credits.txt").write_text(
        f"{channel_art.attribution_block(plates)}\n\nBuilt with --offset {args.offset}.\n"
    )

    print(f"\nbanner       {banner.size[0]}x{banner.size[1]}  -> {out_dir / 'banner.png'}")
    print(f"  safe area  cropped preview   -> {out_dir / 'banner-safe-area.png'}")
    print(f"avatar       {avatar.size[0]}x{avatar.size[1]}    -> {out_dir / 'avatar.png'}")
    print(f"  circular   masked preview    -> {out_dir / 'avatar-circle.png'}")
    print(f"\navatar cropped from: {avatar_plate.citation}")
    print("\nUpload the square avatar.png -- YouTube applies the circular mask itself.")
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    """Render the same plate under each letterbox treatment, side by side.

    This is the artefact for signing off the design decision -- open the output
    and pick.
    """
    cfg = load_config(args.config)
    src = Path(args.image)
    img = imaging.load_image(src.read_bytes())

    out_dir = Path(args.out or cfg.output_dir / "preview")
    out_dir.mkdir(parents=True, exist_ok=True)

    for treatment in ("sampled_paper", "fixed", "black"):
        framed = imaging.frame_vertical(
            img,
            width=cfg.image.width,
            height=cfg.image.height,
            margin_ratio=cfg.image.margin_ratio,
            letterbox=treatment,
            fixed_fill_color=cfg.image.fixed_fill_color,
            border_px=cfg.image.border_px,
            border_color=cfg.image.border_color,
        )
        path = out_dir / f"{src.stem}-{treatment}.png"
        framed.image.save(path)
        print(f"{treatment:<14} fill={framed.fill_color} -> {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="botanical-shorts")
    parser.add_argument("--config", help="path to settings.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="run the daily pipeline")
    p_run.add_argument("--dry-run", action="store_true", help="select and frame only")
    p_run.add_argument("--skip-upload", action="store_true", help="render but do not upload")
    p_run.add_argument(
        "--count", type=int, default=1,
        help="how many plates to produce (for seeding a new channel)",
    )
    p_run.set_defaults(func=cmd_run)

    p_verify = sub.add_parser("verify-bhl", help="confirm BHL field mapping against the live API")
    p_verify.add_argument("--subject", help="subject to probe (default: first configured)")
    p_verify.set_defaults(func=cmd_verify_bhl)

    p_check = sub.add_parser("check-youtube", help="verify YouTube credentials; uploads nothing")
    p_check.add_argument(
        "--video-id", help="also report which channel this video landed on and its state"
    )
    p_check.set_defaults(func=cmd_check_youtube)

    p_find = sub.add_parser("find-subjects", help="search BHL for subject headings that carry titles")
    p_find.add_argument("terms", nargs="+", help="search terms to probe")
    p_find.add_argument("--per-term", type=int, default=8, help="headings to test per term")
    p_find.add_argument("--raw", action="store_true", help="dump raw record keys for diagnosis")
    p_find.set_defaults(func=cmd_find_subjects)

    p_back = sub.add_parser("backfill-history", help="resolve title_id for older history entries")
    p_back.add_argument("--dry-run", action="store_true", help="report without writing")
    p_back.set_defaults(func=cmd_backfill_history)

    p_pool = sub.add_parser("pool-survey", help="measure how deep the usable plate pool is")
    p_pool.add_argument("--limit", type=int, default=1500, help="candidates to walk")
    p_pool.add_argument("--sample", type=int, default=60, help="plates to fetch for image gates")
    p_pool.add_argument("--titles", type=int, help="override source.titles_per_subject")
    p_pool.add_argument(
        "--subjects", nargs="+", help="measure these subjects instead of the configured ones"
    )
    p_pool.set_defaults(func=cmd_pool_survey)

    p_page = sub.add_parser("page-info", help="resolve page ids to their volume and work")
    p_page.add_argument("page_ids", nargs="+", help="BHL page ids to look up")
    p_page.set_defaults(func=cmd_page_info)

    p_art = sub.add_parser("channel-art", help="build the channel banner and profile picture")
    p_art.add_argument(
        "--plates", type=int, default=12, help="how many plates to collect for the banner row"
    )
    p_art.add_argument(
        "--offset",
        type=int,
        default=0,
        help="rotate the starting point in the title list; change it to get a "
             "different draw, keep it to reproduce a build exactly",
    )
    p_art.add_argument("--out", help="output directory")
    p_art.set_defaults(func=cmd_channel_art)

    p_preview = sub.add_parser("preview", help="render letterbox treatments for sign-off")
    p_preview.add_argument("image", help="path to a source plate image")
    p_preview.add_argument("--out", help="output directory")
    p_preview.set_defaults(func=cmd_preview)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    try:
        return args.func(args)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # surface a clean message, full trace when -v
        logging.getLogger(__name__).debug("unhandled error", exc_info=True)
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
