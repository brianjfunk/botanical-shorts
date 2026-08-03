"""Command-line entry points.

    python -m botanical_shorts.cli run           # the daily pipeline
    python -m botanical_shorts.cli run --dry-run # no render, no upload
    python -m botanical_shorts.cli verify-bhl    # confirm live API field names
    python -m botanical_shorts.cli preview IMAGE # compare letterbox treatments
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from . import bhl, imaging, pipeline
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
    result = pipeline.run(cfg, dry_run=args.dry_run, skip_upload=args.skip_upload)
    if not result.accepted:
        print("No publishable plate found this run.", file=sys.stderr)
        return 1
    print(json.dumps(result.summary, indent=2, ensure_ascii=False))
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

    print(f"== GetSubjectMetadata(subject={subject!r}, pubs=t)")
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
        print(f"   {key:<11} = {value}")
    print("\n   Refresh token is valid. Nothing was uploaded.")
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
    p_run.set_defaults(func=cmd_run)

    p_verify = sub.add_parser("verify-bhl", help="confirm BHL field mapping against the live API")
    p_verify.add_argument("--subject", help="subject to probe (default: first configured)")
    p_verify.set_defaults(func=cmd_verify_bhl)

    p_check = sub.add_parser("check-youtube", help="verify YouTube credentials; uploads nothing")
    p_check.set_defaults(func=cmd_check_youtube)

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
