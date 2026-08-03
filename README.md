# Botanical Shorts

An automated pipeline that sources a public-domain historical botanical
illustration from the Biodiversity Heritage Library, frames it as a still
vertical Short, and uploads it to YouTube scheduled to publish ~24 hours out.

The daily steady state is doing nothing: the video auto-publishes unless it's
pulled. Review is one glance at a Slack message to confirm the scan rendered
cleanly.

## The defining constraint: no overlaid text

Nothing is ever composited onto the frame. Classic botanical plates carry their
species name printed or hand-lettered directly on the engraving, and that
lettering is the only typography the channel uses. Re-rendering a name in a
modern font over a 200-year-old lithograph would look wrong, so the pipeline
has no text-layout stage at all. All attribution lives in the video
description.

This has a knock-on effect worth knowing: the pipeline never crops a plate to
fill the frame, only scales it to fit, because cropping risks clipping that
engraved caption at the frame edge.

## Pipeline

```
BHL subject search
  └─ title → item → page traversal, keeping pages BHL flags as illustrations
      └─ history dedupe        (never repeat a plate, or a volume)
        └─ licence gate        (per-item rights, allowlist, fails closed)
          └─ download scan
            └─ resolution + aspect gates
              └─ Claude vision (scan quality: enforced · caption: logged)
                └─ frame to 1080×1920 on sampled aged-paper fill
                  └─ ffmpeg → 2s static h264, no audio
                    └─ YouTube upload, private + publishAt +24h
                      └─ Slack review ping
```

The candidate walk is lazy and stops at the first plate that clears every gate,
so a normal run costs a handful of BHL calls and one vision call.

## Decisions baked into `config/settings.yaml`

| Decision | Setting | Value |
|---|---|---|
| Scope | `source.subjects` | Botanical only; widening to zoological/entomological is appending subjects to that list — nothing else is scope-aware |
| Letterboxing | `image.letterbox` | `sampled_paper` — margins filled with the plate's own paper tone, sampled from its border |
| Caption check | `vision.caption_mode` | `log_only` — the verdict is recorded but not enforced in v1 |
| Duration | `video.duration_seconds` | 2.0s |

### On `caption_mode`

BHL flags *which pages are illustration plates* (that's `PageTypes`, and the
pipeline uses it), but nothing in its schema indicates whether a plate's caption
is engraved on the plate versus typeset on a facing page. So that judgement has
to be visual. In v1 the vision call makes it and logs it without rejecting on
it, so you can check the verdicts against real output before trusting the
filter. Once they look right, set `caption_mode: hard_gate`.

Scan quality is enforced from day one — that's the failure the review gate
exists to catch.

## Setup

### 1. Secrets

Set these as GitHub Actions repository secrets:

| Secret | Where it comes from |
|---|---|
| `BHL_API_KEY` | biodiversitylibrary.org/getapikey |
| `ANTHROPIC_API_KEY` | Reused from the Balm pipeline |
| `YOUTUBE_CLIENT_ID` | Google Cloud OAuth client |
| `YOUTUBE_CLIENT_SECRET` | Google Cloud OAuth client |
| `YOUTUBE_REFRESH_TOKEN` | `scripts/get_youtube_refresh_token.py` |
| `SLACK_WEBHOOK_URL` | Optional; without it the run page is the notification |

### 2. YouTube OAuth

In the Google Cloud console: enable **YouTube Data API v3**, configure the OAuth
consent screen with **External** user type, and **publish the app to
Production**.

That last step matters. An app left in Testing issues refresh tokens that expire
after ~7 days, which would silently break the unattended daily run. Publishing
does not require Google's formal verification review for a solo personal-use app
— it just means clicking through an "unverified app" warning once, during the
consent flow below. You do not need to add yourself as a test user; as project
owner you already have access and the console rejects the attempt as ineligible.

Create an OAuth 2.0 client of type **Desktop app**, download its JSON, then run
locally:

```bash
pip install -r requirements.txt
python scripts/get_youtube_refresh_token.py --client-secrets client_secret.json
```

### 3. Confirm the BHL field mapping

BHL's JSON field names vary between methods, and the published schema was not
reachable from the build environment, so every field read goes through an alias
list in `botanical_shorts/bhl.py`. Run this once against the live API:

```bash
BHL_API_KEY=... python -m botanical_shorts.cli verify-bhl
```

It reports which logical fields resolved, which page types your subjects
actually return, and exits non-zero listing anything unresolved — add the real
key names to `FIELD_ALIASES` if so. **Do this before the first real run**; it's
the one place where a wrong guess would quietly degrade selection.

## Usage

```bash
# Full daily run
python -m botanical_shorts.cli run

# Select and frame only, no render or upload
python -m botanical_shorts.cli run --dry-run

# Render but don't upload -- inspect build/ before committing to a channel post
python -m botanical_shorts.cli run --skip-upload

# Render one plate under all three letterbox treatments, side by side
python -m botanical_shorts.cli preview path/to/plate.jpg
```

The workflow runs daily at 13:10 UTC and is also dispatchable by hand with
`dry_run` / `skip_upload` toggles.

## State

`state/history.json` records every published plate and is committed back by the
workflow. The runner is ephemeral, so that commit is the only durable record of
what's been published — it's what stops the channel repeating itself. The
dedupe is per-volume as well as per-page, so the channel moves between works
rather than walking one book plate by plate.

## Tests

```bash
pip install pytest && python -m pytest tests/ -q
```

Covers the licence gate (including that non-commercial and no-derivatives
plates are refused even when an allowlist entry would otherwise match), the
BHL field-alias tolerance, paper-tone sampling, framing geometry, the vision
gates in both caption modes, and history round-tripping.

## Known gaps

- **Landscape plates are skipped.** Fitted into 9:16 without cropping, a wide
  plate occupies a thin band mid-frame and reads as a tiny picture adrift in
  paper. `image.max_source_aspect` (1.25) skips them rather than publishing a
  weak frame. Most bound plates are portrait, so this costs little pool, but if
  runs start failing to find a candidate this is the first knob to loosen.
- **Vision quality scoring is unvalidated.** `min_scan_quality: 7` is a
  starting guess. Watch the logged scores over the first couple of weeks and
  adjust.
- **Subject strings are unverified.** The three configured subjects are
  plausible BHL subject headings, not confirmed ones — `verify-bhl` will tell
  you whether they return titles.
