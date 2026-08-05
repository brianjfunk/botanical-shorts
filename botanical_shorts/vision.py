"""Claude vision checks on candidate plates.

Two independent judgements, deliberately kept separate because they carry
different authority in v1:

* **Scan quality** -- a real gate. Damage, heavy foxing, bleed-through from the
  reverse page, a plate cropped at the scan edge: these are rejected outright,
  because a bad scan is exactly what Brian's visual review would catch and the
  point is that review takes seconds.
* **Caption embedded** -- recorded but *not* enforced while ``caption_mode`` is
  ``log_only``. This lets the filter's accuracy be validated against real
  output before it starts silently rejecting candidates. Flip to ``hard_gate``
  in config once the logged verdicts look trustworthy.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re
from dataclasses import dataclass, asdict

from PIL import Image

log = logging.getLogger(__name__)

# Downscale before sending: the judgements are about gross scan condition and
# the presence of lettering, neither of which needs full plate resolution.
VISION_MAX_EDGE = 1200


class VisionParseError(RuntimeError):
    """The model answered, but not with usable JSON.

    Distinct from a transport failure: the API is up and responding, so this is
    a property of one response rather than evidence of an outage. Callers that
    stop a run after repeated *transport* failures must not stop on these, or a
    handful of malformed answers ends a batch that was otherwise fine.
    """


class VisionAuthError(RuntimeError):
    """The Anthropic credential was rejected.

    Kept distinct from an ordinary vision failure because it is not a property
    of the candidate: retrying on the next plate cannot help, and treating it
    as a per-candidate rejection burns the whole call budget re-proving the
    same broken credential before reporting a misleading "no publishable plate
    found".
    """

PROMPT = """You are inspecting a scanned page from a historical natural history \
book, digitised by the Biodiversity Heritage Library. It is intended for use as a \
still image in a short video, presented exactly as scanned with no added text.

Assess two things independently.

1. scan_quality (integer 1-10): the physical and photographic condition of THIS \
scan. Lower the score for tears, stains, heavy foxing, text or images bleeding \
through from the reverse side, severe skew, blur, glare, fingers or scanning \
furniture in frame, or an illustration cut off at the edge of the scan. Do NOT \
lower it for the paper simply being aged, toned or yellowed -- that is expected \
and desirable. A clean, complete, sharp plate on toned paper scores 9-10.

2. caption_embedded (boolean): whether ANY printed or hand-lettered text appears ON \
the plate itself as part of the original engraving or lithograph -- including a \
plate number or an artist/publisher/engraver imprint line. Letterpress body text \
on a facing or separate page does not count; it must be part of the plate.

Also report:
- species_name_visible (boolean): whether the plate itself carries the NAME OF THE \
PLANT OR ORGANISM depicted -- a botanical binomial or a common name. This is \
narrower than caption_embedded: a plate number ("1217") or an imprint line \
("Pub. by J. Ridgway") is lettering but is NOT a name. Answer false when the only \
lettering is a number, an imprint, or an engraver credit.
- is_illustration (boolean): is this page primarily a finished pictorial plate? \
Answer false for a page of body text, an index or a blank; for a rough sketchbook \
or notebook page carrying pencil studies and handwritten annotations; and for any \
capture that includes scanning furniture -- a ruler or measuring scale, a colour \
calibration bar or greyscale target, or a library stamp laid beside the page. Also \
answer false for a title page even when it is decorated with drawn flowers or \
ornament: if the lettering is the dominant element of the page, it is a title page \
and not a plate.
- depicts_organism (boolean): is the PRINCIPAL SUBJECT of this plate one or \
more living things -- a plant, animal, fungus, alga or their parts, whether \
whole, dissected or microscopic? Answer false for a portrait or figure study of \
a PERSON (a frontispiece portrait of the author is a finished plate and still \
does not belong), a map or chart, a landscape or architectural view with no \
specimen as its subject, apparatus or equipment, a purely decorative border or \
ornament, and a page of geological strata or bare mineral specimens. A person \
shown incidentally beside a specimen for scale does not make this false.
- is_spread (boolean): does this capture show TWO facing pages photographed \
together, with a gutter or fold running between them -- for example an \
illustration on one side and a page of letterpress text on the other? Answer true \
only for a genuine two-page capture. A single plate is false, however wide it is.
- illustration_side (string): only meaningful when is_spread is true. Which half \
of the capture carries the pictorial plate -- answer "left" or "right" when one \
side holds the illustration and the other is letterpress text, a blank leaf or a \
manuscript page. Answer "both" when both sides are pictorial, and "neither" when \
neither is. Answer "" when is_spread is false.
- subject_summary (string, max 8 words): the subject itself as a short noun \
phrase. Name it directly -- do NOT begin with "Botanical illustration of", "An \
illustration of", "A drawing of" or similar. If a name is legible on the plate, \
prefer it. Good: "Lupinus polyphyllus". "Purple-flowered lupine". Bad: "Botanical \
illustration of a lupine plant with purple flowers".
- issues (array of short strings): specific defects you observed, empty if none.
Keep each under six words, and list at most three.

Respond with ONLY a JSON object with keys: scan_quality, caption_embedded, \
species_name_visible, is_illustration, depicts_organism, is_spread, \
illustration_side, subject_summary, issues."""


@dataclass
class VisionVerdict:
    scan_quality: int
    caption_embedded: bool
    # Narrower than caption_embedded: the plate names the organism, rather than
    # merely carrying a plate number or an engraver's imprint. Recorded only --
    # never gated on, since many fine plates put the name on a facing page.
    species_name_visible: bool
    is_illustration: bool
    # Two facing pages captured together. Framed whole, the plate shares the
    # screen with a page of letterpress and a fold down the middle; the aspect
    # gate only catches spreads wide enough to trip it, which a spread of two
    # tall narrow pages is not.
    is_spread: bool
    subject_summary: str
    issues: list[str]
    # Which half of a two-page capture holds the plate: "left", "right",
    # "both", "neither", or "" when this is not a spread. Defaulted, and
    # placed after the required fields, so the positional constructors used
    # throughout the tests keep working.
    illustration_side: str = ""
    # Narrower than is_illustration, and the distinction matters: an engraved
    # frontispiece portrait of the author is a finished pictorial plate by any
    # reading, and has no business on a natural history channel. Found by Brian
    # in the aspect audit, whose portrait band had deliberately been left
    # un-inspected. Defaults true so a verdict recorded before this field
    # existed is not retroactively rejected.
    depicts_organism: bool = True
    raw: str = ""
    error: str = ""
    # True when the failure was the API itself rather than its answer. Only
    # these count toward a caller's outage breaker.
    error_is_transport: bool = False

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("raw", None)
        return data


def _encode(img: Image.Image) -> tuple[str, str]:
    work = img.convert("RGB")
    longest = max(work.size)
    if longest > VISION_MAX_EDGE:
        scale = VISION_MAX_EDGE / longest
        work = work.resize(
            (max(1, int(work.width * scale)), max(1, int(work.height * scale))), Image.LANCZOS
        )
    buf = io.BytesIO()
    work.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii"), "image/jpeg"


def _parse(text: str) -> dict:
    """Pull the JSON object out of a model response."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise


def inspect_plate(client, img: Image.Image, *, model: str, attempts: int = 2) -> VisionVerdict:
    """Run both vision judgements on one candidate plate.

    Retried once by default. The failures worth retrying are not properties of
    the plate: a 429 or 529 from the API, or a response that came back empty
    and blew up in :func:`_parse`. Treating either as a verdict rejects a
    perfectly good plate and -- worse, since the caller meters vision calls --
    spends budget proving nothing. An auth failure is the exception and is
    raised immediately, because no retry can fix a bad key.
    """
    encoded, media_type = _encode(img)
    raw = ""
    last: Exception | None = None

    for attempt in range(max(1, attempts)):
        try:
            message = client.messages.create(
                model=model,
                # 512 truncated real responses once illustration_side was
                # added -- the JSON arrived cut mid-string, failed to parse,
                # read as an API failure and tripped the caller's outage
                # breaker. The ceiling is not the cost here: the verdict is a
                # short object, and paying for headroom beats losing a batch.
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": encoded,
                                },
                            },
                            {"type": "text", "text": PROMPT},
                        ],
                    }
                ],
            )
            raw = "".join(block.text for block in message.content if block.type == "text")
            try:
                data = _parse(raw)
            except (json.JSONDecodeError, ValueError) as exc:
                raise VisionParseError(f"{exc} (response was {len(raw)} chars)") from exc
            break
        except Exception as exc:
            if getattr(exc, "status_code", None) in (401, 403):
                raise VisionAuthError(
                    "Anthropic rejected the API key (HTTP "
                    f"{getattr(exc, 'status_code', '?')}). Check the ANTHROPIC_API_KEY "
                    "secret; no amount of retrying will help."
                ) from exc
            last = exc
            log.warning(
                "vision inspection failed (attempt %d of %d): %s", attempt + 1, attempts, exc
            )
    else:
        return VisionVerdict(
            scan_quality=0,
            caption_embedded=False,
            species_name_visible=False,
            is_illustration=False,
            is_spread=False,
            subject_summary="",
            issues=[],
            error=str(last),
            # Recorded separately from the message because the caller reacts
            # differently: repeated transport failures mean the API is down and
            # the run should stop, while a malformed answer is one bad response
            # from a working API and should only cost this candidate.
            error_is_transport=not isinstance(last, VisionParseError),
        )

    return VisionVerdict(
        scan_quality=int(data.get("scan_quality") or 0),
        caption_embedded=bool(data.get("caption_embedded")),
        species_name_visible=bool(data.get("species_name_visible")),
        is_illustration=bool(data.get("is_illustration")),
        # Absent from an older response means "not judged", and the safe
        # reading of an unjudged plate is that it is fine -- the field is new,
        # and failing closed would reject every entry harvested before it.
        depicts_organism=bool(data.get("depicts_organism", True)),
        is_spread=bool(data.get("is_spread")),
        illustration_side=str(data.get("illustration_side") or "").strip().lower(),
        subject_summary=str(data.get("subject_summary") or "").strip(),
        issues=[str(i) for i in (data.get("issues") or [])],
        raw=raw,
    )


def passes(
    verdict: VisionVerdict,
    *,
    min_quality: int,
    caption_mode: str,
    allow_spread: bool = False,
) -> tuple[bool, str]:
    """Apply the configured gates to a verdict.

    Returns ``(accepted, reason)``.

    ``allow_spread`` is set by a caller that has already cut the capture at the
    fold and kept the illustrated half. The spread was the reason to reject the
    frame, and once the frame is one page that reason is spent -- so this is a
    statement about what the caller did, not a way to turn the gate off.
    """
    if verdict.error:
        return False, f"vision check errored: {verdict.error}"
    if not verdict.is_illustration:
        return False, "not a pictorial plate"
    if not verdict.depicts_organism:
        return False, "a plate, but not of an organism"
    if verdict.is_spread and not allow_spread:
        return False, "two facing pages captured together"
    if verdict.scan_quality < min_quality:
        issues = "; ".join(verdict.issues) or "no detail given"
        return False, f"scan quality {verdict.scan_quality} < {min_quality} ({issues})"
    if caption_mode == "hard_gate" and not verdict.caption_embedded:
        return False, "no plate-embedded lettering"
    return True, "accepted"
