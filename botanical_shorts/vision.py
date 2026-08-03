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
- is_illustration (boolean): is this page primarily a pictorial plate, rather \
than a page of body text, a title page, an index, or a blank?
- subject_summary (string, max 8 words): the subject itself as a short noun \
phrase. Name it directly -- do NOT begin with "Botanical illustration of", "An \
illustration of", "A drawing of" or similar. If a name is legible on the plate, \
prefer it. Good: "Lupinus polyphyllus". "Purple-flowered lupine". Bad: "Botanical \
illustration of a lupine plant with purple flowers".
- issues (array of short strings): specific defects you observed, empty if none.

Respond with ONLY a JSON object with keys: scan_quality, caption_embedded, \
species_name_visible, is_illustration, subject_summary, issues."""


@dataclass
class VisionVerdict:
    scan_quality: int
    caption_embedded: bool
    # Narrower than caption_embedded: the plate names the organism, rather than
    # merely carrying a plate number or an engraver's imprint. Recorded only --
    # never gated on, since many fine plates put the name on a facing page.
    species_name_visible: bool
    is_illustration: bool
    subject_summary: str
    issues: list[str]
    raw: str = ""
    error: str = ""

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


def inspect_plate(client, img: Image.Image, *, model: str) -> VisionVerdict:
    """Run both vision judgements on one candidate plate."""
    encoded, media_type = _encode(img)
    try:
        message = client.messages.create(
            model=model,
            max_tokens=512,
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
        data = _parse(raw)
    except Exception as exc:
        if getattr(exc, "status_code", None) in (401, 403):
            raise VisionAuthError(
                "Anthropic rejected the API key (HTTP "
                f"{getattr(exc, 'status_code', '?')}). Check the ANTHROPIC_API_KEY "
                "secret; no amount of retrying will help."
            ) from exc
        log.warning("vision inspection failed: %s", exc)
        return VisionVerdict(
            scan_quality=0,
            caption_embedded=False,
            species_name_visible=False,
            is_illustration=False,
            subject_summary="",
            issues=[],
            error=str(exc),
        )

    return VisionVerdict(
        scan_quality=int(data.get("scan_quality") or 0),
        caption_embedded=bool(data.get("caption_embedded")),
        species_name_visible=bool(data.get("species_name_visible")),
        is_illustration=bool(data.get("is_illustration")),
        subject_summary=str(data.get("subject_summary") or "").strip(),
        issues=[str(i) for i in (data.get("issues") or [])],
        raw=raw,
    )


def passes(verdict: VisionVerdict, *, min_quality: int, caption_mode: str) -> tuple[bool, str]:
    """Apply the configured gates to a verdict.

    Returns ``(accepted, reason)``.
    """
    if verdict.error:
        return False, f"vision check errored: {verdict.error}"
    if not verdict.is_illustration:
        return False, "not a pictorial plate"
    if verdict.scan_quality < min_quality:
        issues = "; ".join(verdict.issues) or "no detail given"
        return False, f"scan quality {verdict.scan_quality} < {min_quality} ({issues})"
    if caption_mode == "hard_gate" and not verdict.caption_embedded:
        return False, "no plate-embedded lettering"
    return True, "accepted"
