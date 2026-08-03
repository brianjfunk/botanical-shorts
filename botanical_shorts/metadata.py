"""Video title and description.

All attribution lives here, in the description -- never on the frame. Putting a
credit on-frame would reintroduce exactly the modern typography the channel
rules out.
"""

from __future__ import annotations

from .bhl import PageCandidate
from .vision import VisionVerdict

YOUTUBE_TITLE_LIMIT = 100
YOUTUBE_DESC_LIMIT = 5000


def build_title(candidate: PageCandidate, verdict: VisionVerdict | None) -> str:
    """A short, plain title.

    The plate's own lettering carries the species name visually, so the title
    only needs to be a serviceable handle in search and the Studio list.
    """
    # The year alone is not a title, so pick the subject line first and only
    # then decorate it -- otherwise a missing vision summary yields "(1805)".
    stem = (verdict.subject_summary if verdict else "").strip().rstrip(".")
    if not stem:
        stem = candidate.title.strip()
    if not stem:
        stem = "Botanical illustration"

    title = f"{stem} ({candidate.year})" if candidate.year else stem
    if len(title) > YOUTUBE_TITLE_LIMIT:
        title = title[: YOUTUBE_TITLE_LIMIT - 1].rstrip() + "…"
    return title


def build_description(candidate: PageCandidate, verdict: VisionVerdict | None) -> str:
    lines: list[str] = []

    summary = (verdict.subject_summary if verdict else "").strip()
    if summary:
        lines.append(summary.rstrip(".") + ".")
        lines.append("")

    lines.append("Source")
    lines.append(candidate.citation())
    if candidate.page_url:
        lines.append(candidate.page_url)
    if candidate.source:
        lines.append(f"Digitised by {candidate.source}.")

    rights_bits = [b for b in (candidate.rights, candidate.license_name) if b]
    if rights_bits:
        lines.append("")
        lines.append("Rights: " + " / ".join(rights_bits))
        if candidate.license_url:
            lines.append(candidate.license_url)

    lines.append("")
    lines.append("Presented as scanned, with no added text or digital retouching.")

    text = "\n".join(lines)
    return text[:YOUTUBE_DESC_LIMIT]
