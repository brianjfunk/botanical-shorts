"""Per-item rights/licence gate.

BHL's holdings are a mix of Public Domain, CC0 and various CC licences, and the
status is declared per item -- never assume it across the library. This module
is deliberately allowlist-based: anything the configured allowlist does not
positively match is rejected, so an unrecognised or newly-introduced rights
string fails closed rather than quietly publishing a restricted plate.
"""

from __future__ import annotations

from dataclasses import dataclass

from .bhl import PageCandidate
from .config import LicenseConfig

# Rights strings that carry a share-alike or attribution obligation we would
# have to honour on-frame. Since the channel's defining constraint is no
# overlaid text, these are refused outright even if an allowlist entry would
# otherwise match them.
BLOCKED_MARKERS = ("nc", "non-commercial", "noncommercial", "nd", "no derivative", "in copyright")


@dataclass(frozen=True)
class LicenseVerdict:
    allowed: bool
    reason: str
    matched: str = ""

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return self.allowed


def _norm(value: str) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").replace("-", " ").split())


def _has_blocked_marker(text: str) -> bool:
    tokens = set(text.split())
    for marker in BLOCKED_MARKERS:
        if " " in marker:
            if marker in text:
                return True
        elif marker in tokens:
            return True
    return False


def evaluate(candidate: PageCandidate, cfg: LicenseConfig) -> LicenseVerdict:
    """Decide whether a candidate's declared rights permit reuse here."""
    rights = _norm(candidate.rights)
    license_name = _norm(candidate.license_name)
    license_url = _norm(candidate.license_url)
    haystack = " ".join(x for x in (rights, license_name, license_url) if x)

    if not haystack:
        if cfg.allow_unknown:
            return LicenseVerdict(True, "no rights metadata; allow_unknown is set")
        return LicenseVerdict(False, "BHL reported no rights or licence metadata")

    if _has_blocked_marker(haystack):
        return LicenseVerdict(
            False, f"rights carry a restrictive marker: {candidate.rights or candidate.license_name!r}"
        )

    for allowed in cfg.allowed_rights:
        if _norm(allowed) and _norm(allowed) in rights:
            return LicenseVerdict(True, f"rights status {candidate.rights!r}", matched=allowed)

    for allowed in cfg.allowed_licenses:
        needle = _norm(allowed)
        if needle and (needle in license_name or needle in license_url):
            return LicenseVerdict(True, f"licence {candidate.license_name!r}", matched=allowed)

    return LicenseVerdict(
        False,
        f"rights {candidate.rights!r} / licence {candidate.license_name!r} not on the allowlist",
    )
