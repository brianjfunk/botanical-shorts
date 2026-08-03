"""Per-item rights/licence gate.

BHL's holdings are a mix of Public Domain, CC0 and various CC licences, and the
status is declared per item -- never assume it across the library. This module
is deliberately allowlist-based: anything the configured allowlist does not
positively match is rejected, so an unrecognised or newly-introduced rights
string fails closed rather than quietly publishing a restricted plate.

There are two tracks, because BHL populates its fields differently by kind:

**Public domain** items carry their status only as free rights text
(``RightsStatus`` / ``Rights`` / ``CopyrightStatus``), e.g. *"Public domain. The
BHL considers this work no longer under copyright."* The licence fields are
empty for these. Clear public-domain language in the rights text is therefore
sufficient on its own -- requiring ``LicenseName``/``LicenseUrl`` here would
reject essentially every genuine PD item.

**CC-licensed** items are the opposite: the specific licence lives in
``LicenseName``/``LicenseUrl``, and that is where it must be read from. CC-ish
wording in the rights text alone is *not* accepted, because it does not say
which licence applies -- without that, an NC or ND obligation could pass
unnoticed. Absent licence fields fail closed.

The asymmetry that follows: a **public-domain status** is a status, not a
licence, and BHL declares it in whichever field its upstream source used.
Internet-Archive-sourced records in particular put a bare token
(``NOT_IN_COPYRIGHT``) in the licence field while leaving the rights text
empty. So the PD vocabulary is matched against every rights and licence field,
whereas the CC vocabulary is matched only against the licence fields. Matching
PD against the rights field alone silently rejects every IA-sourced item.
"""

from __future__ import annotations

from dataclasses import dataclass

from .bhl import PageCandidate
from .config import LicenseConfig

# Rights strings that carry a restriction we cannot honour. Since the channel's
# defining constraint is no overlaid text, an attribution-on-frame or
# no-derivatives obligation is refused outright even if an allowlist entry would
# otherwise match.
BLOCKED_MARKERS = ("nc", "non-commercial", "noncommercial", "nd", "no derivative", "in copyright")

# "in copyright" is the marker that matters most and also the one most likely to
# appear negated: BHL's own public-domain vocabulary includes "Not in copyright"
# and "no longer under copyright". Matching it blindly rejects exactly the items
# we most want.
NEGATIONS = ("not ", "no longer ", "never ", "no ")


@dataclass(frozen=True)
class LicenseVerdict:
    allowed: bool
    reason: str
    matched: str = ""

    def __bool__(self) -> bool:  # pragma: no cover - trivial
        return self.allowed


def _norm(value: str) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").replace("-", " ").split())


def _is_negated(text: str, at: int) -> bool:
    """Whether the marker starting at ``at`` is preceded by a negation."""
    prefix = text[:at]
    return any(prefix.endswith(neg) for neg in NEGATIONS)


def _has_blocked_marker(text: str) -> bool:
    for marker in BLOCKED_MARKERS:
        if " " in marker:
            start = text.find(marker)
            while start != -1:
                if not _is_negated(text, start):
                    return True
                start = text.find(marker, start + 1)
        elif marker in text.split():
            # Single-token markers ("nc", "nd") are only meaningful as whole
            # words, so they never match inside "Foundation" or "and".
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
        declared = candidate.rights or candidate.license_name or candidate.license_url
        return LicenseVerdict(False, f"rights carry a restrictive marker: {declared!r}")

    # Public-domain status: accepted from any field, because BHL mirrors
    # whichever field its upstream source populated.
    for allowed in cfg.allowed_rights:
        needle = _norm(allowed)
        if needle and needle in haystack:
            declared = candidate.rights or candidate.license_name or candidate.license_url
            return LicenseVerdict(True, f"public-domain status {declared!r}", matched=allowed)

    # CC licence identity: licence fields only. Rights prose naming Creative
    # Commons without naming the licence is not enough to rule out NC/ND.
    for allowed in cfg.allowed_licenses:
        needle = _norm(allowed)
        if needle and (needle in license_name or needle in license_url):
            return LicenseVerdict(True, f"licence {candidate.license_name!r}", matched=allowed)

    return LicenseVerdict(
        False,
        f"rights {candidate.rights!r} / licence {candidate.license_name!r} not on the allowlist",
    )
