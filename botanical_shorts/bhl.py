"""Biodiversity Heritage Library API v3 client.

Endpoint shape confirmed from the rOpenSci ``rbhl`` client:
``GET https://www.biodiversitylibrary.org/api3?op=<Method>&apikey=<key>&format=json``
returning ``{"Status": "ok", "Result": [...]}``.

BHL's JSON field naming is not perfectly stable across methods (and the
published schema was not reachable at build time), so every field read goes
through :func:`pick`, which tries a list of candidate key names
case-insensitively and returns the first non-empty hit. Run
``python -m botanical_shorts.cli verify-bhl`` against the live API to dump the
real response shape and confirm these mappings.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

import requests

log = logging.getLogger(__name__)

API_BASE = "https://www.biodiversitylibrary.org/api3"

# Full-resolution page scan. BHL serves these directly off the page id.
PAGE_IMAGE_URL = "https://www.biodiversitylibrary.org/pageimage/{page_id}"
PAGE_THUMB_URL = "https://www.biodiversitylibrary.org/pagethumb/{page_id}"
PAGE_VIEW_URL = "https://www.biodiversitylibrary.org/page/{page_id}"
ITEM_VIEW_URL = "https://www.biodiversitylibrary.org/item/{item_id}"

# Candidate key names per logical field, most likely first.
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "item_id": ("ItemID", "Id", "ItemId"),
    "title_id": ("TitleID", "TitleId"),
    "page_id": ("PageID", "PageId"),
    "page_types": ("PageTypes", "PageType"),
    "page_numbers": ("PageNumbers", "PageNumber"),
    "full_title": ("FullTitle", "Title", "ShortTitle", "TitleText"),
    "year": ("Year", "PublicationDate", "PublishDate", "Date"),
    "publisher": ("PublisherName", "Publisher", "PublisherPlace"),
    "authors": ("Authors",),
    "author_name": ("Name", "FullName", "AuthorName"),
    "rights": ("RightsStatus", "Rights", "CopyrightStatus"),
    "rights_statement": ("RightsStatement", "RightsHolder", "Copyright"),
    "license": ("LicenseName", "License", "LicenseUrl"),
    "license_url": ("LicenseUrl", "LicenseURL"),
    "source": ("Source", "SourceIdentifier", "Contributor"),
    "items": ("Items",),
    "pages": ("Pages",),
}


class BHLError(RuntimeError):
    """BHL API returned an error or an unusable response."""


def pick(record: dict[str, Any], logical: str, default: Any = None) -> Any:
    """Read a logical field from a BHL record, tolerating key-name drift."""
    if not isinstance(record, dict):
        return default
    aliases = FIELD_ALIASES.get(logical, (logical,))
    for alias in aliases:
        if alias in record and record[alias] not in (None, "", [], {}):
            return record[alias]
    # Case-insensitive second pass.
    lowered = {str(k).lower(): v for k, v in record.items()}
    for alias in aliases:
        value = lowered.get(alias.lower())
        if value not in (None, "", [], {}):
            return value
    return default


@dataclass
class PageCandidate:
    """One BHL scanned page that is a plausible illustration plate."""

    page_id: str
    item_id: str
    title_id: str
    title: str
    year: str
    publisher: str
    authors: list[str]
    page_types: list[str]
    rights: str
    license_name: str
    license_url: str
    source: str
    subject: str = ""
    extra: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def image_url(self) -> str:
        return PAGE_IMAGE_URL.format(page_id=self.page_id)

    @property
    def thumb_url(self) -> str:
        return PAGE_THUMB_URL.format(page_id=self.page_id)

    @property
    def page_url(self) -> str:
        return PAGE_VIEW_URL.format(page_id=self.page_id)

    @property
    def item_url(self) -> str:
        return ITEM_VIEW_URL.format(item_id=self.item_id)

    def citation(self) -> str:
        """Human-readable source credit for the video description."""
        bits = [self.title.strip()] if self.title else []
        if self.authors:
            bits.append(", ".join(self.authors[:2]))
        if self.year:
            bits.append(str(self.year))
        if self.publisher:
            bits.append(self.publisher)
        head = ". ".join(b for b in bits if b)
        return f"{head}. Biodiversity Heritage Library." if head else "Biodiversity Heritage Library."


class BHLClient:
    def __init__(
        self,
        api_key: str,
        *,
        session: requests.Session | None = None,
        timeout: int = 45,
        max_retries: int = 4,
    ) -> None:
        if not api_key:
            raise BHLError("a BHL API key is required (set BHL_API_KEY)")
        self.api_key = api_key
        self.session = session or requests.Session()
        self.session.headers.setdefault(
            "User-Agent", "botanical-shorts/1.0 (+https://github.com/brianjfunk/botanical-shorts)"
        )
        self.timeout = timeout
        self.max_retries = max_retries

    def call(self, op: str, **params: Any) -> Any:
        """Invoke one API method and return its ``Result`` payload."""
        query = {"op": op, "apikey": self.api_key, "format": "json"}
        query.update({k: v for k, v in params.items() if v is not None})

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(API_BASE, params=query, timeout=self.timeout)
                if resp.status_code >= 500 or resp.status_code == 429:
                    raise BHLError(f"{op}: HTTP {resp.status_code}")
                resp.raise_for_status()
                payload = resp.json()
            except (requests.RequestException, ValueError, BHLError) as exc:
                last_error = exc
                delay = 2 ** (attempt + 1)
                log.warning("BHL %s failed (%s); retrying in %ss", op, exc, delay)
                time.sleep(delay)
                continue

            status = str(payload.get("Status", "")).lower()
            if status and status not in {"ok", "success"}:
                raise BHLError(f"{op}: {payload.get('ErrorMessage') or payload}")
            return payload.get("Result")

        raise BHLError(f"{op}: giving up after {self.max_retries} attempts") from last_error

    # -- API methods -----------------------------------------------------

    def publication_search_advanced(
        self, *, subject: str, page: int = 1, **extra: Any
    ) -> list[dict[str, Any]]:
        result = self.call("PublicationSearchAdvanced", subject=subject, page=page, **extra)
        return _as_list(result)

    def get_title_metadata(self, title_id: str, *, items: bool = True) -> dict[str, Any]:
        result = self.call("GetTitleMetadata", id=title_id, items="t" if items else "f")
        records = _as_list(result)
        return records[0] if records else {}

    def get_item_metadata(self, item_id: str, *, pages: bool = True) -> dict[str, Any]:
        result = self.call("GetItemMetadata", id=item_id, pages="t" if pages else "f")
        records = _as_list(result)
        return records[0] if records else {}


def _as_list(result: Any) -> list[dict[str, Any]]:
    if result is None:
        return []
    if isinstance(result, dict):
        return [result]
    if isinstance(result, list):
        return [r for r in result if isinstance(r, dict)]
    return []


def _page_types(page: dict[str, Any]) -> list[str]:
    raw = pick(page, "page_types", []) or []
    if isinstance(raw, str):
        raw = [raw]
    out: list[str] = []
    for entry in raw:
        if isinstance(entry, dict):
            # Some BHL responses nest as [{"PageTypeName": "Illustration"}].
            value = entry.get("PageTypeName") or entry.get("Name") or entry.get("Type")
            if value:
                out.append(str(value))
        elif entry:
            out.append(str(entry))
    return out


def _authors(title_meta: dict[str, Any]) -> list[str]:
    raw = pick(title_meta, "authors", []) or []
    names: list[str] = []
    for entry in raw if isinstance(raw, list) else []:
        if isinstance(entry, dict):
            name = pick(entry, "author_name")
            if name:
                names.append(str(name).strip().rstrip(","))
        elif entry:
            names.append(str(entry))
    return names


def _year(value: Any) -> str:
    """Normalise BHL's assorted date strings to a bare 4-digit year."""
    text = str(value or "")
    for i in range(len(text) - 3):
        chunk = text[i : i + 4]
        if chunk.isdigit() and 1400 <= int(chunk) <= 2100:
            return chunk
    return ""


def iter_candidates(
    client: BHLClient,
    *,
    subjects: Iterable[str],
    page_types: Iterable[str],
    year_min: int,
    year_max: int,
    titles_per_subject: int,
    max_items_per_title: int,
    max_pages_per_item: int,
    limit: int,
) -> Iterator[PageCandidate]:
    """Walk subject -> title -> item -> page, yielding illustration plates.

    Yields lazily so callers can stop as soon as they have a usable plate,
    rather than paying for the whole traversal every run.
    """
    wanted_types = {t.lower() for t in page_types}
    emitted = 0

    for subject in subjects:
        try:
            titles = client.publication_search_advanced(subject=subject)
        except BHLError as exc:
            log.warning("subject %r search failed: %s", subject, exc)
            continue

        for title_rec in titles[:titles_per_subject]:
            if emitted >= limit:
                return
            title_id = str(pick(title_rec, "title_id") or "")
            if not title_id:
                continue

            year = _year(pick(title_rec, "year"))
            if year and not (year_min <= int(year) <= year_max):
                log.debug("title %s year %s outside window", title_id, year)
                continue

            try:
                title_meta = client.get_title_metadata(title_id)
            except BHLError as exc:
                log.warning("title %s metadata failed: %s", title_id, exc)
                continue

            year = year or _year(pick(title_meta, "year"))
            if year and not (year_min <= int(year) <= year_max):
                continue

            title_text = str(pick(title_meta, "full_title") or pick(title_rec, "full_title") or "")
            publisher = str(pick(title_meta, "publisher") or "")
            authors = _authors(title_meta)
            items = _as_list(pick(title_meta, "items", []))

            for item_rec in items[:max_items_per_title]:
                if emitted >= limit:
                    return
                item_id = str(pick(item_rec, "item_id") or "")
                if not item_id:
                    continue

                try:
                    item_meta = client.get_item_metadata(item_id)
                except BHLError as exc:
                    log.warning("item %s metadata failed: %s", item_id, exc)
                    continue

                rights = str(pick(item_meta, "rights") or "")
                license_name = str(pick(item_meta, "license") or "")
                license_url = str(pick(item_meta, "license_url") or "")
                source = str(pick(item_meta, "source") or "")
                pages = _as_list(pick(item_meta, "pages", []))
                kept_from_item = 0

                for page_rec in pages:
                    if emitted >= limit:
                        return
                    if kept_from_item >= max_pages_per_item:
                        break
                    ptypes = _page_types(page_rec)
                    if not any(pt.lower() in wanted_types for pt in ptypes):
                        continue
                    page_id = str(pick(page_rec, "page_id") or "")
                    if not page_id:
                        continue

                    kept_from_item += 1
                    emitted += 1
                    yield PageCandidate(
                        page_id=page_id,
                        item_id=item_id,
                        title_id=title_id,
                        title=title_text,
                        year=year,
                        publisher=publisher,
                        authors=authors,
                        page_types=ptypes,
                        rights=rights,
                        license_name=license_name,
                        license_url=license_url,
                        source=source,
                        subject=subject,
                        extra={"page": page_rec, "item": item_meta},
                    )


def download_page_image(
    candidate: PageCandidate,
    *,
    session: requests.Session | None = None,
    timeout: int = 90,
) -> bytes:
    """Fetch the full-resolution scan for a candidate page."""
    sess = session or requests.Session()
    resp = sess.get(candidate.image_url, timeout=timeout)
    resp.raise_for_status()
    data = resp.content
    if len(data) < 4096:
        raise BHLError(f"page {candidate.page_id}: image response too small ({len(data)}b)")
    return data
