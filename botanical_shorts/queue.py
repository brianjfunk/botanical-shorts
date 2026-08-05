"""The approved queue: what review produces, and what publishing consumes.

Review used to happen per batch, which put a person in the loop before every
upload. That is the wrong frequency. Reviewing is a queue-filling activity:
look at a hundred and fifty plates once, and the survivors publish on their own
for a month or more.

So the queue is the seam between the two halves. A harvest appends pending
entries; a review marks each approved or rejected and fixes the publish order;
publishing takes from the front and never asks anything of anyone. Refill when
it runs low.

Entries carry everything needed to publish without re-deriving it, because the
run that publishes is a different run on a different machine from the one that
harvested. The framed image is the exception: it is re-derived from the page,
because the same page and the same stored verdict always yield the same frame,
and carrying a megabyte of PNG per plate through the repository would not.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

log = logging.getLogger(__name__)

PENDING = "pending"
APPROVED = "approved"
REJECTED = "rejected"
PUBLISHED = "published"


class Queue:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.entries: list[dict[str, Any]] = []
        if self.path.exists():
            self.entries = json.loads(self.path.read_text())

    # -- reading ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.entries)

    def with_status(self, status: str) -> list[dict[str, Any]]:
        return [e for e in self.entries if e.get("status") == status]

    @property
    def page_ids(self) -> set[str]:
        """Every page the queue knows about, whatever its state.

        Passed to the harvest so a plate already waiting for review is not
        offered a second time.
        """
        return {str(e["page_id"]) for e in self.entries if e.get("page_id")}

    def next_approved(self, count: int) -> list[dict[str, Any]]:
        return self.with_status(APPROVED)[:count]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for entry in self.entries:
            status = str(entry.get("status", PENDING))
            out[status] = out.get(status, 0) + 1
        return out

    # -- writing ------------------------------------------------------------

    def add(self, entries: list[dict[str, Any]]) -> int:
        """Append newly harvested plates, skipping any page already present."""
        known = self.page_ids
        added = 0
        for entry in entries:
            if str(entry.get("page_id")) in known:
                continue
            entry.setdefault("status", PENDING)
            self.entries.append(entry)
            known.add(str(entry["page_id"]))
            added += 1
        return added

    def resolve(self, rejected_indices: set[int], *, order: list[dict[str, Any]]) -> tuple[int, int]:
        """Settle a review: mark rejections, approve the rest, fix the order.

        ``rejected_indices`` are positions in the pending list as the review
        page numbered them. ``order`` is the approved entries in the sequence
        they should publish in -- stored as an explicit rank so the order
        survives the file being rewritten.
        """
        pending = self.with_status(PENDING)
        for i, entry in enumerate(pending):
            entry["status"] = REJECTED if i in rejected_indices else APPROVED

        for rank, entry in enumerate(order):
            entry["publish_rank"] = rank

        # Approved entries sort by rank; everything else keeps its place.
        self.entries.sort(key=lambda e: (e.get("status") != APPROVED, e.get("publish_rank", 1 << 30)))
        return len(order), len(rejected_indices)

    def reconcile(self, history) -> int:
        """Mark as published anything history says is already live.

        History is written upload by upload and is the authority on what
        exists on YouTube; the queue is a plan. When a run dies partway -- the
        daily cap is the ordinary way -- the two disagree, and believing the
        queue would republish videos that are already up. Safe to run at any
        time, and cheap.
        """
        live = {
            str(e["page_id"]): str(e.get("video_id") or "")
            for e in history.entries
            if e.get("page_id") and e.get("video_id")
        }
        fixed = 0
        for entry in self.entries:
            page_id = str(entry.get("page_id"))
            if page_id in live and entry.get("status") != PUBLISHED:
                entry["status"] = PUBLISHED
                entry["video_id"] = live[page_id]
                fixed += 1
        return fixed

    def mark_published(self, page_id: str, video_id: str) -> None:
        for entry in self.entries:
            if str(entry.get("page_id")) == str(page_id):
                entry["status"] = PUBLISHED
                entry["video_id"] = video_id
                return

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, indent=2, ensure_ascii=False))


def parse_review_reply(reply: str) -> set[int]:
    """Turn the line from the review page into zero-based rejected indices.

    Accepts "approve all", "reject 2,5,9", or a bare "2,5,9". The page numbers
    from one so a mistyped code is visible against the captions rather than
    silently off by one.
    """
    raw = reply.strip().lower()
    if not raw or raw in {"approve all", "approve", "all", "none"}:
        return set()
    tokens = raw.replace("reject", " ").replace(",", " ").split()
    out: set[int] = set()
    for token in tokens:
        if not token.isdigit():
            raise ValueError(f"could not read {token!r} as a plate number")
        n = int(token)
        if n < 1:
            raise ValueError("plate numbers start at 1")
        out.add(n - 1)
    return out
