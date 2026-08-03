"""Published-plate history, so the channel never repeats itself.

Stored as JSON in the repo and committed back by the workflow -- the runner is
ephemeral, so the git history *is* the state store.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class History:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: list[dict[str, Any]] = []
        self._page_ids: set[str] = set()
        self._item_ids: set[str] = set()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text() or "[]")
        except json.JSONDecodeError:
            log.warning("history at %s is corrupt; starting empty", self.path)
            return
        if isinstance(data, dict):
            data = data.get("entries", [])
        self.entries = [e for e in data if isinstance(e, dict)]
        self._page_ids = {str(e.get("page_id")) for e in self.entries if e.get("page_id")}
        self._item_ids = {str(e.get("item_id")) for e in self.entries if e.get("item_id")}

    @property
    def page_ids(self) -> set[str]:
        return self._page_ids

    @property
    def item_ids(self) -> set[str]:
        return self._item_ids

    def has_page(self, page_id: str) -> bool:
        return str(page_id) in self._page_ids

    def has_item(self, item_id: str) -> bool:
        """Whether we've already published *any* plate from this volume.

        Used to spread the channel across different works rather than walking
        one book plate by plate.
        """
        return str(item_id) in self._item_ids

    def record(self, entry: dict[str, Any]) -> None:
        entry = dict(entry)
        entry.setdefault("published_at", datetime.now(timezone.utc).isoformat())
        self.entries.append(entry)
        if entry.get("page_id"):
            self._page_ids.add(str(entry["page_id"]))
        if entry.get("item_id"):
            self._item_ids.add(str(entry["item_id"]))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, indent=2, ensure_ascii=False) + "\n")
        log.info("history saved (%d entries) to %s", len(self.entries), self.path)
