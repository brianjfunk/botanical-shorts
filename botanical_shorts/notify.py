"""Daily review notification.

The whole review gate is one glance, so the message has to carry everything
needed to make that glance sufficient: the framed thumbnail, the citation, the
scheduled publish time, and a direct Studio link to pull it.

Slack is used when ``SLACK_WEBHOOK_URL`` is set; otherwise the summary is
written to the GitHub Actions job summary so the run page itself is the notice.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import requests

log = logging.getLogger(__name__)


def _blocks(summary: dict) -> list[dict]:
    citation = summary.get("citation", "")
    publish_at = summary.get("publish_at") or "not scheduled"
    studio_url = summary.get("studio_url", "")
    watch_url = summary.get("url", "")
    thumb = summary.get("thumb_url", "")
    quality = summary.get("scan_quality")
    caption = summary.get("caption_embedded")

    fields = [
        {"type": "mrkdwn", "text": f"*Publishes*\n{publish_at}"},
        {"type": "mrkdwn", "text": f"*Scan quality*\n{quality}/10"},
    ]
    if caption is not None:
        fields.append(
            {"type": "mrkdwn", "text": f"*Caption on plate*\n{'yes' if caption else 'no'}"}
        )

    section: dict = {
        "type": "section",
        "text": {"type": "mrkdwn", "text": f"*{summary.get('title', 'Untitled')}*\n{citation}"},
        "fields": fields,
    }
    if thumb:
        section["accessory"] = {"type": "image", "image_url": thumb, "alt_text": "plate thumbnail"}

    blocks: list[dict] = [
        {"type": "header", "text": {"type": "plain_text", "text": "Today's plate"}},
        section,
    ]

    links = " · ".join(
        part
        for part in (
            f"<{studio_url}|Review in Studio>" if studio_url else "",
            f"<{watch_url}|Watch>" if watch_url else "",
            f"<{summary.get('page_url')}|BHL page>" if summary.get("page_url") else "",
        )
        if part
    )
    if links:
        blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": links}]})
    blocks.append(
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": "Auto-publishes on schedule unless pulled in Studio."}
            ],
        }
    )
    return blocks


def send_slack(summary: dict, webhook_url: str, *, timeout: int = 20) -> bool:
    payload = {
        "text": f"Today's plate: {summary.get('title', 'Untitled')} — publishes {summary.get('publish_at', 'TBD')}",
        "blocks": _blocks(summary),
    }
    try:
        resp = requests.post(webhook_url, json=payload, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("Slack notification failed: %s", exc)
        return False
    return True


def write_job_summary(summary: dict) -> None:
    """Append a human-readable summary to the GitHub Actions run page."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    lines = [
        f"## {summary.get('title', 'Untitled')}",
        "",
        f"- **Publishes:** {summary.get('publish_at') or 'not scheduled'}",
        f"- **Scan quality:** {summary.get('scan_quality')}/10",
        f"- **Caption on plate:** {summary.get('caption_embedded')}",
        f"- **Source:** {summary.get('citation', '')}",
    ]
    if summary.get("studio_url"):
        lines.append(f"- **Review:** {summary['studio_url']}")
    if summary.get("page_url"):
        lines.append(f"- **BHL page:** {summary['page_url']}")
    if summary.get("thumb_url"):
        lines.append("")
        lines.append(f"![plate]({summary['thumb_url']})")
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def notify(summary: dict, *, enabled: bool = True) -> None:
    write_job_summary(summary)
    if not enabled:
        return
    webhook = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
    if webhook:
        if send_slack(summary, webhook):
            log.info("Slack notification sent")
            return
    log.info("no Slack webhook configured; summary:\n%s", json.dumps(summary, indent=2))


def save_summary(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
