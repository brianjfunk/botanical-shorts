"""YouTube Data API v3 upload.

Uploads private with ``publishAt`` set a configurable window out. That gives
Brian a *passive* veto: the video auto-publishes on schedule unless he pulls
it, so the daily steady state is doing nothing.

Auth is a long-lived refresh token held as a GitHub Actions secret. This only
stays long-lived if the Google Cloud OAuth app is published to **Production**;
apps left in Testing issue refresh tokens that expire after ~7 days and would
break the unattended run. See ``scripts/get_youtube_refresh_token.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

log = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
TOKEN_URI = "https://oauth2.googleapis.com/token"


class UploadError(RuntimeError):
    """The upload could not be completed."""


@dataclass(frozen=True)
class UploadResult:
    video_id: str
    url: str
    studio_url: str
    publish_at: str


def build_credentials(client_id: str, client_secret: str, refresh_token: str) -> Credentials:
    creds = Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri=TOKEN_URI,
        client_id=client_id,
        client_secret=client_secret,
        scopes=SCOPES,
    )
    try:
        creds.refresh(Request())
    except Exception as exc:
        raise UploadError(
            "could not refresh the YouTube access token. If the Google Cloud OAuth "
            "app is still in Testing mode its refresh tokens expire after ~7 days; "
            f"publish it to Production and re-run the consent flow. ({exc})"
        ) from exc
    return creds


def scheduled_publish_time(delay_hours: int, *, now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return (now + timedelta(hours=delay_hours)).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def upload_video(
    video_path: Path,
    *,
    title: str,
    description: str,
    tags: list[str],
    category_id: str,
    privacy_status: str,
    publish_at: str | None,
    made_for_kids: bool,
    credentials: Credentials,
) -> UploadResult:
    youtube = build("youtube", "v3", credentials=credentials, cache_discovery=False)

    status: dict[str, object] = {
        "privacyStatus": privacy_status,
        "selfDeclaredMadeForKids": made_for_kids,
    }
    # publishAt is only honoured on a private video; setting it otherwise is
    # rejected by the API.
    if publish_at and privacy_status == "private":
        status["publishAt"] = publish_at

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": category_id,
        },
        "status": status,
    }

    media = MediaFileUpload(str(video_path), mimetype="video/mp4", resumable=True, chunksize=-1)
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    try:
        response = request.execute()
    except HttpError as exc:
        raise UploadError(f"YouTube rejected the upload: {exc}") from exc

    video_id = response.get("id")
    if not video_id:
        raise UploadError(f"upload returned no video id: {response}")

    return UploadResult(
        video_id=video_id,
        url=f"https://www.youtube.com/watch?v={video_id}",
        studio_url=f"https://studio.youtube.com/video/{video_id}/edit",
        publish_at=publish_at or "",
    )
