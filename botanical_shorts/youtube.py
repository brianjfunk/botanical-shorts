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

# Informational only -- Google enforces whatever the refresh token was granted,
# not what is listed here. Kept in step with scripts/get_youtube_refresh_token.py
# so the two do not drift and mislead.
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
]
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
        detail = str(exc)
        if "invalid_grant" in detail:
            hint = (
                "the refresh token is no longer valid. In order of likelihood:\n"
                "  1. The grant was revoked at "
                "https://myaccount.google.com/permissions. Revoking kills every "
                "token issued under it, including one you have just minted but "
                "not yet pasted into the secret -- do not revoke as part of "
                "re-consenting.\n"
                "  2. You re-ran the consent flow but did not paste the new "
                "token into the YOUTUBE_REFRESH_TOKEN secret. Completing the "
                "browser flow does not update it.\n"
                "  3. The OAuth app is still in Testing mode, whose tokens expire "
                "after ~7 days; publish it to Production."
            )
        else:
            hint = "the token exchange was rejected."
        raise UploadError(
            f"could not refresh the YouTube access token: {hint}\n({detail})"
        ) from exc
    return creds


def scheduled_publish_time(delay_hours: int, *, now: datetime | None = None) -> str | None:
    """When the upload should go public, or ``None`` for never.

    Zero (or less) means no schedule at all: the video is uploaded private and
    stays private until published by hand. That is the right default for a
    reviewed batch -- a passive veto window makes sense for one plate a day,
    but a batch of fifteen sharing one deadline would all go public at the same
    moment, and the review pass has already supplied the judgement the veto
    window existed to allow time for.
    """
    if delay_hours <= 0:
        return None
    now = now or datetime.now(timezone.utc)
    return (now + timedelta(hours=delay_hours)).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def check_credentials(client_id: str, client_secret: str, refresh_token: str) -> dict[str, str]:
    """Exchange the refresh token for an access token, without uploading.

    This is the preflight that catches the failure mode the spec called out:
    an OAuth app left in Testing issues refresh tokens that expire after about
    seven days, so the pipeline works for a week and then starts failing at
    3am with no one watching. Running this on every verify catches it while
    it is still a fixable inconvenience.
    """
    creds = build_credentials(client_id, client_secret, refresh_token)
    info: dict[str, str] = {
        "refresh": "ok",
        "expiry": creds.expiry.isoformat() if creds.expiry else "unknown",
        "scopes": ",".join(creds.scopes or []),
    }

    # Identifying the channel needs a broader scope than youtube.upload, so a
    # failure here is a scope limitation, not an auth problem. The refresh
    # above is the signal that matters.
    try:
        youtube = build("youtube", "v3", credentials=creds, cache_discovery=False)
        resp = youtube.channels().list(part="snippet", mine=True).execute()
        items = resp.get("items") or []
        if items:
            info["channel"] = items[0]["snippet"]["title"]
            info["channel_id"] = items[0]["id"]
    except HttpError as exc:
        info["channel"] = f"not readable with this scope ({exc.status_code})"
    except Exception as exc:  # never fail the preflight on an optional lookup
        info["channel"] = f"lookup skipped ({type(exc).__name__})"

    return info


def describe_video(video_id: str, credentials: Credentials) -> dict[str, str]:
    """Report which channel a video landed on and how YouTube processed it.

    Answers the two questions that look identical from the outside when an
    upload "succeeds" but nothing appears in Studio: did it go to the intended
    channel, and did YouTube accept the file? A very short clip can be
    rejected or fail processing after the API has already returned an id.
    """
    youtube = build("youtube", "v3", credentials=credentials, cache_discovery=False)
    resp = (
        youtube.videos()
        .list(part="snippet,status,processingDetails,contentDetails", id=video_id)
        .execute()
    )
    items = resp.get("items") or []
    if not items:
        return {
            "found": "no",
            "note": (
                "No video with this id is visible to these credentials. That "
                "usually means it belongs to a different channel than the one "
                "this refresh token authorises."
            ),
        }

    v = items[0]
    snippet, status = v.get("snippet", {}), v.get("status", {})
    out = {
        "found": "yes",
        "channel_title": snippet.get("channelTitle", "?"),
        "channel_id": snippet.get("channelId", "?"),
        "title": snippet.get("title", "?"),
        "privacy": status.get("privacyStatus", "?"),
        "publish_at": status.get("publishAt", "(none)"),
        "upload_status": status.get("uploadStatus", "?"),
        "duration": v.get("contentDetails", {}).get("duration", "?"),
        "processing": v.get("processingDetails", {}).get("processingStatus", "?"),
    }
    for key, field_name in (
        ("failure_reason", "failureReason"),
        ("rejection_reason", "rejectionReason"),
    ):
        if status.get(field_name):
            out[key] = status[field_name]
    return out


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
