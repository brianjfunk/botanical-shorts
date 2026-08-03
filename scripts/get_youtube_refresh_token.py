#!/usr/bin/env python3
"""One-time local OAuth consent flow to mint a YouTube refresh token.

Run this once on your own machine (not in CI), then store the printed refresh
token as the ``YOUTUBE_REFRESH_TOKEN`` GitHub Actions secret.

    python scripts/get_youtube_refresh_token.py --client-secrets client_secret.json

Before running, in the Google Cloud console:

1. Enable **YouTube Data API v3**.
2. Configure the OAuth consent screen with **External** user type.
3. **Publish the app to Production.** This matters: an app left in Testing
   issues refresh tokens that expire after ~7 days, which would break the
   unattended daily run. Publishing does not require Google's formal
   verification review for a solo personal-use app -- it just means you click
   through an "unverified app" warning once, here.
4. Create an OAuth 2.0 Client ID of type **Desktop app** and download its JSON.

If your Google account has more than one channel (a personal channel plus any
brand accounts), the consent screen shows a channel picker. **Pick the channel
you actually want to publish to** -- the refresh token is bound to that choice,
and an upload to the wrong channel still returns a perfectly valid video id.

You do not need to add yourself as a test user; as project owner you already
have access, and the console rejects the attempt as ineligible.
"""

from __future__ import annotations

import argparse
import sys

try:
    from google_auth_oauthlib.flow import InstalledAppFlow
except ImportError:  # pragma: no cover
    sys.exit("pip install google-auth-oauthlib first")

# youtube.upload alone is write-only: it can publish but cannot read back which
# channel it published to. Adding youtube.readonly lets the preflight name the
# channel before a batch runs -- the difference between catching a wrong-channel
# upload now and discovering it after 30 videos.
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]


AUTH_URI = "https://accounts.google.com/o/oauth2/auth"
TOKEN_URI = "https://oauth2.googleapis.com/token"


def _flow_from_id_and_secret(client_id: str, client_secret: str) -> InstalledAppFlow:
    """Build the flow from the two values directly.

    A hand-written client_secret.json is usually missing the ``installed``
    wrapper and the auth/token URIs that Google's download includes, and the
    resulting error does not say so. If you have the client id and secret --
    they are already in your Actions secrets -- this avoids the file entirely.
    """
    return InstalledAppFlow.from_client_config(
        {
            "installed": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": AUTH_URI,
                "token_uri": TOKEN_URI,
                "redirect_uris": ["http://localhost"],
            }
        },
        SCOPES,
    )


def _flow_from_file(path: str) -> InstalledAppFlow:
    """Load the downloaded client JSON, diagnosing a malformed file clearly."""
    import json
    import os

    if not os.path.exists(path):
        sys.exit(
            f"{path} not found.\n"
            "Download it from Google Cloud Console -> APIs & Services -> "
            "Credentials -> your Desktop app client -> download icon.\n"
            "Or skip the file: pass --client-id and --client-secret instead."
        )
    try:
        data = json.load(open(path))
    except json.JSONDecodeError as exc:
        sys.exit(f"{path} is not valid JSON: {exc}")

    root = data.get("installed") or data.get("web")
    if not root:
        sys.exit(
            f"{path} has no 'installed' or 'web' key, so it is not an OAuth "
            "client file -- a hand-written file usually looks like this.\n"
            "Either download the real one from Google Cloud Console, or pass "
            "--client-id and --client-secret instead."
        )
    missing = [k for k in ("client_id", "client_secret") if not root.get(k)]
    if missing:
        sys.exit(f"{path} is missing: {', '.join(missing)}")

    return InstalledAppFlow.from_client_secrets_file(path, SCOPES)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--client-secrets",
        default="client_secret.json",
        help="path to the Desktop-app OAuth client JSON downloaded from Google Cloud",
    )
    parser.add_argument(
        "--client-id",
        help="OAuth client id; use with --client-secret to skip the JSON file",
    )
    parser.add_argument(
        "--client-secret",
        help="OAuth client secret; use with --client-id to skip the JSON file",
    )
    parser.add_argument(
        "--console",
        action="store_true",
        help="use the console/out-of-band flow instead of a local browser",
    )
    args = parser.parse_args()

    if bool(args.client_id) != bool(args.client_secret):
        sys.exit("--client-id and --client-secret must be given together")

    if args.client_id:
        flow = _flow_from_id_and_secret(args.client_id, args.client_secret)
    else:
        flow = _flow_from_file(args.client_secrets)

    # access_type=offline + prompt=consent is what actually returns a refresh
    # token; without prompt=consent a re-authorisation returns only an access
    # token and the secret you store would be useless.
    kwargs = {"access_type": "offline", "prompt": "consent"}
    creds = (
        flow.run_console(**kwargs) if args.console else flow.run_local_server(port=0, **kwargs)
    )

    if not creds.refresh_token:
        print(
            "No refresh token returned. Revoke the app's access at "
            "https://myaccount.google.com/permissions and run this again.",
            file=sys.stderr,
        )
        return 1

    print("\nStore these as GitHub Actions secrets:\n")
    print(f"YOUTUBE_CLIENT_ID     = {creds.client_id}")
    print(f"YOUTUBE_CLIENT_SECRET = {creds.client_secret}")
    print(f"YOUTUBE_REFRESH_TOKEN = {creds.refresh_token}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
