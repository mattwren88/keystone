#!/usr/bin/env python3
"""Generate a Gmail API refresh token for GitHub Actions secrets."""

import argparse
import json
import sys

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def load_client_info(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "installed" in payload:
        return payload["installed"]
    if "web" in payload:
        return payload["web"]
    raise ValueError("Unsupported OAuth client file format.")


def main():
    parser = argparse.ArgumentParser(
        description="Print Gmail OAuth secrets for GitHub Actions."
    )
    parser.add_argument(
        "--client-secrets",
        required=True,
        help="Path to OAuth client secrets JSON from Google Cloud",
    )
    args = parser.parse_args()

    try:
        client_info = load_client_info(args.client_secrets)
    except (OSError, ValueError) as exc:
        print(f"Failed to read client secrets: {exc}", file=sys.stderr)
        sys.exit(1)

    flow = InstalledAppFlow.from_client_secrets_file(args.client_secrets, SCOPES)
    # Force consent so we always receive a refresh token.
    creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")

    print("\nAdd these as GitHub Actions secrets:")
    print(f"GMAIL_CLIENT_ID={client_info['client_id']}")
    print(f"GMAIL_CLIENT_SECRET={client_info['client_secret']}")
    print(f"GMAIL_REFRESH_TOKEN={creds.refresh_token}")


if __name__ == "__main__":
    main()
