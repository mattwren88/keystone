#!/usr/bin/env python3
"""Fetch labeled Gmail messages and save as .eml files."""

from __future__ import annotations

import base64
import os
import re
from email import policy
from email.header import decode_header
from email.message import Message
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
from pathlib import Path

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
TOKEN_URI = "https://oauth2.googleapis.com/token"


def decode_subject(value: str) -> str:
    parts = decode_header(value or "")
    decoded = []
    for part, encoding in parts:
        if isinstance(part, bytes):
            decoded.append(part.decode(encoding or "utf-8", errors="replace"))
        else:
            decoded.append(part)
    return "".join(decoded).strip()


def sanitize_filename(value: str) -> str:
    value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return value[:80] or "email"


def parse_message(raw_bytes: bytes) -> Message:
    return BytesParser(policy=policy.default).parsebytes(raw_bytes)


def get_label_id(service, label_name: str) -> str | None:
    labels = service.users().labels().list(userId="me").execute().get("labels", [])
    for label in labels:
        if label.get("name", "").lower() == label_name.lower():
            return label.get("id")
    return None


def build_service() -> object:
    client_id = os.environ.get("GMAIL_CLIENT_ID")
    client_secret = os.environ.get("GMAIL_CLIENT_SECRET")
    refresh_token = os.environ.get("GMAIL_REFRESH_TOKEN")
    if not client_id or not client_secret or not refresh_token:
        raise RuntimeError("Missing Gmail OAuth environment variables.")

    creds = Credentials(
        None,
        refresh_token=refresh_token,
        token_uri=TOKEN_URI,
        client_id=client_id,
        client_secret=client_secret,
        scopes=SCOPES,
    )
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def main() -> None:
    label_name = os.environ.get("GMAIL_LABEL", "Keystone College")
    out_dir = Path(os.environ.get("EMAIL_DIR", "emails"))
    max_results = int(os.environ.get("MAX_RESULTS", "50"))
    out_dir.mkdir(parents=True, exist_ok=True)

    service = build_service()
    label_id = get_label_id(service, label_name)
    if not label_id:
        raise RuntimeError(f"Label not found: {label_name}")

    messages: list[dict] = []
    request = service.users().messages().list(
        userId="me", labelIds=[label_id], maxResults=max_results
    )
    while request and len(messages) < max_results:
        response = request.execute()
        messages.extend(response.get("messages", []))
        request = service.users().messages().list_next(request, response)

    for msg in messages[:max_results]:
        msg_id = msg.get("id")
        if not msg_id:
            continue
        if any(path.name.endswith(f"_{msg_id}.eml") for path in out_dir.glob("*.eml")):
            continue

        full = (
            service.users()
            .messages()
            .get(userId="me", id=msg_id, format="raw")
            .execute()
        )
        raw = base64.urlsafe_b64decode(full.get("raw", "").encode("utf-8"))
        message = parse_message(raw)
        subject = decode_subject(message.get("subject", ""))
        date = parsedate_to_datetime(message.get("date")) if message.get("date") else None
        date_stamp = date.strftime("%Y-%m-%d") if date else "unknown-date"

        filename = f"{date_stamp}_{sanitize_filename(subject)}_{msg_id}.eml"
        (out_dir / filename).write_bytes(raw)


if __name__ == "__main__":
    main()
