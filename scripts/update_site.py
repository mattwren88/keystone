#!/usr/bin/env python3
"""Update index.html based on recent emails."""

from __future__ import annotations

import html
import os
import re
from dataclasses import dataclass
from datetime import datetime
from email import policy
from email.header import decode_header
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
from pathlib import Path

MONTHS = [
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
]


def format_date(dt: datetime) -> str:
    return f"{MONTHS[dt.month - 1]} {dt.day}, {dt.year}"


def decode_subject(value: str) -> str:
    parts = decode_header(value or "")
    decoded = []
    for part, encoding in parts:
        if isinstance(part, bytes):
            decoded.append(part.decode(encoding or "utf-8", errors="replace"))
        else:
            decoded.append(part)
    return "".join(decoded).strip()


def html_to_text(payload: str) -> str:
    payload = re.sub(r"<br\s*/?>", "\n", payload, flags=re.I)
    payload = re.sub(r"</p>", "\n", payload, flags=re.I)
    payload = re.sub(r"<[^>]+>", " ", payload)
    payload = html.unescape(payload)
    return payload


def extract_body(message) -> str:
    if message.is_multipart():
        for part in message.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                return part.get_content()
        for part in message.walk():
            if part.get_content_type() == "text/html":
                return html_to_text(part.get_content())
    else:
        if message.get_content_type() == "text/plain":
            return message.get_content()
        if message.get_content_type() == "text/html":
            return html_to_text(message.get_content())
    return ""


def normalize_lines(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip()
        if cleaned:
            lines.append(cleaned)
    return lines


def clean_item(line: str) -> str:
    line = re.sub(r"^[-*]\s*", "", line)
    line = re.sub(r"^\d+\.\s*", "", line)
    line = re.sub(r"^[a-zA-Z]\.?\s*", "", line)
    return line.strip()


def extract_date_from_text(text: str) -> datetime | None:
    mmddyy = re.search(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", text)
    if mmddyy:
        month, day, year = (int(mmddyy.group(1)), int(mmddyy.group(2)), int(mmddyy.group(3)))
        if year < 100:
            year += 2000
        try:
            return datetime(year, month, day)
        except ValueError:
            return None

    month_name = re.search(
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})",
        text,
        re.I,
    )
    if month_name:
        month = [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ].index(month_name.group(1).lower()) + 1
        day = int(month_name.group(2))
        year = datetime.now().year
        try:
            return datetime(year, month, day)
        except ValueError:
            return None
    return None


@dataclass
class ParsedEmail:
    subject: str
    date: datetime | None
    body: str
    lines: list[str]


def load_emails(folder: Path) -> list[ParsedEmail]:
    emails: list[ParsedEmail] = []
    for path in sorted(folder.glob("*.eml")):
        raw = path.read_bytes()
        message = BytesParser(policy=policy.default).parsebytes(raw)
        subject = decode_subject(message.get("subject", ""))
        date = parsedate_to_datetime(message.get("date")) if message.get("date") else None
        if date and date.tzinfo:
            date = date.replace(tzinfo=None)
        body = extract_body(message)
        lines = normalize_lines(body)
        emails.append(ParsedEmail(subject=subject, date=date, body=body, lines=lines))
    return emails


def extract_numbered_list(lines: list[str], header_regex: str) -> list[str]:
    items = []
    header_index = next((i for i, line in enumerate(lines) if re.search(header_regex, line, re.I)), None)
    if header_index is None:
        return items
    for line in lines[header_index + 1 :]:
        if not line:
            if items:
                break
            continue
        if re.match(r"^\d+\.", line):
            item = clean_item(line)
            if item and item not in items:
                items.append(item)
        elif items and not re.match(r"^[a-zA-Z]\.", line):
            break
    return items


def extract_lettered_list(lines: list[str], anchor_regex: str) -> list[str]:
    items = []
    anchor_index = next((i for i, line in enumerate(lines) if re.search(anchor_regex, line, re.I)), None)
    if anchor_index is None:
        return items
    for line in lines[anchor_index + 1 :]:
        if not line:
            if items:
                break
            continue
        if re.match(r"^[a-gA-G]\.", line):
            item = clean_item(line)
            if "http" in item.lower():
                continue
            if item and item not in items:
                items.append(item)
        elif items and not re.match(r"^[a-gA-G]\.", line):
            break
    return items


def extract_action_items(lines: list[str], keywords: list[str], limit: int = 4) -> list[str]:
    results = []
    for line in lines:
        lower = line.lower()
        if any(keyword in lower for keyword in keywords):
            cleaned = clean_item(line)
            if cleaned and cleaned not in results:
                results.append(cleaned)
        if len(results) >= limit:
            break
    return results


def extract_jazz_timing_details(text: str) -> list[str]:
    details = []
    if re.search(r"7\s*[-–]\s*8\s*pm\s*review", text, re.I):
        details.append("7:00-8:00 review set.")
    if re.search(r"8:?10\s*[-–]\s*9:?00\s*new", text, re.I):
        details.append("8:10-9:00 new chart reading (no vocalists first night).")
    if re.search(r"two new", text, re.I):
        details.append("Expect at least two new charts added this spring.")
    return details


def update_simple_tag(html_text: str, attr: str, new_text: str) -> str:
    pattern = re.compile(rf"(<[^>]*{attr}[^>]*>)([^<]*)(</[^>]+>)")
    return pattern.sub(rf"\1{new_text}\3", html_text, count=1)


def replace_list(html_text: str, list_name: str, items: list[str]) -> str:
    if not items:
        return html_text
    pattern = re.compile(rf"(<ul[^>]*data-list=\"{list_name}\"[^>]*>)(.*?)(</ul>)", re.S)
    match = pattern.search(html_text)
    if not match:
        return html_text
    body = match.group(2)
    indent_match = re.search(r"\n([ \t]*)<li", body)
    indent = indent_match.group(1) if indent_match else "              "
    list_markup = "\n" + "\n".join(f"{indent}<li>{html.escape(item)}</li>" for item in items) + "\n"
    return html_text[: match.start(2)] + list_markup + html_text[match.end(2) :]


def main() -> None:
    email_dir = Path(os.environ.get("EMAIL_DIR", "emails"))
    index_path = Path(os.environ.get("INDEX_PATH", "index.html"))

    if not email_dir.exists() or not index_path.exists():
        raise SystemExit("Missing emails directory or index.html")

    emails = load_emails(email_dir)
    if not emails:
        raise SystemExit("No emails found to process")

    latest_email = max((e for e in emails if e.date), key=lambda e: e.date, default=None)
    updated_stamp = format_date(latest_email.date) if latest_email and latest_email.date else format_date(datetime.now())

    symphonic_emails = [e for e in emails if "symphonic" in e.subject.lower()]
    jazz_emails = [e for e in emails if "jazz" in e.subject.lower()]

    symphonic_emails.sort(key=lambda e: e.date or datetime.min, reverse=True)
    jazz_emails.sort(key=lambda e: e.date or datetime.min, reverse=True)

    symphonic_week = None
    if symphonic_emails:
        sym_text = symphonic_emails[0].subject + " " + symphonic_emails[0].body
        symphonic_week = extract_date_from_text(sym_text) or symphonic_emails[0].date

    jazz_week = None
    if jazz_emails:
        jazz_text = jazz_emails[0].subject + " " + jazz_emails[0].body
        jazz_week = extract_date_from_text(jazz_text) or jazz_emails[0].date

    symphonic_pieces: list[str] = []
    symphonic_details: list[str] = []
    for email_item in symphonic_emails:
        symphonic_pieces = extract_numbered_list(
            email_item.lines, r"Symphonic Band Rehearsal Schedule"
        )
        if symphonic_pieces:
            break
    if symphonic_emails:
        symphonic_details = extract_action_items(
            symphonic_emails[0].lines,
            ["handbook", "arrive", "listen", "sign", "lesson", "practice", "new"],
        )

    jazz_pieces: list[str] = []
    for email_item in jazz_emails:
        jazz_pieces = extract_lettered_list(
            email_item.lines, r"keep these charts"  # spring email list
        )
        if jazz_pieces:
            break
    if not jazz_pieces:
        for email_item in jazz_emails:
            jazz_pieces = extract_numbered_list(email_item.lines, r"REVIEW")
            if jazz_pieces:
                break

    jazz_details: list[str] = []
    if jazz_emails:
        jazz_details.extend(extract_jazz_timing_details(jazz_emails[0].body))
        if len(jazz_details) < 4:
            jazz_details.extend(
                extract_action_items(
                    jazz_emails[0].lines,
                    ["chart", "listen", "sheet", "new", "review"],
                    limit=4 - len(jazz_details),
                )
            )

    html_text = index_path.read_text(encoding="utf-8")
    html_text = update_simple_tag(html_text, "data-updated", f"Updated: {updated_stamp}")

    if symphonic_week:
        html_text = update_simple_tag(
            html_text,
            'data-week="symphonic"',
            f"Week of {format_date(symphonic_week)}",
        )
    if jazz_week:
        html_text = update_simple_tag(
            html_text,
            'data-week="jazz"',
            f"Week of {format_date(jazz_week)}",
        )

    html_text = replace_list(html_text, "symphonic-pieces", symphonic_pieces)
    html_text = replace_list(html_text, "symphonic-details", symphonic_details)
    html_text = replace_list(html_text, "jazz-pieces", jazz_pieces)
    html_text = replace_list(html_text, "jazz-details", jazz_details)

    index_path.write_text(html_text, encoding="utf-8")


if __name__ == "__main__":
    main()
