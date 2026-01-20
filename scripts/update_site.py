#!/usr/bin/env python3
"""Update index.html based on recent emails."""

from __future__ import annotations

import html
import json
import os
import re
import urllib.error
import urllib.request
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

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_OPENAI_MODEL = "gpt-5-nano"
DEFAULT_OPENAI_TIMEOUT = 30


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
    line = re.sub(r"^[a-zA-Z][\.\)]\s*", "", line)
    return line.strip()


SIGNATURE_PATTERNS = [
    r"mr\.\s+jeffrey",
    r"director of bands",
    r"keystone\.edu",
    r"office:",
    r"one college green",
]

NOISE_PATTERNS = [
    r"^subject:",
    r"^from:",
    r"^to:",
    r"^cc:",
    r"^sent:",
    r"^delivered-to:",
    r"^reply$",
    r"^reply all$",
    r"^forward$",
]


def strip_signature(lines: list[str]) -> list[str]:
    for idx, line in enumerate(lines):
        if any(re.search(pattern, line, re.I) for pattern in SIGNATURE_PATTERNS):
            return lines[:idx]
    return lines


def is_noise_line(line: str) -> bool:
    lower = line.lower()
    if "http://" in lower or "https://" in lower or "mailto:" in lower:
        return True
    if any(re.search(pattern, line, re.I) for pattern in NOISE_PATTERNS):
        return True
    if lower.startswith("unsubscribe"):
        return True
    return False


def cleanup_lines(lines: list[str]) -> list[str]:
    trimmed = strip_signature(lines)
    return [line for line in trimmed if line and not is_noise_line(line)]


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
        lines = cleanup_lines(normalize_lines(body))
        emails.append(ParsedEmail(subject=subject, date=date, body=body, lines=lines))
    return emails


def clip_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    clipped = text[:limit].rsplit(" ", 1)[0]
    return f"{clipped}..."


def format_email_for_ai(email: ParsedEmail, limit: int = 3000) -> str:
    date = email.date.isoformat() if email.date else "unknown"
    content = "\n".join(email.lines)
    payload = f"Subject: {email.subject}\nDate: {date}\nContent:\n{content}"
    return clip_text(payload, limit)


def build_ai_context(symphonic_emails: list[ParsedEmail], jazz_emails: list[ParsedEmail]) -> str:
    def pack(label: str, emails: list[ParsedEmail]) -> str:
        if not emails:
            return f"{label} emails: none."
        chunks = [format_email_for_ai(email) for email in emails]
        return f"{label} emails:\n" + "\n\n".join(chunks)

    return f"{pack('Symphonic', symphonic_emails)}\n\n{pack('Jazz', jazz_emails)}"


def call_openai(prompt: str) -> dict | None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    model = os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    timeout = int(os.environ.get("OPENAI_TIMEOUT", str(DEFAULT_OPENAI_TIMEOUT)))
    payload = {
        "model": model,
        "temperature": 0.2,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You summarize band rehearsal emails into short lists. "
                    "Return ONLY valid JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    "From the emails, extract rehearsal info as JSON with this shape:\n"
                    "{\n"
                    '  "symphonic": {"pieces": [], "details": [], "notes": []},\n'
                    '  "jazz": {"pieces": [], "details": [], "notes": []}\n'
                    "}\n"
                    "Rules:\n"
                    "- Only use facts from the emails.\n"
                    "- Lists should be 1-4 short bullets, no numbering.\n"
                    "- Omit links, emails, and URLs.\n"
                    "- If info is missing, return an empty list.\n"
                    f"\nEmails:\n{prompt}"
                ),
            },
        ],
    }

    request = urllib.request.Request(
        OPENAI_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError):
        return None

    try:
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)
    except (KeyError, IndexError, TypeError, json.JSONDecodeError):
        return None


def normalize_ai_list(items: object, max_items: int = 4) -> list[str]:
    if not isinstance(items, list):
        return []
    cleaned_items = []
    for item in items:
        if not isinstance(item, str):
            continue
        cleaned = clean_item(item)
        if not cleaned or not is_valid_item(cleaned):
            continue
        if cleaned not in cleaned_items:
            cleaned_items.append(cleaned)
        if len(cleaned_items) >= max_items:
            break
    return cleaned_items


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


def extract_first_numbered_list(lines: list[str]) -> list[str]:
    items = []
    in_list = False
    for line in lines:
        if re.match(r"^\d+\.", line):
            in_list = True
            item = clean_item(line)
            if item and item not in items:
                items.append(item)
            continue
        if in_list:
            if re.match(r"^[a-zA-Z]\.", line):
                continue
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


def is_valid_item(item: str) -> bool:
    lower = item.lower()
    if "http" in lower or "mailto" in lower or "@" in lower:
        return False
    if len(item) > 140:
        return False
    return True


PIECE_EXCLUDE_PATTERNS = [
    r"\bplease\b",
    r"\bwork\b",
    r"\blisten\b",
    r"\breview\b",
    r"\bsign\b",
    r"\barrive\b",
    r"\bdownload\b",
    r"\bsheet\b",
    r"\blink\b",
    r"\bemail\b",
    r"\bchart\b",
    r"\bnew members\b",
]


def normalize_piece_item(item: str) -> str:
    for separator in (" - ", " – ", " — ", ": "):
        if separator in item:
            left, right = item.split(separator, 1)
            if re.search(r"(make sure|review|please|listen|work|note)", right, re.I):
                return left.strip()
    return item.strip()


def is_piece_title(item: str) -> bool:
    cleaned = normalize_piece_item(item)
    lower = cleaned.lower()
    if not cleaned:
        return False
    if len(cleaned) > 70:
        return False
    if any(re.search(pattern, lower) for pattern in PIECE_EXCLUDE_PATTERNS):
        return False
    if re.search(r"\b\d{1,2}:\d{2}\b", cleaned):
        return False
    return True


def filter_piece_items(items: list[str]) -> list[str]:
    filtered = []
    for item in items:
        cleaned = normalize_piece_item(item)
        if is_piece_title(cleaned) and cleaned not in filtered:
            filtered.append(cleaned)
    return filtered


def extract_action_items(lines: list[str], keywords: list[str], limit: int = 4) -> list[str]:
    results = []
    for line in lines:
        lower = line.lower()
        if not re.match(r"^(\d+\.|[-*]\s|[a-zA-Z]\.)", line):
            continue
        if any(keyword in lower for keyword in keywords):
            cleaned = clean_item(line)
            if cleaned and cleaned not in results and is_valid_item(cleaned):
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


def email_target_date(email: ParsedEmail) -> datetime | None:
    return extract_date_from_text(f"{email.subject} {email.body}") or email.date


def sort_by_relevance(emails: list[ParsedEmail]) -> list[ParsedEmail]:
    today = datetime.now().date()

    def score(email: ParsedEmail) -> tuple[int, int, int, datetime]:
        target = email_target_date(email)
        subject = email.subject.lower()
        schedule_hint = 0 if "rehearsal schedule" in subject or "schedule for" in subject else 1
        if not target:
            return (schedule_hint, 3, 9999, email.date or datetime.min)
        delta = (target.date() - today).days
        if 0 <= delta <= 14:
            return (schedule_hint, 0, delta, target)
        if -7 <= delta < 0:
            return (schedule_hint, 1, abs(delta), target)
        return (schedule_hint, 2, abs(delta), target)

    return sorted(emails, key=score)


def main() -> None:
    email_dir = Path(os.environ.get("EMAIL_DIR", "emails"))
    index_path = Path(os.environ.get("INDEX_PATH", "index.html"))

    if not email_dir.exists() or not index_path.exists():
        raise SystemExit("Missing emails directory or index.html")

    emails = load_emails(email_dir)
    if not emails:
        raise SystemExit("No emails found to process")

    updated_stamp = format_date(datetime.now())

    symphonic_emails = [e for e in emails if "symphonic" in e.subject.lower()]
    jazz_emails = [e for e in emails if "jazz" in e.subject.lower()]

    symphonic_sorted = sort_by_relevance(symphonic_emails)
    jazz_sorted = sort_by_relevance(jazz_emails)

    symphonic_primary = None
    symphonic_pieces: list[str] = []
    for email_item in symphonic_sorted:
        symphonic_pieces = extract_numbered_list(
            email_item.lines, r"Symphonic Band Rehearsal Schedule"
        )
        if not symphonic_pieces:
            symphonic_pieces = extract_first_numbered_list(email_item.lines)
        symphonic_pieces = filter_piece_items(symphonic_pieces)
        if symphonic_pieces:
            symphonic_primary = email_item
            break
    if not symphonic_primary and symphonic_sorted:
        symphonic_primary = symphonic_sorted[0]

    symphonic_week = email_target_date(symphonic_primary) if symphonic_primary else None

    symphonic_details: list[str] = []
    if symphonic_primary:
        symphonic_details = extract_action_items(
            symphonic_primary.lines,
            ["handbook", "arrive", "listen", "sign", "lesson", "practice", "new", "recruit"],
        )

    symphonic_notes: list[str] = []

    jazz_primary = None
    jazz_pieces: list[str] = []
    for email_item in jazz_sorted:
        jazz_pieces = extract_lettered_list(
            email_item.lines, r"keep these charts"  # spring email list
        )
        if not jazz_pieces:
            jazz_pieces = extract_numbered_list(email_item.lines, r"REHEARSAL SCHEDULE")
        if not jazz_pieces:
            jazz_pieces = extract_first_numbered_list(email_item.lines)
        jazz_pieces = filter_piece_items(jazz_pieces)
        if jazz_pieces:
            jazz_primary = email_item
            break
    if not jazz_primary and jazz_sorted:
        jazz_primary = jazz_sorted[0]

    jazz_week = email_target_date(jazz_primary) if jazz_primary else None

    jazz_details: list[str] = []
    if jazz_primary:
        jazz_details.extend(extract_jazz_timing_details(jazz_primary.body))
        if len(jazz_details) < 4:
            jazz_details.extend(
                extract_action_items(
                    jazz_primary.lines,
                    ["chart", "listen", "sheet", "new", "review", "recruit"],
                    limit=4 - len(jazz_details),
                )
            )

    jazz_notes: list[str] = []

    ai_email_limit = int(os.environ.get("OPENAI_EMAIL_LIMIT", "2"))
    ai_context = build_ai_context(
        symphonic_sorted[:ai_email_limit],
        jazz_sorted[:ai_email_limit],
    )
    ai_summary = call_openai(ai_context)
    if isinstance(ai_summary, dict):
        symphonic_ai = ai_summary.get("symphonic", {}) if isinstance(ai_summary.get("symphonic"), dict) else {}
        jazz_ai = ai_summary.get("jazz", {}) if isinstance(ai_summary.get("jazz"), dict) else {}

        symphonic_pieces_ai = normalize_ai_list(symphonic_ai.get("pieces", []))
        symphonic_details_ai = normalize_ai_list(symphonic_ai.get("details", []))
        symphonic_notes_ai = normalize_ai_list(symphonic_ai.get("notes", []))

        jazz_pieces_ai = normalize_ai_list(jazz_ai.get("pieces", []))
        jazz_details_ai = normalize_ai_list(jazz_ai.get("details", []))
        jazz_notes_ai = normalize_ai_list(jazz_ai.get("notes", []))

        if symphonic_pieces_ai:
            symphonic_pieces = symphonic_pieces_ai
        if symphonic_details_ai:
            symphonic_details = symphonic_details_ai
        if symphonic_notes_ai:
            symphonic_notes = symphonic_notes_ai

        if jazz_pieces_ai:
            jazz_pieces = jazz_pieces_ai
        if jazz_details_ai:
            jazz_details = jazz_details_ai
        if jazz_notes_ai:
            jazz_notes = jazz_notes_ai

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
    html_text = replace_list(html_text, "symphonic-notes", symphonic_notes)
    html_text = replace_list(html_text, "jazz-pieces", jazz_pieces)
    html_text = replace_list(html_text, "jazz-details", jazz_details)
    html_text = replace_list(html_text, "jazz-notes", jazz_notes)

    index_path.write_text(html_text, encoding="utf-8")


if __name__ == "__main__":
    main()
