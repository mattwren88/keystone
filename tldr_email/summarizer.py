"""Utilities for turning verbose emails into concise TL;DR messages.

The module extracts high-signal lines (e.g., rehearsal notices, dates, and calls to
action) from free-form email text. It returns both a structured representation and a
pre-formatted TL;DR string suitable for SMS or push notifications.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, List


# Common rehearsal- and logistics-related keywords to prioritize.
PRIORITY_KEYWORDS = (
    "rehearsal",
    "sectional",
    "call time",
    "performance",
    "concert",
    "dress",
    "location",
    "room",
    "hall",
    "auditorium",
    "stage",
    "bring",
    "wear",
    "equipment",
    "scores",
    "attendance",
    "required",
    "deadline",
    "update",
)

# Regex patterns for lightweight date and time detection.
DAY_PATTERN = r"\b(Mon(day)?|Tue(sday)?|Wed(nesday)?|Thu(rsday)?|Fri(day)?|Sat(urday)?|Sun(day)?|[0-1]?\d/[0-3]?\d)\b"
TIME_PATTERN = r"\b([0-2]?\d:[0-5]\d(?:\s?(?:am|pm|AM|PM))?|[0-2]?\d\s?(?:am|pm|AM|PM))\b"


@dataclass
class EmailTLDR:
    """Structured summary of an email."""

    highlights: List[str] = field(default_factory=list)
    schedule: List[str] = field(default_factory=list)

    def format(self) -> str:
        """Return a human-friendly TL;DR string."""
        sections = []

        if self.highlights:
            sections.append("Highlights:\n- " + "\n- ".join(self.highlights))

        if self.schedule:
            sections.append("Schedule:\n- " + "\n- ".join(self.schedule))

        if not sections:
            return "TL;DR unavailable: no signal detected."

        return "\n\n".join(sections)


def _normalize_lines(text: str) -> List[str]:
    lines = [line.strip() for line in text.splitlines()]
    # Filter out quoted replies or empty lines.
    return [line for line in lines if line and not line.startswith(">") and not line.lower().startswith("subject:")]


def _contains_keyword(line: str, keywords: Iterable[str]) -> bool:
    lowered = line.lower()
    return any(keyword in lowered for keyword in keywords)


def _looks_like_schedule(line: str) -> bool:
    return bool(re.search(DAY_PATTERN, line, re.IGNORECASE) and re.search(TIME_PATTERN, line))


def _dedupe_preserve_order(lines: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            result.append(line)
    return result


def summarize_email(text: str, *, max_highlights: int = 4) -> EmailTLDR:
    """Generate a TL;DR from raw email text.

    The function prioritizes:
    1. Lines that look like schedule entries (date + time)
    2. Lines containing priority keywords (logistics, requirements, updates)
    3. Fallback to the first few non-empty lines if nothing else matches.
    """

    lines = _normalize_lines(text)

    schedule_lines = [line for line in lines if _looks_like_schedule(line)]
    keyword_lines = [line for line in lines if line not in schedule_lines and _contains_keyword(line, PRIORITY_KEYWORDS)]

    # Balance schedule-heavy emails by reserving part of the budget for keywords/logistics.
    schedule_budget = max(1, max_highlights // 2)
    primary_schedule = schedule_lines[:schedule_budget]

    remaining_slots = max_highlights - len(primary_schedule)
    blended = primary_schedule + keyword_lines[:remaining_slots]

    # If we still have room, append any remaining schedule lines, then fall back to the first lines.
    if len(blended) < max_highlights:
        remaining_slots = max_highlights - len(blended)
        blended.extend(schedule_lines[schedule_budget:schedule_budget + remaining_slots])

    if not blended:
        blended = lines[:max_highlights]

    highlights = _dedupe_preserve_order(blended)
    schedule = _dedupe_preserve_order(schedule_lines[:schedule_budget])

    return EmailTLDR(highlights=highlights, schedule=schedule)


def format_tldr(text: str, *, max_highlights: int = 4) -> str:
    """Convenience wrapper to return only the formatted TL;DR string."""
    return summarize_email(text, max_highlights=max_highlights).format()
