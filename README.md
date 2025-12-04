# Keystone TL;DR helper

A small utility for condensing verbose rehearsal and performance emails into concise TL;DRs that are easy to deliver as notifications.

## How it works
- Scans each email for schedule-like lines (day + time) and logistics keywords (rehearsal, performance, location, bring, wear, etc.).
- Deduplicates repeating lines while preserving their original order.
- Produces a structured `EmailTLDR` object plus a pre-formatted TL;DR string you can send via SMS or push.

## Quick start
```bash
python - <<'PY'
from tldr_email import format_tldr

email = """
Subject: Week 3 Rehearsals
Hi team,

Monday 6:30pm rehearsal in Brooks Hall.
Bring your stand and music.
Thursday 7:00pm sectional in Room 201.

Please wear all black for Friday's performance at 8pm.
"""

print(format_tldr(email))
PY
```

Sample output:
```
Highlights:
- Monday 6:30pm rehearsal in Brooks Hall.
- Thursday 7:00pm sectional in Room 201.
- Bring your stand and music.
- Please wear all black for Friday's performance at 8pm.

Schedule:
- Monday 6:30pm rehearsal in Brooks Hall.
- Thursday 7:00pm sectional in Room 201.
```

## Running tests
```bash
python -m pytest
```
