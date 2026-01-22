# Agent Instructions

Goal: keep `index.html` aligned to the most recent rehearsal emails.

When asked to update content:
- Use the newest `.eml` files in `emails/` for each ensemble (Symphonic, Jazz).
- Update only the weekly content blocks in `index.html`:
  - Week line (`data-week`)
  - Pieces this week
  - Other details
  - Additional notes
  - Updated timestamp (`data-updated`)
- Prefer the schedule for the current/next rehearsal week; ignore older items unless the newest email explicitly references them.
- Do not use PDFs for content extraction unless explicitly requested.
- Leave concert dates, layout, and other sections unchanged.
