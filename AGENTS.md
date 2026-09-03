# Agent Instructions

Goal: keep `index.html` aligned to the most recent director emails.

This is a static page with no build step and no automation — updates happen by hand.

When asked to update content:
- Read the director's recent emails (Gmail, or `.eml` files if provided) for each ensemble
  (Chorale, Symphonic Band, Jazz Ensemble) and for season/concert dates.
- Rehearsals and concerts are defined once, in the `seasonEvents` array inside the `<script>`
  block near the bottom of `index.html`. It drives the countdown banner, the timeline strip, and
  the "Add all dates to your calendar" `.ics` download together — edit it there rather than in
  three places.
- Repertoire per ensemble lives in the "Music to Practice" section (`rep-section`) — update
  pieces, listening links, sheet-music links, and the short notes under each.
- Update the `.updated` bar's date and source email range when you change content.
- Prefer the newest email for each ensemble; only carry forward older items the newest email
  still references.
- Do not use PDFs for content extraction unless explicitly requested.
- Leave the page's layout and visual design unchanged unless asked.
