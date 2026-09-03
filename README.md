# Keystone College Band Briefing

A focused, static one-page summary for Chorale, Symphonic Band, Jazz Ensemble, and the season's
performance dates. The content is curated by hand from director emails and rendered as a single
readable page.

## Quick start

Open `index.html` directly in your browser, or serve the folder:

```bash
python3 -m http.server 3000
```

Then visit `http://localhost:3000`.

## Project layout

- `index.html`: the entire page — markup, styles, and script are all inline
- `assets/css/styles.css`: currently unused (page styles are inline in `index.html`); kept for future use
- `assets/Handbook KC Performance Music Spring 2026.pdf`: handbook link
- `assets/logo.png` / `assets/logo.svg`: branding assets

## Updating content

This site has no build step and no automation — it's a static page you edit by hand.

1. Read the director's recent emails and note what changed: rehearsal dates, repertoire, concert
   dates, recruiting asks, etc.
2. Edit `index.html` directly:
   - The season's rehearsals and concerts live in one place: the `seasonEvents` array near the
     bottom of the file (inside the `<script>` block). Each entry drives the countdown banner,
     the timeline strip, and the "Add all dates to your calendar" download — update it and all
     three stay in sync.
   - The "Music to Practice" section (`rep-section`) lists repertoire per ensemble with listening
     and sheet-music links.
   - The "Fall Dates" list, checklist, and footer are plain HTML further down the page.
3. Update the `.updated` bar's date and email range near the bottom of the page body.
4. Preview locally (see Quick start above) before committing.

## Calendar download

The "Add all dates to your calendar" button generates a `.ics` file in the browser, straight from
`seasonEvents` — there's no separate calendar file to keep in sync. Update `seasonEvents` and the
downloaded calendar updates with it.
