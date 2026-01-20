# Keystone College Band Briefing

A focused, static one-page summary for Symphonic Band, Jazz Band, and upcoming concert dates. The content is curated from director emails and rendered as clean, readable detail cards.

## Quick start

Open `index.html` directly in your browser, or serve the folder:

```bash
python3 -m http.server 3000
```

Then visit `http://localhost:3000`.

## Project layout

- `index.html`: main page content
- `assets/css/styles.css`: site styles
- `assets/keystone-concerts.ics`: calendar download
- `assets/Handbook KC Performance Music Spring 2026.pdf`: handbook link
- `emails/`: optional local email cache (ignored by git)

## Updating content

Plain terms:
- New director emails (labeled `Keystone College`) are pulled via the Gmail API once a day.
- The latest emails are summarized into the Symphonic/Jazz cards.
- The page updates itself automatically; if automation fails, you can run one command locally.

Manual updates:
- Edit `index.html` directly.

Automated updates (local):
- `scripts/update_site.py` pulls labeled Gmail messages (if Gmail env vars are set) and updates `index.html`.
- `scripts/fetch_gmail_emails.py` is optional; it caches `.eml` files in `emails/` for offline parsing.
- Optional Gmail knobs: `GMAIL_LABEL` (default `Keystone College`), `GMAIL_LABEL_ID` (overrides name), `MAX_RESULTS`.

AI-assisted updates (optional):
- If `OPENAI_API_KEY` is set, `scripts/update_site.py` sends recent email content to OpenAI and
  uses the response to update the pieces, other details, and additional notes lists.
- Optional knobs: `OPENAI_MODEL` (default `gpt-5-nano`), `OPENAI_EMAIL_LIMIT` (default `2`).

Token helper:
- `scripts/get_gmail_refresh_token.py` generates the Gmail OAuth refresh token needed by GitHub Actions.

## GitHub Actions automation

A daily workflow runs at 12:00 UTC (7:00 AM EST / 8:00 AM EDT) to pull labeled emails and refresh the page:
- Label: `Keystone College`
- Workflow: `.github/workflows/update-from-emails.yml`

Required repository secrets:
- `GMAIL_CLIENT_ID`
- `GMAIL_CLIENT_SECRET`
- `GMAIL_REFRESH_TOKEN`
- `OPENAI_API_KEY` (optional, enables AI summaries)

The workflow commits only `index.html` (email files remain local and ignored).

## Manual fallback (local)

If the Action doesn’t run, you can update locally:

```bash
python3 scripts/update_site.py
```

If you need to set credentials for a local run, export:

```bash
export GMAIL_CLIENT_ID=...
export GMAIL_CLIENT_SECRET=...
export GMAIL_REFRESH_TOKEN=...
```

If you prefer an `.eml` cache for offline runs:

```bash
python3 scripts/fetch_gmail_emails.py
```

## Notes

- The `emails/` folder and OAuth client JSON are gitignored.
- Update the concert list in `assets/keystone-concerts.ics` if dates change.
