# Band Briefing (single-page app)

A lightweight, static page that lets you paste band director emails or upload a PDF and instantly surfaces dates, rehearsals, and action items. All parsing happens locally in the browser.

## Quick start

1. Open `index.html` directly in your browser, or serve the folder with something like:
   ```bash
   python3 -m http.server 3000
   ```
   then visit `http://localhost:3000`.
2. Paste an email into the text area or drop a PDF. Hit **Parse now** to see dates, rehearsal times, tasks, and links.

Notes:
- PDF extraction uses PDF.js from a CDN; you'll need internet access the first time it loads.
- Date/time extraction is improved with chrono-node (loaded from a CDN); it will fall back to regexes if offline.
- Nothing is sent to any server; the page runs entirely on the client.
