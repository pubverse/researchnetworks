# PubVerse - static site

This folder is the public static site for PubVerse, built to be served by GitHub Pages from the
`pubverse.github.io` repository, at the custom domain pubverse.ai once DNS is pointed at GitHub (see the
CNAME file in this folder). PubVerse itself evaluates a research abstract against the published literature
and reports back on its novelty, methodological strength, and likely impact, in plain language. This site
is affiliated with fortytwodegrees.org, shown in the header and footer next to the 42 compass mark in
assets/compass42.png; that mark and the affiliation line should stay in place on every page.

The site is intentionally thin. The pages (index.html, compass.html, docs.html, and the share page) sit at
the root next to css/app.css, which every page links and which holds the one shared visual system, and the
js folder, which holds config.js, api.js, and ui.js. Page markup should stay small and call into those
shared files rather than re-implement spinners, tabs, word counters, or fetch logic locally. The assets
folder holds the wordmark and the compass mark, and examples holds a couple of public-safe dashboard
snapshots that the docs and compass pages can read directly without a live backend call.

## Talking to the backend

This site does not run any scoring itself and does not talk to a database. Every score, history entry, or
compass run is a call out to the PubVerse backend, a separate service that lives in its own repository and
runs on its own infrastructure. The one place that says where that backend is right now is js/config.js,
which sets window.PV.API_BASE (and a small handful of other shared constants like the word limit and
timeframe options). Load config.js before api.js or ui.js on any page that needs them. To point the site at
a different backend, edit the API_BASE line in js/config.js and the matching `connect-src` and `frame-src`
entries in the Content-Security-Policy meta tag of each page. The only place the production backend host
appears outside js/config.js is that meta tag (a meta tag cannot read config.js). The static site never
stores a password, session token, or hash; the backend owns the login session by way of a cookie it sets.

## Running it locally

From inside this folder, start a plain file server and open the printed address in a browser.

```
python3 -m http.server 8000
```

Then visit http://localhost:8000/index.html. If a local backend is also running (for example http://localhost:8078), point js/config.js at it
first (see above) and add that origin to `connect-src` and `frame-src` in the CSP meta tag of each page, otherwise
the browser blocks the calls silently. An inline `window.PV_API_BASE` script no longer works on docs.html
(`script-src 'self'`). Make these edits locally and never commit them. A plain
file server is enough; nothing here needs a build step.

## Publishing

Push this folder's contents to the `pubverse.github.io` repository (whichever branch Pages is set to build
from, commonly main). Because the repository is named pubverse.github.io, GitHub treats it as a user or
organization Pages site and will publish it once the branch has content, no project settings required
beyond picking the source branch under Settings > Pages. Check that same Settings > Pages screen for the
custom domain field: it should already read pubverse.ai, picked up from the CNAME file committed in this
folder, but if it is blank, enter pubverse.ai there and leave "Enforce HTTPS" on once GitHub issues the
certificate. Leave the CNAME file in place; removing it or editing it to a different value changes what
domain the site answers to.

## What this file will not tell you

This README covers how the site is put together and how to run and publish it. It does not explain how
PubVerse arrives at a score, and it should not gain any content that does. If a future change to this site
needs to describe scoring in any more detail than "novelty, methodological strength, and likely impact,"
that decision belongs in the backend project, not here.

## Security notes

- Every HTML page that loads `js/api.js` (and the `compass.html` redirects) carries a Content-Security-Policy meta tag. `connect-src` and `frame-src` name the API base from `js/config.js`; if that host changes, change the tag too. Inline scripts are still allowed (`index.html` has them); moving them out is the follow-up that would let `script-src` drop `'unsafe-inline'`.
- Fonts are self-hosted under `assets/fonts/` (no request to Google).
- Run-map iframe: `compassRunMapUrl` puts `?pv_token=<export token>` in the iframe URL. This is not the session token: the backend mints it per run, it authorizes only that run's map and exports, and it expires after 30 minutes by default. The token can show up in browser history or logs during that window.
- The run-map iframe is not sandboxed and has no `referrerpolicy` yet. The map is generated content served from the API origin, so a script injected into a generated map would run there with that origin's cookies. A sandbox is not a drop-in fix: the author bridge in `js/compass.js` checks `event.origin` against the frame's own origin, and a sandboxed frame reports `null`. This stays open until the backend side lands (escaping of generated map content, and Referrer-Policy and CSP headers on the map response); the site-side sandbox and `referrerpolicy` should be revisited then.
- Font licenses (SIL OFL 1.1) ship beside the fonts: `assets/fonts/OFL-Jost.txt`, `assets/fonts/OFL-Newsreader.txt`.
