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
a different backend, either edit the API_BASE line in js/config.js, or set window.PV_API_BASE in an inline
script before config.js loads. No page should hardcode a backend URL of its own, and the static site never
stores a password, session token, or hash; the backend owns the login session by way of a cookie it sets.

## Running it locally

From inside this folder, start a plain file server and open the printed address in a browser.

```
python3 -m http.server 8000
```

Then visit http://localhost:8000/index.html. If a local backend is also running, point js/config.js at it
first (see above) so the login veil and the Score, Compass, and Docs pages have something to call. A plain
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
