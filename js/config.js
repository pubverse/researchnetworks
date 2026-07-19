/* PubVerse shared runtime config.
   Load this FIRST on every page, before js/api.js and js/ui.js.
   Exposes a single global, window.PV, that api.js and ui.js hang off of.
   No page should hardcode any of these values. */
(function () {
  'use strict';

  // Backend base URL.
  // No internal host or port is baked into this public file. In production this is
  // the backend's public tunnel URL, set on the line below at deploy time. For
  // local development, override it without editing this file by setting
  // window.PV_API_BASE before this script loads, pointing at your own local
  // backend. The static site never stores any secret, password, or hash.
  // NOTE: this is a temporary public tunnel to the backend. Swap it for the stable
  // api.pubverse.ai once that is set up. Override without redeploy via window.PV_API_BASE.
  var API_BASE = (typeof window.PV_API_BASE === 'string') ? window.PV_API_BASE
    : 'https://soldiers-pace-amended-modelling.trycloudflare.com';

  window.PV = {
    API_BASE: API_BASE,

    // Score page: hard word limit for the abstract. Submit is blocked over this.
    WORD_LIMIT: 1000,

    // Compass page: character cap on the topic input.
    TOPIC_CHAR_LIMIT: 120,

    // Compass page: how far back to look, in months. The selector is built from
    // this list and starts on TIMEFRAME_DEFAULT (the third option, 3 months).
    TIMEFRAMES: [1, 3, 6, 9, 12, 15, 18, 21, 24],
    TIMEFRAME_DEFAULT: 3
  };
})();
