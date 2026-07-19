/* PubVerse shared runtime config.
   Load this FIRST on every page, before js/api.js and js/ui.js.
   Exposes a single global, window.PV, that api.js and ui.js hang off of.
   No page should hardcode any of these values. */
(function () {
  'use strict';

  // Backend base URL.
  // The local default is below. In production this is overridden to the tunnel
  // URL, either by editing this one line at deploy time or by setting
  // window.PV_API_BASE before this script loads. The static site never stores
  // any secret, password, or hash.
  var API_BASE = window.PV_API_BASE || 'http://localhost:8077';

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
