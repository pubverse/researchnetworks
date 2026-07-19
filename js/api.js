/* PubVerse API client. A thin wrapper over fetch, nothing more.
   Every request carries the session cookie (credentials: 'include').
   On success it returns the backend JSON as-is. On any transport or HTTP
   failure it returns a uniform { ok:false, message } so a page can render one
   clean inline error. Business rejections from the backend (a screening gate,
   a topic that is not covered, a promote that does not corroborate) already
   arrive as { ok:false, message } and are passed straight through.
   Requires js/config.js to have loaded first. */
(function () {
  'use strict';

  if (!window.PV || !window.PV.API_BASE) {
    throw new Error('PubVerse: js/config.js must load before js/api.js');
  }

  var NET_ERROR = 'Could not reach the PubVerse service. Check your connection and try again.';

  function statusMessage(status) {
    if (status === 401 || status === 403) return 'Please sign in and try again.';
    if (status === 404) return 'Not found.';
    if (status === 413) return 'That submission is too large to process.';
    if (status === 429) return 'Too many requests. Please wait a moment and retry.';
    if (status >= 500) return 'The PubVerse service hit an error. Please try again shortly.';
    return 'Request failed (' + status + ').';
  }

  function pickMessage(data, status) {
    if (data) {
      if (typeof data.message === 'string' && data.message) return data.message;
      if (typeof data.detail === 'string' && data.detail) return data.detail;
    }
    return statusMessage(status);
  }

  // Core request. Returns parsed JSON on success, or { ok:false, message } on
  // any failure the caller could not otherwise see.
  async function request(path, opts) {
    opts = opts || {};
    var init = {
      method: opts.method || 'GET',
      credentials: 'include',
      headers: {}
    };
    if (opts.body !== undefined && opts.body !== null) {
      init.headers['Content-Type'] = 'application/json';
      init.body = JSON.stringify(opts.body);
    }

    var res;
    try {
      res = await fetch(window.PV.API_BASE + path, init);
    } catch (e) {
      return { ok: false, message: NET_ERROR };
    }

    var text = '';
    try { text = await res.text(); } catch (e) { text = ''; }

    var data = null;
    if (text) {
      try { data = JSON.parse(text); } catch (e) { data = null; }
    }

    if (data !== null && typeof data === 'object') {
      // Backend told us the outcome explicitly (score, promote, compass validate).
      if (typeof data.ok !== 'undefined') return data;
      // Plain success body with no ok flag: { role }, { username, ... }, a list.
      if (res.ok) return data;
      // Error body without an ok flag.
      return { ok: false, status: res.status, message: pickMessage(data, res.status) };
    }

    // No JSON body came back.
    if (res.ok) return { ok: true };
    return { ok: false, status: res.status, message: statusMessage(res.status) };
  }

  // Build a download URL for a compass export. Used as an <a href>, so the
  // session cookie rides along automatically. runId, scope, and cols are
  // encoded; cols may be an array or a comma string.
  function compassExportUrl(runId, scope, cols) {
    var params = new URLSearchParams();
    if (scope) params.set('scope', scope);
    if (cols) params.set('cols', Array.isArray(cols) ? cols.join(',') : cols);
    var qs = params.toString();
    return window.PV.API_BASE + '/api/compass/export/' +
      encodeURIComponent(runId) + '.tsv' + (qs ? '?' + qs : '');
  }

  window.PV.api = {
    // account
    login: function (username, password) {
      return request('/api/login', { method: 'POST', body: { username: username, password: password } });
    },
    logout: function () {
      return request('/api/logout', { method: 'POST' });
    },
    me: function () {
      return request('/api/me');
    },

    // score and history
    score: function (abstract, title) {
      var body = { abstract: abstract };
      if (title) body.title = title;
      return request('/api/score', { method: 'POST', body: body });
    },
    history: function () {
      return request('/api/history');
    },
    deleteHistory: function (id) {
      return request('/api/history/' + encodeURIComponent(id), { method: 'DELETE' });
    },
    promote: function (payload) {
      return request('/api/promote', { method: 'POST', body: payload });
    },

    // compass
    compassValidate: function (topic) {
      return request('/api/compass/validate', { method: 'POST', body: { topic: topic } });
    },
    compassRun: function (topic, monthsBack, email) {
      var body = { topic: topic, months_back: monthsBack };
      if (email) body.email = email;
      return request('/api/compass/run', { method: 'POST', body: body });
    },
    compassPoll: function (runId) {
      return request('/api/compass/run/' + encodeURIComponent(runId));
    },
    compassExportUrl: compassExportUrl
  };
})();
