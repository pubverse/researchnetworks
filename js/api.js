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

  // API_BASE may be '' (same-origin), so check that config defined it, not truthiness.
  if (!window.PV || typeof window.PV.API_BASE !== 'string') {
    throw new Error('PubVerse: js/config.js must load before js/api.js');
  }

  var NET_ERROR = 'Could not reach the PubVerse service. Check your connection and try again.';

  // Session token. The site and API are on different domains, so the httpOnly session cookie is a
  // third-party cookie that Safari and hardened browsers block. We therefore also keep the token in
  // localStorage and send it as a Bearer header, which works regardless of cookie policy.
  var TOKEN_KEY = 'pv_token';
  function getToken() { try { return localStorage.getItem(TOKEN_KEY) || ''; } catch (e) { return ''; } }
  function setToken(t) { try { if (t) localStorage.setItem(TOKEN_KEY, t); } catch (e) {} }
  function clearToken() { try { localStorage.removeItem(TOKEN_KEY); } catch (e) {} }

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
    var tok = getToken();
    if (tok) init.headers['Authorization'] = 'Bearer ' + tok;
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

  // Build a download URL for a compass export, used as an <a href download>. A
  // download link cannot carry an Authorization header, so the token travels in the
  // query. It is NOT the session token: the caller passes a short-lived token
  // scoped to this one run (minted server-side and returned with the run's poll
  // result), so a leaked export URL cannot be replayed as the account. runId,
  // scope, and cols are encoded; cols may be an array or a comma string.
  function compassExportUrl(runId, scope, cols, exportToken) {
    var params = new URLSearchParams();
    if (scope) params.set('scope', scope);
    if (cols) params.set('cols', Array.isArray(cols) ? cols.join(',') : cols);
    if (exportToken) params.set('pv_token', exportToken);
    var qs = params.toString();
    return window.PV.API_BASE + '/api/compass/export/' +
      encodeURIComponent(runId) + '.tsv' + (qs ? '?' + qs : '');
  }

  window.PV.api = {
    // account
    login: function (username, password) {
      return request('/api/login', { method: 'POST', body: { username: username, password: password } })
        .then(function (res) { if (res && res.ok && res.token) setToken(res.token); return res; });
    },
    logout: function () {
      clearToken();
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
    // Covered subjects for the searchable topic picker. Optional q narrows the
    // list server-side as the user types (e.g. ?q=vir). Returns the backend JSON.
    compassCovered: function (q) {
      var path = '/api/compass/covered';
      if (q) path += '?q=' + encodeURIComponent(q);
      return request(path);
    },
    // Ask PubVerse to cover a subject it does not cover yet. The optional email is
    // recorded so the person can be told when that coverage lands.
    compassRequestTopic: function (topic, email) {
      var body = { topic: topic };
      if (email) body.email = email;
      return request('/api/compass/request-topic', { method: 'POST', body: body });
    },
    compassRun: function (topic, monthsBack, email) {
      var body = { topic: topic, months_back: monthsBack };
      if (email) body.email = email;
      return request('/api/compass/run', { method: 'POST', body: body });
    },
    compassPoll: function (runId) {
      return request('/api/compass/run/' + encodeURIComponent(runId));
    },
    compassRuns: function () {
      return request('/api/compass/runs');
    },
    compassExportUrl: compassExportUrl,

    // saved compass topics (Profile). These hit the real server so a saved topic
    // persists across browsers and devices and shows up in Profile.
    compassSaveTopic: function (topic, monthsBack) {
      var body = { topic: topic };
      if (monthsBack != null) body.months_back = monthsBack;
      return request('/api/compass/topics', { method: 'POST', body: body });
    },
    compassTopics: function () {
      return request('/api/compass/topics');
    },
    compassDeleteTopic: function (id) {
      return request('/api/compass/topics/' + encodeURIComponent(id), { method: 'DELETE' });
    }
  };
})();
