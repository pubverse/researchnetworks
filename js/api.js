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
  // A request that timed out is a different situation from one that never connected, and saying
  // so stops a slow assessment being read as a broken site. Scoring an abstract genuinely takes
  // upwards of a minute, so this is the message a waiting user is most likely to see.
  var SLOW_ERROR = 'The PubVerse service is taking longer than usual to answer. It is still ' +
                   'running, so please give it a moment and try again.';

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

    // Time-box every request. Without this, a hung connection (accepted but never answered) leaves
    // fetch pending forever, which strands the sign-in card on its boot spinner. On timeout we abort,
    // which throws below and returns a message that says so.
    //
    // A single blip should not read as an outage. The public path is a home connection behind a
    // tunnel and it does drop briefly, so one transport failure is retried once before we give up.
    // The retry is strictly opt-in per call, because it is not free and it is not always safe:
    //   - GET is retried by default; asking twice costs nothing and changes nothing.
    //   - POST is NOT retried by default. /api/score takes over a minute and writes a history row,
    //     so retrying it would double a slow request and could file the same abstract twice.
    //   - Sign-in opts back in explicitly: a connection that never landed created no server state.
    // A request that reached the server and came back 4xx is a real answer and is never retried.
    var wantRetry = (typeof opts.retry === 'boolean')
      ? opts.retry
      : (init.method === 'GET');

    var attempt = async function () {
      var ctrl = (typeof AbortController !== 'undefined') ? new AbortController() : null;
      var timer = ctrl ? setTimeout(function () { ctrl.abort(); }, opts.timeout || 20000) : null;
      var i = ctrl ? Object.assign({}, init, { signal: ctrl.signal }) : init;
      try {
        return { res: await fetch(window.PV.API_BASE + path, i) };
      } catch (e) {
        // An abort is our own timeout firing, which is a different story from the connection
        // never landing, and the reader deserves to be told which one happened.
        return { err: (e && e.name === 'AbortError') ? 'timeout' : 'network' };
      } finally {
        if (timer) clearTimeout(timer);
      }
    };

    var res;
    var out = await attempt();
    if (out.err && wantRetry) {
      await new Promise(function (r) { setTimeout(r, 1200); });
      out = await attempt();
    }
    if (out.err) {
      return {
        ok: false,
        transport: out.err,
        message: out.err === 'timeout' ? SLOW_ERROR : NET_ERROR
      };
    }
    res = out.res;

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
      // Opts into the one retry: a sign-in that never reached the server created nothing, so
      // asking again is safe, and a brief tunnel blip is exactly what strands people at the door.
      return request('/api/login', { method: 'POST', retry: true,
                                     body: { username: username, password: password } })
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
      // Scoring runs the full grounded pipeline (retrieval + LLM verdict) and can take ~30-60s, well past
      // the 20s default; give it a generous ceiling so a slow score is not misreported as "could not reach".
      // Deliberately NOT retried, and given a long budget. A real assessment measured 60 to 75
      // seconds against a busy GPU, so the ceiling has room over that; retrying would double a
      // slow request and could file the same abstract into the history twice.
      return request('/api/score', { method: 'POST', retry: false, body: body, timeout: 180000 });
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
