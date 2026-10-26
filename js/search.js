/* PubVerse search: authors, funders and subjects.

   Three surfaces, one API (js/api.js):
     - the nav box: a debounced typeahead grouped Authors / Funders / Subjects, ARIA combobox
       pattern, Up/Down/Enter/Esc. At phone width it is folded behind a search button and opens
       full width over the top of the page.
     - the Search tab: a large box, an Advanced disclosure with an ORCID field, and the results.
     - the author and funder views, shown inside the Search tab.

   Views are hash routes so they can be linked and the Back button works:
     #search?q=...   results (&in=authors|funders|topics narrows them to one kind)
     #author/A123   an author (network graph on top, the text beneath)     #funder/F123   a funder
     #space/<slug>   a research-space map (optionally ?author=A123 to place that author on load)
     #compass  #score  #history          the other tabs
   Every string that came from the server is escaped before it touches innerHTML, and only ids that
   match the backend's own patterns are ever turned into links.

   Requires js/config.js and js/api.js. Load after js/compass.js and before the landing glue. */
(function () {
  'use strict';

  if (!window.PV || !window.PV.api) {
    throw new Error('PubVerse: js/config.js and js/api.js must load before js/search.js');
  }
  var api = PV.api;

  function $(s, root) { return (root || document).querySelector(s); }
  function on(node, ev, fn, opt) { if (node) node.addEventListener(ev, fn, opt); }
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  var AUTHOR_RE = /^A\d{1,12}$/, FUNDER_RE = /^F\d{1,12}$/;
  var ORCID_RE = /^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$/;
  var Q_MAX = 120;
  var UNAVAILABLE = 'Search is unavailable right now. Please try again in a few minutes.';

  function reducedMotion() {
    try { return window.matchMedia('(prefers-reduced-motion: reduce)').matches; } catch (e) { return false; }
  }
  function num(n) {
    var x = Number(n);
    return (n == null || n === '' || !isFinite(x)) ? '--' : Math.round(x).toLocaleString('en-US');
  }
  function plural(n, one, many) { return Number(n) === 1 ? one : (many || one + 's'); }

  var regionNames = null;
  try { regionNames = new Intl.DisplayNames(['en'], { type: 'region' }); } catch (e) { regionNames = null; }
  function countryName(c) {
    c = String(c == null ? '' : c).trim();
    if (!c) return '';
    if (/^[A-Za-z]{2}$/.test(c) && regionNames) {
      try { return regionNames.of(c.toUpperCase()) || c; } catch (e) { return c; }
    }
    return c;
  }
  function normOrcid(s) {
    s = String(s == null ? '' : s).trim()
      .replace(/^https?:\/\/(www\.)?orcid\.org\//i, '').replace(/\/+$/, '').toUpperCase();
    return ORCID_RE.test(s) ? s : '';
  }
  function doiUrl(d) {
    d = String(d == null ? '' : d).trim();
    if (/^https?:\/\/(dx\.)?doi\.org\/10\./i.test(d)) return d;
    if (/^doi:/i.test(d)) d = d.slice(4);
    if (/^10\.\d{3,9}\/\S+$/.test(d)) return 'https://doi.org/' + d;
    return '';
  }
  // OpenAlex titles can carry their publisher's markup: whole MathML trees (<mml:math><mml:mi>H</mml:mi>...),
  // <i>, <sub>. Escaped, that markup is safe but prints as forty lines of tags on a phone. Tag-shaped
  // runs are dropped from the plain string BEFORE it is escaped (math boundaries become a space), so
  // what reaches the page is still only escaped text. "p < 0.05" survives: a tag starts with a letter.
  function cleanTitle(t) {
    return String(t == null ? '' : t)
      .replace(/<\/?(?:mml:)?math\b[^<>]*>/gi, ' ')
      .replace(/<\/?[A-Za-z][A-Za-z0-9:_-]*(?:\s[^<>]*)?\/?>/g, '')
      .replace(/\s+/g, ' ').trim();
  }

  // A list entry may be a bare string or an object with a name; never trust either shape.
  function nameOf(x) {
    if (x == null) return '';
    if (typeof x === 'string') return x;
    return x.name || x.display_name || '';
  }
  function authorIdOf(x) {
    var id = x && (x.id || x.author_id);
    return AUTHOR_RE.test(String(id || '')) ? String(id) : '';
  }
  var TOPIC_RE = /^(?:topic|subfield)-\d{1,9}$/;
  // Only topic and subfield hits have a page of their own; domains and fields still open the Compass.
  function topicIdOf(x) {
    var id = x && x.id;
    return TOPIC_RE.test(String(id || '')) ? String(id) : '';
  }
  var COV_NOTE = 'Funding metadata covers about a third of papers; counts are a floor.';
  // Short visible text beside a funder count, with the full note as text a screen reader reads.
  function covMark(note) {
    return ' <span class="cov">partial data<span class="sr-only">. ' + esc(note || COV_NOTE) + '</span></span>';
  }
  function funderIdOf(x) {
    var id = x && (x.id || x.funder_id);
    return FUNDER_RE.test(String(id || '')) ? String(id) : '';
  }

  // What the person reads when a call fails. 503 and 429 are the two that need their own words.
  function errorText(res) {
    if (!res) return UNAVAILABLE;
    if (res.status === 429) {
      var s = parseInt(res.retryAfter, 10);
      return 'Too many searches in a short time. Please wait ' +
        (s > 0 ? s + ' ' + plural(s, 'second') : 'a moment') + ' and try again.';
    }
    if (res.status === 503) return UNAVAILABLE;
    if (res.status === 401 || res.status === 403) return 'Please sign in again to search.';
    if (res.status === 404) return 'That profile is not in the index.';
    if (res.status === 422 || res.status === 400) {
      return res.message || res.error || 'That search could not be read. Try a shorter name or an ORCID iD.';
    }
    if (res.transport) return res.message || UNAVAILABLE;
    if (res.status >= 500) return UNAVAILABLE;
    return res.error || res.message || UNAVAILABLE;
  }

  /* ------------------------------------------------------------ small cache */
  // Back and Forward re-render a view from its route; answering those from a two-minute cache keeps
  // them instant and keeps a person clicking around well inside the per-minute lookup limit.
  var cache = Object.create(null), cacheOrder = [], CACHE_MS = 120000;
  function cached(key, fetcher) {
    var hit = cache[key];
    if (hit && Date.now() - hit.at < CACHE_MS) return Promise.resolve(hit.res);
    return Promise.resolve(fetcher()).then(function (res) {
      if (res && res.ok !== false) {
        cache[key] = { at: Date.now(), res: res };
        cacheOrder = cacheOrder.filter(function (k) { return k !== key; });
        cacheOrder.push(key);
        while (cacheOrder.length > 24) delete cache[cacheOrder.shift()];
      }
      return res;
    });
  }

  /* ---------------------------------------------------------------- routing */

  var tabsRoot = $('#tabsRoot');
  var homeEl = $('#searchHome'), viewEl = $('#searchView');
  var resultsEl = $('#searchResults'), qEl = $('#searchQ');
  var searchHash = '';          // the route the Search tab is showing, restored when its tab is clicked
  var lastResults = '';         // the last results route, for the "Back to results" link
  var scope = 'all';            // Authors | Funders | Subjects | All: what the Search tab and the nav box look for
  var SCOPES = { authors: 'authors', funders: 'funders', topics: 'topics', all: 'authors,funders,topics' };
  var SCOPE_TEXT = {
    all:     { label: 'Name, funder or subject', ph: 'A name, a funder such as NIH, or a subject', noun: 'authors, funders or subjects' },
    authors: { label: 'Author name or ORCID iD', ph: 'A researcher\u2019s name, or an ORCID iD', noun: 'authors' },
    funders: { label: 'Funder name', ph: 'A funder such as NIH or Wellcome', noun: 'funders' },
    topics:  { label: 'Subject', ph: 'A topic or field, such as epigenetics', noun: 'subjects' }
  };
  function scopeParam() { return scope === 'all' ? '' : '&in=' + scope; }
  function setScope(v, opts) {
    if (!SCOPES[v]) v = 'all';
    scope = v;
    Array.prototype.forEach.call(document.querySelectorAll('#searchScope button[data-scope]'), function (b) {
      b.setAttribute('aria-pressed', b.getAttribute('data-scope') === v ? 'true' : 'false');
    });
    var t = SCOPE_TEXT[v], lab = $('label[for="searchQ"]');
    if (lab) lab.textContent = t.label;
    if (qEl) qEl.placeholder = t.ph;
    if (navInput) {
      navInput.placeholder = v === 'all' ? 'Search authors, funders, subjects' : 'Search ' + t.noun;
      navInput.setAttribute('aria-label', 'Search ' + (v === 'all' ? 'authors, funders and subjects' : t.noun));
    }
  }
  var viewSeq = 0;              // bumps on every view change; a late answer for an old view is dropped
  var authed = false, routingClick = false;

  function tabButton(key) { return $('.tabs button[data-tab="' + key + '"]'); }
  function activeTab() {
    var b = $('.tabs button.active');
    return b ? b.getAttribute('data-tab') : '';
  }
  // Switch tabs the way a person would, by clicking, so every listener the landing glue hung on the
  // buttons (Compass start, History load) still runs. The router owns the URL during that click.
  function showTab(key) {
    if (activeTab() === key) return;
    var b = tabButton(key);
    if (!b) return;
    routingClick = true;
    try { b.click(); } finally { routingClick = false; }
  }
  function pushHash(h) {
    try { history.pushState(null, '', h || (location.pathname + location.search)); } catch (e) { /* file:// */ }
  }
  function go(h) {
    if (location.hash === h) route({ scroll: true });
    else location.hash = h;
  }
  function scrollToTabs() {
    if (!tabsRoot) return;
    try { tabsRoot.scrollIntoView({ block: 'start', behavior: reducedMotion() ? 'auto' : 'smooth' }); }
    catch (e) { tabsRoot.scrollIntoView(); }
  }

  function parseRoute(h) {
    h = String(h || '').replace(/^#/, '');
    var m;
    if (!h || h === 'search') return { tab: 'search', view: 'home', scope: 'all' };
    if ((m = /^author\/(A\d{1,12})$/.exec(h))) return { tab: 'search', view: 'author', id: m[1] };
    if ((m = /^topic\/((?:topic|subfield)-\d{1,9})$/.exec(h))) return { tab: 'search', view: 'topic', id: m[1] };
    if ((m = /^funder\/(F\d{1,12})$/.exec(h))) return { tab: 'search', view: 'funder', id: m[1] };
    if ((m = /^space\/([a-z0-9_-]{1,40})(?:\?author=(A\d{1,12}))?$/.exec(h))) {
      return { tab: 'search', view: 'space', id: m[1], author: m[2] || '' };
    }
    if (h.indexOf('search?') === 0) {
      var q = '', kind = 'all';
      try { var sp = new URLSearchParams(h.slice(7)); q = sp.get('q') || ''; kind = sp.get('in') || 'all'; } catch (e) { q = ''; }
      q = q.trim().slice(0, Q_MAX);
      if (!SCOPES[kind]) kind = 'all';
      return q ? { tab: 'search', view: 'results', q: q, scope: kind } : { tab: 'search', view: 'home', scope: kind };
    }
    if (h === 'compass' || h === 'score' || h === 'history') return { tab: h };
    return null;   // an in-page anchor or something we do not own: leave the page alone
  }

  function route(opts) {
    if (!authed) return;
    var r = parseRoute(location.hash);
    if (!r) return;
    showTab(r.tab);
    if (r.tab !== 'search') return;
    searchHash = (location.hash && location.hash !== '#') ? location.hash : '';
    viewSeq++;
    if (r.view === 'home' || r.view === 'results') setScope(r.scope || 'all');
    if (r.view === 'home') showHome();
    else if (r.view === 'results') { lastResults = location.hash; runSearch(r.q); }
    else if (r.view === 'author') openAuthor(r.id);
    else if (r.view === 'funder') openFunder(r.id);
    else if (r.view === 'topic') openTopic(r.id);
    else if (r.view === 'space') openSpace(r.id, r.author);
    if (opts && opts.scroll) scrollToTabs();
  }

  // A tab click records itself in the URL, so Back from the Compass returns to the author you were
  // reading. The Search tab returns to whatever it was last showing.
  Array.prototype.forEach.call(document.querySelectorAll('.tabs button[data-tab]'), function (b) {
    on(b, 'click', function () {
      if (routingClick || !authed) return;
      var key = b.getAttribute('data-tab');
      var want = key === 'search' ? (searchHash || '#search') : '#' + key;
      if (location.hash !== want) pushHash(want);
    });
  });
  on(window, 'hashchange', function () { route({ scroll: true }); });
  on(document, 'pv:authed', function () {
    if (authed) return;
    authed = true;
    var r = parseRoute(location.hash);
    route({ scroll: !!(r && r.tab === 'search' && r.view !== 'home') });
  });

  /* ------------------------------------------------- Compass handoff (subjects) */

  // A subject chip or a topic hit fills the Compass subject box and asks it to check coverage, so
  // the person lands one click from "Find the needles" in that space.
  function openInCompass(subject) {
    subject = String(subject == null ? '' : subject).trim().slice(0, Q_MAX);
    if (!subject) return;
    var b = tabButton('compass');
    if (b && activeTab() !== 'compass') b.click();
    var input = $('#topic');
    if (!input) return;
    input.value = subject;
    input.dispatchEvent(new Event('input', { bubbles: true }));
    var chk = $('#checkBtn');
    if (chk && !chk.disabled) chk.click();
    scrollToTabs();
    try { input.focus({ preventScroll: true }); } catch (e) { input.focus(); }
  }

  /* -------------------------------------------------------------- shared bits */

  function orcidBadge(orcid, link) {
    var o = normOrcid(orcid);
    if (!o) return '';
    if (link) {
      return '<a class="orcid-badge" href="https://orcid.org/' + o + '" target="_blank" rel="noopener"' +
             ' title="ORCID record ' + o + '">ORCID <span class="orcid-id">' + o + '</span></a>';
    }
    return '<span class="orcid-badge" title="Has an ORCID iD">ORCID<span class="sr-only"> iD ' + o + '</span></span>';
  }
  function suspectNote(a) {
    if (!a || !a.suspect_merge) return '';
    return '<p class="suspect-note">This profile may merge several people who share a name. ' +
           'Its counts are probably too high.</p>';
  }
  function placeLine(inst, ctry) {
    var parts = [];
    if (inst) parts.push(esc(inst));
    if (ctry) parts.push(esc(countryName(ctry)));
    return parts.join(' &middot; ');
  }
  function spinnerHtml(cap) {
    return '<div class="compass-load show search-spin"><img src="assets/compass42.png" alt="">' +
           '<div class="cap">' + esc(cap) + '</div></div>';
  }
  function errHtml(text) { return '<div class="err show" role="alert">' + esc(text) + '</div>'; }
  function subjectChip(name, meta) {
    return '<button type="button" class="chip chip-btn" data-subject="' + esc(name) + '"' +
           ' title="Open ' + esc(name) + ' in the Compass">' + esc(name) +
           (meta ? ' <span class="chip-meta">' + esc(meta) + '</span>' : '') + '</button>';
  }
  function setTitle(t) { document.title = t ? (t + ' · PubVerse') : 'PubVerse'; }

  // Delegated once on the Search pane: subject chips open the Compass; [data-go] links route.
  var pane = $('.tabpane[data-tab="search"]');
  on(pane, 'click', function (e) {
    var t = e.target && e.target.closest ? e.target : null;
    if (!t) return;
    var chip = t.closest('[data-subject]');
    if (chip) { e.preventDefault(); openInCompass(chip.getAttribute('data-subject')); return; }
  });

  /* ------------------------------------------------------------- search home */

  function showHome() {
    setTitle('');
    releaseSpaceFrame();
    if (homeEl) homeEl.hidden = false;
    if (viewEl) { viewEl.hidden = true; viewEl.innerHTML = ''; }
    if (resultsEl) resultsEl.innerHTML = '';
    if (qEl) qEl.value = '';
    clearErr();
  }
  // A one-line summary for screen readers; the results list itself is not a live region, so it is
  // not read out in full every time it changes.
  function status(t) { var s = $('#searchStatus'); if (s) s.textContent = t; }
  function clearErr() { var e = $('#searchErr'); if (e) { e.textContent = ''; e.classList.remove('show'); } }
  function showErr(text) { var e = $('#searchErr'); if (e) { e.textContent = text; e.classList.add('show'); } }

  on($('#searchForm'), 'submit', function (e) {
    e.preventDefault();
    var q = (qEl && qEl.value || '').trim().slice(0, Q_MAX);
    if (!q) { showErr('Type a name, a funder or a subject to search for.'); if (qEl) qEl.focus(); return; }
    clearErr();
    go('#search?q=' + encodeURIComponent(q) + scopeParam());
  });
  // The scope control: a click changes what is searched. With a query already typed it searches again.
  on($('#searchScope'), 'click', function (e) {
    var b = e.target.closest && e.target.closest('button[data-scope]');
    if (!b) return;
    var v = b.getAttribute('data-scope');
    if (v === scope) return;
    setScope(v);
    var q = (qEl && qEl.value || '').trim().slice(0, Q_MAX);
    if (q && (resultsEl && resultsEl.innerHTML)) go('#search?q=' + encodeURIComponent(q) + scopeParam());
    else if (ta && ta.close) ta.close();
  });
  function orcidSearch() {
    var f = $('#searchOrcid');
    var o = normOrcid(f && f.value);
    if (!o) {
      showErr('Enter an ORCID iD such as 0000-0002-1825-0097, or its orcid.org link.');
      if (f) f.focus();
      return;
    }
    clearErr();
    setScope(scope === 'all' || scope === 'authors' ? scope : 'authors');
    go('#search?q=' + encodeURIComponent(o) + scopeParam());
  }
  on($('#searchOrcidBtn'), 'click', orcidSearch);
  on($('#searchOrcid'), 'keydown', function (e) {
    if (e.key === 'Enter') { e.preventDefault(); orcidSearch(); }
  });

  /* ---------------------------------------------------------------- results */

  function runSearch(q) {
    var my = viewSeq;
    setTitle('Search');
    if (homeEl) homeEl.hidden = false;
    if (viewEl) { viewEl.hidden = true; viewEl.innerHTML = ''; }
    destroyGraph();
    if (qEl && qEl.value.trim() !== q) qEl.value = q;
    var o = normOrcid(q), of = $('#searchOrcid');
    if (o && of) { of.value = o; var adv = $('#searchAdv'); if (adv) adv.open = true; }
    clearErr();
    resultsEl.innerHTML = spinnerHtml('Searching');
    resultsEl.setAttribute('aria-busy', 'true');
    status('Searching.');
    var sc = scope;
    cached('search:' + sc + ':' + q, function () { return api.search(q, { limit: sc === 'all' ? 20 : 25, types: SCOPES[sc] }); }).then(function (res) {
      if (my !== viewSeq) return;
      resultsEl.removeAttribute('aria-busy');
      if (!res || res.ok === false) { var t = errorText(res); resultsEl.innerHTML = errHtml(t); status(t); return; }
      resultsEl.innerHTML = resultsHtml(res, q, sc);
      var A = (res.authors || []).length, F = (res.funders || []).length, T = (res.topics || []).length;
      status(A + F + T ? [A + ' ' + plural(A, 'author'), F + ' ' + plural(F, 'funder'), T + ' ' + plural(T, 'subject')].join(', ') + ' found.'
                       : 'No matches.');
    });
  }

  function authorItem(a) {
    var id = authorIdOf(a), name = nameOf(a) || 'Unnamed author';
    var topics = (a.topics || []).map(nameOf).filter(Boolean).slice(0, 3);
    var place = placeLine(a.institution, a.country);
    return '<li class="sr-item">' +
      '<div class="sr-line">' + (id ? '<a class="sr-name" href="#author/' + id + '">' + esc(name) + '</a>'
                                     : '<span class="sr-name">' + esc(name) + '</span>') +
      ' ' + orcidBadge(a.orcid, false) + '</div>' +
      (place ? '<div class="sr-sub">' + place + '</div>' : '') +
      '<div class="sr-facts">h-index <b>' + num(a.h_index) + '</b> &middot; <b>' + num(a.works_count) + '</b> ' +
        plural(a.works_count, 'work') + ' &middot; <b>' + num(a.cited_by_count) + '</b> ' +
        plural(a.cited_by_count, 'citation') + '</div>' +
      (topics.length ? '<div class="sr-topics">' + topics.map(esc).join(' &middot; ') + '</div>' : '') +
      suspectNote(a) +
      '</li>';
  }
  function funderItem(f) {
    var id = funderIdOf(f), name = nameOf(f) || 'Unnamed funder';
    var aliases = (f.aliases || []).map(nameOf).filter(function (x) { return x && x !== name; }).slice(0, 4);
    var sub = [];
    if (aliases.length) sub.push('Also ' + aliases.map(esc).join(', '));
    if (f.country) sub.push(esc(countryName(f.country)));
    return '<li class="sr-item">' +
      '<div class="sr-line">' + (id ? '<a class="sr-name" href="#funder/' + id + '">' + esc(name) + '</a>'
                                     : '<span class="sr-name">' + esc(name) + '</span>') + '</div>' +
      (sub.length ? '<div class="sr-sub">' + sub.join(' &middot; ') + '</div>' : '') +
      (f.works_count != null ? '<div class="sr-facts">at least <b>' + num(f.works_count) + '</b> funded ' +
        plural(f.works_count, 'paper') + ' in the index' + covMark(f.coverage_note) + '</div>' : '') +
      '</li>';
  }
  var LEVEL = { domain: 'Domain', field: 'Field', subfield: 'Subfield', topic: 'Topic' };
  function topicItem(t) {
    var name = nameOf(t);
    if (!name) return '';
    var meta = [LEVEL[t.level] || '', t.parent ? 'in ' + t.parent : ''].filter(Boolean).join(' ');
    var tid = topicIdOf(t);
    if (tid && (t.level === 'topic' || t.level === 'subfield')) {
      return '<li class="sr-item sr-topic"><a class="chip chip-btn" href="#topic/' + tid + '">' + esc(name) +
        (meta ? ' <span class="chip-meta">' + esc(meta) + '</span>' : '') + '</a></li>';
    }
    return '<li class="sr-item sr-topic">' + subjectChip(name, meta) + '</li>';
  }
  function groupHtml(id, label, est, shown, itemsHtml, note) {
    var count = (est != null && Number(est) > shown) ? ' <span class="sr-count">about ' + num(est) + '</span>' : '';
    return '<section class="sr-group" aria-labelledby="' + id + '">' +
      '<h3 id="' + id + '">' + esc(label) + count + '</h3>' + (note || '') +
      '<ol class="sr-list">' + itemsHtml + '</ol></section>';
  }
  function resultsHtml(res, q, sc) {
    sc = sc || 'all';
    var A = (res.authors || []).filter(Boolean), F = (res.funders || []).filter(Boolean);
    var T = (res.topics || []).filter(function (t) { return t && nameOf(t); });
    var est = res.estimated || {};
    var orcidMode = res.mode === 'orcid';
    if (!A.length && !F.length && !T.length) {
      return '<p class="sr-empty">' + (orcidMode
        ? 'No author with the ORCID iD <b>' + esc(normOrcid(q) || q) + '</b> is in the index yet.'
        : 'Nothing matches <b>' + esc(q) + '</b> in ' + esc(sc === 'all' ? 'authors, funders or subjects' : SCOPE_TEXT[sc].noun) + '. Check the spelling, try fewer words' +
          (sc === 'all' || sc === 'authors' ? ', or search by ORCID under Advanced' : '') + (sc === 'all' ? '.' : ', or choose All.')) +
        '</p>';
    }
    var html = '';
    if (orcidMode) html += '<p class="mini muted">Exact match on ORCID iD ' + esc(normOrcid(q) || q) + '.</p>';
    if (A.length) {
      html += groupHtml('srAuthors', 'Authors', est.authors, A.length, A.map(authorItem).join(''),
        '<p class="mini muted sr-note">Ranked by name match, then by publication record' +
        (res.data_as_of ? ' (data through ' + esc(res.data_as_of) + ')' : '') + '. ' +
        '<a href="docs.html#search">How authors are ranked</a></p>');
    }
    if (F.length) html += groupHtml('srFunders', 'Funders', est.funders, F.length, F.map(funderItem).join(''),
      '<p class="mini muted sr-note">' + esc((F[0] && F[0].coverage_note) || COV_NOTE) + '</p>');
    if (T.length) {
      html += groupHtml('srTopics', 'Subjects', est.topics, T.length, T.map(topicItem).join(''),
        '<p class="mini muted sr-note">Choose a topic or subfield to see the people in it; other levels open in the Compass.</p>');
    }
    return html;
  }

  /* ------------------------------------------------------------- author view */

  function backLink() {
    var to = lastResults || '#search';
    var label = lastResults ? 'Back to results' : 'New search';
    return '<p class="view-back"><a href="' + esc(to) + '">&larr; ' + label + '</a></p>';
  }
  function beginView(caption) {
    setTitle('');
    releaseSpaceFrame();
    if (homeEl) homeEl.hidden = true;
    if (!viewEl) return;
    viewEl.hidden = false;
    viewEl.innerHTML = backLink() + spinnerHtml(caption);
  }
  function viewFailed(res) {
    if (!viewEl) return;
    viewEl.innerHTML = backLink() + errHtml(errorText(res));
  }
  function focusHeading() {
    var h = viewEl && viewEl.querySelector('h2');
    if (!h) return;
    h.setAttribute('tabindex', '-1');
    try { h.focus({ preventScroll: true }); } catch (e) { /* older browsers scroll; harmless */ }
  }

  function fact(k, v, title) {
    return '<div class="fact"' + (title ? ' title="' + esc(title) + '"' : '') + '><dt>' + esc(k) + '</dt><dd>' + v + '</dd></div>';
  }

  function openAuthor(id) {
    var my = viewSeq;
    beginView('Loading the author');
    cached('author:' + id, function () { return api.author(id); }).then(function (res) {
      if (my !== viewSeq) return;
      if (!res || res.ok === false || !res.author) { viewFailed(res); return; }
      renderAuthor(id, res);
    });
  }

  /* The author map on top of the author page. It is the field-map renderer in its shell build
     (examples/author_map.html) inside an iframe, fed the author's graph from the context-graph route by
     postMessage: the author at the center, linked to the themes and subthemes of their lead and senior papers
     and to the OpenAlex fields and subfields those papers sit in, with co-authors and context authors as
     separate legend classes. The shell draws legend, panels, Advanced settings (people filters), Highlight
     and Top Related Publications. The first build of a profile takes up to a minute: the route answers
     {pending:true} and this page polls it every 2 seconds, for at most 120 seconds. */
  var authorMap = null;
  var MAP_SHELL = 'examples/author_map.html';
  var MAP_MAX_NODES = 600, MAP_MAX_PEOPLE = 260;
  var POLL_MS = 2000, POLL_MAX_MS = 120000;
  var FILTERS_FALLBACK = { min_papers: 5, min_joint: 2, recent_months: 24 };
  function destroyGraph() {
    if (authorMap) { authorMap.timers.forEach(clearTimeout); authorMap.timers = []; }
    authorMap = null;
  }
  function mapStatus(st, t) {
    var el = viewEl && viewEl.querySelector('#avMapStatus');
    if (el && st === authorMap) el.textContent = t || '';
  }
  function cloneGraph(g) {
    return { nodes: (g && g.nodes || []).slice(), edges: (g && g.edges || []).slice(), metadata: (g && g.metadata) || {} };
  }
  // The shell re-mounts on every pv:graph, so a graph is sent once per change, and the load-time retries
  // run only while the shell has not yet answered pv:graph-ready.
  function sendGraph(st) {
    if (st !== authorMap || !st.frame || !st.frame.contentWindow || !st.graph) return;
    st.sent = true; st.ready = false;
    st.frame.contentWindow.postMessage({ type: 'pv:graph', nodes: st.graph.nodes, edges: st.graph.edges, metadata: st.graph.metadata }, location.origin);
  }
  function mountGraph(id) {
    var host = $('#avGraph');
    if (!host) return;
    destroyGraph();
    var st = authorMap = { id: id, my: viewSeq, host: host, graph: null, res: null, token: '', frame: null,
                           timers: [], expanded: Object.create(null), busy: false, started: Date.now(),
                           papers: null, paperSort: '', paperAll: false, showAllCo: false };
    host.innerHTML = spinnerHtml('Building the map');
    wireExports(st);
    loadProfile(st);
  }
  // A {pending:true} answer is not a result: it must never sit in the two-minute cache, so the fetcher reports it
  // as a not-ok answer (cached() keeps only ok ones).
  function fetchProfile(st) {
    return cached('ctx:' + st.id, function () {
      return api.authorContextGraph(st.id).then(function (r) {
        if (r && typeof r === 'object') {
          if (r.pending) return { ok: false, pending: true, message: r.message || '', position: r.position, works: r.works, n_papers: r.n_papers, stats: r.stats };
          r._at = Date.now();
        }
        return r;
      });
    });
  }
  function pendingText(r) {
    var n = r && (r.works != null ? r.works : (r.n_papers != null ? r.n_papers : (r.stats && r.stats.works)));
    var what = (n != null && isFinite(Number(n))) ? num(n) + ' papers' : 'the papers';
    return 'Building the author map: grouping ' + what + ' and writing titles. This takes up to a minute the first time.';
  }
  function loadProfile(st) {
    fetchProfile(st).then(function (res) {
      if (st !== authorMap || st.my !== viewSeq) return;
      if (res && res.pending) {
        if (Date.now() - st.started >= POLL_MAX_MS) {
          st.host.innerHTML = '<p class="mini muted">The map is still being built. Please check back in a minute. ' +
            '<button type="button" class="btn ghost sm" id="avMapRetry">Try again</button></p>';
          on($('#avMapRetry'), 'click', function () { if (st !== authorMap) return; st.started = Date.now(); st.host.innerHTML = spinnerHtml('Building the map'); loadProfile(st); });
          return;
        }
        st.host.innerHTML = spinnerHtml(pendingText(res));
        st.timers.push(setTimeout(function () { if (st === authorMap) loadProfile(st); }, POLL_MS));
        return;
      }
      if (!res || res.ok === false || !res.graph || !Array.isArray(res.graph.nodes)) {
        st.host.innerHTML = '<p class="mini muted">The map is unavailable right now. ' + esc(errorText(res)) + ' The lists below hold the same information as text.</p>';
        return;
      }
      if (res.export_token) { st.token = res.export_token; st.tokenAt = res._at || 0; }
      st.res = res;
      st.graph = cloneGraph(res.graph);
      st.expanded = Object.create(null);
      st.filters = Object.assign({}, FILTERS_FALLBACK, res.filters_default || (res.graph.metadata && res.graph.metadata.filters_default) || {});
      buildMapUi(st);
      fillPrimary(st);
      fillLists(st);
      fillProfilePapers(st);
    });
  }
  function buildMapUi(st) {
    var bar = '<div class="am-bar"><button class="btn ghost sm" id="avMapFull" type="button" aria-controls="avMapFrame">Full screen</button></div>';
    st.host.innerHTML = bar + '<p class="mini muted am-status" id="avMapStatus" role="status" aria-live="polite"></p>' +
      '<div class="space-map am-map"><iframe id="avMapFrame" allow="fullscreen" allowfullscreen ' +
      'title="Interactive map of this author’s work"></iframe></div>' +
      '<p class="mini muted">Click a person for their panel; Expand adds that person’s own co-authors and context authors. ' +
      '<a href="docs.html#maps">How author maps are built</a></p>';
    var frame = st.frame = $('#avMapFrame');
    var cap = st.res && st.res.stats && st.res.stats.cap_note;
    mapStatus(st, cap ? String(cap) : '');
    on($('#avMapFull'), 'click', function () {
      var req = frame.requestFullscreen || frame.webkitRequestFullscreen;
      if (document.fullscreenElement) { document.exitFullscreen(); return; }
      if (req) { try { var p = req.call(frame); if (p && p.catch) p.catch(function () {}); } catch (e) {} }
    });
    on(frame, 'load', function () {
      st.timers.push(setTimeout(function () { if (!st.sent) sendGraph(st); }, 200));
      [2500, 6000].forEach(function (ms) { st.timers.push(setTimeout(function () { if (!st.ready) sendGraph(st); }, ms)); });
    });
    fetch(MAP_SHELL, { method: 'HEAD', cache: 'no-cache' }).then(function (r) {
      if (st !== authorMap) return;
      if (!r.ok) { mapStatus(st, 'The map view is not published yet. The lists below hold the same information as text.'); return; }
      frame.src = MAP_SHELL;
    }, function () { mapStatus(st, 'The map could not be loaded.'); });
  }

  /* ---- the text beneath the map: the same content as lists ---- */
  function nodesOfType(g, type) {
    return (g.nodes || []).filter(function (n) { return n && n.type === type; });
  }
  function labelOf(n) { return cleanTitle(n && (n.title || n.label || n.name) || ''); }
  function papersCount(n) {
    var v = n && (n.n_papers != null ? n.n_papers : n.corpus_size);
    return (v != null && isFinite(Number(v))) ? Number(v) : null;
  }
  function countText(n) { var c = papersCount(n); return c == null ? '' : num(c) + ' ' + plural(c, 'paper'); }
  function yearOf(d) { var m = /^(\d{4})/.exec(String(d == null ? '' : d)); return m ? m[1] : ''; }
  function fillPrimary(st) {
    var box = viewEl && viewEl.querySelector('#avPrimary');
    if (!box || st !== authorMap) return;
    var s = String(st.res && st.res.summary || '').trim();
    var cap = st.res && st.res.stats && st.res.stats.cap_note;
    box.innerHTML = '<h3 id="avPrimaryH">Primary work</h3>' +
      (s ? '<p class="av-summary">' + esc(s) + '</p>' : '<p class="muted">No summary is available for this author yet.</p>') +
      (cap ? '<p class="mini muted">' + esc(cap) + '</p>' : '');
    box.hidden = false;
  }
  function personLi(p, kind) {
    var id = authorIdOf(p), name = p.name || p.label || 'Unnamed author';
    var meta = [];
    var place = placeLine(p.institution, p.country);
    if (place) meta.push(place);
    if (p.works_count != null) meta.push(num(p.works_count) + ' ' + plural(p.works_count, 'paper') + ' in all');
    if (kind === 'co') {
      if (p.n_joint != null) {
        var y = yearOf(p.last_joint);
        meta.push(num(p.n_joint) + ' joint ' + plural(p.n_joint, 'paper') + (y ? ', latest ' + esc(y) : ''));
      }
    }
    var extra = '';
    if (kind === 'co' && Number(p.reattributed) > 0) {
      extra = '<span class="mini muted">OpenAlex lists this person under ' + num((p.also_ids && p.also_ids.length || 1) + 1) +
        ' profiles; joint papers merged here.</span>';
    } else if (kind === 'ctx' && p.reason) {
      extra = '<span class="mini muted">' + esc(p.reason) + '</span>';
    }
    return '<li>' + (id ? '<a href="#author/' + id + '">' + esc(name) + '</a>' : esc(name)) +
      (meta.length ? '<span class="mini muted">' + meta.join(' &middot; ') + '</span>' : '') + extra + '</li>';
  }
  // Default people filters, the same rule the map applies: papers in all >= min_papers; a co-author also needs
  // >= min_joint joint papers unless a joint paper is recent. An unknown count never hides anyone.
  function passesFilter(p, f, co) {
    if (p.works_count != null && Number(p.works_count) < f.min_papers) return false;
    if (co && p.n_joint != null && Number(p.n_joint) < f.min_joint && !p.recent) return false;
    return true;
  }
  function fillLists(st) {
    var box = viewEl && viewEl.querySelector('#avLists');
    if (!box || st !== authorMap) return;
    var g = st.graph, f = st.filters;
    var fields = nodesOfType(g, 'oa_field'), subfields = nodesOfType(g, 'oa_subfield');
    var themes = nodesOfType(g, 'theme'), subs = (g.nodes || []).filter(function (n) { return n && n.type === 'cluster'; });
    var html = '';

    if (fields.length || subfields.length) {
      html += '<section class="av-sec" aria-labelledby="avFieldsH"><h3 id="avFieldsH">Fields and subfields</h3>' +
        '<p class="mini muted">OpenAlex fields and subfields of the lead and senior papers.</p><ul class="tree-list">';
      var used = Object.create(null);
      fields.forEach(function (fl) {
        var fid = String(fl.id || '').replace(/^oa_field::/, '');
        var kids = subfields.filter(function (s) { return String(s.field_id) === fid; });
        kids.forEach(function (s) { used[s.id] = 1; });
        html += '<li><b>' + esc(labelOf(fl)) + '</b>' + (countText(fl) ? ' <span class="mini muted">' + esc(countText(fl)) + '</span>' : '') +
          (kids.length ? '<ul>' + kids.map(function (s) {
            return '<li>' + esc(labelOf(s)) + (countText(s) ? ' <span class="mini muted">' + esc(countText(s)) + '</span>' : '') + '</li>';
          }).join('') + '</ul>' : '') + '</li>';
      });
      subfields.filter(function (s) { return !used[s.id]; }).forEach(function (s) {
        html += '<li>' + esc(labelOf(s)) + (countText(s) ? ' <span class="mini muted">' + esc(countText(s)) + '</span>' : '') + '</li>';
      });
      html += '</ul></section>';
    }

    if (themes.length || subs.length) {
      html += '<section class="av-sec" aria-labelledby="avThemesH"><h3 id="avThemesH">Themes and subthemes</h3>' +
        '<p class="mini muted">Titles are written by our local model from the paper titles; the counts and years are from the papers.</p><ul class="tree-list">';
      var usedSub = Object.create(null);
      function subLi(s) {
        return '<li>' + esc(labelOf(s)) + (s.overview ? ' <span class="mini muted">' + esc(s.overview) + '</span>' : '') + '</li>';
      }
      themes.forEach(function (t) {
        var kids = subs.filter(function (s) { return s.theme != null && s.theme === t.theme; });
        kids.forEach(function (s) { usedSub[s.id] = 1; });
        html += '<li><b>' + esc(labelOf(t)) + '</b>' + (t.overview ? ' <span class="mini muted">' + esc(t.overview) + '</span>' : '') +
          (kids.length ? '<ul>' + kids.map(subLi).join('') + '</ul>' : '') + '</li>';
      });
      subs.filter(function (s) { return !usedSub[s.id]; }).forEach(function (s) { html += subLi(s); });
      html += '</ul></section>';
    }

    var people = nodesOfType(g, 'author');
    var co = people.filter(function (p) { return p.role === 'coauthor'; }).sort(function (a, b) {
      return (Number(b.n_joint) || 0) - (Number(a.n_joint) || 0) || String(b.last_joint || '').localeCompare(String(a.last_joint || ''));
    });
    var cx = people.filter(function (p) { return p.role === 'context'; });
    var coShown = st.showAllCo ? co : co.filter(function (p) { return passesFilter(p, f, true); });
    var cxShown = st.showAllCo ? cx : cx.filter(function (p) { return passesFilter(p, f, false); });
    var hidden = (co.length - coShown.length) + (cx.length - cxShown.length);
    html += '<section class="av-sec" aria-labelledby="avCoH"><h3 id="avCoH">Co-authors</h3>' +
      '<p class="mini muted">Shown with at least ' + num(f.min_papers) + ' papers in all and at least ' + num(f.min_joint) +
      ' joint papers, or a joint paper from the last ' + num(f.recent_months) + ' months.</p>' +
      (coShown.length ? '<ul class="person-list">' + coShown.map(function (p) { return personLi(p, 'co'); }).join('') + '</ul>'
                      : '<p class="muted">No co-authors meet these limits.</p>') + '</section>';
    html += '<section class="av-sec" aria-labelledby="avCxH"><h3 id="avCxH">Context authors</h3>' +
      '<p class="mini muted">People who publish on the same subjects, with at least ' + num(f.min_papers) + ' papers in all.</p>' +
      (cxShown.length ? '<ul class="person-list">' + cxShown.map(function (p) { return personLi(p, 'ctx'); }).join('') + '</ul>'
                      : '<p class="muted">No context authors meet this limit.</p>') + '</section>';
    if (hidden > 0 || st.showAllCo) {
      html += '<p class="btn-row"><button type="button" class="btn ghost sm" id="avShowAll" aria-pressed="' + (st.showAllCo ? 'true' : 'false') + '">' +
        (st.showAllCo ? 'Apply the limits again' : 'Show all (' + num(hidden) + ' more)') + '</button></p>';
    }
    box.innerHTML = html;
    on($('#avShowAll', box), 'click', function () { st.showAllCo = !st.showAllCo; fillLists(st); });
  }

  /* ---- papers, from the profile (all works) ---- */
  var PAPERS_SHOWN = 50;
  function roleName(r) {
    if (typeof r === 'string') return r.toLowerCase();
    if (typeof r === 'number') return ['first', 'middle', 'last'][r & 3] || '';
    return '';
  }
  function profilePapersHtml(st) {
    var list = st.papers.slice();
    if (st.paperSort === 'year') list.sort(function (a, b) { return String(b.date || b.year || '').localeCompare(String(a.date || a.year || '')) || (Number(b.cited) || 0) - (Number(a.cited) || 0); });
    else list.sort(function (a, b) { return (Number(b.cited) || 0) - (Number(a.cited) || 0) || (Number(b.year) || 0) - (Number(a.year) || 0); });
    var total = list.length, shown = st.paperAll ? list : list.slice(0, PAPERS_SHOWN);
    var subTitle = Object.create(null);
    (st.graph.nodes || []).forEach(function (n) { if (n && n.type === 'cluster' && n.id) subTitle[String(n.id).replace(/^sub::/, '')] = labelOf(n); });
    var head = '<p class="mini muted">' + num(total) + ' ' + plural(total, 'paper') + (total > shown.length ? ', showing ' + num(shown.length) : '') + '.</p>';
    var more = total > shown.length ? '<p class="btn-row"><button type="button" class="btn ghost sm" id="avPapersAll">Show all ' + num(total) + ' papers</button></p>' : '';
    return head + '<ol class="paper-list">' + shown.map(function (p) {
      var u = doiUrl(p.doi), t = esc(cleanTitle(p.title) || 'Untitled');
      var r = roleName(p.role), badges = '';
      if (r === 'first') badges += '<span class="role-badge">First author</span>';
      else if (r === 'last') badges += '<span class="role-badge">Last author</span>';
      if (p.corresponding || (typeof p.role === 'number' && (p.role & 4))) badges += '<span class="role-badge corr">Corresponding</span>';
      if (p.claimed_from) badges += '<span class="role-badge claimed" title="OpenAlex lists this paper under a different profile">Filed by OpenAlex under another profile</span>';
      var meta = [];
      if (p.cited != null) meta.push(num(p.cited) + ' ' + plural(p.cited, 'citation'));
      var st2 = p.subtheme ? (subTitle[p.subtheme] || (/^S\d+$/.test(String(p.subtheme)) ? '' : String(p.subtheme))) : '';
      if (st2) meta.push('Subtheme: ' + esc(st2));
      var subj = nameOf(p.topic) || nameOf(p.subfield);
      if (subj) meta.push(esc(subj));
      return '<li><span class="pl-year">' + esc(p.year || '') + '</span><div class="pl-body">' +
        (u ? '<a href="' + esc(u) + '" target="_blank" rel="noopener">' + t + '</a>' : '<span>' + t + '</span>') +
        (badges ? ' <span class="badges">' + badges + '</span>' : '') +
        (meta.length ? '<div class="mini muted">' + meta.join(' &middot; ') + '</div>' : '') +
        '</div></li>';
    }).join('') + '</ol>' + more;
  }
  function fillProfilePapers(st) {
    var box = $('#avPapers');
    var all = st.res && st.res.lists && st.res.lists.papers;
    if (!box || st !== authorMap || !Array.isArray(all) || !all.length) return;
    st.papers = all.filter(function (p) { return p && p.title; });
    st.pagePapers = st.papers.map(function (p) { return { title: p.title, year: p.year, doi: p.doi }; });
    box.innerHTML = profilePapersHtml(st);
    wirePapers(st, box);
  }
  function wirePapers(st, box) {
    on($('#avPapersAll', box), 'click', function () { st.paperAll = true; box.innerHTML = profilePapersHtml(st); wirePapers(st, box); });
  }

  // A tapped person: their own co-authors and context authors, from the people route (no model, no embeddings).
  function expandPerson(st, pid) {
    if (!/^A\d{1,12}$/.test(pid) || st.expanded[pid] || st.busy) return;
    st.busy = true;
    mapStatus(st, 'Adding that person’s co-authors and context authors…');
    cached('people:' + pid, function () { return api.authorPeople(pid); }).then(function (res) {
      st.busy = false;
      if (st !== authorMap) return;
      if (!res || res.ok === false || !Array.isArray(res.authors)) {
        var em = (res && res.status === 404) ? 'That person has nobody more to add on this map.' : errorText(res);
        mapStatus(st, em);
        if (st.frame && st.frame.contentWindow) st.frame.contentWindow.postMessage({ type: 'pv:author-error', message: em }, location.origin);
        return;
      }
      st.expanded[pid] = 1;
      // Merge in place with pv:author-nodes (mode expand): the shell adds the nodes and keeps the clicked person's
      // panel and the camera. Re-posting pv:graph would re-mount the shell and replace the panel.
      var have = Object.create(null), people = 0;
      st.graph.nodes.forEach(function (n) { if (n && n.id != null) { have[n.id] = 1; if (n.type === 'author') people++; } });
      var cand = [], anchorRec = null;
      // The clicked person keeps the role they already have on this map (a co-author stays a co-author, with the
      // shared works and reasons this map gave them); their own list only adds what is new. Someone not yet on
      // the map comes in as a context author.
      var current = null;
      st.graph.nodes.forEach(function (n) { if (n && n.id === pid && n.type === 'author') current = n; });
      (res.authors || []).forEach(function (n) {
        if (!n || n.id == null) return;
        if (n.id === pid) {
          anchorRec = current ? Object.assign({}, n, current) : Object.assign({}, n, { type: 'author', role: 'context' });
          return;
        }
        if (!have[n.id]) cand.push(Object.assign({}, n, { type: 'author' }));
      });
      cand.sort(function (x, y) {
        return (y.role === 'coauthor' ? 1 : 0) - (x.role === 'coauthor' ? 1 : 0) || (Number(y.works_count) || 0) - (Number(x.works_count) || 0);
      });
      var room = Math.max(0, Math.min(24, MAP_MAX_PEOPLE - people, MAP_MAX_NODES - st.graph.nodes.length));
      var sent = cand.slice(0, room), dropped = cand.length - sent.length, added = sent.length;
      var authors = (anchorRec ? [anchorRec] : []).concat(sent), ids = Object.create(null);
      authors.forEach(function (n) { ids[n.id] = 1; });
      sent.forEach(function (n) { have[n.id] = 1; st.graph.nodes.push(n); });
      var edges = [], ekeys = Object.create(null);
      (res.edges || []).forEach(function (e) {
        if (!e || !ids[e.source] || !ids[e.target]) return;
        if (e.edge_type && e.edge_type !== 'author_link') return;
        var k = e.source + '|' + e.target;
        if (ekeys[k]) return;
        ekeys[k] = 1; edges.push({ source: e.source, target: e.target, kind: 'coauthor', weight: e.weight });
        st.graph.edges.push(e);
      });
      if (st.frame && st.frame.contentWindow) {
        st.frame.contentWindow.postMessage({ type: 'pv:author-nodes', mode: 'expand', anchor: pid, authors: authors, edges: edges }, location.origin);
      }
      mapStatus(st, added ? ('Added ' + num(added) + ' ' + plural(added, 'person') + (dropped ? '; ' + num(dropped) + ' more not shown (the map keeps a size cap).' : '.'))
                          : (dropped ? 'The map is at its size cap; nothing more was added.' : 'Nothing new to add for that person.'));
    });
  }
  on(window, 'message', function (ev) {
    var st = authorMap;
    if (!st || !st.frame || !st.frame.contentWindow || ev.source !== st.frame.contentWindow) return;
    if (ev.origin !== location.origin) return;
    var d = ev.data;
    if (!d || typeof d !== 'object' || typeof d.type !== 'string') return;
    var pid = typeof d.id === 'string' ? d.id : (typeof d.author_id === 'string' ? d.author_id : '');
    if (d.type === 'pv:map-ready') { if (!st.sent) sendGraph(st); }
    else if (d.type === 'pv:graph-ready') st.ready = true;
    else if (d.type === 'pv:graph-error') mapStatus(st, 'The map could not be drawn. The lists below hold the same information as text.');
    else if (d.type === 'pv:open-author' && /^A\d{1,12}$/.test(pid)) go('#author/' + pid);
    else if (d.type === 'pv:author-click' && /^A\d{1,12}$/.test(pid)) expandPerson(st, pid);
  });

  /* ------------------------------------------------------------ author exports */

  function saveBlob(blob, name) {
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 1500);
  }
  // Minimal RFC 4180 reader for the papers CSV the backend writes (quotes, doubled quotes, newlines in fields).
  function parseCsv(text) {
    var rows = [], row = [], f = '', q = false, i = 0, c;
    text = String(text || '').replace(/^﻿/, '');
    for (; i < text.length; i++) {
      c = text.charAt(i);
      if (q) {
        if (c === '"') { if (text.charAt(i + 1) === '"') { f += '"'; i++; } else q = false; } else f += c;
      } else if (c === '"') q = true;
      else if (c === ',') { row.push(f); f = ''; }
      else if (c === '\n' || c === '\r') {
        if (c === '\r' && text.charAt(i + 1) === '\n') i++;
        row.push(f); f = ''; if (row.length > 1 || row[0] !== '') rows.push(row); row = [];
      } else f += c;
    }
    if (f !== '' || row.length) { row.push(f); rows.push(row); }
    return rows;
  }
  function bibEsc(s) { return String(s == null ? '' : s).replace(/[{}\\]/g, '').replace(/\s+/g, ' ').trim(); }
  function toBibtex(works) {
    return works.map(function (w, i) {
      var doi = String(w.doi || '').replace(/^https?:\/\/(dx\.)?doi\.org\//i, '');
      var key = 'pubverse' + (doi ? doi.replace(/[^a-zA-Z0-9]/g, '') : (i + 1));
      var lines = ['@article{' + key + ','];
      if (w.title) lines.push('  title = {' + bibEsc(w.title) + '},');
      if (w.venue) lines.push('  journal = {' + bibEsc(w.venue) + '},');
      if (w.year) lines.push('  year = {' + bibEsc(w.year) + '},');
      if (doi) { lines.push('  doi = {' + bibEsc(doi) + '},'); lines.push('  url = {https://doi.org/' + bibEsc(doi) + '},'); }
      lines.push('}');
      return lines.join('\n');
    }).join('\n\n') + '\n';
  }
  function papersFromCsv(text) {
    var rows = parseCsv(text);
    if (rows.length < 2) return [];
    var head = rows[0].map(function (h) { return String(h).trim().toLowerCase(); });
    function col() { for (var a = 0; a < arguments.length; a++) { var k = head.indexOf(arguments[a]); if (k >= 0) return k; } return -1; }
    var ti = col('title'), yi = col('year'), vi = col('venue / source', 'venue', 'source', 'venue/source', 'journal'), di = col('doi');
    return rows.slice(1).map(function (r) {
      return { title: ti >= 0 ? r[ti] : '', year: yi >= 0 ? r[yi] : '', venue: vi >= 0 ? r[vi] : '', doi: di >= 0 ? r[di] : '' };
    }).filter(function (w) { return w.title; });
  }
  function wireExports(st) {
    var box = viewEl && viewEl.querySelector('#avExport');
    if (!box) return;
    var say = box.querySelector('.ex-status');
    function msg(t) { if (say) say.textContent = t || ''; }
    function download(kind, sheet, fname) {
      msg('Preparing the file\u2026');
      function viaFetch() {
        api.authorExportBlob(st.id, kind, sheet).then(function (r) {
          if (!r || !r.ok) { msg((r && r.message) || 'The export could not be made. Please try again.'); return; }
          saveBlob(r.blob, r.filename || fname);
          msg('Download started.');
        });
      }
      // The Compass way: a short-lived, author-scoped export token in the link, never the session token.
      // The one the map call returned is reused while it is young; after that a fresh one is asked for, so a
      // page left open past the token's life never hands out a dead link.
      var young = st.token && st.tokenAt && Date.now() - st.tokenAt < 8 * 60 * 1000;
      (young ? Promise.resolve({ export_token: st.token }) : api.authorExportToken(st.id)).then(function (t) {
        if (t && t.ok !== false && t.export_token) {
          var a = document.createElement('a');
          a.href = api.authorExportUrl(st.id, kind, sheet, t.export_token);
          a.setAttribute('download', fname);
          document.body.appendChild(a); a.click(); document.body.removeChild(a);
          msg('Download started.');
        } else viaFetch();
      });
    }
    var tag = 'pubverse-author-' + st.id;
    Array.prototype.forEach.call(box.querySelectorAll('button[data-ex]'), function (b) {
      on(b, 'click', function () {
        var k = b.getAttribute('data-ex');
        if (k === 'xlsx') download('xlsx', '', tag + '.xlsx');
        else if (k === 'bib') {
          msg('Preparing the file…');
          api.authorExportBlob(st.id, 'csv', 'papers').then(function (r) {
            if (r && r.ok) return r.blob.text().then(papersFromCsv);
            return null;
          }).catch(function () { return null; }).then(function (list) {
            if (!list || !list.length) list = (st.pagePapers || []).map(function (p) {
              return { title: cleanTitle(p.title), year: p.year, doi: p.doi, venue: (p.venue || p.source || '') };
            });
            if (!list.length) { msg('There are no papers to export.'); return; }
            saveBlob(new Blob([toBibtex(list)], { type: 'application/x-bibtex' }), tag + '-papers.bib');
            msg('Download started: ' + num(list.length) + ' ' + plural(list.length, 'entry', 'entries') + '.');
          });
        } else download('csv', k, tag + '-' + k + '.csv');
      });
    });
  }

  function renderAuthor(id, res) {
    var a = res.author || {};
    var name = nameOf(a) || 'Unnamed author';
    var orcid = normOrcid(a.orcid);
    var inst = a.institution || a.last_institution || '';
    var ctry = a.country || a.last_institution_country || '';
    var years = (a.first_year && a.last_year)
      ? (a.first_year === a.last_year ? 'Active ' + esc(a.first_year) : 'Active ' + esc(a.first_year) + '&ndash;' + esc(a.last_year))
      : '';
    var alt = (a.alt_names || []).map(nameOf).filter(function (x) { return x && x !== name; });
    var seen = Object.create(null);
    alt = alt.filter(function (x) { var k = x.toLowerCase(); if (seen[k]) return false; seen[k] = 1; return true; }).slice(0, 6);
    setTitle(name);

    var sub = [placeLine(inst, ctry), years].filter(Boolean).join(' &middot; ');
    var html = backLink() + '<article class="card av" aria-labelledby="avName">' +
      '<header class="av-head"><h2 id="avName">' + esc(name) + '</h2>' +
      (sub ? '<p class="av-sub">' + sub + '</p>' : '') +
      (orcid ? '<p class="av-orcid">' + orcidBadge(orcid, true) + '</p>' : '') +
      (alt.length ? '<p class="mini muted">Also published as ' + alt.map(esc).join('; ') + '</p>' : '') +
      suspectNote(a) + '</header>';

    // The record facts: counts from the public publication record, nothing scored.
    html += '<section class="av-sec" aria-labelledby="avRecord"><h3 id="avRecord">Publication record</h3>' +
      '<dl class="facts">' +
        fact('h-index', num(a.h_index)) +
        fact('Works', num(a.works_count)) +
        fact('Citations', num(a.cited_by_count)) +
        (a.i10_index != null ? fact('i10-index', num(a.i10_index), 'Works cited at least 10 times') : '') +
        (a.last_author_works != null ? fact('Last author', num(a.last_author_works), 'Papers as last author') : '') +
        (a.corresponding_works != null ? fact('Corresponding', num(a.corresponding_works), 'Papers as corresponding author') : '') +
        fact('ORCID', orcid ? 'Linked' : 'None') +
      '</dl>' +
      '<p class="mini muted">Counts come from the OpenAlex snapshot PubVerse indexes, so recent papers may be missing.</p></section>';

    // Their space: subfields first (broad), then topics (narrow), deduplicated by name.
    var chips = [], have = Object.create(null);
    (a.top_subfields || a.subfields || []).forEach(function (s) { var n = nameOf(s); if (n && !have[n.toLowerCase()]) { have[n.toLowerCase()] = 1; chips.push(subjectChip(n, 'subfield')); } });
    (a.top_topics || a.topics || []).forEach(function (s) { var n = nameOf(s); if (n && !have[n.toLowerCase()]) { have[n.toLowerCase()] = 1; chips.push(subjectChip(n, '')); } });
    if (chips.length) {
      html += '<section class="av-sec" aria-labelledby="avSpace"><h3 id="avSpace">Their space</h3>' +
        '<p class="mini muted">Open one of their subjects in the Compass to see what is new in it.</p>' +
        '<div class="chips">' + chips.slice(0, 12).join('') + '</div></section>';
    }

    html += '<section class="av-sec" id="avPrimary" aria-labelledby="avPrimaryH"><h3 id="avPrimaryH">Primary work</h3>' +
      '<p class="muted">Writing the summary…</p></section>';

    html += '<section class="av-sec ag-sec" aria-labelledby="avNetH"><h3 id="avNetH">Author map</h3>' +
      '<p class="mini muted">' + esc(name) + ' at the center, linked to the themes and subthemes of their lead and senior papers ' +
      'and to the OpenAlex fields and subfields those papers sit in. Co-authors and context authors appear with at least 5 papers; ' +
      'change this under Advanced on the map.</p>' +
      '<div id="avGraph"></div></section>';

    html += '<div id="avLists"></div>';

    html += '<section class="av-sec" aria-labelledby="avExH"><h3 id="avExH">Export</h3>' +
      '<div class="btn-row ex-row" id="avExport">' +
        '<button type="button" class="btn ghost sm" data-ex="xlsx">Export (Excel)</button>' +
        '<span class="ex-group" role="group" aria-label="CSV"><span class="ex-lab">CSV</span>' +
          '<button type="button" class="btn ghost sm" data-ex="papers">Papers</button>' +
          '<button type="button" class="btn ghost sm" data-ex="coauthors">Co-authors</button>' +
          '<button type="button" class="btn ghost sm" data-ex="context">Context authors</button></span>' +
        '<button type="button" class="btn ghost sm" data-ex="bib">BibTeX (papers)</button>' +
        '<span class="mini muted ex-status" role="status" aria-live="polite"></span></div>' +
      '<p class="mini muted">The workbook has four sheets: author, papers, co-authors and context authors (with the subthemes they share).</p></section>';

    html += '<div id="avMaps"></div>';

    html += '<section class="av-sec" aria-labelledby="avPapersH">' +
      '<div class="row between rwrap av-papers-head"><h3 id="avPapersH">Papers</h3>' +
      '<div class="seg" role="group" aria-label="Sort papers">' +
        '<button type="button" data-sort="" aria-pressed="true">Most cited</button>' +
        '<button type="button" data-sort="year" aria-pressed="false">Newest</button></div></div>' +
      '<div id="avPapers">' + papersHtml(res.papers || [], res.papers_total) + '</div></section>';

    var funders = (a.top_funders || []).filter(function (f) { return nameOf(f); });
    if (funders.length) {
      html += '<section class="av-sec" aria-labelledby="avFundH"><h3 id="avFundH">Funders</h3><div class="chips">' +
        funders.slice(0, 8).map(function (f) {
          var fid = funderIdOf(f);
          var n = f.n != null ? ' <span class="chip-meta">at least ' + num(f.n) + ' ' + plural(f.n, 'paper') + covMark(f.coverage_note) + '</span>' : '';
          return fid ? '<a class="chip" href="#funder/' + fid + '">' + esc(nameOf(f)) + n + '</a>'
                     : '<span class="chip">' + esc(nameOf(f)) + n + '</span>';
        }).join('') + '</div>' +
        '<p class="mini muted">' + esc(funders[0].coverage_note || COV_NOTE) + ' Unlisted does not mean unfunded.</p></section>';
    }
    html += '</article>';
    viewEl.innerHTML = html;
    focusHeading();
    mountGraph(id);
    if (authorMap) authorMap.pagePapers = res.papers || [];
    fillAuthorMaps(id, (a.top_subfields || a.subfields || []));

    // Sort toggle: refetch rather than re-sort, because "newest" is a different top 50.
    Array.prototype.forEach.call(viewEl.querySelectorAll('.seg button'), function (b) {
      on(b, 'click', function () {
        if (b.getAttribute('aria-pressed') === 'true') return;
        var my = viewSeq, sort = b.getAttribute('data-sort');
        Array.prototype.forEach.call(viewEl.querySelectorAll('.seg button'), function (x) {
          x.setAttribute('aria-pressed', x === b ? 'true' : 'false');
        });
        var box = $('#avPapers');
        box.setAttribute('aria-busy', 'true');
        box.innerHTML = spinnerHtml('Sorting');
        if (authorMap && authorMap.papers) {
          authorMap.paperSort = sort || ''; box.removeAttribute('aria-busy');
          box.innerHTML = profilePapersHtml(authorMap); wirePapers(authorMap, box); return;
        }
        cached('author:' + id + (sort ? ':' + sort : ''), function () { return api.author(id, sort || undefined); }).then(function (r) {
          if (my !== viewSeq) return;
          box.removeAttribute('aria-busy');
          if (!r || r.ok === false) { box.innerHTML = errHtml(errorText(r)); return; }
          box.innerHTML = papersHtml(r.papers || [], r.papers_total);
        });
      });
    });
  }

  var ROLE = { first: 'First author', last: 'Last author' };
  function papersHtml(papers, total) {
    if (!papers.length) return '<p class="muted">No papers listed for this author.</p>';
    var head = (Number(total) > papers.length)
      ? '<p class="mini muted">Showing ' + num(papers.length) + ' of ' + num(total) + ' papers.</p>' : '';
    return head + '<ol class="paper-list">' + papers.map(function (p) {
      var u = doiUrl(p.doi), t = esc(cleanTitle(p.title) || 'Untitled');
      var badges = '';
      if (ROLE[p.role]) badges += '<span class="role-badge">' + ROLE[p.role] + '</span>';
      if (p.is_corresponding) badges += '<span class="role-badge corr">Corresponding</span>';
      if (p.retracted) badges += '<span class="role-badge retracted">Retracted</span>';
      var meta = [];
      if (p.cited_by_count != null) meta.push(num(p.cited_by_count) + ' ' + plural(p.cited_by_count, 'citation'));
      var subj = nameOf(p.topic) || nameOf(p.subfield);
      if (subj) meta.push(esc(subj));
      return '<li><span class="pl-year">' + esc(p.year || '') + '</span><div class="pl-body">' +
        (u ? '<a href="' + esc(u) + '" target="_blank" rel="noopener">' + t + '</a>' : '<span>' + t + '</span>') +
        (badges ? ' <span class="badges">' + badges + '</span>' : '') +
        (meta.length ? '<div class="mini muted">' + meta.join(' &middot; ') + '</div>' : '') +
        '</div></li>';
    }).join('') + '</ol>';
  }

  /* ------------------------------------------------------------- funder view */

  function openFunder(id) {
    var my = viewSeq;
    beginView('Loading the funder');
    cached('funder:' + id, function () { return api.funder(id); }).then(function (res) {
      if (my !== viewSeq) return;
      if (!res || res.ok === false || !res.funder) { viewFailed(res); return; }
      renderFunder(res);
    });
  }
  function renderFunder(res) {
    var f = res.funder || {};
    var name = nameOf(f) || 'Unnamed funder';
    setTitle(name);
    var aliases = (f.aliases || []).map(nameOf).filter(function (x) { return x && x !== name; }).slice(0, 8);
    var subs = (res.subfields || f.top_subfields || f.subfields || []).filter(function (s) { return nameOf(s); });
    var people = (res.top_authors || f.top_authors || []).filter(Boolean);
    var html = backLink() + '<article class="card av" aria-labelledby="fvName">' +
      '<header class="av-head"><h2 id="fvName">' + esc(name) + '</h2>' +
      (f.country ? '<p class="av-sub">' + esc(countryName(f.country)) + '</p>' : '') +
      (aliases.length ? '<p class="mini muted">Also known as ' + aliases.map(esc).join('; ') + '</p>' : '') +
      '</header>' +
      '<section class="av-sec"><dl class="facts">' +
        fact('Funded papers (at least)', num(f.works_count) + covMark(f.coverage_note)) +
        (f.cited_by_count != null ? fact('Citations to them', num(f.cited_by_count)) : '') +
      '</dl></section>';
    if (subs.length) {
      html += '<section class="av-sec" aria-labelledby="fvSubs"><h3 id="fvSubs">Top subfields</h3><div class="chips">' +
        subs.slice(0, 10).map(function (s) {
          return subjectChip(nameOf(s), s.n != null ? 'at least ' + num(s.n) + ' ' + plural(s.n, 'paper') : '');
        }).join('') + '</div></section>';
    }
    if (people.length) {
      html += '<section class="av-sec" aria-labelledby="fvPeople"><h3 id="fvPeople">Authors it funds most</h3>' +
        '<ul class="person-list">' + people.slice(0, 15).map(function (p) {
          var aid = authorIdOf(p), n = nameOf(p) || 'Unnamed author';
          var meta = [];
          if (p.institution) meta.push(esc(p.institution));
          if (p.n != null) meta.push('at least ' + num(p.n) + ' funded ' + plural(p.n, 'paper'));
          return '<li>' + (aid ? '<a href="#author/' + aid + '">' + esc(n) + '</a>' : esc(n)) +
            (meta.length ? '<span class="mini muted">' + meta.join(' &middot; ') + '</span>' : '') + '</li>';
        }).join('') + '</ul></section>';
    }
    html += '<p class="coverage-note">' + esc(f.coverage_note || res.coverage_note || COV_NOTE) +
      ' arXiv carries none. Unlisted does not mean unfunded.</p>';
    html += '</article>';
    viewEl.innerHTML = html;
    focusHeading();
  }

  /* -------------------------------------------------------------- topic view */

  function openTopic(id) {
    var my = viewSeq;
    beginView('Loading the people in this area');
    cached('topic:' + id, function () { return api.topicPeople(id, 25); }).then(function (res) {
      if (my !== viewSeq) return;
      if (!res || res.ok === false || !res.topic) { viewFailed(res); return; }
      renderTopic(res);
    });
  }
  function renderTopic(res) {
    var t = res.topic || {};
    var name = nameOf(t) || 'Unnamed subject';
    setTitle(name);
    var people = (res.people || []).filter(Boolean);
    var sub = [LEVEL[t.level] || 'Subject'];
    if (t.parent) sub.push('in ' + t.parent);
    var html = backLink() + '<article class="card av" aria-labelledby="tvName">' +
      '<header class="av-head"><h2 id="tvName">' + esc(name) + '</h2>' +
      '<p class="av-sub">' + sub.map(esc).join(' ') + '</p></header>' +
      '<section class="av-sec"><dl class="facts">' +
        (t.works_count != null ? fact('Papers', num(t.works_count)) : '') +
      '</dl><p class="btn-row"><button type="button" class="btn" id="tvCompass" data-subject="' + esc(name) + '">Open in Compass</button>' +
      '<span id="tvMap"></span></p></section>' +
      '<section class="av-sec" aria-labelledby="tvPeople"><h3 id="tvPeople">People in this area</h3>';
    if (!people.length) {
      html += '<p class="muted">No authors are listed for this area yet.</p>';
    } else {
      html += '<ul class="person-list">' + people.map(function (p) {
        var aid = authorIdOf(p), n = nameOf(p) || 'Unnamed author';
        var meta = [];
        var place = placeLine(p.institution, p.country);
        if (place) meta.push(place);
        if (p.h_index != null) meta.push('h-index ' + num(p.h_index));
        if (p.first_year && p.last_year) meta.push('papers ' + esc(p.first_year) + '&ndash;' + esc(p.last_year));
        return '<li>' + (aid ? '<a href="#author/' + aid + '">' + esc(n) + '</a>' : esc(n)) +
          (normOrcid(p.orcid) ? ' ' + orcidBadge(p.orcid, false) : '') +
          (meta.length ? '<span class="mini muted">' + meta.join(' &middot; ') + '</span>' : '') + '</li>';
      }).join('') + '</ul>' +
      '<p class="mini muted">People with a paper since ' + esc(res.active_since || '') + ', ordered by publication record' +
      '' + (res.data_as_of ? ' (data through ' + esc(res.data_as_of) + ')' : '') + '. ' +
      '<a href="docs.html#search">How authors are ranked</a></p>';
    }
    html += '</section></article>';
    viewEl.innerHTML = html;
    focusHeading();
    fillTopicMap(t.id);
  }

  /* ------------------------------------------------ research-space maps */

  // A space map exists only for the spaces listed in examples/spaces/index.json. The buttons below
  // appear only when the list says so; a missing or empty list leaves the pages exactly as they were.
  function fillAuthorMaps(authorId, subfields) {
    var bridge = window.PV && PV.mapBridge;
    if (!bridge) return;
    bridge.spaces().then(function (list) {
      var slot = viewEl && viewEl.querySelector('#avMaps');
      if (!slot) return;
      var seen = Object.create(null), links = [];
      (subfields || []).forEach(function (sf) {
        var sp = bridge.spaceForSubfield(list, sf && (sf.id != null ? sf.id : sf));
        if (!sp || seen[sp.slug]) return;
        seen[sp.slug] = 1;
        links.push('<a class="btn ghost sm" href="#space/' + esc(sp.slug) + '?author=' + esc(authorId) + '">Show on the map' +
          ': ' + esc(sp.name) + '</a>');
      });
      if (!links.length) return;
      slot.innerHTML = '<section class="av-sec" aria-labelledby="avMapsH"><h3 id="avMapsH">On the map</h3>' +
        '<p class="mini muted">See where this author sits among the themes of a research space, and who works beside them.</p>' +
        '<div class="btn-row">' + links.join('') + '</div>' +
        '<p class="mini muted">Authors are placed from papers through March 2026; newer preprints are not placed yet. ' +
        '<a href="docs.html#maps">How authors appear on maps</a></p></section>';
    });
  }
  function fillTopicMap(topicId) {
    var bridge = window.PV && PV.mapBridge;
    if (!bridge || !/^subfield-\d{1,9}$/.test(String(topicId || ''))) return;
    bridge.spaces().then(function (list) {
      var slot = viewEl && viewEl.querySelector('#tvMap');
      var sp = bridge.spaceForSubfield(list, topicId);
      if (!slot || !sp) return;
      slot.innerHTML = '<a class="btn ghost" href="#space/' + esc(sp.slug) + '">Open the map</a>';
    });
  }

  var spaceFrame = null, spaceTimers = [];
  function releaseSpaceFrame() {
    destroyGraph();
    spaceTimers.forEach(clearTimeout);
    spaceTimers = [];
    if (spaceFrame && window.PV && PV.mapBridge) PV.mapBridge.detach(spaceFrame);
    spaceFrame = null;
  }
  function openSpace(slug, authorId) {
    var my = viewSeq;
    beginView('Loading the map');
    var bridge = window.PV && PV.mapBridge;
    if (!bridge) { viewFailed(null); return; }
    bridge.spaces().then(function (list) {
      if (my !== viewSeq) return;
      var sp = bridge.spaceBySlug(list, slug);
      if (!sp) { viewEl.innerHTML = backLink() + errHtml('There is no map for that research space yet.'); return; }
      var src = '/examples/spaces/' + encodeURIComponent(sp.slug) + '.html';
      return fetch(src, { method: 'HEAD', cache: 'no-cache' }).then(function (r) {
        if (my !== viewSeq) return;
        if (!r.ok) { viewEl.innerHTML = backLink() + errHtml('The map for ' + sp.name + ' is not published yet.'); return; }
        renderSpace(sp, src, authorId, my);
      }, function () {
        if (my === viewSeq) viewEl.innerHTML = backLink() + errHtml('The map could not be loaded.');
      });
    });
  }
  function renderSpace(sp, src, authorId, my) {
    setTitle(sp.name + ' map');
    var facts = [];
    if (sp.n_papers != null) facts.push(num(sp.n_papers) + ' papers');
    if (sp.n_clusters != null) facts.push(num(sp.n_clusters) + ' subthemes');
    viewEl.innerHTML = backLink() + '<article class="card av space-view" aria-labelledby="spName">' +
      '<header class="av-head"><h2 id="spName">' + esc(sp.name) + '</h2>' +
      (facts.length ? '<p class="av-sub">' + esc(facts.join(' \u00b7 ')) + '</p>' : '') +
      '<p class="mini muted">Search a name or a subject in the map\u2019s search box. An author appears with their co-authors and context authors ' +
      'in this field, or the map says there is no recent work. Authors are placed from papers through March 2026. ' +
      '<a href="docs.html#maps">How authors appear on maps</a></p></header>' +
      '<div class="row between rwrap" style="gap:10px;margin:10px 0 8px">' +
        '<p class="mini muted space-status" id="spStatus" role="status" aria-live="polite"></p>' +
        '<button class="btn ghost sm" id="spFull" type="button" aria-controls="spaceMapFrame">Full screen</button></div>' +
      '<div class="space-map"><iframe id="spaceMapFrame" allow="fullscreen" allowfullscreen ' +
        'title="Interactive map of ' + esc(sp.name) + '"></iframe></div></article>';
    focusHeading();
    var frame = $('#spaceMapFrame'), st = $('#spStatus');
    function say(t) { if (st) st.textContent = t || ''; }
    releaseSpaceFrame();
    spaceFrame = frame;
    PV.mapBridge.attach(frame, { resolve: function () { return sp.slug; }, onStatus: say });
    on($('#spFull'), 'click', function () {
      var req = frame.requestFullscreen || frame.webkitRequestFullscreen;
      if (document.fullscreenElement) { document.exitFullscreen(); return; }
      if (req) { try { var p = req.call(frame); if (p && p.catch) p.catch(function () {}); } catch (e) {} }
    });
    if (authorId) {
      // The map mounts after its document loads; one placement shortly after load and one more a
      // little later, because the place message is idempotent and a first one can arrive before the
      // map is listening.
      on(frame, 'load', function () {
        say('Placing the author on the map\u2026');
        [250, 1800].forEach(function (ms) {
          spaceTimers.push(setTimeout(function () {
            if (my !== viewSeq || spaceFrame !== frame) return;
            PV.mapBridge.placeAuthor(frame, authorId);
          }, ms));
        });
      });
    }
    frame.src = src;
  }

  /* --------------------------------------------------- nav typeahead (combobox) */

  var navForm = $('#navSearch'), navInput = $('#navQ'), navMenu = $('#navSearchMenu');
  var navBtn = $('#navSearchBtn'), navClose = $('#navSearchClose'), navLive = $('#navSearchLive');
  var navEl = navForm && navForm.closest ? navForm.closest('.nav') : null;

  function overlayOpen() { return !!(navForm && navForm.classList.contains('open')); }
  function openOverlay() {
    if (!navForm) return;
    navForm.classList.add('open');
    // The nav is its own stacking context (sticky + backdrop-filter); lift it over the rail so the
    // full-width menu is not cut by the rail's strip down the left edge.
    if (navEl) navEl.classList.add('search-open');
    if (navBtn) navBtn.setAttribute('aria-expanded', 'true');
    if (navInput) navInput.focus();
  }
  function closeOverlay(refocus) {
    if (!overlayOpen()) return;
    navForm.classList.remove('open');
    if (navEl) navEl.classList.remove('search-open');
    if (navBtn) { navBtn.setAttribute('aria-expanded', 'false'); if (refocus) navBtn.focus(); }
  }
  on(navBtn, 'click', function () { overlayOpen() ? closeOverlay(true) : openOverlay(); });
  on(navClose, 'click', function () { ta.close(); closeOverlay(true); });
  // A tap outside the open phone search folds it away, like any other sheet.
  on(document, 'pointerdown', function (e) {
    if (!overlayOpen()) return;
    if (navForm.contains(e.target) || (navBtn && navBtn.contains(e.target))) return;
    ta.close();
    closeOverlay(false);
  });
  // "/" jumps to search from anywhere that is not already a text field.
  on(document, 'keydown', function (e) {
    if (e.key !== '/' || e.ctrlKey || e.metaKey || e.altKey || !authed) return;
    var t = e.target, tag = t && t.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || (t && t.isContentEditable)) return;
    if (!navInput) return;
    e.preventDefault();
    if (navInput.offsetParent === null) openOverlay(); else navInput.focus();
  });

  var ta = (function makeTypeahead(input, menu) {
    var timer = null, seq = 0, items = [], idx = -1, isOpen = false;
    function noop() {}
    if (!input || !menu) return { close: noop };

    function announce(t) { if (navLive) navLive.textContent = t; }
    function setOpen(v) {
      isOpen = v;
      input.setAttribute('aria-expanded', v ? 'true' : 'false');
      menu.hidden = !v;
      if (!v) { idx = -1; input.removeAttribute('aria-activedescendant'); }
    }
    function status(text, cls) {
      items = []; idx = -1;
      input.removeAttribute('aria-activedescendant');
      menu.innerHTML = '<div class="ta-status ' + (cls || '') + '">' + esc(text) + '</div>';
      setOpen(true);
    }
    function optHtml(x) {
      var it = x.it;
      if (x.kind === 'author') {
        var place = placeLine(it.institution, it.country);
        return '<span class="ta-main">' + esc(nameOf(it)) + (normOrcid(it.orcid) ? ' <span class="orcid-dot" title="Has an ORCID iD">iD</span>' : '') + '</span>' +
          '<span class="ta-meta">' + (place ? place + ' &middot; ' : '') + 'h-index ' + num(it.h_index) + '</span>' +
          (it.suspect_merge ? '<span class="ta-meta ta-warn">may merge several people</span>' : '');
      }
      if (x.kind === 'funder') {
        var al = (it.aliases || []).map(nameOf).filter(Boolean).slice(0, 2);
        return '<span class="ta-main">' + esc(nameOf(it)) + '</span>' +
          '<span class="ta-meta">' + (al.length ? esc(al.join(', ')) + ' &middot; ' : '') + 'at least ' + num(it.works_count) + ' funded papers' + covMark(it.coverage_note) + '</span>';
      }
      if (x.kind === 'topic') {
        return '<span class="ta-main">' + esc(nameOf(it)) + '</span>' +
          '<span class="ta-meta">' + esc(LEVEL[it.level] || 'Subject') + (it.parent ? ' in ' + esc(it.parent) : '') + (topicIdOf(it) && (it.level === 'topic' || it.level === 'subfield') ? ' &middot; people in this area' : ' &middot; open in Compass') + '</span>';
      }
      return '<span class="ta-main">See all ' + (scope === 'all' ? '' : esc(SCOPE_TEXT[scope].noun) + ' ') + 'results for “' + esc(x.q) + '”</span>';
    }
    function render(res, q) {
      items = []; idx = -1;
      var html = '', gi = 0;
      function group(label, list, kind) {
        if (!list.length) return;
        var gid = menu.id + 'G' + (gi++);
        html += '<div role="group" aria-labelledby="' + gid + '"><div class="ta-group" id="' + gid + '">' + esc(label) + '</div>';
        list.forEach(function (it) {
          var i = items.length;
          items.push({ kind: kind, it: it });
          html += '<div class="ta-opt" role="option" id="' + menu.id + 'O' + i + '" data-i="' + i + '" aria-selected="false">' +
            optHtml(items[i]) + '</div>';
        });
        html += '</div>';
      }
      var A = (res.authors || []).filter(function (a) { return a && authorIdOf(a); });
      var F = (res.funders || []).filter(function (f) { return f && funderIdOf(f); });
      var T = (res.topics || []).filter(function (t) { return t && nameOf(t); });
      if (!A.length && !F.length && !T.length) {
        status('Nothing matches “' + q + '” in ' + SCOPE_TEXT[scope].noun + '.', 'empty');
        announce('No matches.');
        return;
      }
      var many = scope !== 'all';
      group('Authors', A.slice(0, many ? 8 : 5), 'author');
      group('Funders', F.slice(0, many ? 8 : 3), 'funder');
      group('Subjects', T.slice(0, many ? 8 : 3), 'topic');
      var i = items.length;
      items.push({ kind: 'all', q: q });
      html += '<div class="ta-opt ta-all" role="option" id="' + menu.id + 'O' + i + '" data-i="' + i + '" aria-selected="false">' +
        optHtml(items[i]) + '</div>';
      menu.innerHTML = html;
      setOpen(true);
      announce([A.length && (A.length + ' ' + plural(A.length, 'author')), F.length && (F.length + ' ' + plural(F.length, 'funder')),
                T.length && (T.length + ' ' + plural(T.length, 'subject'))].filter(Boolean).join(', ') +
               '. Use the arrow keys to choose.');
    }
    function move(d) {
      if (!items.length) return;
      idx = idx < 0 ? (d > 0 ? 0 : items.length - 1) : (idx + d + items.length) % items.length;
      Array.prototype.forEach.call(menu.querySelectorAll('.ta-opt'), function (o) {
        var act = +o.getAttribute('data-i') === idx;
        o.classList.toggle('active', act);
        o.setAttribute('aria-selected', act ? 'true' : 'false');
        if (act) {
          input.setAttribute('aria-activedescendant', o.id);
          if (o.scrollIntoView) o.scrollIntoView({ block: 'nearest' });
        }
      });
    }
    function choose(i) {
      var x = items[i];
      if (!x) return;
      close();
      closeOverlay(false);
      input.value = '';
      input.blur();
      if (x.kind === 'author') go('#author/' + authorIdOf(x.it));
      else if (x.kind === 'funder') go('#funder/' + funderIdOf(x.it));
      else if (x.kind === 'topic') {
        if (topicIdOf(x.it) && (x.it.level === 'topic' || x.it.level === 'subfield')) go('#topic/' + topicIdOf(x.it));
        else openInCompass(nameOf(x.it));
      }
      else go('#search?q=' + encodeURIComponent(x.q) + scopeParam());
    }
    function submitAll() {
      var q = input.value.trim().slice(0, Q_MAX);
      if (!q) return;
      close();
      closeOverlay(false);
      input.blur();
      go('#search?q=' + encodeURIComponent(q) + scopeParam());
    }
    function fetchQ(q) {
      var my = ++seq;
      status('Searching…', 'loading');
      api.search(q, { limit: scope === 'all' ? 5 : 8, types: SCOPES[scope], timeout: 8000 }).then(function (res) {
        if (my !== seq || document.activeElement !== input) return;
        if (!res || res.ok === false) {
          var t = errorText(res);
          status(t, 'error');
          announce(t);
          return;
        }
        render(res, q);
      });
    }
    function close() { clearTimeout(timer); setOpen(false); }

    on(input, 'input', function () {
      var q = input.value.trim().slice(0, Q_MAX);
      clearTimeout(timer);
      if (q.length < 2) { seq++; close(); return; }
      timer = setTimeout(function () { fetchQ(q); }, 250);
    });
    on(input, 'keydown', function (e) {
      if (e.key === 'ArrowDown') { e.preventDefault(); if (!isOpen && items.length) setOpen(true); move(1); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); move(-1); }
      else if (e.key === 'Enter') { e.preventDefault(); if (isOpen && idx >= 0) choose(idx); else submitAll(); }
      else if (e.key === 'Escape') {
        if (isOpen) { e.preventDefault(); close(); }
        else if (overlayOpen()) { e.preventDefault(); closeOverlay(true); }
      }
      else if (e.key === 'Tab') close();
    });
    on(input, 'blur', function () { setTimeout(function () { if (document.activeElement !== input) setOpen(false); }, 150); });
    // mousedown would steal focus from the input and close the menu before the click lands.
    on(menu, 'mousedown', function (e) { e.preventDefault(); });
    on(menu, 'click', function (e) {
      var o = e.target.closest && e.target.closest('.ta-opt');
      if (o) choose(+o.getAttribute('data-i'));
    });
    on(navForm, 'submit', function (e) { e.preventDefault(); submitAll(); });
    return { close: close };
  })(navInput, navMenu);

  window.PV.search = { go: go, openInCompass: openInCompass };
})();
