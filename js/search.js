/* PubVerse search: authors, funders and subjects.

   Three surfaces, one API (js/api.js):
     - the nav box: a debounced typeahead grouped Authors / Funders / Subjects, ARIA combobox
       pattern, Up/Down/Enter/Esc. At phone width it is folded behind a search button and opens
       full width over the top of the page.
     - the Search tab: a large box, an Advanced disclosure with an ORCID field, and the results.
     - the author and funder views, shown inside the Search tab.

   Views are hash routes so they can be linked and the Back button works:
     #search?q=...   results            #author/A123   an author     #funder/F123   a funder
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
    if (!h || h === 'search') return { tab: 'search', view: 'home' };
    if ((m = /^author\/(A\d{1,12})$/.exec(h))) return { tab: 'search', view: 'author', id: m[1] };
    if ((m = /^topic\/((?:topic|subfield)-\d{1,9})$/.exec(h))) return { tab: 'search', view: 'topic', id: m[1] };
    if ((m = /^funder\/(F\d{1,12})$/.exec(h))) return { tab: 'search', view: 'funder', id: m[1] };
    if (h.indexOf('search?') === 0) {
      var q = '';
      try { q = new URLSearchParams(h.slice(7)).get('q') || ''; } catch (e) { q = ''; }
      q = q.trim().slice(0, Q_MAX);
      return q ? { tab: 'search', view: 'results', q: q } : { tab: 'search', view: 'home' };
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
    if (r.view === 'home') showHome();
    else if (r.view === 'results') { lastResults = location.hash; runSearch(r.q); }
    else if (r.view === 'author') openAuthor(r.id);
    else if (r.view === 'funder') openFunder(r.id);
    else if (r.view === 'topic') openTopic(r.id);
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
           'Its counts are probably too high, so it is ranked lower.</p>';
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
    go('#search?q=' + encodeURIComponent(q));
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
    go('#search?q=' + encodeURIComponent(o));
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
    if (qEl && qEl.value.trim() !== q) qEl.value = q;
    var o = normOrcid(q), of = $('#searchOrcid');
    if (o && of) { of.value = o; var adv = $('#searchAdv'); if (adv) adv.open = true; }
    clearErr();
    resultsEl.innerHTML = spinnerHtml('Searching');
    resultsEl.setAttribute('aria-busy', 'true');
    status('Searching.');
    cached('search:' + q, function () { return api.search(q, { limit: 20, types: 'authors,funders,topics' }); }).then(function (res) {
      if (my !== viewSeq) return;
      resultsEl.removeAttribute('aria-busy');
      if (!res || res.ok === false) { var t = errorText(res); resultsEl.innerHTML = errHtml(t); status(t); return; }
      resultsEl.innerHTML = resultsHtml(res, q);
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
  function resultsHtml(res, q) {
    var A = (res.authors || []).filter(Boolean), F = (res.funders || []).filter(Boolean);
    var T = (res.topics || []).filter(function (t) { return t && nameOf(t); });
    var est = res.estimated || {};
    var orcidMode = res.mode === 'orcid';
    if (!A.length && !F.length && !T.length) {
      return '<p class="sr-empty">' + (orcidMode
        ? 'No author with the ORCID iD <b>' + esc(normOrcid(q) || q) + '</b> is in the index yet.'
        : 'Nothing matches <b>' + esc(q) + '</b>. Check the spelling, try fewer words, or search by ORCID under Advanced.') +
        '</p>';
    }
    var html = '';
    if (orcidMode) html += '<p class="mini muted">Exact match on ORCID iD ' + esc(normOrcid(q) || q) + '.</p>';
    if (A.length) {
      html += groupHtml('srAuthors', 'Authors', est.authors, A.length, A.map(authorItem).join(''),
        '<p class="mini muted sr-note">Ranked by name match, then by publication record. ' +
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

  var CRED_NOTE = 'Search ranks authors by name match first, then by a credibility measure from 0 to 100 built ' +
    'only from the public publication record: 45% citations (log scale), 35% h-index, 10% for having an ' +
    'ORCID iD, and 10% for last-author and corresponding-author papers, each against fixed anchors so a ' +
    'score means the same after every rebuild. Profiles that look like several people merged under one name count a quarter. It orders search ' +
    'results only; it is never part of a PubVerse novelty, methods or impact score.';

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
      cached('nearby:' + id, function () { return api.authorNearby(id); }).then(function (nb) {
        if (my !== viewSeq) return;
        renderNearby(nb);
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

    // Credibility as the facts it is made of, with the recipe one click away.
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
      '<details class="howto"><summary>How this is computed</summary><p>' + esc(res.credibility_note || CRED_NOTE) + '</p>' +
      (a.credibility != null && isFinite(Number(a.credibility))
        ? '<p>This profile: <b>' + esc(Math.round(Number(a.credibility))) + '</b> of 100.</p>' : '') +
      '<p class="mini muted">Counts come from the OpenAlex snapshot PubVerse indexes, so recent papers may be missing.</p>' +
      '</details></section>';

    // Their space: subfields first (broad), then topics (narrow), deduplicated by name.
    var chips = [], have = Object.create(null);
    (a.top_subfields || a.subfields || []).forEach(function (s) { var n = nameOf(s); if (n && !have[n.toLowerCase()]) { have[n.toLowerCase()] = 1; chips.push(subjectChip(n, 'subfield')); } });
    (a.top_topics || a.topics || []).forEach(function (s) { var n = nameOf(s); if (n && !have[n.toLowerCase()]) { have[n.toLowerCase()] = 1; chips.push(subjectChip(n, '')); } });
    if (chips.length) {
      html += '<section class="av-sec" aria-labelledby="avSpace"><h3 id="avSpace">Their space</h3>' +
        '<p class="mini muted">Open one of their subjects in the Compass to see what is new in it.</p>' +
        '<div class="chips">' + chips.slice(0, 12).join('') + '</div></section>';
    }

    html += '<section class="av-sec" aria-labelledby="avPapersH">' +
      '<div class="row between rwrap av-papers-head"><h3 id="avPapersH">Papers</h3>' +
      '<div class="seg" role="group" aria-label="Sort papers">' +
        '<button type="button" data-sort="" aria-pressed="true">Most cited</button>' +
        '<button type="button" data-sort="year" aria-pressed="false">Newest</button></div></div>' +
      '<div id="avPapers">' + papersHtml(res.papers || [], res.papers_total) + '</div></section>';

    html += '<section class="av-sec" aria-labelledby="avNearbyH"><h3 id="avNearbyH">Nearby</h3>' +
      '<div id="avNearby">' + spinnerHtml('Finding nearby authors') + '</div></section>';

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

  function personList(list, kind) {
    return '<ul class="person-list">' + list.map(function (p) {
      var id = authorIdOf(p), name = nameOf(p) || 'Unnamed author';
      var meta = [];
      var place = placeLine(p.institution, p.country);
      if (place) meta.push(place);
      if (kind === 'co' && p.shared_works != null) meta.push(num(p.shared_works) + ' shared ' + plural(p.shared_works, 'paper'));
      if (kind === 'space' && p.h_index != null) meta.push('h-index ' + num(p.h_index));
      return '<li>' + (id ? '<a href="#author/' + id + '">' + esc(name) + '</a>' : esc(name)) +
        (meta.length ? '<span class="mini muted">' + meta.join(' &middot; ') + '</span>' : '') + '</li>';
    }).join('') + '</ul>';
  }
  function renderNearby(nb) {
    var box = $('#avNearby');
    if (!box) return;
    if (!nb || nb.ok === false) {
      box.innerHTML = '<p class="mini muted">' + esc(errorText(nb)) + '</p>';
      return;
    }
    var co = (nb.coauthors || []).filter(Boolean), sp = (nb.same_space || []).filter(Boolean);
    if (!co.length && !sp.length) { box.innerHTML = '<p class="muted">No nearby authors found.</p>'; return; }
    box.innerHTML = '<div class="nearby">' +
      (co.length ? '<div><h4>Co-authors</h4>' + personList(co.slice(0, 12), 'co') + '</div>' : '') +
      (sp.length ? '<div><h4>Working in the same space</h4>' + personList(sp.slice(0, 12), 'space') + '</div>' : '') +
      '</div>';
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
      '</dl><p><button type="button" class="btn" id="tvCompass" data-subject="' + esc(name) + '">Open in Compass</button></p></section>' +
      '<section class="av-sec" aria-labelledby="tvPeople"><h3 id="tvPeople">People in this area</h3>';
    if (!people.length) {
      html += '<p class="muted">No credible authors are listed for this area yet.</p>';
    } else {
      html += '<ul class="person-list">' + people.map(function (p) {
        var aid = authorIdOf(p), n = nameOf(p) || 'Unnamed author';
        var meta = [];
        var place = placeLine(p.institution, p.country);
        if (place) meta.push(place);
        if (p.h_index != null) meta.push('h-index ' + num(p.h_index));
        return '<li>' + (aid ? '<a href="#author/' + aid + '">' + esc(n) + '</a>' : esc(n)) +
          (normOrcid(p.orcid) ? ' ' + orcidBadge(p.orcid, false) : '') +
          (meta.length ? '<span class="mini muted">' + meta.join(' &middot; ') + '</span>' : '') + '</li>';
      }).join('') + '</ul>' +
      '<p class="mini muted">Ordered by credibility, from the public publication record. ' +
      '<a href="docs.html#search">How authors are ranked</a></p>';
    }
    html += '</section></article>';
    viewEl.innerHTML = html;
    focusHeading();
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
      return '<span class="ta-main">See all results for “' + esc(x.q) + '”</span>';
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
        status('No authors, funders or subjects match “' + q + '”.', 'empty');
        announce('No matches.');
        return;
      }
      group('Authors', A.slice(0, 5), 'author');
      group('Funders', F.slice(0, 3), 'funder');
      group('Subjects', T.slice(0, 3), 'topic');
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
      else go('#search?q=' + encodeURIComponent(x.q));
    }
    function submitAll() {
      var q = input.value.trim().slice(0, Q_MAX);
      if (!q) return;
      close();
      closeOverlay(false);
      input.blur();
      go('#search?q=' + encodeURIComponent(q));
    }
    function fetchQ(q) {
      var my = ++seq;
      status('Searching…', 'loading');
      api.search(q, { limit: 5, types: 'authors,funders,topics', timeout: 8000 }).then(function (res) {
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
