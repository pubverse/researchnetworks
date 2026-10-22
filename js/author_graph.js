/* PubVerse author network: the graph on top of an author's page.

   The author sits in the middle. Their topics are subtheme nodes around them. Co-authors and nearby
   researchers ("deepthinkers") are person nodes with an edge to the author when they wrote together
   and an edge to EVERY subtheme they share with the author. Choosing a person brings out that
   person's own subthemes and neighbors, keeping what is already drawn (click-out expansion); choosing
   a subtheme lights up everyone on it and offers its topic page and the Compass.

   The text list under the graph (js/search.js) is the keyboard and screen-reader equivalent of the
   whole thing; the graph itself also has one tab stop with arrow-key movement between nodes.

   Display only: every field drawn is already returned by the author routes. No contact or social
   fields are read, and nothing here is a model input.

   d3 (the same pinned build the research-space maps embed) is loaded on first use from
   js/vendor/d3.v7.9.0.min.js, so pages that never open an author do not pay for it.

   Usage:  PV.authorGraph.mount(hostElement, {
             center:   { id, name, institution, h_index, credibility },
             topics:   [{ id, name, n }],            // the author's top topics (n = papers)
             fetchPerson(id)  -> Promise<{ topics, coauthors, same_space } | null>,
             fetchTopics(id)  -> Promise<[{id, name}] | null>,   // fallback when shared topics are absent
             openAuthor(id), openCompass(name), topicHref({id, name}) -> '#topic/...' | '',
             maxNodes: 120 })
           -> { addNearby(nearby), destroy(), state() }                                            */
(function () {
  'use strict';
  window.PV = window.PV || {};

  var D3_SRC = 'js/vendor/d3.v7.9.0.min.js?v=1';
  var d3Promise = null;
  function loadD3() {
    if (window.d3 && window.d3.forceSimulation) return Promise.resolve(window.d3);
    if (d3Promise) return d3Promise;
    d3Promise = new Promise(function (resolve, reject) {
      var s = document.createElement('script');
      s.src = D3_SRC;
      s.onload = function () { window.d3 && window.d3.forceSimulation ? resolve(window.d3) : reject(new Error('d3 missing')); };
      s.onerror = function () { d3Promise = null; reject(new Error('d3 failed to load')); };
      document.head.appendChild(s);
    });
    return d3Promise;
  }

  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  function reduced() {
    try { return !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches); } catch (e) { return false; }
  }
  function plural(n, one, many) { return Number(n) === 1 ? one : (many || one + 's'); }
  var AID = /^A\d{1,12}$/;
  var SVGNS = 'http://www.w3.org/2000/svg';

  // A small deterministic hash so a node always starts at the same angle (same page, same picture).
  function hash(str) {
    var h = 2166136261;
    for (var i = 0; i < str.length; i++) { h ^= str.charCodeAt(i); h = Math.imul(h, 16777619); }
    return (h >>> 0) / 4294967296;
  }
  function clip(s, n) { s = String(s || ''); return s.length > n ? s.slice(0, n - 1).replace(/\s+$/, '') + '…' : s; }
  function wrap2(s, n) {
    var words = String(s || '').split(/\s+/), lines = [''];
    words.forEach(function (w) {
      var cur = lines[lines.length - 1];
      if (cur && (cur + ' ' + w).length > n && lines.length < 2) lines.push(w);
      else lines[lines.length - 1] = cur ? cur + ' ' + w : w;
    });
    if (lines.length > 1 && lines[1].length > n) lines[1] = clip(lines[1], n);
    if (lines[0].length > n + 4) lines[0] = clip(lines[0], n + 4);
    return lines;
  }

  /* ---------------------------------------------------------------- normalizers */

  function topicKey(t) {
    if (t.id != null && isFinite(Number(t.id))) return 't' + Number(t.id);
    return 'n:' + String(t.name || '').toLowerCase();
  }
  function personOf(x) {
    if (!x || !AID.test(String(x.id || ''))) return null;
    if (x.suspect_merge) return null;
    var cred = Number(x.credibility);
    return {
      id: x.id,
      name: String(x.name || 'Unnamed author'),
      institution: x.institution || '',
      country: x.country || '',
      h_index: x.h_index != null ? Number(x.h_index) : null,
      credibility: isFinite(cred) ? cred : null,
      shared_works: x.shared_works != null ? Number(x.shared_works) : null,
      why: x.why ? String(x.why) : ''
    };
  }
  // Shared topics, in order of preference: the explicit ids from the API (shared_topic_ids, with
  // shared_topic names), the {id, name} list (shared_topics), the topic names a "why" sentence spells
  // out after a colon. null means "not known"; [] means "known to share none".
  function sharedOf(x) {
    if (Array.isArray(x.shared_topics)) {
      return x.shared_topics.filter(function (t) { return t && (t.id != null || t.name); })
        .map(function (t) { return { id: t.id != null ? Number(t.id) : null, name: String(t.name || '') }; });
    }
    if (Array.isArray(x.shared_topic_ids)) {
      var names = Array.isArray(x.shared_topic) ? x.shared_topic : [];
      return x.shared_topic_ids.map(function (id, i) { return { id: Number(id), name: String(names[i] || '') }; });
    }
    if (typeof x.why === 'string' && x.why.indexOf(':') > 0) {
      var parts = x.why.split(':').slice(1).join(':').split(/;|,\s(?=[A-Z])/).map(function (s) { return s.trim(); }).filter(Boolean);
      if (parts.length) return parts.map(function (n) { return { id: null, name: n }; });
    }
    return null;
  }

  /* ---------------------------------------------------------------- the graph */

  function mount(host, opts) {
    var ctl = {
      addNearby: function () {}, destroy: function () { dead = true; }, state: function () { return {}; }
    };
    var dead = false;
    var pendingNearby = null;
    var inner = null;
    host.innerHTML = '<p class="mini muted ag-loading" role="status">Drawing the network…</p>';
    loadD3().then(function (d3) {
      if (dead) return;
      inner = build(d3, host, opts, function () { return dead; });
      ctl.addNearby = inner.addNearby;
      ctl.state = inner.state;
      ctl.destroy = function () { dead = true; inner.destroy(); };
      if (pendingNearby) inner.addNearby(pendingNearby);
    }, function () {
      if (dead) return;
      host.innerHTML = '<p class="mini muted">The network picture could not be loaded. The lists below show the same people.</p>';
    });
    ctl.addNearby = function (nb) { pendingNearby = nb; };
    return ctl;
  }

  function build(d3, host, opts, isDead) {
    var maxNodes = opts.maxNodes || 120;
    var RM = reduced();
    var compact = host.clientWidth && host.clientWidth < 560;

    var nodes = [], links = [];
    var byId = Object.create(null), linkSet = Object.create(null);
    var branchSeq = 0;
    var selected = null, focusId = null, userZoomed = false;
    var nameIndex = Object.create(null);   // topic name -> node id, to unify name-only topics
    var hidden = { suspects: 0 };
    var width = 600, height = 420;
    // Rings (layout units): the author at the middle, subthemes on the first ring, people outside it.
    var RING = { topic: 120, dt: 190, co: 205 };
    var sim = null;

    /* ---- DOM scaffold -------------------------------------------------------------- */
    host.innerHTML =
      '<div class="ag" data-ag>' +
        '<div class="ag-bar">' +
          '<p class="ag-hint mini muted" id="agHint">Click a person to bring out the people and subthemes around them. Click a subtheme to see who shares it.</p>' +
          '<div class="ag-tools" role="group" aria-label="Zoom the network">' +
            '<button type="button" class="ag-btn" data-act="in" aria-label="Zoom in">+</button>' +
            '<button type="button" class="ag-btn" data-act="out" aria-label="Zoom out">−</button>' +
            '<button type="button" class="ag-btn ag-fit" data-act="fit" aria-label="Fit the whole network">Fit</button>' +
          '</div>' +
        '</div>' +
        '<div class="ag-stage" tabindex="0" role="application" aria-roledescription="network graph" ' +
          'aria-label="Network of this author, their subthemes, co-authors and nearby researchers. Arrow keys move between nodes, Enter opens one, Escape clears." ' +
          'aria-describedby="agHint">' +
          '<svg class="ag-svg" focusable="false"></svg>' +
        '</div>' +
        '<ul class="ag-legend" aria-label="Legend">' +
          '<li><svg width="18" height="18" viewBox="-9 -9 18 18" aria-hidden="true"><circle r="7" class="lg-center"/></svg> This author</li>' +
          '<li><svg width="18" height="18" viewBox="-9 -9 18 18" aria-hidden="true"><path d="M0-8L8 0 0 8-8 0Z" class="lg-topic"/></svg> Subthemes</li>' +
          '<li><svg width="18" height="18" viewBox="-9 -9 18 18" aria-hidden="true"><circle r="6" class="lg-co"/></svg> Co-authors</li>' +
          '<li><svg width="18" height="18" viewBox="-9 -9 18 18" aria-hidden="true"><circle r="6" class="lg-dt"/></svg> Deepthinkers (same space, not co-authors)</li>' +
        '</ul>' +
        '<div class="ag-panel" id="agPanel" role="status" aria-live="polite"></div>' +
      '</div>';
    var root = host.querySelector('[data-ag]');
    var stage = root.querySelector('.ag-stage');
    var svgEl = root.querySelector('.ag-svg');
    var panel = root.querySelector('#agPanel');
    var svg = d3.select(svgEl);
    var gView = svg.append('g').attr('class', 'ag-view');
    var gLinks = gView.append('g').attr('class', 'ag-links');
    var gNodes = gView.append('g').attr('class', 'ag-nodes');

    function measure() {
      width = Math.max(260, stage.clientWidth || host.clientWidth || 600);
      compact = width < 560;
      RING = compact ? { topic: 64, dt: 108, co: 118 } : { topic: 120, dt: 190, co: 205 };
      height = compact ? 360 : 520;
      stage.style.height = height + 'px';
      svg.attr('width', width).attr('height', height).attr('viewBox', '0 0 ' + width + ' ' + height);
    }
    measure();

    /* ---- model --------------------------------------------------------------------- */
    function addLink(a, b, kind, w) {
      if (a === b) return;
      var k = a < b ? a + '|' + b : b + '|' + a;
      if (linkSet[k]) { if (w && w > linkSet[k].w) linkSet[k].w = w; return linkSet[k]; }
      var l = { source: a, target: b, kind: kind, w: w || 1 };
      linkSet[k] = l; links.push(l);
      return l;
    }
    function newNode(id, kind, label, data, branch) {
      var n = byId[id];
      if (n) { n.refs[branch] = 1; return n; }
      n = { id: id, kind: kind, label: label, data: data || {}, refs: {}, expanded: false, loading: false };
      n.refs[branch] = 1;
      n.r = kind === 'center' ? 17 : 9;
      if (branch === 0 && kind !== 'center') n.ring = kind;
      byId[id] = n; nodes.push(n);
      return n;
    }
    function seed(n, ax, ay, radius) {
      var ang = hash(n.id) * Math.PI * 2;
      n.x = ax + Math.cos(ang) * radius; n.y = ay + Math.sin(ang) * radius;
    }
    function topicNode(t, branch, nearNode, share) {
      var key = topicKey(t);
      var id = key;
      if (!byId[id] && t.name && nameIndex[t.name.toLowerCase()]) id = nameIndex[t.name.toLowerCase()];
      var fresh = !byId[id];
      var n = newNode(id, 'topic', t.name || ('Topic ' + t.id), { tid: t.id != null ? Number(t.id) : null, name: t.name || '', share: share || 0, n: t.n || 0 }, branch);
      if (t.name && !nameIndex[t.name.toLowerCase()]) nameIndex[t.name.toLowerCase()] = id;
      if (share && share > (n.data.share || 0)) n.data.share = share;
      n.r = 8 + 9 * Math.sqrt(Math.min(1, n.data.share || 0.15));
      if (fresh && nearNode) seed(n, nearNode.x || 0, nearNode.y || 0, 90);
      return n;
    }
    function personNode(p, kind, branch, nearNode) {
      var id = p.id, ex = byId[id];
      if (ex) {
        ex.refs[branch] = 1;
        if (kind === 'co' && ex.kind === 'dt') { ex.kind = 'co'; }
        return ex;
      }
      var n = newNode(id, kind, p.name, p, branch);
      var c = p.credibility != null ? Math.max(0, Math.min(100, p.credibility)) / 100 : 0.3;
      n.r = 6.5 + 5 * c;
      if (nearNode) seed(n, nearNode.x || 0, nearNode.y || 0, 110);
      return n;
    }
    function linkShared(person, shared, branch) {
      (shared || []).forEach(function (t) {
        if (t.id == null && !t.name) return;
        var tn = topicNode(t, branch, person, 0);
        addLink(person.id, tn.id, 'topic', 1);
      });
    }

    // The author's own subthemes.
    var center = newNode(opts.center.id, 'center', opts.center.name, personOf(opts.center) || { id: opts.center.id, name: opts.center.name,
      institution: opts.center.institution, h_index: opts.center.h_index, credibility: opts.center.credibility }, 0);
    center.x = 0; center.y = 0; center.fx = 0; center.fy = 0;
    var totalN = (opts.topics || []).reduce(function (s, t) { return s + (Number(t.n) || 0); }, 0);
    (opts.topics || []).slice(0, 6).forEach(function (t, i, arr) {
      var share = Number(t.share) || (totalN ? (Number(t.n) || 0) / totalN : 1 / Math.max(1, arr.length));
      var tn = topicNode({ id: t.id, name: t.name, n: t.n }, 0, null, share);
      var ang = (i / arr.length) * Math.PI * 2 - Math.PI / 2;
      tn.x = Math.cos(ang) * RING.topic; tn.y = Math.sin(ang) * RING.topic;
      tn.data.centerTopic = true;
      addLink(center.id, tn.id, 'topic', 1.5);
    });

    function cap(newCount) {
      // Oldest unselected branch collapses first: its nodes go unless a newer branch (or the center)
      // also references them, or they are the selected node.
      var guard = 0;
      while (nodes.length + newCount > maxNodes && guard++ < 50) {
        var oldest = null;
        nodes.forEach(function (n) {
          Object.keys(n.refs).forEach(function (b) { b = +b; if (b > 0 && (oldest === null || b < oldest)) oldest = b; });
        });
        if (oldest === null) break;
        var removed = 0;
        nodes.forEach(function (n) { if (n.refs[oldest]) { delete n.refs[oldest]; } });
        nodes = nodes.filter(function (n) {
          var keep = n.kind === 'center' || Object.keys(n.refs).length > 0 || (selected && selected.id === n.id);
          if (!keep) { delete byId[n.id]; removed++; }
          return keep;
        });
        // a collapsed parent no longer counts as expanded
        nodes.forEach(function (n) { if (n.expanded && !hasAnyChild(n)) n.expanded = false; });
        links = links.filter(function (l) {
          var a = typeof l.source === 'object' ? l.source.id : l.source, b = typeof l.target === 'object' ? l.target.id : l.target;
          var ok = byId[a] && byId[b];
          if (!ok) { delete linkSet[a < b ? a + '|' + b : b + '|' + a]; }
          return ok;
        });
        if (!removed) { /* everything left is protected; stop rather than loop */ }
      }
    }
    function hasAnyChild(n) {
      return links.some(function (l) {
        var a = typeof l.source === 'object' ? l.source.id : l.source, b = typeof l.target === 'object' ? l.target.id : l.target;
        return (a === n.id || b === n.id);
      });
    }

    function addNearby(nb, anchor, limits, branch) {
      if (isDead()) return 0;
      anchor = anchor || center;
      limits = limits || { co: 8, dt: 10 };
      if (!nb || nb.ok === false) { say('The nearby people could not be loaded, so only the subthemes are drawn.'); return 0; }
      var co = (nb.coauthors || []).map(function (x) { return { raw: x, p: personOf(x) }; });
      var dt = (nb.same_space || []).map(function (x) { return { raw: x, p: personOf(x) }; });
      hidden.suspects += co.filter(function (x) { return !x.p && x.raw && x.raw.suspect_merge; }).length +
                         dt.filter(function (x) { return !x.p && x.raw && x.raw.suspect_merge; }).length;
      co = co.filter(function (x) { return x.p; }).slice(0, limits.co);
      dt = dt.filter(function (x) { return x.p; }).slice(0, limits.dt);
      var coIds = Object.create(null); co.forEach(function (x) { coIds[x.p.id] = 1; });
      dt = dt.filter(function (x) { return !coIds[x.p.id] && x.p.id !== anchor.id; });
      var before = nodes.length;
      var br = branch != null ? branch : (anchor === center ? 0 : ++branchSeq);
      cap(co.length + dt.length + 6);
      var need = [];
      co.forEach(function (x) {
        var n = personNode(x.p, 'co', br, anchor);
        addLink(anchor.id, n.id, 'co', Math.max(1, x.p.shared_works || 1));
        var sh = sharedOf(x.raw);
        if (sh) linkShared(n, sh, br); else if (anchor === center) need.push(n);
      });
      dt.forEach(function (x) {
        var n = personNode(x.p, 'dt', br, anchor);
        var sh = sharedOf(x.raw);
        if (sh && sh.length) linkShared(n, sh, br);
        else addLink(anchor.id, n.id, 'near', 0.5);
      });
      relayout();
      if (need.length && opts.fetchTopics) fillSharedFallback(need.slice(0, 6), br);
      return nodes.length - before;
    }

    // An API without shared_topic_ids for co-authors: read each co-author's own topics (one at a
    // time, six at most) and keep the ones the author also has.
    function fillSharedFallback(list, br) {
      var i = 0;
      (function next() {
        if (isDead() || i >= list.length) return;
        var n = list[i++];
        Promise.resolve(opts.fetchTopics(n.id)).then(function (ts) {
          if (isDead() || !byId[n.id]) return;
          var mine = Object.create(null);
          nodes.forEach(function (t) { if (t.kind === 'topic' && t.data.centerTopic) mine[t.id] = t; });
          var any = false;
          (ts || []).forEach(function (t) {
            var k = topicKey(t);
            if (mine[k]) { addLink(n.id, k, 'topic', 1); any = true; }
          });
          if (any) relayout();
        }, function () {}).then(next);
      })();
    }

    /* ---- layout --------------------------------------------------------------------- */
    function linkIds(l) {
      return [typeof l.source === 'object' ? l.source.id : l.source, typeof l.target === 'object' ? l.target.id : l.target];
    }
    function setup() {
      if (sim) sim.stop();
      sim = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(function (d) { return d.id; })
          .distance(function (l) { return l.kind === 'co' ? RING.co : l.kind === 'near' ? 110 : 120; })
          .strength(function (l) { return l.kind === 'co' ? 0.08 : l.kind === 'near' ? 0.05 : 0.12; }))
        .force('charge', d3.forceManyBody().strength(function (d) { return d.kind === 'topic' ? -240 : -70; }).distanceMax(300))
        .force('collide', d3.forceCollide(function (d) { return d.r + (d.kind === 'topic' ? (compact ? 30 : 40) : (compact ? 12 : 16)); }).strength(0.95))
        .force('ring', d3.forceRadial(function (d) { return d.ring ? RING[d.ring] : 0; }, 0, 0).strength(function (d) { return d.ring ? (d.kind === 'topic' ? 0.9 : 0.7) : 0; }))
        .alphaDecay(0.045)
        .on('tick', paintPositions)
        .on('end', function () { if (isDead()) return; placeLabels(); if (!userZoomed) fit(true); });
      sim.stop();
    }
    function relayout() {
      var old = nodes.filter(function (n) { return n.x != null && n._placed && n.kind !== 'center'; });
      setup();
      // Nodes already drawn stay put while the new ones settle around them, then they are released
      // (the live layout then relaxes gently; the static layout simply stops). No animation when
      // reduced motion is on.
      old.forEach(function (n) { n.fx = n.x; n.fy = n.y; });
      sim.alpha(1);
      var ticks = RM ? 280 : 120;
      for (var i = 0; i < ticks; i++) sim.tick();
      old.forEach(function (n) { n.fx = null; n.fy = null; });
      if (RM) sim.stop(); else sim.alpha(old.length ? 0.25 : 0.15).restart();
      nodes.forEach(function (n) { n._placed = true; });
      draw();
      placeLabels();
      if (!userZoomed) fit(RM || !old.length);
    }

    /* ---- drawing --------------------------------------------------------------------- */
    function labelSet() {
      // Everybody gets a tooltip and a list entry; only the center, topics, the selection, its
      // neighbors and the most credible people carry a printed label, so a phone is not a wall of text.
      var people = nodes.filter(function (n) { return n.kind !== 'topic' && n.kind !== 'center'; })
        .sort(function (a, b) { return ((b.data.credibility || 0) - (a.data.credibility || 0)); });
      var set = Object.create(null);
      var nco = compact ? 3 : 6, ndt = compact ? 1 : 4;
      people.filter(function (n) { return n.kind === 'co'; }).slice(0, nco).forEach(function (n) { set[n.id] = 1; });
      people.filter(function (n) { return n.kind === 'dt'; }).slice(0, ndt).forEach(function (n) { set[n.id] = 1; });
      return set;
    }
    function nodeAria(n) {
      if (n.kind === 'topic') return 'Subtheme: ' + n.label;
      if (n.kind === 'center') return 'This author: ' + n.label;
      var bits = [(n.kind === 'co' ? 'Co-author: ' : 'Deepthinker: ') + n.label];
      if (n.data.institution) bits.push(n.data.institution);
      return bits.join(', ');
    }
    function shapePath(n) {
      var r = n.r + 2;
      return 'M0,' + (-r) + 'L' + r + ',0 0,' + r + ' ' + (-r) + ',0Z';
    }
    function draw() {
      var labelled = labelSet();
      var lsel = gLinks.selectAll('line').data(links, function (l) { var i = linkIds(l); return i[0] < i[1] ? i[0] + '|' + i[1] : i[1] + '|' + i[0]; });
      lsel.exit().remove();
      lsel.enter().append('line').merge(lsel)
        .attr('class', function (l) { return 'ag-link ag-link-' + l.kind; })
        .attr('data-a', function (l) { return linkIds(l)[0]; }).attr('data-b', function (l) { return linkIds(l)[1]; })
        .attr('stroke-width', function (l) { return l.kind === 'co' ? Math.min(3.2, 1.2 + Math.log2(1 + l.w) * 0.3) : l.kind === 'near' ? 1 : 1.1; });

      var nsel = gNodes.selectAll('g.ag-node').data(nodes, function (n) { return n.id; });
      nsel.exit().remove();
      var ent = nsel.enter().append('g')
        .attr('role', 'button')
        .attr('data-node-id', function (n) { return n.id; })
        .attr('id', function (n) { return 'agn-' + String(n.id).replace(/[^A-Za-z0-9_-]/g, '_'); });
      ent.append('circle').attr('class', 'ag-hit');
      ent.append('path').attr('class', 'ag-shape-topic');
      ent.append('circle').attr('class', 'ag-shape-person');
      ent.append('title');
      ent.append('text').attr('class', 'ag-label');
      var all = ent.merge(nsel);
      all.attr('class', function (n) { return 'ag-node ag-' + n.kind + (n.expanded ? ' is-expanded' : ''); })
        .attr('data-kind', function (n) { return n.kind; })
        .attr('aria-label', nodeAria);
      all.select('title').text(function (n) {
        return n.kind === 'topic' ? n.label : n.label + (n.data.institution ? ' · ' + n.data.institution : '');
      });
      all.select('.ag-hit').attr('r', function (n) { return Math.max(n.r + 7, 15); });
      all.select('.ag-shape-topic').attr('d', shapePath).style('display', function (n) { return n.kind === 'topic' ? null : 'none'; });
      all.select('.ag-shape-person').attr('r', function (n) { return n.r; }).style('display', function (n) { return n.kind === 'topic' ? 'none' : null; });
      all.each(function (n) {
        var t = d3.select(this).select('text.ag-label');
        t.selectAll('tspan').remove();
        n._showLabel = n.kind === 'topic' || n.kind === 'center' || !!labelled[n.id];
        if (n.kind === 'topic') {
          var lines = wrap2(n.label, compact ? 16 : 22);
          lines.forEach(function (ln, i) { t.append('tspan').attr('x', 0).attr('dy', i ? '1.1em' : 0).text(ln); });
          t.attr('text-anchor', 'middle').attr('y', n.r + 15);
        } else {
          t.append('tspan').attr('x', 0).text(clip(n.label, compact ? 14 : 22));
          if (n.kind === 'center') { t.attr('text-anchor', 'middle').attr('x', 0).attr('y', -(n.r + 9)); t.select('tspan').attr('x', 0); }
          else { t.attr('text-anchor', 'start').attr('x', n.r + 5).attr('y', 4).attr('dy', 0); t.select('tspan').attr('x', n.r + 5); }
        }
        d3.select(this).classed('has-label', n._showLabel && !n._hideLabel);
      });
      all.on('click', function (ev, n) { ev.stopPropagation(); activate(n); });
      if (!RM) {
        all.call(d3.drag().clickDistance(5).filter(function (ev) { return ev.pointerType !== 'touch' && !ev.button; })
          .on('start', function (ev, n) { if (n.kind === 'center') return; if (!ev.active) sim.alphaTarget(0.25).restart(); userZoomed = true; n.fx = n.x; n.fy = n.y; })
          .on('drag', function (ev, n) { if (n.kind === 'center') return; n.fx = ev.x; n.fy = ev.y; })
          .on('end', function (ev, n) { if (n.kind === 'center') return; if (!ev.active) sim.alphaTarget(0); n.fx = null; n.fy = null; }));
      }
      paintPositions();
      paintState();
    }
    // Printed person labels: try right, left, above, below of the node; take the first spot that
    // collides with no other label or node; hide the label when none is free (the tooltip, the list
    // below and the selection panel still name everyone).
    function placeLabels() {
      var fs = compact ? 11 : 11.5, cw = fs * 0.55, lh = fs + 2;
      var boxes = [], i;
      function hit(b, skip) {
        for (var j = 0; j < boxes.length; j++) {
          var c = boxes[j];
          if (b.x < c.x + c.w && b.x + b.w > c.x && b.y < c.y + c.h && b.y + b.h > c.y) return true;
        }
        for (var k = 0; k < nodes.length; k++) {
          var n = nodes[k]; if (n === skip || n.x == null) continue;
          var rr = n.r + 3;
          if (b.x < n.x + rr && b.x + b.w > n.x - rr && b.y < n.y + rr && b.y + b.h > n.y - rr) return true;
        }
        return false;
      }
      nodes.forEach(function (n) {
        if (n.kind === 'topic') {
          var lines = wrap2(n.label, compact ? 16 : 22), w = Math.max.apply(null, lines.map(function (l) { return l.length; })) * cw;
          boxes.push({ x: n.x - w / 2, y: n.y + n.r + 4, w: w, h: lines.length * lh });
        } else if (n.kind === 'center') {
          var w2 = Math.min(n.label.length, compact ? 14 : 22) * cw;
          boxes.push({ x: n.x - w2 / 2, y: n.y - n.r - 9 - lh, w: w2, h: lh });
        }
      });
      var people = nodes.filter(function (n) { return n.kind !== 'topic' && n.kind !== 'center' && n._showLabel; })
        .sort(function (a, b) { return (b.data.credibility || 0) - (a.data.credibility || 0); });
      people.forEach(function (n) {
        var w = Math.min(n.label.length, compact ? 16 : 22) * cw, r = n.r;
        var cands = [
          { x: n.x + r + 5, y: n.y - lh / 2, a: 'start', lx: r + 5, ly: 4 },
          { x: n.x - r - 5 - w, y: n.y - lh / 2, a: 'end', lx: -(r + 5), ly: 4 },
          { x: n.x - w / 2, y: n.y - r - 5 - lh, a: 'middle', lx: 0, ly: -(r + 8) },
          { x: n.x - w / 2, y: n.y + r + 5, a: 'middle', lx: 0, ly: r + 15 }
        ];
        n._hideLabel = true;
        for (i = 0; i < cands.length; i++) {
          var c = cands[i], b = { x: c.x, y: c.y, w: w, h: lh };
          if (!hit(b, n)) { boxes.push(b); n._lbl = c; n._hideLabel = false; break; }
        }
      });
      gNodes.selectAll('g.ag-node').each(function (n) {
        if (n.kind === 'topic' || n.kind === 'center' || !n._lbl) return;
        var t = d3.select(this).select('text.ag-label');
        t.attr('text-anchor', n._lbl.a).attr('x', n._lbl.lx).attr('y', n._lbl.ly);
        t.select('tspan').attr('x', n._lbl.lx);
      });
      paintState();
    }
    function paintPositions() {
      gLinks.selectAll('line')
        .attr('x1', function (l) { return l.source.x; }).attr('y1', function (l) { return l.source.y; })
        .attr('x2', function (l) { return l.target.x; }).attr('y2', function (l) { return l.target.y; });
      gNodes.selectAll('g.ag-node').attr('transform', function (n) { return 'translate(' + (n.x || 0) + ',' + (n.y || 0) + ')'; });
    }

    /* ---- selection & highlight --------------------------------------------------------- */
    function neighborsOf(id) {
      var s = Object.create(null); s[id] = 1;
      links.forEach(function (l) { var i = linkIds(l); if (i[0] === id) s[i[1]] = 1; else if (i[1] === id) s[i[0]] = 1; });
      return s;
    }
    function paintState() {
      var set = selected ? neighborsOf(selected.id) : null;
      gNodes.selectAll('g.ag-node')
        .classed('is-selected', function (n) { return !!selected && n.id === selected.id; })
        .classed('is-dim', function (n) { return !!set && !set[n.id]; })
        .classed('is-lit', function (n) { return !!set && !!set[n.id] && n.id !== selected.id; })
        .classed('is-focus', function (n) { return n.id === focusId; })
        .classed('has-label', function (n) { return (n._showLabel && !n._hideLabel) || (!!set && !!set[n.id]) || (!!selected && selected.id === n.id) || n.id === focusId; });
      gLinks.selectAll('line')
        .classed('is-dim', function (l) { if (!set) return false; var i = linkIds(l); return !(set[i[0]] && set[i[1]] && (i[0] === selected.id || i[1] === selected.id)); })
        .classed('is-lit', function (l) { if (!set) return false; var i = linkIds(l); return i[0] === selected.id || i[1] === selected.id; });
      var f = focusId && byId[focusId];
      if (f) stage.setAttribute('aria-activedescendant', 'agn-' + String(f.id).replace(/[^A-Za-z0-9_-]/g, '_'));
      else stage.removeAttribute('aria-activedescendant');
    }
    function say(t) {
      var p = panel.querySelector('.ag-note');
      if (!p) { p = document.createElement('p'); p.className = 'ag-note mini muted'; panel.appendChild(p); }
      p.textContent = t || '';
    }

    function peopleOn(topicNode) {
      var out = [];
      var ids = neighborsOf(topicNode.id);
      Object.keys(ids).forEach(function (id) { var n = byId[id]; if (n && n.kind !== 'topic') out.push(n); });
      out.sort(function (a, b) { return (a.kind === 'center' ? -1 : 0) - (b.kind === 'center' ? -1 : 0) || (b.data.credibility || 0) - (a.data.credibility || 0); });
      return out;
    }
    function panelHtml() {
      var n = selected;
      if (!n) {
        return '<p class="ag-empty mini muted">Nothing selected. ' + (hidden.suspects ? hidden.suspects + ' ' + plural(hidden.suspects, 'profile') + ' that may merge several people ' + plural(hidden.suspects, 'is', 'are') + ' left out. ' : '') +
          'Showing ' + nodes.length + ' ' + plural(nodes.length, 'node') + '.</p>';
      }
      if (n.kind === 'topic') {
        var ppl = peopleOn(n);
        var href = opts.topicHref ? opts.topicHref({ id: n.data.tid, name: n.label }) : '';
        return '<h4 class="ag-ph">Subtheme: ' + esc(n.label) + '</h4>' +
          '<p class="mini">' + ppl.length + ' ' + plural(ppl.length, 'person is', 'people are') + ' on this subtheme in the picture: ' +
          esc(ppl.slice(0, 8).map(function (p) { return p.label; }).join(', ')) + (ppl.length > 8 ? ' and ' + (ppl.length - 8) + ' more' : '') + '.</p>' +
          '<div class="btn-row">' +
          (href ? '<a class="btn ghost sm" data-ag-act="topic" href="' + esc(href) + '">Topic page</a>' : '') +
          '<button type="button" class="btn ghost sm" data-ag-act="compass">Open in Compass</button>' +
          '<button type="button" class="btn ghost sm" data-ag-act="clear">Clear</button></div>';
      }
      var d = n.data, meta = [];
      if (d.institution) meta.push(d.institution);
      if (d.h_index != null) meta.push('h-index ' + d.h_index);
      var role = n.kind === 'center' ? 'This author' : n.kind === 'co'
        ? 'Co-author' + (d.shared_works ? ', ' + d.shared_works + ' shared ' + plural(d.shared_works, 'paper') : '')
        : (d.why ? 'Deepthinker: ' + d.why : 'Deepthinker in the same space');
      var tn = [];
      links.forEach(function (l) {
        var i = linkIds(l), o = i[0] === n.id ? i[1] : i[1] === n.id ? i[0] : null;
        if (o && byId[o] && byId[o].kind === 'topic') tn.push(byId[o].label);
      });
      return '<h4 class="ag-ph">' + esc(n.label) + '</h4>' +
        '<p class="mini muted">' + esc(role) + (meta.length ? ' · ' + esc(meta.join(' · ')) : '') + '</p>' +
        (tn.length ? '<p class="mini">Subthemes: ' + esc(tn.slice(0, 6).join('; ')) + (tn.length > 6 ? ' and ' + (tn.length - 6) + ' more' : '') + '.</p>' : '') +
        '<div class="btn-row">' +
        (n.kind !== 'center' ? '<button type="button" class="btn ghost sm" data-ag-act="author">Open author page</button>' : '') +
        (n.kind !== 'center' && !n.expanded && !n.loading ? '<button type="button" class="btn ghost sm" data-ag-act="expand">Bring out their neighbors</button>' : '') +
        '<button type="button" class="btn ghost sm" data-ag-act="clear">Clear</button></div>' +
        (n.loading ? '<p class="mini muted ag-note">Bringing out their neighbors…</p>' : '<p class="ag-note mini muted"></p>');
    }
    function renderPanel(keepNote) {
      var note = keepNote ? (panel.querySelector('.ag-note') || {}).textContent : '';
      panel.innerHTML = panelHtml();
      if (note) say(note);
    }

    function select(n) {
      selected = n || null;
      focusId = n ? n.id : focusId;
      paintState();
      renderPanel();
    }
    function activate(n) {
      select(n);
      if (n.kind !== 'topic' && n.kind !== 'center' && !n.expanded && !n.loading) expand(n);
    }
    function expand(n) {
      if (!opts.fetchPerson || n.expanded || n.loading) return;
      n.loading = true; renderPanel();
      Promise.resolve(opts.fetchPerson(n.id)).then(function (res) {
        n.loading = false;
        if (isDead() || !byId[n.id]) return;
        if (!res) { n.expanded = false; renderPanel(); say('Their neighbors could not be loaded. Click the person to try again.'); return; }
        n.expanded = true;
        var before = nodes.length;
        // their own subthemes first, so people who share them attach to the same nodes
        var tops = (res.topics || []).slice(0, 5), tot = tops.reduce(function (s, t) { return s + (Number(t.n) || 0); }, 0);
        var br = ++branchSeq;
        cap(5 + 5 + 6 + 1);
        if (!byId[n.id]) return;
        tops.forEach(function (t) {
          var share = Number(t.share) || (tot ? (Number(t.n) || 0) / tot : 0.2);
          var tn = topicNode({ id: t.id, name: t.name, n: t.n }, br, n, share);
          addLink(n.id, tn.id, 'topic', 1);
        });
        addNearby(res, n, { co: 5, dt: 5 }, br);
        // addNearby used its own branch numbers; keep the person's branch alive for the subthemes
        select(selected && selected.id === n.id ? n : selected);
        say('Brought out ' + (nodes.length - before) + ' ' + plural(nodes.length - before, 'node') + ' around ' + n.label + '.');
      }, function () { n.loading = false; renderPanel(); say('Their neighbors could not be loaded.'); });
    }

    /* ---- zoom & fit -------------------------------------------------------------------- */
    var zoom = d3.zoom().scaleExtent([0.25, 4])
      .filter(function (ev) {
        if (ev.type === 'wheel') return ev.ctrlKey || ev.metaKey;
        if (ev.type.indexOf('touch') === 0) return ev.touches && ev.touches.length > 1;
        return !ev.button && ev.type !== 'dblclick';
      })
      .on('zoom', function (ev) {
        gView.attr('transform', ev.transform);
        if (ev.sourceEvent) userZoomed = true;
      });
    svg.call(zoom);
    svg.on('click', function () { if (selected) select(null); });
    function bounds() {
      var cw = compact ? 6 : 6.4, x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      function add(x, y) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
      nodes.forEach(function (n) {
        if (n.x == null) return;
        var r = n.r + 6;
        add(n.x - r, n.y - r); add(n.x + r, n.y + r);
        if (n.kind === 'topic') {
          var lines = wrap2(n.label, compact ? 16 : 22), w = Math.max.apply(null, lines.map(function (l) { return l.length; })) * cw;
          add(n.x - w / 2, n.y); add(n.x + w / 2, n.y + n.r + 6 + lines.length * 14);
        } else if (n.kind === 'center') {
          add(n.x, n.y - n.r - 22);
        } else if (n._showLabel && !n._hideLabel && n._lbl) {
          var lw = Math.min(n.label.length, compact ? 16 : 22) * cw;
          if (n._lbl.a === 'start') add(n.x + n.r + 8 + lw, n.y);
          else if (n._lbl.a === 'end') add(n.x - n.r - 8 - lw, n.y);
          else { add(n.x - lw / 2, n.y); add(n.x + lw / 2, n.y); add(n.x, n.y + (n._lbl.ly < 0 ? -n.r - 20 : n.r + 20)); }
        }
      });
      return { x0: x0, y0: y0, x1: x1, y1: y1 };
    }
    function fit(instant) {
      if (!nodes.length) return;
      var b = bounds(), w = b.x1 - b.x0, h = b.y1 - b.y0;
      if (!isFinite(w) || w <= 0) return;
      var k = Math.min(1.5, Math.min((width - 16) / w, (height - 16) / h));
      var tx = width / 2 - k * (b.x0 + w / 2), ty = height / 2 - k * (b.y0 + h / 2);
      var t = d3.zoomIdentity.translate(tx, ty).scale(k);
      if (instant || RM) svg.call(zoom.transform, t); else svg.transition().duration(350).call(zoom.transform, t);
      userZoomed = false;
    }
    root.querySelector('.ag-tools').addEventListener('click', function (e) {
      var b = e.target.closest && e.target.closest('button[data-act]');
      if (!b) return;
      var a = b.getAttribute('data-act');
      if (a === 'fit') { fit(true); return; }
      userZoomed = true;
      svg.call(zoom.scaleBy, a === 'in' ? 1.35 : 1 / 1.35);
    });

    /* ---- panel buttons, keyboard --------------------------------------------------------- */
    panel.addEventListener('click', function (e) {
      var b = e.target.closest && e.target.closest('[data-ag-act]');
      if (!b || !selected) return;
      var a = b.getAttribute('data-ag-act');
      if (a === 'clear') select(null);
      else if (a === 'compass' && opts.openCompass) opts.openCompass(selected.label);
      else if (a === 'author' && opts.openAuthor) { e.preventDefault(); opts.openAuthor(selected.id); }
      else if (a === 'expand') expand(selected);
    });
    function order() {
      var rank = { center: 0, topic: 1, co: 2, dt: 3 };
      return nodes.slice().sort(function (a, b) { return rank[a.kind] - rank[b.kind]; });
    }
    function moveFocus(d) {
      var list = order(); if (!list.length) return;
      var i = list.findIndex(function (n) { return n.id === focusId; });
      i = i < 0 ? (d > 0 ? 0 : list.length - 1) : (i + d + list.length) % list.length;
      focusId = list[i].id;
      paintState();
      var n = list[i];
      var live = n.kind === 'topic' ? 'Subtheme ' + n.label : (n.kind === 'center' ? 'This author ' : n.kind === 'co' ? 'Co-author ' : 'Deepthinker ') + (n.kind === 'topic' ? '' : n.label);
      var ln = root.querySelector('.ag-focus-live');
      if (!ln) { ln = document.createElement('span'); ln.className = 'sr-only ag-focus-live'; ln.setAttribute('aria-live', 'polite'); root.appendChild(ln); }
      ln.textContent = live;
    }
    stage.addEventListener('keydown', function (e) {
      var k = e.key;
      if (k === 'ArrowRight' || k === 'ArrowDown') { e.preventDefault(); moveFocus(1); }
      else if (k === 'ArrowLeft' || k === 'ArrowUp') { e.preventDefault(); moveFocus(-1); }
      else if (k === 'Enter' || k === ' ') { var n = focusId && byId[focusId]; if (n) { e.preventDefault(); activate(n); } }
      else if (k === 'Escape') { if (selected) { e.preventDefault(); select(null); } }
      else if (k === '+' || k === '=') { userZoomed = true; svg.call(zoom.scaleBy, 1.35); }
      else if (k === '-' || k === '_') { userZoomed = true; svg.call(zoom.scaleBy, 1 / 1.35); }
      else if (k === '0') { fit(true); }
    });
    stage.addEventListener('focus', function () { if (!focusId) { focusId = center.id; paintState(); } });

    var ro = null;
    if (window.ResizeObserver) {
      var lastW = stage.clientWidth;
      ro = new ResizeObserver(function () {
        if (isDead()) return;
        var w = stage.clientWidth;
        if (Math.abs(w - lastW) < 2) return;
        lastW = w; measure(); if (!userZoomed) fit(true);
      });
      ro.observe(stage);
    }

    relayout();
    renderPanel();

    return {
      addNearby: function (nb) { addNearby(nb, center, { co: 8, dt: 10 }); renderPanel(true); },
      destroy: function () { if (sim) sim.stop(); if (ro) ro.disconnect(); },
      state: function () {
        return { nodes: nodes.length, links: links.length, selected: selected && selected.id,
                 kinds: nodes.reduce(function (a, n) { a[n.kind] = (a[n.kind] || 0) + 1; return a; }, {}), reduced: RM };
      }
    };
  }

  PV.authorGraph = { mount: mount, _sharedOf: sharedOf, _personOf: personOf };
})();
