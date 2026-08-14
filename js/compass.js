/* PubVerse Compass, as a module.
   
   Lifted verbatim out of compass.html's inline IIFE. That script owned a whole page: it did its
   own auth, its own veil, its own boot, and it was never assigned to window.PV, so no other page
   could reach a line of it. Making Compass a tab on the landing page meant the markup moved and
   the behaviour did not, which is how tab 1 shipped as a form whose buttons did nothing.
   
   What was dropped in the move: the auth half (isUser, boot, showLogin, onAuthed, failLogin, the
   sign-out handler and the veil wiring). index.html already does all of it, and two copies in one
   document would mean two api.me() GETs and two api.login() POSTs per click on one form.
   
   What is kept exactly: every function of the compass half, unchanged, so the needle list renders
   in the format it always has.
   
   init() is idempotent -- it is called from the tab-activation hook and guarded there too -- and
   binds by id against the pane, which carries the same ids the page did. */
(function () {
  'use strict';
  var PV = window.PV = window.PV || {};
  var api = PV.api, ui = PV.ui;
  function $(s) { return document.querySelector(s); }
  function on(node, ev, fn) { if (node) node.addEventListener(ev, fn); }

  // Was declared in the auth half; the only name the compass code borrowed across the cut.
  var started = false;

  // ---- run switcher: load + switch between the user's completed needle searches ----
  function loadPastRuns() {
    if (!api.compassRuns) return;
    api.compassRuns().then(function (rows) {
      var sel = $('#pastRuns'), wrap = $('#pastRunsWrap');
      if (!sel || !wrap) return;
      rows = (rows || []).filter(function (r) { return r && r.status === 'done'; });
      if (!rows.length) { wrap.hidden = true; return; }
      var opts = ['<option value="">Your past searches (' + rows.length + ')…</option>'];
      rows.forEach(function (r) {
        var d = r.ts ? new Date(r.ts * 1000).toLocaleDateString() : '';
        opts.push('<option value="' + esc(r.run_id) + '">' + esc(r.topic || 'search') + (d ? ' (' + esc(d) + ')' : '') + '</option>');
      });
      sel.innerHTML = opts.join('');
      wrap.hidden = false;
    });
  }
  on($('#pastRuns'), 'change', function () {
    var rid = this.value;
    if (!rid) return;
    api.compassPoll(rid).then(function (r) {
      if (r && r.dashboard) {
        renderDashboard(r.dashboard, { runId: rid, exportToken: r.export_token });
        announceMap(r, rid);
        var dash = $('#dash'); if (dash) dash.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });

  /* ---------- page wiring (runs once, after auth) ---------- */
  var checkedTopic = null;
  var polling = false;
  var validating = false;
  var lastAutoChecked = null;
  var suggestTimer = null;
  var coveredMap = Object.create(null);   // lowercased covered label -> true

  function initPage() {
    if (started) return;
    started = true;


    var sel = $('#months');
    PV.TIMEFRAMES.forEach(function (m) {
      var o = document.createElement('option');
      o.value = String(m);
      o.textContent = m + (m === 1 ? ' month' : ' months');
      if (m === PV.TIMEFRAME_DEFAULT) o.selected = true;
      sel.appendChild(o);
    });
    sel.value = String(PV.TIMEFRAME_DEFAULT);

    on($('#topic'), 'input', resetValidation);
    on($('#topic'), 'input', onTopicInput);
    on($('#topic'), 'change', onTopicCommit);
    on($('#checkBtn'), 'click', onCheck);
    on($('#findBtn'), 'click', onFind);
    on($('#requestBtn'), 'click', onRequestTopic);

    loadCovered();
    prefillTopicFromQuery();
    loadExample();
  }

  // Preselect the offered window closest to a field's default, so a covered
  // topic starts on a sensible look-back rather than a fixed constant.
  function selectTimeframe(months) {
    var sel = $('#months');
    if (!sel || months == null) return;
    var want = parseInt(months, 10);
    if (isNaN(want)) return;
    var best = null, bestDiff = Infinity;
    PV.TIMEFRAMES.forEach(function (m) {
      var d = Math.abs(m - want);
      if (d < bestDiff) { bestDiff = d; best = m; }
    });
    if (best != null) sel.value = String(best);
  }

  // A saved-topic chip links here with ?topic=...; prefill it for one-click reuse.
  function prefillTopicFromQuery() {
    try {
      var t = new URLSearchParams(window.location.search).get('topic');
      if (t) {
        var input = $('#topic');
        if (input) { input.value = t; input.dispatchEvent(new Event('input')); }
      }
    } catch (e) { /* query parsing is a convenience, never block the page */ }
  }

  function resetValidation() {
    checkedTopic = null;
    lastAutoChecked = null;
    $('#runControls').hidden = true;
    $('#validOk').hidden = true;
    hideRequest();
    ui.clearError('#topicErr');
  }

  /* ---------- covered-topic picker (searchable dropdown over the free-text box) ---------- */
  // Fill the topic datalist with the subjects PubVerse covers. The list is
  // append-only so options never vanish mid-type; the browser filters what it
  // shows by what the user has typed. Best-effort: never blocks the page.
  function loadCovered(q) {
    api.compassCovered(q).then(function (r) {
      addCovered(parseCovered(r));
    }).catch(function () { /* the picker is a convenience, never block on it */ });
  }
  function parseCovered(r) {
    if (!r || r.ok === false) return [];
    var arr = Array.isArray(r) ? r
      : (r.topics || r.covered || r.subjects || r.items || r.fields || []);
    if (!Array.isArray(arr)) return [];
    var out = [];
    arr.forEach(function (it) {
      var label = '';
      if (typeof it === 'string') label = it;
      else if (it && typeof it === 'object')
        label = it.label || it.topic || it.name || it.title || it.subject || it.slug || '';
      label = String(label == null ? '' : label).trim();
      if (label) out.push(label);
    });
    return out;
  }
  function addCovered(labels) {
    var dl = $('#coveredTopics');
    if (!dl || !labels.length) return;
    labels.forEach(function (label) {
      var key = label.toLowerCase();
      if (coveredMap[key]) return;
      coveredMap[key] = true;
      var o = document.createElement('option');
      o.value = label;
      dl.appendChild(o);
    });
  }
  function isCovered(topic) {
    return !!coveredMap[(topic || '').trim().toLowerCase()];
  }

  // As the user types, ask the server for matching covered subjects (it searches
  // by ?q=). Debounced, and merged in, so a large coverage list works without
  // shipping every subject up front.
  function onTopicInput() {
    var v = ($('#topic').value || '').trim();
    if (suggestTimer) clearTimeout(suggestTimer);
    if (v.length < 2) return;
    suggestTimer = setTimeout(function () { loadCovered(v); }, 250);
  }

  // Fired when the field is committed (an option chosen from the dropdown, or
  // focus leaving after an edit). If it matches a covered subject exactly, check
  // it for the user so the run controls appear without a second click.
  function onTopicCommit() {
    var v = ($('#topic').value || '').trim();
    var key = v.toLowerCase();
    if (v && isCovered(v) && lastAutoChecked !== key) {
      lastAutoChecked = key;
      onCheck();
    }
  }

  /* ---------- request an uncovered topic ---------- */
  function showRequest(topic) {
    var panel = $('#requestPanel');
    if (!panel) return;
    panel.setAttribute('data-topic', topic || '');
    $('#requestAsk').hidden = false;
    $('#requestDone').hidden = true;
    ui.clearError('#requestErr');
    var btn = $('#requestBtn');
    if (btn) { btn.disabled = false; btn.textContent = 'Request this topic'; }
    panel.hidden = false;
  }
  function hideRequest() {
    var panel = $('#requestPanel');
    if (panel) panel.hidden = true;
  }
  function looksLikeEmail(s) {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(s);
  }
  function onRequestTopic() {
    var panel = $('#requestPanel');
    var topic = (panel && panel.getAttribute('data-topic')) || ($('#topic').value || '').trim();
    if (!topic) return;
    var email = ($('#reqEmail').value || '').trim();
    ui.clearError('#requestErr');
    if (email && !looksLikeEmail(email)) {
      ui.showError('#requestErr', 'That email does not look right. Correct it, or leave it blank.');
      return;
    }
    var btn = $('#requestBtn');
    btn.disabled = true;
    var label = btn.textContent;
    btn.textContent = 'Sending...';
    api.compassRequestTopic(topic, email || undefined).then(function (r) {
      if (r && r.ok) {
        $('#requestAsk').hidden = true;
        var done = $('#requestDone');
        done.textContent = email
          ? 'Thanks. We noted your interest in "' + topic + '" and will email ' + email + ' when PubVerse covers it.'
          : 'Thanks. We noted your interest in "' + topic + '". Check back as coverage grows.';
        done.hidden = false;
      } else {
        btn.disabled = false;
        btn.textContent = label;
        ui.showError('#requestErr', (r && r.message) || 'Could not send the request just now. Please try again.');
      }
    });
  }

  /* ---------- check topic ---------- */
  function onCheck() {
    var topic = ($('#topic').value || '').trim();
    if (validating) return;
    // Already validated this exact topic as covered and the run controls are up.
    if (topic && checkedTopic === topic && !$('#runControls').hidden) return;
    ui.clearError('#topicErr');
    $('#validOk').hidden = true;
    hideRequest();
    if (!topic) { ui.showError('#topicErr', 'Enter a topic first.'); return; }
    if (topic.length > PV.TOPIC_CHAR_LIMIT) {
      ui.showError('#topicErr', 'That topic is too long. Keep it under ' + PV.TOPIC_CHAR_LIMIT + ' characters.');
      return;
    }
    var btn = $('#checkBtn');
    validating = true;
    btn.disabled = true;
    var label = btn.textContent;
    btn.textContent = 'Checking...';
    api.compassValidate(topic).then(function (r) {
      validating = false;
      btn.disabled = false;
      btn.textContent = label;
      if (r && r.ok) {
        checkedTopic = topic;
        $('#validOk').textContent = r.message || 'This topic is covered. Choose how far back to look and find the needles.';
        $('#validOk').hidden = false;
        $('#runControls').hidden = false;
        hideRequest();
        if (r.default_months != null) selectTimeframe(r.default_months);
      } else {
        ui.showError('#topicErr', (r && r.message) || 'That topic is not covered yet. Try a broader or nearby subject.');
        showRequest(topic);
      }
    });
  }

  /* ---------- run + poll ---------- */
  function onFind() {
    if (polling) return;
    var topic = checkedTopic || ($('#topic').value || '').trim();
    if (!topic) { ui.showError('#topicErr', 'Enter and check a topic first.'); return; }
    var months = parseInt($('#months').value, 10) || PV.TIMEFRAME_DEFAULT;
    var email = ($('#email').value || '').trim();

    ui.clearError('#topicErr');
    $('#findBtn').disabled = true;
    ui.showCompass('#spinner', 'Reading the recent literature. This can take a few minutes.');

    api.compassRun(topic, months, email || undefined).then(function (r) {
      if (!r || r.ok === false || !r.run_id) {
        ui.hideCompass('#spinner');
        $('#findBtn').disabled = false;
        ui.showError('#topicErr', (r && r.message) || 'Could not start the run. Please try again.');
        return;
      }
      pollRun(r.run_id);
    });
  }

  function statusCaption(s) {
    if (s === 'queued') return 'Queued. Waiting for a free slot...';
    if (s === 'running') return 'Reading the recent literature. This can take a few minutes.';
    return 'Working...';
  }

  function pollRun(runId) {
    polling = true;
    var tries = 0, MAX = 480;
    var iv = setInterval(function () {
      tries++;
      api.compassPoll(runId).then(function (r) {
        if (r && r.status === 'done' && r.dashboard) {
          clearInterval(iv); polling = false;
          ui.hideCompass('#spinner');
          $('#findBtn').disabled = false;
          renderDashboard(r.dashboard, { live: true, runId: runId, exportToken: r.export_token });
          announceMap(r, runId);
          $('#dash').scrollIntoView({ behavior: 'smooth' });
        } else if (r && (r.status === 'error' || r.ok === false)) {
          clearInterval(iv); polling = false;
          ui.hideCompass('#spinner');
          $('#findBtn').disabled = false;
          ui.showError('#topicErr', (r && r.message) || 'The run did not finish. Please try again.');
        } else if (tries >= MAX) {
          clearInterval(iv); polling = false;
          ui.hideCompass('#spinner');
          $('#findBtn').disabled = false;
          ui.showError('#topicErr', 'This run is taking longer than expected. If you left an email, we will alert you when it finishes.');
        } else {
          ui.showCompass('#spinner', statusCaption(r && r.status));
        }
      });
    }, 2500);
  }

  /* ---------- worked example on first visit ---------- */
  function loadExample() {
    fetch('/examples/biomath.json', { cache: 'no-cache' }).then(function (res) {
      if (!res.ok) return null;
      return res.json();
    }).then(function (data) {
      if (data) renderDashboard(data, { example: true });
    }).catch(function () { /* the example is a convenience, never block the page on it */ });
  }


  /* The field map belongs to the run, not to this module: the landing page owns the frame and the
     compass page does not have one. So this announces and lets whoever is listening decide, which
     also means a page without a map listener is simply unaffected rather than broken. */
  function announceMap(r, runId) {
    if (!r || !r.map) return;
    try {
      document.dispatchEvent(new CustomEvent('pv:run-map', { detail: {
        state: r.map, runId: runId, topic: r.topic, exportToken: r.export_token
      }}));
    } catch (e) {}
  }

  /* ---------- dashboard rendering (shared by example + live run) ---------- */
  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }
  // Cosmetic tidy for the plain-language reasoning shown to the reader. The
  // backend composes the wording only from the measures and strips any sensitive
  // term server-side before sending, so the page only needs to clear a stray
  // bracketed reference handle like [3] and tidy the spacing it leaves behind.
  function scrub(s) {
    var t = String(s == null ? '' : s);
    t = t.replace(/\[[A-Za-z]?\d+[^\]]*\]/g, '');
    t = t.replace(/\(\s*\)/g, '');
    t = t.replace(/\s{2,}/g, ' ');
    t = t.replace(/\s+([.,;:])/g, '$1');
    return t.trim();
  }
  function fmt(n) {
    if (n == null || n === '') return '-';
    var x = Number(n);
    return isNaN(x) ? esc(n) : x.toLocaleString('en-US');
  }
  function tile(v, k) {
    return '<div class="metric"><div class="v">' + esc(v) + '</div><div class="k">' + esc(k) + '</div></div>';
  }
  function scoreClass(x) {
    var n = Number(x);
    if (n >= 8) return 'good';
    if (n >= 5) return 'warn';
    return 'bad';
  }
  function scoreCell(x) {
    if (x == null || x === '') return '<span class="muted">-</span>';
    return '<span class="tag ' + scoreClass(x) + '">' + esc(x) + ' / 10</span>';
  }

  function renderDashboard(data, opts) {
    opts = opts || {};
    var dash = $('#dash');
    var needles = Array.isArray(data.needles) ? data.needles : [];

    // Honest availability placeholder: { message, needles: [] } with no counts.
    // Render the message plainly rather than a misleading "0 gathered / 0 found".
    if (data.message && !needles.length && data.haystack_count == null) {
      dash.innerHTML = '<div class="card"><h2 style="margin:0 0 .4em">' +
        esc(data.topic || 'Coverage') + '</h2><p class="muted">' +
        esc(scrub(data.message)) + '</p></div>';
      dash.hidden = false;
      return;
    }

    var html = '<div class="card">';

    html += '<div class="row">';
    html += '<h2 style="margin:0">' + esc(data.topic || 'Results') + '</h2>';
    if (opts.example) html += '<span class="tag">worked example</span>';
    html += '</div>';
    if (data.window) html += '<p class="muted" style="margin:.3em 0 0">Looking back over ' + esc(data.window) + '.</p>';

    html += '<div class="scorebar" style="margin-top:16px">';
    html += tile(fmt(data.haystack_count), 'papers gathered');
    html += tile(String(needles.length), needles.length === 1 ? 'needle found' : 'needles found');
    html += '</div>';

    if (data.notes) html += '<div class="verdict">' + esc(scrub(data.notes)) + '</div>';

    // Resolve a needle to its paper URL. Only http(s) links and bare DOIs pass;
    // any other scheme (javascript:, data:) yields no link, so a title or source
    // can never become a live handle. arxiv ids arrive as full https URLs; a bare
    // DOI resolves through doi.org.
    function paperUrl(n) {
      var u = String((n && (n.url || n.doi || n.id)) || '').trim();
      if (/^https?:\/\//i.test(u)) return u;
      if (/^10\./.test(u)) return 'https://doi.org/' + u;
      return '';
    }
    function titleCell(n) {
      var u = paperUrl(n), t = esc(n.title || '');
      return u ? '<a href="' + esc(u) + '" target="_blank" rel="noopener">' + t + '</a>' : t;
    }
    // The source label also links to the paper (doi.org for a DOI, the arxiv page
    // for an arxiv id) so the identifier is a visible, clickable element too.
    function sourceCell(n) {
      var u = paperUrl(n), s = esc(n.source || '');
      return u ? '<a href="' + esc(u) + '" target="_blank" rel="noopener">' + s + '</a>' : s;
    }
    // The prior work the evaluation cited, deduped across every needle in the run.
    function allCitedWorks(ns) {
      var m = {};
      (ns || []).forEach(function (n) {
        (n.cited_works || []).forEach(function (w) {
          var k = w.doi || w.source_id || w.title;
          if (k && !m[k]) m[k] = w;
        });
      });
      return Object.keys(m).map(function (k) { return m[k]; });
    }
    function toBibtex(works) {
      return works.map(function (w, i) {
        var key = 'pubverse' + (w.doi ? w.doi.replace(/[^a-zA-Z0-9]/g, '') : (i + 1));
        var lines = ['@article{' + key + ','];
        if (w.title) lines.push('  title = {' + String(w.title).replace(/[{}]/g, '') + '},');
        if (w.year) lines.push('  year = {' + w.year + '},');
        if (w.doi) lines.push('  doi = {' + w.doi + '},');
        var u = w.url || (w.doi ? 'https://doi.org/' + w.doi : '');
        if (u) lines.push('  url = {' + u + '},');
        lines.push('}');
        return lines.join('\n');
      }).join('\n\n');
    }
    function downloadBib(works, topic) {
      var blob = new Blob([toBibtex(works)], { type: 'application/x-bibtex' });
      var a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'pubverse-cited-works' + (topic ? '-' + String(topic).replace(/[^a-z0-9]+/gi, '-').toLowerCase() : '') + '.bib';
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
      setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
    }
    // Resolve a dimension's dim-local [N#] handles to its cited works.
    function refMap(works) {
      var m = {};
      (works || []).forEach(function (w) {
        (w.refs && w.refs.length ? w.refs : (w.ref ? [w.ref] : [])).forEach(function (r) { if (r) m[r] = w; });
      });
      return m;
    }
    // Render one raw grounded verdict, turning each [N#] into a superscript link to the work it cites.
    function renderVerdict(text, works) {
      var rmap = refMap(works);
      var t = esc(String(text == null ? '' : text));
      t = t.replace(/\[([A-Za-z]?\d+[^\]]*)\]/g, function (whole, inner) {
        var key = String(inner).split(/\s+/)[0];
        var w = rmap[key];
        if (!w) return '';
        var u = w.url || (w.doi ? 'https://doi.org/' + w.doi : '');
        var tip = esc((w.title || '') + (w.year ? ' (' + w.year + ')' : ''));
        return u ? '<sup><a href="' + esc(u) + '" target="_blank" rel="noopener" title="' + tip + '">' + esc(key) + '</a></sup>'
                 : '<sup title="' + tip + '">' + esc(key) + '</sup>';
      });
      return t.replace(/\(\s*\)/g, '').replace(/\s{2,}/g, ' ').trim();
    }
    // Grounding receipts: for the novelty judgment, show the nearest PRIOR WORK it was grounded against,
    // each with its cosine to this paper (bar vs the 0.62 relevance / 0.82 near-duplicate lines) and the
    // retrieved passage of what that prior did. This is exactly the evidence the model judged against, so
    // an expert can verify the call in seconds. Renders only when the per-prior similarity is present.
    function groundingReceipts(works) {
      var ws = (works || []).filter(function (w) { return w.sim != null; })
                .sort(function (a, b) { return (b.sim || 0) - (a.sim || 0); });
      if (!ws.length) return '';
      var rows = ws.map(function (w) {
        var sim = +w.sim, pct = Math.max(0, Math.min(1, sim)) * 100;
        var col = sim >= 0.82 ? '#e0685f' : (sim >= 0.62 ? '#3ea66b' : '#f6c445');
        var src = (w.source_type && w.source_type !== 'dense')
          ? ' <span style="font-size:10px;color:#0b57d0;border:1px solid #b9c9ee;border-radius:3px;padding:0 3px">' + esc(w.source_type) + '</span>' : '';
        var did = w.did ? '<div style="color:#666;font:11px ui-monospace,Menlo,monospace;margin:2px 0 0 20px">' + esc(String(w.did).slice(0, 220)) + '</div>' : '';
        return '<div style="margin:5px 0;line-height:1.4">' +
          '<span style="color:#0b57d0;font-weight:600">[' + esc(w.ref || '?') + ']</span> ' +
          esc(String(w.title || '').slice(0, 88)) + (w.year ? ' <span style="color:#aaa">' + esc(w.year) + '</span>' : '') + src +
          '<span style="display:inline-block;width:90px;height:9px;background:#eef0f4;border-radius:3px;vertical-align:middle;margin:0 6px;position:relative">' +
            '<span style="position:absolute;left:0;top:0;height:100%;border-radius:3px;width:' + pct.toFixed(0) + '%;background:' + col + '"></span></span>' +
          '<span style="font:11px ui-monospace,monospace;color:#444">' + sim.toFixed(2) + '</span>' + did + '</div>';
      }).join('');
      return '<details style="margin-top:6px"><summary class="mini" style="cursor:pointer">what grounded this (' + ws.length + ' nearest prior work)</summary>' +
        '<div class="mini" style="color:#888;margin:2px 0 4px">cosine to this paper: green &ge;0.62 relevant, red &ge;0.82 near-duplicate</div>' + rows + '</details>';
    }
    // The reasoning cell: the model's raw novelty verdict up front, methods + impact in a details.
    // Falls back to the plain summary verdict when a grounded verdict is absent.

    /* "Show me where this one is on the map." The needle list sits directly under the map, so this
       is the obvious question while looking at both. The map is in an iframe -- possibly on another
       host -- so this posts to it rather than reaching into it, and the map widens its own depth
       filter if the paper is deeper than the current setting. Rendered only when a map is actually
       on the page: a button that silently does nothing is worse than no button. */
    function locateBtn(n) {
      if (!n) return '';
      // A needle record does not always carry an explicit id -- the worked example has only a url
      // and a title -- so take whichever handle exists. The map matches on any of them.
      var key = n.id || '';
      if (!key && n.url) {
        var m = String(n.url).match(/(?:abs|10\.\d{4,}\/[^\s]+|content\/)([^\/\s]+)$/);
        key = (m && m[1]) || '';
      }
      if (!key) key = n.title || '';
      if (!key) return '';
      return '<button type="button" class="locate" data-pid="' + esc(String(key)) + '"' +
             ' title="Show this paper on the map" aria-label="Show this paper on the map">i</button>';
    }

    function reasonCell(n) {
      var r = n.reasoning || {}, nov = r.novelty || {}, meth = r.methods || {}, imp = r.impact || {};
      var html = '<div class="verdict-body clamped">' +
                 (nov.text ? renderVerdict(nov.text, nov.works) : esc(scrub(n.verdict))) + '</div>';
      html += groundingReceipts(nov.works);
      if (meth.text || imp.text) {
        html += '<details style="margin-top:6px"><summary class="mini" style="cursor:pointer">methods &amp; impact</summary>';
        if (meth.text) html += '<div style="margin:4px 0"><span class="mini strong">Methods:</span> ' + renderVerdict(meth.text, meth.works) + '</div>';
        if (imp.text) html += '<div style="margin:4px 0"><span class="mini strong">Impact:</span> ' + renderVerdict(imp.text, imp.works) + '</div>';
        html += '</details>';
      }
      // Contribution decomposition triage (present only when PUBVERSE_CONTRIB_DECOMP=1).
      // Shows per-claim delta operators, matched priors, and confidence for expert review.
      if (n.contrib_decomp && n.contrib_decomp.claims && n.contrib_decomp.claims.length) {
        html += renderContribDecomp(n.contrib_decomp);
      }
      // Velocity features (present only when PUBVERSE_DIREC_VEL=1).
      // Shows density/isolation as a compact badge for the triage expert.
      if (n.velocity && n.velocity.density != null) {
        var v = n.velocity;
        var dens = v.density != null ? Math.round(v.density * 100) : null;
        var iso = v.isolation_score != null ? Math.round((1 - v.isolation_score) * 100) : null;
        var dirNov = v.directional_novelty != null ? Math.round(((v.directional_novelty + 1) / 2) * 100) : null;
        html += '<details style="margin-top:4px"><summary class="mini" style="cursor:pointer">position in field</summary>';
        html += '<div style="display:flex;gap:14px;flex-wrap:wrap;margin-top:4px">';
        if (dens != null) html += '<span style="font-size:.75rem;color:var(--muted)">neighbor density: <b style="color:var(--ink)">' + dens + '%</b></span>';
        if (iso != null) html += '<span style="font-size:.75rem;color:var(--muted)">field fit: <b style="color:var(--ink)">' + iso + '%</b></span>';
        if (dirNov != null) html += '<span style="font-size:.75rem;color:var(--muted)">directional novelty: <b style="color:var(--ink)">' + dirNov + '%</b></span>';
        html += '</div></details>';
      }
      // Evidence verification (anti-hallucination gate, always on).
      // Shows whether the judge's cited claims are supported by the retrieved prior text,
      // and flags near-duplicate priors that need full-text comparison.
      if (n.evidence_verified) {
        var ev = n.evidence_verified;
        var sup = (ev.novelty_verification || {}).supported_claims || 0;
        var unsup = (ev.novelty_verification || {}).unsupported_claims || 0;
        var nd = (ev.prior_proximity || {}).near_duplicates || 0;
        var cl = (ev.prior_proximity || {}).close_priors || 0;
        var flags = ev.flags || [];
        if (sup > 0 || unsup > 0 || nd > 0 || cl > 0 || flags.length > 0) {
          var warnColor = unsup > 0 ? 'var(--bad)' : (nd > 0 ? 'var(--warn)' : 'var(--good)');
          html += '<details style="margin-top:4px"><summary class="mini" style="cursor:pointer;color:' + warnColor + '">';
          html += 'evidence check: ' + sup + ' supported';
          if (unsup > 0) html += ', <b>' + unsup + ' unsupported</b>';
          if (nd > 0) html += ', ' + nd + ' near-duplicate';
          html += '</summary>';
          if (flags.length) {
            html += '<div style="margin-top:4px">';
            flags.forEach(function(f) {
              var fc = f.indexOf('unsupported') >= 0 ? 'var(--bad)' :
                       (f.indexOf('near-duplicate') >= 0 ? 'var(--warn)' : 'var(--muted)');
              html += '<div style="font-size:.75rem;color:' + fc + ';margin:2px 0">&#9888; ' + esc(f) + '</div>';
            });
            html += '</div>';
          }
          html += '</details>';
        }
      }
      return html;
    }
    // Render the contribution decomposition as an expandable triage panel.
    // Each claim shows its delta operator (badge), the matched prior (if any),
    // confidence (bar), and whether the union of priors covers it.
    function renderContribDecomp(decomp) {
      var claims = decomp.claims || [];
      var tier = (decomp.aggregate || {}).tier || decomp.novelty_tier || '';
      var tierColors = {high:'var(--good)', moderate:'var(--warn)', low:'var(--bad)',
                        uncontested:'var(--muted)', unknown:'var(--muted)'};
      var tierColor = tierColors[tier] || 'var(--muted)';
      var opLabels = {
        'restate':'restates prior', 'add-constraint':'adds a constraint',
        'change-mechanism':'new mechanism', 'recombine':'recombines priors',
        'scale':'scales to new domain', 'contradict':'contradicts prior',
        'uncontested':'no matching prior', 'error':'error'
      };
      var html = '<details style="margin-top:8px"><summary class="mini" style="cursor:pointer">' +
        'contribution breakdown (' + claims.length + ' claim' + (claims.length !== 1 ? 's' : '') + ')' +
        (tier ? ' <span style="color:' + tierColor + ';font-weight:600">' + esc(tier) + '</span>' : '') +
        '</summary>';
      html += '<div style="margin-top:6px">';
      claims.forEach(function (c, i) {
        var op = c.delta || c.delta_operator || '';
        var conf = c.confidence != null ? c.confidence : (c.delta_confidence != null ? c.delta_confidence : 0);
        var sim = c.best_sim != null ? c.best_sim : '';
        var claimText = (c.claim || c.claim_text || c.central_finding || '').slice(0, 120);
        var union = c.union_covers;
        var opLabel = opLabels[op] || op;
        var opColor = {'restate':'var(--bad)','add-constraint':'var(--warn)',
          'change-mechanism':'var(--good)','recombine':'var(--good)',
          'scale':'var(--warn)','contradict':'var(--good)',
          'uncontested':'var(--muted)'}[op] || 'var(--muted)';
        html += '<div style="margin:6px 0;padding:8px 10px;background:var(--paper);border-radius:var(--r-s);border-left:3px solid ' + opColor + '">';
        html += '<div style="font-size:.85rem;margin-bottom:2px">' + esc(claimText) + '</div>';
        html += '<div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">';
        html += '<span class="tag" style="background:' + opColor + '20;color:' + opColor + '">' + esc(opLabel) + '</span>';
        if (conf > 0) {
          var pct = Math.round(conf * 100);
          html += '<span style="font-size:.75rem;color:var(--muted)">conf ';
          html += '<span style="display:inline-block;width:40px;height:6px;background:#e0e4ea;border-radius:3px;vertical-align:middle;margin:0 4px">';
          html += '<span style="display:block;height:100%;width:' + pct + '%;background:var(--blue);border-radius:3px"></span></span>';
          html += pct + '%</span>';
        }
        if (sim !== '' && sim != null) {
          html += '<span style="font-size:.75rem;color:var(--muted)">sim ' + esc(sim.toFixed ? sim.toFixed(3) : sim) + '</span>';
        }
        if (union === false) {
          html += '<span class="tag" style="background:#e6f6ee;color:var(--good)">union gap</span>';
        }
        html += '</div>';
        html += '</div>';
      });
      var summary = (decomp.aggregate || {}).summary;
      if (summary) {
        html += '<div class="mini" style="margin-top:6px;color:var(--muted)">' + esc(summary) + '</div>';
      }
      html += '</div></details>';
      return html;
    }
    var cited = allCitedWorks(needles);
    if (needles.length) {
      html += '<div style="overflow-x:auto"><table><thead><tr>' +
        '<th>Title</th><th>Source</th><th>Date</th><th>Novelty</th><th>Methods</th><th>Impact</th><th>Why it stood out</th>' +
        '</tr></thead><tbody>';
      needles.forEach(function (n) {
        html += '<tr class="needle-row" data-pid="' + esc(String(n.id || '')) + '">' +
          '<td>' + titleCell(n) + '</td>' +
          '<td class="nowrap">' + sourceCell(n) + '</td>' +
          '<td class="nowrap">' + esc(n.date) + '</td>' +
          '<td>' + scoreCell(n.novelty) + '</td>' +
          '<td>' + scoreCell(n.methods) + '</td>' +
          '<td>' + scoreCell(n.impact) + '</td>' +
          '<td>' + reasonCell(n) + locateBtn(n) + '</td>' +
          '</tr>';
      });
      html += '</tbody></table></div>';
      if (cited.length) {
        html += '<details style="margin-top:16px"><summary class="strong" style="cursor:pointer">Cited relevant works (' + cited.length + ')</summary>' +
                '<p class="muted" style="margin:6px 0">The prior work these needles were evaluated against.</p><ul style="margin:0;padding-left:18px">';
        cited.forEach(function (w) {
          var wu = w.url || (w.doi ? 'https://doi.org/' + w.doi : '');
          var wt = esc(w.title || '(untitled)') + (w.year ? ' (' + esc(w.year) + ')' : '');
          html += '<li style="margin:4px 0">' + (wu ? '<a href="' + esc(wu) + '" target="_blank" rel="noopener">' + wt + '</a>' : wt) + '</li>';
        });
        html += '</ul></details>';
      }
    } else {
      html += '<p class="muted">No needles cleared the bar in this run. That is a real result, not a failure. Most subjects are mostly incremental at any given moment.</p>';
    }

    html += '<div class="row" style="margin-top:20px">';
    if (opts.runId) {
      html += '<button class="btn ghost sm" id="saveTopicBtn">Save topic to my profile</button>';
      html += '<a class="btn ghost sm" id="expNeedles">Export needles (.tsv)</a>';
      html += '<a class="btn ghost sm" id="expHaystack">Export full scope (.tsv)</a>';
      if (cited.length) html += '<a class="btn ghost sm" id="expBib">Cited works (.bib)</a>';
    } else if (opts.example) {
      html += '<span class="muted">This is a completed run you can explore. Enter your own topic above to start a new one.</span>';
    }
    html += '</div>';

    html += '</div>';
    dash.innerHTML = html;
    dash.hidden = false;
    dash.classList.add('fade-in');

    /* Long verdicts are collapsed to a few lines with a "…" trail rather than cut. The judge's
       reasoning against the retrieved prior work is the substance of the card, and a table of
       twenty full paragraphs is unreadable, so it is folded here and opened on demand -- not
       shortened, which is what the pipeline used to do to it at 400 characters. */
    Array.prototype.forEach.call(dash.querySelectorAll('.needle-row td:last-child'), function (cell) {
      var body = cell.querySelector('.verdict-body');
      if (!body) return;
      // Only fold what is actually overflowing; a two-line verdict needs no control.
      if (body.scrollHeight - body.clientHeight < 8) { body.classList.remove('clamped'); return; }
      var more = document.createElement('button');
      more.type = 'button'; more.className = 'more'; more.textContent = 'Show more';
      more.setAttribute('aria-expanded', 'false');
      more.addEventListener('click', function () {
        var open = body.classList.toggle('clamped') === false;
        more.textContent = open ? 'Show less' : 'Show more';
        more.setAttribute('aria-expanded', open ? 'true' : 'false');
      });
      body.parentNode.insertBefore(more, body.nextSibling);
    });

    /* The (i) on a row asks the map to fly to that paper. Delegated, so it survives re-render. */
    dash.addEventListener('click', function (e) {
      var b = e.target && e.target.closest && e.target.closest('button.locate');
      if (!b) return;
      e.preventDefault();
      var frame = document.getElementById('fieldGraphFrame');
      if (!frame || !frame.contentWindow || !frame.getAttribute('src')) return;
      try {
        frame.contentWindow.postMessage({ type: 'pv:focus-node', key: b.getAttribute('data-pid') }, '*');
        var sec = document.getElementById('fieldGraphSection');
        if (sec && !sec.hidden) sec.scrollIntoView({ behavior: 'smooth', block: 'center' });
      } catch (err) { console.error('[pubverse] locate', err); }
    });

    if (opts.runId) {
      var en = $('#expNeedles'), eh = $('#expHaystack'), sb = $('#saveTopicBtn');
      if (en) { en.href = api.compassExportUrl(opts.runId, 'needles', 'id,source,date,title,novelty,methods,impact', opts.exportToken); en.setAttribute('download', ''); }
      if (eh) { eh.href = api.compassExportUrl(opts.runId, 'haystack', 'id,source,date,title', opts.exportToken); eh.setAttribute('download', ''); }
      var eb = $('#expBib');
      if (eb) on(eb, 'click', function (e) { e.preventDefault(); downloadBib(cited, data.topic); });
      if (sb) on(sb, 'click', function () { saveTopic(data.topic, sb); });
    }
  }

  /* Save a topic to the signed-in user's profile, server-side, so it persists
     across browsers and devices and appears in Profile. Success is reported
     only on an ok response from the backend. */
  function saveTopic(topic, btn) {
    if (!topic) return;
    btn.disabled = true;
    var months = parseInt($('#months').value, 10);
    api.compassSaveTopic(topic, isNaN(months) ? undefined : months).then(function (r) {
      if (r && r.ok) {
        btn.textContent = 'Saved to your profile';
      } else {
        btn.disabled = false;
        ui.showError('#topicErr', (r && r.message) || 'Could not save the topic. Please try again.');
      }
    });
  }
  PV.compass = { init: initPage };
})();
