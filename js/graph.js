/* The field map.
 *
 * Renders the payload from compass/build_field_graph.py. Self-contained on purpose: the pipeline's
 * own visualiser (create_interactive_visualization.py) emits ~8.7MB of React plus four cdnjs
 * scripts and transpiles JSX in the browser with babel-standalone, which is fine for a standalone
 * artifact you open on its own and wrong for a section of a landing page that must paint fast and
 * keep working when a CDN does not. So the geometry is copied from that file and the machinery is
 * not.
 *
 * COPIED VERBATIM from create_interactive_visualization.py: the star points
 * (createStarPoints(0,0,size,size/2.5,5)), the hexagon used for methods/computational/experimental
 * clusters, the diamond used for problems clusters, and the pentagon used for results.
 *
 * ONE DELIBERATE DEVIATION. In the pipeline, shape encodes node TYPE -- input article versus
 * cluster subtype, where the impact cluster is a triangle. Here shape encodes paper TIER, because
 * that is what the brief asks for: star = a compass needle, triangle = a paper above the field
 * median, circle = everything else. Triangle therefore means one thing only, and the topic hubs
 * take the pipeline's other shapes (diamond for pains, hexagon for methods, pentagon for impact)
 * rather than competing for it.
 *
 * Layout is deterministic, not a force simulation: hubs on a circle, papers around their own hub at
 * a radius set by the band. A landing page should not spend a second of CPU settling a simulation,
 * and a stable layout means the map looks the same to two people discussing it.
 */
(function () {
  'use strict';
  var PV = window.PV = window.PV || {};

  var COLORS = { pains: '#7b6ef6', impact: '#d98a2b', methods: '#2f6db0',
                 needle: '#c0483f', tri: '#2e9e6b', dot: '#b9b4ab' };

  function starPoints(cx, cy, outer, inner, points) {
    // create_interactive_visualization.py's createStarPoints, same construction.
    var p = [], step = Math.PI / points;
    for (var i = 0; i < points * 2; i++) {
      var r = (i % 2 === 0) ? outer : inner, a = i * step - Math.PI / 2;
      p.push((cx + r * Math.cos(a)).toFixed(2) + ',' + (cy + r * Math.sin(a)).toFixed(2));
    }
    return p.join(' ');
  }
  function polyPoints(cx, cy, r, n, rot) {
    var p = [];
    for (var i = 0; i < n; i++) {
      var a = (i * 2 * Math.PI / n) + (rot || -Math.PI / 2);
      p.push((cx + r * Math.cos(a)).toFixed(2) + ',' + (cy + r * Math.sin(a)).toFixed(2));
    }
    return p.join(' ');
  }
  var HUB_SHAPE = { pains: 4, methods: 6, impact: 5 };   // diamond, hexagon, pentagon

  function load(src) {
    return fetch(src, { cache: 'no-cache' }).then(function (r) {
      if (!r.ok) throw new Error('field map ' + r.status + ' for ' + src);
      return r.json();
    });
  }

  function render(el, data) {
    if (!el || !data || !data.nodes) return;
    var W = el.clientWidth || 900, H = 460, cx = W / 2, cy = H / 2;
    var topics = (data.topics || []).map(function (t) { return t.label; });
    var hub = {};
    topics.forEach(function (t, i) {
      var a = (i * 2 * Math.PI / topics.length) - Math.PI / 2;
      // Ellipse, not a circle: the card is far wider than it is tall, and a min(W,H) radius
      // left the whole map huddled in the middle of a lot of empty paper.
      hub[t] = { x: cx + Math.cos(a) * W * 0.30,
                 y: cy + Math.sin(a) * H * 0.30 };
    });

    // Place each paper around its hub. Angle from a hash of the id so it is stable between loads;
    // radius from the band, so the toggle moves points the viewer can already see.
    var bands = data.nodes.map(function (n) { return n.band; });
    var bmin = Math.min.apply(null, bands), bmax = Math.max.apply(null, bands) || 1;
    data.nodes.forEach(function (n, i) {
      var h = hub[n.topic] || { x: cx, y: cy };
      var seed = 0; for (var k = 0; k < n.id.length; k++) seed = (seed * 31 + n.id.charCodeAt(k)) % 9973;
      var ang = (seed / 9973) * 2 * Math.PI;
      var norm = (n.band - bmin) / ((bmax - bmin) || 1);
      var rad = 30 + (1 - norm) * Math.min(W * 0.16, H * 0.34);  // high band sits closer to its hub
      n._x = h.x + Math.cos(ang) * rad;
      n._y = h.y + Math.sin(ang) * rad;
    });

    var maxDeg = Math.max.apply(null, data.nodes.map(function (n) { return n.degree || 0; })) || 1;
    var state = { degree: 1, bandMin: 0, needlesOnly: false };

    function visible(n) {
      if (n.shape === 'star') return true;               // needles are always visible, as specified
      if ((n.degree || 0) < state.degree) return false;
      var norm = (n.band - bmin) / ((bmax - bmin) || 1);
      return norm >= state.bandMin;
    }

    function draw() {
      var svg = ['<svg viewBox="0 0 ' + W + ' ' + H + '" width="100%" height="' + H +
                 '" role="img" aria-label="Map of the field by topic">'];
      var shown = {};
      data.nodes.forEach(function (n) { if (visible(n)) shown[n.id] = n; });

      (data.edges || []).forEach(function (e) {
        if (e.k === 'topic') {
          var n = shown[e.s]; if (!n) return;
          var h = hub[n.topic]; if (!h) return;
          svg.push('<line x1="' + n._x.toFixed(1) + '" y1="' + n._y.toFixed(1) + '" x2="' +
                   h.x.toFixed(1) + '" y2="' + h.y.toFixed(1) +
                   '" stroke="' + (COLORS[n.topic] || '#ccc') + '" stroke-opacity="' +
                   (n.shape === 'star' ? 0.5 : 0.13) + '" stroke-width="' +
                   (n.shape === 'star' ? 1.2 : 0.6) + '"/>');
        } else if (shown[e.s] && shown[e.t]) {
          var a = shown[e.s], b = shown[e.t];
          svg.push('<line x1="' + a._x.toFixed(1) + '" y1="' + a._y.toFixed(1) + '" x2="' +
                   b._x.toFixed(1) + '" y2="' + b._y.toFixed(1) +
                   '" stroke="#cfcac2" stroke-opacity="0.35" stroke-width="0.5"/>');
        }
      });

      topics.forEach(function (t) {
        var h = hub[t];
        svg.push('<polygon points="' + polyPoints(h.x, h.y, 15, HUB_SHAPE[t] || 6) +
                 '" fill="' + (COLORS[t] || '#888') + '" fill-opacity="0.9"/>');
        svg.push('<text x="' + h.x.toFixed(1) + '" y="' + (h.y + 30).toFixed(1) +
                 '" text-anchor="middle" font-size="12" fill="#6f6b66">' + t + '</text>');
      });

      var stars = 0, tris = 0, dots = 0;
      Object.keys(shown).forEach(function (id) {
        var n = shown[id], title = (n.title || '').replace(/[<&>"]/g, '');
        if (n.shape === 'star') {
          stars++;
          svg.push('<polygon points="' + starPoints(n._x, n._y, 9, 9 / 2.5, 5) +
                   '" fill="' + COLORS.needle + '" stroke="#fff" stroke-width="1">' +
                   '<title>' + title + '</title></polygon>');
        } else if (n.shape === 'triangle') {
          tris++;
          svg.push('<polygon points="' + polyPoints(n._x, n._y, 6, 3) +
                   '" fill="' + COLORS.tri + '" fill-opacity="0.85"><title>' + title + '</title></polygon>');
        } else {
          dots++;
          svg.push('<circle cx="' + n._x.toFixed(1) + '" cy="' + n._y.toFixed(1) +
                   '" r="2.6" fill="' + COLORS.dot + '" fill-opacity="0.75"><title>' + title + '</title></circle>');
        }
      });
      svg.push('</svg>');
      el.querySelector('.fg-canvas').innerHTML = svg.join('');
      var c = el.querySelector('.fg-count');
      if (c) c.textContent = stars + ' needles · ' + tris + ' above average · ' + dots + ' other';
    }

    el.innerHTML =
      '<div class="fg-canvas"></div>' +
      '<div class="fg-controls">' +
      '  <label>Zoom by connections <input type="range" class="fg-deg" min="1" max="' + maxDeg +
      '" value="1" aria-label="Minimum connections"></label>' +
      '  <label>' + (data.band && data.band.label ? data.band.label : 'band') +
      '    <input type="range" class="fg-band" min="0" max="0.95" step="0.05" value="0"' +
      '           aria-label="Minimum band"></label>' +
      '  <span class="fg-count mini muted"></span>' +
      '</div>';

    el.querySelector('.fg-deg').addEventListener('input', function (e) {
      state.degree = parseInt(e.target.value, 10) || 1; draw();
    });
    el.querySelector('.fg-band').addEventListener('input', function (e) {
      state.bandMin = parseFloat(e.target.value) || 0; draw();
    });

    // Default band: the highest setting that still leaves roughly as many context papers on the
    // map as there are needles, which is what the brief asks for -- the sparsest view that is still
    // a field rather than a shortlist. Needles are unconditionally visible either way, so this
    // tunes the surroundings and never the count that matters. Solved for rather than guessed,
    // because a hardcoded 0.8 showed 18 of 320 and read as an empty canvas.
    var needles = data.nodes.filter(function (n) { return n.shape === 'star'; }).length;
    var target = Math.max(needles, 40), best = 0;
    for (var b = 0.95; b >= 0; b -= 0.05) {
      state.bandMin = b;
      if (data.nodes.filter(visible).length >= target) { best = b; break; }
    }
    state.bandMin = best;
    var slider = el.querySelector('.fg-band');
    if (slider) slider.value = String(best);
    draw();
  }

  PV.graph = { load: load, render: render };
})();
