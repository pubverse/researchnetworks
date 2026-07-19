/* PubVerse UI helpers. No framework, vanilla ES6.
   Four small jobs the pages share: the 42 compass spinner, a live counter
   (words for the abstract, chars for the compass topic), the inline error box,
   and tab switching. All class names match css/app.css exactly.
   Requires js/config.js to have loaded first. */
(function () {
  'use strict';

  if (!window.PV) {
    throw new Error('PubVerse: js/config.js must load before js/ui.js');
  }

  // The 42 compass mark. Pages may also place their own spinner markup; if the
  // container already holds an <img>, we leave it alone and just toggle it.
  var COMPASS_SRC = 'assets/compass42.png';

  // Resolve an element from a selector string or pass an element through.
  function el(x) {
    return typeof x === 'string' ? document.querySelector(x) : (x || null);
  }
  function toArray(nodeList) {
    return Array.prototype.slice.call(nodeList);
  }

  /* ------- 42 compass spinner ------- */

  // If the .compass-load container is empty, build the mark and a caption line.
  function ensureCompass(node) {
    if (!node.querySelector('img')) {
      var img = document.createElement('img');
      img.src = COMPASS_SRC;
      img.alt = '';
      node.appendChild(img);
      var cap = document.createElement('div');
      cap.className = 'cap';
      node.appendChild(cap);
    }
    return node;
  }
  function showCompass(target, caption) {
    var node = el(target);
    if (!node) return;
    ensureCompass(node);
    if (caption != null) {
      var cap = node.querySelector('.cap');
      if (cap) cap.textContent = caption;
    }
    node.classList.add('show');
  }
  function hideCompass(target) {
    var node = el(target);
    if (node) node.classList.remove('show');
  }

  /* ------- live counter ------- */

  function wordCount(v) {
    var t = (v || '').trim();
    return t ? t.split(/\s+/).length : 0;
  }

  // Wire an input or textarea to a counter element. Updates on every keystroke,
  // marks the counter red (.over) past the limit, and returns the over-limit
  // boolean from update() so a page can block submit. The returned controller
  // also exposes isOver() and value() for on-demand checks.
  // opts: { mode:'words'|'chars', limit, onChange(over, count) }
  function bindCounter(input, counter, opts) {
    opts = opts || {};
    var inputEl = el(input);
    var counterEl = el(counter);
    var chars = opts.mode === 'chars';
    var limit = opts.limit != null ? opts.limit
      : (chars ? window.PV.TOPIC_CHAR_LIMIT : window.PV.WORD_LIMIT);
    var unit = chars ? '' : ' words';
    var measure = chars ? function (v) { return (v || '').length; } : wordCount;
    var onChange = typeof opts.onChange === 'function' ? opts.onChange : null;

    function readValue() {
      return measure(inputEl ? inputEl.value : '');
    }
    function update() {
      var n = readValue();
      var over = n > limit;
      if (counterEl) {
        counterEl.textContent = n + ' / ' + limit + unit;
        counterEl.classList.toggle('over', over);
      }
      if (onChange) onChange(over, n);
      return over;
    }
    if (inputEl) inputEl.addEventListener('input', update);
    update();

    return {
      update: update,
      value: readValue,
      isOver: function () { return readValue() > limit; }
    };
  }

  // The abstract word counter used by the Score page.
  function bindWordCounter(input, counter, opts) {
    var merged = {};
    if (opts) { for (var k in opts) { if (Object.prototype.hasOwnProperty.call(opts, k)) merged[k] = opts[k]; } }
    merged.mode = 'words';
    return bindCounter(input, counter, merged);
  }

  // The topic character counter used by the Compass page.
  function bindCharCounter(input, counter, opts) {
    var merged = {};
    if (opts) { for (var k in opts) { if (Object.prototype.hasOwnProperty.call(opts, k)) merged[k] = opts[k]; } }
    merged.mode = 'chars';
    return bindCounter(input, counter, merged);
  }

  /* ------- inline error box (.err -> .err.show) ------- */

  function showError(target, message) {
    var node = el(target);
    if (!node) return;
    node.textContent = message || 'Something went wrong. Please try again.';
    node.classList.add('show');
  }
  function clearError(target) {
    var node = el(target);
    if (!node) return;
    node.textContent = '';
    node.classList.remove('show');
  }

  /* ------- tabs (.tabs / .tabpane) ------- */

  // container wraps one .tabs bar and its .tabpane panels. Each button carries
  // data-tab="KEY"; each pane matches by data-tab="KEY" or by id="KEY".
  // Returns a controller with show(key) for programmatic switching.
  function initTabs(container) {
    var root = el(container);
    if (!root) return null;
    var bar = root.querySelector('.tabs');
    if (!bar) return null;
    var buttons = toArray(bar.querySelectorAll('button'));
    var panes = toArray(root.querySelectorAll('.tabpane'));

    function activate(key) {
      if (!key) return;
      buttons.forEach(function (b) {
        b.classList.toggle('active', b.getAttribute('data-tab') === key);
      });
      panes.forEach(function (p) {
        var match = p.getAttribute('data-tab') === key || p.id === key;
        p.classList.toggle('active', match);
      });
    }
    buttons.forEach(function (b) {
      b.addEventListener('click', function () { activate(b.getAttribute('data-tab')); });
    });

    var current = bar.querySelector('button.active') || buttons[0];
    if (current) activate(current.getAttribute('data-tab'));

    return { show: activate };
  }

  window.PV.ui = {
    showCompass: showCompass,
    hideCompass: hideCompass,
    wordCount: wordCount,
    bindCounter: bindCounter,
    bindWordCounter: bindWordCounter,
    bindCharCounter: bindCharCounter,
    showError: showError,
    clearError: clearError,
    initTabs: initTabs
  };
})();
