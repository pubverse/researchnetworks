/* SDSU DRI hub page: the app runs on the same public backend host as PubVerse, on its own HTTPS port (10000).
   Derived from window.PV.API_BASE so no backend host name is written into this public repo. */
(function () {
  'use strict';
  var a = document.getElementById('signin');
  if (!a || !window.PV || !window.PV.API_BASE) return;
  try {
    var u = new URL(window.PV.API_BASE);
    u.port = '10000'; u.pathname = '/'; u.search = ''; u.hash = '';
    a.href = u.href;
  } catch (e) { /* leave the button inert rather than guess */ }
})();
