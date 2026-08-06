#!/usr/bin/env bash
# External uptime probe for pubverse.ai.
#
# WHY THIS EXISTS
#   The on-box watchdog on HPCF cannot report the one failure that matters most: HPCF itself
#   going away. If that machine loses power or its uplink, the watchdog dies with it and no alert
#   is ever sent. On 2026-08-06 a user reported that sign-in was broken while the on-box watchdog
#   logged an unbroken "ok" every 5 minutes.
#
#   So this runs on GitHub's runners instead. It is deliberately outside the house, on a different
#   network, resolving DNS through a different resolver. It needs no secrets and no third-party
#   account: when it fails, the workflow fails, and GitHub emails the repository owner.
#
# THE TRAP IT AVOIDS
#   Run from a GitHub runner, hpcf.tail8ba9b3.ts.net resolves publicly to Tailscale's ingress and
#   everything is honest. Run from a machine on the owner's tailnet, MagicDNS resolves the SAME
#   name to 100.115.159.103, the request travels the tailnet, and the probe can report a cheerful
#   200 straight through a total public outage. So if the name resolves into 100.64.0.0/10 we pin
#   the public addresses explicitly. That keeps this script honest in both places, which matters
#   because it is the one you will reach for when debugging by hand.
#
# Usage:  ops/probe.sh            (exit 0 = all good, exit 1 = something a visitor would hit)
set -uo pipefail

SITE=https://pubverse.ai
API_HOST=hpcf.tail8ba9b3.ts.net
CERT_MIN_DAYS=14

fails=()
note() { printf '  %s\n' "$*"; }
fail() { fails+=("$1"); printf '  FAIL: %s\n' "$1"; }
ok()   { printf '  ok:   %s\n' "$1"; }

# --- work out whether we must pin public addresses -------------------------------------------
resolved=$(getent hosts "$API_HOST" 2>/dev/null | awk '{print $1}' | head -1)
RESOLVE_ARGS=()
case "$resolved" in
  100.6[4-9].*|100.[7-9][0-9].*|100.1[0-1][0-9].*|100.12[0-7].*)
    note "note: $API_HOST resolves to $resolved (tailnet). Pinning public ingress so this stays honest."
    pub=$(dig +short +time=5 @1.1.1.1 "$API_HOST" A 2>/dev/null | grep -E '^[0-9.]+$' | head -1)
    [ -z "$pub" ] && pub=$(dig +short +time=5 @8.8.8.8 "$API_HOST" A 2>/dev/null | grep -E '^[0-9.]+$' | head -1)
    if [ -z "$pub" ]; then
      fail "cannot resolve a public address for $API_HOST on any public resolver"
    else
      RESOLVE_ARGS=(--resolve "$API_HOST:443:$pub")
      note "pinned $API_HOST -> $pub"
    fi
    ;;
  *) note "resolves to ${resolved:-<public resolver>}, using it directly" ;;
esac

api() { curl -s -m 20 "${RESOLVE_ARGS[@]+"${RESOLVE_ARGS[@]}"}" "$@"; }

echo "PubVerse external probe  $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

# --- 1. the static site ------------------------------------------------------------------------
code=$(curl -s -o /tmp/pv_index -m 20 -w '%{http_code}' "$SITE/" 2>/dev/null)
if [ "$code" = 200 ] && grep -q "PubVerse" /tmp/pv_index 2>/dev/null; then
  ok "pubverse.ai serves the page"
else
  fail "pubverse.ai returned HTTP ${code:-no-response} or the body was not the site"
fi

# --- 2. the API answers ------------------------------------------------------------------------
body=$(api "https://$API_HOST/api/health" 2>/dev/null)
case "$body" in
  *'"ok":true'*) ok "/api/health answers" ;;
  *) fail "/api/health did not answer ok:true (got: ${body:-empty})" ;;
esac

# --- 3. the browser preflight for sign-in ------------------------------------------------------
hdrs=$(api -D - -o /dev/null -X OPTIONS "https://$API_HOST/api/login" \
        -H "Origin: $SITE" -H 'Access-Control-Request-Method: POST' \
        -H 'Access-Control-Request-Headers: content-type' 2>/dev/null)
if printf '%s' "$hdrs" | grep -qi "^access-control-allow-origin:.*pubverse.ai"; then
  ok "sign-in CORS preflight allowed"
else
  fail "sign-in preflight not allowed; a browser would report the service as unreachable"
fi

# --- 4. the auth path actually runs ------------------------------------------------------------
# Deliberately wrong credentials. A 401 carrying the app's own JSON proves routing, the users
# file and argon2 are all intact. No secret is needed and none is stored. 429 also proves the
# endpoint is alive (the throttle keys on the caller, and runner IPs rotate, so this cannot
# consume a real visitor's allowance).
resp=$(api -w '\n%{http_code}' -X POST "https://$API_HOST/api/login" \
        -H "Origin: $SITE" -H 'Content-Type: application/json' \
        -d '{"username":"gh-uptime-probe","password":"deliberately-wrong"}' 2>/dev/null)
lcode=$(printf '%s' "$resp" | tail -1)
case "$lcode" in
  401) case "$resp" in
         *'"ok":false'*) ok "sign-in path runs (401 as expected)" ;;
         *) fail "sign-in returned 401 but not the app's JSON; something else is answering" ;;
       esac ;;
  429) ok "sign-in path alive (throttled)" ;;
  *)   fail "sign-in answered HTTP ${lcode:-no-response} where 401 was expected, so auth is broken even though the service is listening" ;;
esac

# --- 5. TLS certificate ------------------------------------------------------------------------
host_for_tls=${RESOLVE_ARGS[1]:-}
tls_target=${host_for_tls#*:443:}
[ -z "$tls_target" ] && tls_target=$API_HOST
end=$(echo | timeout 20 openssl s_client -connect "$tls_target:443" -servername "$API_HOST" 2>/dev/null \
      | openssl x509 -noout -enddate 2>/dev/null | cut -d= -f2)
if [ -z "$end" ]; then
  fail "could not read the TLS certificate for $API_HOST"
else
  left=$(( ( $(date -d "$end" +%s) - $(date +%s) ) / 86400 ))
  if [ "$left" -gt "$CERT_MIN_DAYS" ]; then ok "TLS certificate valid for $left more days"
  else fail "TLS certificate for $API_HOST expires in $left day(s), on $end"; fi
fi

echo
if [ ${#fails[@]} -eq 0 ]; then
  echo "ALL CHECKS PASSED"
  exit 0
fi
echo "${#fails[@]} CHECK(S) FAILED:"
printf '  - %s\n' "${fails[@]}"
echo
echo "This probe runs outside the house, so a failure here is what a real visitor sees."
echo "First things to look at on HPCF:"
echo "  systemctl --user status pubverse-keeper.service"
echo "  tail -20 /home/joneill/pubverse_platform/pubverse_backend/logs/keeper.log"
echo "  tailscale funnel status"
exit 1
