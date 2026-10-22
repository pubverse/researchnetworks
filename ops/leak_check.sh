#!/usr/bin/env bash
# Fail when a tracked file would publish something about the machine behind the site.
#
# This repository is public (GitHub Pages). Generated files have leaked local paths into it before
# (a map's "generated_from" block carried scratchpad paths with the account name in them), so every
# push runs this check: in CI (.github/workflows/hygiene.yml) and, on the build machine, from the
# pre-push hook. It scans tracked text files only and prints file:line for each hit.
#
#   bash ops/leak_check.sh          # exit 0 clean, 1 on a hit
#
# Allowed on purpose: js/config.js names the public API host, which every visitor's browser calls, and the off-box
# uptime probe (ops/probe.sh, .github/workflows/uptime.yml) checks that same host; the probe also names the tailnet
# address RANGE 100.64.0.0/10 in a comment.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

patterns=(
  '/home/[a-z][a-z0-9_-]*/'              # absolute home paths
  '/tmp/claude'                          # tool scratchpads
  '/mnt/fast/|/storage/[a-z]'            # data volumes
  '\b100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]+\.[0-9]+\b'   # tailnet (CGNAT) addresses
  'trycloudflare\.com'                   # throwaway tunnel hosts
  '-----BEGIN [A-Z ]*PRIVATE KEY-----'
  '\b(ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_-]{30,}|tskey-[A-Za-z0-9-]{10,}|xox[baprs]-[A-Za-z0-9-]{10,})'
  'ts\.net'                              # tailnet hostnames (allowed only in js/config.js)
)
hits=0
for p in "${patterns[@]}"; do
  out=$(git grep -nIE -e "$p" -- . ':!ops/leak_check.sh' 2>/dev/null || true)
  if [ "$p" = 'ts\.net' ]; then out=$(printf '%s\n' "$out" | grep -vE '^(js/config\.js|ops/probe\.sh|\.github/workflows/uptime\.yml):' || true); fi
  out=$(printf '%s\n' "$out" | grep -v '100\.64\.0\.0/10' | grep -v '^$' || true)
  if [ -n "$out" ]; then
    printf 'LEAK (%s):\n%s\n' "$p" "$out" | cut -c1-240
    hits=$((hits + 1))
  fi
done
if [ "$hits" -gt 0 ]; then
  echo "leak_check: $hits pattern(s) matched; remove them before publishing." >&2
  exit 1
fi
echo "leak_check: clean"
