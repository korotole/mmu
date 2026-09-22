#!/usr/bin/env bash
# Remove the `stavby` CLI installed by install.sh.  `--purge` also deletes $STAVBY_HOME (data, index, .env).
set -euo pipefail

STAVBY_HOME="${STAVBY_HOME:-$HOME/.stavby}"
PURGE=0
YES=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --purge)  PURGE=1 ;;
    --yes|-y) YES=1 ;;
    --home)   STAVBY_HOME="$2"; shift ;;
    --home=*) STAVBY_HOME="${1#*=}" ;;
    -h|--help) sed -n '2p' "$0" | sed 's/^# //'; echo "usage: $0 [--purge] [--yes] [--home DIR]"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done
STAVBY_HOME="${STAVBY_HOME/#\~/$HOME}"

if command -v uv >/dev/null 2>&1 && uv tool list 2>/dev/null | grep -q '^stavby-rag '; then
  uv tool uninstall stavby-rag
  echo "removed uv tool stavby-rag"
else
  echo "stavby-rag is not installed as a uv tool (nothing to remove)"
fi

if [[ "$PURGE" == 1 ]]; then
  if [[ ! -d "$STAVBY_HOME" ]]; then
    echo "$STAVBY_HOME does not exist"
    exit 0
  fi
  echo "about to delete $STAVBY_HOME ($(du -sh "$STAVBY_HOME" 2>/dev/null | cut -f1)) including .env and the index"
  if [[ "$YES" != 1 ]]; then
    read -r -p "type 'yes' to confirm: " answer
    [[ "$answer" == yes ]] || { echo "aborted"; exit 1; }
  fi
  rm -rf "$STAVBY_HOME"
  echo "removed $STAVBY_HOME"
else
  echo "kept $STAVBY_HOME (use --purge to delete data + .env); ollama models stay in ~/.ollama (ollama rm <model>)"
fi
