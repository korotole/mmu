#!/usr/bin/env bash
# stavby-rag installer: uv + ollama + `stavby` CLI (as a uv tool) + models.  macOS and Linux.
#
#   ./install.sh                     from a checkout: installs this repo; elsewhere: installs from PyPI/git
#   ./install.sh --claude            also store an ANTHROPIC_API_KEY in $STAVBY_HOME/.env (skips the local LLM pull)
#   ./install.sh --no-models         skip all `ollama pull`s
#   ./install.sh --with-index        run `stavby crawl` + `stavby index` at the end
#   ./install.sh --tune-ollama       macOS: flash attention + q8_0 KV cache + 30m keep-alive for the ollama service
#   ./install.sh --home DIR          data + .env location (default ~/.stavby)
#   ./install.sh --with-local-llm    pull qwen3:14b even with --claude;  --extras web,st  choose package extras
#   ./install.sh --from-source       install from the checkout this script lives in (auto when run from a checkout)
#
# Idempotent: every step checks before it acts, so re-running is safe.
set -euo pipefail

# ---------------------------------------------------------------- options
STAVBY_HOME="${STAVBY_HOME:-$HOME/.stavby}"
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"
PACKAGE_SPEC="stavby-rag"                 # what `uv tool install` gets when not run from a checkout
EXTRAS="web"                              # `stavby serve` needs fastapi + uvicorn
PULL_MODELS=1
WITH_INDEX=0
WITH_CLAUDE=0
WITH_LOCAL_LLM=""                         # "" = decide from --claude; 1 = force the qwen3:14b pull
TUNE_OLLAMA=0
FROM_SOURCE=""                            # "" = auto-detect a checkout next to this script

EMBED_MODEL="hf.co/Qwen/Qwen3-Embedding-8B-GGUF:Q8_0"   # 8 GB
RERANK_MODEL="dengcao/Qwen3-Reranker-8B:Q8_0"           # 8.7 GB
LLM_MODEL="qwen3:14b"                                   # 9.3 GB

usage() { # $0 is unreadable when piped (curl | bash), so fall back to the essential lines
  if [[ -r "$0" ]]; then awk 'NR>1 && !/^#/ {exit} NR>1 {sub(/^# ?/, ""); print}' "$0"
  else sed -n '2,11p' <<'EOF' | sed 's/^# \{0,1\}//'
# stavby-rag installer: uv + ollama + `stavby` CLI (as a uv tool) + models.  macOS and Linux.
#
#   curl -fsSL https://raw.githubusercontent.com/korotole/mmu/main/install.sh | bash -s -- --with-index
#   ./install.sh                     from a checkout: installs this repo; elsewhere: installs from PyPI/git
#   ./install.sh --claude            also store an ANTHROPIC_API_KEY in $STAVBY_HOME/.env (skips the local LLM pull)
#   ./install.sh --no-models         skip all `ollama pull`s
#   ./install.sh --with-index        run `stavby crawl` + `stavby index` at the end
#   ./install.sh --tune-ollama       macOS: flash attention + q8_0 KV cache + 30m keep-alive for the ollama service
#   ./install.sh --home DIR          data + .env location (default ~/.stavby)
EOF
  fi
  exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-models)       PULL_MODELS=0 ;;
    --with-index)      WITH_INDEX=1 ;;
    --with-local-llm)  WITH_LOCAL_LLM=1 ;;
    --claude)          WITH_CLAUDE=1 ;;
    --tune-ollama)     TUNE_OLLAMA=1 ;;
    --from-source)     FROM_SOURCE=1 ;;
    --home)            STAVBY_HOME="$2"; shift ;;
    --home=*)          STAVBY_HOME="${1#*=}" ;;
    --extras)          EXTRAS="$2"; shift ;;
    --extras=*)        EXTRAS="${1#*=}" ;;
    -h|--help)         usage 0 ;;
    *) echo "unknown option: $1" >&2; usage 2 ;;
  esac
  shift
done
STAVBY_HOME="${STAVBY_HOME/#\~/$HOME}"

# ---------------------------------------------------------------- helpers
if [[ -t 1 ]]; then
  BOLD=$'\e[1m'; GREEN=$'\e[32m'; YELLOW=$'\e[33m'; RED=$'\e[31m'; DIM=$'\e[2m'; RESET=$'\e[0m'
else
  BOLD=""; GREEN=""; YELLOW=""; RED=""; DIM=""; RESET=""
fi
step() { printf '\n%s==> %s%s\n' "$BOLD" "$*" "$RESET"; }
ok()   { printf '%s  ok%s  %s\n' "$GREEN" "$RESET" "$*"; }
warn() { printf '%swarn%s  %s\n' "$YELLOW" "$RESET" "$*"; }
die()  { printf '%sfail%s  %s\n' "$RED" "$RESET" "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# ---------------------------------------------------------------- 0. platform
OS="$(uname -s)"; ARCH="$(uname -m)"
case "$OS" in
  Darwin) OS_NAME=macos ;;
  Linux)  OS_NAME=linux ;;
  *) die "unsupported OS: $OS (macOS and Linux only)" ;;
esac
step "stavby-rag installer  ($OS_NAME/$ARCH, home: $STAVBY_HOME)"

if [[ -r "$0" && -d "$(dirname "${BASH_SOURCE[0]:-}" 2>/dev/null || echo /nonexistent)" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
else
  SCRIPT_DIR=""   # piped (curl | bash): no file on disk
fi
if [[ -n "$SCRIPT_DIR" && -f "$SCRIPT_DIR/pyproject.toml" ]] && grep -q '^name = "stavby-rag"' "$SCRIPT_DIR/pyproject.toml"; then
  FROM_SOURCE=1
fi
[[ "$FROM_SOURCE" == 1 ]] && PACKAGE_SPEC="$SCRIPT_DIR"
# Not run from a checkout and not on PyPI yet: install straight from the GitHub repo.
if [[ "$FROM_SOURCE" != 1 && "$PACKAGE_SPEC" == "stavby-rag" ]]; then
  PACKAGE_SPEC="git+https://github.com/korotole/mmu.git"
fi

# ---------------------------------------------------------------- 1. uv
step "uv (Python package manager)"
if have uv; then
  ok "uv $(uv --version | awk '{print $2}') present"
else
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  have uv || die "uv installed but not on PATH; open a new shell and re-run"
  ok "uv installed"
fi
# uv tool binaries land here; make sure the current shell sees them.
UV_BIN="$(uv tool dir --bin 2>/dev/null || echo "$HOME/.local/bin")"
case ":$PATH:" in *":$UV_BIN:"*) ;; *) export PATH="$UV_BIN:$PATH" ;; esac

# ---------------------------------------------------------------- 2. ollama
step "ollama (local models)"
if have ollama; then
  ok "ollama $(ollama --version 2>/dev/null | awk '{print $NF}') present"
elif [[ "$OS_NAME" == macos ]] && have brew; then
  brew install ollama
  ok "ollama installed via brew"
elif [[ "$OS_NAME" == linux ]]; then
  if curl -fsSL https://ollama.com/install.sh | sh; then
    ok "ollama installed"
  else
    warn "ollama download failed (proxy/cert?); install it from https://ollama.com/download, then re-run"
  fi
else
  die "ollama not found; install from https://ollama.com/download and re-run"
fi

ollama_up() { curl -fsS --max-time 2 "$OLLAMA_HOST/api/tags" >/dev/null 2>&1; }
if ollama_up; then
  ok "ollama reachable at $OLLAMA_HOST"
else
  if [[ "$OS_NAME" == macos ]] && have brew && brew list ollama >/dev/null 2>&1; then
    brew services start ollama >/dev/null
  elif [[ "$OS_NAME" == linux ]] && have systemctl && systemctl list-unit-files ollama.service >/dev/null 2>&1; then
    sudo systemctl start ollama
  elif have ollama; then
    mkdir -p "$STAVBY_HOME"
    nohup ollama serve >"$STAVBY_HOME/ollama.log" 2>&1 &
    disown || true
  else
    warn "ollama absent; skipping service start (models step will be skipped too)"
  fi
  for _ in $(seq 1 30); do ollama_up && break; sleep 1; done
  ollama_up && ok "ollama started" || warn "ollama not reachable at $OLLAMA_HOST; start it manually (ollama serve)"
fi

# ---------------------------------------------------------------- 3. ollama tuning (macOS)
if [[ "$TUNE_OLLAMA" == 1 ]]; then
  step "ollama memory tuning"
  if [[ "$OS_NAME" != macos ]]; then
    warn "--tune-ollama only handles macOS launchctl; on Linux set the same vars in the ollama systemd unit"
  else
    # Flash attention + q8_0 KV cache halve what the contexts cost (matters most for qwen3:14b at
    # 16k ctx); keep-alive -1 keeps models loaded forever (the web server wants that). Measured on
    # an M4 Pro 24 GB: neither setting nor OLLAMA_NUM_PARALLEL speeds up the reranker itself, and
    # macOS caps Metal's wired memory at ~2/3 of RAM (~16 GB). On 24 GB machines raising the cap
    # is NOT enough for embedder + reranker co-residence (2x Q8_0 8B + macOS > 24 GB, and Ollama
    # also evicts on free system pages); on 32 GB+ machines it helps:
    #   sudo sysctl iogpu.wired_limit_mb=18432     # until reboot
    #   echo iogpu.wired_limit_mb=18432 | sudo tee /etc/sysctl.conf   # persistent
    # launchctl setenv reaches the brew service and GUI app.
    launchctl setenv OLLAMA_FLASH_ATTENTION 1
    launchctl setenv OLLAMA_KV_CACHE_TYPE q8_0
    launchctl setenv OLLAMA_KEEP_ALIVE -1
    ok "launchctl setenv OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_KEEP_ALIVE=-1"
    if [[ "$(sysctl -n iogpu.wired_limit_mb 2>/dev/null || echo 0)" -lt 18432 ]] && \
       [[ "$(sysctl -n hw.memsize 2>/dev/null || echo 0)" -ge 32212254720 ]]; then
      warn "Metal wired limit is ~2/3 of RAM; for warm embedder+reranker run:"
      printf '%s      sudo sysctl iogpu.wired_limit_mb=18432%s\n' "$DIM" "$RESET"
    fi
    if have brew && brew services list 2>/dev/null | grep -q '^ollama.*started'; then
      brew services restart ollama >/dev/null
      for _ in $(seq 1 30); do ollama_up && break; sleep 1; done
      ok "ollama brew service restarted"
    else
      warn "restart ollama (quit the menu-bar app or re-run 'ollama serve') for the new settings to apply"
    fi
    printf '%s      persist across reboots by adding the three launchctl setenv lines to a LaunchAgent or ~/.zprofile%s\n' "$DIM" "$RESET"
  fi
fi

# ---------------------------------------------------------------- 4. the package
step "stavby CLI  (uv tool install ${PACKAGE_SPEC}[${EXTRAS}])"
uv tool install --force --python 3.12 "${PACKAGE_SPEC}[${EXTRAS}]"
have stavby || die "'stavby' not on PATH after install; run: uv tool update-shell"
ok "$(stavby --version 2>/dev/null || echo 'stavby installed') -> $(command -v stavby)"

# ---------------------------------------------------------------- 5. home + .env
step "home directory"
mkdir -p "$STAVBY_HOME/data"
if [[ ! -f "$STAVBY_HOME/.env" ]]; then
  if [[ -f "$SCRIPT_DIR/.env.example" ]]; then
    cp "$SCRIPT_DIR/.env.example" "$STAVBY_HOME/.env"
  else
    cat >"$STAVBY_HOME/.env" <<'ENV'
# stavby-rag configuration (KEY=VALUE, loaded on start; empty values keep defaults).
ANTHROPIC_API_KEY=
STAVBY_ANTHROPIC_MODEL=claude-opus-5
OLLAMA_HOST=http://localhost:11434
STAVBY_OLLAMA_MODEL=qwen3:14b
STAVBY_EMBED_BACKEND=auto
STAVBY_RERANK_BACKEND=auto
ENV
  fi
  ok "created $STAVBY_HOME/.env"
else
  ok "$STAVBY_HOME/.env exists (kept)"
fi
if [[ "$OLLAMA_HOST" != "http://localhost:11434" ]] && ! grep -q "^OLLAMA_HOST=$OLLAMA_HOST" "$STAVBY_HOME/.env"; then
  printf '\nOLLAMA_HOST=%s\n' "$OLLAMA_HOST" >>"$STAVBY_HOME/.env"
fi

if [[ "$WITH_CLAUDE" == 1 ]]; then
  KEY="${ANTHROPIC_API_KEY:-}"
  if [[ -z "$KEY" ]]; then
    read -r -s -p "ANTHROPIC_API_KEY (input hidden): " KEY; echo
  fi
  [[ -n "$KEY" ]] || die "--claude given but no key entered"
  if grep -q '^ANTHROPIC_API_KEY=' "$STAVBY_HOME/.env"; then
    # in-place edit that works with both BSD and GNU sed
    tmp="$(mktemp)"; sed "s|^ANTHROPIC_API_KEY=.*|ANTHROPIC_API_KEY=$KEY|" "$STAVBY_HOME/.env" >"$tmp" && mv "$tmp" "$STAVBY_HOME/.env"
  else
    printf 'ANTHROPIC_API_KEY=%s\n' "$KEY" >>"$STAVBY_HOME/.env"
  fi
  chmod 600 "$STAVBY_HOME/.env"
  ok "ANTHROPIC_API_KEY stored in $STAVBY_HOME/.env (mode 600)"
fi

# ---------------------------------------------------------------- 6. models
if [[ "$PULL_MODELS" == 1 ]]; then
  step "models (embedding 8 GB, reranker 8.7 GB, LLM 9.3 GB; resumable, slow on throttled links)"
  if ! ollama_up; then
    warn "ollama not reachable; skipping model pulls. Later: ollama pull $EMBED_MODEL"
  else
    MODELS=("$EMBED_MODEL" "$RERANK_MODEL")
    if [[ "$WITH_CLAUDE" == 1 && "$WITH_LOCAL_LLM" != 1 ]]; then
      warn "skipping $LLM_MODEL (Claude is the answer backend; add --with-local-llm to pull it anyway)"
    else
      MODELS+=("$LLM_MODEL")
    fi
    INSTALLED="$(ollama list 2>/dev/null | awk 'NR>1 {print $1}')"
    for m in "${MODELS[@]}"; do
      if grep -qxF "$m" <<<"$INSTALLED"; then
        ok "$m already pulled"
      else
        ollama pull "$m"
        ok "$m pulled"
      fi
    done
  fi
fi

# ---------------------------------------------------------------- 7. optional crawl + index
if [[ "$WITH_INDEX" == 1 ]]; then
  step "crawl + index  (crawl ~minutes; index ~25 min with the 8B embedder on Apple silicon)"
  STAVBY_HOME="$STAVBY_HOME" stavby crawl
  STAVBY_HOME="$STAVBY_HOME" stavby index
  ok "index built in $STAVBY_HOME/data/index"
fi

# ---------------------------------------------------------------- 8. summary
step "done"
if [[ "$STAVBY_HOME" != "$HOME/.stavby" ]]; then
  printf '  non-default home: add to your shell profile ->  %sexport STAVBY_HOME=%q%s\n' "$BOLD" "$STAVBY_HOME" "$RESET"
fi
case ":$PATH:" in *":$UV_BIN:"*) ;; *) printf '  add %s to PATH (uv tool update-shell)\n' "$UV_BIN" ;; esac
cat <<SUMMARY

  config:  $STAVBY_HOME/.env
  data:    $STAVBY_HOME/data

  stavby doctor                      # what is configured / reachable
  stavby crawl && stavby index       # mirror the textbook, build the hybrid index
  stavby ask "Co je stavebně technologická studie?"
  stavby serve                       # web UI + API (fastapi/uvicorn)

SUMMARY
