#!/bin/bash
# Pulls every model listed in models.txt and reports what succeeded.
# Requires the Ollama server running: ollama serve

LOG="pull_models.log"
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

if ! ollama list >/dev/null 2>&1; then
  log "ERROR: Ollama server not reachable. Start it with: ollama serve"
  exit 1
fi

ok=(); failed=()
log "=== Pull run started ==="

while IFS= read -r line || [[ -n "$line" ]]; do
  m="${line%%#*}"                        # drop comments, including inline ones
  m="$(echo "$m" | tr -d '[:space:]')"   # drop spaces, tabs, stray \r
  [[ -z "$m" ]] && continue

  log "Pulling $m"
  if ollama pull "$m" < /dev/null; then
    log "OK      $m"; ok+=("$m")
  else
    log "FAILED  $m"; failed+=("$m")
  fi
done < models.txt

log "=== Summary: ${#ok[@]} ok, ${#failed[@]} failed ==="
for m in "${ok[@]}";     do log "  ok      $m"; done
for m in "${failed[@]}"; do log "  FAILED  $m"; done
log "Installed models:"
ollama list | tee -a "$LOG"

[[ ${#failed[@]} -eq 0 ]]   # exit code 1 if anything failed