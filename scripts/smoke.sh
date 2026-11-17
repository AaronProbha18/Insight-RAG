#!/bin/sh
# Smoke test for the running stack: health, one upload, one chat.
# Checks the behaviour contract only; answer quality is measured by the evals.
# Usage: sh scripts/smoke.sh [--down]
set -u
cd "$(dirname "$0")/.."
BASE=http://localhost:8000
SAMPLE=evals/connectors/pdf/corpus/returns-policy.pdf  # synthetic corpus, never real data
QUESTION="What is the general return window?"
failed=0

pass() { echo "PASS  $1"; }
fail() { echo "FAIL  $*"; failed=1; }
finish() {
  [ "${1:-}" = "--down" ] && docker compose down >/dev/null 2>&1
  [ "$failed" = 0 ] && echo "smoke: all steps passed" || echo "smoke: FAILED"
  exit "$failed"
}

docker compose up -d --build >/dev/null 2>&1 || { fail "docker compose up"; finish "${1:-}"; }

# 1. health
i=0
until curl -fsS "$BASE/health" >/dev/null 2>&1; do
  i=$((i + 1))
  if [ "$i" -ge 45 ]; then
    fail "health: app did not become healthy within 90s"
    docker compose logs --tail 20 app
    finish "${1:-}"
  fi
  sleep 2
done
pass "health"

# 2. upload
status=$(curl -s -o /tmp/smoke_upload.json -w '%{http_code}' -F "file=@$SAMPLE" "$BASE/documents")
case "$status" in
  201 | 200) pass "upload (HTTP $status) $(cat /tmp/smoke_upload.json)" ;;
  *)
    fail "upload (HTTP $status): is a model configured? Check .env (INSIGHT_RAG_EMBEDDINGS," \
      "INSIGHT_RAG_EMBEDDING_DIM; in Docker INSIGHT_RAG_MODEL_BASE_URL=http://host.docker.internal:11434)." \
      "A dimension error means: docker compose down -v, then retry."
    finish "${1:-}"
    ;;
esac

# 3. chat
curl -sN -X POST "$BASE/chat" -H 'Content-Type: application/json' \
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$QUESTION\"}]}" >/tmp/smoke_chat.txt
if grep -q '^event: error' /tmp/smoke_chat.txt; then
  fail "chat: error event: $(grep -A1 '^event: error' /tmp/smoke_chat.txt | tail -1)"
elif ! grep -q '^event: done' /tmp/smoke_chat.txt; then
  fail "chat: stream did not end with a done event"
elif grep -q '^event: citations' /tmp/smoke_chat.txt; then
  answer=$(grep -A1 '^event: delta' /tmp/smoke_chat.txt | grep '^data:' | sed 's/^data: {"text": "//; s/"}$//' | tr -d '\n')
  pass "chat answered: $answer"
  echo "      sources: $(grep -A1 '^event: citations' /tmp/smoke_chat.txt | tail -1 | grep -o '"title": "[^"]*", "locator": "[^"]*"' | tr '\n' ' ')"
elif grep -q '^event: abstain' /tmp/smoke_chat.txt; then
  pass "chat abstained (valid behaviour; check the evals if this looks wrong)"
else
  fail "chat: neither citations nor abstain in the stream"
fi

finish "${1:-}"
