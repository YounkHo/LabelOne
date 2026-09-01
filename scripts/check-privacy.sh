#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

PATTERN='(/Users/[^/[:space:]]+|/home/[^/[:space:]]+|[A-Za-z]:\\Users\\[^\\[:space:]]+|BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|(^|[^A-Za-z0-9])(sk|ghp|github_pat)_[A-Za-z0-9_-]{20,})'
CREDENTIAL_PATTERN="(api[_-]?key|token|password)[[:space:]]*[:=][[:space:]]*[\"'][^\$<{][^\"']{7,}"

matches="$({ git ls-files -z --cached --others --exclude-standard | while IFS= read -r -d '' file; do
  [[ "$file" == "scripts/check-privacy.sh" ]] && continue
  grep -nIE -e "$PATTERN" -e "$CREDENTIAL_PATTERN" "$file" 2>/dev/null | sed "s#^#$file:#" || true
done; } || true)"

if [[ -n "$matches" ]]; then
  printf 'Potential private data found in files eligible for commit:\n%s\n' "$matches" >&2
  exit 1
fi

printf 'Repository privacy scan passed.\n'

if [[ "${1:-}" == "--history" ]]; then
  history_emails="$(git log --format='%ae' | sort -u | grep -Ev '(^$|@users\.noreply\.github\.com$|^noreply@)' || true)"
  if [[ -n "$history_emails" ]]; then
    printf 'Git history contains non-anonymous author emails:\n%s\n' "$history_emails" >&2
    printf 'History rewriting affects commit IDs and collaborators; review it separately before publishing.\n' >&2
    exit 2
  fi
  printf 'Git-history author email scan passed.\n'
fi
