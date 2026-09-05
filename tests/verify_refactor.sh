#!/bin/sh
# The refactor gate: everything that must stay true while revv is cleaned up.
#
#   sh tests/verify_refactor.sh            # check against the last commit
#   BASE=<git-ref> sh tests/verify_refactor.sh
#
# Regenerates each behaviour snapshot from BASE and from the working tree and
# requires the two to be byte-identical, then runs the planner suite. Exits
# non-zero on the first failure with the diff on stdout.
set -e

REPO=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
BASE="${BASE:-HEAD}"
OUT="${OUT:-/tmp/revv-verify}"

rm -rf "$OUT"
mkdir -p "$OUT/base"

echo "== extracting $BASE =="
git -C "$REPO" show "$BASE:revv.py"     > "$OUT/base/revv.py"
git -C "$REPO" show "$BASE:install.sh"  > "$OUT/base/install.sh"

run_pair() {           # run_pair <label> <cmd-against-cwd>
    label="$1"
    shift
    echo "== $label =="
    ( cd "$OUT/base" && "$@" ) > "$OUT/$label.before" 2>&1
    ( cd "$REPO"     && "$@" ) > "$OUT/$label.after"  2>&1
    if diff "$OUT/$label.before" "$OUT/$label.after" > "$OUT/$label.diff"; then
        printf '   identical (%s lines)\n' "$(wc -l < "$OUT/$label.after" | tr -d ' ')"
    else
        echo "   DIFFERS:"
        head -40 "$OUT/$label.diff"
        exit 1
    fi
}

# The snapshot harnesses import/read whatever revv.py or install.sh is in the
# directory they run from, so copy them next to the extracted base copies.
cp "$REPO/tests/snapshot_argv.py" "$REPO/tests/snapshot_cli.py" \
   "$REPO/tests/snapshot_install.sh" "$OUT/base/"
mkdir -p "$OUT/base/tests"
cp "$REPO/tests/snapshot_argv.py" "$REPO/tests/snapshot_cli.py" \
   "$REPO/tests/snapshot_install.sh" "$OUT/base/tests/"
ln -s "$REPO/tests/fixtures" "$OUT/base/tests/fixtures" 2>/dev/null || true

run_pair argv    python3 tests/snapshot_argv.py
run_pair cli     python3 tests/snapshot_cli.py
run_pair install sh tests/snapshot_install.sh

echo "== planner suite (144 assertions, unedited) =="
( cd "$REPO" && python3 tests/test_planner.py ) | tail -1

echo
echo "ALL VERIFIED: behaviour is identical to $BASE"
