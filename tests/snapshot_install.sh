#!/bin/sh
# Behaviour snapshot for install.sh.
#
#   sh tests/snapshot_install.sh > before.txt
#   ...refactor install.sh...
#   sh tests/snapshot_install.sh > after.txt
#   diff before.txt after.txt      # must be empty
#
# Sources install.sh with the Main section cut off, then exercises every
# helper that can run without a network, a GPU, or a compiler. Nothing here
# touches $HOME or anything outside /tmp.

REPO=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
HOME_DIR=/tmp/revv-installsnap-home
WORK=/tmp/revv-installsnap-work

rm -rf "$HOME_DIR" "$WORK"
mkdir -p "$HOME_DIR" "$WORK"

scrub() {
    sed -e "s#$HOME_DIR#<home>#g" \
        -e "s#$WORK#<work>#g" \
        -e "s#$REPO#<repo>#g" \
        -e 's#\.tmp\.[0-9][0-9]*#.tmp.<pid>#g' \
        -e 's#"built_at": "[^"]*"#"built_at": "<scrubbed>"#'
}

banner() { printf '\n===== SECTION: %s =====\n' "$1"; }
case_() { printf '\n--- %s ---\n' "$1"; }

# ---------------------------------------------------------------------------
banner "help and argument parsing (real script, subprocess)"

run_case() {         # run_case <label> <cmd...>: print output (scrubbed) then rc
    label="$1"
    shift
    case_ "$label"
    out=$("$@" 2>&1)
    rc=$?
    printf '%s\n' "$out" | scrub
    printf 'rc=%s\n' "$rc"
}

run_case "install.sh --help" sh "$REPO/install.sh" --help
run_case "install.sh -h" sh "$REPO/install.sh" -h
run_case "install.sh --nonsense (usage to stderr, then fail)" \
    sh "$REPO/install.sh" --nonsense
run_case "sh -n install.sh (syntax)" sh -n "$REPO/install.sh"

case_ "shellcheck"
if command -v shellcheck >/dev/null 2>&1; then
    out=$(shellcheck -s sh "$REPO/install.sh" 2>&1)
    rc=$?
    # Line numbers move under a comment-only edit, so normalise them: what
    # this snapshot cares about is which findings exist, not where.
    printf '%s\n' "$out" | scrub | sed 's/^\(In .* line \)[0-9][0-9]*:/\1<n>:/'
    printf 'rc=%s\n' "$rc"
else
    echo "shellcheck: not installed"
fi

# ---------------------------------------------------------------------------
# Source the helper half of install.sh: everything above the Main section.
TRUNC="$WORK/install_helpers.sh"
awk '/^echo "revv installer"$/ { exit } { print }' "$REPO/install.sh" > "$TRUNC"

REVV_HOME="$HOME_DIR"
export REVV_HOME
# shellcheck disable=SC1090
. "$TRUNC"
set +e

banner "sourced helper preamble"
# Deliberately no line counts here: they change when install.sh is refactored
# without its behaviour changing, which is exactly what this snapshot must
# not report as a difference.
printf 'PINNED_COMMIT=%s\nPINNED_BUILD=%s\nREVV_VERSION=%s\n' \
    "$PINNED_COMMIT" "$PINNED_BUILD" "$REVV_VERSION"
printf 'LLAMA_REPO_URL=%s\n' "$LLAMA_REPO_URL"
printf 'PREBUILT_URL=%s\n' "$PREBUILT_URL"
printf 'PREBUILT_SHA256=%s\n' "$PREBUILT_SHA256"
printf 'PREBUILT_CUDA_ARCH=%s\n' "$PREBUILT_CUDA_ARCH"
printf 'BIN_DIR=%s\nSRC_DIR=%s\nLLAMA_SRC=%s\nBUILD_MANIFEST=%s\n' \
    "$BIN_DIR" "$SRC_DIR" "$LLAMA_SRC" "$BUILD_MANIFEST" | scrub
printf 'PATCHES_DIR=%s\nREVV_PY=%s\n' "$PATCHES_DIR" "$REVV_PY" | scrub

# ---------------------------------------------------------------------------
banner "usage() function"
usage 2>&1 | scrub

# ---------------------------------------------------------------------------
banner "is_int"
for v in "" 0 7 007 -1 1.5 abc 12a " 3" 99999999 "1 2"; do
    is_int "$v"
    printf 'is_int(%s) rc=%s\n' "[$v]" "$?"
done

# ---------------------------------------------------------------------------
banner "max_host_gcc_for_cuda"
for pair in "11 0" "11 8" "12 0" "12 3" "12 4" "12 5" "12 6" "12 9" \
            "13 0" "14 2" "19 0" "20 0" "9 0" "x y" " "; do
    # shellcheck disable=SC2086
    set -- $pair
    printf 'cuda %s.%s -> [%s]\n' "${1:-}" "${2:-}" "$(max_host_gcc_for_cuda "${1:-}" "${2:-}" 2>&1)"
done

# ---------------------------------------------------------------------------
banner "find_compatible_hostcxx (machine-dependent: reflects installed g++-N)"
for want in 11 12 13 14 15; do
    out=$(find_compatible_hostcxx "$want" 2>&1)
    rc=$?
    printf 'want_max=%s -> [%s] rc=%s\n' "$want" "$out" "$rc"
done

# ---------------------------------------------------------------------------
banner "is_wsl2 (via REVV_PROC_VERSION_FILE)"
printf 'Linux version 5.15.0-microsoft-standard-WSL2\n' > "$WORK/pv_wsl"
printf 'Linux version 6.1.0-generic\n' > "$WORK/pv_native"
printf 'LINUX VERSION 5.15 MICROSOFT\n' > "$WORK/pv_caps"
: > "$WORK/pv_empty"
for f in pv_wsl pv_native pv_caps pv_empty nope; do
    REVV_PROC_VERSION_FILE="$WORK/$f" is_wsl2
    printf '%s rc=%s\n' "$f" "$?"
done

# ---------------------------------------------------------------------------
banner "parse_glibc / nproc_portable (machine-dependent)"
parse_glibc
printf 'GLIBC_VER=[%s] GLIBC_MAJOR=[%s] GLIBC_MINOR=[%s]\n' \
    "$GLIBC_VER" "$GLIBC_MAJOR" "$GLIBC_MINOR"
printf 'nproc_portable=[%s]\n' "$(nproc_portable)"

# ---------------------------------------------------------------------------
banner "sha256_of"
: > "$WORK/h_empty"
printf 'revv\n' > "$WORK/h_revv"
awk 'BEGIN { while (i++ < 1000) printf "a" }' > "$WORK/h_a1000"
for f in h_empty h_revv h_a1000; do
    out=$(sha256_of "$WORK/$f")
    rc=$?
    printf '%s -> %s rc=%s\n' "$f" "$out" "$rc"
done
out=$(sha256_of "$WORK/does-not-exist")
rc=$?
printf 'missing -> [%s] rc=%s\n' "$out" "$rc"

# ---------------------------------------------------------------------------
banner "detect_cuda_archs"
for v in 86 "75;86" "native"; do
    CUDA_ARCH_REASON=""
    out=$(CUDAARCHS="$v" detect_cuda_archs 2>&1)
    printf 'CUDAARCHS=%s -> [%s]\n' "$v" "$out"
done
case_ "CUDAARCHS unset"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "SKIPPED (nvidia-smi present on this machine)"
else
    CUDA_ARCH_REASON=""
    out=$(unset CUDAARCHS; detect_cuda_archs 2>&1)
    # CUDA_ARCH_REASON is set in a subshell above, so re-run in-process for it
    unset CUDAARCHS
    detect_cuda_archs >/dev/null 2>&1
    printf 'value=[%s] reason=[%s]\n' "$out" "$CUDA_ARCH_REASON"
fi

# ---------------------------------------------------------------------------
banner "check_prebuilt_arch"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "SKIPPED (nvidia-smi present on this machine)"
else
    PREBUILT_FAIL_REASON=""
    out=$(check_prebuilt_arch 2>&1)
    rc=$?
    printf '%s\n' "$out" | scrub
    printf 'rc=%s PREBUILT_FAIL_REASON=[%s]\n' "$rc" "$PREBUILT_FAIL_REASON"
fi

# ---------------------------------------------------------------------------
# The compute-capability parsing in detect_cuda_archs and check_prebuilt_arch
# only runs when nvidia-smi exists. Stub it so both are actually exercised.
banner "compute-cap parsing against a stubbed nvidia-smi"

STUB_DIR="$WORK/stubbin"
mkdir -p "$STUB_DIR"
make_smi() {
    cat > "$STUB_DIR/nvidia-smi" <<STUB
#!/bin/sh
printf '%s' "\$SMI_OUT"
STUB
    chmod +x "$STUB_DIR/nvidia-smi"
}
make_smi

SAVED_PATH="$PATH"
for smi_case in \
    "one-ampere:8.6
" \
    "one-turing:7.5
" \
    "two-same:8.6
8.6
" \
    "two-different:8.6
8.9
" \
    "unsupported-blackwell:12.0
" \
    "mixed-supported-and-not:8.6
12.0
" \
    "not-available:N/A
" \
    "garbage:banana
" \
    "empty:" \
    "blank-line:
" \
    "with-spaces-and-cr: 8.6
"; do
    label=${smi_case%%:*}
    SMI_OUT=${smi_case#*:}
    export SMI_OUT
    case_ "$label"
    printf 'nvidia-smi prints: %s\n' "$(printf '%s' "$SMI_OUT" | tr '\n' '|')"

    PATH="$STUB_DIR:$SAVED_PATH"
    CUDA_ARCH_REASON=""
    archs=$(detect_cuda_archs 2>&1)
    printf 'detect_cuda_archs -> [%s]\n' "$archs"
    detect_cuda_archs >/dev/null 2>&1
    printf '  CUDA_ARCH_REASON=[%s]\n' "$CUDA_ARCH_REASON"

    # NB: a var assignment prefixing a *function* call persists in POSIX sh,
    # so set and unset it explicitly rather than relying on the prefix form.
    unset REVV_ALLOW_ARCH_MISMATCH
    PREBUILT_FAIL_REASON=""
    check_prebuilt_arch > "$WORK/cpa.out" 2>&1
    rc=$?
    sed 's/^/  /' "$WORK/cpa.out" | scrub
    printf '  check_prebuilt_arch rc=%s reason=[%s]\n' "$rc" "$PREBUILT_FAIL_REASON"

    REVV_ALLOW_ARCH_MISMATCH=1
    PREBUILT_FAIL_REASON=""
    check_prebuilt_arch > "$WORK/cpa2.out" 2>&1
    rc=$?
    unset REVV_ALLOW_ARCH_MISMATCH
    printf '  with REVV_ALLOW_ARCH_MISMATCH: rc=%s reason=[%s]\n' \
        "$rc" "$PREBUILT_FAIL_REASON"
    grep -c 'REVV_ALLOW_ARCH_MISMATCH is set' "$WORK/cpa2.out" \
        | sed 's/^/  override-line-count=/'
    PATH="$SAVED_PATH"
done
unset SMI_OUT
rm -f "$WORK/cpa.out" "$WORK/cpa2.out"

# ---------------------------------------------------------------------------
banner "verify_sha256_or_confirm"
printf 'payload\n' > "$WORK/v_file"
good=$(sha256_of "$WORK/v_file")
case_ "matching sha256"
out=$(verify_sha256_or_confirm "$WORK/v_file" "$good" < /dev/null 2>&1)
rc=$?
printf 'rc=%s matched=%s\n' "$rc" "$(printf '%s' "$out" | grep -c 'sha256 verified')"
case_ "mismatching sha256"
out=$(verify_sha256_or_confirm "$WORK/v_file" "0000" < /dev/null 2>&1)
rc=$?
printf '%s\n' "$out" | scrub
printf 'rc=%s\n' "$rc"

# ---------------------------------------------------------------------------
banner "write_manifest (three rungs)"
case_ "source, patched"
write_manifest "source" '"mmvq_iquant_decode.patch", "pr26004-rebased-daef7b687.patch"' \
    "built from source" "" | scrub
scrub < "$BUILD_MANIFEST"
case_ "source, stock"
write_manifest "source" '' "built from source" "" | scrub
scrub < "$BUILD_MANIFEST"
case_ "prebuilt"
write_manifest "prebuilt" '"mmvq_iquant_decode.patch", "pr26004-rebased-daef7b687.patch"' \
    "$PREBUILT_URL" "" | scrub
scrub < "$BUILD_MANIFEST"
case_ "upstream (extra json)"
write_manifest "upstream" "" "https://example.invalid/asset.tar.gz" ',
  "backend": "vulkan"' | scrub
scrub < "$BUILD_MANIFEST"
case_ "revv.py read_build_manifest agrees"
REVV_HOME="$HOME_DIR" python3 -c '
import json, os, sys
sys.path.insert(0, os.environ["REPO"])
import revv
m = revv.read_build_manifest()
m["built_at"] = "<scrubbed>"
print(json.dumps(m, sort_keys=True, indent=2))
' 2>&1 | REPO="$REPO" scrub
rm -f "$BUILD_MANIFEST"

# ---------------------------------------------------------------------------
banner "install_revv_wrapper"
install_revv_wrapper | scrub
scrub < "$BIN_DIR/revv"
ls -l "$BIN_DIR/revv" | awk '{print "mode " $1}'
rm -f "$BIN_DIR/revv"

# ---------------------------------------------------------------------------
banner "report_path"
case_ "BIN_DIR on PATH"
PATH="$BIN_DIR:$PATH" report_path | scrub
case_ "BIN_DIR not on PATH"
report_path | scrub

# ---------------------------------------------------------------------------
banner "find_existing_llama_server / get_llama_version"
mkdir -p "$BIN_DIR"
cat > "$BIN_DIR/llama-server" <<'STUB'
#!/bin/sh
echo "version: 12345 (abcdef0)" >&2
echo "built with fake"
STUB
chmod +x "$BIN_DIR/llama-server"
case_ "present in BIN_DIR"
out=$(find_existing_llama_server)
rc=$?
printf 'found=%s rc=%s\n' "$(printf '%s' "$out" | sed "s#$HOME_DIR#<home>#")" "$rc"
printf 'version=[%s]\n' "$(get_llama_version "$BIN_DIR/llama-server")"
rm -f "$BIN_DIR/llama-server"
case_ "absent"
if command -v llama-server >/dev/null 2>&1; then
    echo "SKIPPED (a real llama-server is on PATH)"
else
    out=$(find_existing_llama_server)
    rc=$?
    printf 'found=[%s] rc=%s\n' "$out" "$rc"
fi

# ---------------------------------------------------------------------------
banner "install_binaries"
mk_build() {
    rm -rf "$LLAMA_SRC"
    mkdir -p "$1"
    for n in $2; do
        printf '#!/bin/sh\necho %s\n' "$n" > "$1/$n"
        chmod +x "$1/$n"
    done
}
show_bin() {
    for n in llama-server llama-cli; do
        if [ -L "$BIN_DIR/$n" ]; then
            printf '%s: symlink -> %s\n' "$n" \
                "$(readlink "$BIN_DIR/$n" | sed "s#$HOME_DIR#<home>#")"
        elif [ -f "$BIN_DIR/$n" ]; then
            printf '%s: regular file\n' "$n"
        else
            printf '%s: absent\n' "$n"
        fi
    done
}
case_ "build/bin/ layout, server + cli"
rm -f "$BIN_DIR/llama-server" "$BIN_DIR/llama-cli"
mk_build "$LLAMA_SRC/build/bin" "llama-server llama-cli"
install_binaries 2>&1 | scrub
show_bin
case_ "build/ layout, server only"
rm -f "$BIN_DIR/llama-server" "$BIN_DIR/llama-cli"
mk_build "$LLAMA_SRC/build" "llama-server"
install_binaries 2>&1 | scrub
show_bin
case_ "nothing built (fail path)"
rm -f "$BIN_DIR/llama-server" "$BIN_DIR/llama-cli"
rm -rf "$LLAMA_SRC"
mkdir -p "$LLAMA_SRC"
out=$( (install_binaries) 2>&1 )
rc=$?
printf '%s\n' "$out" | scrub
printf 'rc=%s\n' "$rc"

# ---------------------------------------------------------------------------
banner "apply_patch"
if ! command -v git >/dev/null 2>&1; then
    echo "SKIPPED (git not installed)"
else
    rm -rf "$LLAMA_SRC"
    mkdir -p "$LLAMA_SRC"
    (
        cd "$LLAMA_SRC" || exit 1
        git init --quiet .
        git config user.email snap@example.invalid
        git config user.name snap
        printf 'one\ntwo\nthree\n' > f.txt
        git add f.txt
        git commit --quiet -m base
        printf 'one\nTWO\nthree\n' > f.txt
        git diff > "$WORK/good.patch"
        git checkout --quiet -- f.txt
        printf 'nine\n' > other.txt
        git add other.txt
        git commit --quiet -m other
        printf 'ninety\n' > other.txt
        git diff > "$WORK/bad.patch"
        git checkout --quiet -- other.txt
        git rm --quiet other.txt
        git commit --quiet -m drop
    ) >/dev/null 2>&1
    case_ "applies cleanly"
    out=$(apply_patch "$WORK/good.patch" 2>&1)
    rc=$?
    printf '%s\n' "$out" | scrub
    printf 'rc=%s\n' "$rc"
    case_ "already applied"
    out=$(apply_patch "$WORK/good.patch" 2>&1)
    rc=$?
    printf '%s\n' "$out" | scrub
    printf 'rc=%s\n' "$rc"
    case_ "does not apply (fail path)"
    out=$( (apply_patch "$WORK/bad.patch") 2>&1 )
    rc=$?
    printf '%s\n' "$out" | scrub | sed -e 's/^  [0-9a-f]\{40\}$/  <sha>/'
    printf 'rc=%s\n' "$rc"
fi

# ---------------------------------------------------------------------------
banner "choose_mode (non-interactive)"
SRC_MODE=""
out=$(choose_mode < /dev/null 2>&1)
printf '%s\n' "$out"
choose_mode < /dev/null >/dev/null 2>&1
printf 'SRC_MODE=%s\n' "$SRC_MODE"

# ---------------------------------------------------------------------------
banner "download_prebuilt (localhost dead port -- no external network)"
download_prebuilt "http://127.0.0.1:1/nope.tar.gz" "$WORK/dl.tmp" > "$WORK/dl.out" 2>&1
rc=$?
scrub < "$WORK/dl.out"
printf 'rc=%s DOWNLOAD_HTTP_CODE=%s\n' "$rc" "$DOWNLOAD_HTTP_CODE"
rm -f "$WORK/dl.tmp" "$WORK/dl.out"

# ---------------------------------------------------------------------------
banner "ensure_prebuilt_downloaded (cached + verified, no network)"
mkdir -p "$REVV_HOME/cache"
printf 'pretend tarball\n' > "$REVV_HOME/cache/$(basename "$PREBUILT_URL")"
SAVED_SHA="$PREBUILT_SHA256"
PREBUILT_SHA256=$(sha256_of "$REVV_HOME/cache/$(basename "$PREBUILT_URL")")
PREBUILT_CACHE_FILE=""
ensure_prebuilt_downloaded > "$WORK/ep.out" 2>&1
rc=$?
scrub < "$WORK/ep.out"
printf 'rc=%s PREBUILT_CACHE_FILE=%s\n' "$rc" \
    "$(printf '%s' "$PREBUILT_CACHE_FILE" | sed "s#$HOME_DIR#<home>#")"
PREBUILT_SHA256="$SAVED_SHA"
rm -rf "$REVV_HOME/cache"

# ---------------------------------------------------------------------------
banner "check_cuda_runtime_libs"
printf '#!/bin/sh\n' > "$WORK/fakebin"
chmod +x "$WORK/fakebin"
out=$(check_cuda_runtime_libs "$WORK/fakebin" 2>&1)
rc=$?
printf '%s\n' "$out" | scrub
printf 'rc=%s\n' "$rc"

# ---------------------------------------------------------------------------
banner "try_prebuilt platform gate (this machine)"
PREBUILT_FAIL_REASON=""
PREBUILT_FAIL_DETAIL=""
try_prebuilt > "$WORK/tp.out" 2>&1
rc=$?
scrub < "$WORK/tp.out"
printf 'rc=%s\nPREBUILT_FAIL_REASON=[%s]\nPREBUILT_FAIL_DETAIL=[%s]\n' \
    "$rc" "$PREBUILT_FAIL_REASON" "$PREBUILT_FAIL_DETAIL"

# ---------------------------------------------------------------------------
banner "do_install_upstream platform gate (this machine)"
out=$( (do_install_upstream) 2>&1 )
rc=$?
printf '%s\n' "$out" | scrub
printf 'rc=%s\n' "$rc"

printf '\n===== END =====\n'

rm -rf "$HOME_DIR" "$WORK"
exit 0
