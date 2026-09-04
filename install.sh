#!/bin/sh
# revv installer -- locates, downloads, or builds llama-server, installs
# the revv CLI.
#
# POSIX sh only: no bashisms, no sudo, no external tools beyond git/cmake/
# python3 (and nvcc when a source build is actually needed) plus curl or
# wget (when a download is actually needed). Safe to run more than once --
# completed work is detected and skipped.
set -e

PINNED_COMMIT="daef7b6874397a5a7c3d7e38b55e2ee0adf7da38"
PINNED_BUILD="b10712"
REVV_VERSION="1.0.0"
LLAMA_REPO_URL="https://github.com/ggml-org/llama.cpp.git"

# revv's own patched, CUDA-enabled prebuilt (rung 1). Not published at the
# time this script was written -- the 404 case below is expected, not a bug.
PREBUILT_URL="https://github.com/mericanii-technologies/revv/releases/download/v1.1.0-binaries/revv-llama-server-1.1.0-linux-x86_64-cuda12.tar.gz"
PREBUILT_SHA256="3522ef73cb9a93b865e0cfe26c6ccb0f7021e3415fc011e844537bf948bc4423"
# The prebuilt binary is compiled for this compute capability only (Ampere /
# 30-series). Other cards rely on driver JIT from PTX, which may simply not
# work -- see check_prebuilt_arch below.
PREBUILT_CUDA_ARCH="8.6"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

fail() {
    printf 'error: %s\n' "$1" >&2
    if [ -n "$2" ]; then
        printf '\n%s\n' "$2" >&2
    fi
    exit 1
}

on_interrupt() {
    printf '\ninstall.sh: interrupted -- nothing after this point ran.\n' >&2
    exit 1
}
trap on_interrupt INT TERM

usage() {
    cat <<'EOF'
Usage: install.sh [OPTIONS]

Installs the revv CLI and locates or installs llama-server, in one of
three ways ("rungs"), most convenient first:

  --prebuilt      (default) Download revv's patched, CUDA-enabled
                  prebuilt binary. Fastest, and the only prebuilt with
                  CUDA on Linux -- but it's a third-party (revv) fork
                  build, not an official upstream release.
  --upstream      Download the official ggml-org/llama.cpp prebuilt.
                  Most trusted, but upstream publishes no Linux CUDA
                  prebuilt: on Linux x86_64 this installs the Vulkan
                  backend instead -- a different backend, with different
                  performance, that none of revv's published numbers
                  describe.
  --source        Build llama.cpp from source: clone the pinned commit,
                  toolchain preflight, compile with CUDA. Patches are
                  applied by default (prompted interactively, or applied
                  automatically when run non-interactively); pass
                  --stock for an unmodified build. Slowest rung to set
                  up, but you compile it yourself.

  --patched       Synonym for --source with patches applied. Older flag
                  name from before the three-rung install; still works.
  --stock         Synonym for --source with no patches applied. Older
                  flag name from before the three-rung install; still
                  works.
  --force-build, --force
                  Reinstall/rebuild even if an llama-server is already
                  available in $REVV_HOME/bin or on PATH.
  --help          Show this help and exit.

Environment:
  REVV_HOME        Where revv stores its binaries (bin/), downloads
                  (cache/), extracted prebuilt runtimes (runtime/),
                  source checkout (src/), and build manifest. Defaults
                  to $HOME/.revv.
  CUDAARCHS        (--source only) Passed through verbatim as
                  -DCMAKE_CUDA_ARCHITECTURES when building llama.cpp
                  with CUDA. Defaults to the compute capability of the
                  detected GPU(s), or 'native' if that can't be
                  determined.
  REVV_ALLOW_UNVERIFIED
                  Set to any non-empty value to install a downloaded binary
                  even when no sha256 tool is available to verify it. Off by
                  default: an unverifiable download is refused, not accepted.
  REVV_ALLOW_ARCH_MISMATCH
                  (--prebuilt only) The revv prebuilt is compiled for sm_86
                  (compute capability 8.6) only. Set to any non-empty value
                  to install it on a different card anyway -- it may fail to
                  load, or fall back to slow driver JIT. Off by default: an
                  explicit --prebuilt on a mismatched card fails with an
                  explanation instead of silently installing a binary that
                  may not run; the automatic (no-flag) path just falls back
                  to --source without needing this.
  REVV_SKIP_TOOLCHAIN_CHECK
                  (--source only) Set to any non-empty value to skip
                  the CUDA/host-compiler preflight check and proceed
                  straight to the build. Not recommended -- that check
                  exists because known-bad pairs fail confusingly, deep
                  inside the build, after a long wait.

With no flag, this script tries --prebuilt first and automatically falls
back to --source (with patches) if the prebuilt doesn't apply to this
machine -- wrong OS/arch, glibc too old, or the release isn't published
yet -- printing exactly which. It never automatically falls back to
--upstream: Vulkan is a real capability difference, not a fallback tier,
so that stays an explicit choice.
EOF
}

# True (0) iff $1 is a plain non-negative integer. Used before doing any
# arithmetic comparison on a version field we parsed out of tool output --
# garbage input (missing tool, unexpected format) should degrade to
# "skip the check", never to a shell arithmetic error.
is_int() {
    case "$1" in
        ''|*[!0-9]*) return 1 ;;
        *) return 0 ;;
    esac
}

# Portable "how many cores do we have" with a safe fallback.
nproc_portable() {
    if command -v nproc >/dev/null 2>&1; then
        nproc
    elif command -v sysctl >/dev/null 2>&1; then
        n=$(sysctl -n hw.ncpu 2>/dev/null)
        if [ -n "$n" ]; then
            printf '%s\n' "$n"
        else
            printf '4\n'
        fi
    else
        printf '4\n'
    fi
}

# ---------------------------------------------------------------------------
# Resolve paths
# ---------------------------------------------------------------------------

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) \
    || fail "could not resolve the directory this script lives in"
REVV_DIR="$SCRIPT_DIR"
PATCHES_DIR="$REVV_DIR/patches"
REVV_PY="$REVV_DIR/revv.py"

REVV_HOME="${REVV_HOME:-$HOME/.revv}"
BIN_DIR="$REVV_HOME/bin"
SRC_DIR="$REVV_HOME/src"
LLAMA_SRC="$SRC_DIR/llama.cpp"
BUILD_MANIFEST="$REVV_HOME/build.json"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

RUNG=""
SRC_MODE=""
FORCE_BUILD=0

while [ $# -gt 0 ]; do
    case "$1" in
        --prebuilt)
            RUNG="prebuilt"
            ;;
        --upstream)
            RUNG="upstream"
            ;;
        --source)
            RUNG="source"
            ;;
        --patched)
            RUNG="source"
            SRC_MODE="patched"
            echo "note: --patched is now spelled '--source' (source build, patches applied). Both still work."
            ;;
        --stock)
            RUNG="source"
            SRC_MODE="stock"
            echo "note: --stock is now spelled '--source --stock' (source build, no patches). Both still work."
            ;;
        --force-build|--force)
            FORCE_BUILD=1
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            usage >&2
            fail "unknown option: $1"
            ;;
    esac
    shift
done

# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

check_python() {
    if ! command -v python3 >/dev/null 2>&1; then
        fail "python3 not found on PATH" \
"revv is a Python 3.9+ CLI with no external dependencies. Install Python 3.9
or newer (e.g. 'brew install python3' on macOS, 'apt install python3' on
Debian/Ubuntu), then re-run this script."
    fi
    PY_VER=$(python3 -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')
    PY_OK=$(python3 -c 'import sys; print(1 if sys.version_info >= (3, 9) else 0)')
    if [ "$PY_OK" != "1" ]; then
        fail "python3 is too old: found $PY_VER, need >= 3.9" \
"Install Python 3.9 or newer and make sure the 'python3' found on PATH
points at it (check with: python3 -c 'import sys; print(sys.executable)')."
    fi
    printf 'python3: %s (OK, >= 3.9)\n' "$PY_VER"
}

check_build_tools() {
    if ! command -v git >/dev/null 2>&1; then
        fail "git not found on PATH" \
"Install git (e.g. 'brew install git' on macOS, 'apt install git' on
Debian/Ubuntu), then re-run this script."
    fi
    if ! command -v cmake >/dev/null 2>&1; then
        fail "cmake not found on PATH" \
"Install CMake (e.g. 'brew install cmake' on macOS, 'apt install cmake' on
Debian/Ubuntu), then re-run this script."
    fi
    if ! command -v nvcc >/dev/null 2>&1; then
        fail "nvcc (the CUDA compiler) not found on PATH" \
"revv has no CPU build path -- llama-server must be built with CUDA support.
Install the NVIDIA CUDA Toolkit (nvcc normally lives under
/usr/local/cuda/bin, or is provided by your distro's 'cuda-toolkit' /
'nvidia-cuda-toolkit' package), make sure it is on PATH, then re-run this
script. If you already have a working llama-server on PATH from elsewhere,
you don't need this: re-run without --force-build and this script will use
it instead of building."
    fi
}

# ---------------------------------------------------------------------------
# Existing llama-server detection
# ---------------------------------------------------------------------------

find_existing_llama_server() {
    if [ -x "$BIN_DIR/llama-server" ]; then
        printf '%s\n' "$BIN_DIR/llama-server"
        return 0
    fi
    if command -v llama-server >/dev/null 2>&1; then
        command -v llama-server
        return 0
    fi
    return 1
}

get_llama_version() {
    v=$("$1" --version 2>&1 | grep -i version | head -n 1)
    if [ -n "$v" ]; then
        printf '%s\n' "$v"
    else
        printf 'version unknown\n'
    fi
}

# ---------------------------------------------------------------------------
# Mode selection (--source rung: patched vs. stock)
# ---------------------------------------------------------------------------

choose_mode() {
    if [ -t 0 ] && [ -t 1 ]; then
        printf 'Build llama.cpp with the revv performance/session-restore patches, or stock upstream?\n'
        printf '  [P] patched  (recommended -- default)\n'
        printf '  [S] stock    (no patches)\n'
        printf 'Choice [P/s]: '
        read -r REPLY_CHOICE || REPLY_CHOICE=""
        case "$REPLY_CHOICE" in
            [Ss]*)
                SRC_MODE="stock"
                ;;
            *)
                SRC_MODE="patched"
                ;;
        esac
    else
        SRC_MODE="patched"
        echo "non-interactive shell: defaulting to --source, patched"
    fi
}

# ---------------------------------------------------------------------------
# llama.cpp source checkout
# ---------------------------------------------------------------------------

setup_llama_src() {
    mkdir -p "$SRC_DIR"
    if [ -d "$LLAMA_SRC/.git" ]; then
        echo "reusing existing checkout at $LLAMA_SRC"
        if ! (cd "$LLAMA_SRC" && git fetch --depth 1 origin "$PINNED_COMMIT" \
              && git checkout --quiet FETCH_HEAD); then
            fail "could not fetch/check out the pinned commit in $LLAMA_SRC" \
"Check network access, or remove the directory and re-run this script to
fetch it fresh:
  rm -rf $LLAMA_SRC"
        fi
        return 0
    fi

    rm -rf "$LLAMA_SRC"
    mkdir -p "$LLAMA_SRC"
    echo "fetching llama.cpp @ $PINNED_COMMIT (shallow)..."
    if (cd "$LLAMA_SRC" \
        && git init --quiet \
        && git remote add origin "$LLAMA_REPO_URL" \
        && git fetch --depth 1 origin "$PINNED_COMMIT" \
        && git checkout --quiet FETCH_HEAD); then
        return 0
    fi

    echo "shallow SHA fetch failed; falling back to a full clone..." >&2
    rm -rf "$LLAMA_SRC"
    if ! git clone "$LLAMA_REPO_URL" "$LLAMA_SRC"; then
        fail "git clone of llama.cpp failed" \
"Check network access and that $LLAMA_REPO_URL is reachable, then re-run
this script."
    fi
    if ! (cd "$LLAMA_SRC" && git checkout --quiet "$PINNED_COMMIT"); then
        fail "could not check out pinned commit $PINNED_COMMIT" \
"The clone succeeded but that commit was not found in it. See
$PATCHES_DIR/PATCHES.md for the pinned commit this build is certified
against."
    fi
}

# ---------------------------------------------------------------------------
# Patch application
# ---------------------------------------------------------------------------

apply_patch() {
    patch_file="$1"
    patch_name=$(basename "$patch_file")

    check_out=""
    if check_out=$(cd "$LLAMA_SRC" && git apply --check "$patch_file" 2>&1); then
        if ! (cd "$LLAMA_SRC" && git apply "$patch_file"); then
            fail "git apply failed for $patch_name even though --check passed" \
"This should not happen. Please report it."
        fi
        echo "  applied: $patch_name"
        return 0
    fi

    if (cd "$LLAMA_SRC" && git apply --reverse --check "$patch_file") >/dev/null 2>&1; then
        echo "  already applied: $patch_name"
        return 0
    fi

    got="$(cd "$LLAMA_SRC" && git rev-parse HEAD 2>/dev/null)"
    fail "patch $patch_name does not apply to $LLAMA_SRC" \
"git apply --check reported:
$check_out

Both patches are verified to apply cleanly to a pristine checkout of
llama.cpp commit $PINNED_COMMIT (build $PINNED_BUILD). The checkout at
$LLAMA_SRC is currently at:
  $got
If that does not match $PINNED_COMMIT, remove the directory and re-run
this script to fetch the pinned commit fresh:
  rm -rf $LLAMA_SRC"
}

# ---------------------------------------------------------------------------
# Build (--source rung)
# ---------------------------------------------------------------------------

install_binaries() {
    mkdir -p "$BIN_DIR"

    SERVER_BIN=""
    for candidate in "$LLAMA_SRC/build/bin/llama-server" "$LLAMA_SRC/build/llama-server"; do
        if [ -x "$candidate" ]; then
            SERVER_BIN="$candidate"
            break
        fi
    done
    if [ -z "$SERVER_BIN" ]; then
        fail "build finished but no llama-server binary was found" \
"Looked in $LLAMA_SRC/build/bin and $LLAMA_SRC/build. Check the build
output above for errors."
    fi
    rm -f "$BIN_DIR/llama-server"
    if ! ln -s "$SERVER_BIN" "$BIN_DIR/llama-server" 2>/dev/null; then
        cp "$SERVER_BIN" "$BIN_DIR/llama-server"
    fi
    chmod +x "$BIN_DIR/llama-server" 2>/dev/null || true
    echo "installed: $BIN_DIR/llama-server -> $SERVER_BIN"

    CLI_BIN=""
    for candidate in "$LLAMA_SRC/build/bin/llama-cli" "$LLAMA_SRC/build/llama-cli"; do
        if [ -x "$candidate" ]; then
            CLI_BIN="$candidate"
            break
        fi
    done
    if [ -n "$CLI_BIN" ]; then
        rm -f "$BIN_DIR/llama-cli"
        if ! ln -s "$CLI_BIN" "$BIN_DIR/llama-cli" 2>/dev/null; then
            cp "$CLI_BIN" "$BIN_DIR/llama-cli"
        fi
        chmod +x "$BIN_DIR/llama-cli" 2>/dev/null || true
        echo "installed: $BIN_DIR/llama-cli -> $CLI_BIN"
    fi
}

# Writes $REVV_HOME/build.json. Shared by all three rungs.
#   $1 install_method  "prebuilt" | "upstream" | "source"
#   $2 patch_list       e.g. '"mmvq_iquant_decode.patch", "..."' or ''
#   $3 source_str        short human string: a URL for downloads, or
#                        "built from source"
#   $4 extra_json        optional extra keys, e.g. ',\n  "backend": "vulkan"'
#                        (must already include its own leading comma)
write_manifest() {
    method="$1"
    patch_list="$2"
    source_str="$3"
    extra="$4"
    built_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    tmp_manifest="$BUILD_MANIFEST.tmp.$$"
    cat > "$tmp_manifest" <<MANIFEST_EOF
{
  "base_commit": "$PINNED_COMMIT",
  "patches": [$patch_list],
  "built_at": "$built_at",
  "revv_version": "$REVV_VERSION",
  "install_method": "$method",
  "source": "$source_str"$extra
}
MANIFEST_EOF
    mv "$tmp_manifest" "$BUILD_MANIFEST"
    echo "wrote build manifest: $BUILD_MANIFEST"
}

# Picks the -DCMAKE_CUDA_ARCHITECTURES value. Building the full default
# fan-out (50/61/70/75/80/86/89/90...) is far slower and produces a much
# larger binary than a build pinned to the card(s) actually present, so we
# try hard to detect it rather than let cmake fall back on its own default.
#
# Sets CUDA_ARCH_REASON as a side effect, for logging by the caller.
detect_cuda_archs() {
    if [ -n "$CUDAARCHS" ]; then
        CUDA_ARCH_REASON="from CUDAARCHS"
        printf '%s\n' "$CUDAARCHS"
        return 0
    fi

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        CUDA_ARCH_REASON="could not detect; letting CMake probe the local card"
        printf 'native\n'
        return 0
    fi

    raw=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null) || raw=""

    # Convert each "8.6"-style line to "86", dedupe, and join with ';'
    # (CMake's list separator). Bail to "BAD" on anything that doesn't look
    # like a compute capability -- e.g. "N/A" -- so the caller falls back
    # to 'native' instead of handing cmake garbage.
    result=$(printf '%s\n' "$raw" | tr -d ' \r' | awk '
        NF == 0 { next }
        !/^[0-9]+\.[0-9]+$/ { bad = 1; exit }
        {
            gsub(/\./, "", $0)
            if (!($0 in seen)) {
                seen[$0] = 1
                out = out (out == "" ? "" : ";") $0
                n++
            }
        }
        END {
            if (bad || out == "") { print "BAD"; exit }
            print out
            print n
        }
    ')
    archs=$(printf '%s\n' "$result" | sed -n '1p')

    if [ "$archs" = "BAD" ] || [ -z "$archs" ]; then
        CUDA_ARCH_REASON="could not detect; letting CMake probe the local card"
        printf 'native\n'
        return 0
    fi

    count=$(printf '%s\n' "$result" | sed -n '2p')
    if [ "$count" -gt 1 ]; then
        CUDA_ARCH_REASON="detected, $count GPUs"
    else
        first_cc=$(printf '%s\n' "$raw" | tr -d ' \r' | head -n 1)
        CUDA_ARCH_REASON="detected from nvidia-smi compute_cap $first_cc"
    fi
    printf '%s\n' "$archs"
}

# ---------------------------------------------------------------------------
# CUDA / host-compiler / glibc preflight
# ---------------------------------------------------------------------------
#
# A real fresh-user install lost ~30 minutes here: CUDA 12.6's headers
# against glibc 2.43 + gcc-15 fail deep inside the nvcc build with
# confusing template/__builtin errors that look like an llama.cpp bug but
# are actually a toolchain mismatch -- and the previous version of this
# script only surfaced that after cmake configure and part of a build had
# already run. Catch it here instead, before any of that starts.

# Is this WSL2? Only needs a case-insensitive substring match on
# /proc/version, but the path is read from a variable (not hardcoded) so
# tests can point it at a fixture instead of the real kernel file.
is_wsl2() {
    proc_version_file="${REVV_PROC_VERSION_FILE:-/proc/version}"
    [ -r "$proc_version_file" ] || return 1
    tr '[:upper:]' '[:lower:]' < "$proc_version_file" 2>/dev/null | grep -q microsoft
}

# Parses "ldd --version"'s first line into GLIBC_VER / GLIBC_MAJOR /
# GLIBC_MINOR. Shared by the source-build toolchain preflight and the
# --prebuilt rung's platform check, so there's exactly one place that
# knows how to read glibc's version out of ldd.
parse_glibc() {
    glibc_line=$(ldd --version 2>/dev/null | head -n 1)
    GLIBC_VER=$(printf '%s\n' "$glibc_line" | awk '{print $NF}')
    GLIBC_MAJOR=${GLIBC_VER%%.*}
    GLIBC_MINOR=${GLIBC_VER#*.}
}

# Prints the maximum host-gcc major version a given CUDA major/minor
# supports, per NVIDIA's documented compiler support matrix. Empty output
# means "no data for this CUDA version" -- the caller skips the pair check
# rather than risk a false failure on an untabulated CUDA release.
max_host_gcc_for_cuda() {
    cmaj="$1"
    cmin="$2"
    case "$cmaj" in
        11)
            printf '11\n'
            ;;
        12)
            if [ "$cmin" -le 3 ]; then
                printf '12\n'
            elif [ "$cmin" -le 5 ]; then
                printf '13\n'
            else
                printf '14\n'
            fi
            ;;
        1[3-9]|[2-9][0-9])
            printf '15\n'
            ;;
        *)
            printf '\n'
            ;;
    esac
}

# Looks for an older g++ already installed that satisfies max gcc major
# $1, newest-to-oldest so the suggestion is as close to the broken one as
# possible. Prints the command name (e.g. "g++-13") and returns 0 if
# found; returns 1 with no output otherwise.
find_compatible_hostcxx() {
    want_max="$1"
    for cand in g++-14 g++-13 g++-12 g++-11; do
        cand_major=${cand#g++-}
        if [ "$cand_major" -le "$want_max" ] && command -v "$cand" >/dev/null 2>&1; then
            printf '%s\n' "$cand"
            return 0
        fi
    done
    return 1
}

check_toolchain() {
    if [ -n "$REVV_SKIP_TOOLCHAIN_CHECK" ]; then
        echo "REVV_SKIP_TOOLCHAIN_CHECK is set: skipping the CUDA/host-compiler"
        echo "preflight check and proceeding anyway."
        return 0
    fi

    cuda_release=$(nvcc --version 2>/dev/null | grep -o 'release [0-9][0-9]*\.[0-9][0-9]*' | head -n 1)
    cuda_ver=${cuda_release#release }
    CUDA_MAJOR=${cuda_ver%%.*}
    CUDA_MINOR=${cuda_ver#*.}

    hostcxx="${CUDAHOSTCXX:-g++}"
    hostver=$("$hostcxx" -dumpfullversion -dumpversion 2>/dev/null | head -n 1)
    if [ -z "$hostver" ]; then
        hostver=$(gcc -dumpversion 2>/dev/null | head -n 1)
    fi
    HOST_GCC_MAJOR=${hostver%%.*}

    parse_glibc

    printf 'toolchain: CUDA %s, g++ %s, glibc %s\n' \
        "${cuda_ver:-unknown}" "${HOST_GCC_MAJOR:-unknown}" "${GLIBC_VER:-unknown}"

    if is_int "$CUDA_MAJOR" && is_int "$GLIBC_MAJOR" && is_int "$GLIBC_MINOR"; then
        if [ "$CUDA_MAJOR" -lt 13 ] \
           && { [ "$GLIBC_MAJOR" -gt 2 ] || { [ "$GLIBC_MAJOR" -eq 2 ] && [ "$GLIBC_MINOR" -ge 42 ]; }; }; then
            echo ""
            echo "warning: glibc $GLIBC_VER is newer than CUDA $cuda_ver's headers expect."
            echo "         CUDA toolkits before 13.x are known to break against glibc 2.42+"
            echo "         (confusing nvcc template/__builtin errors, not an llama.cpp bug)."
            echo "         cuda-toolkit-13-3 or newer resolves this."
        fi
    fi

    if is_int "$CUDA_MAJOR" && is_int "$CUDA_MINOR" && is_int "$HOST_GCC_MAJOR"; then
        max_gcc=$(max_host_gcc_for_cuda "$CUDA_MAJOR" "$CUDA_MINOR")
        if [ -n "$max_gcc" ] && [ "$HOST_GCC_MAJOR" -gt "$max_gcc" ]; then
            echo ""
            echo "############################################################"
            echo "# KNOWN-BAD TOOLCHAIN PAIR"
            echo "############################################################"
            echo "CUDA $cuda_ver supports host gcc up to major $max_gcc; found g++ $HOST_GCC_MAJOR."
            echo "This combination fails deep inside the nvcc build with confusing"
            echo "template/__builtin errors that look like an llama.cpp bug but are"
            echo "actually this mismatch. Better to catch it now than burn 30 minutes"
            echo "watching cmake configure and part of a build fail."

            suggestion=$(find_compatible_hostcxx "$max_gcc") || suggestion=""

            detail="Two ways to fix this:

  a) (recommended) Install a newer CUDA toolkit -- cuda-toolkit-13-3 or
     newer supports g++ $HOST_GCC_MAJOR."
            if [ -n "$suggestion" ]; then
                detail="$detail

  b) Or point nvcc at the older compiler already on this machine:
       CUDAHOSTCXX=$suggestion sh install.sh --source"
            else
                detail="$detail

  b) Or install an older host compiler nvcc supports, e.g.:
       sudo apt install g++-$max_gcc
     then:
       CUDAHOSTCXX=g++-$max_gcc sh install.sh --source"
            fi

            if is_wsl2; then
                detail="$detail

You're on WSL2: see this repo's README.md, WSL2 section, for the exact
apt commands (NVIDIA's wsl-ubuntu repo keyring plus cuda-toolkit-13-3)
for remedy (a)."
            fi

            detail="$detail

Set REVV_SKIP_TOOLCHAIN_CHECK=1 to bypass this check and proceed anyway
(not recommended -- you will very likely hit the build failure this is
trying to save you from)."

            fail "CUDA $cuda_ver + g++ $HOST_GCC_MAJOR is a known-bad host-compiler pair" "$detail"
        fi
    fi
}

do_build() {
    check_build_tools
    check_toolchain
    setup_llama_src

    if [ "$SRC_MODE" = "patched" ]; then
        echo "applying patches..."
        apply_patch "$PATCHES_DIR/mmvq_iquant_decode.patch"
        apply_patch "$PATCHES_DIR/pr26004-rebased-daef7b687.patch"
    else
        echo "--stock: building unmodified upstream at $PINNED_COMMIT"
        echo "note: the +10% raw / +2.5% end-to-end kernel gain and the"
        echo "18x session-restore capability are skipped in this build."
        echo "'revv doctor' will report this build as unpatched."
    fi

    njobs=$(nproc_portable)

    CUDA_ARCH=$(detect_cuda_archs)
    echo "CUDA architecture: $CUDA_ARCH ($CUDA_ARCH_REASON)"
    echo "  (pinned instead of the default arch fan-out -- significantly"
    echo "  faster to build and a much smaller binary)"

    # cmake caches CMAKE_CUDA_ARCHITECTURES in build/CMakeCache.txt. If a
    # previous run configured it with a different value, cmake will happily
    # keep using the stale one instead of picking up ours -- drop just the
    # cache file (not the whole build dir) so the configure step below
    # genuinely re-runs with the architecture we just chose.
    cache_file="$LLAMA_SRC/build/CMakeCache.txt"
    if [ -f "$cache_file" ]; then
        cached_arch=$(sed -n 's/^CMAKE_CUDA_ARCHITECTURES:STRING=//p' "$cache_file")
        if [ -n "$cached_arch" ] && [ "$cached_arch" != "$CUDA_ARCH" ]; then
            echo "warning: cached CMAKE_CUDA_ARCHITECTURES ($cached_arch) differs from"
            echo "         the target ($CUDA_ARCH) -- removing stale $cache_file"
            echo "         so cmake reconfigures with the new value."
            rm -f "$cache_file"
        fi
    fi

    echo "configuring build (cmake, CUDA on)..."
    if ! (cd "$LLAMA_SRC" && cmake -B build -DGGML_CUDA=ON \
          -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
          -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH"); then
        fail "cmake configure failed" \
"See the cmake output above. Common causes: the CUDA toolkit is not fully
installed, or a stale build directory is left over from a previous failed
attempt (try: rm -rf $LLAMA_SRC/build)."
    fi

    echo "building (this can take a while; using $njobs parallel job(s))..."
    if ! (cd "$LLAMA_SRC" && cmake --build build --config Release -j "$njobs"); then
        fail "build failed" \
"See the compiler output above. If CUDA headers or libraries are missing,
install the full CUDA Toolkit and re-run this script."
    fi

    install_binaries

    if [ "$SRC_MODE" = "patched" ]; then
        patch_list='"mmvq_iquant_decode.patch", "pr26004-rebased-daef7b687.patch"'
    else
        patch_list=''
    fi
    write_manifest "source" "$patch_list" "built from source" ""
}

# ---------------------------------------------------------------------------
# Prebuilt install (rung 1: revv's patched CUDA binary)
# ---------------------------------------------------------------------------

sha256_of() {
    file="$1"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$file" 2>/dev/null | awk '{print $1}'
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$file" 2>/dev/null | awk '{print $1}'
    else
        return 1
    fi
}

# Downloads $1 to $2 with curl if available, else wget. Both show a
# progress meter by default, so nothing extra is needed for that. Hard
# fails (there is no rung that doesn't eventually need one of these) if
# neither is installed.
download_with_progress() {
    url="$1"
    dest="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -fL -o "$dest" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$dest" "$url"
    else
        fail "neither curl nor wget found on PATH" \
"revv needs curl or wget to download prebuilt binaries. Install one (e.g.
'brew install curl' on macOS, 'apt install curl' on Debian/Ubuntu), or use
--source to build llama.cpp from source instead."
    fi
}

# Like download_with_progress, but for the prebuilt rung specifically:
# classifies *why* a failed download failed, so the caller can tell "the
# release isn't published yet" (benign, expected right now) apart from "this
# machine can't reach GitHub" (the user's network). Sets DOWNLOAD_HTTP_CODE
# to the HTTP status if one was obtained (e.g. "404"), or "000" if the
# request never got far enough to get one at all (DNS/route/TLS/timeout).
# Returns 0 on a clean 200, 1 otherwise -- never calls fail() itself, this
# rung already has its own soft-fail/auto-fallback convention.
download_prebuilt() {
    url="$1"
    dest="$2"
    DOWNLOAD_HTTP_CODE="000"

    if command -v curl >/dev/null 2>&1; then
        # No -f: with -f, curl discards the response and we lose the status
        # code on a 404 along with it. Plain -sS -w gets us the code either
        # way, and a 404's response body is a tiny HTML page we're about to
        # rm -f regardless.
        code=$(curl -sS -L -o "$dest" -w '%{http_code}' "$url" 2>/dev/null)
        curl_status=$?
        is_int "$code" && DOWNLOAD_HTTP_CODE="$code"
        [ "$curl_status" -eq 0 ] && [ "$DOWNLOAD_HTTP_CODE" = "200" ] && return 0
        return 1
    fi

    if command -v wget >/dev/null 2>&1; then
        wget_log="$dest.wgetlog.$$"
        if wget -o "$wget_log" -O "$dest" "$url"; then
            rm -f "$wget_log"
            DOWNLOAD_HTTP_CODE="200"
            return 0
        fi
        wget_status=$?
        # wget exposes no numeric status the way curl's -w does; exit 8
        # ("server issued an error response") is the closest signal that we
        # got a real HTTP response and it was bad, rather than never
        # reaching the server at all. Grep its log for the specific case we
        # care about distinguishing; anything else stays "unknown" and gets
        # the generic message.
        if [ "$wget_status" -eq 8 ] && grep -q '404' "$wget_log" 2>/dev/null; then
            DOWNLOAD_HTTP_CODE="404"
        fi
        rm -f "$wget_log"
        return 1
    fi

    fail "neither curl nor wget found on PATH" \
"revv needs curl or wget to download prebuilt binaries. Install one (e.g.
'brew install curl' on macOS, 'apt install curl' on Debian/Ubuntu), or use
--source to build llama.cpp from source instead."
}

# Verifies $1 against expected sha256 $2. If neither sha256sum nor shasum
# is installed, warns loudly and proceeds only if the shell is
# non-interactive or the user explicitly confirms -- never silently.
verify_sha256_or_confirm() {
    file="$1"
    expected="$2"
    if command -v sha256sum >/dev/null 2>&1 || command -v shasum >/dev/null 2>&1; then
        got=$(sha256_of "$file") || got=""
        if [ "$got" = "$expected" ]; then
            echo "  sha256 verified: $got"
            return 0
        fi
        echo "  sha256 mismatch: expected $expected, got ${got:-<none>}" >&2
        return 1
    fi

    echo "" >&2
    echo "warning: neither sha256sum nor shasum is available on this machine --" >&2
    echo "         the download cannot be verified against its known-good sha256." >&2
    if [ -t 0 ] && [ -t 1 ]; then
        printf 'Continue anyway, unverified? [y/N]: ' >&2
        read -r REPLY_VERIFY || REPLY_VERIFY=""
        case "$REPLY_VERIFY" in
            [Yy]*) return 0 ;;
            *) return 1 ;;
        esac
    fi
    # Fail CLOSED. This is a binary download; "could not check" must not
    # silently become "checked". sha256sum ships with coreutils on every
    # mainstream Linux, so reaching here at all is unusual and worth stopping
    # for. The escape hatch is explicit and has to be typed on purpose.
    if [ -n "${REVV_ALLOW_UNVERIFIED:-}" ]; then
        echo "REVV_ALLOW_UNVERIFIED is set: continuing without verification." >&2
        return 0
    fi
    echo "         Refusing to install an unverified binary." >&2
    echo "         Install coreutils (sha256sum), or re-run with" >&2
    echo "         REVV_ALLOW_UNVERIFIED=1 if you accept the risk," >&2
    echo "         or use --source to build it yourself instead." >&2
    return 1
}

# Checked before downloading anything (no point spending 78 MB finding this
# out afterward). Returns 0 to proceed -- either it matches, or the check
# itself couldn't be done and we don't block on a missing check. Returns 1
# with PREBUILT_FAIL_REASON set on a genuine mismatch, so the existing
# auto-fallback machinery in try_prebuilt()/main drops to --source; an
# explicit --prebuilt then surfaces that reason as a hard failure the same
# way a 404 or a bad sha256 would, unless REVV_ALLOW_ARCH_MISMATCH is set.
check_prebuilt_arch() {
    sm_target=$(printf '%s' "$PREBUILT_CUDA_ARCH" | tr -d '.')

    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "  warning: 'nvidia-smi' not found -- could not verify GPU architecture" >&2
        echo "           against the prebuilt's sm_$sm_target target. Proceeding anyway." >&2
        return 0
    fi

    raw=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null) || raw=""

    # One compute-capability value per line, deduped. Same "bail to BAD on
    # anything that doesn't look like a compute capability" approach as
    # detect_cuda_archs -- a partial/garbage nvidia-smi output should
    # degrade to "could not verify", not to a false mismatch report.
    caps=$(printf '%s\n' "$raw" | tr -d ' \r' | awk '
        NF == 0 { next }
        !/^[0-9]+\.[0-9]+$/ { bad = 1; exit }
        {
            if (!($0 in seen)) {
                seen[$0] = 1
                out = out (out == "" ? "" : "\n") $0
            }
        }
        END {
            if (bad) { print "BAD"; exit }
            if (out != "") print out
        }
    ')

    if [ -z "$caps" ] || [ "$caps" = "BAD" ]; then
        echo "  warning: could not parse a GPU compute capability from nvidia-smi --" >&2
        echo "           could not verify it against the prebuilt's sm_$sm_target target." >&2
        echo "           Proceeding anyway." >&2
        return 0
    fi

    mismatch=0
    for cc in $caps; do
        [ "$cc" = "$PREBUILT_CUDA_ARCH" ] || mismatch=1
    done
    [ "$mismatch" -eq 0 ] && return 0

    detected=$(printf '%s\n' "$caps" | tr '\n' ',' | sed 's/,$//')
    echo ""
    echo "warning: GPU architecture mismatch."
    echo "         The revv prebuilt is compiled for sm_$sm_target (compute capability"
    echo "         $PREBUILT_CUDA_ARCH) only. This machine reports: $detected."
    echo "         The binary may fail to load, or fall back to slow driver JIT"
    echo "         compilation from PTX, on this card."
    echo "         The reliable path here is --source: it auto-detects and compiles"
    echo "         for the local architecture."

    if [ -n "$REVV_ALLOW_ARCH_MISMATCH" ]; then
        echo "         REVV_ALLOW_ARCH_MISMATCH is set: proceeding with the sm_$sm_target"
        echo "         prebuilt anyway."
        return 0
    fi

    PREBUILT_FAIL_REASON="GPU compute capability $detected does not match the prebuilt's sm_$sm_target target (override with REVV_ALLOW_ARCH_MISMATCH=1)"
    return 1
}

# Downloads the revv prebuilt into $REVV_HOME/cache/, reusing it if
# already present and sha256-verified (idempotent: re-running does not
# re-fetch ~78 MB). Sets PREBUILT_CACHE_FILE on success; sets
# PREBUILT_FAIL_REASON (one line) and returns 1 on failure -- 404, network
# error, or a sha256 mismatch -- without printing; the caller reports it.
# On a 404 or network error, also sets PREBUILT_FAIL_DETAIL to a longer,
# case-specific explanation for main's explicit-rung fail() path.
ensure_prebuilt_downloaded() {
    cache_dir="$REVV_HOME/cache"
    mkdir -p "$cache_dir"
    fname=$(basename "$PREBUILT_URL")
    dest="$cache_dir/$fname"

    if [ -f "$dest" ]; then
        got=$(sha256_of "$dest" 2>/dev/null) || got=""
        if [ "$got" = "$PREBUILT_SHA256" ]; then
            echo "  reusing cached, verified download: $dest"
            PREBUILT_CACHE_FILE="$dest"
            return 0
        fi
        echo "  cached file at $dest is missing or does not match the expected sha256; re-downloading."
        rm -f "$dest"
    fi

    tmp="$dest.tmp.$$"
    echo "  downloading $PREBUILT_URL ..."
    if ! download_prebuilt "$PREBUILT_URL" "$tmp"; then
        rm -f "$tmp"
        case "$DOWNLOAD_HTTP_CODE" in
            404)
                PREBUILT_FAIL_REASON="the prebuilt release is not published yet (HTTP 404) -- this is expected until the first binary release is published"
                PREBUILT_FAIL_DETAIL="This is expected right now: the revv prebuilt release hasn't been
published yet. Use --source to build llama.cpp from source in the
meantime, or --upstream for the official (Vulkan) prebuilt."
                ;;
            000)
                PREBUILT_FAIL_REASON="could not reach $PREBUILT_URL (network error -- check connectivity, DNS, or a proxy)"
                PREBUILT_FAIL_DETAIL="This looks like a local network problem (no route, DNS failure, TLS
error, or timeout) rather than the release itself. Check connectivity
(and any proxy settings), then re-run; or use --source to build
llama.cpp from source instead."
                ;;
            *)
                PREBUILT_FAIL_REASON="download failed for $PREBUILT_URL (HTTP $DOWNLOAD_HTTP_CODE)"
                ;;
        esac
        return 1
    fi

    if ! verify_sha256_or_confirm "$tmp" "$PREBUILT_SHA256"; then
        rm -f "$tmp"
        PREBUILT_FAIL_REASON="sha256 verification failed for the downloaded prebuilt archive"
        return 1
    fi

    mv "$tmp" "$dest"
    PREBUILT_CACHE_FILE="$dest"
    return 0
}

# Checks that the dynamic loader can resolve every shared library the
# real binary needs -- the prebuilt's CUDA libs come from the system, not
# the archive, so this is the one part of "install" that a download alone
# can't guarantee. Never fails hard; reports and lets the user fix it.
check_cuda_runtime_libs() {
    bin_path="$1"
    if ! command -v ldd >/dev/null 2>&1; then
        echo "note: 'ldd' not found -- skipping the CUDA runtime library check"
        return 0
    fi
    ldd_out=$(ldd "$bin_path" 2>/dev/null) || ldd_out=""
    missing_count=$(printf '%s\n' "$ldd_out" | grep -c "not found" || true)
    if is_int "$missing_count" && [ "$missing_count" -gt 0 ]; then
        echo ""
        echo "warning: the dynamic loader cannot resolve these shared libraries at"
        echo "         runtime:"
        printf '%s\n' "$ldd_out" | grep "not found" | awk '{print "           " $1}'
        echo "         This means the CUDA *runtime* (not the full CUDA Toolkit) is"
        echo "         missing. Install just the runtime, e.g.:"
        echo "           sudo apt install cuda-cudart-12-6 libcublas-12-6"
        echo "         (far smaller than the full CUDA Toolkit that --source needs)."
    else
        echo "CUDA runtime libraries: OK (ldd reports nothing missing for $bin_path)"
    fi
}

# Rung 1. Returns 0 on success (installed + manifest written). Returns 1
# and sets PREBUILT_FAIL_REASON, having already printed "skip prebuilt:
# <reason>", if any precondition or the download/verify step fails --
# never calls fail() for those, so the auto-fallback path can fall
# through to --source cleanly.
try_prebuilt() {
    PREBUILT_FAIL_REASON=""
    PREBUILT_FAIL_DETAIL=""

    uname_s=$(uname -s)
    if [ "$uname_s" != "Linux" ]; then
        PREBUILT_FAIL_REASON="not running on Linux (uname -s reports '$uname_s')"
        echo "  skip prebuilt: $PREBUILT_FAIL_REASON"
        return 1
    fi

    uname_m=$(uname -m)
    if [ "$uname_m" != "x86_64" ]; then
        PREBUILT_FAIL_REASON="not running on x86_64 (uname -m reports '$uname_m')"
        echo "  skip prebuilt: $PREBUILT_FAIL_REASON"
        return 1
    fi

    if ! command -v ldd >/dev/null 2>&1; then
        PREBUILT_FAIL_REASON="could not determine the glibc version ('ldd' not found)"
        echo "  skip prebuilt: $PREBUILT_FAIL_REASON"
        return 1
    fi
    parse_glibc
    if ! is_int "$GLIBC_MAJOR" || ! is_int "$GLIBC_MINOR"; then
        PREBUILT_FAIL_REASON="could not parse a glibc version out of 'ldd --version'"
        echo "  skip prebuilt: $PREBUILT_FAIL_REASON"
        return 1
    fi
    if [ "$GLIBC_MAJOR" -lt 2 ] || { [ "$GLIBC_MAJOR" -eq 2 ] && [ "$GLIBC_MINOR" -lt 38 ]; }; then
        PREBUILT_FAIL_REASON="glibc $GLIBC_VER is older than the 2.38 minimum the revv prebuilt requires"
        echo "  skip prebuilt: $PREBUILT_FAIL_REASON"
        return 1
    fi
    echo "  platform OK for the revv prebuilt: Linux x86_64, glibc $GLIBC_VER"

    if ! check_prebuilt_arch; then
        echo "  skip prebuilt: $PREBUILT_FAIL_REASON"
        return 1
    fi

    if ! ensure_prebuilt_downloaded; then
        echo "  skip prebuilt: $PREBUILT_FAIL_REASON"
        return 1
    fi

    runtime_dir="$REVV_HOME/runtime/prebuilt"
    rm -rf "$runtime_dir"
    mkdir -p "$runtime_dir"
    if ! tar -xzf "$PREBUILT_CACHE_FILE" -C "$runtime_dir" --strip-components=1; then
        fail "failed to extract $PREBUILT_CACHE_FILE" \
"The sha256 checksum matched, so this looks like a local problem (e.g. no
'tar', or a full disk) rather than a bad download. Remove the cache file and
re-run to fetch it fresh:
  rm -f $PREBUILT_CACHE_FILE"
    fi
    chmod +x "$runtime_dir/llama-server" "$runtime_dir/llama-server.bin" 2>/dev/null || true
    if [ ! -x "$runtime_dir/llama-server" ]; then
        fail "extracted prebuilt archive has no llama-server launcher" \
"Looked for $runtime_dir/llama-server. Remove the cache file and re-run to
fetch it fresh:
  rm -f $PREBUILT_CACHE_FILE"
    fi

    mkdir -p "$BIN_DIR"
    rm -f "$BIN_DIR/llama-server"
    ln -s "$runtime_dir/llama-server" "$BIN_DIR/llama-server"
    echo "installed: $BIN_DIR/llama-server -> $runtime_dir/llama-server"

    if [ -x "$runtime_dir/llama-server.bin" ]; then
        check_cuda_runtime_libs "$runtime_dir/llama-server.bin"
    fi

    write_manifest "prebuilt" \
        '"mmvq_iquant_decode.patch", "pr26004-rebased-daef7b687.patch"' \
        "$PREBUILT_URL" ""
    return 0
}

# ---------------------------------------------------------------------------
# Upstream install (rung 2: official ggml-org/llama.cpp prebuilt)
# ---------------------------------------------------------------------------

# Rung 2, explicit only -- this script never auto-falls-back into it,
# because Vulkan is a different backend with different performance, not a
# strictly-worse-but-safe substitute for CUDA. Fails hard (not a soft
# "skip") on an unsupported platform, since the user asked for this rung
# by name.
do_install_upstream() {
    echo "installing the official upstream llama.cpp prebuilt (rung 2)..."
    echo "note: ggml-org/llama.cpp publishes no Linux CUDA prebuilt. On Linux"
    echo "x86_64 the official prebuilt is the Vulkan backend -- it runs on your"
    echo "NVIDIA GPU, but is a different backend with different performance;"
    echo "none of revv's published numbers describe it. Use --source for a CUDA"
    echo "build of the official upstream kernels, or --prebuilt (the default)"
    echo "for revv's patched CUDA build."
    echo ""

    uname_s=$(uname -s)
    if [ "$uname_s" != "Linux" ]; then
        fail "the upstream prebuilt this script knows how to install is Linux x86_64 only (uname -s reports '$uname_s')" \
"Use --source to build llama.cpp from source on this machine instead."
    fi
    uname_m=$(uname -m)
    if [ "$uname_m" != "x86_64" ]; then
        fail "the upstream prebuilt this script knows how to install is Linux x86_64 only (uname -m reports '$uname_m')" \
"Use --source to build llama.cpp from source on this machine instead."
    fi

    cache_dir="$REVV_HOME/cache"
    mkdir -p "$cache_dir"
    asset="llama-${PINNED_BUILD}-bin-ubuntu-vulkan-x64.tar.gz"
    url="https://github.com/ggml-org/llama.cpp/releases/download/${PINNED_BUILD}/${asset}"
    dest="$cache_dir/$asset"

    if [ -f "$dest" ]; then
        echo "reusing cached download: $dest"
    else
        tmp="$dest.tmp.$$"
        echo "downloading $url ..."
        if ! download_with_progress "$url" "$tmp"; then
            rm -f "$tmp"
            fail "download of the upstream prebuilt failed: $url" \
"Check network access, or use --source to build llama.cpp from source
instead."
        fi
        mv "$tmp" "$dest"
    fi
    echo "note: this is the official upstream release asset. It has no revv-"
    echo "pinned sha256 to check it against, so it is verified only by having"
    echo "come from the official GitHub release URL above, plus a check below"
    echo "that it actually extracts and contains llama-server."

    runtime_dir="$REVV_HOME/runtime/upstream"
    rm -rf "$runtime_dir"
    mkdir -p "$runtime_dir"
    if ! tar -xzf "$dest" -C "$runtime_dir" --strip-components=1; then
        fail "failed to extract $dest" \
"The download may be corrupt. Remove it and re-run to fetch it fresh:
  rm -f $dest"
    fi
    chmod +x "$runtime_dir/llama-server" 2>/dev/null || true
    if [ ! -x "$runtime_dir/llama-server" ]; then
        fail "extracted upstream archive does not contain an executable llama-server" \
"Looked for $runtime_dir/llama-server. Remove the cached download and
re-run to fetch it fresh:
  rm -f $dest"
    fi

    mkdir -p "$BIN_DIR"
    rm -f "$BIN_DIR/llama-server"
    ln -s "$runtime_dir/llama-server" "$BIN_DIR/llama-server"
    echo "installed: $BIN_DIR/llama-server -> $runtime_dir/llama-server"

    write_manifest "upstream" "" "$url" ',
  "backend": "vulkan"'
}

# ---------------------------------------------------------------------------
# revv CLI wrapper
# ---------------------------------------------------------------------------

install_revv_wrapper() {
    mkdir -p "$BIN_DIR"
    wrapper="$BIN_DIR/revv"
    cat > "$wrapper" <<WRAPPER_EOF
#!/bin/sh
exec python3 "$REVV_PY" "\$@"
WRAPPER_EOF
    chmod +x "$wrapper"
    echo "installed: $wrapper"
}

report_path() {
    case ":$PATH:" in
        *":$BIN_DIR:"*)
            echo "$BIN_DIR is already on PATH."
            ;;
        *)
            echo "Add $BIN_DIR to your PATH:"
            echo ""
            echo "    export PATH=\"$BIN_DIR:\$PATH\""
            echo ""
            echo "Add that line to your shell profile (~/.zshrc, ~/.bashrc, etc.)"
            echo "to make it permanent."
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

echo "revv installer"
echo "  source dir: $REVV_DIR"
echo "  REVV_HOME:   $REVV_HOME"
echo ""

mkdir -p "$BIN_DIR" "$SRC_DIR"

check_python

EXISTING=""
DID_INSTALL=0
if [ "$FORCE_BUILD" -ne 1 ] && EXISTING=$(find_existing_llama_server); then
    VER=$(get_llama_version "$EXISTING")
    echo ""
    echo "llama-server already available: $EXISTING"
    echo "  $VER"
    echo "skipping install (pass --force-build to reinstall)."
    DID_INSTALL=1
else
    if [ -z "$RUNG" ]; then
        echo ""
        echo "no install method given -- trying the default, --prebuilt, first."
        if try_prebuilt; then
            DID_INSTALL=1
        else
            echo "  falling back to --source."
            RUNG="source"
        fi
    fi

    if [ "$DID_INSTALL" -ne 1 ]; then
        case "$RUNG" in
            prebuilt)
                echo ""
                echo "installing llama-server (--prebuilt)..."
                if ! try_prebuilt; then
                    fail "prebuilt install failed: $PREBUILT_FAIL_REASON" \
"${PREBUILT_FAIL_DETAIL:-Use --source to build llama.cpp from source on this machine instead.}"
                fi
                ;;
            upstream)
                echo ""
                do_install_upstream
                ;;
            source)
                if [ -z "$SRC_MODE" ]; then
                    choose_mode
                fi
                echo ""
                echo "building llama-server (--source, $SRC_MODE)..."
                do_build
                ;;
        esac
    fi
fi

echo ""
install_revv_wrapper

echo ""
report_path

echo ""
echo "Next steps:"
echo "  revv doctor            check this machine and report what is possible"
echo "  revv get               download the certified model (~10.2 GiB)"
echo "  revv up                start the stack in the background on port 8080"
echo ""
echo "Already have a Qwen3.8 GGUF from ollama or LM Studio? Skip 'revv get':"
echo "  revv adopt             register it in place, no second download"
echo ""
echo "Then point your tool at http://127.0.0.1:8080/v1 and run 'revv compare'."
