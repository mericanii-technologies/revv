#!/bin/sh
# revv installer -- locates or builds llama-server, installs the revv CLI.
#
# POSIX sh only: no bashisms, no sudo, no external tools beyond git/cmake/
# python3 (and nvcc when a build is actually needed). Safe to run more than
# once -- completed work is detected and skipped.
set -e

PINNED_COMMIT="daef7b6874397a5a7c3d7e38b55e2ee0adf7da38"
PINNED_BUILD="b10712"
REVV_VERSION="1.0.0"
LLAMA_REPO_URL="https://github.com/ggml-org/llama.cpp.git"

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

Installs the revv CLI and locates or builds llama-server.

Options:
  --patched       Build llama.cpp with the revv performance and session-
                  restore patches applied. This is the recommended mode,
                  and the automatic default when running non-interactively.
  --stock         Build llama.cpp from the pinned commit with no patches
                  applied. Skips the +10% raw / +2.5% end-to-end kernel
                  gain and the 18x session-restore speedup; 'revv doctor'
                  will report the build as unpatched.
  --force-build   Build from source even if an llama-server is already
                  available in $REVV_HOME/bin or on PATH.
  --help          Show this help and exit.

Environment:
  REVV_HOME        Where revv stores its binaries, source checkout, and build
                  manifest. Defaults to $HOME/.revv.
  CUDAARCHS        Passed through verbatim as -DCMAKE_CUDA_ARCHITECTURES when
                  building llama.cpp with CUDA. Defaults to the compute
                  capability of the detected GPU(s), or 'native' if that
                  can't be determined.

If neither --patched nor --stock is given and a build is needed, this
script prompts when run from an interactive terminal, and otherwise
defaults to --patched.
EOF
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

MODE=""
FORCE_BUILD=0

while [ $# -gt 0 ]; do
    case "$1" in
        --patched)
            MODE="patched"
            ;;
        --stock)
            MODE="stock"
            ;;
        --force-build)
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
# Mode selection
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
                MODE="stock"
                ;;
            *)
                MODE="patched"
                ;;
        esac
    else
        MODE="patched"
        echo "non-interactive shell: defaulting to --patched"
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
# Build
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

write_manifest() {
    built_at=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    if [ "$MODE" = "patched" ]; then
        patch_list='"mmvq_iquant_decode.patch", "pr26004-rebased-daef7b687.patch"'
    else
        patch_list=''
    fi
    tmp_manifest="$BUILD_MANIFEST.tmp.$$"
    cat > "$tmp_manifest" <<MANIFEST_EOF
{
  "base_commit": "$PINNED_COMMIT",
  "patches": [$patch_list],
  "built_at": "$built_at",
  "revv_version": "$REVV_VERSION"
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

do_build() {
    check_build_tools
    setup_llama_src

    if [ "$MODE" = "patched" ]; then
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
    write_manifest
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
if [ "$FORCE_BUILD" -ne 1 ] && EXISTING=$(find_existing_llama_server); then
    VER=$(get_llama_version "$EXISTING")
    echo ""
    echo "llama-server already available: $EXISTING"
    echo "  $VER"
    echo "skipping build (pass --force-build to rebuild)."
else
    if [ -z "$MODE" ]; then
        choose_mode
    fi
    echo ""
    echo "building llama-server ($MODE)..."
    do_build
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
