#!/usr/bin/env python3
"""Behavior-snapshot harness for revv.py.

Prints a large, fully deterministic text snapshot of revv's planner/argv/
format behavior to stdout. Run this before and after a refactor; the two
outputs must be byte-identical.

    python3 tests/snapshot_argv.py > before.txt
    # ... refactor revv.py ...
    python3 tests/snapshot_argv.py > after.txt
    diff before.txt after.txt

Stdlib only, no test framework. Never modifies revv.py or any other file.
"""

import argparse
import contextlib
import io
import json
import math
import os
import shutil
import sys

# ---------------------------------------------------------------------------
# Determinism setup -- must happen before `import revv`.
# ---------------------------------------------------------------------------

REVV_HOME = "/tmp/revv-snapshot-home"
shutil.rmtree(REVV_HOME, ignore_errors=True)
os.makedirs(REVV_HOME, exist_ok=True)

os.environ["REVV_HOME"] = REVV_HOME
os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "dumb"
os.environ["COLUMNS"] = "80"

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)
import revv  # noqa: E402

revv._COLOR = False
revv._git_sha = lambda: None

FIXTURES = os.path.join(HERE, "fixtures")


# ---------------------------------------------------------------------------
# Small printing helpers
# ---------------------------------------------------------------------------

def section(name):
    print("\n===== SECTION: %s =====" % name)


def case(label):
    print("--- %s ---" % label)


def kv(label, value):
    print("%s = %r" % (label, value))


def scrub(text):
    """Replace machine/location-specific substrings with stable placeholders."""
    if text is None:
        return text
    return text.replace(REVV_HOME, "<home>").replace(REPO_ROOT, "<repo>")


# ---------------------------------------------------------------------------
# Fixtures / targets / drafters
# ---------------------------------------------------------------------------

FIXTURE_NAMES = sorted(
    ["gemma_like.gguf", "head_no_think.gguf", "qwen_like.gguf", "think_no_head.gguf"]
)
QWEN_LIKE_PATH = os.path.join(FIXTURES, "qwen_like.gguf")

TARGETS = []  # List[Tuple[str, GGUFInfo]]
for build_key in sorted(revv.BUILDS):
    info = revv.read_gguf(QWEN_LIKE_PATH)
    info.file_size = int(revv.BUILDS[build_key]["size"])
    info.path = os.path.join("/models", str(revv.BUILDS[build_key]["file"]))
    TARGETS.append(("build:%s" % build_key, info))
for fname in FIXTURE_NAMES:
    info = revv.read_gguf(os.path.join(FIXTURES, fname))
    TARGETS.append(("raw:%s" % fname, info))

# head_no_think.gguf HAS an MTP draft head (no thinking switch); think_no_head
# has NO draft head but DOES support thinking. Confirmed by reading the fixtures:
# revv.read_gguf(head_no_think.gguf).has_mtp_head == True
# revv.read_gguf(think_no_head.gguf).has_mtp_head == False
DRAFTERS = [
    ("none", None),
    ("mtp", revv.read_gguf(os.path.join(FIXTURES, "head_no_think.gguf"))),
    ("plain", revv.read_gguf(os.path.join(FIXTURES, "think_no_head.gguf"))),
]
MTP_DRAFTER = DRAFTERS[1][1]

EXE = "/opt/llama/llama-server"
PORT = 41414


# ---------------------------------------------------------------------------
# 1. env
# ---------------------------------------------------------------------------

def section_env():
    section("env")
    kv("__version__", revv.__version__)
    kv("sorted(BUILDS)", sorted(revv.BUILDS))
    kv("MODEL_LINES", revv.MODEL_LINES)
    kv("DEPRECATED_LINES", revv.DEPRECATED_LINES)
    kv("TIERS", revv.TIERS)
    kv("CONTEXT_LADDER", revv.CONTEXT_LADDER)
    kv("KV_MIB_PER_TOKEN", revv.KV_MIB_PER_TOKEN)
    kv("FOOTPRINT_BASE_MIB", revv.FOOTPRINT_BASE_MIB)
    kv("VRAM_MARGIN_MIB", revv.VRAM_MARGIN_MIB)
    kv("MEASURED_PEAK_MARGIN_MIB", revv.MEASURED_PEAK_MARGIN_MIB)
    kv("VRAM_MIN_FREE_MIB", revv.VRAM_MIN_FREE_MIB)

    case("all module-level ALL_CAPS constants")
    # Path-shaped constants: printed as their basename instead of the full
    # (machine-specific) path, per the harness spec.
    path_like_names = {
        "REVV_HOME", "MODELS_DIR", "BIN_DIR", "RUN_DIR", "RUN_FILE",
        "LOG_DIR", "BUILD_MANIFEST",
    }
    allowed_types = (str, int, float, tuple, list, dict, bool)
    for name in sorted(vars(revv)):
        if not name.isupper():
            continue
        value = vars(revv)[name]
        if not isinstance(value, allowed_types):
            continue
        if name in path_like_names and isinstance(value, str):
            print("%s = <path:%s>" % (name, os.path.basename(value)))
            continue
        try:
            text = repr(value)
        except Exception:
            print("%s = <unrepresentable>" % name)
            continue
        if " at 0x" in text:
            print("%s = <skipped: non-deterministic repr>" % name)
            continue
        text = scrub(text)
        print("%s = %s" % (name, text))


# ---------------------------------------------------------------------------
# 2. gguf
# ---------------------------------------------------------------------------

def section_gguf():
    section("gguf")
    for tlabel, info in TARGETS:
        case(tlabel)
        print("basename         = %r" % os.path.basename(info.path))
        print("file_size        = %r" % info.file_size)
        print("version          = %r" % info.version)
        print("arch             = %r" % info.arch)
        print("n_layer          = %r" % info.n_layer)
        print("n_embd           = %r" % info.n_embd)
        print("n_head           = %r" % info.n_head)
        print("n_head_kv        = %r" % info.n_head_kv)
        print("n_vocab          = %r" % info.n_vocab)
        print("n_ctx_train      = %r" % info.n_ctx_train)
        print("supports_thinking= %r" % info.supports_thinking)
        print("has_mtp_head     = %r" % info.has_mtp_head)
        print("dominant_quant   = %r" % info.dominant_quant)
        print("type_counts      = %r" % sorted(info.type_counts.items()))
        print("type_bytes       = %r" % sorted(info.type_bytes.items()))
        print("n_tensors        = %r" % info.n_tensors)
        print("data_offset      = %r" % info.data_offset)
        print("tensor_data_bytes= %r" % info.tensor_data_bytes)
        print("mtp_tensors      = %r" % info.mtp_tensors)

        print("identify_build       = %r" % revv.identify_build(info))
        print("is_certified_file    = %r" % revv.is_certified_file(info))
        print("has_measured_peak    = %r" % revv.has_measured_peak(info))
        print("vram_margin_for(0,None)      = %r"
              % revv.vram_margin_for(info, 0, None))
        print("vram_margin_for(100,None)    = %r"
              % revv.vram_margin_for(info, 100, None))
        print("vram_margin_for(0,mtp-draft) = %r"
              % revv.vram_margin_for(info, 0, MTP_DRAFTER))

        verdict, why = revv.classify(info, info.path)
        print("classify[0] (verdict) = %s" % verdict)
        print("classify[1] (why):")
        print(why)


# ---------------------------------------------------------------------------
# 3. peaks
# ---------------------------------------------------------------------------

KVS = ("f16", "q8_0", "q4_0")
CTXS = (4096, 8192, 16384, 32768, 65536)


def section_peaks():
    section("peaks")
    for tlabel, info in TARGETS:
        case("model_peak_mib target=%s" % tlabel)
        for k in KVS:
            print("kv_mib_per_token(%-5s) = %r" % (k, revv.kv_mib_per_token(info, k)))
            for c in CTXS:
                print("  model_peak_mib(ctx=%-6d kv=%-5s) = %r"
                      % (c, k, revv.model_peak_mib(info, c, k)))

    case("estimated_peak_mib (target-independent)")
    for c in revv.CONTEXT_LADDER:
        for k in KVS:
            print("estimated_peak_mib(ctx=%-6d kv=%-5s) = %r"
                  % (c, k, revv.estimated_peak_mib(c, k)))

    for dlabel, draft in DRAFTERS:
        case("draft_overhead_mib drafter=%s" % dlabel)
        for c in revv.CONTEXT_LADDER:
            for k in KVS:
                print("draft_overhead_mib(ctx=%-6d kv=%-5s) = %r"
                      % (c, k, revv.draft_overhead_mib(draft, c, k)))


# ---------------------------------------------------------------------------
# 4. tiers
# ---------------------------------------------------------------------------

def section_tiers():
    section("tiers")
    case("tier_for")
    for f in range(4000, 26001, 250):
        print("tier_for(%d) = %r" % (f, revv.tier_for(f)))

    case("plan_context")
    for free in (9000, 11000, 12044, 15000, 23000, 30000):
        for k in KVS:
            for pref in (8192, 16384, 32768, 65536):
                print("plan_context(free=%-6d kv=%-5s pref=%-6d) = %r"
                      % (free, k, pref, revv.plan_context(free, k, pref)))


# ---------------------------------------------------------------------------
# 5. plans
# ---------------------------------------------------------------------------

def section_plans():
    section("plans")
    # tier_for(11000) is None on this build's numbers (VRAM_MIN_FREE_MIB sits
    # above 11000), which would make plan_launch(info, None, ...) raise
    # KeyError. Adapted: fall back to "12gb", matching how test_planner.py's
    # WSL2 profile is actually exercised (tier="12gb" is passed explicitly
    # there; a WSL2 3060 is still a 12gb-tier card).
    hw_list = [
        ("REF_3060", revv.tier_for(12044) or "12gb", 12044),
        ("WSL2_3060", revv.tier_for(11000) or "12gb", 11000),
        ("forced-12gb", "12gb", None),
        ("forced-16gb", "16gb", None),
        ("forced-24gb", "24gb", None),
    ]
    ctx_args = (None, 8192, 32768)
    passthroughs = ((), ("--foo", "bar", "--n-gpu-layers", "5"))
    n_blocks = 0

    for tlabel, info in TARGETS:
        for hwlabel, tier, free_mib in hw_list:
            for ctx_arg in ctx_args:
                for dlabel, draft in DRAFTERS:
                    n_blocks += 1
                    plan = revv.plan_launch(info, tier, ctx_arg, free_mib, draft)
                    print("--- plan target=%s hw=%s tier=%s free=%s ctx_arg=%s "
                          "draft=%s ---"
                          % (tlabel, hwlabel, tier, free_mib, ctx_arg, dlabel))
                    print("ctx = %r" % plan.ctx)
                    print("kv = %r" % plan.kv)
                    print("use_spec = %r" % plan.use_spec)
                    print("thinking_off = %r" % plan.thinking_off)
                    print("estimated_peak = %r" % plan.estimated_peak)
                    if free_mib is not None and plan.estimated_peak is not None:
                        headroom = free_mib - plan.estimated_peak
                    else:
                        headroom = "n/a"
                    print("headroom = %r" % headroom)
                    print("draft_path = %r"
                          % (os.path.basename(plan.draft_path)
                             if plan.draft_path else None))
                    print("draft_spec_type = %r" % plan.draft_spec_type)
                    print("ctx_checkpoints = %r" % plan.ctx_checkpoints)
                    print("n_cpu_moe = %r" % plan.n_cpu_moe)
                    print("build_name = %r" % plan.build_name)
                    print("n_threads = %r" % plan.n_threads)
                    print("levers = %r" % plan.levers)
                    print("is_noop = %r" % plan.is_noop)
                    print("notes:")
                    for note in plan.notes:
                        print("    %s" % note)
                    print("mode_description(REVV) = %r"
                          % revv.mode_description(revv.MODE_REVV, plan))
                    print("mode_description(STOCK) = %r"
                          % revv.mode_description(revv.MODE_STOCK, plan))

                    argv_by_mode_empty = {}
                    for mode in (revv.MODE_REVV, revv.MODE_STOCK):
                        for passthrough in passthroughs:
                            argv = revv.build_server_argv(
                                EXE, "/models/target.gguf", plan, PORT, mode,
                                passthrough)
                            if not passthrough:
                                argv_by_mode_empty[mode] = argv
                            print("argv mode=%s passthrough=%r:" % (mode, passthrough))
                            print("  list  = %r" % argv)
                            print("  shell = %s"
                                  % " ".join(revv._shell_quote(a) for a in argv))
                    print("modes_identical = %r"
                          % (argv_by_mode_empty[revv.MODE_REVV]
                             == argv_by_mode_empty[revv.MODE_STOCK]))
    print("\nTOTAL plan blocks = %d" % n_blocks)


# ---------------------------------------------------------------------------
# 6. builds_and_lines
# ---------------------------------------------------------------------------

class _FakeHostRam(object):
    """Pin revv.host_ram_mib() for the duration of a block, then restore it."""

    def __init__(self, total):
        self.total = total

    def __enter__(self):
        self.saved = revv.host_ram_mib
        if self.total is None:
            revv.host_ram_mib = lambda: (None, None)
        else:
            total = self.total
            revv.host_ram_mib = lambda: (total, total)
        return self

    def __exit__(self, *exc):
        revv.host_ram_mib = self.saved
        return False


def section_builds_and_lines():
    section("builds_and_lines")

    case("stock_spec / stock_description")
    for build_name in sorted(revv.BUILDS) + [None, "no-such-build"]:
        print("stock_spec(%r) = %r" % (build_name, revv.stock_spec(build_name)))
        print("stock_description(%r) = %r"
              % (build_name, revv.stock_description(build_name)))

    case("bench_reference")
    for build_name in sorted(revv.BUILDS) + [None, "no-such-build"]:
        for patched in (True, False):
            print("bench_reference(%r, patched=%r) = %r"
                  % (build_name, patched, revv.bench_reference(build_name, patched)))

    case("hf_url (certified builds)")
    for build_name, spec in sorted(revv.BUILDS.items()):
        if spec.get("certified"):
            print("hf_url(%r) = %r"
                  % (build_name,
                     revv.hf_url(str(spec["file"]), str(spec["repo"]))))

    case("resolve_model_line")
    for name in ["moe", "dense", "  MoE  ", "speed", "flagship", "12gb",
                 "nonsense", ""]:
        print("resolve_model_line(%r):" % name)
        result = revv.resolve_model_line(name)
        print("  -> %r" % result)

    case("get_build_choice / default_build_for_host, by host RAM")
    for total in (None, 4096, 8192, 16384, 24576, 65536):
        with _FakeHostRam(total):
            print("host_ram total=%r" % total)
            print("  default_build_for_host() = %r" % (revv.default_build_for_host(),))
            for arg in [None, "moe", "dense", "speed", "flagship",
                        "12gb", "16gb", "24gb"]:
                print("  get_build_choice(%r) = %r"
                      % (arg, revv.get_build_choice(arg)))


# ---------------------------------------------------------------------------
# 7. registry
# ---------------------------------------------------------------------------

def section_registry():
    section("registry")

    case("_slugify")
    for label in ["qwen3:latest", "TheBloke/Foo/model-Q4",
                  "  ..weird__name!! ", "", "ALLCAPS"]:
        print("_slugify(%r) = %r" % (label, revv._slugify(label)))

    case("_unique_name")
    models = {}
    n1 = revv._unique_name("qwen3-latest", "/path/a.gguf", models)
    print("fresh base                     -> %r" % n1)
    models[n1] = {"path": "/path/a.gguf"}
    n2 = revv._unique_name("qwen3-latest", "/path/a.gguf", models)
    print("same base + same path          -> %r" % n2)
    n3 = revv._unique_name("qwen3-latest", "/path/b.gguf", models)
    print("same base + different path     -> %r" % n3)
    models[n3] = {"path": "/path/b.gguf"}
    c1 = revv._unique_name("foo", "/p1", models)
    models[c1] = {"path": "/p1"}
    c2 = revv._unique_name("foo", "/p2", models)
    models[c2] = {"path": "/p2"}
    c3 = revv._unique_name("foo", "/p3", models)
    models[c3] = {"path": "/p3"}
    print("collision chain of 3           -> %r" % [c1, c2, c3])

    case("save_registry / registry.json bytes")
    revv.save_registry(models)
    with open(revv.registry_path(), "rb") as fh:
        raw = fh.read()
    print(scrub(raw.decode("utf-8")))

    case("load_registry (sorted)")
    print(sorted(revv.load_registry().items()))

    case("registry_lookup")
    fixture_path = os.path.join(FIXTURES, "qwen_like.gguf")
    models["exists-fixture"] = {"path": fixture_path}
    models["missing-file"] = {"path": "/nonexistent/path/to/model.gguf"}
    revv.save_registry(models)
    print("registry_lookup(existing file) = %r"
          % scrub(revv.registry_lookup("exists-fixture") or ""))
    print("registry_lookup(missing path)  = %r"
          % revv.registry_lookup("missing-file"))
    print("registry_lookup(unregistered)  = %r"
          % revv.registry_lookup("totally-unregistered-name"))

    try:
        os.remove(revv.registry_path())
    except OSError:
        pass


# ---------------------------------------------------------------------------
# 8. formatters
# ---------------------------------------------------------------------------

LONG_TEXT = (
    "The quick brown fox jumps over the lazy dog while revv plans a context "
    "window that fits the free VRAM it was actually given, not the VRAM the "
    "card claims to have in total, because the difference between those two "
    "numbers is exactly what turns a certified configuration into a CUDA OOM."
)


def section_formatters():
    section("formatters")

    case("gib")
    for n in (0, 1024 ** 3, 3 * 1024 ** 3 + 500 * 1024 ** 2, -(1024 ** 3)):
        print("gib(%r) = %r" % (n, revv.gib(n)))

    case("mib")
    for n in (0, 1, 1024, 123456789, -500):
        print("mib(%r) = %r" % (n, revv.mib(n)))

    case("_human_size")
    for n in (0, 500, 1024, 1048576, 1073741824, 10 * 1024 ** 3):
        print("_human_size(%r) = %r" % (n, revv._human_size(n)))

    case("_format_eta")
    for s in (0, 59, 60, 3599, 3600, 90000, float("nan"), float("inf"), -5):
        print("_format_eta(%r) = %r" % (s, revv._format_eta(s)))

    case("_uptime")
    for s in (0, 59, 60, 3599, 3600, 90000):
        print("_uptime(%r) = %r" % (s, revv._uptime(s)))

    case("_shell_quote")
    for s in ["plain", "with space", "quote's here",
              "a/path-1.2_3:4=5,6+7%8@9", "semi;colon", "$(danger)", ""]:
        print("_shell_quote(%r) = %r" % (s, revv._shell_quote(s)))

    case("_wrap width=40")
    for line in revv._wrap(LONG_TEXT, 40):
        print("  %r" % line)
    case("_wrap width=66")
    for line in revv._wrap(LONG_TEXT, 66):
        print("  %r" % line)

    case("_sse_token_count")
    body_timings_usage = (
        b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":" there"}}],'
        b'"timings":{"predicted_per_second":37.9,"predicted_n":42}}\n\n'
        b'data: {"choices":[{"delta":{}}],"usage":{"completion_tokens":42}}\n\n'
        b'data: [DONE]\n\n'
    )
    body_content_only = (
        b'data: {"choices":[{"delta":{"content":"a"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"b"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"c"}}]}\n\n'
        b'data: [DONE]\n\n'
    )
    body_reasoning_draft = (
        b'data: {"choices":[{"delta":{"reasoning_content":"thinking..."}}]}\n\n'
        b'data: {"choices":[{"delta":{"reasoning_content":"more"}}],'
        b'"timings":{"draft_n":10,"draft_n_accepted":7}}\n\n'
        b'data: [DONE]\n\n'
    )
    for label, body in (("timings+usage", body_timings_usage),
                        ("content-only", body_content_only),
                        ("reasoning+draft", body_reasoning_draft)):
        print("_sse_token_count(%s) = %r" % (label, revv._sse_token_count(body)))

    case("server_supports / thinking_off_flags (EXE does not exist)")
    print("server_supports(EXE, '--reasoning') = %r"
          % revv.server_supports(EXE, "--reasoning"))
    print("thinking_off_flags(EXE) = %r" % revv.thinking_off_flags(EXE))

    case("mode_description(plan=None)")
    print("mode_description(REVV, None) = %r"
          % revv.mode_description(revv.MODE_REVV, None))
    print("mode_description(STOCK, None) = %r"
          % revv.mode_description(revv.MODE_STOCK, None))


# ---------------------------------------------------------------------------
# 9. inspect
# ---------------------------------------------------------------------------

def section_inspect():
    section("inspect")
    for fname in FIXTURE_NAMES:
        case(fname)
        path = os.path.join(FIXTURES, fname)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = revv.cmd_inspect(argparse.Namespace(file=path))
        print(scrub(buf.getvalue()))
        print("return code = %r" % rc)


# ---------------------------------------------------------------------------
# 10. doctor
# ---------------------------------------------------------------------------

def section_doctor():
    section("doctor")
    if shutil.which("nvidia-smi") is not None:
        print("SKIPPED: nvidia-smi present")
        return
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = revv.cmd_doctor(argparse.Namespace())
    print(scrub(buf.getvalue()))
    print("return code = %r" % rc)


# ---------------------------------------------------------------------------
# 11. cli_help
# ---------------------------------------------------------------------------

SUBCOMMANDS = ["doctor", "get", "inspect", "adopt", "serve", "up", "down",
               "status", "toggle", "compare", "bench", "update", "uninstall"]


def _capture_help(parser, argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        try:
            parser.parse_args(argv)
        except SystemExit:
            pass
    return buf.getvalue()


def section_cli_help():
    section("cli_help")
    parser = revv.build_parser()

    case("top-level --help")
    print(_capture_help(parser, ["--help"]))

    for cmd in SUBCOMMANDS:
        case("%s --help" % cmd)
        print(_capture_help(parser, [cmd, "--help"]))


# ---------------------------------------------------------------------------
# 12. host
# ---------------------------------------------------------------------------

def section_host():
    section("host")
    print("(machine-dependent; stable on one machine, which is all this "
          "snapshot needs)")
    print("physical_core_count() = %r" % revv.physical_core_count())
    print("host_ram_mib() = %r" % (revv.host_ram_mib(),))


# ---------------------------------------------------------------------------

def main():
    # Everything is captured into a buffer first and scrubbed as a whole pass
    # at the end, so any absolute path that leaks into argv (e.g. an external
    # drafter's GGUFInfo.path, which is built from this file's own location)
    # is normalized too, not just the handful of spots that scrub it locally.
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        section_env()
        section_gguf()
        section_peaks()
        section_tiers()
        section_plans()
        section_builds_and_lines()
        section_registry()
        section_formatters()
        section_inspect()
        section_doctor()
        section_cli_help()
        section_host()
        print("\n===== END =====")
    sys.stdout.write(scrub(buf.getvalue()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
