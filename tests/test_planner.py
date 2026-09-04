#!/usr/bin/env python3
"""Planner regression tests.

    python3 tests/test_planner.py

Stdlib only, no test framework — revv has no dependencies and neither does its
test suite. Every case here corresponds to a bug that actually shipped or was
caught in review; the comment on each says which.

Needs the synthetic GGUF fixtures. Regenerate them with tests/make_fixtures.py
if tests/fixtures/ is missing.
"""

import os
import socket
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import revv  # noqa: E402

FIXTURES = os.path.join(HERE, "fixtures")
CERT_SIZE = int(revv.BUILDS[revv.DEFAULT_BUILD]["size"])
CERT_NAME = str(revv.BUILDS[revv.DEFAULT_BUILD]["file"])

_failures = []


def check(name, got, want):
    ok = got == want
    print("  %-58s %s" % (name, "ok" if ok else "FAIL got=%r want=%r"
                          % (got, want)))
    if not ok:
        _failures.append(name)


def certified_like():
    """The 27B: certified size and name, hybrid geometry, both levers."""
    info = revv.read_gguf(os.path.join(FIXTURES, "qwen_like.gguf"))
    info.file_size = CERT_SIZE
    info.path = os.path.join("/models", CERT_NAME)
    return info


def section(title):
    print("\n" + title)


def test_context_checkpoints():
    """Default 32 checkpoints/slot at ~150 MiB each are allocated lazily, so a
    config near the ceiling passes its health check and dies on request two
    with a cudaGraphInstantiate error naming neither memory nor checkpoints."""
    section("context checkpoints (-ctxcp)")
    info = certified_like()
    base = revv.model_peak_mib(info, 16384, "q8_0")

    # Boundary is exact: at the threshold we leave the default alone.
    p = revv.plan_launch(info, "12gb", 16384, base + revv.CHECKPOINT_HEADROOM_MIB)
    check("headroom == threshold -> default left alone", p.ctx_checkpoints, None)
    p = revv.plan_launch(info, "12gb", 16384,
                         base + revv.CHECKPOINT_HEADROOM_MIB - 1)
    check("headroom one below threshold -> disabled", p.ctx_checkpoints, 0)

    # A clean 12GB card running the certified config IS near the ceiling.
    p = revv.plan_launch(info, "12gb", None, 12287)
    check("certified config on a clean 12GB card -> disabled",
          p.ctx_checkpoints, 0)
    check("...and it is named as a lever",
          any("checkpoint" in x for x in p.levers), True)
    check("...and the reason is in the notes",
          any("second request" in n for n in p.notes), True)

    # Plenty of room: do not touch a default we have no reason to touch.
    p = revv.plan_launch(info, "12gb", None, 24476)
    check("24GB card -> default left alone", p.ctx_checkpoints, None)

    # Never guess when the GPU is unknown (forced tier, no nvidia-smi).
    p = revv.plan_launch(info, "12gb", None, None)
    check("free VRAM unknown -> default left alone", p.ctx_checkpoints, None)

    # The flag must actually reach the command line, and only in revv mode.
    p = revv.plan_launch(info, "12gb", None, 12287)
    argv = revv.build_server_argv("/x/llama-server", "/m.gguf", p, 8080,
                                  revv.MODE_REVV, [])
    check("argv carries -ctxcp 0", "-ctxcp" in argv and
          argv[argv.index("-ctxcp") + 1] == "0", True)
    argv = revv.build_server_argv("/x/llama-server", "/m.gguf", p, 8080,
                                  revv.MODE_STOCK, [])
    check("STOCK mode stays stock (no -ctxcp)", "-ctxcp" in argv, False)


def test_kv_and_context():
    """A Gemma-4-12B ran 2.5% SLOWER than stock because revv applied quantized
    KV to a model with room to spare; quantizing is a compute tax that only
    pays when it buys capacity."""
    section("KV precision and context sizing")
    info = certified_like()

    p = revv.plan_launch(info, "12gb", None, 12287)
    check("certified on 12GB -> ctx 16384", p.ctx, 16384)
    check("certified on 12GB -> q8_0 (f16 does not fit)", p.kv, "q8_0")
    check("certified on 12GB -> peak 11830", p.estimated_peak, 11830)

    p = revv.plan_launch(info, "12gb", None, 11744)
    check("WSL2-style 11744 free -> ctx reduced to 8192", p.ctx, 8192)

    p = revv.plan_launch(info, "12gb", None, 24476)
    check("24GB -> f16 KV (the faster kernel, and it fits)", p.kv, "f16")

    # A small model on a roomy card must not be taxed.
    small = revv.read_gguf(os.path.join(FIXTURES, "gemma_like.gguf"))
    p = revv.plan_launch(small, "12gb", None, 12287)
    check("small model on 12GB -> f16 KV, not quantized", p.kv, "f16")

    # Explicit --ctx was once snapped UP to the smallest ladder rung, silently
    # handing out more context than asked for.
    for want in (1024, 2048, 8192):
        p = revv.plan_launch(info, "12gb", want, 12287)
        check("explicit --ctx %d honoured exactly" % want, p.ctx, want)

    # More free VRAM must never yield less context.
    ctxs = [revv.plan_launch(info, "12gb", None, f).ctx
            for f in (11744, 12287, 13000, 24476)]
    check("context is monotonic in free VRAM", ctxs == sorted(ctxs), True)


def test_levers():
    """revv's levers are properties of the MODEL. Claiming them on a model that
    cannot use them is how the Gemma regression happened."""
    section("per-model levers")
    cases = {
        "gemma_like": (False, False, True),    # no head, no thinking -> no-op
        "qwen_like": (True, True, False),
        "head_no_think": (True, False, False),
        "think_no_head": (False, True, False),
    }
    for name, (spec, think, noop) in cases.items():
        info = revv.read_gguf(os.path.join(FIXTURES, name + ".gguf"))
        p = revv.plan_launch(info, "12gb", None, 12287)
        check("%s: speculation" % name, p.use_spec, spec)
        check("%s: thinking off" % name, p.thinking_off, think)
        check("%s: is_noop" % name, p.is_noop, noop)


def test_certified_identity():
    """revv adopt registers ollama blobs whose filename is a content hash. When
    identity was matched on name alone the certified model fell through to the
    geometric KV estimate and lost two-thirds of its context."""
    section("certified-model identity")
    info = certified_like()
    by_name = revv.plan_launch(info, "12gb", None, 11744)
    info.path = "/home/u/.ollama/models/blobs/sha256-c0b7c3038681ed"
    by_size = revv.plan_launch(info, "12gb", None, 11744)
    check("adopted blob plans same context as the named file",
          by_size.ctx, by_name.ctx)
    check("adopted blob plans same KV as the named file",
          by_size.kv, by_name.kv)
    check("is_certified_file matches on size alone",
          revv.is_certified_file(info), True)


def test_drafter():
    """An external drafter's weights and KV have to be charged against free
    VRAM, or adding one silently pushes the plan into an OOM."""
    section("external drafter")
    info = certified_like()
    drafter = revv.read_gguf(os.path.join(FIXTURES, "gemma_like.gguf"))
    drafter.file_size = 3_000_000_000

    without = revv.plan_launch(info, "12gb", 8192, 24476)
    with_d = revv.plan_launch(info, "12gb", 8192, 24476, drafter)
    check("drafter increases the estimated peak",
          with_d.estimated_peak > without.estimated_peak, True)
    check("no-head drafter selects draft-simple",
          with_d.draft_spec_type, "draft-simple")

    mtp_drafter = revv.read_gguf(os.path.join(FIXTURES, "qwen_like.gguf"))
    p = revv.plan_launch(info, "12gb", 8192, 24476, mtp_drafter)
    check("MTP-head drafter selects draft-mtp", p.draft_spec_type, "draft-mtp")


def speed_like():
    """The 35B MoE speed line, identified by its exact size."""
    info = revv.read_gguf(os.path.join(FIXTURES, "qwen_like.gguf"))
    info.file_size = int(revv.BUILDS["Q3_K_XL_35B"]["size"])
    info.path = "/models/" + str(revv.BUILDS["Q3_K_XL_35B"]["file"])
    return info


def test_speed_tier():
    """The MoE line keeps most of its weights in HOST RAM, so estimating its
    VRAM from file size over-counts by gigabytes and collapses the context."""
    section("speed tier (35B MoE)")
    info = speed_like()
    check("identified by size", revv.identify_build(info), "Q3_K_XL_35B")
    check("peak anchored on the measurement, not the file size",
          revv.model_peak_mib(info, 16384, "q8_0"), 11832)

    p = revv.plan_launch(info, "12gb", None, 12287)
    check("certified context survives planning", p.ctx, 16384)
    check("certified KV survives planning", p.kv, "q8_0")
    check("expert offload is applied", p.n_cpu_moe, 16)
    argv = revv.build_server_argv("/x/llama-server", info.path, p, 8080,
                                  revv.MODE_REVV, [])
    check("argv carries --n-cpu-moe 16",
          "--n-cpu-moe" in argv and argv[argv.index("--n-cpu-moe") + 1] == "16",
          True)
    check("STOCK mode does not offload experts",
          "--n-cpu-moe" in revv.build_server_argv(
              "/x/llama-server", info.path, p, 8080, revv.MODE_STOCK, []),
          False)
    check("host-RAM requirement is mentioned",
          any("host RAM" in n or "host RAM" in n.lower() for n in p.notes), True)

    verdict, _ = revv.classify(info, info.path)
    check("labelled as the speed line", verdict, "CERTIFIED (speed line)")

    # The flagship must be untouched by any of this.
    f = certified_like()
    pf = revv.plan_launch(f, "12gb", None, 12287)
    check("flagship still plans ctx 16384 / q8_0 / 11830 peak",
          (pf.ctx, pf.kv, pf.estimated_peak, pf.n_cpu_moe),
          (16384, "q8_0", 11830, None))


def test_speed_tier_is_the_mtp_build():
    """Two HuggingFace repos publish a file with this exact name. The plain
    -GGUF one has NO draft head (16,845,511,648 bytes, 733 tensors, 40 layers);
    without it speculation silently does not run and 48.5 t/s is unreachable.
    Guard the size and the repo so nobody 'simplifies' this back."""
    section("speed tier points at the MTP build")
    spec = revv.BUILDS["Q3_K_XL_35B"]
    check("repo is the MTP one", spec["repo"],
          "unsloth/Qwen3.6-35B-A3B-MTP-GGUF")
    check("size is the MTP build's", spec["size"], 17227569440)
    check("NOT the headless build's size", spec["size"] == 16845511648, False)


def test_port_fallback():
    """A real WSL2 box had an unrelated service squatting 8080."""
    section("port selection")
    sock = socket.socket()
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    busy = sock.getsockname()[1]
    sock.listen(1)
    try:
        check("a busy port reads as busy",
              revv.port_is_free("127.0.0.1", busy), False)
        # Explicit port is exact-or-fail: it must NOT silently move.
        try:
            revv.resolve_port("127.0.0.1", busy)
            moved = True
        except SystemExit:
            moved = False
        check("explicit --port on a busy port fails rather than moving",
              moved, False)
        free = revv.resolve_port("127.0.0.1", None)
        check("auto selection returns a bindable port",
              revv.port_is_free("127.0.0.1", free), True)
    finally:
        sock.close()


def main():
    if not os.path.isdir(FIXTURES):
        print("fixtures missing: run python3 tests/make_fixtures.py first")
        return 2
    test_context_checkpoints()
    test_kv_and_context()
    test_levers()
    test_certified_identity()
    test_drafter()
    test_speed_tier()
    test_speed_tier_is_the_mtp_build()
    test_port_fallback()
    print("\n%s" % ("ALL PASSED" if not _failures
                    else "%d FAILED: %s" % (len(_failures), ", ".join(_failures))))
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
