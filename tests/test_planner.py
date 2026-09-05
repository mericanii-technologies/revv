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
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import revv  # noqa: E402

FIXTURES = os.path.join(HERE, "fixtures")
CERT_SIZE = int(revv.BUILDS[revv.DEFAULT_BUILD]["size"])
CERT_NAME = str(revv.BUILDS[revv.DEFAULT_BUILD]["file"])

# An RTX 3060 12GB reports 12,288 MiB total, but 244 MiB of that is reserved
# by the driver and shows up in neither `used` nor `free` -- so `memory.free`
# on an otherwise-idle card maxes out at 12,288 - 244 = 12,044 MiB. A previous
# revision of this suite planned against a 12,287 MiB fixture, which is
# physically unreachable on real hardware; that unreachable fixture was green
# while the speed tier could not actually reach its certified c=16384 on any
# real card. REF_3060_FREE_MIB replaces it with the true idle ceiling.
REF_3060_FREE_MIB = 12044
# A 3060 under WSL2, with Windows itself holding onto roughly 1.2 GB of the
# card before revv ever launches.
WSL2_3060_FREE_MIB = 11000

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


class fake_host_ram(object):
    """Pin host_ram_mib() for the duration of a block, then restore it.

    The RAM-based default must be tested against chosen values, not against
    whatever the machine running the suite happens to have.
    """

    def __init__(self, total_mib, avail_mib=None):
        self.total = total_mib
        self.avail = avail_mib if avail_mib is not None else total_mib
        self.saved = None

    def __enter__(self):
        self.saved = revv.host_ram_mib
        revv.host_ram_mib = lambda: (self.total, self.avail)
        return self

    def __exit__(self, *exc):
        revv.host_ram_mib = self.saved
        return False


def test_build_names():
    """Build lines are named for what they are (moe / dense); the old names
    still work but are deprecated, because they were inverted against the
    editing measurement rather than merely old."""
    section("build names and deprecated aliases")

    check("moe -> the 35B MoE build",
          revv.resolve_model_line("moe"), "Q3_K_XL_35B")
    check("dense -> the 27B dense build",
          revv.resolve_model_line("dense"), "IQ3_XXS")
    check("names are case- and space-insensitive",
          revv.resolve_model_line("  MoE  "), "Q3_K_XL_35B")

    # The aliases are inverted relative to the old naming, which is the whole
    # reason they are deprecated rather than silently mapped.
    check("alias speed -> moe (the 35B)",
          revv.resolve_model_line("speed"), "Q3_K_XL_35B")
    check("alias flagship -> dense (the 27B)",
          revv.resolve_model_line("flagship"), "IQ3_XXS")
    check("every alias maps to a real line",
          all(new in revv.MODEL_LINES for new, _ in
              revv.DEPRECATED_LINES.values()), True)
    check("a non-line name is not a line", revv.resolve_model_line("12gb"), None)
    check("an unknown name is not a line",
          revv.resolve_model_line("nonsense"), None)

    # Both certified builds must be reachable by name. This is the regression
    # that mattered: `revv get speed` used to resolve the line correctly and
    # then have it overwritten by the tier default, so it fetched the dense
    # build while printing nothing about it.
    check("get speed fetches the MoE build, not the dense one",
          revv.get_build_choice("speed")[0], "Q3_K_XL_35B")
    check("get moe fetches the MoE build",
          revv.get_build_choice("moe")[0], "Q3_K_XL_35B")
    check("get dense fetches the dense build",
          revv.get_build_choice("dense")[0], "IQ3_XXS")
    check("an explicit line needs no explanation",
          revv.get_build_choice("moe")[1], None)

    check("the two lines cover both certified builds",
          sorted(revv.MODEL_LINES.values()),
          sorted(n for n, s in revv.BUILDS.items() if s["certified"]))
    check("every build is labelled with a live line name",
          sorted({str(s["line"]) for s in revv.BUILDS.values()}),
          sorted(revv.MODEL_LINES))


def test_ram_based_default():
    """With no build named, host RAM decides: it is the only requirement the
    two certified builds do not share. The MoE build streams 16 blocks of
    experts from host RAM and wants ~8-9 GiB free for them."""
    section("RAM-based default build")

    with fake_host_ram(32 * 1024):
        build, why = revv.default_build_for_host()
        check("32 GiB host -> MoE build", build, "Q3_K_XL_35B")
        check("...and it says why", "RAM" in (why or ""), True)
    with fake_host_ram(revv.MOE_DEFAULT_HOST_RAM_MIB):
        check("exactly at the threshold -> MoE build",
              revv.default_build_for_host()[0], "Q3_K_XL_35B")
    with fake_host_ram(revv.MOE_DEFAULT_HOST_RAM_MIB - 1):
        check("one MiB below the threshold -> dense build",
              revv.default_build_for_host()[0], "IQ3_XXS")
    with fake_host_ram(16 * 1024):
        build, why = revv.default_build_for_host()
        check("16 GiB host -> dense build", build, "IQ3_XXS")
        check("...and it says why", "RAM" in (why or ""), True)
    with fake_host_ram(8 * 1024):
        check("8 GiB host -> dense build",
              revv.default_build_for_host()[0], "IQ3_XXS")

    # Unknown RAM must fail toward the build with no host-RAM requirement.
    saved = revv.host_ram_mib
    try:
        revv.host_ram_mib = lambda: (None, None)
        build, why = revv.default_build_for_host()
        check("unreadable host RAM -> dense build", build, "IQ3_XXS")
        check("...and it says why", bool(why), True)
    finally:
        revv.host_ram_mib = saved

    # A VRAM tier does not pick a line: every tier ships the same weights.
    with fake_host_ram(32 * 1024):
        check("a VRAM tier still asks the host-RAM question",
              revv.get_build_choice("12gb")[0], "Q3_K_XL_35B")
        check("no argument asks the same question",
              revv.get_build_choice(None)[0], "Q3_K_XL_35B")
    with fake_host_ram(16 * 1024):
        check("...and answers it differently on a smaller host",
              revv.get_build_choice("24gb")[0], "IQ3_XXS")

    # The build revv calibrates against is not the build it defaults to.
    check("DEFAULT_BUILD is still the measured reference build",
          revv.DEFAULT_BUILD, "IQ3_XXS")
    check("...and is still what identify_build matches",
          revv.identify_build(certified_like()), revv.DEFAULT_BUILD)


def test_context_checkpoints():
    """Default 32 checkpoints/slot at ~150 MiB each are allocated lazily, so a
    config near the ceiling passes its health check and dies on request two
    with a cudaGraphInstantiate error naming neither memory nor checkpoints."""
    section("context checkpoints (-ctxcp)")
    info = certified_like()
    # The flagship now runs the n-gram+MTP chain too, so its own ~100 MiB has
    # to be in the boundary math or this test drifts by exactly that amount.
    base = revv.model_peak_mib(info, 16384, "q8_0") + revv.SPEC_NGRAM_CHAIN_MIB

    # Boundary is exact: at the threshold we leave the default alone.
    p = revv.plan_launch(info, "12gb", 16384, base + revv.CHECKPOINT_HEADROOM_MIB)
    check("headroom == threshold -> default left alone", p.ctx_checkpoints, None)
    p = revv.plan_launch(info, "12gb", 16384,
                         base + revv.CHECKPOINT_HEADROOM_MIB - 1)
    check("headroom one below threshold -> disabled", p.ctx_checkpoints, 0)

    # A clean 12GB card running the certified config IS near the ceiling.
    p = revv.plan_launch(info, "12gb", None, REF_3060_FREE_MIB)
    check("certified config on a clean 12GB card -> disabled",
          p.ctx_checkpoints, 0)
    check("...and it is named as a lever",
          any("checkpoint" in x for x in p.levers), True)
    check("...and the reason is in the notes",
          any("second request" in n for n in p.notes), True)

    # Plenty of room: do not touch a default we have no reason to touch.
    p = revv.plan_launch(info, "12gb", None, 24476)
    check("24GB card -> default left alone", p.ctx_checkpoints, None)

    # Never guess when the GPU is unknown (forced tier, no nvidia-smi): the
    # planner now refuses to leave checkpoints at their default and forces
    # them off instead, since it has no VRAM reading to know it is safe.
    p = revv.plan_launch(info, "12gb", None, None)
    check("free VRAM unknown -> checkpoints forced off", p.ctx_checkpoints, 0)

    # The flag must actually reach the command line, and only in revv mode.
    p = revv.plan_launch(info, "12gb", None, REF_3060_FREE_MIB)
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

    p = revv.plan_launch(info, "12gb", None, REF_3060_FREE_MIB)
    # On a real (not the old, unreachable 12,287 MiB) 12GB card, the
    # chain-aware boundary math (see test_flagship_ngram_chain) caps the
    # flagship at 12288, not 16384.
    check("certified on 12GB -> ctx 12288", p.ctx, 12288)
    check("certified on 12GB -> q8_0 (f16 does not fit)", p.kv, "q8_0")
    # 11746 is model_peak_mib(info, 12288, "q8_0") plus the chain's own
    # ~100 MiB, which now runs on the flagship too.
    check("certified on 12GB -> peak is ctx-12288 peak + chain",
          p.estimated_peak,
          revv.model_peak_mib(info, 12288, "q8_0") + revv.SPEC_NGRAM_CHAIN_MIB)

    # A WSL2-sized card forces a lower rung than the reference card above.
    # This value is unchanged by MEASURED_PEAK_MARGIN_MIB: the flagship runs
    # the n-gram chain, so its peak carries an estimated term and keeps the
    # wide 250 MiB margin (see vram_margin_for).
    p = revv.plan_launch(info, "12gb", None, 11744)
    check("WSL2-style 11744 free -> ctx reduced to 6144 (chain counted)",
          p.ctx, 6144)

    p = revv.plan_launch(info, "12gb", None, 24476)
    check("24GB -> f16 KV (the faster kernel, and it fits)", p.kv, "f16")

    # A small model on a roomy card must not be taxed.
    small = revv.read_gguf(os.path.join(FIXTURES, "gemma_like.gguf"))
    p = revv.plan_launch(small, "12gb", None, REF_3060_FREE_MIB)
    check("small model on 12GB -> f16 KV, not quantized", p.kv, "f16")

    # Explicit --ctx was once snapped UP to the smallest ladder rung, silently
    # handing out more context than asked for.
    for want in (1024, 2048, 8192):
        p = revv.plan_launch(info, "12gb", want, REF_3060_FREE_MIB)
        check("explicit --ctx %d honoured exactly" % want, p.ctx, want)

    # More free VRAM must never yield less context.
    ctxs = [revv.plan_launch(info, "12gb", None, f).ctx
            for f in (11744, REF_3060_FREE_MIB, 13000, 24476)]
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
        p = revv.plan_launch(info, "12gb", None, REF_3060_FREE_MIB)
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

    p = revv.plan_launch(info, "12gb", None, REF_3060_FREE_MIB)
    check("certified context survives planning", p.ctx, 16384)
    check("certified KV survives planning", p.kv, "q8_0")
    # This build was certified WITH the n-gram chain already running (11,832
    # MiB peak includes it), so the planner must not add SPEC_NGRAM_CHAIN_MIB
    # a second time -- that would needlessly shrink its context.
    check("peak is not double-counting the chain", p.estimated_peak, 11832)
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
    check("labelled as the moe line", verdict, "CERTIFIED (moe line)")

    # The speed tier's OWN settings (n-cpu-moe, host RAM notes) must not leak
    # onto the flagship. Its peak now carries the chain's ~100 MiB, same as
    # the speed tier -- that part is shared, not speed-tier-specific.
    f = certified_like()
    pf = revv.plan_launch(f, "12gb", None, REF_3060_FREE_MIB)
    # On a real 12GB card the flagship's own chain-aware boundary math caps
    # it at 12288 (see test_flagship_ngram_chain) -- still q8_0, still no
    # n_cpu_moe, since that lever belongs to the speed tier only.
    check("flagship plans ctx 12288 / q8_0, no n_cpu_moe",
          (pf.ctx, pf.kv, pf.n_cpu_moe), (12288, "q8_0", None))
    check("flagship peak is ctx-12288 peak + chain",
          pf.estimated_peak,
          revv.model_peak_mib(f, 12288, "q8_0") + revv.SPEC_NGRAM_CHAIN_MIB)


def test_thread_heuristic():
    """-t is a CPU-MoE-only lever: measured on a 3060+Ryzen 3600, -t 8 is
    +14.4% over the server default and the full logical core count LOSES
    5-15% to oversubscription on this bandwidth-bound decode. It must never
    appear on a build whose experts stay on the GPU."""
    section("thread heuristic (-t)")
    check("physical_core_count is clamped to [4, 8]",
          4 <= revv.physical_core_count() <= 8, True)

    speed = speed_like()
    p = revv.plan_launch(speed, "12gb", None, REF_3060_FREE_MIB)
    check("n_cpu_moe > 0 -> n_threads is set", p.n_threads is not None, True)
    check("n_threads is clamped to [4, 8]", 4 <= p.n_threads <= 8, True)
    argv = revv.build_server_argv("/x/llama-server", speed.path, p, 8080,
                                  revv.MODE_REVV, [])
    check("argv carries -t <n_threads>",
          "-t" in argv and argv[argv.index("-t") + 1] == str(p.n_threads),
          True)
    check("STOCK mode does not set -t",
          "-t" in revv.build_server_argv(
              "/x/llama-server", speed.path, p, 8080, revv.MODE_STOCK, []),
          False)

    # No CPU-MoE offload -> no thread flag, on either certified line.
    flagship = certified_like()
    pf = revv.plan_launch(flagship, "12gb", None, REF_3060_FREE_MIB)
    check("flagship (no n_cpu_moe) -> n_threads is None", pf.n_threads, None)
    argv_f = revv.build_server_argv("/x/llama-server", flagship.path, pf, 8080,
                                    revv.MODE_REVV, [])
    check("flagship argv never carries -t", "-t" in argv_f, False)


def test_speed_tier_drafter_stack():
    """The n-gram+MTP drafter chain is a strict addition over MTP alone
    (first-success-wins chain). Originally certified ONLY for the n_cpu_moe
    speed tier; certified 2026-09-05 on the flagship too (2.81-6.10x on
    editing workloads, 1.00x/inert on plain generation, byte-identical
    output on all 4 workloads) -- it now ships on every build that
    speculates through its own MTP head."""
    section("drafter chain (n-gram + MTP), both tiers")
    speed = speed_like()
    p = revv.plan_launch(speed, "12gb", None, REF_3060_FREE_MIB)
    argv = revv.build_server_argv("/x/llama-server", speed.path, p, 8080,
                                  revv.MODE_REVV, [])
    check("speed tier argv carries the n-gram+MTP chain",
          "--spec-type" in argv and
          argv[argv.index("--spec-type") + 1] == revv.SPEC_TYPE_CHAIN, True)
    check("speed tier argv carries --spec-ngram-simple-size-m 256",
          "--spec-ngram-simple-size-m" in argv and
          argv[argv.index("--spec-ngram-simple-size-m") + 1] == "256", True)

    # The flagship gets the identical chain now -- no longer speed-tier-only.
    flagship = certified_like()
    pf = revv.plan_launch(flagship, "12gb", None, REF_3060_FREE_MIB)
    argv_f = revv.build_server_argv("/x/llama-server", flagship.path, pf, 8080,
                                    revv.MODE_REVV, [])
    check("flagship argv carries the n-gram+MTP chain",
          "--spec-type" in argv_f and
          argv_f[argv_f.index("--spec-type") + 1] == revv.SPEC_TYPE_CHAIN, True)
    check("flagship argv carries --spec-ngram-simple-size-m 256",
          "--spec-ngram-simple-size-m" in argv_f and
          argv_f[argv_f.index("--spec-ngram-simple-size-m") + 1] == "256", True)
    check("flagship no longer ships plain draft-mtp alone",
          argv_f[argv_f.index("--spec-type") + 1] == revv.SPEC_TYPE, False)

    # An external drafter still takes precedence over the built-in chain, on
    # either line.
    drafter = revv.read_gguf(os.path.join(FIXTURES, "gemma_like.gguf"))
    drafter.file_size = 3_000_000_000
    pd = revv.plan_launch(speed, "12gb", 8192, 24476, drafter)
    argv_d = revv.build_server_argv("/x/llama-server", speed.path, pd, 8080,
                                    revv.MODE_REVV, [])
    check("external drafter overrides the n-gram+MTP chain",
          "--spec-ngram-simple-size-m" in argv_d, False)
    check("external drafter argv carries --spec-draft-model",
          "--spec-draft-model" in argv_d, True)

    pd_f = revv.plan_launch(flagship, "12gb", 8192, 24476, drafter)
    argv_df = revv.build_server_argv("/x/llama-server", flagship.path, pd_f,
                                     8080, revv.MODE_REVV, [])
    check("external drafter overrides the chain on the flagship too",
          "--spec-ngram-simple-size-m" in argv_df, False)


def test_flagship_ngram_chain():
    """Certified 2026-09-05: the n-gram+MTP chain, previously speed-tier
    only, now ships on the flagship. Editing workloads 40.3 -> 222.8 / 246.0
    / 113.3 t/s (2.81-6.10x); pure generation 35.17 -> 35.16 t/s (1.00x,
    inert -- the matcher just misses and falls through); outputs
    byte-identical to plain-MTP on all 4 workloads. It costs ~100 MiB of
    VRAM (measured 11,956 vs 11,854 MiB at c=16384), so the planner has to
    count it before sizing context: left uncounted, a 12GB card would be
    handed ctx=16384 with only ~330 MiB of headroom once the chain is
    actually running, below the ~400 MiB comfort line noted when this was
    certified. The shipped ceiling is ctx=12288."""
    section("flagship n-gram chain sizing (certified 2026-09-05)")
    info = certified_like()
    peak12 = revv.model_peak_mib(info, 12288, "q8_0")

    # A free-VRAM reading tight enough that 16384 no longer clears the
    # chain-inclusive margin, but loose enough that 12288 still leaves >=400
    # MiB of headroom -- i.e. a normal 12GB card, tighter than the pristine
    # 12,287 fixture used elsewhere in this file (which predates the chain
    # shipping on the flagship and still has slack to spare at 16384).
    free = peak12 + revv.SPEC_NGRAM_CHAIN_MIB + 420

    p = revv.plan_launch(info, "12gb", None, free)
    check("12GB reference: chain-aware sizing lands on 12288, not 16384",
          p.ctx, 12288)
    check("12GB reference: headroom stays >= 400 MiB",
          free - p.estimated_peak >= 400, True)

    argv = revv.build_server_argv("/x/llama-server", info.path, p, 8080,
                                  revv.MODE_REVV, [])
    check("chain flags present on the flagship plan",
          "--spec-type" in argv and
          argv[argv.index("--spec-type") + 1] == revv.SPEC_TYPE_CHAIN and
          "--spec-ngram-simple-size-m" in argv, True)

    # The speed tier's plan is unchanged: it was certified WITH the chain
    # already running (11,832 MiB peak includes it), so it must not pay the
    # cost a second time.
    speed = speed_like()
    ps = revv.plan_launch(speed, "12gb", None, REF_3060_FREE_MIB)
    check("speed tier: ctx unchanged at 16384", ps.ctx, 16384)
    check("speed tier: peak unchanged at 11832 (no double-counted chain)",
          ps.estimated_peak, 11832)


def test_speed_tier_is_the_mtp_build():
    """Two HuggingFace repos publish a file with this exact name. The plain
    -GGUF one has NO draft head (16,845,511,648 bytes, 733 tensors, 40 layers);
    without it speculation silently does not run and 55.9 t/s is unreachable.
    Guard the size and the repo so nobody 'simplifies' this back."""
    section("speed tier points at the MTP build")
    spec = revv.BUILDS["Q3_K_XL_35B"]
    check("repo is the MTP one", spec["repo"],
          "unsloth/Qwen3.6-35B-A3B-MTP-GGUF")
    check("size is the MTP build's", spec["size"], 17227569440)
    check("NOT the headless build's size", spec["size"] == 16845511648, False)


def test_real_hardware_fixtures():
    """The old 12,287 MiB fixture used across this file is physically
    unreachable: an RTX 3060 12GB tops out at 12,044 MiB free once the driver
    takes its 244 MiB, so a suite that only ever planned against 12,287 was
    green while the speed tier could not reach its certified c=16384 on any
    real card. These are the two real profiles that matter: an idle 3060, and
    a 3060 under WSL2 with Windows holding back roughly 1.2 GB more."""
    section("real-hardware VRAM fixtures")

    speed = speed_like()
    p = revv.plan_launch(speed, "12gb", None, REF_3060_FREE_MIB)
    check("reference 3060: speed tier reaches certified 16384", p.ctx, 16384)
    check("reference 3060: speed tier peak is the measured 11832",
          p.estimated_peak, 11832)
    check("reference 3060: speed tier still disables checkpoints",
          p.ctx_checkpoints, 0)

    flagship = certified_like()
    pf = revv.plan_launch(flagship, "12gb", None, REF_3060_FREE_MIB)
    check("reference 3060: flagship lands on 12288", pf.ctx, 12288)
    check("reference 3060: flagship still disables checkpoints",
          pf.ctx_checkpoints, 0)

    speed_wsl = speed_like()
    p_wsl = revv.plan_launch(speed_wsl, "12gb", None, WSL2_3060_FREE_MIB)
    check("WSL2 3060: speed tier steps down from 16384", p_wsl.ctx < 16384,
          True)

    flagship_wsl = certified_like()
    pf_wsl = revv.plan_launch(flagship_wsl, "12gb", None, WSL2_3060_FREE_MIB)
    check("WSL2 3060: flagship steps down from 16384", pf_wsl.ctx < 16384,
          True)

    check("measured-peak builds use the 150 MiB margin",
          revv.vram_margin_for(speed_like()), revv.MEASURED_PEAK_MARGIN_MIB)

    # Guard so nobody reintroduces the unreachable fixture value.
    check("the 12287 fixture is unreachable on real hardware",
          REF_3060_FREE_MIB < 12287, True)


def test_forced_tier_guard():
    """--tier forces a speed tier without ever reading nvidia-smi. With no
    free-VRAM reading, the planner has no basis to know a stack of lazily
    allocated context checkpoints is safe, so it must always disable them
    and say so, rather than leaving the (possibly unsafe) default alone."""
    section("forced tier (--tier, no VRAM reading)")

    pf = revv.plan_launch(certified_like(), "12gb", None, None)
    check("forced tier disables checkpoints (flagship)",
          pf.ctx_checkpoints, 0)

    ps = revv.plan_launch(speed_like(), "12gb", None, None)
    check("forced tier disables checkpoints (speed)",
          ps.ctx_checkpoints, 0)

    info = certified_like()
    p = revv.plan_launch(info, "12gb", None, None)
    argv = revv.build_server_argv("/x/llama-server", info.path, p, 8080,
                                  revv.MODE_REVV, [])
    check("forced tier argv carries -ctxcp 0",
          "-ctxcp" in argv and argv[argv.index("-ctxcp") + 1] == "0", True)

    check("forced tier warns that VRAM was not read",
          any("--tier was given" in n for n in p.notes), True)


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


def test_stock_is_per_build():
    """STOCK is not one config, it is a property of the build. `-ngl 99` on
    the MoE build is not a slow baseline, it is a CUDA OOM (llama-server asks
    for 15,499 MiB on a 12 GiB card), so STOCK for that build is the
    certified `-ngl 30 -c 16384` baseline instead of "every layer on the
    GPU"."""
    section("STOCK is defined per build")

    moe = speed_like()
    p_moe = revv.plan_launch(moe, "12gb", None, REF_3060_FREE_MIB)
    stock_argv = revv.build_server_argv("/x/llama-server", moe.path, p_moe,
                                        8080, revv.MODE_STOCK, [])
    check("MoE stock uses -ngl 30, not 99",
          stock_argv[stock_argv.index("-ngl") + 1], "30")
    check("MoE stock pins c=16384",
          stock_argv[stock_argv.index("-c") + 1], "16384")
    check("MoE stock does not offload experts",
          "--n-cpu-moe" in stock_argv, False)
    check("MoE stock does not speculate",
          "--spec-type" in stock_argv, False)
    check("MoE stock leaves KV at the default",
          "-ctk" in stock_argv, False)

    dense = certified_like()
    p_dense = revv.plan_launch(dense, "12gb", None, REF_3060_FREE_MIB)
    dense_stock_argv = revv.build_server_argv("/x/llama-server", dense.path,
                                              p_dense, 8080,
                                              revv.MODE_STOCK, [])
    check("dense stock still uses -ngl 99",
          dense_stock_argv[dense_stock_argv.index("-ngl") + 1], "99")
    check("dense stock uses the planned ctx",
          dense_stock_argv[dense_stock_argv.index("-c") + 1],
          str(p_dense.ctx))

    revv_argv = revv.build_server_argv("/x/llama-server", moe.path, p_moe,
                                       8080, revv.MODE_REVV, [])
    check("MoE revv mode still uses -ngl 99",
          revv_argv[revv_argv.index("-ngl") + 1], "99")
    check("MoE revv mode still offloads experts",
          "--n-cpu-moe" in revv_argv, True)

    check("stock_spec is None for the dense build",
          revv.stock_spec("IQ3_XXS"), None)
    check("stock_spec exists for the MoE build",
          revv.stock_spec("Q3_K_XL_35B") is not None, True)
    check("stock_description names the -ngl 30 fit",
          "-ngl 30" in revv.stock_description("Q3_K_XL_35B"), True)


def test_modes_are_not_identical():
    """compare refuses when the two arms would launch byte-identical argv --
    an A/B that never differs is not an A/B."""
    section("anti-fake A/B")

    for label, info in (("MoE", speed_like()), ("dense", certified_like())):
        p = revv.plan_launch(info, "12gb", None, REF_3060_FREE_MIB)
        stock_argv = revv.build_server_argv("/x/llama-server", info.path, p,
                                            8080, revv.MODE_STOCK, [])
        revv_argv = revv.build_server_argv("/x/llama-server", info.path, p,
                                           8080, revv.MODE_REVV, [])
        check("%s: stock and revv argv differ" % label,
              stock_argv == revv_argv, False)
        if label == "MoE":
            check("MoE arms differ in -ngl",
                  stock_argv[stock_argv.index("-ngl") + 1] !=
                  revv_argv[revv_argv.index("-ngl") + 1], True)


def test_bench_reference():
    """The old code graded every build against the dense reference, so a MoE
    result of 40.0 t/s (a 28% regression) scored ratio 40.0/36.94 = 1.08 and
    printed "on target". bench_reference() fixes this by keying the
    reference to the build actually loaded."""
    section("bench reference is per build")

    check("dense reference is the bench-protocol figure",
          round(revv.bench_reference("IQ3_XXS", True)[1], 2),
          round(revv.BENCH_REF_PATCHED, 2))
    check("MoE reference is its certified 55.9",
          revv.bench_reference("Q3_K_XL_35B", True)[1],
          revv.BUILDS["Q3_K_XL_35B"]["decode_ts"])
    check("unpatched target is 2.5% lower",
          round(revv.bench_reference("Q3_K_XL_35B", False)[0], 2),
          round(55.9 / 1.025, 2))
    check("an unregistered model falls back to dense",
          round(revv.bench_reference(None, True)[1], 2),
          round(revv.BENCH_REF_PATCHED, 2))
    check("the MoE reference names the MoE build",
          "moe" in revv.bench_reference("Q3_K_XL_35B", True)[2].lower(), True)

    # Regression guard for the actual bug: under the old dense-only
    # reference, a MoE result of 40.0 t/s (a 28% regression from 55.9)
    # scored 40.0 / (BENCH_REF_PATCHED / 1.025) = 1.08 and printed "on
    # target". The per-build reference below correctly fails it.
    moe_target = revv.bench_reference("Q3_K_XL_35B", False)[0]
    check("MoE at 55.94 is on target",
          0.95 <= 55.94 / moe_target <= 1.05, True)
    check("a MoE regression to 40 t/s is NOT on target",
          (40.0 / moe_target) < 0.95, True)
    # This documents the bug the fix closes: the old dense-only target would
    # have waved the same 40.0 t/s regression through.
    check("the old dense target would have passed it",
          (40.0 / (revv.BENCH_REF_PATCHED / 1.025)) >= 0.95, True)

    dense_target = revv.bench_reference("IQ3_XXS", False)[0]
    check("dense at 37.20 is on target",
          0.95 <= 37.20 / dense_target <= 1.05, True)


def test_failed_start_does_not_move_the_mode():
    """A start that never becomes healthy must leave the mode label alone.

    Backend.start() used to set self.mode before launching, so a mode switch
    that OOMed left the status endpoint reporting the mode that had just
    failed to load, with no process behind it -- `revv status` said STOCK
    while the backend was dead. Observed for real on the MoE build, whose
    stock arm could not fit before the per-build STOCK definition landed.
    """
    section("a failed start does not move the mode label")
    info = certified_like()
    plan = revv.plan_launch(info, "12gb", None, REF_3060_FREE_MIB)
    log = os.path.join(tempfile.mkdtemp(prefix="revv-qa-"), "llama-server.log")
    # /bin/false exits immediately, which is what a CUDA OOM at load looks
    # like to the supervisor: the process is gone before /health ever answers.
    backend = revv.Backend("/bin/false", "/m.gguf", "12gb", plan, [], log)
    check("backend starts out in revv mode", backend.mode, revv.MODE_REVV)

    raised = False
    try:
        backend.start(revv.MODE_STOCK, wait_s=5.0)
    except RuntimeError:
        raised = True
    check("a start that never gets healthy raises", raised, True)
    check("...and the mode label did NOT move to stock",
          backend.mode, revv.MODE_REVV)
    check("...and alive() reports the truth", backend.alive(), False)


def main():
    if not os.path.isdir(FIXTURES):
        print("fixtures missing: run python3 tests/make_fixtures.py first")
        return 2
    test_build_names()
    test_ram_based_default()
    test_context_checkpoints()
    test_kv_and_context()
    test_levers()
    test_certified_identity()
    test_drafter()
    test_speed_tier()
    test_speed_tier_is_the_mtp_build()
    test_thread_heuristic()
    test_speed_tier_drafter_stack()
    test_flagship_ngram_chain()
    test_real_hardware_fixtures()
    test_forced_tier_guard()
    test_port_fallback()
    test_stock_is_per_build()
    test_modes_are_not_identical()
    test_bench_reference()
    test_failed_start_does_not_move_the_mode()
    print("\n%s" % ("ALL PASSED" if not _failures
                    else "%d FAILED: %s" % (len(_failures), ", ".join(_failures))))
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
