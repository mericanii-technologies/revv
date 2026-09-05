#!/usr/bin/env python3
"""revv -- by Mericanii.

Runs Qwen3.8-27B GGUFs on consumer NVIDIA GPUs at a measured, published
configuration. Every number this tool prints comes from BENCHMARKS.md.

Python 3.9+, standard library only, no external dependencies. That is a
hard constraint: revv has to run on a fresh box before anything is installed.
"""

import argparse
import dataclasses
import http.client
import json
import math
import os
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import (Any, BinaryIO, Callable, Dict, List, NamedTuple, Optional,
                    Sequence, Tuple)
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

__version__ = "1.0.0"


def _git_sha() -> Optional[str]:
    """Short SHA of the revv checkout, when revv is running from a git tree."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        out = subprocess.run(["git", "-C", here, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else None


def version_string() -> str:
    sha = _git_sha()
    return "revv %s (%s)" % (__version__, sha) if sha else "revv %s" % __version__

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REVV_HOME = os.environ.get("REVV_HOME") or os.path.join(
    os.path.expanduser("~"), ".revv")
MODELS_DIR = os.path.join(REVV_HOME, "models")
BIN_DIR = os.path.join(REVV_HOME, "bin")
BUILD_MANIFEST = os.path.join(REVV_HOME, "build.json")

# ---------------------------------------------------------------------------
# The certified configuration.
#
# Measured on: RTX 3060 12GB (sm_86, driver 535.309.01), Ryzen 5 3600, DDR4,
# Ubuntu 24.04, headless. Protocol: thinking off, greedy, 400 new tokens,
# 1 discarded warmup + 4 measured requests, decode rate from llama-server's
# own timings. Do not edit without a new measurement in BENCHMARKS.md.
#
# The two speed figures come from two different measurement sessions and are
# both real; revv reports whichever matches the build actually installed.
# ---------------------------------------------------------------------------

CERT_TS_PATCHED = 36.7    # kernel-patched, MTP path A/B (stock arm 35.8 same session)
CERT_TS_STOCK = 34.39     # upstream llama.cpp, shipping row of the re-certification
CERT_TS_NOSPEC = 20.0     # same weights, speculation off -- the raw floor
CERT_HUMANEVAL = 92.7     # HumanEval-164, thinking off, greedy
CERT_PEAK_MIB = 11958     # peak DURING requests, not after load. See BENCHMARKS.md:
                          # three configs pass a load-time check then OOM mid-request.
CERT_ACCEPT = 0.781       # MTP draft acceptance, shipping config, novel prompt

# The standing harness canary: under the corrected protocol a HumanEval task
# averages 158.8 completion tokens. Above ~350 means the model is still
# emitting reasoning and the thinking switch is not taking effect -- which is
# exactly the bug that made every pre-2026-09-02 quality number wrong.
THINK_LEAK_TOKENS = 350

# The headline figure, used wherever "what revv delivers" is meant.
CERT_TS = CERT_TS_PATCHED

# What `revv bench` itself measures on the reference box.
#
# These are NOT the same numbers as CERT_* above and must not be compared with
# them. The certification harness and the bench harness use different prompts,
# which changes MTP acceptance and therefore decode rate: on one box, in one
# session, the same flagship build reads 37.86 t/s under this protocol and
# 34.39 under the certification protocol. Bench compares your machine against
# the figure measured with the protocol bench actually runs, otherwise every
# result would read a few percent high.
BENCH_REF_PATCHED = 37.86     # flagship + kernel patch, RTX 3060, revv bench
BENCH_REF_NOSPEC = 22.5       # same weights, speculation off (revv compare, STOCK)
BENCH_PEAK_MIB = 11830        # peak during requests for the flags revv launches

# v1.1 candidate: the ASCII-vocab-pruned flagship on the same merged build.
# Faster and roomier, and byte-identical to the certified baseline on a 25-task
# HumanEval spot-check -- but 25 tasks is not a certification, so it is not the
# default and is not quoted as a quality result.
V11_TS = 40.10
V11_PEAK_MIB = 11502

# A 12GB card has ~12044 MiB usable. The certified config peaks at 11958.
# That is 86 MiB of headroom, so an X server or a stray process is the
# difference between "works" and "CUDA out of memory".
VRAM_IDLE_WARN_MIB = 250

# ---------------------------------------------------------------------------
# Fitting the model to the VRAM that is actually available.
#
# Total VRAM is the wrong number to plan against. Windows/WSL2 reserves roughly
# 1-1.5 GB of a card for the desktop compositor, so a "12GB" card there offers
# noticeably less than 12 GB to CUDA, and the certified c=16384 config OOMs on
# hardware that a total-VRAM check waves through. revv plans against
# nvidia-smi's memory.free instead.
#
# Cost model:  peak(ctx, kv) = FOOTPRINT_BASE_MIB + ctx * KV_MIB_PER_TOKEN[kv]
# FOOTPRINT_BASE_MIB is weights plus compute buffers, back-solved from the
# measured 11,830 MiB peak at c=16384 with q8_0 KV -- the flags revv actually
# launches, on an RTX 3060. The KV rate comes from the measured ~368 MiB per
# 16K tokens at q4_0 (~23 KiB/token), doubled for q8_0 and again for f16.
#
# Corroborated by a fresh WSL2 install: this predicts 11,462 MiB at c=8192, and
# that machine reported 12,006 MiB in use with ~544 MiB reserved by Windows.
# ---------------------------------------------------------------------------

KV_MIB_PER_TOKEN = {"q4_0": 0.0225, "q8_0": 0.0449, "f16": 0.0898}
FOOTPRINT_BASE_MIB = 11830 - 16384 * KV_MIB_PER_TOKEN["q8_0"]

# Descending. Below 4096 the model is too cramped to be useful for code work.
CONTEXT_LADDER = [65536, 49152, 32768, 24576, 16384, 12288, 8192, 6144, 4096]

# Headroom left unclaimed: CUDA allocators fragment, and the 12GB tier has
# historically sat 86 MiB from the ceiling.
VRAM_MARGIN_MIB = 250

# The same headroom, for a build whose peak is a MEASUREMENT rather than an
# estimate. 250 MiB exists to cover the geometric estimator's error; a build in
# BUILDS with a `peak_mib` has no estimator error to cover, because the number
# came off this hardware with the allocator's real fragmentation already in it.
#
# Applying the estimator's margin to a measured peak excluded the certified
# configuration from its own planner. An RTX 3060 12GB reports 12,288 MiB total
# but only 12,044 MiB free -- 244 MiB is reserved by the driver and appears in
# neither `used` nor `free`. The speed tier's measured peak is 11,832 MiB at
# c=16384, so 11,832 + 250 = 12,082 > 12,044 and the ladder stepped down to
# c=12288 on every 12GB card in existence, while tests/test_planner.py asserted
# the certified 16384 against a free_mib=12287 fixture that no such card can
# ever report.
#
# Certification for the 150: 11,832 MiB peak at c=16384 with -ctxcp 0, verified
# across consecutive deep requests -- i.e. past the request-two checkpoint
# allocation that kills configs which only pass a load-time check. Real headroom
# 212 MiB against the 12,044 usable ceiling.
MEASURED_PEAK_MARGIN_MIB = 150


def has_measured_peak(info: "GGUFInfo") -> bool:
    """Is this file's peak a registry MEASUREMENT rather than an estimate?

    Decides which of the two margins above applies. Mirrors the condition
    model_peak_mib() uses to anchor on `peak_mib`, so the two never disagree
    about which arm a given file is on.
    """
    build_name = identify_build(info)
    spec = BUILDS.get(build_name) if build_name else None
    return bool(spec is not None and spec.get("peak_mib"))


def vram_margin_for(info: "GGUFInfo", chain_mib: int = 0,
                    draft: Optional["GGUFInfo"] = None) -> int:
    """Headroom to leave unclaimed for this file.

    The reduced margin is for a peak that is PURELY a measurement. The moment
    an estimated term is added on top of the measured `peak_mib` -- the n-gram
    chain's ~100 MiB, or an external drafter's weights plus its KV -- the total
    is part measurement, part estimate, and that estimate is exactly what the
    250 MiB exists to cover. So a mixed sum falls back to the wide margin.

    This is not hypothetical. The flagship's measured peak is 11,830 MiB, and
    with the chain running the planner charges it 11,930. Granting that sum the
    narrow margin would let the flagship take c=16384 with as little as 150 MiB
    of headroom on a card reporting ~12,080 MiB free, undoing the chain-aware
    step-down to c=12288 that was certified on 2026-09-05 for exactly this
    reason. The speed tier is unaffected: its 11,832 MiB was measured WITH the
    chain already running, so chain_mib is 0 there and its peak stays purely
    measured.
    """
    if has_measured_peak(info) and chain_mib == 0 and draft is None:
        return MEASURED_PEAK_MARGIN_MIB
    return VRAM_MARGIN_MIB


# llama-server keeps up to 32 context checkpoints PER SLOT by default
# (upstream PR #15293), and on this model each one is ~150 MiB. They are
# allocated lazily, so the first one lands during the SECOND request -- after
# the health check has already passed. The server therefore starts fine, serves
# one request, and dies on the next with a cudaGraphInstantiate error that
# names neither memory nor checkpoints. Any config running close to the ceiling
# must turn them off. Measured: configs near the ceiling "required -ctxcp 0 to
# survive at all".
#
# INTERACTION, for whoever wires up session save/restore later: the restore
# patch depends on these checkpoints (its measured run used -ctxcp 8), so a
# config that disables them cannot also offer restore. No conflict today
# because revv v1.0 does not expose save/restore, but the two features are
# mutually exclusive at the VRAM ceiling and something has to give.
CHECKPOINT_MIB_EACH = 150
CHECKPOINT_HEADROOM_MIB = 500


def host_ram_mib() -> Tuple[Optional[int], Optional[int]]:
    """(total, available) host RAM in MiB, or (None, None) if unknown.

    A new dimension for MoE tiers: --n-cpu-moe streams expert weights from host
    RAM, so a config can fit VRAM perfectly and still thrash or get OOM-killed
    on a machine with too little RAM. MemAvailable is the right field -- it
    accounts for reclaimable page cache, which MemFree does not.
    """
    try:
        with open("/proc/meminfo", "r") as fh:
            fields = {}
            for line in fh:
                parts = line.split(":")
                if len(parts) == 2:
                    fields[parts[0]] = parts[1].strip().split()[0]
        total = int(fields["MemTotal"]) // 1024
        avail = int(fields.get("MemAvailable", fields["MemFree"])) // 1024
        return total, avail
    except (OSError, KeyError, ValueError, IndexError):
        pass
    # macOS fallback: total only, so revv can still say something on a dev box.
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip().isdigit():
            return int(out.stdout.strip()) // (1024 * 1024), None
    except (OSError, subprocess.SubprocessError):
        pass
    return None, None


def physical_core_count() -> int:
    """Physical (not logical/hyperthreaded) core count, clamped to [4, 8].

    Measured on a 3060 + Ryzen 3600 (6 physical / 12 logical): -t 8 is +14.4%
    over the server's default thread count, but the full logical count (12)
    LOSES 5-15% -- llama.cpp oversubscribes the physical cores on a
    bandwidth-bound decode, and SMT siblings just fight over the same memory
    bus. Output was verified bit-identical across -t 3..12, so this is a pure
    speed lever with no quality risk. /proc/cpuinfo's unique (physical id,
    core id) pairs is the one field that survives SMT; os.cpu_count() // 2 is
    the fallback when /proc is unavailable (e.g. macOS dev boxes), and 6 --
    the value this was measured on -- is the last resort.
    """
    try:
        pairs = set()
        physical_id = None
        core_id = None
        with open("/proc/cpuinfo", "r") as fh:
            for line in fh:
                if line.startswith("physical id"):
                    physical_id = line.split(":", 1)[1].strip()
                elif line.startswith("core id"):
                    core_id = line.split(":", 1)[1].strip()
                    if physical_id is not None:
                        pairs.add((physical_id, core_id))
        if pairs:
            return max(4, min(8, len(pairs)))
    except (OSError, ValueError, IndexError):
        pass
    logical = os.cpu_count()
    if logical:
        return max(4, min(8, logical // 2))
    return 6


def estimated_peak_mib(ctx: int, kv: str) -> int:
    return int(round(FOOTPRINT_BASE_MIB
                     + ctx * KV_MIB_PER_TOKEN.get(kv, KV_MIB_PER_TOKEN["q8_0"])))


def plan_context(free_mib: int, kv: str, preferred_ctx: int
                 ) -> Optional[Tuple[int, int]]:
    """Largest ladder context <= preferred that fits, with margin.

    None means even the smallest rung does not fit, i.e. this card cannot run
    the model. Saying so up front is kinder than a CUDA OOM three minutes into
    loading weights.
    """
    for ctx in CONTEXT_LADDER:
        if ctx > preferred_ctx:
            continue
        peak = estimated_peak_mib(ctx, kv)
        if peak + VRAM_MARGIN_MIB <= free_mib:
            return ctx, peak
    return None


# Bytes per KV element by cache precision. q8_0 is 34 bytes per 32 elements,
# q4_0 is 18 per 32.
KV_BYTES_PER_ELEM = {"f16": 2.0, "q8_0": 34.0 / 32.0, "q4_0": 18.0 / 32.0}

# Weights aside, a loaded server costs this much in compute buffers and
# scratch. Back-solved from the certified model: 11,094 base - 10,428 of
# weights. Roughly constant across models of this class.
COMPUTE_OVERHEAD_MIB = 666


def identify_build(info: "GGUFInfo") -> Optional[str]:
    """Which registry build is this file, by exact size then by name.

    Size first because `revv adopt` registers ollama blobs whose filename is a
    content hash; matching on name alone silently demotes a certified file.
    """
    for name, spec in BUILDS.items():
        if info.file_size == spec["size"]:
            return name
    base = os.path.basename(info.path)
    for name, spec in BUILDS.items():
        if base == spec["file"]:
            return name
    return None


def is_certified_file(info: "GGUFInfo") -> bool:
    """Is this the exact file the measured numbers came from?

    Matched on size as well as name, because `revv adopt` registers ollama
    blobs whose filename is a content hash. Name-only matching silently
    demoted the certified model to the geometric KV estimate, which
    over-estimates threefold on this hybrid architecture and cost the user
    two-thirds of their context for no reason.
    """
    if info.file_size == BUILDS[DEFAULT_BUILD]["size"]:
        return True
    return os.path.basename(info.path) == str(BUILDS[DEFAULT_BUILD]["file"])


def kv_mib_per_token(info: "GGUFInfo", kv: str) -> Optional[float]:
    """KV cache cost per token, or None if the header does not say enough.

    The certified model gets its MEASURED rate. Everything else is estimated
    from attention geometry, which is exact for dense models and conservative
    (an over-estimate) for hybrids like the certified one, where most layers
    carry no KV at all. Over-estimating is the safe direction: it makes revv
    reach for a smaller context rather than OOM.
    """
    if identify_build(info) == DEFAULT_BUILD:
        return KV_MIB_PER_TOKEN.get(kv)
    if not (info.n_layer and info.n_embd and info.n_head and info.n_head_kv):
        return None
    n_embd_kv = float(info.n_embd) * info.n_head_kv / info.n_head
    per_token_bytes = 2.0 * info.n_layer * n_embd_kv * KV_BYTES_PER_ELEM[kv]
    return per_token_bytes / (1024.0 * 1024.0)


def model_peak_mib(info: "GGUFInfo", ctx: int, kv: str) -> Optional[int]:
    """Estimated peak VRAM, anchored on a measurement when we have one.

    The file-size term is only valid when the whole model is resident. A
    mixture-of-experts build launched with --n-cpu-moe keeps most of its
    weights in HOST RAM, so counting the file size as VRAM over-estimates it by
    gigabytes -- enough to collapse the context to a quarter of the certified
    value. For any build we have actually measured, anchor on that number and
    move only the KV term, which is the part that genuinely scales with
    context.
    """
    rate = kv_mib_per_token(info, kv)
    if rate is None:
        return None

    build_name = identify_build(info)
    spec = BUILDS.get(build_name) if build_name else None
    if spec is not None and spec.get("peak_mib"):
        anchor_tier = TIERS["12gb"]
        anchor_ctx = int(anchor_tier["ctx"])
        anchor_rate = kv_mib_per_token(info, str(anchor_tier["kv"])) or rate
        base = float(spec["peak_mib"]) - anchor_ctx * anchor_rate
        return int(round(base + ctx * rate))

    weights = info.file_size / (1024.0 * 1024.0)
    return int(round(weights + COMPUTE_OVERHEAD_MIB + ctx * rate))


# The least free VRAM that can run anything at all: the smallest ladder rung.
VRAM_MIN_FREE_MIB = (estimated_peak_mib(CONTEXT_LADDER[-1], "q8_0")
                     + VRAM_MARGIN_MIB)

# Turing (7.5) is the floor: the i-quant kernels revv relies on take a code
# path that does not exist on older architectures.
MIN_COMPUTE_CAPABILITY = (7, 5)

# ---------------------------------------------------------------------------
# Model registry
#
# One model ships. More VRAM does NOT buy a bigger quant, it buys context:
# above ~2.9bpw this model's HumanEval is statistically indistinguishable
# from its own uncompressed anchor (IQ3_XXS 92.7% vs Q8 93.3%), so spending
# gigabytes on more weight bits buys nothing measurable. See BENCHMARKS.md
# section "Why one model".
# ---------------------------------------------------------------------------

HF_REPO = "unsloth/Qwen3.8-27B-GGUF"
HF_REPO_35B = "unsloth/Qwen3.6-35B-A3B-MTP-GGUF"

# Two certified lines. FLAGSHIP is the 27B dense model; SPEED is a 35B
# mixture-of-experts model whose experts stream from host RAM, which is why it
# is faster despite being a bigger file: only ~3B parameters are active per
# token. They tie on our instruments; see README for where they differ.
#
# Sizes are exact bytes from the HuggingFace API. They are the download's
# integrity check: a truncated or CDN-mangled file is caught before it ever
# reaches llama-server.
BUILDS: Dict[str, Dict[str, object]] = {
    "IQ3_XXS": {
        "file": "Qwen3.8-27B-UD-IQ3_XXS.gguf",
        "repo": HF_REPO,
        "size": 10934860704,
        "line": "flagship",
        "certified": True,
        "humaneval": 92.7,
        "decode_ts": 37.9,
        "peak_mib": 11830,
        "note": "the flagship: 27B dense, the build every published number "
                "was measured on",
    },
    "Q3_K_XL_35B": {
        "file": "Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf",
        "repo": HF_REPO_35B,
        # The MTP repo, NOT the plain one. Both publish a file with this exact
        # name; the plain repo's build (16,845,511,648 bytes, 733 tensors, 40
        # layers) has NO draft head, so speculation silently would not run and
        # the 55.9 t/s figure would not be reachable. Verified by parsing both
        # headers over a ranged HTTP fetch: this one has 41 layers, 753
        # tensors, and blk.40.nextn.*.
        "size": 17227569440,
        "line": "speed",
        "certified": True,
        "humaneval": 92.68,
        # 55.9 t/s: certified 2026-09 with three flag changes over the
        # original 48.5 t/s baseline -- -t 8 (CPU-MoE thread heuristic) and
        # an n-gram+MTP drafter stack (ngram-simple,draft-mtp). Quality
        # re-verified unchanged: HE-164 153/164 vs 152/164 (p=1.0),
        # edit-compliance 34/34 vs 33/34 (p=1.0). See BENCHMARKS.md.
        "decode_ts": 55.9,
        "peak_mib": 11832,
        # MoE: experts live in host RAM and stream in, so this line has a
        # second requirement the flagship does not have.
        "n_cpu_moe": 16,
        "host_ram_mib": 8192,
        "note": "the speed tier: 35B mixture-of-experts, 55.9 t/s (2.52x "
                "stock), ties the flagship on our instruments. Needs ~8 GiB "
                "of free host RAM on top of the VRAM, because the experts "
                "stream from it.",
    },
    "Q2_K_XL": {
        "file": "Qwen3.8-27B-UD-Q2_K_XL.gguf",
        "repo": HF_REPO,
        "size": 9828981664,
        "line": "flagship",
        "certified": False,
        "humaneval": 93.3,
        "note": "1.03 GiB smaller and the same HumanEval, but edit-format "
                "compliance is 67.6% vs 94.1% (p=0.0117) -- it breaks in agent "
                "loops, not on benchmarks. Use it only if VRAM forces you to.",
    },
    "IQ2_XXS": {
        "file": "Qwen3.8-27B-UD-IQ2_XXS.gguf",
        "repo": HF_REPO,
        "size": 7266070528,
        "line": "flagship",
        "certified": False,
        "humaneval": 78.0,
        "note": "no MTP draft head (stripped below ~8.4 GiB): no speculation, "
                "so ~20 t/s not ~37, and 15 points of HumanEval gone. "
                "Small is doubly penalised. Not recommended.",
    },
}

# The two lines a user can ask for by name.
MODEL_LINES = {"flagship": "IQ3_XXS", "speed": "Q3_K_XL_35B"}

DEFAULT_BUILD = "IQ3_XXS"

# Tier -> runtime configuration. Only the 12GB tier is certified; the others
# are the same certified weights with the extra VRAM spent on context, which
# is a derived setting, not a measured one.
TIERS: Dict[str, Dict[str, object]] = {
    "12gb": {"min_mib": VRAM_MIN_FREE_MIB, "ctx": 16384, "kv": "q8_0",
             "certified": True,
             "desc": "certified: 36.7 t/s, 92.7% HumanEval, 11,958 MiB peak"},
    "16gb": {"min_mib": 15000, "ctx": 32768, "kv": "q8_0",
             "certified": False,
             "desc": "certified weights, context raised to 32K (not separately measured)"},
    "24gb": {"min_mib": 23000, "ctx": 65536, "kv": "f16",
             "certified": False,
             "desc": "certified weights, 64K context and f16 KV (not separately measured)"},
}

TIER_ORDER = ["24gb", "16gb", "12gb"]  # highest first, for detection

# Speculation is the whole speed story: MTP n=2 is +68% and measured
# quality-neutral (135/164 with vs 136/164 without, p=1.0). n>=3 showed
# greedy non-reproducibility in one observation and is not shipped.
SPEC_TYPE = "draft-mtp"
SPEC_N_MAX = 2

# The n-gram+MTP drafter chain. Originally shipped speed-tier-only, then
# certified on the flagship too (2026-09-05): editing workloads 40.3 ->
# 222.8 / 246.0 / 113.3 t/s (2.81-6.10x), pure generation 35.17 -> 35.16 t/s
# (1.00x, inert -- the model isn't reusing anything there, so the n-gram
# matcher just misses and falls through), outputs byte-identical to
# plain-MTP on all 4 workloads. It now applies to EVERY build that
# speculates through its own MTP head, not just the n_cpu_moe (host-RAM
# offload) tier -- see build_server_argv. llama.cpp runs speculation chains
# first-success-wins: an n-gram hit skips the MTP pass for that token, so
# this is a strict addition over MTP alone, not a substitute for it.
# size-m=256 was verified byte-identical to MTP-only output; see
# BENCHMARKS.md for the size-m sweep (rising through 256, so this is a
# floor, not a ceiling) and the CRLF warning (0.83 -> 0.11 acceptance on
# CRLF text; keep repos LF).
SPEC_TYPE_CHAIN = "ngram-simple,draft-mtp"
SPEC_NGRAM_SIZE_M = 256

# The chain's own VRAM cost. Small but real, and it has to be counted before
# the context ladder picks a rung or a "certified" config can OOM once the
# chain is actually running. Measured on the flagship: 11,956 MiB vs
# 11,854 MiB at c=16384, both otherwise-identical launches -- a delta of
# ~100 MiB. Only added for builds whose registered peak_mib does NOT already
# bake the chain in: the n_cpu_moe speed tier was certified WITH the chain
# from the start (11,832 MiB peak already includes it), so adding this again
# there would double-count it and needlessly shrink its context.
SPEC_NGRAM_CHAIN_MIB = 100


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

def _use_color() -> bool:
    return (sys.stdout.isatty()
            and os.environ.get("NO_COLOR") is None
            and os.environ.get("TERM", "") != "dumb")


_COLOR = _use_color()


def _c(code: str, text: str) -> str:
    return "\033[%sm%s\033[0m" % (code, text) if _COLOR else text


def bold(t: str) -> str:
    return _c("1", t)


def green(t: str) -> str:
    return _c("32", t)


def yellow(t: str) -> str:
    return _c("33", t)


def red(t: str) -> str:
    return _c("31", t)


def dim(t: str) -> str:
    return _c("2", t)


OK, WARN, FAIL = "ok", "warn", "fail"
_TAG_COLOR = {OK: green, WARN: yellow, FAIL: red}


def status(tag: str, text: str, detail: str = "") -> None:
    # Pad on the uncoloured word: ANSI escapes are invisible on screen but not
    # zero-width to str formatting, so padding the coloured string misaligns.
    pad = " " * (6 - len(tag))
    print("  [%s]%s%s" % (_TAG_COLOR[tag](tag), pad, text))
    if detail:
        for line in detail.splitlines():
            print("         %s" % line)


def die(msg: str, fix: str = "") -> None:
    """Exit with a message that tells the user what to do about it."""
    # stdout is block-buffered when piped; without this the error lands above
    # the output it refers to.
    sys.stdout.flush()
    print("%s %s" % (red("error:"), msg), file=sys.stderr)
    if fix:
        print("\n%s\n  %s" % (bold("fix:"), fix.replace("\n", "\n  ")),
              file=sys.stderr)
    sys.exit(1)


def gib(n: float) -> str:
    return "%.2f GiB" % (n / (1024.0 ** 3))


def mib(n: int) -> str:
    return "{:,} MiB".format(n)


# ---------------------------------------------------------------------------
# GGUF header reader   [spliced: gguf unit]
# ---------------------------------------------------------------------------

GGUF_MAGIC = 0x46554747  # b"GGUF" read as a little-endian u32
SUPPORTED_VERSIONS = (2, 3)
DEFAULT_ALIGNMENT = 32
QK_K = 256

# GGUF key-value value type ids -> meaning (see module docstring / spec).
_VT_UINT8 = 0
_VT_INT8 = 1
_VT_UINT16 = 2
_VT_INT16 = 3
_VT_UINT32 = 4
_VT_INT32 = 5
_VT_FLOAT32 = 6
_VT_BOOL = 7
_VT_STRING = 8
_VT_ARRAY = 9
_VT_UINT64 = 10
_VT_INT64 = 11
_VT_FLOAT64 = 12

# Byte width of every *fixed-size* value type (everything except STRING and
# ARRAY, which are variable-length and handled specially).
_FIXED_SIZE: Dict[int, int] = {
    _VT_UINT8: 1,
    _VT_INT8: 1,
    _VT_UINT16: 2,
    _VT_INT16: 2,
    _VT_UINT32: 4,
    _VT_INT32: 4,
    _VT_FLOAT32: 4,
    _VT_BOOL: 1,
    _VT_UINT64: 8,
    _VT_INT64: 8,
    _VT_FLOAT64: 8,
}

# ggml tensor type id -> canonical name. Gaps are real (removed/reserved
# ids in upstream ggml); do not assume contiguity.
GGML_TYPE_NAMES: Dict[int, str] = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    6: "Q5_0",
    7: "Q5_1",
    8: "Q8_0",
    9: "Q8_1",
    10: "Q2_K",
    11: "Q3_K",
    12: "Q4_K",
    13: "Q5_K",
    14: "Q6_K",
    15: "Q8_K",
    16: "IQ2_XXS",
    17: "IQ2_XS",
    18: "IQ3_XXS",
    19: "IQ1_S",
    20: "IQ4_NL",
    21: "IQ3_S",
    22: "IQ2_S",
    23: "IQ4_XS",
    24: "I8",
    25: "I16",
    26: "I32",
    27: "I64",
    28: "F64",
    29: "IQ1_M",
    30: "BF16",
    34: "TQ1_0",
    35: "TQ2_0",
    39: "MXFP4",
    40: "NVFP4",
    41: "Q1_0",
}

# ggml type name -> (block_elems, bytes_per_block). Used to compute tensor
# byte sizes from ne[] dims without touching the data section.
BLOCK_SIZES: Dict[str, Tuple[int, int]] = {
    "F32": (1, 4),
    "F16": (1, 2),
    "BF16": (1, 2),
    "F64": (1, 8),
    "I8": (1, 1),
    "I16": (1, 2),
    "I32": (1, 4),
    "I64": (1, 8),
    "Q4_0": (32, 18),
    "Q4_1": (32, 20),
    "Q5_0": (32, 22),
    "Q5_1": (32, 24),
    "Q8_0": (32, 34),
    "Q8_1": (32, 40),
    "IQ4_NL": (32, 18),
    "MXFP4": (32, 17),
    "NVFP4": (64, 36),
    "Q1_0": (128, 18),
    "Q2_K": (256, 2 + 2 + QK_K // 16 + QK_K // 4),
    "Q3_K": (256, 2 + QK_K // 4 + QK_K // 8 + 12),
    "Q4_K": (256, 2 + 2 + QK_K // 2 + 12),
    "Q5_K": (256, 2 + 2 + QK_K // 2 + QK_K // 8 + 12),
    "Q6_K": (256, 2 + QK_K // 2 + QK_K // 4 + QK_K // 16),
    "Q8_K": (256, 4 + QK_K + QK_K // 8),
    "IQ2_XXS": (256, 2 + QK_K // 4),
    "IQ2_XS": (256, 2 + QK_K // 4 + QK_K // 32),
    "IQ3_XXS": (256, 2 + QK_K // 4 + QK_K // 8),
    "IQ1_S": (256, 2 + QK_K // 8 + QK_K // 16),
    "IQ3_S": (256, 2 + QK_K // 4 + QK_K // 8 + QK_K // 32 + 4),
    "IQ2_S": (256, 2 + QK_K // 4 + QK_K // 16),
    "IQ4_XS": (256, 2 + 2 + QK_K // 2 + QK_K // 64),
    "IQ1_M": (256, QK_K // 8 + QK_K // 16 + QK_K // 32),
    "TQ1_0": (256, 2 + 4 * 13),
    "TQ2_0": (256, 2 + 64),
}


class GGUFError(Exception):
    """Raised for anything wrong with a GGUF file: bad magic, unsupported
    version, truncation, or an unknown value/array-element type id."""


@dataclasses.dataclass
class TensorInfo:
    name: str
    dims: Tuple[int, ...]
    type_name: str
    n_bytes: Optional[int]


@dataclasses.dataclass
class GGUFInfo:
    path: str
    file_size: int
    version: int
    alignment: int
    n_tensors: int
    n_kv: int
    arch: Optional[str]
    name: Optional[str]
    n_vocab: Optional[int]
    n_layer: Optional[int]
    n_ctx_train: Optional[int]
    n_embd: Optional[int]
    n_head: Optional[int]
    n_head_kv: Optional[int]
    supports_thinking: bool
    file_type: Optional[int]
    dominant_quant: str
    type_counts: Dict[str, int]
    type_bytes: Dict[str, int]
    mtp_tensors: List[str]
    data_offset: int
    tensor_data_bytes: int

    @property
    def has_mtp_head(self) -> bool:
        return bool(self.mtp_tensors)


class _ArrayMeta(NamedTuple):
    """Placeholder stored in the kv dict for ARRAY-typed values: we record
    only the element type and count (the payload has already been walked
    and discarded, never materialized)."""

    element_type: int
    count: int


class _Reader:
    """Thin wrapper around a buffered binary file that turns short reads
    (truncated file) into a GGUFError instead of silently returning fewer
    bytes than requested."""

    def __init__(self, f: BinaryIO, path: str) -> None:
        self.f = f
        self.path = path

    def read(self, n: int) -> bytes:
        start = self.f.tell()
        data = self.f.read(n)
        if len(data) < n:
            raise GGUFError(
                "file is truncated or incomplete: expected {} bytes at "
                "offset {} but only got {} in {!r}; the file may not have "
                "finished downloading -- re-download it".format(n, start, len(data), self.path)
            )
        return data

    def seek_forward(self, n: int) -> None:
        # A plain seek (no read) is how we skip large/uninteresting value
        # payloads without paging them through Python; any resulting
        # out-of-bounds position is caught by the next read() call.
        if n:
            self.f.seek(n, os.SEEK_CUR)

    def tell(self) -> int:
        return self.f.tell()


def _read_u32(r: _Reader) -> int:
    return struct.unpack("<I", r.read(4))[0]


def _read_u64(r: _Reader) -> int:
    return struct.unpack("<Q", r.read(8))[0]


def _read_string(r: _Reader) -> str:
    length = _read_u64(r)
    data = r.read(length)
    return data.decode("utf-8", errors="replace")


def _skip_array_payload(r: _Reader, element_type: int, count: int) -> None:
    """Walks (but does not store) `count` elements of `element_type`,
    leaving the file cursor positioned right after the array."""
    if element_type == _VT_STRING:
        # Variable stride: each element must be walked individually.
        for _ in range(count):
            slen = _read_u64(r)
            r.seek_forward(slen)
    elif element_type == _VT_ARRAY:
        # Nested arrays (rare/hypothetical): each element declares its own
        # element_type + count per the spec's recursive definition.
        for _ in range(count):
            nested_type = _read_u32(r)
            nested_count = _read_u64(r)
            _skip_array_payload(r, nested_type, nested_count)
    elif element_type in _FIXED_SIZE:
        # Fixed stride: one bulk seek instead of `count` tiny reads.
        r.seek_forward(_FIXED_SIZE[element_type] * count)
    else:
        raise GGUFError(
            "unknown GGUF array element type id {} at offset {}".format(element_type, r.tell())
        )


def _read_value(r: _Reader, value_type: int) -> Any:
    """Reads a scalar/string value fully, or for ARRAY walks and discards
    the payload and returns an _ArrayMeta(element_type, count)."""
    if value_type == _VT_UINT8:
        return r.read(1)[0]
    if value_type == _VT_INT8:
        return struct.unpack("<b", r.read(1))[0]
    if value_type == _VT_UINT16:
        return struct.unpack("<H", r.read(2))[0]
    if value_type == _VT_INT16:
        return struct.unpack("<h", r.read(2))[0]
    if value_type == _VT_UINT32:
        return _read_u32(r)
    if value_type == _VT_INT32:
        return struct.unpack("<i", r.read(4))[0]
    if value_type == _VT_FLOAT32:
        return struct.unpack("<f", r.read(4))[0]
    if value_type == _VT_BOOL:
        return r.read(1)[0] != 0
    if value_type == _VT_STRING:
        return _read_string(r)
    if value_type == _VT_ARRAY:
        element_type = _read_u32(r)
        count = _read_u64(r)
        _skip_array_payload(r, element_type, count)
        return _ArrayMeta(element_type, count)
    if value_type == _VT_UINT64:
        return _read_u64(r)
    if value_type == _VT_INT64:
        return struct.unpack("<q", r.read(8))[0]
    if value_type == _VT_FLOAT64:
        return struct.unpack("<d", r.read(8))[0]
    raise GGUFError("unknown GGUF value type id {} at offset {}".format(value_type, r.tell()))


def _is_mtp_tensor(name: str) -> bool:
    """MTP / multi-token-prediction draft-head tensors, e.g.
    'blk.64.nextn.embed_tokens.weight'. Case-insensitive."""
    lower = name.lower()
    if lower.split(".").count("nextn") > 0:
        # covers ".nextn." and a leading/trailing "nextn" component too,
        # but we still check the explicit prefix case below for names
        # with no dot separator at all (just "nextn").
        return True
    if ".nextn." in lower:
        return True
    if lower.startswith("nextn."):
        return True
    return lower == "nextn"


def read_gguf(path: str) -> GGUFInfo:
    file_size = os.path.getsize(path)
    with open(path, "rb") as raw:
        r = _Reader(raw, path)

        magic = _read_u32(r)
        if magic != GGUF_MAGIC:
            raise GGUFError(
                "not a GGUF file: {!r} (magic 0x{:08x} != 0x{:08x}); "
                "a .safetensors or .bin file is not a GGUF file".format(path, magic, GGUF_MAGIC)
            )

        version = _read_u32(r)
        if version not in SUPPORTED_VERSIONS:
            raise GGUFError(
                "unsupported GGUF version {} (only versions {} are supported)".format(
                    version, SUPPORTED_VERSIONS
                )
            )

        tensor_count = _read_u64(r)
        kv_count = _read_u64(r)

        kv: Dict[str, Any] = {}
        for _ in range(kv_count):
            key = _read_string(r)
            value_type = _read_u32(r)
            kv[key] = _read_value(r, value_type)

        tensor_infos: List[TensorInfo] = []
        type_counts: Dict[str, int] = {}
        type_bytes: Dict[str, int] = {}
        mtp_tensors: List[str] = []
        token_embd_dims: Optional[Tuple[int, ...]] = None

        for _ in range(tensor_count):
            name = _read_string(r)
            n_dims = _read_u32(r)
            dims = tuple(_read_u64(r) for _ in range(n_dims))
            ggml_type = _read_u32(r)
            _offset = _read_u64(r)  # relative to data section; unused here

            type_name = GGML_TYPE_NAMES.get(ggml_type, "UNKNOWN({})".format(ggml_type))
            block_info = BLOCK_SIZES.get(type_name)
            if block_info is not None:
                block_elems, bytes_per_block = block_info
                n_bytes: Optional[int] = math.prod(dims) * bytes_per_block // block_elems
            else:
                n_bytes = None

            tensor_infos.append(TensorInfo(name=name, dims=dims, type_name=type_name, n_bytes=n_bytes))

            type_counts[type_name] = type_counts.get(type_name, 0) + 1
            if n_bytes is not None:
                type_bytes[type_name] = type_bytes.get(type_name, 0) + n_bytes

            if _is_mtp_tensor(name):
                mtp_tensors.append(name)

            if name == "token_embd.weight":
                token_embd_dims = dims

        # --- alignment & data offset -------------------------------------------------
        alignment_val = kv.get("general.alignment")
        if alignment_val is None:
            alignment = DEFAULT_ALIGNMENT
        else:
            alignment = int(alignment_val)
            if alignment <= 0 or (alignment & (alignment - 1)) != 0:
                raise GGUFError(
                    "general.alignment must be a nonzero power of two, got {}".format(alignment_val)
                )

        offs = r.tell()
        data_offset = ((offs + alignment - 1) // alignment) * alignment

        # --- scalar metadata lookups ---------------------------------------------------
        arch = kv.get("general.architecture")
        arch = arch if isinstance(arch, str) else None

        name_val = kv.get("general.name")
        name_val = name_val if isinstance(name_val, str) else None

        file_type_val = kv.get("general.file_type")
        file_type = file_type_val if isinstance(file_type_val, int) else None

        n_layer = kv.get("{}.block_count".format(arch)) if arch else None
        n_layer = n_layer if isinstance(n_layer, int) else None

        n_ctx_train = kv.get("{}.context_length".format(arch)) if arch else None
        n_ctx_train = n_ctx_train if isinstance(n_ctx_train, int) else None

        n_embd = kv.get("{}.embedding_length".format(arch)) if arch else None
        n_embd = n_embd if isinstance(n_embd, int) else None

        n_head = kv.get("{}.attention.head_count".format(arch)) if arch else None
        n_head = n_head if isinstance(n_head, int) else None

        # GQA: KV cache is sized by the KV head count, not the query heads.
        n_head_kv = (kv.get("{}.attention.head_count_kv".format(arch))
                     if arch else None)
        n_head_kv = n_head_kv if isinstance(n_head_kv, int) else n_head

        # Whether the packaged chat template implements a thinking switch at
        # all. If it does not, revv's --reasoning off is a no-op on this model
        # and claiming it as a lever would be dishonest.
        template = kv.get("tokenizer.chat_template")
        supports_thinking = (isinstance(template, str)
                             and "enable_thinking" in template)

        # --- n_vocab: tokens array count, then {arch}.vocab_size, then token_embd dims[1]
        n_vocab: Optional[int] = None
        tokens_meta = kv.get("tokenizer.ggml.tokens")
        if isinstance(tokens_meta, _ArrayMeta) and tokens_meta.element_type == _VT_STRING:
            n_vocab = tokens_meta.count
        if n_vocab is None and arch:
            vocab_size = kv.get("{}.vocab_size".format(arch))
            if isinstance(vocab_size, int):
                n_vocab = vocab_size
        if n_vocab is None and token_embd_dims is not None and len(token_embd_dims) >= 2:
            n_vocab = token_embd_dims[1]

        # --- dominant quant: largest total bytes, excluding norm/bias-only float types
        excluded = {"F32", "F16", "BF16"}
        candidates = {k: v for k, v in type_bytes.items() if k not in excluded and v > 0}
        if candidates:
            dominant_quant = max(candidates, key=candidates.get)
        elif type_bytes:
            dominant_quant = max(type_bytes, key=type_bytes.get)
        else:
            dominant_quant = ""

        tensor_data_bytes = sum(t.n_bytes for t in tensor_infos if t.n_bytes is not None)

        return GGUFInfo(
            path=path,
            file_size=file_size,
            version=version,
            alignment=alignment,
            n_tensors=tensor_count,
            n_kv=kv_count,
            arch=arch,
            name=name_val,
            n_vocab=n_vocab,
            n_layer=n_layer,
            n_ctx_train=n_ctx_train,
            n_embd=n_embd,
            n_head=n_head,
            n_head_kv=n_head_kv,
            supports_thinking=supports_thinking,
            file_type=file_type,
            dominant_quant=dominant_quant,
            type_counts=type_counts,
            type_bytes=type_bytes,
            mtp_tensors=mtp_tensors,
            data_offset=data_offset,
            tensor_data_bytes=tensor_data_bytes,
        )


# ---------------------------------------------------------------------------
# Resumable downloader   [spliced: download unit]
# ---------------------------------------------------------------------------

USER_AGENT = "revv/1.0"
CHUNK_SIZE = 1024 * 1024  # 1 MiB, per spec
BACKOFF_CAP_SECONDS = 30.0
PROGRESS_WINDOW_SECONDS = 5.0  # moving-average window for the rate display
TTY_REDRAW_INTERVAL_SECONDS = 0.1  # throttle redraws to ~10/s

# Exceptions that indicate a transient network problem worth retrying.
_RETRYABLE_EXCEPTIONS = (
    socket.timeout,
    urllib.error.URLError,
    http.client.IncompleteRead,
    ConnectionResetError,
)


class DownloadError(Exception):
    """Raised for any download failure; the message tells the user how to fix it."""


def _build_request(url: str, method: str, range_header: Optional[str] = None) -> urllib.request.Request:
    headers = {"User-Agent": USER_AGENT}
    if range_header is not None:
        headers["Range"] = range_header
    return urllib.request.Request(url, headers=headers, method=method)


def head_size(url: str, timeout: float = 30.0) -> Optional[int]:
    """Content-Length of the final resource, following redirects. None if unknown."""
    req = _build_request(url, "HEAD")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            length = resp.headers.get("Content-Length")
    except (urllib.error.URLError, socket.timeout, http.client.HTTPException):
        # HEAD is a best-effort probe; callers treat None as "unknown size",
        # not a fatal error, so failures here should not raise.
        return None
    if length is None:
        return None
    try:
        return int(length)
    except ValueError:
        return None


def _format_eta(seconds: float) -> str:
    if seconds != seconds or seconds == float("inf"):  # nan or inf: rate unknown
        return "--:--"
    seconds_int = int(seconds)
    hours, rem = divmod(seconds_int, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return "%d:%02d:%02d" % (hours, minutes, secs)
    return "%02d:%02d" % (minutes, secs)


class _ProgressReporter:
    """Renders download progress to stderr; only instantiated when progress=True."""

    def __init__(self, total: Optional[int]) -> None:
        self._is_tty = sys.stderr.isatty()
        self._total: Optional[int] = total
        self._downloaded = 0
        self._last_draw_time = 0.0
        self._last_pct_step_reported = -10  # so 0% line is not force-printed
        self._samples: List[Tuple[float, int]] = []

    def set_total(self, total: Optional[int]) -> None:
        self._total = total

    def set_downloaded(self, n: int) -> None:
        self._downloaded = n
        now = time.time()
        self._samples.append((now, n))
        cutoff = now - PROGRESS_WINDOW_SECONDS
        # Keep at least two samples so a rate can always be computed.
        while len(self._samples) > 2 and self._samples[1][0] < cutoff:
            self._samples.pop(0)

    def note_retry(self, attempt: int, max_retries: int, backoff: float) -> None:
        sys.stderr.write(
            "download interrupted (attempt %d/%d), retrying in %.0fs...\n" % (attempt, max_retries, backoff)
        )
        sys.stderr.flush()

    def _rate_bytes_per_sec(self) -> float:
        if len(self._samples) < 2:
            return 0.0
        t0, b0 = self._samples[0]
        t1, b1 = self._samples[-1]
        dt = t1 - t0
        if dt <= 0:
            return 0.0
        return (b1 - b0) / dt

    def _format_line(self) -> str:
        gib = 1024.0 ** 3
        mib = 1024.0 ** 2
        downloaded_gib = self._downloaded / gib
        rate = self._rate_bytes_per_sec()
        rate_mib_s = rate / mib
        if self._total:
            total_gib = self._total / gib
            pct = min(100.0, self._downloaded * 100.0 / self._total)
            remaining = max(0, self._total - self._downloaded)
            eta = remaining / rate if rate > 0 else float("inf")
            return "%.2f/%.2f GiB (%5.1f%%) %6.2f MiB/s ETA %s" % (
                downloaded_gib,
                total_gib,
                pct,
                rate_mib_s,
                _format_eta(eta),
            )
        return "%.2f GiB downloaded, %6.2f MiB/s" % (downloaded_gib, rate_mib_s)

    def maybe_draw(self) -> None:
        now = time.time()
        if self._is_tty:
            if now - self._last_draw_time < TTY_REDRAW_INTERVAL_SECONDS:
                return
            self._last_draw_time = now
            sys.stderr.write("\r" + self._format_line() + "    ")
            sys.stderr.flush()
            return
        if not self._total:
            return  # nothing sane to print every 10% without a known total
        pct = self._downloaded * 100.0 / self._total
        step = int(pct // 10) * 10
        if step > self._last_pct_step_reported:
            self._last_pct_step_reported = step
            sys.stderr.write(self._format_line() + "\n")
            sys.stderr.flush()

    def finish(self) -> None:
        if self._is_tty:
            sys.stderr.write("\r" + self._format_line() + "    \n")
            sys.stderr.flush()
        elif self._total and self._last_pct_step_reported < 100:
            self._last_pct_step_reported = 100
            sys.stderr.write(self._format_line() + "\n")
            sys.stderr.flush()


def _attempt_download(
    url: str,
    part_path: str,
    offset: int,
    timeout: float,
    reporter: Optional[_ProgressReporter],
    known_total: Optional[int],
) -> Optional[int]:
    """Performs a single HTTP GET attempt, writing into part_path. Returns the
    best-known total size (may be unchanged from known_total). Raises on any
    transient network error so the caller's retry loop can resume from disk."""
    range_header = "bytes=%d-" % offset if offset > 0 else None
    req = _build_request(url, "GET", range_header)

    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        if e.code == 416:
            # .part already covers the full length; caller falls through to
            # the size check rather than treating this as an error.
            e.close()
            return known_total if known_total is not None else offset
        if e.code == 404:
            e.close()
            raise DownloadError(
                "HTTP 404 Not Found for %s. The file may have been moved or "
                "renamed upstream; double-check the URL." % url
            ) from e
        if 400 <= e.code < 500:
            e.close()
            raise DownloadError(
                "HTTP %d %s for %s. This is a client-side error and will not "
                "be retried; check the URL and any required credentials." % (e.code, e.reason, url)
            ) from e
        # 5xx and other unexpected codes are transient; HTTPError is itself a
        # URLError subclass so re-raising lets the caller's retry loop catch it.
        e.close()
        raise

    total = known_total
    with resp:
        status = resp.status
        content_length = resp.headers.get("Content-Length")
        content_range = resp.headers.get("Content-Range")

        if status == 206:
            if content_range:
                try:
                    total = int(content_range.rsplit("/", 1)[1])
                except (ValueError, IndexError):
                    pass
            elif content_length is not None:
                try:
                    total = offset + int(content_length)
                except ValueError:
                    pass
            write_offset = offset
            truncate = False
        else:
            # Status 200 (or anything else without an exception): the server
            # ignored our Range header, so the body is the WHOLE resource.
            # Truncating and restarting from zero is the only way to avoid
            # corrupting the .part file by appending a full body onto it.
            if content_length is not None:
                try:
                    total = int(content_length)
                except ValueError:
                    pass
            write_offset = 0
            truncate = offset > 0

        if reporter is not None:
            reporter.set_total(total)
            reporter.set_downloaded(write_offset)

        mode = "r+b" if os.path.exists(part_path) else "wb"
        f = open(part_path, mode)
        try:
            if truncate:
                f.seek(0)
                f.truncate(0)
            f.seek(write_offset)
            written = write_offset
            while True:
                chunk = resp.read(CHUNK_SIZE)
                if not chunk:
                    break
                f.write(chunk)
                written += len(chunk)
                if reporter is not None:
                    reporter.set_downloaded(written)
                    reporter.maybe_draw()
        finally:
            # Runs on success, on a network exception, and on KeyboardInterrupt
            # alike, so the .part file is always left flushed and resumable.
            f.flush()
            f.close()

    return total


def download(
    url: str,
    dest: str,
    expected_size: Optional[int] = None,
    progress: bool = True,
    max_retries: int = 5,
    timeout: float = 60.0,
) -> int:
    """Download url to dest, resuming if a partial file exists.
    Returns the final byte size. Idempotent: if dest already exists and
    (expected_size is None or matches), returns immediately without network I/O."""
    dest_dir = os.path.dirname(os.path.abspath(dest))
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    if os.path.exists(dest):
        existing_size = os.path.getsize(dest)
        if expected_size is None or existing_size == expected_size:
            return existing_size

    part_path = dest + ".part"
    reporter = _ProgressReporter(expected_size) if progress else None
    total_size = expected_size
    attempt = 0

    try:
        while True:
            part_size = os.path.getsize(part_path) if os.path.exists(part_path) else 0
            try:
                total_size = _attempt_download(url, part_path, part_size, timeout, reporter, total_size)
                break
            except _RETRYABLE_EXCEPTIONS as e:
                attempt += 1
                if attempt > max_retries:
                    raise DownloadError(
                        "Download of %s failed after %d attempt(s): %s. A partial "
                        "file was kept at %s; rerun the download to resume, or "
                        "check your network connection." % (url, attempt, e, part_path)
                    ) from e
                backoff = min(2 ** (attempt - 1), BACKOFF_CAP_SECONDS)
                if reporter is not None:
                    reporter.note_retry(attempt, max_retries, backoff)
                time.sleep(backoff)
    except KeyboardInterrupt:
        # The .part file was already flushed/closed by _attempt_download's
        # finally block, so it is safe to resume on the next run.
        sys.stderr.write("\ninterrupted - rerun the same command to resume\n")
        sys.stderr.flush()
        raise

    final_size = os.path.getsize(part_path)
    if expected_size is not None and final_size != expected_size:
        raise DownloadError(
            "Downloaded size %d bytes does not match expected size %d bytes for "
            "%s. Delete %s and retry the download." % (final_size, expected_size, url, part_path)
        )

    os.replace(part_path, dest)
    if reporter is not None:
        reporter.finish()
    return final_size



# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

class GPU:
    def __init__(self, name: str, total_mib: int, used_mib: int,
                 free_mib: int, driver: str,
                 cc: Optional[Tuple[int, int]]) -> None:
        self.name = name
        self.total_mib = total_mib
        self.used_mib = used_mib
        # Reported by the driver, NOT total-minus-used: on WSL2 the host
        # reserves memory that shows up in neither total nor used.
        self.free_mib = free_mib
        self.driver = driver
        self.cc = cc

    @property
    def reserved_mib(self) -> int:
        """VRAM the driver accounts for in neither used nor free."""
        return max(0, self.total_mib - self.used_mib - self.free_mib)


def detect_gpus() -> Tuple[List[GPU], Optional[str]]:
    """Return (gpus, error). error is a human-readable reason if detection failed."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return [], "nvidia-smi not found on PATH"
    query = ("name,memory.total,memory.used,memory.free,driver_version,"
             "compute_cap")
    try:
        out = subprocess.run(
            [exe, "--query-gpu=" + query, "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError) as exc:
        return [], "could not run nvidia-smi: %s" % exc
    if out.returncode != 0:
        detail = (out.stderr or out.stdout).strip().splitlines()
        return [], "nvidia-smi failed: %s" % (detail[0] if detail else
                                              "exit %d" % out.returncode)

    gpus: List[GPU] = []
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            total, used, free = int(parts[1]), int(parts[2]), int(parts[3])
        except ValueError:
            continue
        cc = None   # type: Optional[Tuple[int, int]]
        # compute_cap is not supported by every driver version; absence is not
        # an error, it just means we cannot check the Turing floor.
        if len(parts) >= 6 and re.match(r"^\d+\.\d+$", parts[5]):
            major, minor = parts[5].split(".")
            cc = (int(major), int(minor))
        gpus.append(GPU(parts[0], total, used, free, parts[4], cc))
    if not gpus:
        return [], "nvidia-smi ran but reported no GPUs"
    return gpus, None


def tier_for(free_mib: int) -> Optional[str]:
    """Tier from FREE VRAM. Total lies wherever the host reserves memory."""
    for name in TIER_ORDER:
        if free_mib >= int(TIERS[name]["min_mib"]):
            return name
    return None


# ---------------------------------------------------------------------------
# llama-server discovery
# ---------------------------------------------------------------------------

def find_llama_server() -> Optional[str]:
    """revv's own build wins over whatever is on PATH: we know its provenance."""
    local = os.path.join(BIN_DIR, "llama-server")
    if os.path.isfile(local) and os.access(local, os.X_OK):
        return local
    return shutil.which("llama-server")


def llama_server_version(exe: str) -> Optional[str]:
    try:
        out = subprocess.run([exe, "--version"], capture_output=True,
                             text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    # llama-server prints its version banner on stderr.
    blob = (out.stderr or "") + (out.stdout or "")
    m = re.search(r"version:\s*(\d+)\s*\(([0-9a-f]+)\)", blob)
    if m:
        build, commit = m.group(1), m.group(2)
        # A clone without git-describe metadata reports "build 1" while the
        # commit stays correct. Real llama.cpp build numbers are five digits,
        # so anything tiny is metadata loss, not an ancient binary. Verify by
        # commit, never by build number.
        if len(build) < 4:
            return "commit %s (build number unreliable: reported %s)" % (
                commit, build)
        return "build %s (%s)" % (build, commit)
    for line in blob.splitlines():
        if "version" in line.lower():
            return line.strip()
    return None


def read_build_manifest() -> Optional[Dict[str, object]]:
    """install.sh records what it built and which patches went in.

    There is no way to detect an applied source patch from a compiled binary,
    so provenance is tracked at build time or not at all.
    """
    try:
        with open(BUILD_MANIFEST, "r") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


# ---------------------------------------------------------------------------
# Local model inventory
# ---------------------------------------------------------------------------

def local_models() -> List[str]:
    if not os.path.isdir(MODELS_DIR):
        return []
    return sorted(os.path.join(MODELS_DIR, f) for f in os.listdir(MODELS_DIR)
                  if f.endswith(".gguf"))


def resolve_model(arg: Optional[str]) -> str:
    """Turn a path, an adopted name, a build name, a bare filename, or nothing
    into a real path on disk."""
    if arg:
        if os.path.isfile(arg):
            return arg
        adopted = registry_lookup(arg)
        if adopted is not None:
            return adopted
        if arg in BUILDS:
            path = os.path.join(MODELS_DIR, str(BUILDS[arg]["file"]))
            if os.path.isfile(path):
                return path
            die("build %s is not downloaded" % arg,
                "revv get   # downloads the certified build")
        candidate = os.path.join(MODELS_DIR, arg)
        if os.path.isfile(candidate):
            return candidate
        die("no such model: %s" % arg,
            "revv doctor   # lists the models revv can see\n"
            "revv adopt    # registers GGUFs ollama or LM Studio already has")

    found = local_models()
    registered = sorted(load_registry())
    if not found and not registered:
        die("no models found in %s" % MODELS_DIR,
            "revv get      # downloads the certified build (~10.2 GiB)\n"
            "revv adopt    # reuses a GGUF ollama or LM Studio already has")
    # Prefer the certified build wherever it is: it is the only one revv has
    # numbers for.
    certified = os.path.join(MODELS_DIR, str(BUILDS[DEFAULT_BUILD]["file"]))
    if certified in found:
        return certified
    if len(found) == 1 and not registered:
        return found[0]
    if not found and len(registered) == 1:
        path = registry_lookup(registered[0])
        if path is not None:
            return path
    choices = [os.path.basename(f) for f in found] + registered
    die("several models available, pick one",
        "revv serve %s\n\navailable: %s"
        % (choices[0], ", ".join(choices)))
    return ""   # unreachable


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

def cmd_doctor(args: argparse.Namespace) -> int:
    print(bold("revv %s  --  doctor" % __version__))
    problems = 0

    print("\n" + bold("GPU"))
    gpus, err = detect_gpus()
    tier: Optional[str] = None
    if err is not None:
        status(FAIL, err)
        print("         revv needs an NVIDIA GPU. There is no CPU or Apple\n"
              "         Silicon path in v1.0 -- the whole configuration is\n"
              "         CUDA-specific and would be dishonest to pretend at.")
        problems += 1
    else:
        for i, g in enumerate(gpus):
            cc_txt = ("compute capability %d.%d" % g.cc) if g.cc else \
                     "compute capability unknown (old driver)"
            detail = ("%s total, %s in use, %s free\ndriver %s, %s"
                      % (mib(g.total_mib), mib(g.used_mib), mib(g.free_mib),
                         g.driver, cc_txt))
            if g.reserved_mib > 0:
                detail += ("\n%s reserved by the host (WSL2/desktop) and not "
                           "available to CUDA" % mib(g.reserved_mib))
            status(OK if g.free_mib >= VRAM_MIN_FREE_MIB else FAIL,
                   "GPU %d: %s" % (i, g.name), detail)
            if g.cc is not None and g.cc < MIN_COMPUTE_CAPABILITY:
                status(FAIL, "architecture too old (need Turing / 7.5 or newer)")
                problems += 1
        best = max(gpus, key=lambda g: g.free_mib)
        tier = tier_for(best.free_mib)
        if tier is None:
            status(FAIL, "only %s free; revv needs at least %s"
                   % (mib(best.free_mib), mib(VRAM_MIN_FREE_MIB)),
                   "The weights alone are 10.2 GiB. Even the smallest context\n"
                   "revv will run (%d) needs %s.\n"
                   "If this card has %s total, something else is holding it:\n"
                   "close other GPU processes, or run headless."
                   % (CONTEXT_LADDER[-1],
                      mib(estimated_peak_mib(CONTEXT_LADDER[-1], "q8_0")),
                      mib(best.total_mib)))
            problems += 1
        else:
            t = TIERS[tier]
            status(OK, "tier: %s" % tier.upper(), str(t["desc"]))
            plan = plan_context(best.free_mib, str(t["kv"]), int(t["ctx"]))
            if plan is None:
                status(FAIL, "no context size fits in %s free"
                       % mib(best.free_mib))
                problems += 1
            else:
                ctx, peak = plan
                if ctx == int(t["ctx"]):
                    status(OK, "context: %s" % "{:,}".format(ctx),
                           "the full size for this tier; ~%s peak, %s free"
                           % (mib(peak), mib(best.free_mib)))
                else:
                    status(WARN, "context: %s (reduced from %s)"
                           % ("{:,}".format(ctx),
                              "{:,}".format(int(t["ctx"]))),
                           "%s at the full size needs ~%s but only %s is free.\n"
                           "revv will use %s automatically; override with --ctx."
                           % ("{:,}".format(int(t["ctx"])),
                              mib(estimated_peak_mib(int(t["ctx"]),
                                                     str(t["kv"]))),
                              mib(best.free_mib), "{:,}".format(ctx)))
        if len(gpus) > 1:
            status(WARN, "%d GPUs found; revv uses one" % len(gpus),
                   "Multi-GPU split is untested. Pin with CUDA_VISIBLE_DEVICES.")

    print("\n" + bold("llama-server"))
    exe = find_llama_server()
    if exe is None:
        status(FAIL, "llama-server not found",
               "Looked in %s and on PATH." % BIN_DIR)
        print("         Run ./install.sh -- it downloads a prebuilt binary;\n"
              "         no compiler needed. See README for the three install\n"
              "         paths and what each one asks you to trust.")
        problems += 1
    else:
        status(OK, exe, llama_server_version(exe) or "version unknown")
        manifest = read_build_manifest()
        if manifest is None:
            status(WARN, "build provenance unknown",
                   "This binary was not built by revv, so the kernel patch\n"
                   "status cannot be determined. Expect ~%.1f t/s rather\n"
                   "than %.1f if it is stock upstream."
                   % (CERT_TS_STOCK, CERT_TS))
        else:
            patches = manifest.get("patches") or []
            base = manifest.get("base_commit", "unknown")
            # Which rung of the install ladder produced this binary. It decides
            # what performance to expect and how much to trust it, so it is
            # worth stating rather than inferring from the patch list alone.
            method = str(manifest.get("install_method") or "source")
            backend = str(manifest.get("backend") or "cuda")
            origin = str(manifest.get("source") or "")
            rung = {
                "prebuilt": ("revv prebuilt binary (patched, CUDA)",
                             "Downloaded, not built here. A Mericanii fork\n"
                             "build of upstream at %s." % base),
                "upstream": ("official llama.cpp prebuilt (%s)" % backend,
                             "Official upstream binary, so the most trusted\n"
                             "rung -- but upstream ships no CUDA build for\n"
                             "Linux, so this is the %s backend. revv's\n"
                             "published numbers are CUDA and do NOT apply."
                             % backend),
                "source": ("built from source", "base commit %s" % base),
            }.get(method, ("unknown install method: %s" % method, ""))
            detail = rung[1] + (("\n" + origin) if origin else "")
            status(OK if method != "upstream" else WARN,
                   "install: %s" % rung[0], detail)

            if backend != "cuda":
                status(WARN, "not a CUDA build",
                       "Speculation and the kernel patch are CUDA-specific.\n"
                       "For the certified configuration: ./install.sh --prebuilt")
            elif "mmvq_iquant_decode.patch" in patches:
                status(OK, "kernel patch applied", "base commit %s" % base)
            else:
                status(WARN, "kernel patch NOT applied",
                       "base commit %s\nExpect ~%.1f t/s instead of %.1f (-2.5%%).\n"
                       "Get the patched build with: ./install.sh --prebuilt"
                       % (base, CERT_TS_STOCK, CERT_TS))

    print("\n" + bold("Models"))
    found = local_models()
    if not found:
        status(WARN, "no models in %s" % MODELS_DIR, "Run: revv get")
    for path in found:
        name = os.path.basename(path)
        try:
            info = read_gguf(path)
        except GGUFError as exc:
            status(FAIL, name, str(exc))
            problems += 1
            continue
        verdict, _ = classify(info, name)
        tag = OK if verdict.startswith("CERTIFIED") else WARN
        status(tag, "%s  %s  %s" % (name, gib(info.file_size),
                                    info.dominant_quant), verdict)

    print("\n" + bold("Verdict"))
    if problems == 0 and tier is not None and exe is not None:
        if tier == "12gb":
            print("  Ready. The certified configuration on this box:")
            print("    %.1f t/s decode, %.1f%% HumanEval-164, %s peak VRAM."
                  % (CERT_TS, CERT_HUMANEVAL, mib(CERT_PEAK_MIB)))
        else:
            print("  Ready. Certified weights at the %s tier (%s)."
                  % (tier.upper(), TIERS[tier]["ctx"]))
            print("  Speed and quality were measured on 12GB; this tier only")
            print("  raises the context, so expect the same or slightly better.")
        print("\n  Next:  %s" % bold("revv serve"))
        return 0
    print("  %d problem(s) above. Fix those first." % problems)
    return 1


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------

def classify(info: "GGUFInfo", filename: str) -> Tuple[str, str]:
    """Return (verdict line, explanation).

    Three states, and the distinction that matters is the draft head: without
    it there is no speculative decoding, which is the difference between
    ~37 t/s and ~20 t/s. Nothing else about a GGUF changes speed that much.
    """
    build_name = identify_build(info)
    spec = BUILDS.get(build_name) if build_name else None
    if spec is not None and spec.get("certified"):
        detail = ("This is the exact file the published numbers were measured "
                  "on.\n%.1f t/s decode, %.1f%% HumanEval-164, %s peak VRAM."
                  % (float(spec["decode_ts"]), float(spec["humaneval"]),
                     mib(int(spec["peak_mib"]))))
        if spec.get("host_ram_mib"):
            detail += ("\nMixture-of-experts: also needs ~%s of free host RAM, "
                       "because\nthe expert layers stream from it."
                       % mib(int(spec["host_ram_mib"])))
        return ("CERTIFIED (%s line)" % spec.get("line", "?"), detail)
    # Quoting the certified Qwen figures at a Gemma or Llama file would be
    # meaningless: those numbers are properties of one model on one card.
    same_family = bool(info.arch and info.arch.lower().startswith("qwen"))
    if not same_family:
        if info.has_mtp_head:
            return ("COMPATIBLE (has a draft head -- speculation will run, "
                    "no numbers for this model)",
                    "This is not the model revv was certified on (arch: %s),\n"
                    "so none of the published speed or quality figures apply.\n"
                    "It has a draft head, so speculative decoding will work.\n"
                    "Measure it yourself with revv bench."
                    % (info.arch or "unknown"))
        return ("COMPATIBLE (no draft head -- revv's levers may not apply)",
                "This is not the model revv was certified on (arch: %s), and\n"
                "it has no MTP draft head, so speculative decoding cannot run.\n"
                "If its chat template also has no thinking mode, revv has no\n"
                "lever left and will serve it with the best-known stock config\n"
                "rather than pretend to tune it. revv serve will tell you which\n"
                "case you are in.\n"
                "\n"
                "No built-in draft head. Speculation is still available via an\n"
                "external drafter: revv serve --draft <file.gguf>. Community MTP\n"
                "drafts exist for some models (the Gemma-4 family on HF, for\n"
                "one). Experimental and uncertified -- revv will not download a\n"
                "third-party drafter for you, and acceptance is a property of\n"
                "the target/drafter pair, so measure it with revv bench."
                % (info.arch or "unknown"))
    if info.has_mtp_head:
        return ("COMPATIBLE (has draft head -- full speed expected, "
                "numbers not certified)",
                "The MTP draft head is present, so speculative decoding will\n"
                "work and speed should land near the certified figure.\n"
                "Quality is unmeasured: revv has no numbers for this file.")
    return ("COMPATIBLE (no draft head -- speculation unavailable, "
            "expect ~%.0f not ~%.0f t/s)" % (CERT_TS_NOSPEC, CERT_TS),
            "No blk.N.nextn.* tensors. Quantizers below ~8.4 GiB strip the\n"
            "MTP head, and some conversion scripts drop it. Without it\n"
            "speculative decoding cannot run and you lose ~40% of decode\n"
            "speed. revv will start the server with speculation disabled.\n"
            "\n"
            "No built-in draft head. Speculation is still available via an\n"
            "external drafter: revv serve --draft <file.gguf>. Community MTP\n"
            "drafts exist for some models (the Gemma-4 family on HF, for one).\n"
            "Experimental and uncertified -- revv will not download a\n"
            "third-party drafter for you, and acceptance is a property of the\n"
            "target/drafter pair, so measure it with revv bench.")


def cmd_inspect(args: argparse.Namespace) -> int:
    path = args.file
    if not os.path.isfile(path):
        candidate = os.path.join(MODELS_DIR, path)
        if os.path.isfile(candidate):
            path = candidate
        else:
            die("no such file: %s" % args.file)
    try:
        info = read_gguf(path)
    except GGUFError as exc:
        die(str(exc))
        return 1  # unreachable; keeps type checkers honest

    print(bold(os.path.basename(path)))
    print("  path            %s" % path)
    print("  size            %s  (%s bytes)"
          % (gib(info.file_size), "{:,}".format(info.file_size)))
    print("  gguf version    %d" % info.version)
    print("  architecture    %s" % (info.arch or "unknown"))
    print("  quantization    %s" % info.dominant_quant)
    print("  vocab           %s" % ("{:,}".format(info.n_vocab)
                                    if info.n_vocab else "unknown"))
    print("  layers          %s" % (info.n_layer if info.n_layer else "unknown"))
    print("  train context   %s" % ("{:,}".format(info.n_ctx_train)
                                    if info.n_ctx_train else "unknown"))
    print("  tensors         %d" % info.n_tensors)

    print("\n  " + bold("tensor types"))
    ranked = sorted(info.type_bytes.items(), key=lambda kv: -kv[1])
    for tname, nbytes in ranked:
        print("    %-10s %5d tensors  %10s"
              % (tname, info.type_counts.get(tname, 0), gib(nbytes)))

    print("\n  " + bold("MTP draft head"))
    if info.has_mtp_head:
        print("    present -- %d tensors" % len(info.mtp_tensors))
        for name in info.mtp_tensors[:6]:
            print("      %s" % name)
        if len(info.mtp_tensors) > 6:
            print("      ... and %d more" % (len(info.mtp_tensors) - 6))
    else:
        print("    %s" % red("absent"))

    verdict, why = classify(info, path)
    print("\n  " + bold("verdict"))
    colour = green if verdict.startswith("CERTIFIED") else yellow
    print("    %s" % colour(verdict))
    for line in why.splitlines():
        print("    %s" % dim(line))
    return 0


# ---------------------------------------------------------------------------
# adopt: reuse GGUFs already downloaded by ollama or LM Studio
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Found record
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Found:
    path: str          # absolute path to the GGUF (or, for ollama, its blob)
    source: str         # "ollama" | "lmstudio"
    label: str          # "qwen3:latest" or "TheBloke/foo/model-Q4"
    size: int           # bytes


# ---------------------------------------------------------------------------
# Ollama discovery
#
# <root>/manifests/<registry>/<namespace>/<name>/<tag>  -- a JSON manifest
# <root>/blobs/sha256-<hex>                              -- the actual data
# We only ever open() these files for reading; nothing here writes into an
# ollama tree.
# ---------------------------------------------------------------------------

_OLLAMA_MODEL_MEDIA_TYPE = "application/vnd.ollama.image.model"


def _default_ollama_root() -> str:
    env = os.environ.get("OLLAMA_MODELS")
    if env:
        return env
    return os.path.join(os.path.expanduser("~"), ".ollama", "models")


def _ollama_label(manifest_path: str, manifests_dir: str) -> str:
    # manifest_path is <manifests_dir>/<registry>/<namespace>/<name>/<tag>.
    # "library" is ollama's default namespace and is conventionally dropped
    # from the short name; any other namespace is kept so e.g. a third-party
    # publisher's "foo/bar:latest" doesn't collide with the official "bar".
    rel = os.path.relpath(manifest_path, manifests_dir)
    parts = rel.split(os.sep)
    tag = parts[-1]
    name = parts[-2] if len(parts) >= 2 else tag
    namespace = parts[-3] if len(parts) >= 3 else ""
    if namespace and namespace != "library":
        return "%s/%s:%s" % (namespace, name, tag)
    return "%s:%s" % (name, tag)


def _parse_ollama_manifest(manifest_path: str, manifests_dir: str,
                            blobs_dir: str) -> Optional[Found]:
    # Every failure mode here (unreadable file, bad JSON, missing/odd
    # fields, absent blob) returns None rather than raising: one broken
    # manifest must never take down a scan of an otherwise-healthy store.
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            raw = f.read()
    except OSError:
        return None  # unreadable (permissions, dangling symlink, ...)

    try:
        manifest = json.loads(raw)
    except ValueError:
        return None  # malformed or truncated JSON

    if not isinstance(manifest, dict):
        return None
    layers = manifest.get("layers")
    if not isinstance(layers, list):
        return None

    model_layer = None
    for layer in layers:
        if isinstance(layer, dict) and layer.get("mediaType") == _OLLAMA_MODEL_MEDIA_TYPE:
            model_layer = layer
            break
    if model_layer is None:
        return None  # no model-mediaType layer in this manifest

    digest = model_layer.get("digest")
    if not isinstance(digest, str) or ":" not in digest:
        return None
    algo, _, hexpart = digest.partition(":")
    if algo != "sha256" or not hexpart:
        return None

    blob_path = os.path.join(blobs_dir, "sha256-%s" % hexpart)
    if not os.path.isfile(blob_path):
        return None  # manifest references a blob that isn't (or no longer) there

    try:
        size = int(model_layer.get("size") or os.path.getsize(blob_path))
    except (TypeError, ValueError, OSError):
        return None

    label = _ollama_label(manifest_path, manifests_dir)
    return Found(path=blob_path, source="ollama", label=label, size=size)


def scan_ollama(root: Optional[str] = None) -> List[Found]:
    if root is None:
        root = _default_ollama_root()
    manifests_dir = os.path.join(root, "manifests")
    blobs_dir = os.path.join(root, "blobs")
    found: List[Found] = []
    if not os.path.isdir(manifests_dir):
        return found

    # onerror=no-op: a directory we can't list (permissions) is skipped
    # rather than raising out of os.walk.
    for dirpath, _dirnames, filenames in os.walk(manifests_dir, onerror=lambda _e: None):
        for fname in filenames:
            item = _parse_ollama_manifest(os.path.join(dirpath, fname),
                                           manifests_dir, blobs_dir)
            if item is not None:
                found.append(item)
    return found


# ---------------------------------------------------------------------------
# LM Studio discovery
# ---------------------------------------------------------------------------

_LMSTUDIO_MAX_DEPTH = 6
_MIN_GGUF_BYTES = 1024 * 1024  # below this, LM Studio left a partial download


def _default_lmstudio_roots() -> List[str]:
    roots: List[str] = []
    env = os.environ.get("LMSTUDIO_MODELS_DIR")
    if env:
        roots.append(env)
    home = os.path.expanduser("~")
    roots.append(os.path.join(home, ".lmstudio", "models"))
    roots.append(os.path.join(home, ".cache", "lm-studio", "models"))
    roots.append(os.path.join(home, "Library", "Application Support", "LM Studio", "models"))
    return roots


def _scan_lmstudio_root(root: str) -> List[Found]:
    results: List[Found] = []
    root = os.path.abspath(root)
    # os.walk defaults to followlinks=False, so a symlinked subdirectory is
    # never descended into -- that alone satisfies "follow no symlinks out
    # of the root" for directories. Symlinked *files* are still listed and
    # opened normally, which is exactly what the test fixtures rely on to
    # avoid copying multi-gigabyte models.
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda _e: None):
        rel = os.path.relpath(dirpath, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth >= _LMSTUDIO_MAX_DEPTH:
            dirnames[:] = []  # prune: stop descending, but this dir's own files still count

        for fname in filenames:
            if not fname.lower().endswith(".gguf"):
                continue
            fpath = os.path.join(dirpath, fname)
            try:
                size = os.path.getsize(fpath)
            except OSError:
                continue  # dangling symlink or a race with a deletion
            if size < _MIN_GGUF_BYTES:
                continue  # partial download

            relfile = os.path.relpath(fpath, root)
            label = os.path.splitext(relfile)[0].replace(os.sep, "/")
            results.append(Found(path=fpath, source="lmstudio", label=label, size=size))
    return results


def scan_lmstudio(roots: Optional[List[str]] = None) -> List[Found]:
    if roots is None:
        roots = _default_lmstudio_roots()
    found: List[Found] = []
    for root in roots:
        if root and os.path.isdir(root):
            found.extend(_scan_lmstudio_root(root))
    return found


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def registry_path() -> str:
    return os.path.join(REVV_HOME, "registry.json")


def load_registry() -> Dict[str, Dict[str, str]]:
    path = registry_path()
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        status(WARN, "registry.json is corrupt or unreadable -- treating as empty", path)
        return {}
    if not isinstance(data, dict) or not isinstance(data.get("models"), dict):
        status(WARN, "registry.json has an unexpected shape -- treating as empty", path)
        return {}
    return data["models"]


def save_registry(models: Dict[str, Dict[str, str]]) -> None:
    # Atomic write: a crash or concurrent `revv adopt` mid-write leaves either
    # the old registry or the new one on disk, never a half-written file.
    os.makedirs(REVV_HOME, exist_ok=True)
    path = registry_path()
    tmp = path + ".tmp"
    payload = {"version": 1, "models": models}
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def registry_lookup(name: str) -> Optional[str]:
    entry = load_registry().get(name)
    if not entry:
        return None
    path = entry.get("path")
    if not path or not os.path.exists(path):
        # A stale entry (the source file moved/was deleted since adoption)
        # must look exactly like "not registered", not surface a confusing
        # llama-server error further down the call chain.
        return None
    return path


# ---------------------------------------------------------------------------
# adopt
# ---------------------------------------------------------------------------

def _slugify(label: str) -> str:
    """"qwen3:latest" -> "qwen3-latest"; runs of non-alphanumerics collapse
    to one "-" and leading/trailing "-" are dropped, without regex (not on
    the allowed stdlib list for this module)."""
    chars = [ch if ch.isalnum() else "-" for ch in label.lower()]
    parts = [p for p in "".join(chars).split("-") if p]
    return "-".join(parts)


def _unique_name(base: str, path: str, models: Dict[str, Dict[str, str]]) -> str:
    """A registry key for `path`: re-adopting the same path under the same
    base name is idempotent (returns `base` again, no rename); a genuinely
    different model that slugifies to the same base gets "-2", "-3", ..."""
    if base not in models or models[base].get("path") == path:
        return base
    n = 2
    while True:
        candidate = "%s-%d" % (base, n)
        if candidate not in models or models[candidate].get("path") == path:
            return candidate
        n += 1


def cmd_adopt(args: argparse.Namespace) -> int:
    # READ-ONLY GUARANTEE: this function (and everything it calls) only ever
    # reads inside an Ollama or LM Studio directory -- no write, move,
    # delete, or chmod. The only file it ever writes is revv's own
    # registry.json, via save_registry()'s atomic tmp-then-replace.
    source = getattr(args, "source", None)
    do_all = bool(getattr(args, "all", False))
    dry_run = bool(getattr(args, "dry_run", False))

    ollama_root = _default_ollama_root()
    lmstudio_roots = _default_lmstudio_roots()

    found: List[Found] = []
    searched: List[str] = []
    if source in (None, "ollama"):
        searched.append(os.path.join(ollama_root, "manifests"))
        found.extend(scan_ollama(ollama_root))
    if source in (None, "lmstudio"):
        searched.extend(lmstudio_roots)
        found.extend(scan_lmstudio(lmstudio_roots))

    if not found:
        # Neither tool installed is a perfectly normal machine, not an error.
        print(bold("no models found."))
        print("  looked in:")
        for path in searched:
            tag = "exists" if os.path.isdir(path) else "not found"
            print("    %s  %s" % (path, dim("(%s)" % tag)))
        return 0

    models = load_registry()
    adopted_names: List[str] = []
    n_adopted = 0
    n_skipped = 0

    for item in sorted(found, key=lambda f: (f.source, f.label)):
        print(bold("%s  %s" % (item.label, dim("(%s)" % item.source))))
        try:
            info = read_gguf(item.path)
        except GGUFError as exc:
            status(FAIL, "could not read GGUF header", str(exc))
            n_skipped += 1
            continue

        is_qwen = info.arch is not None and info.arch.lower().startswith("qwen")
        verdict, _why = classify(info, item.path)

        status(OK, "size=%s  quant=%s  mtp=%s"
               % (gib(item.size), info.dominant_quant,
                  "yes" if info.has_mtp_head else "no"))
        status(OK, "verdict: %s" % verdict)

        if not is_qwen and not do_all:
            status(WARN, "skipped (not a Qwen model: %s)" % (info.arch or "unknown"))
            n_skipped += 1
            continue
        if not is_qwen and do_all:
            status(WARN, "adopting anyway (--all): revv's numbers do not apply to this model")

        base = _slugify(item.label) or "model"
        name = _unique_name(base, item.path, models)
        models[name] = {
            "path": item.path,
            "source": item.source,
            "label": item.label,
            "quant": info.dominant_quant,
            "mtp": info.has_mtp_head,
        }
        adopted_names.append(name)
        n_adopted += 1
        status(OK, "registered as %s" % name)

    if not dry_run:
        save_registry(models)

    print()
    print(bold("%d adopted, %d skipped" % (n_adopted, n_skipped)))
    if dry_run:
        print("  (dry run -- registry not written)")
    for name in adopted_names:
        print("  next: %s" % green("revv serve %s" % name))
    return 0


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------

def hf_url(filename: str, repo: Optional[str] = None) -> str:
    return "https://huggingface.co/%s/resolve/main/%s?download=true" % (
        repo or HF_REPO, filename)


def cmd_get(args: argparse.Namespace) -> int:
    build = args.build
    if build is None:
        if args.tier and args.tier.lower() in MODEL_LINES:
            build = MODEL_LINES[args.tier.lower()]
        elif args.tier:
            tier = args.tier.lower()
            if tier not in TIERS:
                die("unknown target: %s" % args.tier,
                    "a model line:  %s\n"
                    "or a VRAM tier: %s"
                    % (", ".join(sorted(MODEL_LINES)), ", ".join(sorted(TIERS))))
        else:
            gpus, err = detect_gpus()
            if err is not None:
                print("%s %s -- assuming the 12GB tier." % (yellow("note:"), err))
                tier = "12gb"
            else:
                best = max(gpus, key=lambda g: g.free_mib)
                detected = tier_for(best.free_mib)
                if detected is None:
                    die("%s has only %s free (of %s)"
                        % (best.name, mib(best.free_mib), mib(best.total_mib)),
                        "revv needs %s free. See README.md."
                        % mib(VRAM_MIN_FREE_MIB))
                tier = detected
                print("Detected %s -> %s tier." % (best.name, tier.upper()))
        # Every tier ships the same weights; see the note on BUILDS.
        build = DEFAULT_BUILD

    if build not in BUILDS:
        die("unknown build: %s" % build,
            "one of: %s" % ", ".join(sorted(BUILDS)))

    spec = BUILDS[build]
    filename = str(spec["file"])
    dest = os.path.join(MODELS_DIR, filename)
    if os.path.isfile(dest) and not args.force:
        print("Already present: %s" % dest)
        print("Run %s to verify it." % bold("revv inspect %s" % filename))
        return 0

    if not spec["certified"]:
        print("%s %s is not the certified build." % (yellow("note:"), build))
        print("       %s" % spec["note"])

    url = hf_url(filename, str(spec.get("repo") or HF_REPO))
    line = str(spec.get("line") or "")
    print("Downloading %s%s" % (build, " (%s line)" % line if line else ""))
    if spec.get("decode_ts"):
        print("  measured %.1f t/s, %.1f%% HumanEval-164 on an RTX 3060"
              % (float(spec["decode_ts"]), float(spec["humaneval"])))
    if spec.get("host_ram_mib"):
        print("  needs ~%s of free host RAM as well as the VRAM"
              % mib(int(spec["host_ram_mib"])))
    print("  from  %s" % url.split("?")[0])
    print("  to    %s" % dest)
    size = head_size(url)
    if size:
        print("  size  %s" % gib(size))
        free = shutil.disk_usage(os.path.dirname(dest) or ".").free \
            if os.path.isdir(MODELS_DIR) else shutil.disk_usage(
                os.path.expanduser("~")).free
        if free < size * 1.05:
            die("not enough disk space: %s free, need %s"
                % (gib(free), gib(size * 1.05)),
                "Free up space, or set REVV_HOME to a bigger volume:\n"
                "REVV_HOME=/mnt/big/.revv revv get")
    print()
    try:
        final = download(url, dest, expected_size=size)
    except DownloadError as exc:
        die(str(exc))
        return 1
    print("\nDownloaded %s" % gib(final))

    try:
        info = read_gguf(dest)
    except GGUFError as exc:
        die("the downloaded file does not parse as a GGUF: %s" % exc,
            "Delete it and retry:\n  rm %s\n  revv get %s" % (dest, build))
        return 1
    verdict, _ = classify(info, dest)
    print("Verified: %s, %s vocab, draft head %s"
          % (info.dominant_quant,
             "{:,}".format(info.n_vocab) if info.n_vocab else "?",
             "present" if info.has_mtp_head else "ABSENT"))
    print("Status:   %s" % verdict)
    print("\nNext:  %s" % bold("revv serve"))
    return 0


# ---------------------------------------------------------------------------
# serve: a supervised llama-server behind a stable local port
#
# The user's tools point at one port and never move. Behind it revv runs
# llama-server on an ephemeral port and can restart it in a different
# configuration without the client noticing. That is what makes `revv toggle`
# and `revv compare` possible, and it is the same trick llama-swap uses.
# ---------------------------------------------------------------------------

MODE_REVV = "revv"
MODE_STOCK = "stock"

# What llama-server advertises at /v1/models. In single-model mode the
# "model" field of a request is ignored, but tools still want a name.
MODEL_ALIAS = "revv"

# STOCK is llama.cpp's own defaults for the three levers revv changes:
# speculation, KV precision, and the thinking switch. Same weights, same GPU,
# same context. It is a control for revv's configuration, NOT a measurement of
# ollama or of anyone else's product.
MODE_HELP = {
    MODE_REVV: "tuned for this model",
    MODE_STOCK: "llama.cpp defaults: no speculation, f16 KV, thinking on",
}


def mode_description(mode: str, plan: Optional["LaunchPlan"]) -> str:
    """Describe the mode in terms of what it does to THIS model.

    A fixed string here would claim speculation and a thinking switch on models
    that have neither, which is how revv came to run a Gemma file 2.5% slower
    than stock while reporting it as the tuned configuration.
    """
    if mode != MODE_REVV or plan is None:
        return MODE_HELP[mode]
    if plan.is_noop:
        return "no lever applies to this model; best-known stock config"
    return "tuned: " + ", ".join(plan.levers)


class LaunchPlan:
    """What revv will actually do to this model, and why.

    revv's flag set was certified on one model. Applied blindly to a different
    one it can be a net LOSS: a field report measured a Gemma-4-12B running
    2.5% SLOWER in revv mode than stock, because none of the levers applied --
    no draft head so no speculation, no thinking switch to disable, and
    quantized KV, which is a compute tax that only pays for itself when VRAM is
    tight. A 7 GB model on a 12 GB card is not tight. So each lever is decided
    per model rather than assumed.
    """

    def __init__(self, ctx: int, kv: str, use_spec: bool, thinking_off: bool,
                 notes: List[str], estimated_peak: Optional[int],
                 draft_path: Optional[str] = None,
                 draft_spec_type: Optional[str] = None,
                 ctx_checkpoints: Optional[int] = None,
                 n_cpu_moe: Optional[int] = None,
                 build_name: Optional[str] = None,
                 n_threads: Optional[int] = None) -> None:
        self.ctx = ctx
        self.kv = kv
        self.use_spec = use_spec
        self.thinking_off = thinking_off
        self.notes = notes
        self.estimated_peak = estimated_peak
        # An external drafter: a second, smaller GGUF that proposes tokens the
        # target model verifies. Experimental and uncertified -- acceptance is
        # a property of the PAIR, so a drafter that works well for one target
        # can be worthless for a finetune of it.
        self.draft_path = draft_path
        self.draft_spec_type = draft_spec_type
        # None = leave llama-server's default. 0 = disable, because the
        # checkpoint allocation would not fit and would kill request two.
        self.ctx_checkpoints = ctx_checkpoints
        # MoE lines stream expert weights from host RAM; this is how many
        # expert layers stay on the CPU. Certified per build, not guessed.
        self.n_cpu_moe = n_cpu_moe
        self.build_name = build_name
        # CPU-MoE offload puts host-RAM bandwidth on the critical path for
        # every token, which makes the thread count a decode-speed lever, not
        # just a load-time one. None = leave llama-server's default; only set
        # when n_cpu_moe is also set.
        self.n_threads = n_threads

    @property
    def levers(self) -> List[str]:
        active = []
        if self.draft_path:
            active.append("speculation via external drafter %s [experimental]"
                          % os.path.basename(self.draft_path))
        elif self.use_spec:
            # Certified on both tiers: n-gram hits skip the MTP pass entirely
            # (first-success-wins), so this is a strict addition over MTP
            # alone, not a replacement.
            active.append("speculation (n-gram + MTP n=%d drafter chain)"
                          % SPEC_N_MAX)
        if self.thinking_off:
            active.append("thinking off")
        if self.kv != "f16":
            active.append("%s KV (for capacity)" % self.kv)
        if self.ctx_checkpoints == 0:
            active.append("context checkpoints off (VRAM)")
        if self.n_cpu_moe is not None:
            active.append("%d expert layers on CPU" % self.n_cpu_moe)
        if self.n_threads is not None:
            active.append("-t %d (CPU-MoE decode)" % self.n_threads)
        return active

    @property
    def is_noop(self) -> bool:
        """True when revv mode is materially the same as stock.

        No draft head, no thinking switch, and f16 KV already optimal means
        there is nothing left for revv to change. Pretending otherwise would
        make `revv compare` a fake A/B.
        """
        return (not self.use_spec and not self.draft_path
                and not self.thinking_off and self.kv == "f16")


def draft_overhead_mib(draft: Optional["GGUFInfo"], ctx: int,
                       kv: str) -> int:
    """VRAM the external drafter adds: its weights plus its own KV cache."""
    if draft is None:
        return 0
    weights = draft.file_size / (1024.0 * 1024.0)
    rate = kv_mib_per_token(draft, kv) or 0.0
    return int(round(weights + ctx * rate))


def plan_launch(info: "GGUFInfo", tier: str, explicit_ctx: Optional[int],
                free_mib: Optional[int],
                draft: Optional["GGUFInfo"] = None) -> LaunchPlan:
    t = TIERS[tier]
    preferred = explicit_ctx if explicit_ctx is not None else int(t["ctx"])
    notes = []      # type: List[str]

    # A recognised build brings its own certified settings. These are measured
    # for that specific file, so they beat anything the generic estimator
    # would derive.
    build_name = identify_build(info)
    spec = BUILDS.get(build_name) if build_name else None
    n_cpu_moe = None    # type: Optional[int]
    if spec is not None and spec.get("n_cpu_moe") is not None:
        n_cpu_moe = int(spec["n_cpu_moe"])
        need_ram = int(spec.get("host_ram_mib") or 0)
        total_ram, avail_ram = host_ram_mib()
        notes.append("mixture-of-experts build: %d expert layers stay on the "
                     "CPU and stream from host RAM, which is where the speed "
                     "comes from" % n_cpu_moe)
        if avail_ram is not None and need_ram and avail_ram < need_ram:
            notes.append("WARNING only %s of host RAM is available and this "
                         "build wants ~%s. It will thrash or be OOM-killed; "
                         "close something, or use the flagship line instead"
                         % (mib(avail_ram), mib(need_ram)))
        elif total_ram is not None and need_ram and total_ram < need_ram + 4096:
            notes.append("WARNING this machine has %s of RAM total and the "
                         "build wants ~%s free. That is tight once the OS and "
                         "your editor are counted" % (mib(total_ram),
                                                      mib(need_ram)))
        elif avail_ram is not None:
            notes.append("host RAM ok: %s available, ~%s needed"
                         % (mib(avail_ram), mib(need_ram)))

    # CPU-MoE offload makes host RAM bandwidth the bottleneck for every
    # token, and -t controls how many threads compete for it. Measured on a
    # 3060 + Ryzen 3600: -t 8 is +14.4% over the server default, while the
    # full logical count loses 5-15% to oversubscription. Only relevant when
    # experts are actually on the CPU -- a GPU-resident build has no such
    # bottleneck.
    n_threads = None    # type: Optional[int]
    if n_cpu_moe:
        n_threads = physical_core_count()
        notes.append("-t %d: physical core count (clamped 4-8), tuned for "
                     "CPU-MoE decode where host RAM bandwidth is the "
                     "bottleneck" % n_threads)

    # An external drafter takes precedence: the user asked for it explicitly,
    # and it is the only way to speculate on a file with no built-in head.
    draft_spec_type = None      # type: Optional[str]
    if draft is not None:
        draft_spec_type = (SPEC_TYPE if draft.has_mtp_head
                           else "draft-simple")
        notes.append("external drafter %s (%s, %s) -- speculation is "
                     "EXPERIMENTAL and uncertified; acceptance depends on the "
                     "target/drafter pair, so check it with revv bench"
                     % (os.path.basename(draft.path), draft.dominant_quant,
                        gib(draft.file_size)))
        if draft.n_vocab and info.n_vocab and draft.n_vocab != info.n_vocab:
            notes.append("WARNING vocab mismatch: target %s vs drafter %s. "
                         "llama-server will usually refuse this pair"
                         % ("{:,}".format(info.n_vocab),
                            "{:,}".format(draft.n_vocab)))

    use_spec = info.has_mtp_head
    if not use_spec and draft is None:
        notes.append("no MTP draft head in this file, so speculative decoding "
                     "is off (that is where most of revv's speed comes from); "
                     "an external drafter can supply it: --draft <file.gguf>")

    # The built-in n-gram+MTP chain (see SPEC_TYPE_CHAIN in build_server_argv)
    # runs whenever the file speculates on its own MTP head and there is no
    # external drafter overriding it. It costs real VRAM, so that cost has to
    # be in the budget before the context ladder picks a rung -- except for
    # the n_cpu_moe speed tier, whose certified peak_mib already measures the
    # chain running; counting it twice there would shrink its context for no
    # reason.
    chain_active = use_spec and draft is None and n_cpu_moe is None
    chain_mib = SPEC_NGRAM_CHAIN_MIB if chain_active else 0

    # How much headroom to leave unclaimed: 150 MiB when the peak below is
    # purely a measurement, 250 when any part of it is estimated. Decided here
    # rather than at the top of the function because it depends on chain_mib
    # and the drafter, which are the estimated terms. See vram_margin_for().
    margin = vram_margin_for(info, chain_mib, draft)

    thinking_off = info.supports_thinking
    if not thinking_off:
        notes.append("this model's chat template has no thinking switch, so "
                     "there is nothing to disable")

    # Context: the largest rung that fits, measured at q8_0 so the choice is
    # about capacity rather than precision. An EXPLICIT --ctx is honoured
    # exactly and never snapped to the ladder -- silently handing a user more
    # context than they asked for is how you turn a deliberate choice into an
    # OOM.
    ctx = preferred
    if explicit_ctx is not None:
        peak = model_peak_mib(info, ctx, str(t["kv"]))
        if peak is not None and free_mib is not None:
            peak += draft_overhead_mib(draft, ctx, str(t["kv"])) + chain_mib
            if peak + margin > free_mib:
                notes.append("--ctx %s needs ~%s but only %s is free; expect a "
                             "CUDA OOM" % ("{:,}".format(ctx), mib(peak),
                                           mib(free_mib)))
    elif free_mib is not None:
        chosen = None
        for cand in CONTEXT_LADDER:
            if cand > preferred:
                continue
            peak = model_peak_mib(info, cand, "q8_0")
            if peak is not None:
                peak += draft_overhead_mib(draft, cand, "q8_0") + chain_mib
            if peak is None or peak + margin <= free_mib:
                chosen = cand
                break
        if chosen is None:
            # Nothing fits at q8_0. Retry at q4_0 before giving up context:
            # halving the cache is cheaper than an eighth of the context.
            for cand in CONTEXT_LADDER:
                if cand > preferred:
                    continue
                peak = model_peak_mib(info, cand, "q4_0")
                if peak is not None:
                    peak += draft_overhead_mib(draft, cand, "q4_0") + chain_mib
                if peak is None or peak + margin <= free_mib:
                    chosen = cand
                    break
        if chosen is None:
            chosen = CONTEXT_LADDER[-1]
        if chosen != preferred:
            notes.append("context reduced %s -> %s to fit %s free"
                         % ("{:,}".format(preferred), "{:,}".format(chosen),
                            mib(free_mib)))
        ctx = chosen

    # KV precision by need, not by habit. f16 is the faster kernel; quantizing
    # is a compute tax paid only to buy capacity we would not otherwise have.
    kv = str(t["kv"])
    if free_mib is None:
        pass                            # tier was forced; keep its setting
    else:
        f16_peak = model_peak_mib(info, ctx, "f16")
        if f16_peak is not None:
            f16_peak += draft_overhead_mib(draft, ctx, "f16") + chain_mib
        if f16_peak is not None and f16_peak + margin <= free_mib:
            if kv != "f16":
                notes.append("f16 KV fits (~%s of %s free) and is the faster "
                             "kernel, so revv is not quantizing the cache"
                             % (mib(f16_peak), mib(free_mib)))
            kv = "f16"
        else:
            for cand in ("q8_0", "q4_0"):
                peak = model_peak_mib(info, ctx, cand)
                if peak is not None:
                    peak += draft_overhead_mib(draft, ctx, cand) + chain_mib
                if peak is None or peak + margin <= free_mib:
                    kv = cand
                    break
            else:
                kv = "q4_0"
            if f16_peak is not None:
                notes.append("f16 KV would need ~%s, over the %s free, so the "
                             "cache is %s to fit"
                             % (mib(f16_peak), mib(free_mib), kv))

    peak = model_peak_mib(info, ctx, kv)
    if peak is not None:
        peak += draft_overhead_mib(draft, ctx, kv) + chain_mib
        if free_mib is not None and peak + margin > free_mib:
            # Two distinct situations, and saying "exceeds" for both is simply
            # false: peak can be UNDER free and still trip this, because the
            # test includes the safety margin. A user who reads "~11,530 MiB
            # exceeds 11,640 MiB free" and can do arithmetic stops trusting
            # the rest of the output. Name the drafter only when there is one.
            why = " once the drafter is counted" if draft is not None else ""
            if peak > free_mib:
                notes.append("WARNING estimated peak ~%s exceeds the %s free%s;"
                             " reduce --ctx if this OOMs"
                             % (mib(peak), mib(free_mib), why))
            else:
                notes.append("WARNING estimated peak ~%s leaves only ~%s of the "
                             "%s free%s, under the %s revv keeps in reserve for "
                             "allocator fragmentation; reduce --ctx if this OOMs"
                             % (mib(peak), mib(free_mib - peak), mib(free_mib),
                                why, mib(margin)))

    # Context checkpoints are allocated lazily, so leaving them on near the
    # ceiling produces a server that passes its health check and then dies on
    # the second request. Turn them off before that can happen.
    ctx_checkpoints = None      # type: Optional[int]
    if free_mib is None:
        # Tier forced with --tier, so there is no VRAM reading and the ladder
        # never ran: ctx is whatever the tier declares, which on the 12GB tier
        # is 16384 -- the rung a real 12GB card cannot hold. Headroom here is
        # not "large", it is UNKNOWN, and the failure mode of guessing wrong is
        # the nasty one: the server passes its health check, serves one
        # request, and dies on the next with a cudaGraphInstantiate error that
        # names neither memory nor checkpoints. Disabling checkpoints costs
        # ~2.7% on short prompts and gains at depth, so it is the cheap side of
        # the bet. Always take it when we are flying blind.
        ctx_checkpoints = 0
        notes.append("WARNING --tier was given, so revv did NOT read free "
                     "VRAM: the context above is the tier's declared %s, not a "
                     "measured fit, and could OOM. Context checkpoints are "
                     "disabled (-ctxcp 0) because headroom is unknown. Drop "
                     "--tier to let revv size this against the actual card."
                     % "{:,}".format(ctx))
    elif peak is not None:
        headroom = free_mib - peak
        if headroom < CHECKPOINT_HEADROOM_MIB:
            ctx_checkpoints = 0
            notes.append("context checkpoints disabled (-ctxcp 0): only ~%s "
                         "would be left after load, and the first checkpoint "
                         "alone wants ~%s. Left on, this config passes its "
                         "health check and then dies on the second request"
                         % (mib(max(0, headroom)), mib(CHECKPOINT_MIB_EACH)))

    return LaunchPlan(ctx, kv, use_spec, thinking_off, notes, peak,
                      draft.path if draft else None, draft_spec_type,
                      ctx_checkpoints, n_cpu_moe, build_name, n_threads)


def build_server_argv(exe: str, model: str, plan: LaunchPlan, port: int,
                      mode: str, passthrough: Sequence[str]) -> List[str]:
    """Apply the plan. Which levers are in the revv arm is decided per model."""
    argv = [exe, "-m", model, "-ngl", "99", "-c", str(plan.ctx),
            # Certified at one slot. Concurrency splits the context and was
            # never measured, and the 12GB tier has 86 MiB of headroom.
            "--parallel", "1",
            # A stable id in /v1/models, so clients keep working across a mode
            # switch and users have one short name to configure. Cosmetic: it
            # is identical in both modes and cannot affect the measurement.
            "-a", MODEL_ALIAS,
            "--host", "127.0.0.1", "--port", str(port)]
    if mode == MODE_REVV:
        argv += [
            "-fa", "on",
            "-ctk", plan.kv, "-ctv", plan.kv,
            # The RAM prompt cache delivers ZERO reuse on this hybrid
            # architecture (measured against a control) while costing ~1 GB
            # per 8K tokens. It is off on both counts.
            "--cache-ram", "0",
            "--no-cache-idle-slots",
        ]
        if plan.ctx_checkpoints is not None:
            argv += ["-ctxcp", str(plan.ctx_checkpoints)]
        if plan.n_cpu_moe is not None:
            argv += ["--n-cpu-moe", str(plan.n_cpu_moe)]
        if plan.n_threads is not None:
            argv += ["-t", str(plan.n_threads)]
        argv += [
            # The thinking switch is read by the jinja engine, so --jinja must
            # be set and must come first. Disabling thinking is the largest
            # single effect in the stack: ~2.8x wall-clock per task.
            "--jinja",
        ]
        # Only claim the levers this model can actually use. Passing
        # --spec-type to a file with no draft head makes llama-server fail to
        # start with an opaque message; disabling a thinking mode the template
        # does not implement is simply a lie in the log.
        if plan.thinking_off:
            argv += thinking_off_flags(exe)
        if plan.draft_path:
            # An external drafter. draft-mtp when the file carries an MTP head,
            # draft-simple for an ordinary small model used as the drafter.
            argv += ["--spec-type", plan.draft_spec_type or "draft-simple",
                     "--spec-draft-model", plan.draft_path,
                     # Keep the drafter on the GPU; a CPU drafter's latency
                     # eats the entire speculation win at batch 1.
                     "--spec-draft-ngl", "99",
                     "--spec-draft-n-max", str(SPEC_N_MAX)]
        elif plan.use_spec:
            # The certified drafter chain: stack an n-gram matcher in front
            # of MTP. Certified 2026-09-05 on BOTH tiers -- originally
            # speed-tier-only, now shipped on the flagship too (2.81-6.10x on
            # editing workloads, inert and byte-identical on plain
            # generation). Do not widen this further than "any build that
            # speculates through its own MTP head" without re-measuring;
            # acceptance is a property of the specific target model.
            argv += ["--spec-type", SPEC_TYPE_CHAIN,
                     "--spec-draft-n-max", str(SPEC_N_MAX),
                     "--spec-ngram-simple-size-m", str(SPEC_NGRAM_SIZE_M)]
    argv += list(passthrough)
    return argv


_HELP_CACHE = {}    # type: Dict[str, str]


def server_supports(exe: str, flag: str) -> bool:
    """Does this llama-server build accept `flag`?

    Probed from --help once per binary. Needed because --chat-template-kwargs
    is deprecated upstream in favour of --reasoning, but older builds have only
    the former, and revv has to run on both.
    """
    text = _HELP_CACHE.get(exe)
    if text is None:
        try:
            out = subprocess.run([exe, "--help"], capture_output=True,
                                 text=True, timeout=30)
            text = (out.stdout or "") + (out.stderr or "")
        except (OSError, subprocess.SubprocessError):
            text = ""
        _HELP_CACHE[exe] = text
    return flag in text


def thinking_off_flags(exe: str) -> List[str]:
    """The flags that disable thinking server-side, for this build.

    `--reasoning off` and `--chat-template-kwargs '{"enable_thinking":false}'`
    are the same mechanism: arg.cpp maps both onto
    default_template_kwargs["enable_thinking"]="false", which the server seeds
    every request from. --reasoning is simply the spelling that is not on an
    upstream removal path.
    """
    if server_supports(exe, "--reasoning"):
        return ["--reasoning", "off"]
    return ["--chat-template-kwargs", '{"enable_thinking":false}']


DEFAULT_PORT = 8080
PORT_FALLBACK_TRIES = 8


def port_is_free(host: str, port: int) -> bool:
    sock = socket.socket()
    try:
        # SO_REUSEADDR matches what ThreadingHTTPServer will do, so this probe
        # answers the same question the real bind will ask a moment later.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def resolve_port(host: str, explicit: Optional[int]) -> int:
    """An explicit --port is exact-or-fail; the default walks forward.

    Ports get taken by things that have nothing to do with revv -- the WSL2
    field box had an unrelated python service on 8080 -- and failing to start
    over that is a worse default than moving one port along and saying so.
    """
    if explicit is not None:
        if not port_is_free(host, explicit):
            die("port %d on %s is already in use" % (explicit, host),
                "Something else is listening there. Free it, or pick another:\n"
                "  revv up --port %d\n"
                "Leaving --port off lets revv find a free port itself."
                % (explicit + 1))
        return explicit
    for candidate in range(DEFAULT_PORT, DEFAULT_PORT + PORT_FALLBACK_TRIES):
        if port_is_free(host, candidate):
            return candidate
    die("ports %d-%d on %s are all in use"
        % (DEFAULT_PORT, DEFAULT_PORT + PORT_FALLBACK_TRIES - 1, host),
        "Pick one explicitly:  revv up --port 9000")
    return DEFAULT_PORT      # unreachable


def _free_port() -> int:
    """Ask the kernel for an unused port. Small race, local-only, acceptable."""
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


class Backend:
    """Owns the llama-server child process and can restart it in a new mode."""

    def __init__(self, exe: str, model: str, tier: str, plan: LaunchPlan,
                 passthrough: Sequence[str], log_path: str) -> None:
        self.exe = exe
        self.model = model
        self.tier = tier
        self.plan = plan
        self.passthrough = list(passthrough)
        self.log_path = log_path
        self.proc = None            # type: Optional[subprocess.Popen]
        self.port = 0
        self.mode = MODE_REVV
        self.lock = threading.Lock()
        # Called after every successful start so the supervisor can persist the
        # new backend pid. `revv down` needs that pid from DISK, because if the
        # supervisor was killed the status endpoint died with it.
        self.on_change = None       # type: Optional[Callable[[], None]]

    def argv(self, mode: str, port: int) -> List[str]:
        return build_server_argv(self.exe, self.model, self.plan, port, mode,
                                 self.passthrough)

    def start(self, mode: str, wait_s: float = 600.0) -> None:
        self.port = _free_port()
        self.mode = mode
        argv = self.argv(mode, self.port)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        log = open(self.log_path, "ab", 0)
        log.write(b"\n=== revv: starting backend in %s mode ===\n"
                  % mode.encode())
        try:
            self.proc = subprocess.Popen(argv, stdout=log, stderr=log,
                                         stdin=subprocess.DEVNULL)
        except OSError as exc:
            raise RuntimeError("could not start %s: %s" % (self.exe, exc))
        if not self._await_health(wait_s):
            self.stop()
            raise RuntimeError(
                "llama-server did not become healthy.\n"
                "Last lines of %s:\n%s" % (self.log_path, _tail(self.log_path)))
        if self.on_change is not None:
            self.on_change()

    def _await_health(self, wait_s: float) -> bool:
        url = "http://127.0.0.1:%d/health" % self.port
        deadline = time.time() + wait_s
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                return False    # died during load, usually CUDA OOM
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        return True
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.4)
        return False

    def stop(self) -> None:
        proc = self.proc
        self.proc = None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)

    def switch(self, mode: str) -> None:
        """Restart in the other mode. The weights stay in the page cache, so
        this is a reload from RAM, not from disk."""
        with self.lock:
            if mode == self.mode and self.proc is not None:
                return
            self.stop()
            self.start(mode)

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None


def _tail(path: str, n: int = 15) -> str:
    try:
        with open(path, "rb") as fh:
            lines = fh.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return "(no log)"
    return "\n".join("  " + ln for ln in lines[-n:])


class _Stats:
    """Rolling telemetry for the console log and the response header.

    This is instrumentation, not measurement: `revv bench` and `revv compare`
    are the instruments that follow the certified protocol.
    """

    def __init__(self) -> None:
        self.last_tps = 0.0
        self.n_requests = 0


def _sse_token_count(body: bytes) -> Tuple[int, Optional[float],
                                           Optional[int], int, int]:
    """Walk a server-sent-event body: (content chunks, t/s if the server
    reported it, completion tokens if reported)."""
    chunks = 0
    tps = None      # type: Optional[float]
    n_tok = None    # type: Optional[int]
    draft_n = 0
    draft_acc = 0
    for raw in body.split(b"\n"):
        if not raw.startswith(b"data: "):
            continue
        payload = raw[6:].strip()
        if payload == b"[DONE]":
            continue
        try:
            obj = json.loads(payload.decode("utf-8", "replace"))
        except ValueError:
            continue
        choices = obj.get("choices")
        if isinstance(choices, list) and choices:
            delta = choices[0].get("delta") or {}
            # Reasoning tokens are tokens. In STOCK mode the model may spend
            # its whole budget in the reasoning channel and emit no content at
            # all; ignoring that would report zero work done.
            if delta.get("content") or delta.get("reasoning_content"):
                chunks += 1
        timings = obj.get("timings")
        if isinstance(timings, dict) and timings.get("predicted_per_second"):
            tps = float(timings["predicted_per_second"])
            n_tok = int(timings.get("predicted_n") or 0) or None
        if isinstance(timings, dict) and timings.get("draft_n"):
            draft_n = int(timings.get("draft_n") or 0)
            draft_acc = int(timings.get("draft_n_accepted") or 0)
        usage = obj.get("usage")
        if isinstance(usage, dict) and usage.get("completion_tokens"):
            n_tok = int(usage["completion_tokens"])
    return chunks, tps, n_tok, draft_n, draft_acc


def make_proxy_handler(backend: Backend, stats: _Stats, quiet: bool):
    class Handler(BaseHTTPRequestHandler):
        # HTTP/1.0 with connection-close lets us stream a body of unknown
        # length straight through without re-chunking it. Local hop, so the
        # cost of a new connection per request is irrelevant.
        protocol_version = "HTTP/1.0"
        server_version = "revv/" + __version__

        def log_message(self, fmt: str, *a: object) -> None:
            pass    # we print our own, more useful, one-liner

        def _control(self, path: str) -> bool:
            if not path.startswith("/_revv/"):
                return False
            body = b""
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                body = self.rfile.read(length)
            action = path[len("/_revv/"):].split("?")[0]
            if action == "status":
                self._json(200, self._status_obj())
            elif action in ("toggle", "mode"):
                want = MODE_STOCK if backend.mode == MODE_REVV else MODE_REVV
                if action == "mode" and body:
                    try:
                        want = json.loads(body.decode("utf-8"))["mode"]
                    except (ValueError, KeyError, TypeError):
                        self._json(400, {"error": "expected {\"mode\": \"revv\""
                                                  " or \"stock\"}"})
                        return True
                if want not in (MODE_REVV, MODE_STOCK):
                    self._json(400, {"error": "unknown mode: %s" % want})
                    return True
                t0 = time.time()
                try:
                    backend.switch(want)
                except RuntimeError as exc:
                    self._json(500, {"error": str(exc)})
                    return True
                obj = self._status_obj()
                obj["switch_seconds"] = round(time.time() - t0, 2)
                print("[revv] mode -> %s (%.1fs)"
                      % (backend.mode.upper(), obj["switch_seconds"]))
                self._json(200, obj)
            else:
                self._json(404, {"error": "no such control endpoint"})
            return True

        def _status_obj(self) -> Dict[str, object]:
            plan = backend.plan
            return {"mode": backend.mode,
                    "mode_description": mode_description(backend.mode, plan),
                    "model": os.path.basename(backend.model),
                    "tier": backend.tier,
                    "context": plan.ctx,
                    "kv": plan.kv,
                    "line": (BUILDS[plan.build_name].get("line")
                             if plan.build_name in BUILDS else None),
                    # Which levers this model can actually use, so toggle and
                    # compare can refuse to stage a meaningless A/B.
                    "levers": plan.levers,
                    "tuning_is_noop": plan.is_noop,
                    "speculation": plan.use_spec and backend.mode == MODE_REVV,
                    "backend_port": backend.port,
                    # `revv down` uses this to reap an orphaned llama-server if
                    # the supervisor died without cleaning up after itself.
                    "backend_pid": (backend.proc.pid if backend.proc else 0),
                    "supervisor_pid": os.getpid(),
                    "requests": stats.n_requests,
                    "last_decode_tps": round(stats.last_tps, 2)}

        def _json(self, code: int, obj: Dict[str, object]) -> None:
            payload = json.dumps(obj, indent=2).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(payload)

        def _forward(self, method: str) -> None:
            path = self.path
            if self._control(path):
                return
            if not backend.alive():
                self._json(503, {"error": "the llama-server backend is not "
                                          "running; check the revv console"})
                return
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else None

            conn = http.client.HTTPConnection("127.0.0.1", backend.port,
                                              timeout=3600)
            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in ("host", "connection",
                                            "content-length")}
            if body is not None:
                headers["Content-Length"] = str(len(body))
            t0 = time.time()
            try:
                conn.request(method, path, body=body, headers=headers)
                resp = conn.getresponse()
            except (OSError, http.client.HTTPException) as exc:
                conn.close()
                self._json(502, {"error": "backend request failed: %s" % exc})
                return

            self.send_response(resp.status)
            for key, value in resp.getheaders():
                # send_response() already emitted Server and Date; forwarding
                # the backend's copies would duplicate both headers.
                if key.lower() in ("connection", "transfer-encoding",
                                   "content-length", "server", "date"):
                    continue
                self.send_header(key, value)
            self.send_header("X-Revv-Mode", backend.mode)
            # The rate of the PREVIOUS generation: headers must be written
            # before the body, so this request's own rate does not exist yet.
            self.send_header("X-Revv-Last-Decode-TPS", "%.2f" % stats.last_tps)
            self.send_header("Connection", "close")
            self.end_headers()

            captured = bytearray()
            t_first = None      # type: Optional[float]
            # read1() hands back whatever has already arrived; plain read(n)
            # blocks until it can fill n bytes, which stalls token streaming
            # until 64 KB has piled up.
            read_chunk = getattr(resp, "read1", None) or resp.read
            try:
                while True:
                    chunk = read_chunk(65536)
                    if not chunk:
                        break
                    if t_first is None:
                        t_first = time.time()
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    if len(captured) < 4 * 1024 * 1024:
                        captured += chunk
            except (BrokenPipeError, ConnectionResetError):
                return          # client hung up mid-stream; not our problem
            finally:
                conn.close()
            self._record(method, path, bytes(captured), t0, t_first)

        def _record(self, method: str, path: str, body: bytes,
                    t0: float, t_first: Optional[float]) -> None:
            if "/completion" not in path:
                return
            stats.n_requests += 1
            now = time.time()
            n_tok = None        # type: Optional[int]
            tps = None          # type: Optional[float]
            if body.startswith(b"data:") or b"\ndata: " in body[:4096]:
                chunks, tps, n_tok, _, _ = _sse_token_count(body)
                if tps is None and t_first is not None and chunks > 1:
                    tps = (chunks - 1) / max(now - t_first, 1e-6)
                if n_tok is None:
                    n_tok = chunks
            else:
                try:
                    obj = json.loads(body.decode("utf-8", "replace"))
                except ValueError:
                    return
                timings = obj.get("timings")
                if isinstance(timings, dict):
                    tps = timings.get("predicted_per_second")
                    n_tok = timings.get("predicted_n")
                usage = obj.get("usage") or {}
                n_tok = n_tok or usage.get("completion_tokens")
            if tps:
                stats.last_tps = float(tps)
            if quiet:
                return
            ttft = "-" if t_first is None else "%.2fs" % (t_first - t0)
            print("[revv] %s %s  mode=%s  %s tok  %s  ttft %s  %.1fs"
                  % (method, path, backend.mode.upper(),
                     n_tok if n_tok else "?",
                     "%.1f t/s" % tps if tps else "-",
                     ttft, now - t0))
            sys.stdout.flush()

        def do_GET(self) -> None:
            self._forward("GET")

        def do_POST(self) -> None:
            self._forward("POST")

        def do_DELETE(self) -> None:
            self._forward("DELETE")

        def do_OPTIONS(self) -> None:
            self._forward("OPTIONS")

    return Handler


def _pick_tier(explicit: Optional[str], quiet: bool = False
               ) -> Tuple[str, Optional[int]]:
    """Return (tier, free_mib). free_mib is None when the tier was forced."""
    if explicit is not None:
        if explicit not in TIERS:
            die("unknown tier: %s" % explicit,
                "one of: %s" % ", ".join(sorted(TIERS)))
        return explicit, None
    gpus, err = detect_gpus()
    if err is not None:
        die("cannot detect a GPU: %s" % err,
            "revv v1.0 is NVIDIA-only. If nvidia-smi works but revv cannot see\n"
            "it, force a tier:  revv serve --tier 12gb")
    best = max(gpus, key=lambda g: g.free_mib)
    detected = tier_for(best.free_mib)
    if detected is None:
        die("%s has only %s free (of %s total)"
            % (best.name, mib(best.free_mib), mib(best.total_mib)),
            "revv needs %s free: the weights are 10.2 GiB before any context.\n"
            "Close other GPU processes and try again. On WSL2 the host\n"
            "reserves 1-1.5 GB, which can put a 12GB card under the line."
            % mib(VRAM_MIN_FREE_MIB))
    if best.reserved_mib > 0 and not quiet:
        print("%s %s of %s is reserved by the host and unavailable to CUDA."
              % (dim("note:"), mib(best.reserved_mib), mib(best.total_mib)))
    return str(detected), best.free_mib


def print_plan(plan: LaunchPlan) -> None:
    """Say which levers apply to THIS model, and which do not and why."""
    if plan.is_noop:
        print("  %s this model gains nothing from revv's tuned mode --"
              % yellow("note:"))
        print("         serving with the best-known stock config.")
    else:
        print("  levers   %s" % ", ".join(plan.levers))
    for note in plan.notes:
        for i, line in enumerate(_wrap(note, 66)):
            print("           %s%s" % ("" if i else "- ", line)
                  if i == 0 else "             %s" % line)


def _wrap(text: str, width: int) -> List[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w) if cur else w
    if cur:
        lines.append(cur)
    return lines


def cmd_serve(args: argparse.Namespace, passthrough: Sequence[str]) -> int:
    exe = find_llama_server()
    if exe is None:
        die("llama-server not found",
            "./install.sh   # downloads a prebuilt binary into %s" % BIN_DIR)
        return 1
    model = resolve_model(args.model)
    tier, free_mib = _pick_tier(args.tier, quiet=args.print_command)

    # A model without the draft head cannot speculate, and passing the flags
    # anyway makes llama-server fail to start with an opaque message.
    try:
        info = read_gguf(model)
    except GGUFError as exc:
        die("cannot read %s: %s" % (model, exc))
        return 1
    mode = MODE_STOCK if args.stock else MODE_REVV

    draft_info = None       # type: Optional[GGUFInfo]
    if args.draft:
        if not os.path.isfile(args.draft):
            die("no such draft model: %s" % args.draft,
                "--draft takes a path to a .gguf file you already have.\n"
                "revv never downloads third-party drafters for you.")
            return 1
        try:
            draft_info = read_gguf(args.draft)
        except GGUFError as exc:
            die("cannot read the draft model: %s" % exc)
            return 1
    plan = plan_launch(info, tier, args.ctx, free_mib, draft_info)

    # Resolved before anything binds or prints, so --print-command shows the
    # port the server would actually use.
    port = resolve_port(args.host, args.port)

    if args.print_command:
        argv = build_server_argv(exe, model, plan, port, mode, passthrough)
        print(" ".join(_shell_quote(a) for a in argv))
        return 0

    print(bold("revv %s  --  serve" % __version__))
    print("  model    %s" % os.path.basename(model))
    if args.port is None and port != DEFAULT_PORT:
        print("  %s port %d was busy, using %d instead"
              % (yellow("note:"), DEFAULT_PORT, port))
    print("  tier     %s   context %s   KV %s"
          % (tier.upper(), "{:,}".format(plan.ctx), plan.kv))
    if plan.estimated_peak:
        print("  vram     ~%s estimated peak%s"
              % (mib(plan.estimated_peak),
                 " of %s free" % mib(free_mib) if free_mib else ""))
    print_plan(plan)

    backend = Backend(exe, model, tier, plan, passthrough,
                      os.path.join(REVV_HOME, "logs", "llama-server.log"))
    print("\n  starting llama-server (first load pages ~%s off disk)..."
          % gib(info.file_size))
    sys.stdout.flush()
    try:
        backend.start(mode)
    except RuntimeError as exc:
        die(str(exc),
            "On a 12GB card the usual cause is VRAM: the certified config\n"
            "peaks at %s of ~12,044 MiB. Close the desktop session, or:\n"
            "  revv serve --ctx 8192" % mib(CERT_PEAK_MIB))
        return 1

    stats = _Stats()
    handler = make_proxy_handler(backend, stats, args.quiet)
    try:
        httpd = ThreadingHTTPServer((args.host, port), handler)
    except OSError as exc:
        backend.stop()
        die("cannot bind %s:%d (%s)" % (args.host, port, exc),
            "Something else is on that port. Pick another:\n"
            "  revv serve --port 8081")
        return 1
    httpd.daemon_threads = True

    print("\n  %s  http://%s:%d/v1" % (bold("api"), args.host, port))
    print("  mode     %s -- %s" % (bold(backend.mode.upper()),
                                   mode_description(backend.mode, plan)))
    print("  backend  llama-server on 127.0.0.1:%d" % backend.port)
    print("  log      %s" % backend.log_path)
    print("\n  Point any OpenAI-compatible tool at the api url above; the port")
    print("  stays put even when the backend restarts. Model name: %s"
          % bold(MODEL_ALIAS))
    print("    %s   switch between certified and stock" % bold("revv toggle"))
    print("    %s  measure both, side by side" % bold("revv compare"))
    print("    %s    the certified-protocol benchmark" % bold("revv bench"))
    print("\n  Ctrl-C to stop.\n")
    sys.stdout.flush()

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()

    # `revv up` runs this very function detached, so serve is the only writer of
    # the run file; up and status just read it.
    started_at = time.time()

    def _persist() -> None:
        write_run_file(pid=os.getpid(), port=port, host=args.host,
                       model=os.path.basename(model), tier=tier,
                       backend_pid=(backend.proc.pid if backend.proc else 0),
                       started_at=started_at,
                       log=os.path.join(LOG_DIR, "revv.log"))

    _persist()
    backend.on_change = _persist

    # SIGTERM is how `revv down` asks us to stop. Turning it into
    # KeyboardInterrupt routes it through the same cleanup path as Ctrl-C --
    # without this the default handler would kill us and orphan llama-server
    # holding 11 GB of VRAM.
    def _on_term(signum: int, frame: object) -> None:
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, _on_term)

    try:
        while True:
            time.sleep(0.5)
            if not backend.alive() and not backend.lock.locked():
                print("\n%s the llama-server backend exited. Last log lines:\n%s"
                      % (red("error:"), _tail(backend.log_path)),
                      file=sys.stderr)
                return 1
    except KeyboardInterrupt:
        print("\n  stopping...")
    finally:
        httpd.shutdown()
        backend.stop()
        clear_run_file()
    print("  stopped.")
    return 0


# ---------------------------------------------------------------------------
# toggle
# ---------------------------------------------------------------------------

def _control(url: str, action: str, payload: Optional[Dict[str, object]] = None,
             timeout: float = 900.0) -> Dict[str, object]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else b""
    req = urllib.request.Request(
        url.rstrip("/") + "/_revv/" + action, data=data,
        headers={"Content-Type": "application/json", "User-Agent": "revv/1.0"},
        method="POST" if payload is not None or action != "status" else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def default_url() -> str:
    """Where a client command should look, absent an explicit --url.

    Reads the run file, so `revv bench` still finds the server after the port
    fallback moved it off 8080.
    """
    run = read_run_file()
    if run and run.get("port"):
        return "http://%s:%s" % (run.get("host") or "127.0.0.1", run["port"])
    return "http://127.0.0.1:%d" % DEFAULT_PORT


def _no_server(url: str, exc: object) -> None:
    die("no revv server at %s (%s)" % (url, exc),
        "Start one in another terminal:\n  revv serve\n"
        "This command talks to a running `revv serve`, not to llama-server.")


def cmd_toggle(args: argparse.Namespace) -> int:
    args.url = args.url or default_url()
    try:
        before = _control(args.url, "status")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        _no_server(args.url, exc)
        return 1
    if before.get("tuning_is_noop") and not args.mode:
        print("%s revv mode and stock are the same configuration for this "
              "model." % yellow("note:"))
        print("      No draft head, no thinking switch, and f16 KV is already")
        print("      optimal, so there is nothing for the toggle to change.")
        print("      Pass an explicit mode if you want to switch anyway.")
        return 0
    want = args.mode or (MODE_STOCK if before["mode"] == MODE_REVV else MODE_REVV)
    if want == before["mode"]:
        print("Already in %s mode." % str(want).upper())
        return 0
    print("Switching %s -> %s ..." % (str(before["mode"]).upper(),
                                      str(want).upper()))
    print("  the weights stay in the page cache, so this is usually 10-15 s.")
    sys.stdout.flush()
    try:
        after = _control(args.url, "mode", {"mode": want})
    except (urllib.error.URLError, OSError, ValueError) as exc:
        die("the switch failed: %s" % exc,
            "Check the `revv serve` console for the llama-server error.")
        return 1
    if after.get("error"):
        die(str(after["error"]))
    print("\n  mode  %s -- %s" % (bold(str(after["mode"]).upper()),
                                  after["mode_description"]))
    print("  took  %.1fs" % float(after.get("switch_seconds") or 0.0))
    print("  the api url did not change; your client did not notice.")
    return 0


# ---------------------------------------------------------------------------
# compare
#
# The demo. Same prompt, same weights, same GPU, both modes, one table.
# ---------------------------------------------------------------------------

class GenResult:
    """One timed generation, measured two ways on purpose.

    `server_tps` is llama-server's own decode rate -- the same quantity
    `revv bench` reports, and the one to quote. `client_tps` is what this
    client observed over the stream. The two differ by whatever is lost
    between llama-server and here: the revv proxy, SSE framing, and the host's
    loopback networking. On a native Linux box that gap is ~1%; a large gap is
    a property of the host, not of the model, and `compare` says so rather
    than silently reporting the lower number.
    """

    def __init__(self, server_tps: Optional[float], client_tps: float,
                 ttft: float, n_tok: int, wall: float, draft_n: int = 0,
                 draft_accepted: int = 0) -> None:
        self.server_tps = server_tps
        self.client_tps = client_tps
        self.ttft = ttft
        self.n_tok = n_tok
        self.wall = wall
        self.draft_n = draft_n
        self.draft_accepted = draft_accepted

    @property
    def acceptance(self) -> Optional[float]:
        return self.draft_accepted / self.draft_n if self.draft_n else None

    @property
    def tps(self) -> float:
        """The figure to report: the server's, when it gives us one."""
        return self.server_tps if self.server_tps else self.client_tps

    @property
    def overhead_pct(self) -> Optional[float]:
        if not self.server_tps or not self.client_tps:
            return None
        return (self.client_tps - self.server_tps) / self.server_tps * 100.0


def _timed_generation(base: str, max_tokens: int,
                      timeout: float) -> GenResult:
    """Stream one generation and time it from both ends.

    Streaming, because time-to-first-token is half the point: it separates
    prefill from decode without needing the server's own counters.
    """
    payload = {
        "messages": [{"role": "user", "content": BENCH_PROMPT}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "seed": 1234,
        "stream": True,
        "cache_prompt": False,
        # Ask for the server's own decode rate on every chunk. Without this,
        # timings ride only the final response (server-context.cpp: "populate
        # timings if this is final response or timings_per_token is enabled"),
        # whose shape varies across builds and OAI-compat paths. If we miss it
        # we silently fall back to a client-side wall-clock rate that includes
        # proxy and loopback cost -- which is how compare and bench came to
        # disagree by 13% on one host while agreeing on another.
        "timings_per_token": True,
        # No chat_template_kwargs: whether the model thinks must be decided by
        # the running mode, which is the thing being compared.
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "revv/1.0"})
    t0 = time.time()
    t_first = None      # type: Optional[float]
    body = bytearray()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        # Line-at-a-time: an SSE event ends at a newline, and read(n) would
        # block until n bytes accumulated, destroying the time-to-first-token
        # measurement this function exists to take.
        for line in resp:
            # Time to FIRST TOKEN, whichever channel it comes out of: with
            # thinking on, the first thing the model emits is reasoning.
            if t_first is None and line.startswith(b"data: ") and (
                    b'"content":"' in line
                    or b'"reasoning_content":"' in line):
                t_first = time.time()
            body += line
    t_end = time.time()
    wall = t_end - t0
    chunks, server_tps, n_tok, draft_n, draft_acc = _sse_token_count(
        bytes(body))
    if t_first is None:
        t_first = t0
    if n_tok is None:
        n_tok = chunks
    # Client-observed decode rate: tokens after the first, over the time spent
    # streaming them. Never mixed with the server figure -- reported alongside.
    client_tps = (chunks - 1) / max(t_end - t_first, 1e-6) if chunks > 1 else 0.0
    return GenResult(server_tps, client_tps, t_first - t0, int(n_tok), wall,
                     draft_n, draft_acc)


def cmd_compare(args: argparse.Namespace) -> int:
    base = (args.url or default_url()).rstrip("/")
    try:
        start_status = _control(base, "status")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        _no_server(base, exc)
        return 1
    original = str(start_status["mode"])

    if start_status.get("tuning_is_noop"):
        print(bold("revv %s  --  compare" % __version__))
        print("  model    %s" % start_status["model"])
        print("\n  %s there is nothing to compare for this model."
              % yellow("note:"))
        print("  It has no MTP draft head, its chat template has no thinking")
        print("  switch, and f16 KV already fits -- so revv mode and stock are")
        print("  the same configuration. Running the A/B anyway would produce")
        print("  two numbers that differ only by noise and imply a speedup")
        print("  that is not there.")
        print("\n  revv is still serving this model with the best-known stock")
        print("  config. Use %s to measure it." % bold("revv bench"))
        print("  %s tells you before you serve whether a file has the"
              % bold("revv inspect"))
        print("  draft head that revv's speed actually depends on.")
        return 0

    print(bold("revv %s  --  compare" % __version__))
    print("  model    %s" % start_status["model"])
    print("  prompt   the certified bench prompt, greedy, one request per mode")
    print("  budget   %d tokens max; each mode stops when it is done"
          % args.max_tokens)
    print("  warmup   one exchange per mode is run and discarded, so switching\n"
          "           cost never lands inside a timed window (same rule as\n"
          "           `revv bench`)\n")

    rows = []
    try:
        for mode in (MODE_STOCK, MODE_REVV):
            print("  %-5s switching..." % mode.upper(), end="")
            sys.stdout.flush()
            _control(base, "mode", {"mode": mode})
            # Discarded warmup. `revv bench` has always done this; compare did
            # not, which made the two instruments different protocols wearing
            # the same name. Aligning them removes restart cost as a variable
            # whether or not it is large on any given host.
            print("\r  %-5s warming up...     " % mode.upper(), end="")
            sys.stdout.flush()
            _timed_generation(base, max_tokens=min(args.max_tokens, 200),
                              timeout=args.timeout)
            print("\r  %-5s generating...     " % mode.upper(), end="")
            sys.stdout.flush()
            res = _timed_generation(
                base, max_tokens=args.max_tokens, timeout=args.timeout)
            rows.append((mode, res))
            print("\r  %-5s done.               " % mode.upper())
    except (urllib.error.URLError, OSError, ValueError) as exc:
        die("comparison failed: %s" % exc,
            "Check the `revv serve` console for the llama-server error.")
        return 1
    finally:
        try:
            _control(base, "mode", {"mode": original})
        except (urllib.error.URLError, OSError, ValueError):
            pass

    print("\n  %-7s %11s %8s %9s %9s" % ("mode", "decode t/s", "ttft",
                                         "tokens", "wall"))
    print("  " + "-" * 47)
    for mode, res in rows:
        print("  %-7s %11.1f %7.2fs %9d %8.1fs"
              % (mode.upper(), res.tps, res.ttft, res.n_tok, res.wall))

    stock = rows[0][1]
    fast = rows[1][1]
    print()
    if stock.tps > 0:
        print("  decode rate   %.2fx faster" % (fast.tps / stock.tps))
    if stock.wall > 0:
        print("  time to done  %.2fx faster  (%.1fs -> %.1fs)"
              % (stock.wall / fast.wall, stock.wall, fast.wall))
    for mode, res in rows:
        if res.acceptance is not None:
            print("  acceptance    %s %.3f (%d/%d drafted tokens kept)"
                  % (mode.upper(), res.acceptance, res.draft_accepted,
                     res.draft_n))
    if fast.n_tok and stock.n_tok > fast.n_tok:
        print("  tokens spent  %d -> %d  (STOCK thinks out loud; revv does not)"
              % (stock.n_tok, fast.n_tok))

    # Transport diagnostic. These figures are llama-server's own, so they match
    # `revv bench`. If this client saw materially fewer tokens per second than
    # the server produced, the difference is the proxy plus SSE framing plus
    # this host's loopback, and it is worth naming -- it is the most likely
    # explanation for a compare/bench disagreement on any given box.
    gaps = [r.overhead_pct for _, r in rows if r.overhead_pct is not None]
    if gaps and min(gaps) < -5.0:
        worst = min(gaps)
        print("\n  %s this client observed %.0f%% fewer tokens/second than"
              % (yellow("note:"), abs(worst)))
        print("        llama-server reported. The table above quotes the")
        print("        server's own figure, which is what `revv bench` uses.")
        print("        The gap is transport cost on this host (proxy, SSE,")
        print("        loopback) and is usually near zero on native Linux.")
    if any(r.server_tps is None for _, r in rows):
        print("\n  %s the server did not report decode timings, so the rates"
              % yellow("note:"))
        print("        above are client-side wall-clock and include transport.")

    print("\n  " + dim("STOCK = the same weights on the same GPU with"
                       " llama.cpp's defaults for"))
    print("  " + dim("speculation, KV precision and the thinking switch. It is"
                     " a control for"))
    print("  " + dim("revv's configuration, not a measurement of any other"
                     " product."))
    print("  " + dim("Restored to %s mode." % original.upper()))
    return 0


# ---------------------------------------------------------------------------
# Daemon lifecycle: up / down / status
#
# revv is meant to be plumbing you forget about. `revv up` detaches the same
# stack `revv serve` runs in the foreground. No systemd or launchd units in
# v1.0 -- setsid is enough, and autostart can come later.
# ---------------------------------------------------------------------------

RUN_DIR = os.path.join(REVV_HOME, "run")
RUN_FILE = os.path.join(RUN_DIR, "revv.json")
LOG_DIR = os.path.join(REVV_HOME, "logs")


def write_run_file(**fields: object) -> None:
    os.makedirs(RUN_DIR, exist_ok=True)
    tmp = RUN_FILE + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(fields, fh, indent=2)
    os.replace(tmp, RUN_FILE)


def read_run_file() -> Optional[Dict[str, object]]:
    try:
        with open(RUN_FILE, "r") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def clear_run_file() -> None:
    try:
        os.remove(RUN_FILE)
    except OSError:
        pass


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True     # exists, owned by someone else
    return True


def cmd_up(args: argparse.Namespace) -> int:
    existing = read_run_file()
    if existing and pid_alive(int(existing.get("pid") or 0)):
        print("revv is already up (pid %s) on http://%s:%s/v1"
              % (existing.get("pid"), existing.get("host"),
                 existing.get("port")))
        print("Use %s to inspect it, %s to stop it."
              % (bold("revv status"), bold("revv down")))
        return 0
    clear_run_file()

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, "revv.log")
    argv = [sys.executable, os.path.abspath(__file__), "serve", "--quiet"]
    if args.model:
        argv.append(args.model)
    # up resolves the port itself and hands the child an EXPLICIT one, so it
    # knows where to poll for readiness. The child then binds exact-or-fail,
    # which is what we want: we already checked it was free.
    port = resolve_port(args.host, args.port)
    if args.port is None and port != DEFAULT_PORT:
        print("  %s port %d was busy, using %d instead"
              % (yellow("note:"), DEFAULT_PORT, port))
    for flag, value in (("--port", port), ("--host", args.host),
                        ("--ctx", args.ctx), ("--tier", args.tier),
                        ("--draft", args.draft)):
        if value is not None:
            argv += [flag, str(value)]
    if args.stock:
        argv.append("--stock")

    print("Starting revv in the background...")
    print("  log  %s" % log_path)
    sys.stdout.flush()
    log = open(log_path, "ab", 0)
    log.write(b"\n=== revv up ===\n")
    # start_new_session detaches from this terminal's process group, so the
    # stack survives the terminal closing without needing nohup.
    proc = subprocess.Popen(argv, stdout=log, stderr=log,
                            stdin=subprocess.DEVNULL, start_new_session=True)

    base = "http://%s:%d" % (args.host, port)
    deadline = time.time() + args.timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            print("%s revv failed to start. Last log lines:\n%s"
                  % (red("error:"), _tail(log_path, 20)), file=sys.stderr)
            return 1
        try:
            st = _control(base, "status", timeout=3)
        except (urllib.error.URLError, OSError, ValueError):
            time.sleep(0.5)
            continue
        print("\n  %s   http://%s:%d/v1" % (bold("api"), args.host, port))
        print("  mode  %s -- %s" % (bold(str(st["mode"]).upper()),
                                    st["mode_description"]))
        print("  model %s" % st["model"])
        print("  pid   %d" % proc.pid)
        print("\n  %s to see it, %s to stop it, %s for the demo."
              % (bold("revv status"), bold("revv down"), bold("revv compare")))
        return 0
    proc.terminate()
    print("%s revv did not come up within %ds. Last log lines:\n%s"
          % (red("error:"), int(args.timeout), _tail(log_path, 20)),
          file=sys.stderr)
    return 1


def cmd_down(args: argparse.Namespace) -> int:
    run = read_run_file() or {}
    url = getattr(args, "url", None)
    if not run and not url:
        print("revv is not running (no %s)." % RUN_FILE)
        return 0
    # An explicit --url is authoritative: it is how you stop an instance that
    # is not the one in the run file, e.g. started on another port.
    base = url.rstrip("/") if url else \
        "http://%s:%s" % (run.get("host"), run.get("port"))

    # Ask the supervisor for both pids first. If it is already gone we still
    # have to make sure no llama-server is left orphaned holding VRAM.
    pid = int(run.get("pid") or 0)
    backend_pid = 0
    try:
        st = _control(base, "status", timeout=3)
        backend_pid = int(st.get("backend_pid") or 0)
        pid = int(st.get("supervisor_pid") or 0) or pid
    except (urllib.error.URLError, OSError, ValueError):
        # Unreachable is exactly the case where an orphan is most likely, so
        # fall back to what the supervisor wrote to disk.
        backend_pid = int(run.get("backend_pid") or 0)
        if url and not run:
            die("nothing is listening at %s" % base,
                "Check the port, or run `revv down` with no --url to stop\n"
                "the instance recorded in %s" % RUN_FILE)
            return 1

    if not pid_alive(pid):
        print("revv supervisor (pid %d) is already gone." % pid)
    else:
        print("Stopping revv (pid %d)..." % pid)
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as exc:
            die("could not signal pid %d: %s" % (pid, exc),
                "Stop it by hand:  kill %d" % pid)
        deadline = time.time() + 30.0
        while time.time() < deadline and pid_alive(pid):
            time.sleep(0.3)
        if pid_alive(pid):
            print("  it ignored SIGTERM; sending SIGKILL.")
            try:
                os.kill(pid, signal.SIGKILL)
            except OSError:
                pass
            time.sleep(1.0)

    # Never leave a llama-server holding 11 GB of VRAM.
    if backend_pid and pid_alive(backend_pid):
        print("  reaping orphaned llama-server (pid %d)..." % backend_pid)
        try:
            os.kill(backend_pid, signal.SIGTERM)
            deadline = time.time() + 20.0
            while time.time() < deadline and pid_alive(backend_pid):
                time.sleep(0.3)
            if pid_alive(backend_pid):
                os.kill(backend_pid, signal.SIGKILL)
        except OSError:
            pass
    clear_run_file()
    print("Stopped.")
    return 0


def _uptime(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return "%ds" % seconds
    if seconds < 3600:
        return "%dm %ds" % (seconds // 60, seconds % 60)
    return "%dh %dm" % (seconds // 3600, (seconds % 3600) // 60)


def cmd_status(args: argparse.Namespace) -> int:
    run = read_run_file()
    base = args.url.rstrip("/") if args.url else (
        "http://%s:%s" % (run.get("host"), run.get("port")) if run else None)
    if base is None:
        print("revv is not running.")
        print("Start it with %s (background) or %s (foreground)."
              % (bold("revv up"), bold("revv serve")))
        return 1
    try:
        st = _control(base, "status", timeout=5)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print("%s no revv server at %s (%s)" % (red("down:"), base, exc))
        if run:
            print("  a stale %s says pid %s; clear it with %s"
                  % (RUN_FILE, run.get("pid"), bold("revv down")))
        return 1

    print(bold("revv %s  --  status" % __version__))
    print("  api      %s/v1" % base)
    print("  mode     %s -- %s" % (bold(str(st["mode"]).upper()),
                                   st["mode_description"]))
    print("  model    %s   (send \"%s\" as the model name)"
          % (st["model"], MODEL_ALIAS))
    if st.get("line"):
        print("  line     %s" % str(st["line"]).upper())
    print("  tier     %s   context %s   KV %s"
          % (str(st["tier"]).upper(),
             "{:,}".format(int(st.get("context") or 0)), st.get("kv", "?")))
    if st.get("tuning_is_noop"):
        print("  tuning   %s"
              % yellow("no lever applies to this model; serving stock config"))
    else:
        print("  levers   %s" % ", ".join(st.get("levers") or ["none"]))
    if run and run.get("started_at"):
        print("  uptime   %s   pid %s"
              % (_uptime(time.time() - float(run["started_at"])),
                 run.get("pid")))
    print("  requests %s" % st["requests"])
    last = float(st.get("last_decode_tps") or 0.0)
    print("  last     %s" % ("%.1f t/s on the most recent generation" % last
                             if last else "no generations yet"))

    gpus, err = detect_gpus()
    if err is None and gpus:
        g = max(gpus, key=lambda x: x.total_mib)
        head = g.total_mib - g.used_mib
        print("  vram     %s used of %s  (%s free)"
              % (mib(g.used_mib), mib(g.total_mib), mib(head)))
        if g.used_mib > CERT_PEAK_MIB and head < 200:
            print("           %s that is close to the ceiling."
                  % yellow("note:"))
    return 0


def _shell_quote(s: str) -> str:
    return s if re.match(r"^[\w@%+=:,./-]+$", s) else "'" + s.replace(
        "'", "'\"'\"'") + "'"


# ---------------------------------------------------------------------------
# bench
#
# The community-feedback instrument. Reproduces the re-certification protocol:
# 4 sequential requests, a code prompt, 400 new tokens, greedy, thinking off.
# ---------------------------------------------------------------------------

BENCH_PROMPT = (
    "Write a complete Python implementation of an LRU cache with a fixed "
    "capacity. Use a dict plus a doubly linked list so that get and put are "
    "both O(1). Include the node class, full link/unlink handling, and "
    "docstrings. Do not use collections.OrderedDict or functools.lru_cache."
)
BENCH_N_PREDICT = 400
BENCH_REQUESTS = 4


def _post_json(url: str, payload: Dict[str, object],
               timeout: float) -> Dict[str, object]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json", "User-Agent": "revv/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class BenchResult:
    def __init__(self, ts: float, n_tok: int, wall: float, source: str,
                 reasoning_chars: int, draft_n: int = 0,
                 draft_accepted: int = 0) -> None:
        self.ts = ts
        self.n_tok = n_tok
        self.wall = wall
        self.source = source
        self.reasoning_chars = reasoning_chars
        self.draft_n = draft_n
        self.draft_accepted = draft_accepted

    @property
    def acceptance(self) -> Optional[float]:
        """Fraction of drafted tokens the target model kept.

        Only meaningful with speculation on. It is a property of the
        target/drafter PAIR, not of either model alone -- which is why an
        external drafter has to be measured rather than assumed.
        """
        return self.draft_accepted / self.draft_n if self.draft_n else None


def _bench_once(base: str, timeout: float) -> BenchResult:
    payload = {
        "messages": [{"role": "user", "content": BENCH_PROMPT}],
        "max_tokens": BENCH_N_PREDICT,
        "temperature": 0.0,   # greedy, exactly as certified
        "top_k": 1,
        "seed": 1234,
        "stream": False,
        "cache_prompt": False,  # a warm prefix would replay the previous answer
        # Deliberately NO chat_template_kwargs. A per-request kwarg is merged
        # OVER the server's default, so sending it here would mask a broken
        # server-side flag and measure a configuration nobody ships.
    }
    t0 = time.time()
    body = _post_json(base + "/v1/chat/completions", payload, timeout)
    wall = time.time() - t0

    usage = body.get("usage") or {}
    n_tok = int(usage.get("completion_tokens") or 0)
    reasoning = ""
    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(msg, dict):
            reasoning = msg.get("reasoning_content") or ""

    timings = body.get("timings")
    d_n, d_acc = 0, 0
    if isinstance(timings, dict):
        d_n = int(timings.get("draft_n") or 0)
        d_acc = int(timings.get("draft_n_accepted") or 0)
    if isinstance(timings, dict) and timings.get("predicted_per_second"):
        # llama-server's own decode rate: excludes prefill, and is the exact
        # quantity the re-certification table reports.
        return BenchResult(float(timings["predicted_per_second"]),
                           int(timings.get("predicted_n") or n_tok), wall,
                           "server", len(reasoning), d_n, d_acc)
    if n_tok:
        return BenchResult(n_tok / wall, n_tok, wall, "wall-clock",
                           len(reasoning), d_n, d_acc)
    raise RuntimeError("server returned no token counts")


# The thinking probe.
#
# A per-request chat_template_kwargs is merged OVER the server's default, so a
# probe that sends the kwarg cannot tell a working server-side flag from a
# broken one -- it tests the request path instead. Arm A therefore sends
# nothing and measures the shipped configuration; arm C turns thinking ON to
# prove the detector actually fires. Without arm C, a zero in arm A is
# unfalsifiable: a model that never emits reasoning_content would look the same
# as one that is correctly suppressed.
PROBE_PROMPT = "What is 17 * 23? Answer briefly."
PROBE_MAX_TOKENS = 220


def _probe_reasoning(base: str, kwargs: Optional[Dict[str, bool]],
                     timeout: float) -> int:
    """Return the number of reasoning characters the server emitted."""
    payload = {
        "messages": [{"role": "user", "content": PROBE_PROMPT}],
        "max_tokens": PROBE_MAX_TOKENS,
        "temperature": 0.0,
        "top_k": 1,
        "seed": 1234,
        "stream": False,
        "cache_prompt": False,
    }   # type: Dict[str, object]
    if kwargs is not None:
        payload["chat_template_kwargs"] = kwargs
    body = _post_json(base + "/v1/chat/completions", payload, timeout)
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return 0
    msg = choices[0].get("message")
    if not isinstance(msg, dict):
        return 0
    return len(msg.get("reasoning_content") or "")


def thinking_check(base: str, timeout: float) -> Tuple[bool, str]:
    """Three-arm check. Returns (ok, printable report)."""
    arms = [
        ("A  server-side flag only  (no kwarg)", None),
        ("B  per-request kwarg      (false)   ", {"enable_thinking": False}),
        ("C  positive control       (true)    ", {"enable_thinking": True}),
    ]
    results = []
    for label, kwargs in arms:
        try:
            n = _probe_reasoning(base, kwargs, timeout)
        except (urllib.error.URLError, OSError, ValueError, RuntimeError) as exc:
            return False, "    probe failed: %s" % exc
        results.append((label, n))

    lines = ["    %-38s %s" % (label, "%d reasoning chars" % n)
             for label, n in results]
    a, c = results[0][1], results[2][1]
    if a == 0 and c > 0:
        lines.append("    %s thinking is off in the shipped configuration,"
                     % green("PASS:"))
        lines.append("          and arm C proves the detector fires when it is on.")
        return True, "\n".join(lines)
    if a > 0:
        lines.append("    %s the server-side flag is NOT taking effect."
                     % red("FAIL:"))
        lines.append("          The model is thinking on every request: ~2.8x")
        lines.append("          slower per task, and the BENCHMARKS.md quality")
        lines.append("          numbers do not apply. Check that --jinja is set")
        lines.append("          and that your GGUF's chat template honours")
        lines.append("          enable_thinking.")
        return False, "\n".join(lines)
    lines.append("    %s arm A is clean but the positive control never fired,"
                 % yellow("INCONCLUSIVE:"))
    lines.append("          so this probe cannot distinguish 'thinking correctly")
    lines.append("          suppressed' from 'this template never emits a")
    lines.append("          reasoning block at all'. Treat the speed numbers as")
    lines.append("          valid and the thinking state as unverified.")
    return True, "\n".join(lines)


def cmd_bench(args: argparse.Namespace) -> int:
    base = (args.url or default_url()).rstrip("/")
    print(bold("revv %s  --  bench" % __version__))
    print("  target   %s" % base)
    print("  protocol %d requests, %d new tokens, greedy, thinking off"
          % (BENCH_REQUESTS, BENCH_N_PREDICT))

    try:
        with urllib.request.urlopen(base + "/health", timeout=10) as resp:
            resp.read()
    except (urllib.error.URLError, OSError) as exc:
        die("no server at %s (%s)" % (base, exc),
            "Start one in another terminal:\n  revv serve\n"
            "Or point bench elsewhere:\n  revv bench --url http://host:8080")
        return 1

    print("\n  warmup...", end="")
    sys.stdout.flush()
    try:
        _bench_once(base, args.timeout)
    except (urllib.error.URLError, OSError, RuntimeError, ValueError) as exc:
        print()
        die("the warmup request failed: %s" % exc,
            "Check the llama-server log. A 12GB card OOMs here if a desktop\n"
            "session is holding VRAM: peak during generation is %s."
            % mib(CERT_PEAK_MIB))
        return 1
    print(" done (excluded from the result)\n")

    runs: List[BenchResult] = []
    for i in range(BENCH_REQUESTS):
        try:
            r = _bench_once(base, args.timeout)
        except (urllib.error.URLError, OSError, RuntimeError, ValueError) as exc:
            die("request %d failed: %s" % (i + 1, exc))
            return 1
        runs.append(r)
        print("  request %d   %6.2f t/s   %4d tokens   %5.2f s wall"
              % (i + 1, r.ts, r.n_tok, r.wall))

    rates = sorted(r.ts for r in runs)
    mean = sum(rates) / len(rates)
    median = rates[len(rates) // 2] if len(rates) % 2 else \
        (rates[len(rates) // 2 - 1] + rates[len(rates) // 2]) / 2.0
    spread = (rates[-1] - rates[0]) / mean * 100.0
    leaked = any(r.reasoning_chars > 0 for r in runs)

    print("\n  " + bold("result"))
    print("    decode      %.2f t/s mean, %.2f median" % (mean, median))
    print("    spread      %.1f%% across %d requests (noise floor is ~1%%)"
          % (spread, len(rates)))
    print("    measured by %s" % ("llama-server timings"
                                  if runs[0].source == "server"
                                  else "wall clock (server timings absent)"))
    total_draft = sum(r.draft_n for r in runs)
    if total_draft:
        acc = sum(r.draft_accepted for r in runs) / float(total_draft)
        print("    acceptance  %.3f  (%d of %d drafted tokens kept)"
              % (acc, sum(r.draft_accepted for r in runs), total_draft))
        # The certified built-in head sits at 0.781 on a novel prompt. An
        # external drafter is a different pair and has no reference value.
        if acc < 0.35:
            print("    %s acceptance this low usually means the drafter is a"
                  % yellow("note:"))
            print("          poor match for this target; speculation may be a")
            print("          net loss. Compare against --stock to be sure.")
    elif any(r.source == "server" for r in runs):
        print("    acceptance  n/a (no speculation active)")

    if leaked:
        print("\n  " + red("thinking is LEAKING") + " -- the measured requests"
              " returned reasoning blocks.")

    # Compare against the figure that matches the build actually installed,
    # not the headline. Flagging a healthy stock build as "slow" would be noise.
    manifest = read_build_manifest()
    patched = bool(manifest and "mmvq_iquant_decode.patch"
                   in (manifest.get("patches") or []))
    # Compare against the patched figure only when we know the build is
    # patched; the +2.5% would otherwise be charged to the user's hardware.
    target = BENCH_REF_PATCHED if patched else BENCH_REF_PATCHED / 1.025
    target_label = ("kernel-patched build" if patched else
                    "stock build" if manifest else
                    "stock build (assumed: unknown provenance)")

    print("\n  " + bold("reference (RTX 3060 12GB, sm_86, c=16384, this "
                        "same protocol)"))
    print("    %-44s %6.1f t/s" % ("flagship, MTP n=2, q8_0 KV, patched",
                                   BENCH_REF_PATCHED))
    print("    %-44s %6.1f t/s" % ("speculation off (what MTP buys you)",
                                   BENCH_REF_NOSPEC))
    print("    %-44s %6.1f t/s" % ("v1.1 candidate (ASCII prune, not default)",
                                   V11_TS))
    print("    comparing against the %s (%.1f t/s)" % (target_label, target))
    print(dim("    Certification used a different prompt and reads 34.4-36.7"))
    print(dim("    t/s for the same build; see BENCHMARKS.md. Do not mix them."))

    print("\n  " + bold("reading"))
    ratio = mean / target
    if ratio >= 0.95:
        print("    %s within 5%% of the reference (%.1f t/s)."
              % (green("on target:"), target))
    elif mean < BENCH_REF_NOSPEC * 1.10:
        print("    %s %.1f t/s sits in the no-speculation regime."
              % (red("speculation is not running:"), mean))
        print("    Almost always a model without the MTP draft head. Check:")
        print("      revv inspect <your.gguf>")
    elif ratio >= 0.85:
        print("    %s %.0f%% of reference. Usual causes: a hotter or"
              % (yellow("slightly low:"), ratio * 100))
        print("    power-limited card, a slower CPU on the host-side draft")
        print("    path, or another process sharing the GPU.")
    else:
        print("    %s %.0f%% of reference. Check nvidia-smi for other"
              % (red("well below:"), ratio * 100))
        print("    processes, and confirm -ngl put every layer on the GPU.")
    print("\n  " + bold("thinking check") + dim("  (3 arms, ~%d tokens each)")
          % PROBE_MAX_TOKENS)
    ok, report = thinking_check(base, args.timeout)
    print(report)

    print("\n  Numbers off? That is the useful case -- open an issue with this")
    print("  output, `revv doctor`, and your GPU model. BENCHMARKS.md documents")
    print("  exactly how the reference figures were produced.")
    return 0


# ---------------------------------------------------------------------------
# Self-update / uninstall
# ---------------------------------------------------------------------------

def _run_git(git_args: List[str], cwd: str) -> "subprocess.CompletedProcess":
    return subprocess.run(["git"] + git_args, cwd=cwd, capture_output=True,
                          text=True)


def cmd_update(args: argparse.Namespace) -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isdir(os.path.join(here, ".git")):
        die("revv was not installed from git (%s has no .git), so there is "
            "nothing to pull" % here,
            "Install from git instead:\n"
            "  git clone https://github.com/mericanii-technologies/revv")
        return 1

    if args.check:
        fetch = _run_git(["fetch"], here)
        if fetch.returncode != 0:
            die("git fetch failed:\n%s"
                % (fetch.stderr.strip() or fetch.stdout.strip()),
                "Check your network connection and try again.")
            return 1
        behind = _run_git(["rev-list", "--count", "HEAD..@{u}"], here)
        if behind.returncode != 0:
            die("could not compare against the upstream branch:\n%s"
                % (behind.stderr.strip() or behind.stdout.strip()),
                "This checkout may not be tracking a remote branch. Set one "
                "with:\n  git -C %s branch --set-upstream-to=origin/<branch>"
                % here)
            return 1
        count = int(behind.stdout.strip() or "0")
        sha = _git_sha() or "unknown SHA"
        if count == 0:
            print("already up to date (%s)." % sha)
        else:
            print("%d commit%s behind (%s). Run `revv update` to pull."
                  % (count, "" if count == 1 else "s", sha))
        return 0

    # Refuse to pull over local modifications -- never discard anyone's work.
    dirty = _run_git(["status", "--porcelain"], here)
    if dirty.returncode != 0:
        die("git status failed:\n%s"
            % (dirty.stderr.strip() or dirty.stdout.strip()))
        return 1
    if dirty.stdout.strip():
        die("local changes in %s would be overwritten by an update:\n%s"
            % (here, dirty.stdout.rstrip()),
            "Commit or stash your changes first:\n  git -C %s stash" % here)
        return 1

    old_sha = _git_sha()

    # --ff-only is deliberate: never create a merge commit in someone's
    # checkout on their behalf.
    pull = _run_git(["pull", "--ff-only"], here)
    if pull.returncode != 0:
        die("git pull --ff-only failed:\n%s"
            % (pull.stderr.strip() or pull.stdout.strip()),
            "This usually means the checkout has diverged from upstream or "
            "there is no network. Inspect it with:\n"
            "  git -C %s log --oneline --left-right --graph HEAD...@{u}"
            % here)
        return 1

    new_sha = _git_sha()
    if new_sha == old_sha:
        print("already up to date (%s)." % (new_sha or "unknown SHA"))
        return 0

    count_out = _run_git(["rev-list", "--count", "%s..%s" % (old_sha, new_sha)],
                         here)
    count = count_out.stdout.strip() if count_out.returncode == 0 else "?"
    print("updated %s -> %s (%s commit%s)"
          % (old_sha or "unknown", new_sha or "unknown", count,
             "" if count == "1" else "s"))

    print("\n" + bold("what changed"))
    changelog_path = os.path.join(here, "CHANGELOG.md")
    try:
        with open(changelog_path, "r") as fh:
            lines = fh.read().splitlines()
    except OSError:
        lines = None
    if lines is None:
        print("  (no CHANGELOG.md found)")
    else:
        start = next((i for i, l in enumerate(lines) if l.startswith("## [")),
                     None)
        if start is None:
            print("  (CHANGELOG.md has no ## [version] heading)")
        else:
            end = next((i for i in range(start + 1, len(lines))
                        if lines[i].startswith("## [")), len(lines))
            section = lines[start:end]
            while section and not section[-1].strip():
                section.pop()
            for line in section[:40]:
                print("  %s" % line)
            if len(section) > 40:
                print("  ...")

    print("\n" + bold("after this update"))
    manifest = read_build_manifest()
    if manifest is not None:
        method = str(manifest.get("install_method") or "source")
        base = manifest.get("base_commit", "unknown")
        print("  llama-server install: %s (base %s)" % (method, base))
        if method in ("prebuilt", "upstream"):
            print("  `git pull` does not update the installed llama-server "
                  "binary -- if the pinned build changed, re-run: "
                  "./install.sh")
    print("  %s" % version_string())
    return 0


def _dir_size(path: str) -> Tuple[int, int]:
    """Total bytes and file count under path."""
    total = 0
    count = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            count += 1
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total, count


def _human_size(n: int) -> str:
    for unit, factor in (("GiB", 1024 ** 3), ("MiB", 1024 ** 2),
                         ("KiB", 1024)):
        if n >= factor:
            return "%.2f %s" % (n / factor, unit)
    return "%d B" % n


def _confirm(prompt: str) -> bool:
    try:
        answer = input(prompt)
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")


def cmd_uninstall(args: argparse.Namespace) -> int:
    # Refuse anywhere dangerous before touching a single file. REVV_HOME is
    # env-controlled, so a bad value here must never turn into rm -rf /.
    home = os.path.realpath(REVV_HOME)
    dangerous = {"/", os.path.realpath(os.path.expanduser("~")), ""}
    if not REVV_HOME.strip() or home in dangerous:
        die("REVV_HOME resolves to %r, which looks like your home directory "
            "or filesystem root, not revv's state directory" % home,
            "Point REVV_HOME at revv's actual state dir (default ~/.revv) "
            "before running uninstall.")
        return 1

    # Stop anything running first -- never leave a llama-server holding VRAM
    # for a stack whose state is about to disappear.
    run = read_run_file()
    if run and pid_alive(int(run.get("pid") or 0)):
        print("Stopping the running revv stack first...")
        cmd_down(argparse.Namespace(url=None))
        print()

    if not os.path.isdir(REVV_HOME):
        print("Nothing to remove: %s does not exist." % REVV_HOME)
        return 0

    models_dir = os.path.realpath(MODELS_DIR)
    models_inside_home = os.path.isdir(MODELS_DIR) and (
        models_dir == home or models_dir.startswith(home + os.sep))

    # Models are a separate decision, so they are never in state_entries even
    # when (as is normal) they live inside REVV_HOME.
    state_entries = []
    for name in sorted(os.listdir(REVV_HOME)):
        full = os.path.join(REVV_HOME, name)
        if models_inside_home and os.path.realpath(full) == models_dir:
            continue
        state_entries.append(full)

    state_size = 0
    for entry in state_entries:
        if os.path.isdir(entry):
            s, _n = _dir_size(entry)
            state_size += s
        elif os.path.isfile(entry):
            state_size += os.path.getsize(entry)

    models_size = 0
    models_count = 0
    if os.path.isdir(MODELS_DIR):
        models_size, models_count = _dir_size(MODELS_DIR)

    print(bold("this will remove"))
    if state_entries:
        print("  %s  (%s)" % (REVV_HOME, _human_size(state_size)))
        for entry in state_entries:
            print("    %s" % os.path.relpath(entry, REVV_HOME))
    else:
        print("  %s has nothing to remove aside from models." % REVV_HOME)
    if os.path.isdir(MODELS_DIR):
        print("\n" + bold("separately, downloaded models"))
        print("  %s  (%s, %d file%s)"
              % (MODELS_DIR, _human_size(models_size), models_count,
                 "" if models_count == 1 else "s"))
    print()

    is_tty = sys.stdin.isatty()

    if args.yes:
        remove_state = True
    elif is_tty:
        remove_state = _confirm(
            "Remove revv state at %s (%s)? [y/N] "
            % (REVV_HOME, _human_size(state_size)))
    else:
        die("uninstall needs confirmation and stdin is not a terminal",
            "Re-run non-interactively with the flags that answer for you:\n"
            "  revv uninstall --yes             # remove state, keep models\n"
            "  revv uninstall --yes --models    # remove state and models")
        return 1

    if not os.path.isdir(MODELS_DIR) or args.keep_models:
        remove_models = False
        if args.keep_models and os.path.isdir(MODELS_DIR):
            print("Keeping models at %s (--keep-models)." % MODELS_DIR)
    elif args.yes and args.models:
        remove_models = True
    elif args.yes:
        remove_models = False
        print("Keeping models (pass --models with --yes to remove them too).")
    elif is_tty:
        remove_models = _confirm(
            "Also remove downloaded models at %s (%s, %d file%s)? [y/N] "
            % (MODELS_DIR, _human_size(models_size), models_count,
               "" if models_count == 1 else "s"))
    else:
        remove_models = False   # unreachable: the state branch above already
                                 # died when stdin is not a tty and not --yes

    if not remove_state and not remove_models:
        print("Nothing removed.")
        return 0

    errors = []

    if remove_state:
        for entry in state_entries:
            try:
                if os.path.isdir(entry) and not os.path.islink(entry):
                    shutil.rmtree(entry, ignore_errors=False)
                else:
                    os.remove(entry)
            except OSError as exc:
                errors.append("%s: %s" % (entry, exc))
        if errors:
            status(FAIL, "some revv state could not be removed",
                   "\n".join(errors))
        else:
            status(OK, "removed revv state", REVV_HOME)

    if remove_models:
        try:
            shutil.rmtree(MODELS_DIR, ignore_errors=False)
            status(OK, "removed downloaded models", MODELS_DIR)
        except OSError as exc:
            errors.append("%s: %s" % (MODELS_DIR, exc))
            status(FAIL, "models could not be fully removed", str(exc))
    elif os.path.isdir(MODELS_DIR):
        status(OK, "kept downloaded models", MODELS_DIR)

    print()
    remaining = sorted(os.listdir(REVV_HOME)) if os.path.isdir(REVV_HOME) \
        else []
    if remaining:
        print("Remaining in %s:" % REVV_HOME)
        for name in remaining:
            print("  %s" % name)
    else:
        print("%s is now empty." % REVV_HOME)
    print("The git checkout itself is untouched -- removing that is your "
          "call:\n  rm -rf %s" % os.path.dirname(os.path.abspath(__file__)))

    return 1 if errors else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="revv",
        description="revv, by Mericanii -- Qwen3.8-27B on consumer NVIDIA GPUs.",
        epilog="Start with: revv doctor")
    p.add_argument("--version", action="version",
                   version=version_string())
    sub = p.add_subparsers(dest="command")

    sub.add_parser("doctor", help="check this machine and report what is possible")

    g = sub.add_parser("get", help="download the certified model")
    g.add_argument("tier", nargs="?", help="12gb / 16gb / 24gb (default: detect)")
    g.add_argument("--build", help="download a specific build: %s"
                   % ", ".join(sorted(BUILDS)))
    g.add_argument("--force", action="store_true",
                   help="re-download even if the file is already present")

    i = sub.add_parser("inspect", help="parse a GGUF and report its capabilities")
    i.add_argument("file", help="path to a .gguf file")

    a = sub.add_parser("adopt",
                       help="find and register GGUFs already on this machine")
    a.add_argument("--source", choices=["ollama", "lmstudio"],
                   help="scan only one store (default: both)")
    a.add_argument("--all", action="store_true",
                   help="adopt non-Qwen models too; revv's numbers will not apply")
    a.add_argument("--dry-run", action="store_true",
                   help="show what would be adopted without writing the registry")

    def add_serve_flags(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("model", nargs="?",
                        help="path, filename, build name, or adopted name")
        sp.add_argument("--port", type=int, default=None,
                        help="exact port, or omit to use %d and walk forward "
                             "if it is busy" % DEFAULT_PORT)
        sp.add_argument("--host", default="127.0.0.1")
        sp.add_argument("--ctx", type=int, help="override the context size")
        sp.add_argument("--tier", help="force a tier instead of detecting one")
        sp.add_argument("--stock", action="store_true",
                        help="start in STOCK mode (llama.cpp defaults)")
        sp.add_argument("--draft", metavar="FILE.GGUF",
                        help="external draft model for speculative decoding, "
                             "for targets with no built-in head "
                             "(experimental, uncertified)")

    s = sub.add_parser(
        "serve", help="run the stack in the foreground (verbose; for debugging)",
        epilog="Unknown flags are passed through to llama-server unchanged.")
    add_serve_flags(s)
    s.add_argument("--quiet", action="store_true",
                   help="suppress the per-request log line")
    s.add_argument("--print-command", action="store_true",
                   help="print the llama-server command and exit")

    u = sub.add_parser("up", help="start the stack in the background")
    add_serve_flags(u)
    u.add_argument("--timeout", type=float, default=600.0,
                   help="seconds to wait for the model to load")

    d = sub.add_parser("down",
                       help="stop the background stack and its llama-server")
    d.add_argument("--url", help="stop a specific instance instead of the "
                                 "one recorded in the run file")

    st = sub.add_parser("status", help="show mode, model, port, uptime, VRAM")
    st.add_argument("--url", help="query a specific instance")

    t = sub.add_parser("toggle",
                       help="switch between revv and STOCK without moving the port")
    t.add_argument("mode", nargs="?", choices=[MODE_REVV, MODE_STOCK],
                   help="switch to a specific mode (default: the other one)")
    t.add_argument("--url", default=None, help="default: the running instance, else port %d" % DEFAULT_PORT)

    c = sub.add_parser("compare",
                       help="run the same prompt through both modes, side by side")
    c.add_argument("--url", default=None, help="default: the running instance, else port %d" % DEFAULT_PORT)
    # STOCK thinks out loud and needs room to actually finish; capping it
    # turns the headline ratio into a lower bound instead of a measurement.
    c.add_argument("--max-tokens", type=int, default=2048)
    c.add_argument("--timeout", type=float, default=900.0)

    b = sub.add_parser("bench", help="time a running server against the reference")
    b.add_argument("--url", default=None, help="default: the running instance, else port %d" % DEFAULT_PORT)
    b.add_argument("--timeout", type=float, default=300.0)

    up_ = sub.add_parser(
        "update", help="pull the latest revv from git and report what changed")
    up_.add_argument("--check", action="store_true",
                     help="only report whether an update is available; "
                          "do not pull")

    un = sub.add_parser(
        "uninstall",
        help="remove revv's own state (binaries, cache, logs); models are "
             "a separate decision")
    un.add_argument("--yes", action="store_true",
                    help="skip the confirmation prompt for state removal")
    un.add_argument("--models", action="store_true",
                    help="also remove downloaded models (still prompted "
                         "unless --yes)")
    un.add_argument("--keep-models", action="store_true",
                    help="never remove or prompt about downloaded models")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    # serve forwards unrecognised flags to llama-server; every other command
    # treats an unknown flag as a typo and should say so.
    if raw and raw[0] == "serve":
        args, passthrough = parser.parse_known_args(raw)
    else:
        args, passthrough = parser.parse_args(raw), []

    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    os.makedirs(MODELS_DIR, exist_ok=True)

    handlers = {
        "doctor": cmd_doctor, "get": cmd_get, "inspect": cmd_inspect,
        "adopt": cmd_adopt, "up": cmd_up, "down": cmd_down,
        "status": cmd_status, "toggle": cmd_toggle, "compare": cmd_compare,
        "bench": cmd_bench,
        "update": cmd_update, "uninstall": cmd_uninstall,
    }
    if args.command == "serve":
        return cmd_serve(args, passthrough)
    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        return 0
    return handler(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)
    except BrokenPipeError:
        # e.g. `revv doctor | head`
        os._exit(0)
