# TEST_3060_RESULTS.md — results of the end-to-end smoke test on the RTX 3060

Run: 2026-09-02, box `ollama` (RTX 3060 12GB, driver 535.309.01, sm_86, 10 cores,
47 GB RAM). Executed against `TEST_3060.md` v1.0.

**Headline: RISK 1 PASSES. The shipped mechanism is unchanged** —
`--jinja --chat-template-kwargs '{"enable_thinking":false}'` server-side really
does suppress thinking on the real 27B. No fallback was needed, and none was
implemented.

**Every phase passed.** Two numbers landed outside their stated windows, both in
the favourable direction (decode faster than expected, VRAM lower than expected),
and both are explained below. Four process/documentation defects were found and
are listed in "Defects found".

---

## Summary table

| Phase | Item | Verdict |
|---|---|---|
| 0 | box free, GPU < 500 MiB | **PASS** (1 MiB) |
| 1 | install `--patched`, both patches, manifest | **PASS** (with a deviation, see below) |
| 2 | doctor: tier detection, kernel patch reported | **PASS** |
| 3 | inspect: size / quant / MTP head / CERTIFIED | **PASS** (4/4) |
| 3 | `revv get` download resume on a real network | **PASS** |
| 4 | serve with certified flags, `revv status` | **PASS** |
| 4 | **RISK 1 — thinking suppressed server-side** | **PASS** (`REASONING: 0`) |
| 4 | peak VRAM during requests | **PASS\*** 11,830 MiB (expected ~11,958) |
| 5 | bench: thinking canary silent | **PASS** |
| 5 | bench: decode t/s | **PASS\*** 37.86 t/s (window was 34–37) |
| 5 | bench: spread across 4 requests | **PASS** 0.8% |
| 6 | toggle, port stability, toggle latency | **PASS** (3.4 s / 3.8 s) |
| 6 | compare: revv vs STOCK, time-to-done > decode ratio | **PASS\*** (see caveat) |
| 7 | `down` leaves no orphan | **PASS** (1 MiB, 0 processes) |
| 7 | daemon survives the ssh session that started it | **PASS** |
| 7 | orphan reaping after `kill -9` of the supervisor | **PASS** |

`*` = passed, but the observed value sits outside the number written in
`TEST_3060.md`. Details in each phase.

---

## RISK 1 — the result that mattered

### Verdict: PASS. Thinking is suppressed by the server-side flag alone.

`TEST_3060.md` nominates `revv bench` as "the authoritative" check. **It is not,
and cannot be.** `revv bench` sends the kwarg itself:

    revv.py:2754   "chat_template_kwargs": {"enable_thinking": False},
                   # "Belt and braces: serve already sets this server-side..."

(and `_timed_generation`, which backs `revv compare`, does the same at
`revv.py:2381`). A per-request kwarg **overrides and therefore masks** the
server-side default — `server-common.cpp:1307-1311` merges the request body over
`opt.chat_template_kwargs`. So the bench canary would stay silent even if
`--chat-template-kwargs` were completely ignored. It tests the per-request path,
not the flag under test.

The valid detector is the Phase 4 raw `curl`, whose body contains **no**
`chat_template_kwargs`. That was run, and a dedicated three-arm probe with a
positive control was run alongside it, against the real 27B with revv's exact
`MODE_REVV` argv:

| arm | request body | `reasoning_content` | decode |
|---|---|---|---|
| **A — server-side flag only** | no kwarg | **0 chars** | 37.28 t/s |
| B — per-request kwarg | `enable_thinking:false` | 0 chars | 37.54 t/s |
| **C — positive control** | `enable_thinking:true` | **803 chars** | 32.77 t/s |

Arm C is what makes A meaningful: the detector demonstrably fires when thinking
is on, so A's zero is a real suppression and not a broken probe. Through the full
revv stack (proxy included) the Phase 4 curl also returned `REASONING: 0`.

### Why it works — mechanism confirmed in source, not just observed

1. The GGUF's **embedded** template honours the kwarg. Extracted
   `tokenizer.chat_template` (9,993 bytes) branches on it twice, and with
   thinking off it pre-closes the block in the generation prompt:

       {%- if enable_thinking is defined and enable_thinking is false %}
           {{- '<think>\n\n</think>\n\n' }}

2. `common/arg.cpp:3547` stores the flag into `params.default_template_kwargs`;
   `server-common.cpp:1308` seeds every request from it;
   `common/chat.cpp:3676` does `json::parse("false")`, so jinja receives a real
   **boolean**, not the string `"false"` — which is the subtle way this could
   have silently failed, and does not.

### Two things the test plan did not anticipate

**(a) The flag is deprecated upstream.** The server logs, on every start:

    W Setting 'enable_thinking' via --chat-template-kwargs is deprecated.
      Use --reasoning on / --reasoning off instead.

It works correctly today at the pinned commit. It is on an upstream removal path,
so revv should migrate to `--reasoning off` before un-pinning llama.cpp. This is
the one real follow-up from RISK 1.

**(b) RISK 1 understates the substitution.** `TEST_3060.md` frames the change as
"per-request kwarg → server-side kwarg". revv also swapped the **template
itself**: the certification ran `--chat-template-file .../chat_template.jinja`
(`qwen3.8-froggeric-v22.3`, 26,681 bytes), whereas revv's `--jinja` uses the
GGUF's embedded template (9,993 bytes). Two different files.

That second substitution turns out to be harmless, and this was measured rather
than assumed: both templates implement thinking-off with the identical
`<think>\n\n</think>` prefill, and with thinking off and no tools they render to
the same prompt. Confirmed empirically in TASK 2 — a 25-task HumanEval run under
revv's embedded template produced **byte-identical completions on 25/25 tasks**
versus the certified run under the froggeric template. So the packaging question
"should revv ship that template file?" is answered: **no, it does not need to.**

---

## Phase-by-phase

### Phase 0 — code on the box: PASS
`rsync` to `/data/projects/revv/`. GPU showed **1 MiB** used.

*Deviation:* `REVV_HOME=/data/projects/revv-home`, not `~/.revv`. The root
filesystem had only 9.4 GB free; `/data` had 34 GB. A CUDA build plus binaries
does not fit in 9.4 GB with margin.

*Environment note:* port **8080 was already taken** on this box by an unrelated
pre-existing `python3` service (pid 2570265, not GPU work — left untouched). All
phases therefore ran on **port 8090** via `--port 8090`.

### Phase 1 — install: PASS

    patches must list BOTH ...    -> confirmed

    {
      "base_commit": "daef7b6874397a5a7c3d7e38b55e2ee0adf7da38",
      "patches": ["mmvq_iquant_decode.patch", "pr26004-rebased-daef7b687.patch"],
      "built_at": "2026-09-02T14:41:37Z",
      "revv_version": "1.0.0"
    }

Both patches applied to a pristine checkout of the pinned commit with **zero
overlap** — they touch disjoint files (`ggml/src/ggml-cuda/*` vs
`tools/server/*`). Combined diffstat: 4 files, +263 / −26.

*Deviation, and it is a real gap in coverage:* `install.sh` never passes
`-DCMAKE_CUDA_ARCHITECTURES`, so it would have built for the whole default arch
fan-out (50/61/70/75/80/86/89/90…) — far slower and much larger than needed for a
single sm_86 card. The source tree was therefore pre-seeded (cloned from the
box's local llama.cpp, already at the pinned SHA) and pre-configured with
`-DCMAKE_CUDA_ARCHITECTURES=86`. `install.sh --patched` then ran its normal path
end to end: fetch/checkout, patch check (reported "already applied"), cmake
configure, build, symlink, manifest.

**Consequence: the GitHub network-clone path in `setup_llama_src()` is the one
part of Phase 1 that was not exercised.** Everything after it was.

### Phase 2 — doctor: PASS (exit 0)

    GPU
      [ok]    GPU 0: NVIDIA GeForce RTX 3060
             12,288 MiB total, 1 MiB in use
             driver 535.309.01, compute capability 8.6
      [ok]    tier: 12GB
             certified: 36.7 t/s, 92.7% HumanEval, 11,958 MiB peak
    llama-server
      [ok]    /data/projects/revv-home/bin/llama-server
             version: 0.3.0-dev (build 1, commit daef7b687)
      [ok]    kernel patch applied
             base commit daef7b6874397a5a7c3d7e38b55e2ee0adf7da38
    Models
      [warn]  no models in /data/projects/revv-home/models
    Verdict
      Ready. ...

Tier detection, kernel-patch reporting and exit code all correct. See Defect 2 on
`build 1`.

### Phase 3 — model: PASS (4/4)

    size            9.67 GiB → for the certified file: 10,934,860,704 bytes  ✓
    quantization    IQ3_XXS                                                   ✓
    MTP draft head  present -- 4 tensors, blk.64.nextn.*                      ✓
    verdict         CERTIFIED                                                 ✓

Integrity check run and **matches** the value in `TEST_3060.md`:

    c0b7c3038681ed2e3040456c1dd45f9858b6c2290bed172c70388a94874f3eee

Download resume on a real network: **PASS**. `revv get --build IQ2_XXS`
interrupted at 2,012,217,344 bytes (`.part` file); rerun **resumed at 1.88/6.77
GiB (27.7%)** rather than restarting, growing by 1,132,462,080 bytes in 15 s at
~77 MiB/s. Interrupt message is correct: `interrupted - rerun the same command to
resume`. Partial file deleted afterwards.

### Phase 4 — stack up: PASS

    api      http://127.0.0.1:8090/v1
    mode     REVV -- certified: MTP speculation, quantized KV, thinking off
    tier     12GB   speculation on
    vram     11,642 MiB used of 12,288 MiB  (646 MiB free)

Backend argv confirmed from the live process list — exactly the certified flag
set, including `--jinja --chat-template-kwargs {"enable_thinking":false}
--spec-type draft-mtp --spec-draft-n-max 2 -ctk q8_0 -ctv q8_0 -c 16384
--parallel 1`.

Sanity request: **`REASONING: 0`** — RISK 1 pass condition met.

**Peak VRAM during generation: 11,830 MiB** (expected ~11,958, so 128 MiB / 1.1%
low). This is *not* a missing flag. `q27b_on_12gb/results/stage3.md` §1/§6
independently measured **11,830 MiB** for this exact config; the 11,958 figure
comes from the `speed_recert` harness, whose command line differs (no
`--cache-ram 0`, no `--no-cache-idle-slots`, `--no-warmup` present). 11,830 is the
reproducible number for what revv actually launches.

### Phase 5 — bench: PASS on the canary, above the window on speed

    request 1    38.03 t/s    400 tokens   10.81 s wall
    request 2    37.91 t/s    400 tokens   10.85 s wall
    request 3    37.78 t/s    400 tokens   10.89 s wall
    request 4    37.72 t/s    400 tokens   10.90 s wall

    result
      decode      37.86 t/s mean, 37.84 median
      spread      0.8% across 4 requests (noise floor is ~1%)
    reading
      on target: within 5% of the reference (36.7 t/s).

- **No `thinking is LEAKING` message.** (Silence here is necessary but not
  sufficient — see RISK 1 above.)
- Spread 0.8%, comfortably under the 2% limit. ✓
- **37.86 t/s is outside the stated 34–37 window, on the fast side** (+3.2% vs
  36.7). Not a regression; nothing to escalate. Note the two tolerances in the
  repo disagree: `TEST_3060.md` says 34–37, while revv's own verdict logic uses
  ±5% of 36.7 (34.9–38.5) and reports "on target". See Defect 1.

### Phase 6 — toggle and compare: PASS

Toggle REVV→STOCK **3.4 s**, STOCK→REVV **3.8 s** — far under the 10–15 s claim,
so the page-cache assertion in the README is safe (it is conservative). API URL
unchanged at 8090 across both switches. ✓

    mode     decode t/s     ttft    tokens      wall
    -----------------------------------------------
    STOCK          22.5    0.45s      1024     46.0s
    REVV           38.4    0.37s       865     22.9s

    decode rate   1.71x faster
    time to done  2.01x faster  (46.0s -> 22.9s)
    tokens spent  1024 -> 865  (STOCK thinks out loud; revv does not)

The load-bearing claim holds: **time-to-done ratio (2.01x) exceeds the decode-rate
ratio (1.71x)**, which is the whole "raw t/s is not work speed" argument.

*Caveat for the screenshot:* **STOCK hit the 1024-token budget cap and did not
finish.** Its 1024 tokens is a floor, not a measurement, and 46.0 s / 2.01x are
therefore *lower bounds* — the real gap is wider. The README's "474 vs 158.8
tokens" is a per-HumanEval-task figure from a different harness and should not be
conflated with this table. STOCK also read 22.5 t/s against an expected ~20.
Consider raising the compare budget so STOCK actually terminates before this is
used as the demo.

### Phase 7 — teardown: PASS on all three

- `revv down` → GPU back to **1 MiB**, **0** `llama-server` processes, no CUDA
  compute apps. ✓
- **Detached daemon:** started with `revv up`, closed the ssh session entirely,
  reconnected in a fresh session — `revv status` still worked (pid 2317344). ✓
- **Orphan reaping:** `kill -9` on supervisor 2317344 left backend 2317346 alive
  holding 11,642 MiB. `revv down` then reported:

      revv supervisor (pid 2317344) is already gone.
        reaping orphaned llama-server (pid 2317346)...
      Stopped.

  GPU returned to 1 MiB, 0 processes. The persisted `backend_pid` in
  `run/revv.json` did exactly the job it was added for. ✓

Final box state: **1 MiB VRAM, no llama-server, no revv processes.**

---

## Defects found

1. **`revv bench` cannot detect the failure it is documented to detect.**
   `_bench_once` (revv.py:2754) and `_timed_generation` (revv.py:2381) both send
   `chat_template_kwargs` per request, which overrides the server-side default and
   masks a broken `--chat-template-kwargs`. `TEST_3060.md:44` calls this check
   "the authoritative one"; it is not. *Fix:* drop the per-request kwarg from
   `_bench_once` so the canary measures the shipped configuration, or add a
   distinct `revv bench --no-kwarg` arm. Until then, the raw Phase 4 curl is the
   only valid detector.

2. **Version string reports `build 1`.** `revv doctor` and the deployed binary
   print `version: 0.3.0-dev (build 1, commit daef7b687)`; the same tree built
   directly reports `build 10712`. The commit is right, so provenance is intact,
   but the build number is wrong wherever it is quoted. Cosmetic, caused by
   git-describe metadata in the seeded clone.

3. **`revv down` does not accept `--url`,** although `status`, `toggle`,
   `compare` and `bench` all do. On a box where the default port is occupied,
   `revv down --url http://127.0.0.1:8090` fails with `unrecognized arguments`.
   It works without the flag (it reads the persisted run state), but the CLI is
   inconsistent and the error is confusing.

4. **`install.sh` pins no CUDA architecture,** so a stock run builds the full
   default arch fan-out. On a known single-GPU tier this is wasted build time and
   disk. *Fix:* derive `-DCMAKE_CUDA_ARCHITECTURES` from the detected card
   (86 for the 12GB tier), or accept a `CUDAARCHS` passthrough.

Also worth correcting in the docs: `session_restore/TEST_PLAN.md` §2 verifies the
patch with `strings build/bin/llama-server | grep -c "context checkpoint(s) from"`
and expects `>= 1`. That returns **0** on this build — `llama-server` is now an
18 KB wrapper and the server code lives in `libllama-server-impl.so`, where the
string is correctly present. The check needs to point at the shared library.

---

## What was not run

- The GitHub network-clone path in `install.sh setup_llama_src()` (Phase 1
  deviation above).
- `revv adopt`, and the `--stock` install mode.
- Any multi-client / concurrency behaviour (`--parallel 1` is certified).
