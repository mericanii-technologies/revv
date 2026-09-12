# Limits, other hardware, and what revv touches

This file holds the detail that [README.md](README.md) summarises. Every figure
in it was measured on our own hardware, with the protocols in
[BENCHMARKS.md](BENCHMARKS.md).

## What it does under the hood

Seven things, each measured separately on the rig in BENCHMARKS.md §2.

- **Thinking mode off.** These models ship with reasoning on, which roughly
  triples the tokens spent per answer. Off: same pass rate, **2.8× faster
  wall-clock per task** (4.79 s vs 13.38 s, 158.8 vs 474 tokens). Bigger than
  every flag below combined.
- **MTP self-speculation.** The model file carries its own draft head. Depth 2
  is worth **+68%**. Quality-neutral by measurement (identical per-task outcomes
  across all 164 HumanEval tasks, p=1.0) but **not bit-exact** — 3 of 5 greedy
  probes differ from a no-speculation run, because the batched verify pass sums
  floats in a different order. Depth 3+ loses more than it gains.
- **An n-gram drafter chained in front of it.** Editing work is mostly
  re-emitting file content already visible in the prompt, so a literal n-gram
  matcher lands long runs for free: **2.8–6.1× on editing, byte-identical
  output, and exactly zero effect on writing new code** (1.00×, no cost — it
  misses and falls through).
- **A thread count for CPU-offloaded experts.** The MoE build streams 16 expert
  layers from host RAM, which puts RAM bandwidth on the critical path. Setting
  `-t` to the physical core count is worth **+14%**; the default
  oversubscribes. Output is bit-identical across the whole sweep.
- **Context sized to free VRAM, not total.** revv reads `memory.free` and picks
  the largest context off a ladder that fits with a margin. On WSL2, where
  Windows reserves 1–1.5 GB, that means a smaller context instead of an OOM.
- **A checkpoint guard near the ceiling.** llama.cpp keeps 32 context
  checkpoints per slot at ~150 MiB each, allocated lazily, so a tight config
  loads, passes its health check, serves one request, then dies on the second
  with an error mentioning neither memory nor checkpoints. revv sets `-ctxcp 0`
  whenever the planned peak leaves under 500 MiB free.
- **q8_0 KV cache.** On the dense 27B it is not a speed win — quantized KV is
  measurably *slower* than f16 at every depth we tested there, because it moves
  attention onto a compute-bound kernel; it ships because f16 does not fit that
  model on 12GB. On the hybrid MoE the opposite holds at depth: q8_0 decodes
  1.7× faster than f16 at 15K tokens in context (BENCHMARKS.md §21), so the
  planner keeps q8_0 on the MoE line even when f16 fits. Measured both ways
  on both builds rather than assumed.

## Reading the results table honestly

The table is in [README.md](README.md). Reading it honestly:

- The stock column is llama.cpp with defaults and a file that fits. The 35B
  figure is a stock `llama-server` at `-ngl 30`, no tuning, from the same
  campaign as the 55.9. The 27B figures are one `revv compare` session (22.5
  stock, 38.4 revv); `revv bench` uses a different prompt and reads 37.9, the
  reference figure it compares your machine against. Both in BENCHMARKS.md §15.
- **An ollama-style default is a worse starting point than either.** The popular
  4-bit file does not fit a 12GB card; it spills to CPU and runs at **2–4.5
  t/s** (our measured point: 2.12 t/s). Picking a file that fits is most of the
  first jump.
- **Editing is a range because it depends on how much of the answer is already
  in the prompt.** Rewriting a file you pasted in hits the top; inventing new
  code hits none of it. Both figures are byte-identical to the same config with
  the matcher off, so nothing is traded for the speed.
- **On our editing instrument the 35B is both faster and more capable** — 9/34
  vs 4/34 first-attempt, p=0.039. The 27B produces well-formed edits (34/34
  format compliance) that are more often wrong. That is why this README names
  the builds by what they are rather than by tier. What we have *not* measured:
  the 27B is a generation newer, dense, and needs no host-RAM headroom, and
  vendor reasoning claims for it are unreplicated by us. Its case is real and
  unmeasured, not disproven.
- No head-to-head prefill number: the two figures we have (~500 t/s for the 27B,
  205 t/s for the 35B) were taken under different configs on different dates.

## The two systems, in detail

The results table comparing the two systems is in [README.md](README.md); the
raw record is in [TEST_WSL2_RESULTS.md](TEST_WSL2_RESULTS.md). The `revv
compare` sessions behind it, which the README table does not carry:

| | Linux box | Windows PC, WSL2 |
|---|---|---|
| dense: compare, time to done | 22.5 → 38.4 t/s, 46.0 → 22.9 s | 21.9 → 34.6 t/s, 94.2 → 25.2 s |
| MoE: compare, time to done | 22.2 → 55.9 t/s, not measured | 18.7 → 41.9 t/s, 110.1 → 24.5 s |

Three caveats. The Linux MoE stock figure is a stock `llama-server` at
`-ngl 30` from the same campaign, not a `revv compare` session. The PC's MoE
compare ran at 4,096 context with q4_0 KV, before the planner fix that lifted
that box to 12,288. And in both compare sessions the stock arm ran out its token
budget without finishing (1,024 and 2,048), so those time-to-done figures are
lower bounds, and not comparable across columns.

The dense build lands 4.4% off the headless box (36.25 against 37.9) because it
sits entirely on the GPU; all the desktop costs it is one context rung, 8,192
instead of 12,288. The MoE build loses about 16% (47.21 against 55.9) because
its 16 expert layers stream from host RAM, and here that path is worse in three
ways at once: two fewer threads, Hyper-V between the process and RAM, and
Windows running alongside. The stock arm dropped by the same fraction (18.7
against 22.2 t/s), which is how we know the cause is the machine and not revv's
configuration. Quality is identical by construction: same file, same flags, same
binary; the n-gram chain is byte-identical, and MTP is quality-neutral by paired
HumanEval.

## Peak VRAM and host RAM, per build

| | moe | dense |
|---|---|---|
| model file | Qwen3.6-35B-A3B, UD-Q3_K_XL | Qwen3.8-27B, UD-IQ3_XXS |
| download | 16.0 GiB | 10.2 GiB |
| peak VRAM | 11,832 MiB | 11,822 MiB |
| host RAM needed | ~8 GiB free, on top of the VRAM | not a factor |

## Limits and known issues

- **12GB is the real floor, and it is tight.** revv refuses to start below
  11,457 MiB of free VRAM, the least the dense build has been measured to
  run in (4096 context, 276 MiB to spare) plus a 150 MiB reserve. A 12GB 3060 reports 12,288 MiB and offers about
  12,044; the rest is driver-reserved. Certified configs land with 212–222 MiB
  of real headroom, so a desktop session on the same card can cause an OOM.
- **WSL2 gets less, and it moves.** If Windows holds more than ~830 MiB of
  the card, revv refuses outright; below that it serves a smaller context.
  Both are intended. Windows's share is not fixed: on our second 3060 it
  read 1,022 MiB with a browser open and 275 MiB with the desktop cleared,
  and revv plans against whatever is free at launch. Close GPU-using apps
  before `revv up`, and expect an OOM if you open them mid-session. With the
  desktop cleared (11.5–11.8 GB free) that box serves the MoE build at
  8,192–12,288 context and the dense build at 8,192; both measured in
  BENCHMARKS.md §19. WSL2 also caps Linux at half the PC's RAM by default,
  which can hide enough of it to rule out the MoE build; `memory=24GB` in
  `%UserProfile%\.wslconfig` fixes that on a 32 GB machine.
- **The planner's headroom figure is an upper bound that drifts.** It anchors on
  a measured peak and scales only the KV term, so it is optimistic by ~26 MiB at
  the anchor context and ~104 MiB three rungs down the ladder. Shipped configs
  still clear our ≥200 MiB standard, but the small-context margin is thinner
  than it looks.
- **MTP speculation is not bit-exact.** Quality-neutral by a full paired
  HumanEval-164 run, not byte-identical. If you need reproducible bytes, turn
  speculation off.
- **The n-gram matcher needs LF line endings.** It is a literal byte match; a
  repo checked out with CRLF drops acceptance from 0.83 to 0.11.
- **The prebuilt has been installed on exactly one machine other than the one
  that built it**: a second RTX 3060 under WSL2, Ubuntu 26.04, where it served
  both builds, the dense one at 8,192 context and the MoE one at 12,288
  (BENCHMARKS.md §19, raw record in
  [TEST_WSL2_RESULTS.md](TEST_WSL2_RESULTS.md)). That is the whole of the
  independent evidence so far.
- Every number here is one card, one protocol, one workload type. Speculation
  speedup is a property of the content: +110% on code, −2% to −4% on prose.

## Other cards

The planner's rules are general — read free VRAM not total, size context to
fit, disable checkpoints near the ceiling, don't quantize KV for speed. Only
these two builds are certified. Certification takes days per model.

**Other cards.** Every number above is from one RTX 3060. On a bigger card
revv runs the same two files and the speed levers still apply, because they
are properties of the model, not the card: thinking off, the MTP head, and
the n-gram chain all come with the file. What changes is the context: a 16 GB
card gets 32K, a 24 GB card gets 64K with f16 KV, neither separately
measured. Two things to know. The stock baseline is better on a big card,
since a 4-bit file already fits, so expect the *ratio* to shrink even as the
absolute number rises. And the MoE build keeps 16 expert layers on the CPU
regardless of VRAM, because that is the certified config; on a 24 GB card
the whole model fits on the GPU and `revv serve moe --n-cpu-moe 0` (flags
after the model name go straight to llama-server) will likely be much
faster. Nobody has measured it. Larger quants of the same two models, such as
Unsloth's Q4_K_XL files, run the same way, uncertified. Below 12 GB of free
VRAM revv refuses to start. RTX 50-series needs `./install.sh --source`.

## Supported, in detail

- **Models:** Qwen3.6-35B-A3B and Qwen3.8-27B in the GGUF builds above. Other
  quants of the same two models run; `revv inspect` tells you what a file
  supports, since some third-party conversions strip the draft head.
- **Hardware:** NVIDIA, 12GB or more of *free* VRAM, Turing or newer, Linux
  (WSL2 works).

## Not supported, and why

- Under 12GB free VRAM, AMD, Apple Silicon, native Windows, CPU-only,
  multi-GPU splitting.
- FP8, AWQ, EXL3. We tested EXL3: it tied on quality, ran slower, and saved
  less VRAM than we had estimated (EXPERIMENTS.md §3).
- **Other model families run but may gain nothing.** revv's speed comes from
  properties of the model, not the server: speculation needs a draft head in
  the file, and the thinking-off win needs a chat template with a thinking mode
  to turn off. A model with neither gets no benefit — we measured one case
  running 2.5% *slower* in revv mode than stock. revv now derives flags per
  model, says plainly when no lever applies, and serves the best-known stock
  config instead of staging a meaningless A/B.

## What revv touches, and how to remove it

revv is experimental, and you are helping test it, so it is built to leave
no trace. It writes to exactly two places:

- **The clone directory** — the code you checked out. Nothing is written
  there except `__pycache__`.
- **`~/.revv`** (or `$REVV_HOME`) — its own llama-server under `bin/`, the
  downloaded runtime, model files under `models/`, logs, and a small
  registry. `revv doctor` prints the path.

It does not: edit your shell profile, use sudo, install Python packages,
write to `/usr/local`, or modify anything it finds. If you already have
llama.cpp, ollama or LM Studio installed, they are left exactly as they were.
`install.sh` installs its own `llama-server` copy even when one is on your
PATH, because revv's numbers were measured on its patched build; pass
`--system-llama-server` if you want yours used instead. `revv adopt` reads
the ollama and LM Studio model directories and never writes to them. The
server binds only to `127.0.0.1`, on 8080 or the next free port.

To remove it:

```
./revv.py uninstall      # stops the server, shows what it will delete, asks
rm -rf revv              # the clone
```

`uninstall` treats downloaded models as a separate question, so you can keep
the 16 GB file and drop everything else. `revv uninstall --yes --models`
removes all of it without asking. Deleting `~/.revv` by hand is equivalent.

The pieces are independent. The two llama.cpp patches in `patches/` are plain
diffs you can read or build without (`./install.sh --source --stock`), the
model files are unmodified Unsloth GGUFs any llama.cpp can load, and
`revv serve --print-command` prints the exact llama-server command line it
would run, so you can run the same configuration by hand with no revv at all.

See also: [README.md](README.md), [BENCHMARKS.md](BENCHMARKS.md), [EXPERIMENTS.md](EXPERIMENTS.md), [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
