# revv

by [Mericanii](https://github.com/mericanii-technologies). Apache-2.0.

revv is a small terminal program, Python standard library only, that launches
llama.cpp with a measured configuration for one of two Qwen coding models on a
12GB-or-larger NVIDIA card, and serves a plain OpenAI-compatible endpoint on
localhost. Any harness that speaks that API — opencode, aider, Continue, your
own script — points at it and works. revv does not touch the harness, send your
prompts anywhere, or train anything. It picks flags, keeps the server alive,
and gets out of the way. Every number below was measured on an RTX 3060 12GB;
protocols are in [BENCHMARKS.md](BENCHMARKS.md), and the work that did not pan
out is in [EXPERIMENTS.md](EXPERIMENTS.md).

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
- **q8_0 KV cache.** Not a speed win — quantized KV is measurably *slower* than
  f16 at every depth we tested, because it moves attention onto a compute-bound
  kernel. It is a capacity trade, and f16 does not fit this model on 12GB. We
  measured it both ways rather than assuming.

## Results

RTX 3060 12GB, Ubuntu 24.04, headless.

| | 35B-A3B (MoE) | 27B (dense) |
|---|---|---|
| model | Qwen3.6-35B-A3B, UD-Q3_K_XL | Qwen3.8-27B, UD-IQ3_XXS |
| download | 16.0 GiB | 10.2 GiB |
| generation, stock flags | 22.2 t/s | 22.5 t/s |
| generation, revv | **55.9 t/s** (2.52×) | **37.9 t/s** |
| editing, revv | 63 → ~188 t/s mean, 243 peak (+197%) | 40 → 113–246 t/s (2.8–6.1×) |
| context revv serves | 16,384 | 12,288 |
| peak VRAM | 11,832 MiB | 11,822 MiB |
| host RAM needed | ~8 GiB free, on top of the VRAM | not a factor |
| HumanEval-164 | 153/164 | 152/164 |
| multi-file editing, 34 tasks | 9/34 first try, 16/34 overall | 4/34 first try, 8/34 overall |

Reading that table honestly:

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

## Install

You need Linux and a working NVIDIA driver (`nvidia-smi` prints a table).

**1. Linux, prebuilt — simplest.**

```
git clone https://github.com/mericanii-technologies/revv && cd revv
./install.sh          # downloads a prebuilt llama-server; no compiler needed
./revv.py doctor      # check the card before downloading 16 GB
./revv.py get moe     # or: ./revv.py get dense
./revv.py up
```

> **TODO (founder):** the prebuilt GitHub Release is **not published yet**. As
> of 2026-09-05 the releases list is empty and the asset URL 404s, so
> `install.sh` falls through to building from source for everyone. This section
> is written for how it will read once the release is tagged. Publish the sm_86
> artifact, then delete this block.

Preconditions, checked first: Linux x86_64, glibc 2.38+ (Ubuntu 24.04+; 22.04
will not work), and the CUDA *runtime* libraries
`libcudart`/`libcublas`/`libcublasLt`, which need no compiler. The binary is
built for sm_86 (30-series); other cards fall back to driver JIT, where
`--source` is the reliable path. If a precondition fails, `install.sh` names it
and falls back to building.

**2. Windows, via WSL2.** The rule that breaks most first runs: **install the
NVIDIA driver on the Windows host, before touching WSL2. Never install an
NVIDIA driver inside the WSL2 guest** — CUDA is passed through from the host
driver. A working `nvidia-smi` inside Ubuntu does not mean the CUDA toolkit is
present.

```
wsl --install -d Ubuntu          # PowerShell (admin), reboot, open Ubuntu
sudo apt update && sudo apt install -y git cmake build-essential
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt update
sudo apt install -y cuda-toolkit-13-3    # or the newest cuda-toolkit-13-x listed
export PATH=/usr/local/cuda/bin:$PATH    # add to ~/.bashrc too
# then follow the Linux steps
```

Ubuntu's packaged CUDA and older 12.x toolkits fail against recent glibc/gcc,
which is why the NVIDIA WSL repo is used above. Expect a smaller context:
Windows reserves roughly 1–1.5 GB of the card, revv plans against free VRAM,
and will pick 8192 or lower. `revv doctor` shows the reservation and the choice.

**3. From source — the fallback.** `./install.sh --source` clones llama.cpp at
the pinned commit, applies the two patches in `patches/` (or `--stock` for
none), and builds with CUDA. You can read every patch first. It needs cmake, a
CUDA toolkit, and a host compiler `nvcc` accepts — that chain is the single
biggest obstacle to a first working install, and the reason the prebuilt
exists. `./install.sh --upstream` fetches official llama.cpp binaries instead,
but on Linux those are Vulkan, not CUDA: it runs, the kernel patch does not
apply, and none of these numbers were measured on it. `revv doctor` reports
which path built the binary you are running.

## Usage

Five commands do everything:

```
./revv.py doctor      # what this machine can run, and what it will pick
./revv.py get moe     # download a certified file (resumable)
./revv.py up          # start in the background on 127.0.0.1:8080
./revv.py status      # mode, model, port, uptime, VRAM
./revv.py down        # stop everything, including orphaned servers
```

Then point any OpenAI-compatible client at it:

```
export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=revv
```

The two builds are named `moe` (the 35B-A3B) and `dense` (the 27B). The old
keywords `speed` and `flagship` still work, but they print a deprecation note,
because they were inverted against the editing result above. `revv get` with no
argument picks by host RAM — 24 GB or more gets the MoE build, less gets the
dense one, since the MoE build's CPU-resident experts want ~8–9 GB — and says
which and why. If you have the RAM, `revv get moe` is the one to start with.

To see the difference on your own card: `revv compare` runs one prompt through
stock and revv mode back to back, `revv bench` measures your decode rate
against the reference, and `revv toggle` switches modes without moving the
port. `revv inspect <file>` explains any GGUF you already have, and
`revv adopt` finds models pulled through ollama or LM Studio.

## Supported

- **Models:** Qwen3.6-35B-A3B and Qwen3.8-27B in the GGUF builds above. Other
  quants of the same two models run; `revv inspect` tells you what a file
  supports, since some third-party conversions strip the draft head.
- **Hardware:** NVIDIA, 12GB or more of *free* VRAM, Turing or newer, Linux
  (WSL2 works).

The planner's rules are general — read free VRAM not total, size context to
fit, disable checkpoints near the ceiling, don't quantize KV for speed. Only
these two builds are certified. Certification takes days per model.

## Not supported

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

## Limits and known issues

- **12GB is the real floor, and it is tight.** revv refuses to start below
  11,528 MiB of free VRAM. A 12GB 3060 reports 12,288 MiB and offers about
  12,044; the rest is driver-reserved. Certified configs land with 212–222 MiB
  of real headroom, so a desktop session on the same card can cause an OOM.
- **WSL2 gets less.** If the host reserves more than ~760 MiB, revv refuses
  outright; below that it serves a smaller context. Both are intended.
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
- **The prebuilt has never been installed on a machine other than the one that
  built it**, and revv has not been run end to end on a second GPU.
- Every number here is one card, one protocol, one workload type. Speculation
  speedup is a property of the content: +110% on code, −2% to −4% on prose.

## Feedback

Hardware reports, corrections, questions: **contact@mericanii.com**, or open an
issue. `revv bench` plus `revv doctor` output from a card we have not tested is
the most useful contribution right now.

Credits: [llama.cpp](https://github.com/ggml-org/llama.cpp) does the heavy
lifting; quantized files by [Unsloth](https://huggingface.co/unsloth); models by
the Qwen team.
