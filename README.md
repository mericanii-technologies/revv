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

## Tested on two systems

Everything above is one headless Linux box. On 2026-09-08 we installed the
prebuilt on a Windows 11 desktop running WSL2, monitor attached to the card, and
ran both builds through the same five commands. Raw record:
[TEST_WSL2_RESULTS.md](TEST_WSL2_RESULTS.md); tables in BENCHMARKS.md §19.

| | Linux box | Windows PC, WSL2 |
|---|---|---|
| card, OS | RTX 3060 12GB, Ubuntu 24.04 headless | RTX 3060 12GB driving the display, WSL2 Ubuntu 26.04 |
| host | 10-vCPU KVM guest of a Ryzen 5 3600, 47 GB RAM, `-t 8` | Ryzen 5 3600 (6c/12t), 32 GB RAM with 24 GB given to WSL2, `-t 6` |
| free VRAM at launch | 12,044 MiB | 11,516–11,841 MiB, moving with the desktop |
| dense: context, bench | 12,288, 37.9 t/s | 8,192, 36.25 t/s |
| dense: compare, time to done | 22.5 → 38.4 t/s, 46.0 → 22.9 s | 21.9 → 34.6 t/s, 94.2 → 25.2 s |
| MoE: context, bench | 16,384, 55.9 t/s | 12,288, 47.21 t/s |
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


Preconditions, checked first: Linux x86_64, glibc 2.38+ (Ubuntu 24.04+; 22.04
will not work), and an NVIDIA driver providing `libcuda.so.1`. The CUDA
*runtime* libraries `libcudart`/`libcublas`/`libcublasLt` ship inside the
archive, so no compiler and no CUDA toolkit are needed. The binary is built
for sm_75/80/86/89/90 (Turing through Hopper/Ada); **RTX 50-series (sm_120) is
not included** — use `--upstream` or `--source` there. If a precondition
fails, `install.sh` names it and falls back to building.

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
and will drop a rung: on our WSL2 box that is 8,192 for the dense build and
12,288 for the MoE one. `revv doctor` shows the reservation and the choice.

**3. From source — the fallback.** `./install.sh --source` clones llama.cpp at
the pinned commit, applies the two patches in `patches/` (or `--stock` for
none), and builds with CUDA. You can read every patch first. It needs cmake, a
CUDA toolkit, and a host compiler `nvcc` accepts — that chain is the single
biggest obstacle to a first working install, and the reason the prebuilt
exists. `./install.sh --upstream` fetches official llama.cpp binaries instead,
but on Linux those are Vulkan, not CUDA: it runs, the kernel patch does not
apply, and none of these numbers were measured on it. `revv doctor` reports
which path built the binary you are running.

## Install problems we hit, and what to do

Everything below is from the WSL2 install on 2026-09-08. Items marked *fixed in
revv* need only an update; the rest are on your side.

- **`apt install` fails with 404s on a fresh WSL image.** The image ships a
  stale package index. Run `sudo apt update` before anything else.
- **`nvidia-smi: command not found` over SSH or in a script, though it works in
  a terminal.** WSL2 puts it in `/usr/lib/wsl/lib`, and only login shells add
  that to PATH. *Fixed in revv:* revv falls back to that path itself.
- **Ubuntu shows half your RAM.** WSL2 caps the guest at 50% of the host by
  default, which can hide enough to rule out the MoE build (it wants ~8–9 GB
  free, on top of the VRAM). Put `memory=24GB` under `[wsl2]` in
  `%UserProfile%\.wslconfig`, a Windows file you write from PowerShell or from
  Linux at `/mnt/c/Users/<you>/.wslconfig`, then `wsl --shutdown` from
  PowerShell. *Fixed in revv:* `revv doctor` now says this on WSL2.
- **Linux starts throwing I/O errors or "Bus error" and the VM dies.** The WSL
  virtual disk lives on `C:` and grows on demand, so when `C:` fills the guest's
  disk fails under it. Free space on `C:`, or put models on another drive with
  `REVV_HOME`. Interrupted model downloads resume.
- **Closing the last Ubuntu window kills the VM**, and with it a running server
  and any download in flight. Keep one open, or set `vmIdleTimeout=-1`.
- **Windows desktop apps hold 0.5–1 GB of the card.** revv plans against what is
  free at launch, so it will refuse or serve a smaller context. Close browsers
  and the like before `revv up`; opening them mid-session can OOM the server.
- **`revv up` refuses on a card that should fit, or plans the MoE build far too
  small.** Both were ours, fixed on 2026-09-08 in `3f9542d` and `4d126fc`. If
  you cloned before that date, `git pull` or `revv update`.
- **The installer says to install `cuda-cudart`.** A false warning, fixed in
  `3f9542d`; the CUDA runtime is inside the archive. Ignore it on an old clone.
- **You already have an `llama-server` on PATH.** revv installs its own anyway
  and says so, because its numbers were measured on the patched build. Pass
  `--system-llama-server` to use yours instead.

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

## Supported

- **Models:** Qwen3.6-35B-A3B and Qwen3.8-27B in the GGUF builds above. Other
  quants of the same two models run; `revv inspect` tells you what a file
  supports, since some third-party conversions strip the draft head.
- **Hardware:** NVIDIA, 12GB or more of *free* VRAM, Turing or newer, Linux
  (WSL2 works).

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

## Feedback

Hardware reports, corrections, questions: **contact@mericanii.com**, or open an
issue. `revv bench` plus `revv doctor` output from a card we have not tested is
the most useful contribution right now.

Credits: [llama.cpp](https://github.com/ggml-org/llama.cpp) does the heavy
lifting; quantized files by [Unsloth](https://huggingface.co/unsloth); models by
the Qwen team.
