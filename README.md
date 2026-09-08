# revv

by [Mericanii](https://github.com/mericanii-technologies). Apache-2.0.

revv is a small terminal program, Python standard library only, that launches
llama.cpp with a measured configuration for one of two Qwen coding models on a
12GB-or-larger NVIDIA card, and serves a plain OpenAI-compatible endpoint on
localhost. It is not a new inference engine: llama.cpp does the work, and revv
picks the flags, keeps the server alive, and gets out of the way. Every number
in this repository was measured on our own hardware, and the protocols are in
[BENCHMARKS.md](BENCHMARKS.md).

## What you can do with it

- **Two certified builds.** `moe` is Qwen3.6-35B-A3B, UD-Q3_K_XL, 16.0 GiB to
  download, **55.9 t/s** generation. `dense` is Qwen3.8-27B, UD-IQ3_XXS, 10.2
  GiB, **37.9 t/s**. Both on an RTX 3060 12GB.
- **Any harness that speaks the OpenAI API** — opencode, aider, Continue, your own
  script — points at it and works. revv does not touch the harness, send your
  prompts anywhere, or train anything.
- The speed comes from three levers, each measured on its own. **Thinking mode
  off** is 2.8× faster wall-clock per task at the same pass rate, and is bigger
  than the rest combined. **MTP self-speculation** at depth 2, the draft head
  inside the model file, is worth +68% and is quality-neutral by paired
  HumanEval. **An n-gram drafter chained in front of it** is worth 2.8–6.1× on
  editing work, byte-identical, with no effect on writing new code.
- **`doctor`** says what this machine can run and the exact plan it will use.
  **`get`** downloads a certified file with a hash check, and resumes.
  **`up`**, **`down`** and **`status`** run the server.
- **`bench`** measures your decode rate against our reference, **`compare`**
  runs one prompt through stock and revv mode back to back, and **`toggle`**
  switches modes without moving the port.
- **`adopt`** reuses GGUFs already pulled through ollama or LM Studio,
  read-only; **`inspect`** explains any GGUF you have, including whether it
  still carries a draft head; **`update`** and **`uninstall`** are one command.
- **`serve --print-command`** prints the exact llama-server command line revv
  would run, so you can run the same configuration by hand with no revv at all.
- **A prebuilt binary with the CUDA runtime bundled**, so a first install needs
  no compiler and no CUDA toolkit.
- **Context sized to the VRAM that is actually free**, not the total, with a
  checkpoint guard near the ceiling that stops a die-on-second-request failure.
- **It installs into `~/.revv` and the clone directory, nothing else.** No
  sudo, no shell-profile edits, no Python packages. An existing llama.cpp,
  ollama or LM Studio install is left exactly as it was.

## Results on two systems

The Linux box is a Proxmox VM: Ubuntu 24.04 headless, 10 vCPU of a Ryzen 5
3600, 47 GB RAM, an RTX 3060 12GB by passthrough, `-t 8`. The Windows PC is
Windows 11 with WSL2 Ubuntu 26.04: Ryzen 5 3600 (6c/12t), 32 GB RAM with 24 GB
given to WSL2, the same model of 3060 driving the display, `-t 6`. On
2026-09-08 we installed the prebuilt on the PC and ran both builds through it.

| | Linux box | Windows PC, WSL2 |
|---|---|---|
| free VRAM at launch | 12,044 MiB | 11,516–11,841 MiB, moving with the desktop |
| `moe`: generation, stock flags | 22.2 t/s | 18.7 t/s |
| `moe`: generation, revv | **55.9 t/s** (2.52×) | **47.21 t/s** |
| `moe`: editing, revv | 63 → ~188 t/s mean, 243 peak (+197%) | not run |
| `moe`: context served | 16,384 | 12,288 |
| `moe`: HumanEval-164 | 153/164 | not run |
| `moe`: editing instrument, 34 tasks | 9/34 first try, 16/34 overall | not run |
| `dense`: generation, stock flags | 22.5 t/s | 21.9 t/s |
| `dense`: generation, revv | **37.9 t/s** | **36.25 t/s** |
| `dense`: editing, revv | 40 → 113–246 t/s (2.8–6.1×) | not run |
| `dense`: context served | 12,288 | 8,192 |
| `dense`: HumanEval-164 | 152/164 | not run |
| `dense`: editing instrument, 34 tasks | 4/34 first try, 8/34 overall | not run |

Quality is identical by construction across the two machines: same file, same
flags, same binary. The dense build lands within 5% of the headless box (36.25
against 37.9) because it sits entirely on the GPU, and all the desktop costs it
is one context rung. The MoE build reaches 84% of it (47.21 against 55.9)
because its 16 expert layers stream from host RAM, and on the PC that path
carries two fewer threads, Hyper-V, and Windows running alongside. On our
editing instrument the MoE build is both faster and more capable than the dense
one, 9/34 against 4/34 first-attempt, p=0.039, which is why the builds are
named for what they are rather than by tier.

Protocols and every number: [BENCHMARKS.md](BENCHMARKS.md), §19 for the second
machine; raw record [TEST_WSL2_RESULTS.md](TEST_WSL2_RESULTS.md); caveats and
provenance [LIMITS.md](LIMITS.md).

## Install

```
git clone https://github.com/mericanii-technologies/revv && cd revv
./install.sh          # downloads a prebuilt llama-server; no compiler needed
./revv.py doctor      # check the card before downloading 16 GB
./revv.py get moe     # or: ./revv.py get dense
./revv.py up
```

Preconditions, checked first: Linux x86_64, glibc 2.38+ (Ubuntu 24.04+; 22.04
will not work), an NVIDIA driver providing `libcuda.so.1` (`nvidia-smi` prints
a table), and 12 GB of free VRAM. The MoE build additionally wants about 24 GB
of system RAM, since its CPU-resident experts want ~8–9 GB free on top of the
VRAM; if a precondition fails, `install.sh` names it and falls back to building.

On WSL2, install the NVIDIA driver on the Windows host and never inside the
guest, because CUDA is passed through from the host driver. Raise the guest's
RAM with `memory=24GB` under `[wsl2]` in `%UserProfile%\.wslconfig`, since WSL2
caps it at half the host by default and that can hide enough RAM to rule out
the MoE build. Expect a smaller context, because Windows reserves roughly
1–1.5 GB of the card and revv plans against what is free; the rest of the WSL2
detail is in [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

`./install.sh --source` builds llama.cpp from the pinned commit with the two
patches in `patches/` instead; it needs cmake and a CUDA toolkit, and is the
path for RTX 50-series cards.

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
keywords `speed` and `flagship` still work but print a deprecation note,
because they were inverted against the editing result above. `revv get` with no
argument picks by host RAM, 24 GB or more gets the MoE build and less gets the
dense one, and says why. If you have the RAM, start with `revv get moe`.

`./revv.py uninstall` stops the server, shows what it will delete, and asks;
downloaded models are a separate question, so you can keep the 16 GB file and
drop everything else.

## Supported, and not

Supported: Qwen3.6-35B-A3B and Qwen3.8-27B in the GGUF builds named above
(other quants of the same two models run, uncertified); NVIDIA, 12 GB or more
of *free* VRAM, Turing or newer; Linux x86_64, and Windows through WSL2.

Not supported: under 12 GB free VRAM, AMD, Apple Silicon, native Windows,
CPU-only, multi-GPU splitting; RTX 50-series (sm_120) on the prebuilt, which
needs `./install.sh --source`. Other model families run but may gain nothing:
revv's speed comes from the model rather than the server, so a file with no
draft head and no thinking mode to turn off has no lever to pull. The reasons,
and what we measured, are in [LIMITS.md](LIMITS.md).

## Read more

- [TROUBLESHOOTING.md](TROUBLESHOOTING.md): install problems we hit, by symptom.
- [LIMITS.md](LIMITS.md): limits and known issues, other cards, what revv touches.
- [BENCHMARKS.md](BENCHMARKS.md): protocols and every number.
- [EXPERIMENTS.md](EXPERIMENTS.md): what we tried that did not pan out, and why.
- [TEST_WSL2_RESULTS.md](TEST_WSL2_RESULTS.md): the raw second-machine record.
- [CHANGELOG.md](CHANGELOG.md): what changed, release by release.

## Feedback and credits

Hardware reports, corrections, questions: **contact@mericanii.com**, or open an
issue. `revv bench` plus `revv doctor` output from a card we have not tested is
the most useful contribution right now.

Credits: [llama.cpp](https://github.com/ggml-org/llama.cpp) does the heavy
lifting; quantized files by [Unsloth](https://huggingface.co/unsloth); models by
the Qwen team.
