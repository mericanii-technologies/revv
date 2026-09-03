# revv

by [Mericanii](https://github.com/mericanii-technologies). Runs Qwen3.8-27B properly on a consumer NVIDIA GPU.

Most setups run this model far below what the hardware allows — the default
model file doesn't fit a 12GB card, speculation is off, and a template default
makes every answer ~3× longer than it needs to be. revv is a small terminal
tool that applies a measured configuration and gives you a standard
OpenAI-compatible endpoint. Everything below was measured on an RTX 3060;
protocols are in [BENCHMARKS.md](BENCHMARKS.md).

## What it does

- **Picks a model file that actually fits your VRAM.** The popular 4-bit file
  (15.4GB) spills off a 12GB card and runs at ~5 tok/s. The 3-bit file (10.9GB)
  fits, runs 4× faster, and scores the same on HumanEval within noise
  (92.7% vs 94.5%, n=164, statistically indistinguishable).
- **Turns on the model's built-in speculative decoding** (MTP head). Lossless —
  output is byte-identical — and ~1.7× faster. Off in every default setup we checked.
- **Disables thinking mode by default.** This model ships with reasoning on,
  which triples tokens per answer. With it off: same pass rate, ~2.8× faster
  per task.
- **Tunes KV cache and context** to measured values, not folklore (quantized
  KV is a capacity trade, not a speed win — we measured it both ways).
- Optional patches (in `patches/`, pending upstream): a CUDA kernel fix
  (+3% decode) and session save/restore for this model family (resume an 8K
  session in 0.9s instead of 16.7s).

## Results (RTX 3060 12GB, same weights throughout)

| setup | decode | notes |
|---|---|---|
| default 4-bit file, ollama-style settings | 4.7 tok/s | model spills to CPU; thinking on |
| right-sized 3-bit file, stock flags | ~20 tok/s | fits VRAM, no speculation |
| **revv** | **37.9 tok/s** | + ~2.8× fewer tokens per task |

On a 12-task workload (10 HumanEval + 2 long-context), the default setup
finished 5/12 in 28.6 minutes; revv finished 12/12 in 1.5 minutes.
Quality: 92.7% HumanEval-164, statistically equal to the uncompressed Q8
model (93.3%) under the same protocol.

Verified output samples — physics sim, SVG drawing, threaded code, CUDA —
with how each was checked: [examples/](examples/)

## Starting from zero (no model downloaded yet)

**Linux:**
```
# either let revv fetch the certified file directly (recommended):
git clone https://github.com/mericanii-technologies/revv && cd revv
./install.sh && ./revv.py get && ./revv.py up

# or, if you prefer ollama for model management:
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen3.8:27b        # note: this pulls a 4-bit file that won't fit 12GB cards
./revv.py adopt                # revv finds it, warns about the fit, offers the right file
```

`./install.sh` no longer compiles anything by default — it downloads a prebuilt
binary. See [Install paths](#install-paths-and-what-you-are-trusting) if you
would rather not run a third-party build.

**Windows:** use WSL2 (revv needs Linux + the NVIDIA driver's WSL CUDA support):
```
wsl --install -d Ubuntu        # from PowerShell (admin), reboot, open Ubuntu
# then, inside Ubuntu — note: a working nvidia-smi does NOT mean the CUDA
# toolkit is installed; install.sh needs cmake and nvcc to build llama-server.
# Use NVIDIA's WSL repo and a RECENT toolkit (Ubuntu's packaged CUDA and older
# 12.x toolkits fail against new glibc/gcc — verified on a real setup):
sudo apt update && sudo apt install -y cmake build-essential
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb && sudo apt update
sudo apt install -y cuda-toolkit-13-3   # or newest cuda-toolkit-13-x listed
export PATH=/usr/local/cuda/bin:$PATH   # add to ~/.bashrc too
# then follow the Linux steps above

# WSL2 VRAM note: Windows typically reserves ~1-1.5GB of the GPU. revv reads
# free VRAM (not total) and automatically picks the largest context that fits,
# so on WSL2 it will usually choose 8192 instead of 16384 and tell you why.
# `./revv.py doctor` shows the reservation and the choice. --ctx overrides it.
```

**Windows native (PowerShell, no WSL) — manual config, untested by us:**
the revv tool itself needs Linux for now (installer + daemon), but the
configuration it applies works with llama.cpp's official Windows CUDA build:
```powershell
# 1. get llama.cpp: download the latest cudart+bin win-cuda zip from
#    https://github.com/ggml-org/llama.cpp/releases and unzip
# 2. get the certified model file (10.9GB):
Invoke-WebRequest -Uri "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-UD-IQ3_XXS.gguf" -OutFile Qwen3.8-27B-UD-IQ3_XXS.gguf
# 3. run with the certified flags:
.\llama-server.exe -m .\Qwen3.8-27B-UD-IQ3_XXS.gguf -ngl 99 -fa on -c 16384 `
  -ctk q8_0 -ctv q8_0 --spec-type draft-mtp --spec-draft-n-max 2 --parallel 1 `
  --jinja --reasoning off --cache-ram 0 --port 8080
```
That's the whole trick minus the conveniences (no adopt/toggle/bench/patches).
Our numbers are from Linux; Windows-native reports welcome. A native Windows
revv is on the list.

`./revv.py get` downloads the certified file (unsloth Qwen3.8-27B-UD-IQ3_XXS,
10.9GB) with resume support. `./revv.py inspect <file>` explains any GGUF you
already have.

## Quickstart

```
git clone https://github.com/mericanii-technologies/revv && cd revv
./install.sh          # downloads a prebuilt llama-server. No compiler needed.
./revv.py adopt       # finds Qwen3.8 models you already downloaded via ollama
                      # (or: ./revv.py get   to download the certified file)
./revv.py up          # starts in the background on localhost:8080
```

Then point anything OpenAI-compatible at it:

```
export OPENAI_BASE_URL=http://localhost:8080/v1
export OPENAI_API_KEY=revv
```

To see the difference on your own card: `./revv.py compare` runs the same
prompt in stock mode and revv mode, back to back. `./revv.py bench` measures
your decode speed against our reference numbers. `./revv.py down` stops everything.

## Install paths, and what you are trusting

`install.sh` has three rungs. The default downloads a binary, because the CUDA
build chain — cmake, then the toolkit, then a host compiler `nvcc` accepts,
then a glibc that toolkit accepts — is the single biggest obstacle to a first
working install. One reported setup lost about half an hour to it.

| | command | what you get | what you trust |
|---|---|---|---|
| **1. default** | `./install.sh` | our patched llama-server, CUDA, the certified config | a **third-party fork build** signed off by Mericanii |
| **2. upstream** | `./install.sh --upstream` | the official llama.cpp prebuilt | **official upstream binaries**, nothing of ours |
| **3. source** | `./install.sh --source` | you compile it, patched or `--stock` | **your own machine** — you can read every patch first |

**The awkward fact about rung 2.** Upstream llama.cpp publishes prebuilt CUDA
binaries **for Windows only**. For Linux it ships CPU, Vulkan, ROCm and SYCL —
no CUDA. So "official prebuilt" on a Linux NVIDIA box means the *Vulkan* build:
it will run, but it is a different backend, the CUDA kernel patch does not
apply to it, and **none of the numbers in this README were measured on it**.
That is why rung 2 is a deliberate choice and revv will never fall back to it
silently. Verified against the GitHub API on 2026-09-03 for builds b10712,
b10770 and b10776.

**Rung 2 is also the older-distro option.** The upstream Vulkan build needs only
glibc 2.34, so it runs on Ubuntu 22.04, where our CUDA prebuilt (glibc 2.38)
does not. If you are on an older distro and do not want to compile, that is the
path — at the cost of the Vulkan backend's unmeasured performance.

Rung 1 requirements, checked before it will install: Linux x86_64, **glibc 2.38
or newer** (Ubuntu 24.04+; 22.04 will not work), and the CUDA *runtime*
libraries — `libcudart`, `libcublas`, `libcublasLt`, which are far smaller than
the full toolkit and need no compiler. The binary is built for **sm_86**
(Ampere, 30-series); other cards rely on driver JIT and `--source` is the
reliable path there. If any precondition fails, `install.sh` says which one and
falls back to rung 3 automatically.

Every download is pinned to a tested release tag, never "latest", and verified
by sha256. `revv doctor` reports which rung produced the binary you are running.

**This path is new.** The prebuilt has been packaged and its provenance
verified, but it has not yet been installed from scratch on a machine other
than the one that built it. If it fails on yours, that report is genuinely
useful — open an issue with `revv doctor` output.

## Supported

- Qwen3.8-27B GGUF files (any quant; `inspect` tells you what a file supports —
  some third-party conversions strip the speculation head)
- NVIDIA GPUs, 12GB+ VRAM, Turing or newer, Linux
- Community finetunes of the same architecture (works; our quality numbers don't transfer)

**Other models will run, but may gain nothing.** revv's speed comes from
levers that are properties of the model, not of the server: speculative
decoding needs an MTP draft head in the file, and the thinking-off win needs a
chat template that has a thinking mode to turn off. A model with neither —
most GGUFs — gets no benefit, and a measured field case ran 2.5% *slower* in
revv mode than stock before revv learned to check. **Models without a draft
head may see little or no gain — `revv inspect` will tell you before you
serve.** revv now picks its flags per model, says which levers apply, and when
none do it says so plainly and serves the best-known stock config instead of
staging a meaningless A/B. If your model has no built-in draft head you can
still get speculation by supplying your own drafter — `revv serve --draft
small-model.gguf` (experimental, uncertified; community MTP drafts exist for
some families, revv will not download one for you, and `revv bench` reports
the acceptance rate so you can judge whether that pair is worth it).

## Roadmap

**1. A prebuilt Linux CUDA binary, hosted on Releases. — SHIPPING.**
`./install.sh` now downloads instead of compiling. The remaining work is
breadth: the binary is built for sm_86 only, so a multi-architecture build is
needed before it covers most cards, and it has not yet been installed from
scratch on a machine other than the one that built it. The original note is
kept below because it is still why this mattered. Right now the single
biggest thing standing between a new user and a working setup is not revv and
not the model — it is the CUDA build gauntlet: cmake, then the CUDA toolkit,
then a host-compiler version nvcc accepts, then a glibc that toolkit accepts.
A real fresh install burned about half an hour walking that chain before
landing on `cuda-toolkit-13-3`. `install.sh` now detects the common mismatches
early and tells you the fix instead of failing deep in a build, but detecting a
problem is a consolation prize. A prebuilt binary deletes the problem. Until it
ships, building from source is the supported path.

**2. Auto-sizing beyond context.** revv now sizes context to free VRAM. The
same treatment for KV precision and speculation depth is the natural next step.

**3. Certification on a second GPU tier.** Everything here is one card. A 16GB
and a 24GB tier need their own measurements before they stop being derived
settings.

## Not supported (yet)

- <12GB VRAM, AMD, Apple Silicon, Windows (WSL2 works)

**Under 12GB — what we think would happen (unmeasured):** on an 8GB card
(3070/3070 Ti class) the certified 10.9GB file spills to CPU and we'd expect
~8-12 tok/s. The 2-bit file (6.8GB) fits and should be fast — possibly
35-50 tok/s given the memory bandwidth — but quality drops measurably
(~78% HumanEval vs 92.7%, and instruction-following degrades more than that
number suggests). Fine for chat, not recommended for agent work. `revv doctor`
will tell you which case you're in. If you try it anyway, `revv bench` +
an issue report would genuinely help — nobody has published numbers for
these cards yet. The proper 8GB tier arrives when a strong-enough smaller
model exists.
- FP8 / AWQ / EXL3 formats
- Other model families — the pipeline generalizes; certification takes days per model

Numbers here come from one card and one protocol. If yours differ, run
`./revv.py bench` and open an issue — hardware reports are the most useful
contribution right now.

Feedback, hardware reports, questions: **contact@mericanii.com** or open an issue.

Credits: [llama.cpp](https://github.com/ggml-org/llama.cpp) does the heavy
lifting; quantized files by [Unsloth](https://huggingface.co/unsloth);
model by the Qwen team. Apache-2.0.
