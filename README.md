# revv

by [Mericanii](https://github.com/mericanii-technologies). Apache-2.0.

revv runs three certified coding models fast on a 12GB NVIDIA card. It is a
small terminal program that launches llama.cpp with a configuration we
measured, and serves an OpenAI-compatible endpoint on localhost. llama.cpp
does the work; revv picks the flags. Not a new engine, not a fine-tune.

## The models

| name | model | file | download | tested on |
|---|---|---|---|---|
| `moe` | Qwen3.6-35B-A3B | Unsloth UD-Q3_K_XL (MTP build) | 16.0 GiB | RTX 3060 12GB, Linux and WSL2 |
| `dense` | Qwen3.8-27B | Unsloth UD-IQ3_XXS | 10.2 GiB | RTX 3060 12GB, Linux and WSL2 |
| `gemma` | Google Gemma 4 26B-A4B | bartowski IQ3_XXS + sidecar MTP head | 11.3 + 0.3 GiB | RTX 3060 12GB, Linux |

These three files, on this card, are the whole of what is certified. Other
quants of the same models run, uncertified. Other model families run but may
gain nothing, because the speed comes from the model file, not the server.
`moe` needs about 24 GB of system RAM; `gemma` needs about 4 GB; `dense` needs
none beyond the VRAM. Pick `moe` for editing and agent work -- it is the
strongest line on our editing instrument. Pick `gemma` for raw speed or a
16 GB-RAM machine -- it is the fastest line measured, but the weakest editor.
Pick `dense` when RAM is the constraint and `gemma`'s speed is not needed.

## What it does

- Picks measured llama.cpp flags for the model and the card: thinking off,
  the model's own draft head for speculation, an n-gram drafter for editing,
  context sized to the VRAM that is actually free.
- Serves `http://127.0.0.1:8080/v1` for any OpenAI-compatible tool: opencode,
  aider, Continue, your own script.
- `doctor` says what your machine can run; `bench` measures it against our
  reference; `compare` runs stock and revv back to back on the same card.
- Reuses GGUFs you already have from ollama or LM Studio, read-only.
- Installs into `~/.revv` and the clone directory only. No sudo, no profile
  edits, no Python packages. Existing llama.cpp, ollama and LM Studio installs
  are left alone. `uninstall` removes it.

Every command, and where the speed comes from: [COMMANDS.md](COMMANDS.md).

## Results

RTX 3060 12GB. Linux box: Proxmox VM, Ubuntu 24.04 headless, 10 vCPU of a
Ryzen 5 3600, 47 GB RAM, GPU by passthrough. Windows PC: Windows 10, WSL2
Ubuntu 26.04, Ryzen 5 3600, 24 GB of 32 GB given to WSL2, same 3060 driving
the display.

| | stock llama.cpp | revv, Linux box | revv, Windows PC |
|---|---|---|---|
| `moe` generation | 22.2 t/s | **55.9 t/s**, 16K context | 47.2 t/s, 12K context |
| `moe` editing | 63 t/s | ~188 t/s mean, 243 peak | not run |
| `dense` generation | 22.5 t/s | **37.9 t/s**, 12K context | 36.3 t/s, 8K context |
| `dense` editing | 40 t/s | 113–246 t/s | not run |
| `gemma` generation | not shippable (OOM, 116 MiB headroom) | **70.7 t/s**, 16K context | not run |
| `gemma` editing | not shippable (OOM) | 287 t/s | not run |

At 8K context the dense build runs draft depth 3, 40 t/s instead of 38, measured quality-neutral.

Quality: `moe` and `dense` tie their uncompressed anchor on HumanEval-164
(153/164 and 152/164); `gemma` scores 157/164, tied with both within noise. On
a 34-task multi-file editing instrument `moe` solved 9/34 first try against
`dense` 4/34 (p=0.039); `gemma` solved 7/34 overall against `moe`'s 16/34
(p=0.012) and 1/34 first try against `moe`'s 9/34 (p=0.008) on the same 34
tasks, which is why it is the speed line, not the editor. The Windows PC gets
less context because the desktop holds part of the card, and `moe` runs at
84% of the box because its expert layers stream from system RAM through the
WSL2 layer.

Protocols and every number: [BENCHMARKS.md](BENCHMARKS.md). Second machine:
[TEST_WSL2_RESULTS.md](TEST_WSL2_RESULTS.md).

## Install

Linux x86_64 (Ubuntu 24.04 or newer), an NVIDIA driver (`nvidia-smi` works),
12 GB of free VRAM. No compiler, no CUDA toolkit: the prebuilt bundles the
runtime.

```
git clone https://github.com/mericanii-technologies/revv && cd revv
./install.sh
./revv.py doctor        # what this machine can run
./revv.py get moe       # or: get dense / get gemma
./revv.py up
```

On WSL2: driver on Windows only, never inside the guest; give WSL2 24 GB with
`memory=24GB` in `%UserProfile%\.wslconfig`; expect one context rung less.
Anything that goes wrong: [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
RTX 50-series: `./install.sh --source`.

## Use

```
export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=revv
```

Point your tool at that. `./revv.py status` shows what is running,
`./revv.py down` stops it, `./revv.py uninstall` removes revv and asks
separately about the downloaded models.

`revv up moe --long` serves 128K context on the MoE build: 47 t/s at short
context, about 17 t/s with the context full, 22 expert blocks on the CPU.
The default 16K profile is 56 t/s. Both measured; BENCHMARKS.md §20.

`revv up moe --streams 4` serves four requests at once. Speculation is off in
that mode (the shipped build cannot run the draft head per slot), so a single
stream is slower, but total output is higher: measured 69 t/s aggregate on the
MoE and 47 on the dense build at four streams, 79 and 57 at eight. Use it for
parallel agents; use the default for one chat.

## Supported

NVIDIA with 12 GB or more of free VRAM, Turing or newer; Linux, or Windows
through WSL2; the three models above. Not: under 12 GB free, AMD, Apple
Silicon, native Windows, CPU-only, multi-GPU. Details, other cards, known
issues: [LIMITS.md](LIMITS.md).

## Paper

[Cheaper intelligence on P.O.S. hardware: what one RTX 3060 can be pushed to infer](paper/cheaper-intelligence.pdf) (PDF, 11 pages). The argument, the numbers, the 22 failed ideas, and what it means for the consumer installed base.

## More

[COMMANDS.md](COMMANDS.md) every command and the levers ·
[TROUBLESHOOTING.md](TROUBLESHOOTING.md) problems by symptom ·
[LIMITS.md](LIMITS.md) limits, other cards, what revv touches ·
[BENCHMARKS.md](BENCHMARKS.md) every number ·
[EXPERIMENTS.md](EXPERIMENTS.md) what failed and why ·
[CHANGELOG.md](CHANGELOG.md)

Reports from cards we have not tested (`revv bench` and `revv doctor` output),
corrections, questions: **contact@mericanii.com** or open an issue.
Credits: [llama.cpp](https://github.com/ggml-org/llama.cpp),
[Unsloth](https://huggingface.co/unsloth), the Qwen team.
