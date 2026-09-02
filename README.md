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

## Quickstart

```
git clone https://github.com/mericanii-technologies/revv && cd revv
./install.sh          # checks python3 + llama.cpp, offers a patched build
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

## Supported

- Qwen3.8-27B GGUF files (any quant; `inspect` tells you what a file supports —
  some third-party conversions strip the speculation head)
- NVIDIA GPUs, 12GB+ VRAM, Turing or newer, Linux
- Community finetunes of the same architecture (works; our quality numbers don't transfer)

## Not supported (yet)

- <12GB VRAM, AMD, Apple Silicon, Windows (WSL2 works)
- FP8 / AWQ / EXL3 formats
- Other model families — the pipeline generalizes; certification takes days per model

Numbers here come from one card and one protocol. If yours differ, run
`./revv.py bench` and open an issue — hardware reports are the most useful
contribution right now.

Credits: [llama.cpp](https://github.com/ggml-org/llama.cpp) does the heavy
lifting; quantized files by [Unsloth](https://huggingface.co/unsloth);
model by the Qwen team. Apache-2.0.
