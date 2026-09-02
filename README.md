# revv

**by Mericanii**

### Your GPU has more in it. Proven.

## What this is

revv is a terminal app that runs Qwen3.8-27B GGUFs on a consumer NVIDIA GPU at a
configuration that was actually measured on real hardware, not assumed from a
spec sheet or a forum post. What makes it different from another "fast local
LLM" script: the numbers below are published with the exact protocol that
produced them, including the ones that went against us.

## The receipts

| setup | decode t/s | HumanEval-164 | source |
|---|---|---|---|
| naive offload, out of the box | 2.12 (band 2-4.5) | — | our measurement |
| tuned community recipe | ~9.7 claimed, 6.6-8.5 replicated | — | third-party claim, not ours |
| same weights, llama.cpp defaults (no speculation) | 20.00 | — | our measurement |
| revv certified | 36.7 | 92.7% | our measurement |

Peak VRAM at the certified config: **11,958 MiB** on a 12GB card. Full protocol,
hardware, and raw numbers are in [BENCHMARKS.md](./BENCHMARKS.md).

Two honesty notes, up front, so nobody has to dig for them:

- The 36.7 t/s figure is the kernel-patched build. The same shipping config on
  stock, unpatched llama.cpp measures 34.39 t/s. Both numbers are real and both
  are ours.
- Comparing 36.7 against the 2.12 row is a ~17x ratio, but that row is a naive,
  out-of-the-box setup. Against a **tuned** community baseline the honest
  comparison is **~2.2x**, and that is the number to hold us to. Anyone quoting
  a bare "default is ~6 t/s" figure is citing something nobody measured.

## Quickstart — three paths in

### a. Certified (download the tested weights)

```
./install.sh
revv doctor
revv get
revv up
```

- `./install.sh` — builds the patched llama-server.
- `revv doctor` — checks GPU / VRAM / driver / llama-server / models, reports
  your tier and what's possible.
- `revv get` — downloads the certified GGUF from HuggingFace (resumable,
  verifies exact byte size, then parses it).
- `revv up` — starts the stack in the background; detached, survives the
  terminal closing, logs to `~/.revv/logs/`.

### b. Adopt (reuse what you already downloaded)

```
revv adopt
```

Finds Qwen3.8 GGUFs that ollama or LM Studio already downloaded and registers
them so `serve` can use them. Read-only on their stores. No second 10 GB
download.

### c. Bring your own GGUF

```
revv serve /path/to/your.gguf
```

Runs the stack in the foreground against any GGUF you already have on disk.

For all three: point your tool at `http://127.0.0.1:8080/v1`, then run
`revv compare` to see the difference on your own hardware.

## Use with your coding tool

revv does not touch, wrap, patch, or replace your coding harness. It gives you
a local OpenAI-compatible endpoint and nothing more. Whatever you already use
keeps working.

**The generic case, and the trap.** The official OpenAI SDKs (Python and Node)
read `OPENAI_BASE_URL`. LiteLLM-based tools (aider and many wrappers) read
`OPENAI_API_BASE`. These are different environment variables. Setting both is
harmless:

```
export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
export OPENAI_API_BASE=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=revv        # any non-empty string; the SDKs refuse an empty key
```

**aider** ([docs](https://aider.chat/docs/llms/openai-compat.html)):

```
export OPENAI_API_BASE=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=revv
aider --model openai/revv
```

Equivalent flags exist: `--openai-api-base` and `--openai-api-key`.

**opencode** ([docs](https://opencode.ai/docs/providers/)) is configured by
file, not environment. Global config lives at `~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "revv": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "revv (local)",
      "options": { "baseURL": "http://127.0.0.1:8080/v1" },
      "models": { "revv": { "name": "Qwen3.8-27B (revv)" } }
    }
  }
}
```

Then select the model as `revv/revv` (provider-id/model-id).

**Caveat that will cost you an hour if you skip it:** use `127.0.0.1`, not
`localhost`. llama-server binds IPv4, and `localhost` can resolve to `::1`
first on some systems. The model name to configure, everywhere, is `revv`.

## The commands

| command | what it does |
|---|---|
| `revv doctor` | check GPU / VRAM / driver / llama-server / models, report the tier and what is possible |
| `revv get [tier]` | download the certified GGUF from HuggingFace (resumable; verifies exact byte size, then parses it) |
| `revv adopt` | find GGUFs ollama or LM Studio already downloaded and register them so serve can use them, read-only on their stores |
| `revv inspect <f.gguf>` | parse the header: quant, size, vocab, layers, and whether the MTP draft head is present |
| `revv up` | start the stack in the background (detached; survives the terminal closing; logs to `~/.revv/logs/`) |
| `revv down` | stop it, and reap the llama-server if it was orphaned |
| `revv status` | mode, model, port, uptime, last measured t/s, VRAM |
| `revv serve` | the same stack in the foreground, verbose, for debugging; unknown flags pass through to llama-server |
| `revv toggle [mode]` | switch between revv and STOCK without moving the port |
| `revv compare` | run the same prompt through both modes, side by side |
| `revv bench` | run the certified-protocol benchmark against a running server |

Flags worth knowing: `--port` (default 8080), `--host`, `--ctx`, `--tier`,
`--stock`. Run `revv serve --print-command` if you just want the exact
llama-server command line printed instead of executed.

## How it works

`revv up` / `revv serve` bind a small forwarder on the user-facing port and run
llama-server on an ephemeral internal port behind it. Switching modes restarts
only the backend; the user-facing port never moves, so client tools never
notice. Weights stay in the page cache across a restart, so a switch is
typically 10-15 s rather than a cold load.

Two modes:

- **revv** — the certified configuration: MTP speculation, q8_0 KV, thinking off.
- **STOCK** — llama.cpp's defaults for exactly those three levers: no
  speculation, f16 KV, thinking on. Same weights, same GPU, same context.

STOCK is a control for revv's own configuration. It is **not** a measurement of
ollama, LM Studio, or anyone else's product, and that applies everywhere
`revv compare` shows up.

## Supported / not supported

**Supported:**
- Qwen3.8-27B GGUFs
- NVIDIA, 12GB+ VRAM
- Turing (compute 7.5) or newer
- Linux

**Compatible but not certified:**
- GGUFs from other quantizers and finetunes of this model. The single biggest
  speed determinant is whether the build kept the MTP draft head: builds
  without it (typically anything below ~8.4 GiB) cannot speculate and land
  near 20 t/s instead of the certified 36.7 t/s. Run `revv inspect` before you
  commit to a download — it reports whether the draft head is present.

**Not supported:**
- FP8 / AWQ / EXL3 — not yet
- Under 12GB VRAM — not yet
- AMD — not yet
- Apple Silicon — not yet
- CPU-only — not yet

Also not in v1.0: no command for session save/restore (the patch exists in
`patches/`, the CLI just doesn't drive it), no multi-GPU splitting, no
autostart/systemd/launchd unit, no fine-tuning, quantizing, or training.

## Why these numbers are real

The full measurement protocol — hardware, prompt, sampling, warm-up and
discard rules — is published in [BENCHMARKS.md](./BENCHMARKS.md), not
summarized away here. Failures and retractions are published in that same
document alongside the wins: an early "86-148 t/s" ngram speculation claim was
retracted after it turned out to be a benchmark artifact from replaying the
same prompt, and a thinking-mode bug in the eval harness invalidated weeks of
this project's own quality numbers before it was caught. `revv bench` runs the
identical protocol against your own server, on your own hardware, so you can
reproduce these numbers or contradict them. If your numbers disagree with
ours, we want to hear about it.

## Honest limitations

- The 12GB tier has about 86 MiB of headroom (11,958 MiB peak of roughly
  12,044 MiB usable). A desktop session running on the same card will push it
  into a CUDA OOM. Run headless.
- Speculation gain depends on content: +110% on code, -2 to -4% on prose. A
  prose-heavy workload will not see the headline number.
- One card, one model family, one workload measured. Quality is measured on
  HumanEval-164 and edit-format compliance (IQ3_XXS 94.1% vs Q2_K_XL 67.6%,
  p=0.0117), not on your codebase.

## Requirements / install notes

Python 3.9+, standard library only, no pip dependencies. The CUDA toolkit is
only needed if you build llama.cpp yourself via `install.sh`. No sudo,
anywhere. Models land in `~/.revv/models`, logs in `~/.revv/logs`.

## Credits

revv stands on this ecosystem, it doesn't replace it:

- **llama-swap** — the stable-port, swappable-backend pattern `revv toggle` uses.
- **llamafile** — the bar for distribution simplicity.
- **llama.cpp** — revv is a configuration and measurement layer on top of it.
  Both patches in `patches/` are proposed upstream, not forked.
- **unsloth** — the Dynamic GGUF quants the certified build comes from.

## Licence

Apache-2.0. See [LICENSE](./LICENSE).
