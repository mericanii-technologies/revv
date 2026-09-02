# Changelog

All notable changes to revv are documented here.
This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-09-02

### Added
- First public release.
- `revv doctor`: detects GPU, VRAM, driver and compute capability, locates llama-server, reports the tier and whether the certified configuration fits.
- `revv get`: resumable HTTPS download of the certified GGUF from HuggingFace, verified against an exact expected byte size.
- `revv adopt`: registers Qwen GGUFs that ollama or LM Studio already downloaded, read-only on their stores, so there is no second download.
- `revv inspect`: minimal GGUF v3 header parser reporting quantization, size, vocabulary, layers, and whether the MTP draft head is present.
- `revv up` / `revv down` / `revv status`: background daemon lifecycle, with orphaned llama-server processes reaped on shutdown.
- `revv serve`: the same stack in the foreground for debugging; unknown flags pass through to llama-server.
- `revv toggle`: switches between the certified revv configuration and stock llama.cpp defaults without moving the user-facing port.
- `revv compare`: runs one prompt through both modes and prints them side by side.
- `revv bench`: the certified-protocol benchmark, four requests of 400 greedy tokens, compared against the published reference.
- A stable local forwarder on port 8080 backed by llama-server on an internal port, so client tools are unaffected by backend restarts.
- `patches/mmvq_iquant_decode.patch`: CUDA i-quant decode speedup, +10.1% raw and +2.5% on the shipping speculative-decoding configuration.
- `patches/pr26004-rebased-daef7b687.patch`: llama.cpp PR 26004 rebased onto the pinned commit, making llama-server slot save/restore work on this hybrid recurrent architecture (18x on the first request after restore).
- `install.sh`: no-sudo bootstrap that locates or builds llama.cpp with CUDA, optionally with both patches applied.
- BENCHMARKS.md: the full measurement record, including retractions.

### Verified

- End-to-end smoke test on an RTX 3060 (2026-09-02): every phase passed.
  37.86 t/s flagship at 11,830 MiB peak, 0.8% spread, `revv compare` at 2.01x
  time-to-done, toggle in 3.4 s, clean teardown and orphan reaping. Thinking
  suppression, the one open risk, was confirmed against a positive control and
  the GGUF's embedded chat template was shown byte-identical to the external
  one on 25/25 HumanEval tasks — so no template file ships.

### Fixed

- `revv bench`'s thinking canary could not detect the failure it documented:
  every request sent `chat_template_kwargs` itself, which is merged over the
  server default and masked a broken server-side flag. The measured requests
  now send no kwarg, and a three-arm probe (no kwarg / kwarg false / positive
  control) reports PASS, FAIL or INCONCLUSIVE explicitly.
- Launch flag migrated from the upstream-deprecated
  `--chat-template-kwargs '{"enable_thinking":false}'` to `--reasoning off`.
  Identical mechanism; revv probes `--help` and falls back automatically on
  builds that predate it.
- `revv down` accepts `--url`, like every other subcommand, and works without
  a run file so an instance on a non-default port can be stopped.
- `revv --version` reports the git short SHA when running from a checkout, and
  `revv doctor` no longer trusts llama.cpp's build number when git-describe
  metadata is missing — it reports the commit, which is what actually
  identifies the build.
- `install.sh` pins `CMAKE_CUDA_ARCHITECTURES` to the detected card instead of
  building the whole default fan-out, honours `CUDAARCHS`, falls back to
  `native`, and clears a stale CMake cache entry.
- `revv bench` compares against 37.9 t/s, measured with the bench protocol,
  rather than the 34.4/36.7 certification figures taken with a different
  prompt. `revv compare`'s default budget is 2048 tokens so the STOCK arm
  finishes instead of hitting the cap.

### Naming

- Released as **revv**, by Mericanii. The project was developed under the
  working name MLS (Mericanii Local Stack); that name never shipped, so there
  is nothing to migrate. Anything still referring to `mls`, `MLS_HOME`, or
  `~/.mls` predates this release and is stale.

### Certified configuration
Qwen3.8-27B-UD-IQ3_XXS with MTP speculation at n=2, q8_0 KV cache, 16384 context, flash attention on, all layers on GPU, and thinking disabled server-side. Measured on an RTX 3060 12GB: 36.7 t/s decode with the kernel patch (34.4 t/s stock), 92.7% on HumanEval-164, 11,958 MiB peak VRAM.

### Known limitations
- NVIDIA only; no AMD, Apple Silicon, or CPU path.
- Requires 12GB or more of VRAM and compute capability 7.5 or newer.
- Multi-GPU splitting is untested.
- Session save and restore is not wired into the CLI; the patch is carried but must be driven through llama-server's own endpoints.
- No autostart unit; `revv up` uses a detached process.
- The 12GB tier leaves roughly 86 MiB of headroom, so a desktop session on the same card will exhaust VRAM. Run headless.
