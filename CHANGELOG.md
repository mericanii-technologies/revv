# Changelog

All notable changes to revv are documented here.
This project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-09-02

### Added

- **Zero-compile install.** `./install.sh` now downloads a prebuilt
  `llama-server` instead of building one. Three rungs: our patched CUDA
  prebuilt (default), the official upstream prebuilt (`--upstream`), or a
  source build (`--source`, the old path with all its preflights; `--patched`
  and `--stock` still work as aliases). Downloads are pinned to a tested
  release tag, never "latest", and sha256-verified. If Linux/x86_64/glibc-2.38
  preconditions fail, or the download does, it says which one and falls back to
  the source build — but it never falls back to `--upstream` silently, because
  that is a different backend. `revv doctor` reports which rung produced the
  binary in use.
- Documented the trust tradeoff in the README, including the awkward fact that
  upstream llama.cpp ships **no Linux CUDA prebuilt** (Windows only; verified
  against the GitHub API across builds b10712/b10770/b10776), so "official
  prebuilt" on a Linux NVIDIA box means the Vulkan backend, which none of our
  numbers were measured on.


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

### Roadmap

- **A prebuilt Linux CUDA binary on Releases is the next priority.** The
  from-source CUDA build (cmake, toolkit, host compiler, glibc) is the largest
  obstacle to a first working install; a fresh WSL2 setup spent roughly 30
  minutes on toolchain mismatches alone. `install.sh` now catches the common
  ones early, but the real fix is not making people build at all.

### Verified

- End-to-end smoke test on an RTX 3060 (2026-09-02): every phase passed.
  37.86 t/s flagship at 11,830 MiB peak, 0.8% spread, `revv compare` at 2.01x
  time-to-done, toggle in 3.4 s, clean teardown and orphan reaping. Thinking
  suppression, the one open risk, was confirmed against a positive control and
  the GGUF's embedded chat template was shown byte-identical to the external
  one on 25/25 HumanEval tasks — so no template file ships.

### Added

- **External draft models.** `revv serve --draft <file.gguf>` / `revv up
  --draft ...` enables speculative decoding on targets with no built-in MTP
  head, via llama-server's `--spec-draft-model`. revv reads the drafter's
  header to choose `draft-mtp` or `draft-simple`, keeps it on the GPU, charges
  its weights and KV against free VRAM when sizing context, and warns on a
  vocabulary mismatch before llama-server refuses the pair. Experimental and
  uncertified: revv never downloads a third-party drafter, and `revv bench`
  now reports the acceptance rate, because acceptance is a property of the
  target/drafter PAIR — a drafter that suits a base model can be worthless on
  a finetune of it. `revv compare` reports acceptance per mode too.

### Fixed

- **The certified model was demoted when adopted from ollama.** Model identity
  was matched on filename, but `revv adopt` registers ollama blobs whose
  filename is a content hash, so the certified file fell through to the
  geometric KV estimate — which over-estimates threefold on this hybrid
  architecture and cost two-thirds of the context (ctx 4,096 + q4_0 instead of
  8,192 + q8_0 at 11,744 MiB free). Identity is now matched on exact file size
  as well as name.
- An explicit `--ctx` below the smallest ladder rung was silently snapped
  upward (`--ctx 2048` became 4096), quietly handing out more context than
  asked for. Explicit contexts are now honoured exactly, with a warning if the
  arithmetic says they will not fit.
- When no context fitted at q8_0, the planner jammed context to the floor and
  KV to q4_0 together; it now retries the ladder at q4_0 first, since halving
  the cache is cheaper than losing seven-eighths of the context.

### Changed

- **revv now picks its flags per model instead of applying the Qwen-certified
  set to everything.** A field report measured a Gemma-4-12B running 2.5%
  *slower* in revv mode than stock: no draft head so no speculation, no thinking
  switch to disable, and quantized KV, which the kernel profile shows is a
  compute tax that only pays when VRAM is tight. `serve`/`up` now decide each
  lever from what `inspect` reads — speculation only when a draft head exists,
  the thinking flag only when the chat template implements one, and KV precision
  by need (f16 when it fits at the chosen context, quantized only to buy
  capacity) — and print the choice with its reason. When no lever applies revv
  says "this model gains nothing from revv's tuned mode -- serving with the
  best-known stock config", and `toggle`/`compare` report that state instead of
  staging an A/B whose two arms are the same configuration. The certified Qwen
  config is provably unchanged by this: on a clean 12 GB card it still resolves
  to ctx 16,384, q8_0 KV and speculation on.



- **`revv compare` and `revv bench` now measure the same quantity.** A field
  report showed them disagreeing by 14% on one machine (compare 29.7 t/s vs
  bench 34.31). compare now requests `timings_per_token`, so llama-server's own
  decode rate is available on every streamed chunk instead of only the final
  one, whose shape varies by build; it reports the client-observed rate
  alongside and flags a gap above 5% as transport cost on that host; and it
  discards one warmup exchange per mode, as bench always has. The proposed
  cold-path-after-toggle explanation was tested with a warm control arm and did
  not hold: no restart-specific penalty above 2.6%, and the discrepancy runs in
  opposite directions on two different machines.



- **Context and tier are now chosen from FREE VRAM, not total.** Windows/WSL2
  reserves 1-1.5 GB of the GPU, so the certified c=16384 config OOMed on a
  12GB card that passed a total-VRAM check. `doctor`, `serve` and `up` read
  `nvidia-smi memory.free`, report any host reservation, and automatically
  select the largest context that fits with a 250 MiB margin, printing what
  was chosen and why. `--ctx` still overrides, and warns if the arithmetic
  says it will not fit.
- `install.sh` preflights the CUDA/host-compiler/glibc combination and fails
  early with both remedies rather than dying deep in a build.

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
