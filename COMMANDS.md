# What revv does, command by command

The README keeps one line per thing. This is the longer account. Every claim
here is measured; the protocols are in [BENCHMARKS.md](BENCHMARKS.md).

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

## Where the speed comes from

Three levers, each measured on its own, each a property of the model file
rather than of the server. That is why other model families may gain nothing:
a file with no draft head and no thinking mode to turn off has no lever to pull.

- **Thinking mode off.** The models ship with reasoning on, which roughly
  triples the tokens spent per answer. Off: same pass rate, 2.8× faster
  wall-clock per task. Bigger than everything else combined.
- **MTP self-speculation.** The model file carries its own draft head. Depth 2
  is worth +68%, quality-neutral by a paired HumanEval-164 run, not bit-exact.
- **An n-gram drafter chained in front of it.** Editing work mostly re-emits
  text already in the prompt, so a literal matcher lands long runs for free:
  2.8–6.1× on editing, byte-identical, no effect on writing new code.

The seven-lever long form, including the ones that are capacity trades rather
than speed wins, is in [LIMITS.md](LIMITS.md).
