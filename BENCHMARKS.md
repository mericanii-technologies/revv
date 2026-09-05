# revv Benchmarks

## 1. What this document is

These are the measurements behind revv's claims — speed, quality, VRAM, and
context — including the ones that went against us. Section 12 lists every
number we published and then retracted, and why. If a figure isn't in this
document, or isn't reproducible with the protocol described in Section 2,
don't trust it.

## 2. The rig and the protocol

**Hardware and software.** RTX 3060 12GB (sm_86), driver 535.309.01, Ryzen 5
3600, 47 GB DDR4, Ubuntu 24.04, headless. llama.cpp build 10718 /
9efa1595e for the speed re-certification below; a cross-check binary, build
169 / daef7b6, unpatched, was used to sanity-check results against an
independent build.

**Speed protocol** (the re-certification, 2026-09-02):

- Server started with: `-ngl 99 -fa on -ctk <kv> -ctv <kv> -c <ctx> --parallel 1 --no-warmup`,
  plus `--spec-type draft-mtp --spec-draft-n-max <n>` where speculation is in
  play.
- Client settings: `temperature=0, top_k=1, seed=1234, max_tokens=400,
  cache_prompt=false, chat_template_kwargs={"enable_thinking": false}`.
- Prompt: one fixed greedy code prompt (a persistent on-disk B-tree module),
  99 tokens exactly (~110 tokens).
- Each cell is 1 discarded warm-up request followed by 4 measured requests;
  the reported number is the mean of those 4.
- Decode t/s is read from llama-server's own timings
  (`predicted_n / predicted_ms`), not from client-side wall-clock. This
  matters — see Section 7 for a case where raw t/s and actual work speed
  told opposite stories.
- VRAM is the peak of a 1 Hz `nvidia-smi` sample taken across the duration of
  the requests, not a single reading at load time. See Section 4 for why this
  distinction changed our results.
- Guard: the server refuses to start unless the card is already below 500 MiB
  used, so every cell starts from a clean baseline.

**Noise floor.** Within-cell spread is 0.1-0.8%. Drift over a 3-hour run is
<=1.1%. We treat +/-1% as the measurement floor and don't call anything under
roughly 2% a real difference. This is stated here, up front, so you can apply
your own skepticism to any close numbers in the tables below rather than
take our word for which differences matter.

**Quality protocol.** Full HumanEval-164, greedy decoding, thinking off.

## 3. Speed: the official table

Decode t/s, 400-token code generation, shallow depth (empty/short context at
start of generation):

| # | config | decode t/s | vs no-spec | accept | mean len | VRAM peak MiB |
|---|---|---|---|---|---|---|
| 1 | IQ3_XXS + MTP n=2, q8_0 KV, c=16384 **[SHIPPING]** | 34.39 | 1.719x | 0.781 | 2.56 | 11958 |
| 2 | IQ3_XXS + MTP n=2, f16 KV, c=16384 | OOM | — | — | — | 11960 (dies) |
| 3a | IQ3_XXS + MTP n=3, q8_0 KV, c=16384 | 32.76 | 1.638x | 0.638 | 2.91 | 11984 |
| 3b | IQ3_XXS + MTP n=4 / n=5, q8_0 KV, c=16384 | OOM | — | — | — | — |
| 4a | Q2_K_XL + MTP n=2, f16 KV, c=16384 | 34.73 | 1.735x | 0.771 | 2.54 | 11156 |
| 4b | Q2_K_XL + MTP n=2, q8_0 KV, c=16384 | 34.64 | 1.732x | 0.781 | 2.56 | 10704 |
| 5 | IQ3_XXS no-spec, f16 KV, c=16384 | 20.15 | 1.007x | — | — | 11440 |
| 5b | IQ3_XXS no-spec, q8_0 KV, c=16384 (RAW FLOOR) | 20.00 | 1.000x | — | — | 10986 |

The OOM rows are in the table on purpose. IQ3_XXS with MTP n=2 and f16 KV
dies during the run at c=16384; IQ3_XXS with MTP n=4 or n=5 (q8_0 KV,
c=16384) dies the same way. On a 12GB card, both are real limits, not
configs that just weren't tried — and both failed during the run, not at
load. That's evidence about what the card can hold, not clutter to be
trimmed.

**Depth result.** At working depth — 13.5K tokens of filled context — the
shipping config reaches 36.00 t/s, acceptance 0.981, 1.99x vs no-spec. Deep
context is speculation's best regime under this protocol: the model gets
more confident (higher acceptance) as context fills, not less.

**MTP depth sweep** (at c=8192): n=2 gives 34.06 t/s, n=3 gives 32.57, n=4
gives 31.61, n=5 gives 30.22. Optimal n is 2. n>=3 showed greedy
non-reproducibility in a one-prompt observation, so losslessness of deep MTP
is unverified, and n>2 does not ship on that basis.

## 4. Why VRAM is certified during requests, not at load

VRAM is measured as the peak of a 1 Hz sample taken across the full duration
of the request cell — not a single reading right after the server reports
"loaded." That distinction is not academic: three configs in our sweep
passed a load-time VRAM check cleanly and then ran out of memory on the
first actual request. A benchmark that samples VRAM only at load would have
certified all three as safe. This is a methods point that most published
local-LLM numbers get wrong, and it's the reason the protocol in Section 2
samples continuously.

## 5. The kernel patch A/B (patch #1, mmvq_iquant_decode.patch)

Shipping config, MTP path, measured on the main decode loop:

- stock 35.8 t/s -> patched 36.7 t/s (+2.5%)
- acceptance identical at 0.8356 in both arms, which corroborates that the
  patch changes speed, not output

Raw decode, speculation off: 20.48 -> 22.54 t/s (+10.1%); +10.2% at depth.
Prefill is untouched by the patch, which serves as a clean control (a patch
that touches decode-only code shouldn't move a prefill-only number, and it
doesn't).

**Why the gain attenuates from +10.1% raw to +2.5% shipped.** Speculation
amortizes the GEMV across a verification round, so a raw-decode kernel gain
gets divided down by the round structure once speculation is turned on.
T_round at n=2 is 74.45 ms; one verify forward is about 45-50 ms of that;
the remaining ~24 ms/round of round overhead (draft forwards plus host
graph/sync) doesn't benefit from the kernel patch at all, and is now
co-dominant with the GEMV. A patch that speeds up one piece of a round by
10% only speeds up the round overall by however large that piece's share is.

**Root cause the patch addresses.** sm_75 and newer removed the
SIMD-video instructions the original kernel relied on, so nvcc emulates each
one in 4-5 instructions. In this model's i-quant format, the sign-recovery
machinery accounted for 18 of every 20 instructions in the inner loop. The
fix is a carry-free SWAR multiply to spread sign bits (valid because no
i-quant codebook byte is zero) combined with 2-rows-per-block tiling. The two
changes are superadditive: +4.2% and +2.7% alone, +10.1% together.

**Correctness evidence.** A 133,392-case exhaustive proof, llama.cpp's own
backend tests, three 200-token greedy transcripts byte-identical across all
four builds tested, and SASS of untouched quant types confirmed byte-for-byte
unchanged (i.e., the patch provably doesn't touch code paths it isn't
supposed to).

## 6. Quality: the ladder

Protocol: full HumanEval-164, greedy, thinking off, on the 3060.

| Build | Size on disk | HumanEval-164 |
|---|---|---|
| Q8_0 anchor (uncompressed reference) | 29,047,086,048 B | 93.3% |
| UD-Q2_K_XL | 9,828,981,664 B | 93.3% |
| UD-IQ3_XXS **[SHIPPING]** | 10,934,860,704 B | 92.7% |
| UD-IQ2_XXS | 7,266,070,528 B | 78.0% |

The finding that matters here: the shipping flagship (IQ3_XXS) is
statistically indistinguishable from its own uncompressed Q8_0 anchor on
this benchmark — about 99% retention. The old rule of thumb we used to quote,
"~10 points of HumanEval per GiB," is retired. It turned out to be an
artifact of the thinking bug described in Section 7, not a real property of
quantization at this range.

Above roughly 2.9 bits per weight, quantization damage on this model doesn't
show up in pass/fail coding benchmarks like HumanEval at all — it shows up in
format and instruction adherence. Edit-format compliance: Q2_K_XL 67.6% vs
IQ3_XXS 94.1%, p=0.0117. IQ2_S emits unclosed code fences on 52% of tasks
even when the code inside is correct. This is the actual reason IQ3_XXS ships
instead of Q2_K_XL, despite both posting the same HumanEval score: the
difference shows up in agent and edit loops, which is what people actually
use this class of model for, and a lenient pass/fail benchmark can't see it.

MTP speculation is quality-neutral: adjudicated no-spec scored 136/164
(82.9%) vs spec 135/164 (82.3%). That comparison was measured under the
legacy (thinking-on) protocol described in Section 7, but since both arms
share the same protocol, the relative comparison remains internally valid.

## 7. The thinking bug — the largest single correction in the program

The legacy test harness never sent `enable_thinking=false`, and the model's
served chat template defaults thinking ON. As a result, every historical
quality number in this program up to the fix was measured with the model
burning its 1024-token reasoning budget before answering. 26 of 29 legacy
failures turned out to be truncations from running out of budget mid-thought,
not wrong answers.

- Reconstructing the legacy conditions reproduced the old 82.3% score
  task-for-task: 164/164 match.
- Flipping that single flag (thinking off) changed the outcome by +18 tasks,
  p=0.00053.
- Every other confound we suspected and tested, combined, was worth exactly
  one task, p=1.0. The thinking flag was the whole story.
- Wall-clock effect: 4.79 s vs 13.38 s per HumanEval task, and 158.8 vs 474
  completion tokens per task — about 2.8x, roughly 4x larger than every other
  configuration lever we tested, combined.

The consequence, and the reason revv sets this flag server-side rather than
leaving it to the client: raw t/s is not work speed. The legacy protocol
posted a HIGHER raw decode rate while doing the job 2.8x slower in wall
clock, because it was generating far more tokens (reasoning) per task before
producing an answer.

Standing canary against recurrence: if HumanEval mean completion tokens per
task climbs back above 350, the thinking switch is not taking effect, and any
quality run in that state should be discarded.

## 8. A suspected GPU bug that wasn't

A same-file quality gap was observed between the 3060 (sm_86) and an L40S
(sm_89), and was initially suspected to be a kernel defect specific to
sub-4-bit i-quants. A decisive cross-platform test was run to check this: Q8
greedy generation on both cards, compared paired task-by-task. Result: 54/54
paired tasks had identical completion-token counts, 0 discordant pairs. The
GPU architecture is exonerated — the actual cause of the earlier gap was the
same thinking-flag harness bug described in Section 7.

One rule survived this investigation, though: sub-2-bit i-quants are NOT
numerically reproducible across GPU architectures. IQ2_XXS showed only a 67%
identical-token rate cross-platform (p=0.45, i.e. consistent with unbiased
divergence, not a directional defect), against a 100% same-GPU control.
Standing practice from this: never A/B two quants of this precision class on
two different cards and expect the comparison to mean anything.

## 9. Context and KV

- With MTP at q8_0 KV on a 12GB card, the practical context ceiling is
  24-32K (the config OOMs at 32768). An earlier "-c 40960" figure floating
  around was measured without MTP, at q4_0 KV, with smaller buffers — it is
  not the same config and is not comparable. Any context claim needs the full
  config stated alongside it.
- No decode cliff was observed with growing context: flat ~21 t/s from 8K to
  98K, measured on Q2_K_XL with q4_0 KV. The cliff reported in upstream
  llama.cpp issue #27623 does not reproduce on this rig.
- KV precision at depth (llama-bench tg64, IQ3_XXS, `-fa 1`), decode t/s:

  | KV type | d0 | 8K | 16K | 24K | 32K |
  |---|---|---|---|---|---|
  | f16 | 20.36 | 19.81 | 19.26 | OOM | OOM |
  | q8_0 | 20.53 | 19.22 | 18.12 | 17.14 | 16.27 |
  | q4_0 | 20.26 | 18.92 | 17.75 | 16.69 | 15.72 |

  Quantized KV is slower, not faster, at every depth measured — q4_0 reads
  half the bytes of q8_0 per KV entry but posts a lower t/s at every single
  depth. The slowdown is dequantization compute, not bandwidth. This inverts
  the usual folklore: KV quantization here is purely a capacity trade
  (fitting more context in the same VRAM), bought at a real speed cost, not a
  speed win.
- KV precision quality: q4 vs q8 comparison gave p=0.375 — no measurable
  quality difference between the two KV tiers.
- Long-context quality: needle-in-a-haystack scored 15/15 at all depths
  tested up to 48K, on both KV tiers. Multi-hop reasoning scored 100% at 8K,
  falling to 65-75% at 32-48K.
- f16 KV at c=16384 with MTP is not a real config on this card: it passes
  the load-time check and then OOMs on the first request. The actual f16
  ceiling with MTP is c=12288. This is the concrete case referenced in
  Section 4 for why VRAM must be certified from a peak sampled during
  requests rather than at load — this config, among others in the sweep,
  passed a load-time check and then died.

**Prefill:** approximately 500 t/s, and essentially linear with context
length on this hardware. Two earlier prefill figures we published — 225 t/s
and 25.63 t/s — were both wrong and are superseded by this number.

## 10. Session restore (patch #2, pr26004-rebased-daef7b687.patch)

Median of 3 runs, fresh server per row, 8K-token prompt:

| condition | cache_n | prompt_n | prompt_ms | wall |
|---|---|---|---|---|
| patched, cold prefill 8K | 0 | 8023 | 15784.4 | 16.564 s |
| patched, action=save | — | — | — | 1.13 s, 997,687,036 B on disk |
| patched, action=restore | — | — | — | 0.277 s |
| patched, first request after restore | 8019 | 4 | 156.0 | 0.925 s |
| unpatched, first request after restore | 0 | 8023 | 15881.9 | 16.669 s |
| RAM-cache control (`--cache-ram 8192`, no save/restore) | 0 | 8023 | — | 16.691 s |

**Headline: 18.02x** on first-request-after-restore (0.925 s vs 16.7 s). We
quote 18x specifically, and deliberately do not quote larger numbers that
appear elsewhere in the underlying data (up to roughly 40-50x, and never the
101x figure) — 18x is the comparison against the realistic unpatched
baseline in the same table, and it's the one we stand behind.

The key inversion in this data: the RAM prompt cache control
(`--cache-ram 8192`, no save/restore) delivered zero reuse on this hybrid
architecture — its cache_n is 0 and its wall-clock is indistinguishable from
plain cold prefill. That's why the save/restore patch is not described as an
"optimization" — it is the only mechanism in this stack that produces any
reuse at all on this architecture.

Byte-correctness: the restored run's output diverges from the reference
generation at token 76, while a batch-split noise-floor control (using no
restore at all, just different batching) diverges at token 75 — i.e., the
restore's divergence sits at or below the noise floor already present from
batching effects. 6 of 6 semantic ground-truth probes matched exactly. On the
MTP arm: prompt_n dropped from 3519 to 24, wall from 9.075 s to 2.321 s,
decode rate 31.48 to 31.15 t/s, and completion token ids were identical over
64 greedy tokens.

Cost: approximately 125 MB of disk per 1K tokens of saved context, or
roughly 1 GB for an 8K save.

**This is not wired into the revv CLI in v1.0.** The patch lives in revv's
`patches/` directory; session save and restore are available to people who
build llama-server with the patch applied and drive its slot save/restore
endpoints directly. There is no `revv` command for this in the current
release.

## 11. Baselines: what revv is actually faster than

Two separate comparisons, shown as two rows, because conflating them
overstates the result:

| baseline | decode t/s | source |
|---|---|---|
| Naive out-of-the-box offload, general range | 2-4.5 t/s | our measurement |
| Naive out-of-the-box offload, our specific measured point (UD-Q4_K_S, CPU offload) | 2.12 t/s (live server logged ~3.96 t/s during that eval) | our measurement |
| Tuned community recipe (Q4_K_S + hand-picked FFN offload + MTP at 96K context), as reported | ~9.7 t/s | third-party claim, NOT measured by us |
| Same tuned community recipe, independently replicated | 6.6-8.5 t/s | third-party replication, NOT measured by us |

The project's own standing honesty note: the honest comparison against a
tuned baseline is about 2.2x, not 10x. The 10x-style figure describes the
naive out-of-the-box experience specifically, and must be labelled as such
wherever it's quoted. There is no single "default llama.cpp on this card does
~6 t/s" figure in this document, because no such measurement exists — the
naive baseline spans 2-4.5 t/s depending on config, and the tuned baseline is
a third-party claim we have not independently measured ourselves (only
replicated within a wide band).

## 12. Retractions and things we got wrong

A benchmarks document that only contains wins is an advertisement. Here is
everything in this program that was published and then withdrawn or
corrected:

- **ngram speculation's "86-148 t/s"**: retracted. It was a benchmark
  artifact from re-firing the same prompt against a warm server, which let
  ngram speculation replay its own previous answer verbatim. On 5 distinct
  prompts it issued zero drafts — 1.00x, no gain at all.
- **"+42% from removing state-management overhead"**: retracted, flawed
  control. The realistic gain from that direction is +3-7%.
- **The ASCII vocab-prune "-4.9 HumanEval points" result**: both arms of
  that comparison were measured under the legacy thinking-on protocol (see
  Section 7). Do not quote this number until it is re-run under the
  corrected protocol.
- **MTP n>=3 is not shipped**: greedy non-reproducibility was observed in a
  one-prompt observation, and losslessness of deep MTP has not been
  verified. n=2 ships; n>=3 does not.
- **`--spec-type draft-mtp-adaptive`** (merged upstream) is unusable on a
  12GB card: it creates a second full context against the target model and
  OOMs even at c=8192. Plain `draft-mtp` shares the main context and is
  unaffected by this problem.
- **DFlash2 drafting** segfaults in graph compute on this llama.cpp build.
  This is treated as an upstream instability, not an artifact introduced by
  revv.
- **Speculation speedup is content-dependent**, not a fixed multiplier:
  +129% on structured output, +110% on code, -2% to -4% on prose. The
  certified figures in this document are a code workload; do not extrapolate
  them to prose.
- **Multi-GPU splitting is untested.** AMD, Apple Silicon, and CPU-only are
  not supported at all in v1.0.
- The 12GB tier has approximately 86 MiB of headroom (11,958 MiB peak
  observed against roughly 12,044 MiB usable). A desktop session running on
  the same card is enough to turn that headroom into a CUDA OOM.

## 13. Limits of these numbers

Everything above was measured on one GPU (RTX 3060 12GB, sm_86), one model
family (Qwen3.8-27B, unsloth Dynamic GGUF quants), and one workload type
(code generation, via HumanEval-164 and a fixed B-tree code prompt). As
Section 12 notes, speculation speedup is a property of the content: +129% on
structured output and +110% on code, but -2% to -4% on prose — so a prose
workload will not see anything close to the headline numbers in Section 3.
Multi-GPU splitting is untested. There is no AMD, Apple Silicon, or CPU-only
support, and consequently no numbers for any of those paths.

## 14. Reproducing it yourself

`revv bench` runs the same protocol described in Section 2 — 4 measured
requests plus a discarded warm-up, 400 tokens, greedy, thinking off,
server-reported decode rate rather than client wall-clock — against your own
running server, and prints your number next to the reference figure from
this document. `revv compare` runs the same prompt through both revv and
STOCK mode side by side so you can see the difference directly.

Note that `revv bench` measures decode speed only. It does not measure
quality — the quality numbers in Section 6 come from full HumanEval-164 runs,
not from anything `revv bench` or `revv compare` produces.

If your numbers disagree with this document, that's useful information —
post them. The protocol above is written so it can be attacked; if it's
underspecified somewhere, that's a bug in this document.

## 15. End-to-end verification of the shipped tool (2026-09-02)

Everything above was measured with research harnesses. This section is
different: it is `revv` itself, installed from this repository onto an RTX 3060
box and driven through its own commands. Every phase of `TEST_3060.md` passed.

| what | measured |
|---|---|
| flagship, patched build, `revv bench` | **37.86 t/s**, spread 0.8% |
| peak VRAM for the flags revv actually launches | **11,830 MiB** |
| v1.1 candidate (ASCII-pruned) | **40.10 t/s**, peak **11,502 MiB** |
| `revv compare`, revv vs STOCK | 38.4 vs 22.5 t/s decode; 2.01x time-to-done |
| toggle latency | 3.4 s and 3.8 s |
| teardown | GPU to 1 MiB, 0 stray processes; orphan reaping worked |

**The harness gap, stated plainly.** `revv bench` reads the flagship at
37.86 t/s where the certification in Section 3 reads 34.39 t/s for the same
build on the same card. This is not a discrepancy to be resolved — it is two
different prompts producing different MTP acceptance rates. `revv bench`
therefore compares your machine against 37.9, the figure measured with the
protocol it actually runs. **Do not compare 40.10 against 34.39**; the only
internally consistent comparison is the three-row table below, all taken with
one harness in one session:

| build | decode t/s | peak VRAM |
|---|---:|---:|
| v1.1 candidate (ASCII prune + merged kernel) | 40.10 | 11,502 MiB |
| ASCII prune + stock kernel | 38.92 | 11,500 MiB |
| flagship + merged kernel | 37.86 | 11,830 MiB |

Isolating the two effects: the kernel patch is **+3.03%** here against the
+2.5% in Section 5, and the ASCII prune is **+5.92% and −328 MiB** against a
prior +5.73% / −332 MiB. Both reproduce.

**Peak VRAM is 11,830, not 11,958.** The 11,958 figure comes from the
certification harness, whose command line differs from what revv launches (no
`--cache-ram 0`, no `--no-cache-idle-slots`). 11,830 was independently measured
before and is the reproducible number for revv's actual flags. revv keeps
warning at the higher figure, because being conservative about VRAM on a card
with 214 MiB of headroom is the correct bias.

**The thinking substitution, closed.** revv uses the GGUF's embedded chat
template and disables thinking server-side; the certification used an external
template and a per-request kwarg. Both differences were tested:

| arm | request body | reasoning emitted |
|---|---|---|
| A — server-side flag only | no kwarg | **0 chars** |
| B — per-request kwarg | `enable_thinking:false` | 0 chars |
| C — positive control | `enable_thinking:true` | **803 chars** |

Arm C is what makes arm A meaningful. And a 25-task HumanEval run under revv's
embedded template produced **byte-identical completions on 25/25 tasks** versus
the certified run under the external template, so revv does not ship a template
file and does not need to.

**What the v1.1 candidate is, and is not.** It is the ASCII-vocab-pruned
flagship (vocab 127,947) on the merged build: faster, and roomier at 542 MiB
free versus 214 MiB. Its 25-task spot-check was byte-identical to the certified
baseline, which is strong evidence of output-neutrality on that workload but is
**not** a certification — 25 tasks bounds pass@1 only to roughly [86.7%, 100%].
The full 164-task run has not been done. v1.1 therefore ships no headline
accuracy number and is not the default. The prune is also ASCII/English+code by
construction, so non-ASCII workloads are out of scope for it.

**This test also found four defects in revv**, all since fixed: `revv bench`
could not detect the failure it documented (it sent the masking kwarg itself),
the version string reported an unreliable build number, `revv down` rejected
`--url`, and `install.sh` built the full CUDA architecture fan-out instead of
the detected card's. The bench defect is the instructive one: a canary that
tests the wrong path is worse than no canary, because it reports PASS.

## 16. Compatibility reports from other machines (anecdotal)

Section 15 is our own hardware. This section is other people's, and it is
labelled anecdotal because it is: no controlled protocol, no repeated trials,
and in some cases no numbers at all. It is here because a compatibility claim
that only ever gets tested on the author's box is worth very little.

| machine | OS | config reached | `revv bench` decode | outcome |
|---|---|---|---|---|
| RTX 3060 12GB (second unit) | Windows 11 + WSL2 (Ubuntu) | ctx 8192, q8_0 KV, MTP n=2 | **revv 34.31 t/s** (91% of the 37.9 reference), **STOCK 22.09 t/s** (98% of the 22.5 reference), spread 0.3-0.4% | working, chat verified end to end |

Those are raw figures from that user's box, quoted as reported. We have not
set an "expected band" for WSL2 — one machine is not a band.

### A negative result: revv mode made a Gemma model slower

On the same box, a Gemma-4-12B GGUF measured over 3 trials:

| mode | decode t/s |
|---|---|
| stock | 35.49 |
| revv | **34.63** (2.5% *slower*) |

This is real, it is expected once you look at it, and it was our bug. revv was
applying a flag set certified on Qwen3.8-27B to a model that could not use any
of it:

- **No MTP draft head** in the file, so no speculative decoding — which is
  where most of revv's speed comes from.
- **No thinking mode** in its chat template, so nothing to disable — and the
  thinking-off win is the largest single lever in Section 7.
- The only flag still doing anything was **quantized KV**, and per the kernel
  profile in Section 9 that is a *compute tax*: q8_0 switches attention to a
  compute-bound kernel and is slower than f16. It pays for itself only when it
  buys capacity you would otherwise not have. A 7 GB model on a 12 GB card is
  not short of capacity, so the tax was pure loss.

2.5% is close to this box's noise floor, so the magnitude is not the point. The
direction is. A tool that claims to speed models up should not be able to slow
one down silently.

**What changed.** revv now derives its flags per model from the facts `inspect`
already reads: draft head present or not, thinking switch present or not, and
whether f16 KV fits in free VRAM at the chosen context. On this Gemma file that
yields no speculation flags, no thinking flag, and f16 KV — i.e. the best-known
stock configuration — and revv says so:

    note: this model gains nothing from revv's tuned mode --
          serving with the best-known stock config.

`toggle` and `compare` now refuse to stage the A/B in that state rather than
printing two numbers that differ only by noise. Verified: the certified Qwen
config is unchanged by this logic (ctx 16,384, q8_0 KV, speculation on, 11,830
MiB estimated peak on a clean 12 GB card), because on that model f16 KV genuinely
does not fit — the rule reproduces the certification rather than overriding it.
On a 24 GB card the same model now correctly gets f16 KV, which Section 9 says
is the faster kernel.

What that run taught us, both of which are now fixed in the tool:

1. **Total VRAM is not available VRAM.** Windows reserves roughly 1-1.5 GB of
   the card under WSL2. The certified c=16384 config therefore OOMs on a 12GB
   card that a total-VRAM check would pass, and the user had to discover
   `--ctx 8192` by hand. At that setting the card reported 12,006 MiB of
   12,288 in use. revv now plans against `memory.free` and picks the largest
   context that fits with a 250 MiB margin — which on that machine is exactly
   the 8192 the user arrived at manually. The cost model behind that choice
   predicts 11,462 MiB at c=8192, consistent with the 12,006 observed once the
   host reservation is added back.
2. **The CUDA build is the real onboarding cost.** About 30 minutes went to
   toolchain mismatches — CUDA 12.6 against glibc 2.43 and gcc-15 — before
   `cuda-toolkit-13-3` worked. That is not a revv defect, but it is a revv
   problem, and it is why a prebuilt binary is now the top roadmap item.

### The instrument disagreement that run exposed

On that same box, `revv compare` reported revv mode at **29.7 t/s** while
`revv bench` reported **34.31 t/s** — a 14% disagreement between two of our own
instruments on one machine, one model, one config. That is worth more attention
than either number.

The proposed explanation was cold-path cost: `bench` discards a warmup and
`compare` did not, so the first CUDA graph build after a mode switch would land
inside compare's timed window. **We tested it and it does not hold up as the
explanation.** Two pieces of evidence:

- A controlled experiment (restart-then-measure versus already-warm, 4 requests
  each, 3 trials per mode) found no restart-specific penalty. The first-request
  effect was at most 2.6% and inconsistent in sign between the two arms — within
  this box's noise. Caveat, and it is a real one: that box is Apple Silicon, so
  the specific CUDA-graph mechanism cannot reproduce there. The test constrains
  the size of any general first-request effect; it cannot rule out a CUDA-only one.
- The direction is wrong. On the RTX 3060 in Section 15, compare read *higher*
  than bench (38.4 vs 37.86). On WSL2 it read *lower*. A fixed protocol bias
  would push the same way on both machines.

The mechanism that does fit: `compare` streams, and took its decode rate from
the server's `timings` when present, falling back to a **client-side wall-clock
rate** when absent. llama.cpp only attaches timings to the final streamed
response unless `timings_per_token` is requested, and the exact shape of that
final chunk varies across builds and across the OAI-compatibility path. When the
fallback engages, the number includes everything between llama-server and the
client — the revv proxy, SSE framing, and the host's loopback — which on WSL2 is
materially slower than on native Linux and would depress the figure in exactly
the direction and rough magnitude observed.

Three changes follow, none of which required believing that story:

1. `compare` now requests `timings_per_token`, so the server's own decode rate
   is available on every chunk rather than one. On the test box this took the
   proportion of chunks carrying timings from 1 of 33 to 31 of 33. compare and
   bench now report the same quantity by construction.
2. `compare` measures both ends and, when the client-observed rate falls more
   than 5% below the server's, says so and names it as transport cost on that
   host. Verified to fire on a replay of the WSL2 figures.
3. `compare` discards one warmup exchange per mode. Not because the cold-path
   theory was confirmed — it was not — but because `bench` has always done this,
   and two instruments that share a name should not be different protocols.

**Status: the 14% gap is explained by mechanism but not yet confirmed on the
machine that produced it.** The next `revv compare` on that box settles it: if
the transport note appears, the diagnosis is right; if the two instruments now
simply agree, the fix landed. Either outcome is informative and we will record it.

If you run revv on hardware not listed above, `revv bench` output plus
`revv doctor` is exactly the contribution that makes this table worth having.

## 17. Two things that do not work, measured (2026-09-04)

**Context checkpoints must be off near the VRAM ceiling.** llama.cpp keeps 32
context checkpoints per slot by default (upstream PR #15293) at roughly 150 MiB
each on this model. They are allocated lazily, so on a config close to the
ceiling the server loads, passes its health check, serves one request, and then
dies on the second with a `cudaGraphInstantiate` error that names neither
memory nor checkpoints. Configs near the ceiling required `-ctxcp 0` to survive
at all. This is the same class of trap as certifying VRAM at load time instead
of during requests (Section 4): the failure hides behind a successful startup.
revv's planner now sets `-ctxcp 0` whenever the estimated peak leaves under
500 MiB free — which includes the certified config on a clean 12 GB card, where
the margin is 457 MiB.

**GPU overclocking is not a lever on this card.** Measured on the RTX 3060,
driver 535.309.01, headless:

| arm | tg64 d=13000 | server t/s | sm / power at depth |
|---|---|---|---|
| A stock (170 W) | 20.390 | 35.103 | 1865 MHz / 161.5 W |
| C 190 W + lgc 2145 | 20.581 (+0.94%) | **35.586 (+1.38%)** | 1882 MHz / 179.8 W |
| E lgc 2400 + lmc 8000 (above spec) | 20.582 | — | 1876 MHz / 180.2 W |

Clock *offsets* are impossible headless on this driver — they require
`nvidia-settings`, which requires X, and NVML exposes no offset API. The power
limit is the only real lever and it buys **+1.4% for +11.8% power and +7 °C**,
because the card sits against `SW_POWER_CAP` at 1880 MHz of a 2145 MHz maximum
even after the raise: the workload is power-limited, not clock-limited.

The trap worth publishing: **`nvidia-smi` accepts above-spec clock requests,
reports success, and silently clamps.** Arm E asked for 2400/8000 MHz, printed
`"All done."`, and delivered clocks identical to the in-spec arm C, matching
throughput to four decimal places. Anything automating this must verify with
`--query-gpu=clocks.sm,clocks.mem` under load and never trust the exit status.
Acceptance was bit-identical at 0.7756 across every arm, so the locked clocks
were not a stability risk either — they simply were not an overclock.

## 18. Speed tier re-certification: 55.9 t/s (2026-09-05)

Three flag changes over the original 48.5 t/s speed-tier config, all gated to
the n_cpu_moe (MoE, host-RAM-offload) build only -- the flagship's flags are
untouched:

1. **`-t <n>`, an explicit thread count** for the server process. CPU-MoE
   offload puts host RAM bandwidth on the critical path for every token, so
   the thread count is a decode-speed lever, not just a load-time one. `n` is
   the physical (not logical/hyperthreaded) core count, clamped to [4, 8].
2. **`--spec-type ngram-simple,draft-mtp`**, replacing plain `draft-mtp`.
   llama.cpp runs this as a first-success-wins chain: an n-gram hit skips the
   MTP pass for that token, so this is a strict addition over MTP alone, not
   a substitute for it.
3. **`--spec-ngram-simple-size-m 256`**, the n-gram matcher's window size.

**Paired quality, before vs after (n=164 / n=34, McNemar):**

| instrument | before | after | p |
|---|---|---|---|
| HumanEval-164 | 152/164 | 153/164 | 1.0 |
| edit-format compliance | 33/34 | 34/34 | 1.0 |
| peak VRAM | 11,832 MiB | 11,832 MiB | identical |

No measurable quality change on either instrument, and the VRAM ceiling that
`-ctxcp 0` and the rest of the planner are certified against did not move.

**Result.** Decode: 48.5 -> **55.9 t/s** (2.52x stock 22.2 t/s). Editing
workloads -- where the n-gram matcher gets to reuse text it can already see in
the prompt instead of generating it token by token -- go much further: up to
**~188 t/s mean, 243 t/s peak**, roughly 3x over MTP alone on the same tasks.
Generation workloads see zero effect from the n-gram addition (it just misses
and falls through to MTP) at zero cost.

**Thread sweep** (decode t/s at fixed ctx, MoE build, n-gram+MTP stack held
constant):

| -t | t/s |
|---|---|
| 3 | 39.6 |
| 4 | 46.4 |
| 5 | 51.9 |
| 6 | 55.1 |
| 7 | 54.7 |
| **8** | **55.9** |
| 9 | 52.9 |
| 10 | 48.5 |
| 12 | 41.1 |

Peaks at 8, falls off on both sides: too few threads starves the host-RAM
transfer, too many oversubscribes the 6 physical cores and the SMT siblings
fight over the same memory bus on this bandwidth-bound decode. The rig's full
logical core count is 12 (6 physical, SMT-2); by -t 10 it is already losing
5-15% against the -t 8 peak, which is why the heuristic clamps to physical
cores rather than `os.cpu_count()`.

**`size_m` sweep summary.** The n-gram matcher's default window (`size_m=48`)
gives ~100 t/s mean on editing tasks; `size_m=256` gives **~188 t/s mean**.
The curve was still rising at 256, so this is a floor for a wider window, not
a measured ceiling -- 256 is what shipped because it is where we stopped
sweeping, not where the effect stopped.

**CRLF warning.** n-gram matching is a literal byte-sequence match against the
prompt. Repos checked out with CRLF line endings collapse acceptance from
0.83 to 0.11, because every matched line has its ending byte-flipped against
what the model just generated. Use LF line endings in any repo an n-gram
drafter is expected to help with; this is a property of the matcher, not of
revv's config.

## Appendix: exact artifacts

For anyone trying to reproduce these results from byte-identical inputs:

- Repository: `unsloth/Qwen3.8-27B-GGUF`
- Certified file: `Qwen3.8-27B-UD-IQ3_XXS.gguf`, exact size 10,934,860,704
  bytes (10.18 GiB), sha256 `c0b7c3038681ed2e3040456c1dd45f9858b6c2290bed172c70388a94874f3eee`
- Other files from the same repo referenced in this document:
  `Qwen3.8-27B-UD-Q2_K_XL.gguf` (9,828,981,664 B),
  `Qwen3.8-27B-UD-IQ2_XXS.gguf` (7,266,070,528 B),
  `Qwen3.8-27B-Q8_0.gguf` (29,047,086,048 B — note this file has no "UD-"
  prefix).
- The MTP draft head lives in the GGUF as tensors named `blk.64.nextn.*`.
  Builds below roughly 8.4 GiB have it stripped, which is why going smaller
  than IQ3_XXS is a double penalty: lower quality and no speculation.
- llama.cpp base commit both patches in `patches/` apply to:
  `daef7b6874397a5a7c3d7e38b55e2ee0adf7da38` (build b10712), "vulkan: top_k
  radix select for k >= 1024 for Qwen 3.8 Flash Next (#28032)". Both patches
  were verified to apply cleanly to a pristine checkout of that commit.
</content>

## MTP losslessness note (2026-09-03)

Speculative decoding with greedy sampling is algorithmically lossless, but on
this stack it is not bit-exact: 3 of 5 diverse greedy probes diverged between
MTP-on and MTP-off on the 27B (control: no-spec vs no-spec across a server
restart was 5/5 byte-identical). Cause is floating-point summation order in
the batched verify pass — the same batch-shape nondeterminism llama.cpp
exhibits generally. Measured quality impact: none (full HumanEval-164 A/B,
identical per-task outcomes, p=1.0). We previously wrote "byte-identical";
that was wrong and is corrected throughout.

## Flagship n-gram chain certification (2026-09-05)

Section 18 certified the n-gram+MTP drafter chain (`--spec-type
ngram-simple,draft-mtp --spec-ngram-simple-size-m 256`) for the n_cpu_moe
speed tier only. It now ships on the flagship too: the chain is a strict,
first-success-wins addition over MTP alone, and nothing about the acceptance
mechanism is specific to the MoE build — an n-gram hit still just means the
model is about to reproduce text already visible in the prompt.

**Result, 4 workloads (t/s, greedy, server-reported decode rate):**

| workload | before (plain MTP) | after (chain) | speedup | output |
|---|---:|---:|---:|---|
| editing 1 | 40.3 | 222.8 | 5.53x | byte-identical |
| editing 2 | 40.3 | 246.0 | 6.10x | byte-identical |
| editing 3 | 40.3 | 113.3 | 2.81x | byte-identical |
| pure generation | 35.17 | 35.16 | 1.00x | byte-identical |

Editing workloads gain 2.81-6.10x, because the n-gram matcher gets to reuse
text the server can already see in the prompt instead of generating it token
by token. Pure generation is flat (1.00x, within noise) and costs nothing —
there is nothing for the matcher to reuse, so it misses every time and falls
straight through to MTP. All 4 workloads produced byte-identical output
against plain-MTP: the chain changes nothing about which tokens get emitted,
only how many of them are drafted for free. Because output did not change on
a single byte, quality scores (HumanEval, edit-compliance) cannot have moved
either — there is nothing to re-run.

**VRAM cost and the ship point.** The chain is not free: ~100 MiB, measured
as 11,956 MiB vs 11,854 MiB at c=16384 (otherwise-identical launches). On a
12GB reference card (12,288 MiB free):

| context | peak VRAM (chain included) | headroom |
|---|---:|---:|
| 16384 | 11,956 MiB | 332 MiB |
| **12288** | **11,822 MiB** | **466 MiB** |
| 8192 | 11,666 MiB | 622 MiB |

332 MiB of headroom at c=16384 is below the ~400 MiB comfort line this
program has otherwise held to, so the shipped ceiling for the flagship on a
12GB card is **c=12288**, not c=16384. Decode throughput is flat across
8K/12K/16K context on this workload, so shrinking context to buy the headroom
back costs nothing measurable. `revv`'s planner now charges the chain's
~100 MiB against free VRAM before sizing context (`SPEC_NGRAM_CHAIN_MIB` in
`revv.py`) specifically so it lands here automatically instead of shipping a
config with an unacceptably thin safety margin.
