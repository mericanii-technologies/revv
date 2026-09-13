# Google Gemma 4 26B-A4B-it, IQ3_XXS — certification on the RTX 3060 12GB

Run 2026-09-13 16:55–17:40 UTC on the reference box, as serial-queue job
`08_gemma4_26b`, rc=0, 0.75 h wall clock. Same protocol as the two certified
Qwen builds (`BENCHMARKS.md` §2 speed, §6 quality, §18 MoE re-certification).

Raw log: `/data/scratch/queue/logs/08_gemma4_26b.log` on the box.
Machine-readable: `/data/scratch/gemma4/results.json`.
Per-task records: `/data/scratch/gemma4/he_revv.jsonl`, `he_stock.jsonl`,
`poly/edit_gemma4_26b.jsonl`.

**Headline: it is the fastest thing we have measured on this card — 70.65 t/s at
16K context, +26.4% over the certified Qwen MoE — and it scores 157/164 on
HumanEval. It is also, on our editing instrument, clearly worse than the Qwen
MoE. Both halves of `BENCHMARKS.md` §16's negative result on Gemma 4 are now
formally refuted.**

---

## Summary table

| field | value |
|---|---|
| file | `google_gemma-4-26B-A4B-it-IQ3_XXS.gguf` — 12,160,200,320 B / 11.33 GiB (verified byte-exact against the HF API) |
| sidecar draft head | `mtp-google_gemma-4-26B-A4B-it-Q4_0.gguf` — 321,145,344 B / 0.30 GiB |
| repo | `bartowski/google_gemma-4-26B-A4B-it-GGUF` (both files) |
| licence | Apache-2.0 (`general.license` in the GGUF itself; link `ai.google.dev/gemma/docs/gemma_4_license`). Gemma 4 Terms of Use / Prohibited Use Policy referenced separately |
| architecture | `gemma4`, 30 layers, 128 experts / 8 active, `expert_feed_forward_length` 704, sliding window 1024 on 25 of 30 layers, train ctx 262,144 |
| **context served** | **16,384** (ladder: 8192, 12288 and 16384 all fit at the *same* placement) |
| expert blocks on CPU | **`--n-cpu-moe 4` of 30** — 87% of the model stays resident |
| **peak VRAM (revv-style)** | **11,580 MiB** of 12,044 usable |
| **headroom** | **464 MiB** (standard is ≥ 200) |
| peak VRAM (stock) | 11,928 MiB → **116 MiB headroom, fails the standard** |
| **stock decode** | **62.03 t/s** (spread 4.99%) |
| **revv-style decode** | **70.65 t/s** (spread 0.24%, min 70.57 / max 70.75) |
| revv-style decode @ c=8192 | 70.38 t/s (context is nearly free on this model) |
| **editing t/s (revv-style)** | **287.00 t/s** (3 × ~1,500-token file re-emit) |
| editing t/s (stock) | 59.21 t/s → the n-gram chain is worth **4.85×** on edit work |
| decode t/s on real HumanEval work | 112.8 t/s revv-style vs 62.7 t/s stock — **1.80×** |
| draft acceptance | 0.669 on the fixed generation prompt; 0.678 across all 164 HumanEval tasks (32,040 / 47,234) |
| **HumanEval-164, revv-style** | **157/164 (95.7%)** |
| **HumanEval-164, stock** | **157/164 (95.7%)** |
| **editing instrument (polyglot, 34 Python)** | **1/34 first-attempt, 5/34 overall, 27/34 edit-format** — see the truncation caveat below |
| host RAM | 47 GB box, ~38 GB available; only 4/30 blocks offloaded, so host-RAM pressure is far below the Qwen MoE's ~8 GiB |
| reference: Qwen3.6-35B-A3B MoE | 55.9 t/s, 153/164, 9/34 first-attempt (16/34 overall), 34/34 format, 11,832 MiB |
| reference: Qwen3.8-27B dense | 37.9 t/s, 152/164, 4/34 first-attempt (8/34 overall), 34/34 format |

---

## The argv that worked

Shipped (revv-style) configuration, 16K context:

```
q-srv -m google_gemma-4-26B-A4B-it-IQ3_XXS.gguf \
  -ngl 99 -c 16384 -fa on -ctk q8_0 -ctv q8_0 -ctxcp 0 -t 8 \
  --cache-ram 0 --no-cache-idle-slots \
  --host 127.0.0.1 --port 18098 --no-warmup --jinja --parallel 1 \
  --reasoning off --n-cpu-moe 4 \
  --spec-type ngram-simple,draft-mtp --spec-draft-n-max 2 \
  --spec-ngram-simple-size-m 256 \
  -md mtp-google_gemma-4-26B-A4B-it-Q4_0.gguf
```

Stock baseline (for the A/B only — it does not meet the headroom standard):

```
q-srv -m google_gemma-4-26B-A4B-it-IQ3_XXS.gguf \
  -ngl 99 -c 8192 -fa on --host 127.0.0.1 --port 18098 \
  --no-warmup --jinja --parallel 1 --n-cpu-moe 2
```

Client, both arms: `temperature=0, top_k=1, seed=1234, cache_prompt=false,
chat_template_kwargs={"enable_thinking": false}`.

**Resolved flag questions.** The sidecar flag in the certified v11 build is
`--spec-draft-model` / `-md` / `--model-draft`; there is no bare `--draft`. The
full chain `--spec-type ngram-simple,draft-mtp` loaded and drafted on the very
first attempt (dry run: 43.7 t/s, draft 90/81 at `--n-cpu-moe 16`), so no
fallback to plain `draft-mtp` or to no-speculation was ever needed. Binary:
build 1 / commit `daef7b687`, which already carries `gemma4` and
`gemma4_assistant` graph classes.

---

## What `revv inspect` said

**Main file** — `architecture gemma4`, 30 layers, 658 tensors, vocab 262,144,
`MTP draft head: absent`, verdict `COMPATIBLE (no draft head — revv's levers
may not apply)`. Its second paragraph does correctly point at
`revv serve --draft <file.gguf>` and names the Gemma 4 family on HF.

**Sidecar** — `architecture gemma4-assistant`, 4 layers, 49 tensors,
`MTP draft head: present — 2 tensors` (`nextn.pre_projection.weight`,
`nextn.post_projection.weight`), verdict `COMPATIBLE (has a draft head —
speculation will run, no numbers for this model)`.

**Thinking switch** — read directly out of the GGUF's `tokenizer.chat_template`
(18,681 chars, 390 lines), no server needed:

```
line 186: {%- set enable_thinking = enable_thinking | default(false) -%}
line 193: {%- if enable_thinking -%}
line 194: {{- '<|think|>\n' -}}
line 384: {%- if not enable_thinking -%}
```

Exactly as the desk research predicted: the switch exists and **defaults off**.
No `<think` tag appeared in any sampled output.

---

## Findings

**1. `BENCHMARKS.md` §16 is refuted on both counts.** That section concluded a
Gemma 4 GGUF has no draft head and no thinking mode, and that revv could
therefore only slow it down. Measured here: the draft head exists as a
first-party sidecar and drafts at 0.67–0.68 acceptance; the thinking switch
exists and defaults off; and the revv-style configuration is **+13.9% faster
than stock on generation and 4.85× faster on editing**, while simultaneously
serving 2× the context with 4× the headroom. The 34.63-vs-35.49 t/s measurement
that produced §16 stands as a measurement of a different, headless, dense
Gemma-4-12B build; its generalisation does not.

**2. It stayed resident, which is the whole ballgame.** `MODELS_12GB.md` §4.1
predicted Class A, 55–85 t/s, with the caveat that Gemma 4's active FFN
footprint is 1.73× the Qwen MoE's and "if it ends up heavily offloaded that band
collapses. Keep it resident." It needed only **4 of 30** expert blocks on the
host link, against the Qwen MoE's 16 of 41, and landed at 70.65 t/s — inside the
predicted band and above its midpoint. The prediction and its caveat were both
correct.

**3. Context is nearly free on this model.** 8,192 / 12,288 / 16,384 all fit at
the *same* `--n-cpu-moe 4`, costing 124 MiB of peak VRAM in total across that
whole range (11,350 → 11,474 MiB), and 16K decodes *no slower* than 8K (70.65 vs
70.38 t/s, inside the ±1% noise floor). That is sliding-window attention on 25
of 30 layers doing exactly what §4.1 said it would. There is likely a lot more
context available here than we probed for.

**4. Speculation is quality-neutral, again.** Stock and revv-style both scored
157/164 on the same 164 problems, sharing 5 of 7 failures and differing on 2
each (HumanEval/108 and /129 revv-only; /130 and /132 stock-only). Two
discordant pairs each way is McNemar p = 1.0. This reproduces the MTP
losslessness note on a second architecture and a second drafter topology
(external sidecar rather than embedded head).

**5. The editing score is real but the instrument was mis-calibrated for this
model, and the number as printed is a floor.** All 7 `no_valid_python_block`
failures are truncation: every one of them ran to `completion_tokens` of 4,096
(= 2 attempts × the 2,048-token cap) or close to it (2,964 / 3,388 / 3,599),
i.e. the model was still writing when the budget ran out and never closed its
fence. The 2,048-token budget was calibrated on the 27B, which averages 887
tokens per task (the Qwen MoE, the real comparator, averages 1,296); Gemma 4
averages **2,233**. So `27/34` is an artifact of the
cap, not a format-compliance defect, and `5/34 overall` is a lower bound — 7 of
34 tasks were never given a chance to finish. **Re-running this instrument at
`--max-tokens 4096` is the single highest-value follow-up, roughly 30 minutes.**

That said, do not read the caveat as rescuing the result. On the 27 tasks that
*did* produce well-formed edits, Gemma 4 passed 5; the Qwen MoE passed 16 of 34
at 34/34 format compliance. (The follow-up below settles this: it reaches 7.) Even crediting Gemma every truncated task it could
plausibly have won, it does not obviously reach 16. And its failure mode on the
tasks it did complete is telling: `affine-cipher` failed on an exact
error-message-string assertion (`"Key 'a' must be coprime to 26."` vs the
required `"a and m must be coprime."`), which is spec adherence, not coding.

**6. Stock does not meet our VRAM standard.** At c=8192 the smallest placement
that loads at all is `--n-cpu-moe 2`, and it peaks at 11,928 MiB — **116 MiB of
headroom, below the 200 MiB standard**. `--n-cpu-moe 0` OOMs outright. So the
"stock" column here is a measurement, not a shippable configuration: the only
configuration of this model that is certifiable on this card is the revv-style
one. Note also that revv-style wins while carrying *more* offload (4 blocks vs
2) — the levers more than pay for the extra host-link traffic.

**7. A docs nit.** `revv inspect` on the main file leads with
`COMPATIBLE (no draft head — revv's levers may not apply)`. For the Gemma 4
family specifically that headline is now known to be too pessimistic, since
Google ships a first-party sidecar for every size. The body text already points
at `--draft`; the headline should too.

---

## Verdict against the Qwen MoE

Gemma 4 26B-A4B is **faster and cheaper to serve than the certified Qwen MoE and
statistically indistinguishable from it on HumanEval, but it loses the editing
instrument, which is the one that has historically predicted agent-loop
usefulness.** It decodes 70.65 t/s against 55.9 (+26.4%), serves the same 16K
context at 11,580 MiB against 11,832, from a file 4.7 GiB smaller, with 4 of 30
expert blocks offloaded instead of 16 of 41 — so it needs far less host RAM and
far less of the 25.7 GB/s link, and it has 464 MiB of headroom to spend instead
of 212. On HumanEval it scores 157/164 against 153/164, a 4-task difference at
n=164 that is not significant and that we should not market. On the 34-task
editing instrument it scores 5/34 overall against 16/34 — and although that
number is confounded by a token cap that truncated 7 tasks, the confound does
not plausibly close a gap that large, and the failures it *did* complete look
like instruction-adherence misses rather than reasoning misses. The honest
reading is the one `MODELS_12GB.md` §4 warned about in advance: **none of these
models was being measured on the expectation of a quality win, and this one did
not deliver one.** Its case is speed, footprint and licence — 26% more decode,
4.7 GiB less disk, and clean Apache-2.0 — and it is a strong case for a
speed-tier line. It is not yet a case for displacing the Qwen MoE as the build
we point agent work at. Re-run the editing instrument at `--max-tokens 4096`
before making that call final.

---

# Follow-up, 2026-09-13 17:45–18:12 UTC: the editing instrument at `--max-tokens 4096`

Serial-queue job `09_gemma4_editing_4096`, rc=0, 27 min wall clock. Log:
`/data/scratch/queue/logs/09_gemma4_editing_4096.log`. Data:
`/data/scratch/gemma4/poly4096/`.

**Purpose.** Finding 5 above argued that the 2,048-token cap truncated 7 of 34
tasks and that `5/34` was therefore a floor. This run doubles the cap to test
that claim. Everything else is held fixed: same 34 tasks, same seed 1337
sample, same two-attempt protocol, `--attempts 2 --concurrency 2
--request-timeout 1800`, same revv-style server configuration.

**Two deliberate deviations, both stated up front.** (a) The server ran at
`-c 16384` rather than the first run's `-c 8192`, because attempt 2 appends the
full attempt-1 reply to the conversation — at `max_tokens=4096` the second
prompt reaches ~6.3K tokens and `c=8192` would have overflowed. 16,384 is the
shipped configuration anyway and decodes within noise of 8,192 (70.65 vs 70.38
t/s), so this is not a speed or capability confound. (b) The instrument ran from
a byte-identical copy, `polyglot_edit_tok.py`, differing from
`cert35b_harness/polyglot_edit.py` only by four added lines that record
per-attempt `completion_tokens` and `finish_reason`. The protocol itself is
untouched.

## Result

| metric | max_tokens 2048 | **max_tokens 4096** | Qwen3.6-35B-A3B MoE |
|---|---:|---:|---:|
| first-attempt solved | 1/34 | **1/34** | **9/34** |
| solved overall | 5/34 | **7/34** | **16/34** |
| edit-format compliance | 27/34 | **30/34** | **34/34** |
| mean completion tokens / task | 2,233 | 3,426 | 1,296 |
| wall time | 1,096 s | 1,629 s | — |
| peak VRAM | 11,456 MiB (c=8192) | 11,600 MiB (c=16384) | 11,832 MiB |

**Does anything still hit the cap? Yes, and this is the finding.** 19 of 66
attempts (29%) ended with `finish_reason=length` at 4,096 tokens, spread across
15 of 34 tasks. The per-attempt token distribution is sharply bimodal:

```
n=66  min=151  p25=487  median=877  p75=4096  p90=4096  max=4096  mean=1765
finish_reason: stop 47, length 19
```

A median attempt costs 877 tokens — *less* than the Qwen MoE's 1,296 average.
Gemma 4 is not a verbose model. It has a runaway mode that it enters on roughly
a third of attempts and never exits, and doubling the budget simply let those
attempts burn twice as many tokens. A third doubling would not fix this either.

## Paired comparison against the certified Qwen MoE

Same 34 task ids, the Qwen MoE arm being its shipped chain configuration
(`results/cpu_wave1/edit_Q3_K_XL_ncm16_t8_ngrammtp_m256_edit34_3060`), McNemar
exact test on discordant pairs:

| metric | Gemma 4 | Qwen MoE | Qwen-only wins | Gemma-only wins | p |
|---|---:|---:|---:|---:|---:|
| **overall** | **7/34** | **16/34** | 10 | 1 | **0.0117** |
| **first-attempt** | **1/34** | **9/34** | 8 | 0 | **0.0078** |

Gemma solves exactly one task the Qwen MoE misses (`dominoes`) and misses ten it
solves (`affine-cipher`, `beer-song`, `bowling`, `grade-school`, `hangman`,
`poker`, `proverb`, `simple-linked-list`, `two-bucket`,
`variable-length-quantity`). On first attempt it wins nothing at all.

Raising the cap moved Gemma strictly forward and cost it nothing — `dominoes`
and `transpose` newly solved, none lost, and three tasks (`connect`,
`scale-generator`, `simple-linked-list`) had their format failure repaired.
Four tasks still fail format at 4,096 (`food-chain`, `sgf-parsing`, `transpose`,
`zebra-puzzle`), all of them runaways.

And the decisive diagnostic: **truncation is not what separates solved from
unsolved.** Of the 15 tasks with at least one capped attempt, 4 were solved
(27%); of the 19 with no capped attempt at all, 3 were solved (16%). The tasks
that ran clean inside the budget did no better than the ones that ran away.

## Reading: the gap holds

**It narrows by two tasks and then holds, and the holding is now statistically
significant on paired data rather than a comparison of two totals.** Doubling
the token budget was the right control to run and it did what Finding 5
predicted at the margin — overall went 5/34 → 7/34, format compliance 27/34 →
30/34, nothing regressed — so the original `5/34` was indeed a floor and the
`27/34` compliance figure was indeed partly an artifact. But the floor was low,
not the ceiling: against the Qwen MoE's 16/34 and 9/34 the gap is 7-to-16
overall (McNemar p = 0.0117) and 1-to-9 on first attempt (p = 0.0078), with
Gemma winning a single task in the entire 34 and winning none on first attempt.
That is a broad capability gap on this instrument, not a trade of strengths, and
it is the same shape of result the 27B posted against the 35B in
`flagship_polyglot`. The token-cap story also inverts on inspection: Gemma's
median attempt is *shorter* than Qwen's average, so this was never a verbosity
problem — it is a failure to terminate on about a third of attempts, which is an
instruction-adherence defect of the same family as the `affine-cipher`
exact-error-string miss noted in job 08. **The verdict from job 08 stands
unchanged and is now better evidenced: Gemma 4 26B-A4B is the faster, smaller,
cheaper-to-serve model and is tied on HumanEval, but it is not the model to
point agent or edit-loop work at.** Its case is the speed tier. No further
box hours on the editing instrument are warranted; if anything else gets
measured here it should be why a third of attempts fail to emit a stop token,
since that is a serving-level defect that might be fixable in the sampler or
the template rather than a property of the weights.
