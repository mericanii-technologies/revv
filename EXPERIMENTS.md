# What we tried, what failed, and why

The negative-results record for the program behind revv: running a 27B-class
coding model well on a 12GB consumer GPU, 2026-08-27 to 2026-09-05.

Most of this work did not ship. It is written down because the failures were
more informative than the wins, and because anyone deciding whether to spend
their own time on one of these ideas deserves the measurement rather than the
intuition. Tables and protocols for what did ship are in
[BENCHMARKS.md](BENCHMARKS.md).

Rig unless stated: RTX 3060 12GB (sm_86), driver 535.309.01, Ryzen 5 3600,
47 GB DDR4, Ubuntu 24.04, headless. Some arms ran on rented L40S or AWS
instances and say so. Noise floor is about 1%; we do not call anything under
2% a difference.

---

## 1. The mistake that reset the program

**The thinking-mode harness bug, found 2026-09-02.** Our harness never sent
`enable_thinking=false`, and the model's chat template defaults thinking on.
Every quality number we had was measured with the model spending its 1024-token
budget on reasoning before answering. 26 of 29 recorded failures were
truncations, not wrong answers.

We found it while chasing something else. Quality on a rented L40S came out far
higher than on the 3060 with the same files, and for about a day the leading
hypothesis was a kernel defect in sub-4-bit i-quants on sm_86 — an upstream
correctness bug that would have affected a lot of people. It was not. A paired
cross-platform run of the uncompressed model gave 54/54 identical
completion-token counts, 0 discordant. The GPU was fine. Our flag was missing.

Reconstructing the old conditions reproduced the old score task-for-task
(164/164 match). Flipping the one flag was worth +18 tasks (p=0.00053). Every
other confound we suspected, combined, was worth exactly one task (p=1.0).

Corrected ladder (HumanEval-164, greedy, thinking off, 3060): IQ2_XXS 78.0% ·
Q2_K_XL 93.3% · IQ3_XXS 92.7% · Q8_0 anchor 93.3%.

Two consequences. The "~10 HumanEval points per GiB" exchange rate we had
published is retired — it was an artifact of the bug. Above roughly 2.9 bits
per weight, quantization damage on this model does not show in pass/fail coding
benchmarks at all; it shows in format and instruction adherence. And we added a
canary: if HumanEval mean completion tokens per task climbs above 350, the
switch is not taking effect and the run is discarded.

The wall-clock effect is the largest lever in the program: 4.79 s vs 13.38 s
per task, 158.8 vs 474 tokens. Roughly 4× larger than every configuration lever
combined. Raw tokens/sec is not work speed — the buggy protocol posted a
*higher* decode rate while doing the job 2.8× slower.

## 2. Three things we measured that generalise

**The α-precision law.** Across 34 configurations of one model at different
bit-widths, same-model twin disagreement follows 1−α₁ = 1.35·2^(−1.13·bpw),
R²=0.965: drafter-target agreement error halves for every ~0.9 bits removed,
with a knee below about 2.1 bpw, not a cliff. The constant is family-specific
and we have not measured the dense-model exponent. The useful part is the
shape — you can price a drafter before building it.

**The no-arbitrage law.** Bytes buy quality at one price on this model, and
nothing we tested bought them cheaper. We tested the direct version — a more
heavily quantized base plus a larger correction adapter at matched total bytes
— at three rungs of the ladder. Adapters lost by 2.0–2.7× at every rung.
Break-even needed 57–69% capture of the quantization residual; measured capture
was 13–17%. The residual's spectral *shape* turns out to be invariant along the
ladder: capture at matched rank is identical to 0.3 points across bases while
error magnitude spans 3.24×. Adapter headroom is a constant of the
architecture, not of the bit-rate. Structural pruning lost the same way, and
independent literature on expert pruning agrees that at equal bytes ≥3-bit
quantization beats pruning. Caveat: the HumanEval-based version of this
argument rested on the retired exchange rate. What survives the thinking-bug
correction is the residual-capture arithmetic, which does not depend on that
protocol.

**The cheap-helper theorem.** At batch 1 a helper model must be architecturally
small, not numerically small. Using a quantized twin of the target as its own
drafter: 0 of 25 configurations beat no speculation. The cause is cost, not
acceptance — a twin draft step costs 0.70–0.97 of a target step, and quartering
the drafter's bytes cut step time only 28%, because at batch 1 the step is
latency-bound. The built-in MTP head wins with one of the *worst* acceptance
rates we measured (0.738) purely because its cost ratio is 0.07–0.09. Confirmed
on a dense model too (0 of 6). Third confirmation from an unrelated lane: for
prompt-admission scoring a free lexical scorer beat every model-based scorer,
and the embedding one cost +325 ms for worse accuracy.

## 3. What failed

### Quality and model surgery

**Correction adapters (EoRA-class).** Build a low-rank adapter that corrects
quantization error so a smaller quant matches a larger one. The pipeline worked
— BF16-vs-dequant residual, imatrix-whitened, written as a GGUF LoRA, 505 of
866 tensors correctable. The best adapter (445 MiB) bought −0.44% perplexity,
while a byte spent on quantization bits removed 1.2–3.3× more error at every
rank; and 445 MiB does not exist in a 12GB budget without dropping speculation.
Unsloth's dynamic quantization had already spent its bits along the directions
the adapter would recover: capture 4.9% against a 1.5% noise floor. Correction
adapters and dynamic bit-allocation are substitutes, not complements. A 0.6B
pilot had said 29.5% capture and "go"; the real 27B said 4.9% and "no". Small
pilots overstate low-rank capture by roughly the hidden-dimension ratio. $3.10.

**Vocabulary pruning.** Strip 120K non-ASCII tokens to reclaim 555 MB. First
measured at −4.9 HumanEval points (p=0.0215) and killed; re-measured under the
corrected protocol at 93.29% vs 93.90% (p=1.0) and exonerated. It buys real
context: 57,344 vs 40,960 max with speculation on. The original damage was
mostly thinking-bug truncations — our clearest case for why every pre-09-02
verdict had to be re-audited. Still unshipped: it is an English-and-code
artifact by construction and we did not want that as a default.

**Structural pruning for speed.** A published 20B depth-pruned,
distillation-healed variant (64→44 layers) at a similar file size: faster
(28.44 vs 21.09 t/s) and worse on both instruments we had. Someone else paid
the pruning compute and the result still lost to quantizing the full model to
the same size. Caveat: those quality numbers were n=20 HumanEval, which our own
noise analysis says cannot resolve a 10-point gap. The claim we stand behind is
the byte-for-byte one from the adapter ladder, not this run.

**Expert pruning (REAP).** Dead by the same arithmetic and by independent
literature. Not run.

**EXL3 as an alternative quantization rail.** Our earlier "it doesn't fit"
claim had been wrong — it compared download size to VRAM, and exllamav3 forces
the embedding to host RAM — so the gate passed and we ran the battery. 3.0bpw
92.68%, 3.5bpw 93.29%, vs GGUF IQ3_XXS 93.90%; all McNemar p≥0.73, mutually
indistinguishable, both rails on the same ~93% ceiling. The "+1.3 GiB headroom"
from the earlier analysis was an apples-to-oranges KV comparison; matched-KV
saving is ~0.2 GiB, and the derived budget had under-counted real overhead by
0.61 GiB. About 9% slower at matched settings. Measured beats derived. $2.33.

### Drafters and speculation

**Training our own drafter.** A small block drafter on a personal corpus, to
beat the built-in MTP head. Run A proved the trainer works: block acceptance
0.7343 vs a bigram baseline of 0.2660 (n=3741 windows, 10k paired bootstrap),
untrained control exactly 0.0000. Run v2 spent 10× the compute plus an anneal
and reached 0.798 against a fitted ceiling of 0.767. It still failed the bar:
beating the MTP head already in the file needs mean 3.85 of 7 slots accepted.
Block acceptance is bounded by 7·p₁, so that needs per-slot p₁ ≥ 0.710; best
observed was 0.459, unresponsive to both levers. About 5× short, not a near
miss — an architecture problem, not a scaling problem. Also: in-training
512-window evals ranked our hyperparameters backwards, and paired bootstrap on
held-out data was required to see the true order. $67.67 across both runs.

**Block drafters from other projects (DSpark, DFlash).** Published checkpoints
for our exact model family, after an audited third-party 3090 study predicted
60–64 t/s. Measured −8.3% and −9.3% against the built-in MTP head: 12GB forces
the drafter to displace target layers, so it must beat MTP *and* repay what it
evicted. The trap is sharper than the numbers. The DSpark GGUF declares
`general.architecture=dflash`, loads without error through the dflash loader,
reports 4,722 healthy-looking drafted tokens, and runs at 0.0051 acceptance for
−62% throughput. A non-zero draft count is not evidence that a drafter works.
Acceptance is the required canary.

**DFlash2 on a 12GB card.** Every squeeze we had — vocab-pruned target,
vocab-pruned drafter, q4 KV, small micro-batch — got it to load at 12,002 of
12,044 usable MiB, where it segfaulted in graph compute during decode. On a
later upstream build that fixed the segfault it ran at −8.1% and +844 MiB. The
first failure was upstream instability, the second is the displacement
arithmetic above. Two clean reproduction stack traces were the useful output.

**`--spec-type draft-mtp-adaptive`.** The adaptive-depth path merged upstream
claims 2.60× on code. It creates a second full context against the target model
and OOMs even at c=8192 with 900 MiB free. Plain `draft-mtp` shares the main
context and is unaffected. The published number is datacenter-only until that
changes.

**Draft trees.** Not started, after the arithmetic. A 64-token trie is one
forward pass on a dense model; on this hybrid architecture the per-position
runtime cost puts break-even at mean accepted >4.41 against a best published
figure of 2.65.

**Algebraic state rewind, and a retraction.** The gated delta-net state update
inverts exactly via Sherman-Morrison, so a rejected speculative step could in
principle be rewound instead of snapshotted. The algebra is exact; the numerics
are dead. The inverse is an expansive map while forward replay is contractive:
at fp32, depth-3 rewind had p99 error 1.2e-3 and worst case 183%; bf16 went to
NaN. Forward replay is bit-identical, 0 ULP. *The retraction:* we had published
"+42% available from removing state-management overhead", based on a control
that used an n-gram drafter — which takes zero state snapshots, as its own VRAM
column showed. The 8.60 ms/token we attributed to state management is verify
and graph cost no checkpoint scheme can touch. Realistic gain is +3–7%. Check
that your control exercises the thing you are controlling for.

**The n-gram numbers, retracted.** We measured n-gram speculation at 86–148 t/s
and briefly treated it as the headline. Re-firing the *same* prompt against a
warm server lets the drafter replay its own previous answer, mean accepted run
56.7 tokens. On 5 distinct prompts it issued zero drafts: 1.00×, no gain. The
benchmark measured the harness. The idea survived, correctly scoped — n-grams
pay when the model re-emits text it can already see, which is what code editing
is, and that is where it certified honestly (§4).

**Inverted twin: big model in RAM, small model on GPU.** Host the uncompressed
29GB model in system RAM as verifier, use the quantized GPU model as its
drafter. The mechanism works — 1.15 → 1.63 t/s at draft depth 8, +42% — and the
magnitude does not; depth 16 goes negative from over-drafting. Cross-precision
greedy agreement is far lower than same-model intuition suggests (0.33 at depth
8) and a DDR4 verify pass is slow; even DDR5 scaling lands near 4 t/s. The
transferable finding: speculation speedup is a property of the *content*, not a
fixed multiplier — +129% on structured output, +110% on code, −2% to −4% on
prose in the same setup.

### Scheduling and MoE

**Hot-expert SSD tiering.** Trace which experts a MoE actually uses, keep the
hot ones resident, stream the rest. We traced 16.19M routed calls over 60
prompts in 4 domains. Dead by measurement: pooled expert skew 1.83× against a
kill gate of 2×; 90% of calls need 85.7% of the experts (uniform needs 90%);
the hot set is neither code-specific (Jaccard 0.239 vs cross-domain 0.247) nor
persistent (0.200 decile stability); and an LRU cache beat any static hot set
at every size — the operating system already runs a better policy than we would
have shipped. The per-layer imbalance factors of 6–80× quoted in papers are not
cacheability: a cache holds (layer, expert) pairs, and the union across layers
washes the skew out. Two side findings closed the lane harder than the physics.
40% of that model's bytes are a per-layer embedding table llama.cpp already
lazy-streams from disk, so the tierable mass was never the experts; and its
license required a separately negotiated agreement for AI coding products.
$1.75.

**Dynamic expert cache.** A published 1.91× on a similar MoE on a 12GB card
made this look worth building. We simulated it against our routing trace first.
Dispatch granularity is the killer: llama.cpp runs a layer on GPU only if *all*
of its top-8 experts are resident, so the joint hit rate is what matters, and
it is small — 12.5% of the pool cached gives 12% joint hits and 1.12×; 25%
gives 1.38×. Reaching 1.91× needs about 50% cached, which is just static
placement, which we already ship. And fetching a missing expert over PCIe
(0.57 ms Gen4, 1.2 ms Gen3) loses to computing it on the CPU, measured at
0.536 ms. The per-*access* hit rate does clear 1.91× at 12.5% cache, but
capturing that means computing some of a layer's experts on GPU and some on CPU
at once — intra-layer expert splitting, the one structural door we have not
closed. $0.

**Expert deferral.** The GPU sits idle about 48% of decode on the MoE build
while CPU-resident experts compute. The obvious fix is to let the CPU result
join the residual stream a layer or more late so the two overlap. We proved the
scheduling works first: a toy on a rented GPU showed concurrent CPU+CUDA split
execution with zero ggml internal changes and ~100% overlap efficiency,
bit-identical to serial. Then we simulated the *staleness* before building the
scheduler, and it died at every depth. HumanEval-164: d=1 153/164 (p=1.0), d=2
151 (p=0.61), d=3 139 (p=0.0015). But multi-file editing (34 aider-polyglot
tasks) collapsed at d=1: 13/34 → 3/34 first-attempt, ten regressions and zero
improvements, p=0.00195; d=2 and d=3 both 1/34. d=1 is the minimum depth that
buys any overlap, so no configuration survives, and zero improvements means
systematic capability loss rather than churn. This is the most important
instrument result in the program: **HumanEval would have green-lit d=1.** A
265-token self-contained function cannot show damage that compounds across an
1100-token multi-file edit. Benchmark *shape*, not benchmark quality. The kill
cost about $5 and a day against a 1–2 week build.

### Kernels and numerics

**Rewriting the GDN kernel.** Verifying more than one drafted token per step
costs 8.01 ms per extra token, which we had attributed to the gated-delta-net
state machinery being ALU-bound. Wrong: source reading plus our own batch probe
put verify-2 at 1.108× of verify-1, not the 1.79× the true pathology looks like
elsewhere. GDN's share of the 8.01 ms is bounded at 9–11%, so a rewrite buys
about +1.4%. At batch 1, weight traffic outweighs state traffic 68:1. Two
upstream recurrent-kernel rewrites shipped in the same period reporting
"generation unchanged, within noise." One side finding survived: the ~150 MiB
VRAM step per draft depth, which is what caps depth on 12GB, is the
per-position state snapshot planes (48 × 3.146 MB = 144 MiB, matching 152–156
measured).

**Routing small verify batches from MMVQ to MMQ.** Our 3-token verify batch
routes to a kernel that re-decodes the sub-4-bit codebook once per column. A
~10-line routing change sends small batches to the tiled kernel; an open
upstream issue asks for exactly this and reports ~9% elsewhere. We measured a
regression in all three regimes: dense build 35.33 → 29.11 t/s (−17.6%), MoE
build −5.6%, and at the n-gram chain's much larger verify batches (about 76,
twelve times past the predicted crossover) still −2.76%. The patch does remove
the per-column decode — marginal cost fell 7.81 → 1.52 ms/token — but the tiled
kernel pays a ~73 ms fixed entry cost at small batch. Crossover is at N=6; we
run N=3. A falsification arm with the threshold set to 6 reproduced the
unpatched build exactly: identical throughput, acceptance identical to 16
digits, identical output hash. The method lesson is bigger than the patch. Our
own runbook's rule — fit a line through a batch-size sweep, check the slope —
would have **passed** this change (fitted slope 3.98, r²=0.561: a step plus a
shallow slope, not a line), while the pre-registered prediction made at the
actual operating point (28.85) matched the measured result (29.11). What
survives is that the lever is sized: a batched kernel decoding the codebook
once across columns is worth about +5.5 t/s on the dense build.

**Rebasing onto a newer upstream with MoE kernel fusion.** An upstream fusion
PR landed shortly after our pin and looked like free speed. Measured −4.6% at
draft depth 2, slower at every depth. The fusion is not bit-exact: it perturbs
the MTP head's logits enough to drop acceptance from 0.773 to 0.704, and gives
back more in re-drafted tokens than it saves in kernel time. "Close enough"
kernels are not free under speculation, and acceptance is the cheap canary.

**TurboQuant KV.** A 4.125-bits-per-value KV codec from a third-party fork, to
buy context. Extraction is feasible — about 2,700 lines, zero conflict with our
patches — and the prize is 25.4 MiB per 1K context against q4_0's 27.0, about
6% more context than a KV type we can already use. Dead trade: the codec
materializes into an f16 scratch buffer so it can never beat f16, and the fork
carries a global −1.70% throughput and +182 MiB tax that cancels the win at our
context sizes. While measuring it we found the fork silently alters f16 and
q8_0 attention numerics through an unguarded comparison compiled into every
type instance — the leading suspect for that −1.70%, and a reason to be careful
comparing against non-upstream baselines.

**NVFP4 weights.** Evaluated as a quantization rail including on
current-generation hardware. Settled negative. Not run on our rig.

**Overclocking.** Clock offsets are impossible headless on this driver — they
need `nvidia-settings`, which needs X, and the management library exposes no
offset API. The only real lever is the power limit, and 170 W → 190 W bought
+1.4% for +11.8% power and +7 °C: the card sits against its software power cap
at 1880 MHz of a 2145 MHz maximum, so the workload is power-limited, not
clock-limited. The trap worth publishing is that `nvidia-smi` accepts
above-spec clock requests, prints "All done.", and silently clamps. The
above-spec arm delivered clocks identical to the in-spec arm and throughput
matching to four decimal places. Anything automating this must verify
`clocks.sm` and `clocks.mem` under load and never trust the exit status.

**Sparse prefill.** Not started, after fitting our own prefill curve. The
attackable term is 4.5% / 8.5% / 18.9% of prefill at 8K / 16K / 41K. A perfect
implementation saves at most 18 seconds at our context ceiling.

## 4. What worked, and the mechanism

**Thinking off — 2.8× wall-clock per task.** §1. Same pass rate, about a third
of the tokens. revv sets it server-side because leaving it to the client is how
we got it wrong ourselves.

**MTP self-speculation — +68%.** The model file contains its own draft head.
Depth 2 is the largest configuration lever; depth 3+ loses more acceptance than
it gains and is additionally VRAM-capped. Quality-neutral by measurement:
identical per-task outcomes across all 164 HumanEval tasks, p=1.0. **Not
bit-exact** — 3 of 5 diverse greedy probes differed between speculation on and
off, against a 5/5 identical control across a server restart, because of
floating-point summation order in the batched verify pass. We previously wrote
"byte-identical"; that was wrong and is corrected throughout.

**Thread count on the MoE build — +14.4%.** `-t 8` alone took the CPU-offloaded
configuration from 48.8 to 55.86 t/s. The reference box is a 6-core/12-thread
host presented as 10 vCPUs and the default oversubscribes; the sweep peaks at 8
and falls off both sides (t=6: 55.06, t=10: 48.50, t=12: 41.10). The mechanism
surprised us: the slope of the placement law is thread-invariant (0.5360 vs
0.5353 ms/layer at t=8 and t=10), so the win is not faster expert compute but
about 1.0–1.4 ms of fixed per-forward-pass overhead. Output was bit-identical
across t=3 through t=12: one hash, 22 cells.

**The n-gram drafter chain — 2.8–6.1× on editing, nothing elsewhere.** An
n-gram matcher chained in front of the MTP head, first-success-wins. On editing
tasks the model is largely re-emitting file content already visible in the
prompt, so the matcher hits long verbatim runs. Dense build: 40.3 → 222.8 /
246.0 / 113.3 t/s on three editing workloads (5.53× / 6.10× / 2.81×) and
35.17 → 35.16 on a pure-generation control, byte-identical on all four. MoE
build: +197% mean on the same three tasks (63 → 188 t/s, peak 243.6), also
byte-identical, generation acceptance unchanged. Acceptance is the wrong scalar
for this class of drafter — the chain runs at 0.48–0.93 against MTP's ~1.0 and
is still 2.8–6.1× faster, because one long verbatim run lands many accepted
tokens per verify pass; 0.4832 acceptance still bought 2.81×. Practical
warning: the matcher is a literal byte match, so a repo checked out with CRLF
drops acceptance from 0.83 to 0.11.

**Placement-aware quantization — +10.7%.** Choose a tensor's quantization by
which chip will execute it. On the MoE build, 16 blocks of experts live in host
RAM and run on the CPU. ggml has an AVX2 repack path for q4_K but none for
q3_K, so the CPU-resident experts were paying an unpacking tax the GPU-resident
ones were not. Retyping only those tensors moved throughput 55.84 → 59.68 t/s
(+6.9%) against a pre-registered prediction of +2. *The falsification arm is
the proof:* q6_K, with *more* bits and also no repack path, ran 5% **slower**
than stock. Speed moves opposite to bit-width here, so the mechanism is the
kernel dispatch table, not bandwidth or precision.

That first artifact did not ship. Its HumanEval-50 scores were 45/50 and 44/50
against a 48/50 baseline — not significant alone (p=0.375 / 0.125) — but 8 of 9
discordant tasks favoured the baseline and acceptance had moved 0.773 → 0.642,
confirming the weights really had changed. Cause: requantizing
already-quantized blocks with no imatrix. The shippable version inverts the
recipe — CPU-resident expert blocks taken natively from a higher-precision
published file, everything else from the lower one, as a pure tensor merge with
**zero** requantized tensors. That artifact measured 62.08 t/s (+10.7% vs 56.06
stock), HumanEval-164 151/164 (p=0.69), multi-file editing 9/34 first-attempt /
16/34 overall (p=1.000), passing both gates. Fidelity bought speed, not just
quality: two artifacts with identical CPU-layer types differed by 4.1 t/s
purely because one's acceptance had collapsed. Requantization noise costs
forward passes through the draft head; both axes move together.

**A CUDA kernel patch — +10.1% raw, +2.5% shipped.** sm_75 and newer removed
the SIMD-video instructions the sub-4-bit i-quant kernel relied on, so the
compiler emulates each in 4–5 instructions; the sign-recovery machinery was 18
of every 20 instructions in the inner loop. A carry-free SWAR multiply to
spread the sign bits (valid because no i-quant codebook byte is zero) plus
2-rows-per-block tiling gave +4.2% and +2.7% separately, +10.1% together.
Correctness: a 133,392-case exhaustive proof, upstream's own backend tests,
byte-identical greedy transcripts, machine code of untouched quant types
verified unchanged. *The attenuation is the finding.* On the shipping
speculative config it is worth +2.5% (35.8 → 36.7 t/s) with acceptance
identical at 0.8356 in both arms. Speculation amortizes the matrix-vector
kernel across a verify round, so a raw-decode gain is divided by the round
structure: the round takes 74.45 ms at depth 2, of which one verify forward is
45–50 ms, and the remaining ~24 ms of draft forwards and host graph work does
not benefit at all — and is now co-dominant with the kernel the patch fixes.

**Session save and restore — 18×.** Re-entering an 8K-token session takes
0.925 s instead of 16.7 s. The control that matters: llama.cpp's own RAM prompt
cache delivers *zero* reuse on this hybrid architecture — reported cache hit 0,
wall-clock indistinguishable from a cold prefill. This is not an optimization
on top of an existing mechanism; it is the only mechanism that produces reuse
here. Not wired into the revv CLI; the patch is in `patches/`.

## 5. The rules we ended up with

1. **Nothing ships on HumanEval alone.** Expert deferral at d=1 scored p=1.0 on
   HumanEval-164 and collapsed multi-file editing from 13/34 to 3/34. The
   editing instrument is mandatory for any change to the graph or the numerics.
2. **Edit-format compliance is a format check, not a correctness check.** It
   held at 34/34 → 32/34 across every deferral depth while solve rate fell 77%.
3. **Judge at the operating point, never by a fitted slope.** The MMVQ routing
   change would have passed a slope threshold and lost 17.6% where we run.
4. **Acceptance is a required canary and not a sufficient one.** It caught the
   fusion regression and the broken third-party drafter. It is the wrong scalar
   for block drafters, and it is *expected* to move whenever weights genuinely
   change — then check it moves monotonically in fidelity and use a real
   quality battery for the verdict.
5. **Certify VRAM from a peak sampled during requests, not at load.** Three
   configurations in one sweep passed a load-time check and died on the first
   request.
6. **Measure headroom against the usable ceiling, not the nominal one.** A 12GB
   3060 reports 12,288 MiB and gives about 12,044; the difference is
   driver-reserved. Our standard is ≥200 MiB of real headroom across
   consecutive deep requests. A config certified against the nominal figure
   looks about 244 MiB safer than it is.
7. **Never benchmark speculation by re-firing one prompt.** It measures the
   cache.
8. **Never A/B two quantizations on two different cards.** Sub-2-bit i-quants
   showed a 67% cross-platform identical-token rate against a 100% same-GPU
   control.
9. **Perplexity orders risk; it does not decide.** It mis-ordered deferral depth
   in the wrong direction (+14.5% at the depth HumanEval called clean).
10. **Measured beats derived.** Every VRAM budget we computed on paper
    under-counted the real thing.
11. **State the full configuration with any context claim.** Our own "40,960
    context" figure was measured without speculation, at a different KV
    precision, with smaller buffers, and was not comparable to the config it
    was quoted beside.

## 6. What this cost

Recorded figures: the adapter investigation $3.99 including the ladder
extension, the α-precision curve $7.66, drafter training $67.67 across two
runs, the quality battery $19, the uncompressed anchor $6, EXL3 $2.33, MoE
tiering $1.75, the scheduling toy $0.70, expert deferral about $5, and the
expert-cache simulation $0.

The two decisions that paid best were both simulations run before writing any
code: the expert-cache trace replay and the deferral staleness sim. Together
they cost about $5 and closed two multi-week build projects.

Corrections welcome: **contact@mericanii.com**, or open an issue. If a number
here is wrong, that is worth knowing.
