# TEST_WSL2_RESULTS.md: results of the end-to-end smoke test on the second machine (WSL2, RTX 3060)

Run: 2026-09-08, a Windows 10 desktop running WSL2 with Ubuntu 26.04 LTS
(glibc 2.43, Python 3.14.4, git 2.53). NVIDIA driver 610.74 installed on
Windows, CUDA passed through to the guest. RTX 3060 12GB with the monitor
attached to it. AMD Ryzen 5 3600, 6 cores / 12 threads, no integrated GPU.
32 GB RAM. C: is an SSD (the WSL virtual disk lives there); D: is a 2 TB
HDD. Install was driven over SSH from another machine, then the five revv
commands were run exactly as a user would.

Reference machine for comparison ("the Linux box"): RTX 3060 12GB, Ubuntu
24.04 headless, 10-vCPU KVM guest of a Ryzen 5 3600 host, 47 GB RAM, `-t 8`.
Its certified numbers: dense build serves ctx 12,288, `revv bench` 37.9
t/s; MoE build serves ctx 16,384, `revv bench` 55.9 t/s, peak VRAM 11,832
MiB, 12,044 MiB free VRAM at launch.

**Headline: the prebuilt installs and runs correctly on a machine that did
not build it, both builds serve, thinking suppression holds, and teardown
is clean.** Two installer defects and two planner defects were found and
fixed the same day. No quality regression is possible or claimed: same
model file, same flags, same binary as the Linux box.

---

## Summary table

| Item | Verdict |
|---|---|
| install: prebuilt path, sha256, `revv doctor` | **PASS** |
| dense build: forced measurement, ctx 4096 and 8192 | **PASS** (both clear the >=200 MiB standard) |
| dense build: unforced `revv up`, refused pre-fix | blocked by 12 MiB, then fixed |
| dense build: unforced `revv up`, post-fix | **PASS** ctx 8,192, q8_0 KV |
| dense build: `revv bench` | **PASS** 36.25 t/s (within 5% of 37.9) |
| dense build: thinking-off check | **PASS** 0 / 0 / 120 reasoning chars |
| dense build: `revv compare` | **PASS** decode 1.58x, time to done 3.74x |
| dense build: `revv down` | **PASS** clean |
| MoE build: download and `revv inspect` | **PASS** CERTIFIED, draft head present |
| MoE build: WSL2 RAM cap | blocked at 15 GB, fixed via `.wslconfig` |
| MoE build: VM disk fill mid-download | blocked, fixed by freeing C: space |
| MoE build: unforced `revv up moe`, pre-fix | **PASS** but degraded, ctx 4,096, q4_0 KV |
| MoE build: `revv bench`, pre-fix | **PASS** 47.02 t/s (84% of reference 55.9) |
| MoE build: thinking-off check | **PASS** 0 / 0 / 511 reasoning chars |
| MoE build: `revv compare`, pre-fix | **PASS** decode 2.24x, time to done 4.49x |
| MoE build: forced measurement, ctx 4096/8192/12288/16384 | **PASS** 3 of 4 rungs clear the standard |
| MoE build: unforced `revv up moe`, post-fix | **PASS** ctx 12,288, q8_0 KV |
| MoE build: `revv bench`, post-fix | **PASS** 47.21 t/s |
| MoE build: `revv down` | **PASS** clean |

---

## Install

`./install.sh` took the prebuilt path. The platform check passed (Linux
x86_64, glibc 2.43). It downloaded the 723 MB archive at about 7 MB/s,
verified the sha256, prefix `c4af4f1f`, and installed to `~/.revv/bin`.

`revv doctor` reported: `install: revv prebuilt binary (patched, CUDA)`,
`kernel patch applied`.

This is the first install of the prebuilt binary on a machine that did not
build it. Two installer defects were found and fixed the same day, in
commit `3f9542d`:

1. A helper function overwrote the caller's `dest` variable, so the
   downloaded archive stayed under its temporary name.
2. The runtime-library check ran `ldd` without the bundled library path on
   the search path, and printed a false "install cuda-cudart" warning. The
   runtime is inside the archive.

---

## Dense build (Qwen3.8-27B, UD-IQ3_XXS)

The model file was already on disk from a Sept 2 field test, so no
download was needed for this build.

### Refusal, then forced measurement

An unforced `revv up` initially **refused**: 11,516 MiB free against a
generic floor of 11,528 MiB, short by 12 MiB.

Forced measurement protocol: q8_0 KV, the n-gram + MTP chain on, `-ctxcp
0`, two consecutive context-filling requests per rung, `nvidia-smi`
sampled every second, whole-process peak recorded.

| ctx | free before | process peak | headroom | notes |
|---|---|---|---|---|
| 4096 | 11,583 MiB | 11,307 MiB | 276 MiB | 4,056-token prompts, 40-token completions, 34.9 t/s |
| 8192 | 11,583 MiB | 11,253 MiB | 330 MiB | 6,927-token prompts, 900-token completions, 19.2 t/s |

A short unforced request at 4096 ran 36.3 t/s, chain acceptance 0.95, mean
draft length 2.9. Both rungs clear the certification standard of at least
200 MiB measured headroom.

### Planner fix

Fixed in commit `3f9542d`: these rungs now carry their measured figures
with a 150 MiB margin instead of anchor arithmetic, which had charged
11,378 MiB for them (71 MiB too pessimistic) with a 250 MiB margin. The
refusal floor is now derived from the smallest measured rung: 11,457 MiB.

### Post-fix, unforced

    revv up
      planned ctx 8,192, q8_0 KV

- Single request: 37.9 t/s.
- `revv bench`: 36.25 t/s mean, spread 2.4%, chain acceptance 0.793,
  verdict "on target: within 5% of the reference (37.9)".
- Thinking-off check: **PASS**. Three arms A/B/C produced 0 / 0 / 120
  reasoning characters. (Arm A is the server-side flag only, with no
  per-request kwarg; arm B is the per-request kwarg; arm C is the
  positive control with `enable_thinking` true. Arm C firing is what
  makes A meaningful.)
- `revv compare`: STOCK 21.9 t/s, 2,048 tokens, 94.2 s. REVV 34.6 t/s, 865
  tokens, 25.2 s. Decode 1.58x, time to done 3.74x. Chain acceptance
  0.519.
- `revv down`: clean.

The dense build is nearly identical to the Linux box, because it sits
entirely on the GPU. It only loses a context rung (8,192 rather than
12,288) to the desktop's share of the card's VRAM.

---

## MoE build (Qwen3.6-35B-A3B, UD-Q3_K_XL, 16.04 GiB)

`revv get moe` downloaded the file at about 40 MiB/s. `revv inspect`
verdict: CERTIFIED (moe line), draft head present.

### Two things that blocked the run

1. **WSL2 RAM cap.** The guest showed only 15 GB of RAM, because WSL2
   defaults to capping the guest at half the host's RAM. Fixed by writing
   `%UserProfile%\.wslconfig` with a `[wsl2]` section containing
   `memory=24GB`, `swap=0`, `vmIdleTimeout=-1`, then `wsl --shutdown`
   from PowerShell.
2. **VM died mid-download with I/O errors on the root filesystem.** Cause:
   the C: drive had filled, and the WSL virtual disk lives on C: and
   grows on demand, so the guest's disk failed. The user freed space on
   C:, and the download resumed from 85%.

### Unforced, before the planner fix

    revv up moe
      planned ctx 4,096, q4_0 KV

- Requests: 42.6 and 40.9 t/s, chain acceptance 0.735.
- `revv bench`: 47.02 t/s, spread 0.7%, acceptance 0.668, verdict "84% of
  reference (55.9)".
- Thinking-off check: **PASS**, arms 0 / 0 / 511 reasoning characters.
- `revv compare`: STOCK 18.7 t/s, 2,048 tokens, 110.1 s. REVV 41.9 t/s,
  1,010 tokens, 24.5 s. Decode 2.24x, time to done 4.49x. Chain
  acceptance 0.411.

### Forced measurement, same protocol

q8_0 KV, chain on, `-ctxcp 0`, context-filling requests, `nvidia-smi`
sampled every second.

| ctx | free before | process peak | headroom | notes |
|---|---|---|---|---|
| 4096 | 11,841 MiB | 11,407 MiB | 434 MiB | 3,008-token prompts, 600-token completions |
| 8192 | 11,841 MiB | 11,487 MiB | 354 MiB | 6,927-token prompts, 900-token completions |
| 12288 | 11,820 MiB | 11,521 MiB | 299 MiB | 11,154-token prompts, 900-token completions |
| 16384 | 11,820 MiB | 11,659 MiB | 161 MiB | under the 200 MiB standard; the Linux box's certified 11,832 MiB stands for this rung |

Three of the four rungs clear the >=200 MiB standard. The 16384 rung does
not, and no forced certification was substituted for it on this box; the
Linux box's certified figure is what stands.

### Why the planner got this build wrong

The KV term is small on this hybrid-attention model, so anchor arithmetic
extrapolated down from the certified 16384 peak over-charged the small
rungs by 200 to 400 MiB, and dropped the build to ctx 4,096 with q4_0 KV
on 11,555 MiB free.

### Planner fix

Fixed in commit `4d126fc`: the three measured rungs carry their own
figures, a monotonic guard stops any estimate for a larger context or a
wider KV type from undercutting a measured smaller configuration, and
`revv doctor` now plans against the real model file on disk.

### Post-fix, unforced

    revv up moe
      planned ctx 12,288, q8_0 KV
      292 MiB free at load

- Single request: 49.3 t/s.
- `revv bench`: 47.21 t/s, acceptance 0.600.
- `revv down`: clean.

### Why the MoE build still trails the Linux box

The shortfall against the Linux box (47.2 vs 55.9, about 84%) is
expected and has known causes: two fewer threads (`-t 6` on this box's
six physical cores against `-t 8` on the Linux box's 10-vCPU guest), host
RAM reached through the Hyper-V guest, Windows running alongside, and the
desktop's share of the card costing a context rung. The stock arm
dropped by a comparable fraction (18.7 t/s here against 22.2 t/s on the
Linux box), which is the evidence that the cause is the machine and not
revv's configuration.

---

## Other observations

- Windows's share of the card moved during the session between 1,022 MiB
  (browser open), 932, 600, 568, and 275 MiB (desktop cleared). revv
  plans against free VRAM at the moment of launch, so this changes which
  context rung it picks, and can cause an OOM if a GPU-using app is
  opened mid-session.
- This box runs `-t 6` (six physical cores) against `-t 8` on the Linux
  box.

---

## Defects found

1. **Installer: `dest` variable clobbered.** A helper function
   overwrote the caller's `dest` variable, so the downloaded archive
   stayed under its temporary name. Fixed same day, commit `3f9542d`.

2. **Installer: false "install cuda-cudart" warning.** The
   runtime-library check ran `ldd` without the bundled library path on
   the search path, and reported a missing CUDA runtime even though the
   runtime ships inside the archive. Fixed same day, commit `3f9542d`.

3. **Planner: dense build refused a machine that had enough VRAM.**
   Anchor arithmetic charged 11,378 MiB for the small dense rungs, 71
   MiB more pessimistic than the measured figures, with a 250 MiB
   margin on top, which pushed the refusal floor to 11,528 MiB, 12 MiB
   above the 11,516 MiB actually free. Fixed same day, commit
   `3f9542d`: the small rungs now carry their measured figures with a
   150 MiB margin, and the floor derives from the smallest measured
   rung (11,457 MiB).

4. **Planner: MoE build undersized on a machine that had enough VRAM.**
   The KV term is small on this hybrid-attention model, so anchor
   arithmetic extrapolated down from the certified 16384-context peak
   over-charged the small rungs by 200 to 400 MiB, dropping the build
   to ctx 4,096 with q4_0 KV on 11,555 MiB free instead of a larger
   context with q8_0 KV. Fixed same day, commit `4d126fc`: the three
   measured rungs carry their own figures, a monotonic guard stops a
   larger-context or wider-KV estimate from undercutting a measured
   smaller configuration, and `revv doctor` plans against the real
   model file on disk.

All four defects were found and fixed on 2026-09-08, the same day as
this test run.

---

## What was not run

- No HumanEval or editing-instrument quality run was done on this box;
  quality is identical by construction (same model file, same flags,
  same binary as the Linux box).
- No forced 16384-context certification for the MoE build on this box;
  the forced measurement at that rung fell under the 200 MiB standard,
  so the Linux box's certified figure stands in its place.
- `revv adopt`.
- The `--source` or `--upstream` install path.
- Any multi-client concurrency behaviour.
