# Patches

Both patches in this directory apply to the pinned llama.cpp base commit
`daef7b6874397a5a7c3d7e38b55e2ee0adf7da38` (build b10712, "vulkan: top_k
radix select for k >= 1024 for Qwen 3.8 Flash Next (#28032)"). This was
verified by applying each one to a pristine checkout of exactly that
commit.

| File | sha256 |
|---|---|
| `mmvq_iquant_decode.patch` | `d70533dc2db7c836c01055d35aba31feaf5fe5ebbebbcb8cca763b0a58861ad7` |
| `pr26004-rebased-daef7b687.patch` | `a1d09de1c7777e43a9b6c97472a705830e7d6e1dfcecff6b0253d5d7c69baa23` |

`install.sh --patched` (the default) applies both. `install.sh --stock`
applies neither.

## mmvq_iquant_decode.patch

Speeds up sub-4-bit i-quant matrix-vector decode on CUDA. Two changes,
which are superadditive when combined:

1. A carry-free SWAR multiply that spreads i-quant sign bits, valid
   because no i-quant codebook byte is zero.
2. 2-rows-per-block tiling in the mmvq kernel at `ncols_dst==1`.

**Root cause it addresses:** sm_75+ removed the SIMD-video instructions
and nvcc emulates them in 4-5 instructions each, leaving the sign
machinery as 18 of every 20 inner-loop instructions.

**Measured effect:** +4.2% and +2.7% individually, +10.1% together on
raw decode (20.48 -> 22.54 t/s), and +2.5% end-to-end on the shipping
speculative-decoding config (35.8 -> 36.7 t/s) -- the gain attenuates
because speculation amortizes the GEMV across a round. Prefill is
untouched (a clean control).

**Correctness evidence:** a 133,392-case exhaustive proof, llama.cpp's
backend tests, 3x200-token greedy transcripts byte-identical across all
four builds, and SASS of untouched quant types byte-for-byte unchanged.

**Files touched:** `ggml/src/ggml-cuda/mmvq.cu`,
`ggml/src/ggml-cuda/vecdotq.cuh`.

## pr26004-rebased-daef7b687.patch

A REBASE of upstream llama.cpp PR #26004 onto the pinned commit -- the
upstream diff does not apply as-is; one context line had drifted.

It makes llama-server's slot save/restore actually work on this hybrid
recurrent architecture by appending state checkpoints to the slot save
file, because a recurrent state cannot be rewound and therefore cannot be
reconstructed from the final state alone.

**Measured effect:** first request after restore takes 0.925 s instead
of 16.7 s = 18.0x; save 1.13 s writing 997,687,036 bytes; restore
0.277 s. The control that matters: llama.cpp's RAM prompt cache
(`--cache-ram`) delivers ZERO reuse on this architecture (re-prefills all
8023 tokens, 16.691 s), so this patch is not an optimization, it is the
only working mechanism. Cost is roughly 125 MB of disk per 1K tokens.

**Files touched:** `tools/server/server-context.cpp`,
`tools/server/tests/unit/test_slot_save.py`.

**Not wired into the CLI:** session save/restore is NOT wired into the
`revv` CLI in v1.0. This patch is carried so that people who build with it
can drive llama-server's slot save/restore endpoints themselves.

## Upstream status

Both patches are PENDING UPSTREAM and not merged.

llama.cpp's AGENTS.md requires human-owned contributions, so these will
be submitted personally by the author rather than machine-generated. The
SWAR-signs change is staged as the low-risk first PR; the rows-per-block
change needs multi-GPU validation before submission.

The mmvq work also caught nvcc silently compiling `__byte_perm`'s sign
mode to identity, which is independently worth reporting upstream.

## How to apply by hand

If you don't trust `install.sh`, do it yourself:

```sh
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout daef7b6874397a5a7c3d7e38b55e2ee0adf7da38

git apply --check /path/to/revv/patches/mmvq_iquant_decode.patch
git apply /path/to/revv/patches/mmvq_iquant_decode.patch

git apply --check /path/to/revv/patches/pr26004-rebased-daef7b687.patch
git apply /path/to/revv/patches/pr26004-rebased-daef7b687.patch

cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build --config Release -j "$(nproc)"
```

## If a patch fails to apply

It means the tree is not at the pinned commit. Check what commit is
actually checked out:

```sh
git -C <your-llama.cpp-dir> rev-parse HEAD
```

If it does not print `daef7b6874397a5a7c3d7e38b55e2ee0adf7da38`, get back
to the pinned commit and try again:

```sh
git -C <your-llama.cpp-dir> fetch --depth 1 origin daef7b6874397a5a7c3d7e38b55e2ee0adf7da38
git -C <your-llama.cpp-dir> checkout FETCH_HEAD
```
