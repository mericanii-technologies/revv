#!/usr/bin/env python3
"""Regenerate the synthetic GGUF fixtures used by tests/test_planner.py.

    python3 tests/make_fixtures.py

These files are NOT real models. They contain no weights (every tensor is
zero-filled F32) and llama.cpp cannot load or run them -- they exist purely
so revv's GGUF metadata parser and launch planner (arch/geometry detection,
MTP-draft-head detection, "enable_thinking" chat-template detection) have
something strictly-valid to read in tests, without depending on a multi-GB
real model file or a scratch directory outside the repo.

This script is self-contained (stdlib only: struct, os, sys) and writes into
tests/fixtures/ relative to its own location, so `git clone` + `python3
tests/make_fixtures.py` is enough to make the test suite runnable. It
consolidates two earlier scratch generators (one for MTP-head detection, one
for chat-template/head-count detection) into a single file with no
cross-imports.

GGUF v3 layout written here (little-endian):

  header:
    magic        u32  0x46554747  ("GGUF")
    version      u32  3
    tensor_count u64
    kv_count     u64
  kv[kv_count]:
    key   = u64 length + utf8 bytes
    value_type u32
    value (scalar, string, or array per value_type)
  tensor_info[tensor_count]:
    name    = u64 length + utf8 bytes
    n_dims  u32
    dims    u64 * n_dims
    ggml_type u32
    offset  u64   (relative to the start of the data section)
  padding to next multiple of general.alignment (default 32) -> data section
  tensor data, each tensor at its declared relative offset

Only ggml type 0 (F32, block size 1, 4 bytes/elem) is used, so a tensor's
byte size is simply prod(dims) * 4. Every tensor's dims in this fixture set
are chosen so that size is already a multiple of `alignment`, which means
tensors pack back-to-back with zero inter-tensor padding -- that keeps
`data_offset + tensor_data_bytes == file_size` exactly, which is asserted
below for every fixture written.
"""

import os
import struct
import sys

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3

VT_UINT32 = 4
VT_STRING = 8
VT_ARRAY = 9

GGML_TYPE_F32 = 0
DEFAULT_ALIGNMENT = 32


# ---------------------------------------------------------------------------
# Low-level GGUF v3 packing helpers
# ---------------------------------------------------------------------------

def pack_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_string(key: str, value: str) -> bytes:
    return pack_string(key) + struct.pack("<I", VT_STRING) + pack_string(value)


def kv_uint32(key: str, value: int) -> bytes:
    return pack_string(key) + struct.pack("<I", VT_UINT32) + struct.pack("<I", value)


def kv_array_string(key: str, values) -> bytes:
    out = pack_string(key) + struct.pack("<I", VT_ARRAY)
    out += struct.pack("<I", VT_STRING) + struct.pack("<Q", len(values))
    for v in values:
        out += pack_string(v)
    return out


def tensor_nbytes(dims) -> int:
    n = 1
    for d in dims:
        n *= d
    return n * 4  # F32, block size 1, 4 bytes/elem


def build_gguf(path, arch, block_count, embedding_length, context_length,
                tokens, tensors, head_count=None, head_count_kv=None,
                chat_template=None, alignment=DEFAULT_ALIGNMENT):
    """Write one synthetic GGUF v3 file to `path`.

    tensors: list of (name, dims_tuple), all written as F32 (type 0).
    chat_template: optional tokenizer.chat_template string value. revv only
    checks for the literal substring "enable_thinking" in it, so fixtures
    use short template strings that either contain or omit exactly that
    substring. head_count / head_count_kv, if given, are written as
    {arch}.attention.head_count[_kv].

    Returns (data_offset, tensor_data_bytes) for the caller's own
    verification -- callers should also re-derive this from the file on
    disk rather than trust these numbers blindly.
    """

    kv_items = [
        kv_string("general.architecture", arch),
        kv_uint32("{}.block_count".format(arch), block_count),
        kv_uint32("{}.embedding_length".format(arch), embedding_length),
        kv_uint32("{}.context_length".format(arch), context_length),
    ]
    if head_count is not None:
        kv_items.append(kv_uint32("{}.attention.head_count".format(arch), head_count))
    if head_count_kv is not None:
        kv_items.append(kv_uint32("{}.attention.head_count_kv".format(arch), head_count_kv))
    if chat_template is not None:
        kv_items.append(kv_string("tokenizer.chat_template", chat_template))
    kv_items.append(kv_array_string("tokenizer.ggml.tokens", tokens))

    kv_count = len(kv_items)
    kv_bytes = b"".join(kv_items)

    # Compute each tensor's relative offset, aligning the cursor before each
    # tensor (a no-op for this fixture set since every size is already a
    # multiple of `alignment`, but done properly so the generator is correct
    # in general, not just for these particular dims).
    offsets = []
    cur = 0
    for _name, dims in tensors:
        cur = ((cur + alignment - 1) // alignment) * alignment
        offsets.append(cur)
        cur += tensor_nbytes(dims)

    tensor_info_bytes = bytearray()
    for (name, dims), off in zip(tensors, offsets):
        tensor_info_bytes += pack_string(name)
        tensor_info_bytes += struct.pack("<I", len(dims))
        for d in dims:
            tensor_info_bytes += struct.pack("<Q", d)
        tensor_info_bytes += struct.pack("<I", GGML_TYPE_F32)
        tensor_info_bytes += struct.pack("<Q", off)

    header = struct.pack("<IIQQ", GGUF_MAGIC, GGUF_VERSION, len(tensors), kv_count)
    pre_data = header + kv_bytes + bytes(tensor_info_bytes)

    data_offset = ((len(pre_data) + alignment - 1) // alignment) * alignment
    pad1 = b"\x00" * (data_offset - len(pre_data))

    data_section = bytearray()
    for (name, dims), off in zip(tensors, offsets):
        assert len(data_section) <= off, "generator bug: offsets went backwards"
        if len(data_section) < off:
            data_section += b"\x00" * (off - len(data_section))
        data_section += b"\x00" * tensor_nbytes(dims)  # 0.0 floats

    with open(path, "wb") as f:
        f.write(pre_data)
        f.write(pad1)
        f.write(bytes(data_section))

    tensor_data_bytes = sum(tensor_nbytes(dims) for _name, dims in tensors)
    return data_offset, tensor_data_bytes


# ---------------------------------------------------------------------------
# Fixture set -- must match the case table in tests/test_planner.py exactly
# ---------------------------------------------------------------------------

TOKENS = ["<unk>", "<s>", "</s>", "hi", "the", "a", "!"]  # 7 tokens -> n_vocab
CTX_TRAIN = 32768

TEMPLATE_NO_THINK = (
    "{% for message in messages %}{{ message['role'] }}: "
    "{{ message['content'] }}\n{% endfor %}"
)
TEMPLATE_WITH_THINK = (
    "{% for message in messages %}{{ message['role'] }}: "
    "{{ message['content'] }}\n{% endfor %}"
    "{% if enable_thinking is defined and enable_thinking %}<think>{% endif %}"
)

BASE_TENSORS = [
    ("token_embd.weight", (8, 7)),
    ("blk.0.attn_norm.weight", (8,)),
    ("blk.1.attn_norm.weight", (8,)),
    ("output.weight", (8, 7)),
]

NEXTN_TENSORS = [
    ("blk.64.nextn.embed_tokens.weight", (8, 7)),
    ("blk.64.nextn.eh_proj.weight", (8, 8)),
    ("blk.64.nextn.enorm.weight", (8,)),
]

FIXTURE_SPECS = {
    # gemma_like: no nextn head, no thinking switch. Neither lever applies.
    "gemma_like.gguf": dict(
        arch="gemma4", block_count=48, embedding_length=3584,
        context_length=CTX_TRAIN, head_count=16, head_count_kv=8,
        chat_template=TEMPLATE_NO_THINK, tokens=TOKENS,
        tensors=BASE_TENSORS,
    ),
    # qwen_like: has nextn head, has thinking switch. Both levers apply.
    "qwen_like.gguf": dict(
        arch="qwen35", block_count=64, embedding_length=5120,
        context_length=CTX_TRAIN, head_count=40, head_count_kv=8,
        chat_template=TEMPLATE_WITH_THINK, tokens=TOKENS,
        tensors=BASE_TENSORS + NEXTN_TENSORS,
    ),
    # head_no_think: has nextn head, no thinking switch. Same geometry as qwen_like.
    "head_no_think.gguf": dict(
        arch="qwen35", block_count=64, embedding_length=5120,
        context_length=CTX_TRAIN, head_count=40, head_count_kv=8,
        chat_template=TEMPLATE_NO_THINK, tokens=TOKENS,
        tensors=BASE_TENSORS + NEXTN_TENSORS,
    ),
    # think_no_head: no nextn head, has thinking switch. Same geometry as qwen_like.
    "think_no_head.gguf": dict(
        arch="qwen35", block_count=64, embedding_length=5120,
        context_length=CTX_TRAIN, head_count=40, head_count_kv=8,
        chat_template=TEMPLATE_WITH_THINK, tokens=TOKENS,
        tensors=BASE_TENSORS,
    ),
}


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    fixtures_dir = os.path.join(here, "fixtures")
    os.makedirs(fixtures_dir, exist_ok=True)

    # revv itself is the ground truth for "did this file parse the way we
    # intended" -- import it directly (repo root is the parent of tests/)
    # rather than re-deriving GGUF-reading logic here, which would let this
    # script and revv.py silently drift apart.
    sys.path.insert(0, os.path.dirname(here))
    import revv  # noqa: E402  (import after sys.path mutation, by design)

    failed = []

    for filename, spec in FIXTURE_SPECS.items():
        path = os.path.join(fixtures_dir, filename)
        data_offset, tensor_data_bytes = build_gguf(path, **spec)

        info = revv.read_gguf(path)
        size = os.path.getsize(path)
        exact = (info.data_offset + info.tensor_data_bytes == size)

        print("%-20s size=%-6d arch=%-8s n_vocab=%-4s mtp_head=%-5s "
              "thinking=%-5s n_head=%s/%s%s"
              % (filename, size, info.arch, info.n_vocab, info.has_mtp_head,
                 info.supports_thinking, info.n_head, info.n_head_kv,
                 "" if exact else "  EXACT-SIZE-CHECK-FAILED"))

        if not exact:
            failed.append(filename)
            continue

        # Cross-check the low-level packer's own offset/size bookkeeping
        # against what revv independently parsed back out of the file.
        assert data_offset == info.data_offset, (
            "%s: generator data_offset=%d != revv data_offset=%d"
            % (filename, data_offset, info.data_offset)
        )
        assert tensor_data_bytes == info.tensor_data_bytes, (
            "%s: generator tensor_data_bytes=%d != revv tensor_data_bytes=%d"
            % (filename, tensor_data_bytes, info.tensor_data_bytes)
        )

    if failed:
        print("\n%d fixture(s) FAILED verification: %s"
              % (len(failed), ", ".join(failed)))
        return 1

    print("\nwrote %d synthetic GGUF fixtures to %s" % (len(FIXTURE_SPECS), fixtures_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
