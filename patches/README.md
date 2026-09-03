# Patches

Both apply on top of [ggml-org/llama.cpp#27861](https://github.com/ggml-org/llama.cpp/pull/27861)
(`csantiago78:moe-expert-cache`, commit `bccbacd`). They also cherry-pick cleanly onto
[#28243](https://github.com/ggml-org/llama.cpp/pull/28243) with #27861 applied, which is the
combination used for the speculative-decoding measurements.

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
git fetch origin refs/pull/28243/head && git checkout FETCH_HEAD      # MTP / qwen4exp
git remote add cs https://github.com/csantiago78/llama.cpp.git
git fetch --depth 2 cs moe-expert-cache
git cherry-pick bccbacdb8945680f1cfc7e6bffd1e59014705750              # expert cache, no conflicts
git apply ../patches/0001-msvc-file-locking.patch
git apply ../patches/0002-moe-cache-small-batches.patch
```

## 0001 — MSVC file locking

`ggml-cpu.c` uses `flockfile` / `funlockfile` in the `GGML_MOE_LOG` diagnostic block. Those are
POSIX; MSVC has `_lock_file` / `_unlock_file` instead, so the link fails with:

```
ggml-cpu.c.obj : error LNK2019: unresolved external symbol flockfile
ggml-cpu.c.obj : error LNK2019: unresolved external symbol funlockfile
```

Wrapped in `#if defined(_MSC_VER)`. MinGW is unaffected either way.

## 0002 — let the expert cache serve small batches

Upstream gates the cache graph on `n_tokens == 1`, so anything that decodes more than one token at
a time — speculative decoding above all — falls back to computing every expert on the CPU. On a
weak host CPU that is a large regression, which makes MTP a net loss rather than a net win.

This patch:

- replaces the hard `n_tokens == 1` gate with `n_tokens <= LLAMA_MOE_CACHE_MAX_TOKENS`, an
  environment variable defaulting to **1** (so behaviour is unchanged unless you opt in);
- fixes the id->slot lookup for `n_tokens > 1`. `mcache->dev_table` is `[1, n_expert]` while
  `selected_experts` is `[n_expert_used, n_tokens]`, and `ggml_get_rows` asserts
  `a->ne[2] == b->ne[1]`. The table is broadcast with `ggml_repeat_4d` to `[1, n_expert, n_tokens]`
  first, and the result reshaped to `[n_expert_used, n_tokens]`.

Everything else already generalises: the CPU-side skip in `ggml_compute_forward_mul_mat_id` tests
the table per `(token, expert)` pair, and the two `mul_mat_id` chains still sum to the exact result.
The LRU observation callback keeps its own `n_tokens > 4` guard, so prefill-sized batches use the
cache without polluting the eviction order.

Measured with `LLAMA_MOE_CACHE_MAX_TOKENS=4` and `--spec-draft-n-max 3` on Qwen3.8-Flash-Next
(2x RX 6950XT, Vulkan): decode 18.9-19.1 -> **21-24 tok/s**, despite giving up ~50 cache slots to
the draft head. Without the patch, the same configuration is *slower* than not drafting at all.

Untested: values above 4. Raising the limit to the ubatch size would let prefill use the cache too,
which is the obvious next experiment — see [../docs/benchmarks.md](../docs/benchmarks.md).
