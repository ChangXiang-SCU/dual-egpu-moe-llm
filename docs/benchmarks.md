# Measurements

Hardware: GPD WIN Max 2 (Ryzen 7 7840U, 8C/16T, 64 GB LPDDR5), 2x RX 6950XT 16 GB in OCuLink docks,
both links Gen4 x4, Resizable BAR on. Windows 11, Vulkan backend.

Model: `unsloth/Qwen3.8-Flash-Next-GGUF` **UD-Q2_K_XL** — 78.9 GB, 176.9 B parameters, 48 MoE
layers, 512 experts per layer, 10 routed per token, 262144 native context.

Method: 6-8 rounds of a fixed prompt mix (Chinese prose, English explanation, Python, lists),
256 generated tokens each, streamed, one server slot. The first 2-3 rounds are cache warm-up and
are excluded from the ranges. Prefill measured separately with `cache_prompt: false` at several
prompt lengths. Hit rate is the engine's own cumulative counter (`--verbose`).

## Deployed configuration

```
LLAMA_MOE_CACHE_MAX_TOKENS=1
llama-server -m Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf \
  -dev Vulkan1,Vulkan2 -sm layer -ngl 99 -ot exps=CPU \
  --moe-expert-cache 265 --moe-expert-cache-inserts 2 \
  -fa on -c 262144 -ctk q8_0 -ctv q8_0 -ub 16 -b 512 -t 8 -np 1
```

**17.7-19.6 tok/s decode, ~18 tok/s prefill, at the full native 256k context.**

## Expert cache size vs decode speed

`-c 32768`, no speculative decoding, `-t 8`.

| slots/layer | VRAM used by cache | decode tok/s | cumulative hit rate |
|---|---|---|---|
| 0 (cache off) | 0 | **8.2** | — |
| 128 | 11 GB | 12-13 | ~85 % |
| 256 | 22 GB | 17.4-17.8 | 92 % |
| **280** | 24 GB | **18.9-19.1** | 93 % (95 % instantaneous) |
| 300 | 26 GB | **9.5** | — |

300 slots is past the per-card budget: allocation succeeds but the driver backs part of it with
host memory and throughput halves. **The usable ceiling on a 16 GB card here is about 15 GB
total** — model shard + cache + KV + compute buffer. Diminishing returns are sharp: 256 -> 280 slots
costs 2 GB of VRAM for 1 point of hit rate and about 1 tok/s.

## Context length: the compute buffer is the real cost, not KV

Flash-Next is a hybrid linear-attention model, so its KV cache is tiny even at 256k. What actually
competes with the expert cache is the **compute buffer**, and that is governed by `-ub`:

| context | `-ub` | KV/card | compute buffer/card | max slots | decode tok/s | prefill @957 tok |
|---|---|---|---|---|---|---|
| 32k | 512 | 76 MB | 687 MB | 280 | 18.9-19.1 | — |
| 64k | 512 | 153 MB | 949 MB | 265 | 17.5-17.7 | — |
| 128k | 512 | 306 MB | 973 MB | 275 | 17.2-17.9 | — |
| 256k | 512 | 612 MB | 2521 MB | ~200 | 15-16 | — |
| 256k | 128 | 612 MB | 1021 MB | 235 | 16.4-17.3 | 15.4 |
| **256k** | **16** | 612 MB | **275 MB** | **265** | **17.8-19.6** | **18.2** |

Dropping `-ub` from 128 to 16 frees 750 MB per card — worth 30 extra cache slots — **and** is
faster at prefill, because with experts on the host CPU a smaller ubatch has better locality. It
costs nothing at decode, which is single-token regardless. Full 256k context now costs essentially
nothing versus 32k.

At 256k with 265+ slots and `-ub 128` the allocation spills and decode drops to 8-9 tok/s. This
initially looked like a context-length scaling bug; a control run at 256k with 200 slots recovered
to 15-16 tok/s, confirming it is purely a VRAM budget effect.

Qwen documents YaRN (`factor 4.0`) to reach 1 M, but warns that static YaRN degrades shorter
inputs, so it is not a sensible default. Prefill is the practical limit anyway: at ~18 tok/s,
filling 256k would take four hours.

## Speculative decoding (MTP)

Requires [#28243](https://github.com/ggml-org/llama.cpp/pull/28243) for `qwen4exp` MTP, plus
[patch 0002](../patches/0002-moe-cache-small-batches.patch) — without it, verification batches of
2-4 tokens bypass the cache and run entirely on the CPU, which is slower than not drafting at all.

`LLAMA_MOE_CACHE_MAX_TOKENS=4`, `--spec-draft-n-max 3`, draft head on the second card, `-c 32768`:

| config | cache VRAM | decode tok/s | hit rate |
|---|---|---|---|
| 280 slots, no MTP | 24 GB | 18.9-19.1 | 93 % |
| 220 slots + `mtp-...-Q4_K_M` | 19 + 2.6 GB | 18.4-24.6 | 85 % |
| 230 slots + `mtp-...-shared-Q8_0` | 20 + 2.6 GB | **21-24** | 86 % (91 % inst.) |

Worth roughly +15 % net: the draft head costs 2.6 GB of VRAM, which costs ~50 cache slots and
~7 points of hit rate, and it still comes out ahead. Most of the hit-rate drop is the smaller
cache, not the drafting.

## Correctness: where the cache path can and cannot be used

Greedy decoding (`temperature 0`, `top_k 1`, fixed seed) on four fixed prompts — two short, two
with a 3000-character prefix — comparing each configuration against the same build with
`LLAMA_MOE_CACHE_MAX_TOKENS=1`. **Output hashes, not just plausibility.**

| max batch through the cache | result |
|---|---|
| 1 (upstream behaviour) | reference |
| 2-4, via MTP verification | **byte-identical to reference** on all four prompts |
| 16 | correct and well-formed, but not byte-identical |
| 32, 64, 128 | **corrupted** — a short arithmetic prompt is misread, long prompts emit `!!!!!!...` |

Two distinct failures are stacked at >=32:

1. **Vulkan topk-moe fusion misfires.** The fusion matcher identifies its pattern by fixed node
   offsets (`cgraph->nodes[node_idx + 4]` and similar), and the cache path inserts a `repeat` and a
   `get_rows` into that region. Running with `GGML_VK_DISABLE_GRAPH_OPTIMIZE=1` **fixes the short
   prompts** at batch 128, which confirms this one.
2. **The expert tables mutate mid-prefill.** A long prompt spans many ubatches. Between them,
   `llama_moe_cache_step()` publishes uploads queued during earlier decoding, so the CPU chain
   (which reads `host_table` at compute time) and the device chain (which reads `dev_table` when
   the GPU op runs) can see different table versions. The partition between them is then neither
   exclusive nor exhaustive. Disabling graph optimisation does not help here, and neither does
   `LLAMA_GRAPH_REUSE_DISABLE=1`. Fixing it needs the tables frozen or double-buffered for the
   duration of a multi-ubatch prefill — a change to the PR's synchronisation design, not a patch.

So the prefill-through-cache experiment is **not shipped**, despite being fast:

| prompt tokens | cache off for prefill | cache on for prefill (`-ub 128`) |
|---|---|---|
| 258 | 16.6 tok/s | 22.6 |
| 957 | 15.4 | 25.7 |
| 3722 | 17.4 | **30.5** |

+75 % and growing with prompt length — but wrong. Reported upstream rather than deployed. The
safe ceiling today is `LLAMA_MOE_CACHE_MAX_TOKENS=4`, which is exactly what MTP needs.

## Baselines and dead ends

| approach | result |
|---|---|
| llama.cpp mainline, both GPUs, `-ot exps=CPU`, no cache | 8.2 tok/s |
| llama.cpp mainline, layer split, experts statically partly on GPU | 6.1 tok/s |
| MoE4All / `infr`, single GPU paged expert cache | 18 tok/s decode, 6-17 prefill, 8k context |
| MoE4All `infr multi` | data-parallel only — different models per GPU |
| smaller quant to fit experts in 32 GB VRAM | would need ~1.5 bpw; not viable |

The static-partition result is the interesting one: putting 43 % of the experts permanently on the
GPUs and the rest on the CPU gives 6.1 tok/s, while an LRU cache holding a similar *fraction* gives
19. Expert selection is near-uniform in the long run but strongly clustered in the short run, so
what matters is not how much you can hold but what share of accesses land on the GPU — 93 % versus
about 43 %.

## Quantisation options (headers read over HTTP range requests, nothing downloaded)

| quant | experts | n-gram table | rest | total | effective bits/param on experts |
|---|---|---|---|---|---|
| UD-Q2_K_XL | 46.1 GB | 28.8 GB | 4.0 GB | 78.9 GB | 3.05 |
| UD-IQ3_XXS | 48.6 GB | 28.8 GB | 4.5 GB | 82.0 GB | 3.22 |
| UD-Q3_K_XL | 55.8 GB | 28.8 GB | 5.4 GB | 90.0 GB | 3.70 |

The 51B n-gram table is **byte-identical across all three** — unsloth keeps it at IQ4_NL — so a
quant upgrade only changes expert weights. And "Q2_K_XL" is not 2-bit: the dynamic quantisation
leaves enough tensors at higher precision that the experts average 3.05 bits.

RAM planning is subtler than "experts must fit in RAM". The experts are mmap'd; the ones sitting in
the VRAM cache have their host pages read only on insertion, so the OS can evict them. The
resident working set is closer to *experts minus what the cache covers* than to the full tensor,
which makes UD-Q3_K_XL less obviously out of reach on 64 GB than the raw numbers suggest. Untested.
