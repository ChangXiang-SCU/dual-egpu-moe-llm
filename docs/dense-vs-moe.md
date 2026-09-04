# The line that actually matters: does the model fit in VRAM?

All the engineering in this repo went into making a 176B MoE usable on 32 GB of VRAM. It worked —
17.8-19.6 tok/s decode at full 256k context, up from 8.2. Then the same machine ran a **27B dense**
model at 6-bit, and beat it on every axis that matters.

Both configurations use ~31 of the 32.7 GB available. Both run the full 262144 native context. The
difference is where the weights live.

| | Qwen3.8-Flash-Next 176B MoE | Qwen3.8-27B dense |
|---|---|---|
| quantisation | UD-Q2_K_XL (experts avg **3.05 bpw**) | UD-Q6_K (**~6.6 bpw**, near-lossless) |
| weights | 46 GB of experts in host RAM, 23 GB LRU cache in VRAM | 22 GB, **entirely in VRAM** |
| VRAM used | 31.0 / 32.7 GB | 31.4 / 32.7 GB |
| context | 262144 | 262144 |
| **decode** | 17.8-19.6 tok/s | **20.9 tok/s** |
| **prefill, 957-token prompt** | 18.2 tok/s | **~430 tok/s** |
| **prefill, 3722-token prompt** | 17.4 tok/s | **429 tok/s** |
| **prefill, 14834-token prompt** | ~18 tok/s (about 13.7 min) | **403 tok/s (36.8 s)** |
| time to first token | 1.7-2.2 s | **0.41-0.51 s** |
| quality, 8 mixed prompts | no measurable advantage | equal or better on 2 of 8 |

**24x on prefill.** That is the whole story.

## Why

The expert cache serves ~93 % of decode lookups from VRAM, so decode only pays the CPU for the
remaining 7 %. That is survivable — 19 vs 21 tok/s.

Prefill is different. The cache path is gated to single-token batches (raising that gate is
[numerically broken above batch 16](benchmarks.md), for two independent reasons), so **100 % of
prefill expert compute runs on the host CPU**. On a Ryzen 7 7840U that is ~18 tok/s no matter what
else you tune. Cache size, context length, speculative decoding and PCIe generation are all
second-order next to a term that is 100 % CPU.

The dense model never touches the CPU at all. Every layer is resident on a GPU; `-sm layer` splits
them across the two cards and the only traffic between cards is one activation vector per layer
boundary, which a Gen4 x4 link handles without noticing.

## What this means if you are planning a build

On a machine where **VRAM < weights**, the useful question is not "how big a model can I page in?"
It is "what is the largest model that fits *entirely*, and is it good enough?" Paging a much larger
model in gets you a fraction of a bit of extra quality and costs you an order of magnitude of
prefill — which is exactly the thing you notice when you paste a document, a codebase, or a long
conversation.

Concretely, on 32 GB:

- A ~27-32B dense model at Q6_K (~22-25 GB) fits with room for a large KV cache. This is the
  sweet spot.
- A ~30B MoE with ~3B active (Q5/Q6, ~25 GB) also fits entirely and would decode faster still,
  at some cost in quality per parameter. Worth testing if decode speed dominates your use.
- A 100B+ MoE does not fit and never will. It is a fun engineering problem — the rest of this repo
  is that problem — but it is not the fast answer.

The second GPU was still worth buying. It is what took the dense model from 3-bit at 8k context on
one card to 6-bit at 256k on two. It just turned out that the payoff was in the dense model, not in
the giant MoE it was bought for.

## Deployed configuration

```
llama-server -m Qwen3.8-27B-UD-Q6_K.gguf \
  -dev Vulkan1,Vulkan2 -sm layer -ngl 99 -ts 52,48 \
  -fa on -c 262144 -ctk q8_0 -ctv q8_0 -ub 512 -b 2048 \
  -t 8 -np 1 --jinja --reasoning off
```

`-ts 52,48` matters at this fill level. With the default split the two cards land at 14,866 and
16,153 MiB — the second card sits at 98.7 % and a long request tips it into host memory, which
halves throughput silently. Rebalancing gives 15,604 / 15,772 MiB with identical speed and real
headroom.

Speculative decoding (MTP) is worth +36 % decode (21 -> 28.5 tok/s) but the draft head plus its own
KV cache does not fit at 256k: measured 31,969 MiB with the cards at 98.9 %, spilling to 3.3 tok/s.
It was tested at three different tensor splits and with a q4_0 draft KV cache; none fit. At 64k it
fits comfortably, so that is kept as a second profile.

## How VRAM was actually measured

`llama-server --list-devices` reports the same "free" figure whether a model is loaded or not — it
reads the Vulkan heap budget, not live usage. Real per-adapter usage on Windows comes from the
performance counters:

```powershell
(Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage').CounterSamples |
  Where-Object {$_.CookedValue -gt 500MB} |
  ForEach-Object { '{0,7:N0} MiB' -f ($_.CookedValue/1MB) }
```

Every VRAM number on this page comes from that counter while the server was serving.
