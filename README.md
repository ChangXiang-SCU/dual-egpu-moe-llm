# A 176B MoE on a handheld, over two eGPUs

Notes, patches and measurements from getting **Qwen3.8-Flash-Next (176B MoE, 512 experts/layer)**
to run across **two RX 6950XT eGPUs** hanging off a **GPD WIN Max 2 (Ryzen 7 7840U, 64 GB RAM)**
by OCuLink, on Windows 11, with the Vulkan backend.

End result: **17.7–19.6 tok/s decode and ~18 tok/s prefill at the model's full native 256k
context**, up from 8.2 tok/s with the same two cards and no expert cache.

Three things here are worth someone else's time:

1. **[A BIOS fix for running two large-BAR eGPUs at once](docs/bios-dual-egpu-pfmmio.md)** — not
   specific to LLMs. If your second 16 GB eGPU makes allocations fail once Resizable BAR is on,
   this is probably why. I could not find this written up anywhere.
2. **[Building llama.cpp with Vulkan on Windows without the Vulkan SDK](docs/build-windows-vulkan-no-sdk.md)** —
   for machines where you cannot run the LunarG installer (no admin rights), plus the `glslc`
   version trap that breaks the shader build.
3. **[Two patches](patches/)** against [ggml-org/llama.cpp#27861](https://github.com/ggml-org/llama.cpp/pull/27861)
   (GPU-resident LRU cache for host-offloaded MoE experts): one to make it link under MSVC, one to
   let the cache serve small multi-token batches so speculative decoding can use it.

Plus [the measurements](docs/benchmarks.md), including the parts that did not work.

## The problem

Qwen3.8-Flash-Next's GGUF (`unsloth/Qwen3.8-Flash-Next-GGUF`, UD-Q2_K_XL) is 78.9 GB. Reading the
tensor table, that splits into:

| part | size | behaviour |
|---|---|---|
| `ffn_*_exps` — 512 experts × 48 layers | **46.1 GB** | 10 of 512 picked per token per layer; over a few hundred tokens essentially all of them get touched, so this is the hot set |
| `per_layer_token_embd.weight` — the 51B n-gram table | **28.8 GB** | 320 M rows × 160 dims; a few rows looked up per token. mmap keeps only the touched pages resident — tens of MB in practice |
| attention / norms / router | 3.2 GB | fits on the GPUs |
| token embedding + output head | 0.8 GB | |

So 32 GB of VRAM cannot hold the experts, and never will. The n-gram table is a red herring for
memory planning: it is a lookup table, it stays mmap'd on the SSD, and it is byte-identical across
every quantisation unsloth publishes (verified by reading the headers of UD-Q2_K_XL / UD-IQ3_XXS /
UD-Q3_K_XL over HTTP range requests — 28.8 GB in all three).

The useful framing is a three-tier memory hierarchy:

```
VRAM  32 GB   attention+router 3.4 GB  |  LRU cache of hot experts ~23 GB  |  KV+compute ~1 GB
RAM   64 GB   all 46 GB of experts, mmap-resident            + ~4 GB rest
SSD           the 28.8 GB n-gram table (only touched pages ever reach RAM)
```

PR #27861 implements exactly the missing middle: a per-layer LRU cache of expert slices in VRAM,
placed in the device buffer of that layer's router — so with `-sm layer` each GPU caches the
experts for the layers it owns, and the slots add up across cards.

## What it takes

```
llama-server -m Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf \
  -dev Vulkan1,Vulkan2 -sm layer -ngl 99 -ot exps=CPU \
  --moe-expert-cache 265 --moe-expert-cache-inserts 2 \
  -fa on -c 262144 -ctk q8_0 -ctv q8_0 -ub 16 -b 512 \
  -t 8 -np 1 --jinja --reasoning off
```

- `-ot exps=CPU` keeps the 46 GB of experts in host memory.
- `-sm layer` splits the 48 layers across the two cards.
- `--moe-expert-cache 265` gives each layer 265 of its 512 experts a VRAM slot (~23 GB total).
  Cached experts are computed on the GPU; the ~7 % that miss are computed on the CPU.
- `-np 1` matters: with more than one server slot, decode batches exceed one token and skip the
  cache path entirely.
- `-ub 16` is the non-obvious one. At 256k it cuts the compute buffer from 2.5 GB to 275 MB per
  card — worth 65 extra cache slots — *and* it is faster at prefill than `-ub 128` (18.2 vs
  15.4 tok/s on a 957-token prompt), because with experts in host memory a smaller ubatch has
  better locality. Decode is single-token either way, so it costs nothing there.

## Results

Qwen3.8-Flash-Next UD-Q2_K_XL, 2× RX 6950XT (OCuLink Gen4 x4 each), 7840U, 64 GB, Vulkan, decode
measured after cache warm-up:

| config | VRAM for cache | decode | hit rate |
|---|---|---|---|
| no expert cache (attention on GPU only) | 0 | **8.2 tok/s** | — |
| 128 slots/layer, 32k ctx | 11 GB | 12–13 | ~85 % |
| 256 slots/layer, 32k ctx | 22 GB | 17.4–17.8 | 92 % |
| 280 slots/layer, 32k ctx | 24 GB | 18.9–19.1 | 93 % / 95 % instantaneous |
| 235 slots, 256k ctx, `-ub 128` | 20 GB | 16.4–17.3 | — |
| **265 slots, 256k ctx, `-ub 16` (deployed)** | 23 GB | **17.8–19.6** | — |
| 300 slots, 32k ctx | 26 GB | 9.5 — *spilled to host* | — |
| with MTP, 230 slots, 32k ctx | 20 + 2.6 GB | 21–24 | 86 % |
| for reference: MoE4All/infr, single card | ~12 GB | 18 | 8k ctx only |

Prefill is the remaining wall: **~18 tok/s**, because batches larger than
`LLAMA_MOE_CACHE_MAX_TOKENS` skip the cache and every expert is computed on the 7840U. Letting
prefill use the cache is worth +75 % (30.5 tok/s on a 3722-token prompt) but is **numerically
broken above batch 16** for two separate reasons, both documented in
[docs/benchmarks.md](docs/benchmarks.md). It is reported upstream rather than shipped. Every
configuration here was checked with greedy decoding against output hashes, not just eyeballed for
plausibility.

See [docs/benchmarks.md](docs/benchmarks.md) for the full trade-off curves, the speculative-decoding
numbers, the VRAM cliff, and the quantisation survey.

## Hardware notes

- GPD WIN Max 2 (2023, Ryzen 7 7840U), BIOS 0.27. Both OCuLink ports run **Gen4 x4** — the native
  one and the one adapted from the M.2 slot (the first adapter board only wired two lanes; a
  different board fixed it).
- Resizable BAR must be enabled (`Advanced -> PCI Devices Common Settings -> Re-Size BAR Support`);
  it can take two reboots to stick. `pnputil /enum-devices /resources` lies about this — GPU-Z's
  Advanced -> PCIe Resizable BAR page and an actual large Vulkan allocation are the ground truth.
- Running **both** 16 GB cards at once needs the PFMMIO change described in
  [docs/bios-dual-egpu-pfmmio.md](docs/bios-dual-egpu-pfmmio.md).
- OCuLink link speed is worth checking: Gen3 vs Gen4 was 14.4 vs ~18 tok/s on the paging engine.

## Credit

- [ggml-org/llama.cpp#27861](https://github.com/ggml-org/llama.cpp/pull/27861) by **csantiago78** —
  the expert cache this all rests on. The patches here are small fixes on top, offered back.
- [ggml-org/llama.cpp#28243](https://github.com/ggml-org/llama.cpp/pull/28243) — MTP / `qwen4exp`
  support, needed for the speculative-decoding experiments.
- [kryptic-sh/infr](https://github.com/kryptic-sh/infr) and the MoE4All fork — the single-GPU paged
  expert engine used as the baseline here.
- Quantisations from [unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF).

## License

Notes and documentation: CC BY 4.0. Patches: MIT, matching llama.cpp.
