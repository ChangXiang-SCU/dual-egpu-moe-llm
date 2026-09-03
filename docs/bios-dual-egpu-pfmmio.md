# Two large-BAR eGPUs on one machine: the PFMMIO padding setting

**Symptom.** One 16 GB eGPU with Resizable BAR works fine. Plug in a second one and both cards
still enumerate, both drive displays, `--list-devices` shows both with 16368 MiB — but any attempt
to allocate a large amount of VRAM fails. On the Vulkan backend it surfaces as:

```
ggml_vulkan: queue_submit: A device memory allocation has failed
```

The allocation that fails can be well under the 16 GB the card reports free. Shrinking the
allocation, moving the display output to the iGPU, and adding system RAM all fail to help.

**Cause.** OCuLink ports are hot-plug PCIe bridges. When firmware assigns MMIO windows to a
hot-plug bridge it reserves ("pads") an address range so a device can be inserted later. A 16 GB
Resizable BAR lives in **64-bit prefetchable** MMIO space, but on this firmware the padding was
being applied in the **32-bit** window, which is far too small to hold two 16 GB BARs. One card
squeezes in; the second one gets a window that the driver cannot actually back.

**Fix.** In BIOS (on a GPD WIN Max 2, `Del` at boot, then `ALT+F5` -> `F4` -> re-enter `Del` for the
hidden Advanced menu):

```
Advanced
└── PCI Devices Common Settings
    └── PCI Hot-Plug Settings
        ├── PFMMIO 32 bit Resources Padding ....... Disabled
        └── PFMMIO 64 bit Resources Padding ....... Enabled -> 8G      (largest this BIOS offers)
```

Save, reboot. Both cards now allocate their full VRAM.

Measured side effect on this machine: DMA import of a large model dropped from ~3 minutes to
1 min 43 s, and both cards reached the same throughput the single card had (14.0 tok/s each on the
paging engine used at the time).

## Notes

- The setting names vary by vendor. Look for *PFMMIO padding*, *Above 4G MMIO*, *MMIO High Base /
  Size*, or *PCIe hot-plug resource padding*. The principle is the same: the padding must be in the
  64-bit prefetchable window, and it must be at least as large as the BARs you intend to map.
- 8G of padding was enough for two 16 GB cards here. If your firmware offers more and you have more
  or larger cards, take more.
- Verify Resizable BAR with **GPU-Z -> Advanced -> PCIe Resizable BAR** (BAR0 should read 16384 MB),
  not with `pnputil /enum-devices /resources`, which reports a stale 256 MB regardless.
- Windows reports the GPU function's `CurrentLinkSpeed` as x16 on Navi 21 because the die has an
  internal PCIe switch. That is not your cable link. Read the link speed from GPU-Z instead.
- Enabling Resizable BAR sometimes needs two or three full power cycles before it takes.

## Why it matters beyond LLMs

Any workload that wants two large-BAR GPUs on a consumer or handheld platform hits this: dual eGPU
rendering, multi-GPU compute over Thunderbolt/OCuLink docks, or simply a desktop with a hot-plug
capable slot. The failure mode is misleading — the cards are present and healthy, and only
allocation fails — so it tends to get blamed on drivers or on the docks.
