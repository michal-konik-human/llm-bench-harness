# Benchmark report — 20260919-231520

Config: **new-model-bielik-11b**  

Run at: 2026-09-19 23:15:20 CEST


## Verdict

- ✅ **bielik-11b / throughput** (210): pp512 **2888.3** tok/s, tg128 **45.1** tok/s
- ✅ **bielik-11b / thermal** (210): pp4096 **2694.5** tok/s, tg1024 **44.7** tok/s

✅ Hardware error counters (AER, ECC) unchanged — results are not contaminated by bus or memory errors.

## Hardware state at run time

Recorded automatically. Without these values the results below are not comparable with anything.


| Parameter | Value |
|---|---|
| CPU | AMD Ryzen 7 9700X 8-Core Processor |
| RAM | 60.5 GB |
| BIOS | F13c |
| GPU 0 | AMD Radeon AI PRO R9700 — gfx1201, 64 CU, 0000:03:00.0 |
| GPU 1 | AMD Radeon AI PRO R9700 — gfx1201, 64 CU, 0000:06:00.0 |
| GPU 2 | AMD Radeon Graphics — gfx1036, 2 CU, 0000:14:00.0 |
| GPU under test | index 0 (0000:03:00.0) |
| VRAM | 32624 MiB |
| Power cap | 210 |

## Software state at run time

| Parameter | Value |
|---|---|
| Kernel | 7.0.0-31-generic |
| Kernel cmdline | `BOOT_IMAGE=/boot/vmlinuz-7.0.0-31-generic root=UUID=2177f55b-d49b-4b63-b447-4b5d00d2e414 ro quiet splash vt.handoff=7` |
| GPU driver | amdgpu |
| ROCm | 7.2.4 |
| llama.cpp | `987498f` (2026-09-15) |
| llama.cpp build flags | `CMAKE_BUILD_TYPE=Release`, `GGML_CUDA=OFF`, `GGML_HIP=ON`, `GGML_VULKAN=OFF`, `GPU_TARGETS=gfx1201` |

## Measurement parameters

| Parameter | Value |
|---|---|
| Runs per cell | 3 — **separate OS processes** |
| Cooldown between runs | 45 s |
| Thermal guard | abort at junction ≥ 95 °C |

**Configurations used** — env vars and flags affect the result, so they are recorded verbatim:

| Config | Environment | llama-bench flags |
|---|---|---|
| throughput | `HIP_VISIBLE_DEVICES=0` | `-ngl 999 -sm none -p 512 -n 128` |
| thermal | `HIP_VISIBLE_DEVICES=0` | `-ngl 999 -sm none -p 4096 -n 1024` |

## Results

| Model | Config | Cap | Test | Mean tok/s | min | max | Spread | Peak junction |
|---|---|---|---|---:|---:|---:|---:|---:|
| bielik-11b | throughput | 210 | pp512 | 2888.34 | 2879.78 | 2898.12 | 0.6% | 40 °C |
| bielik-11b | throughput | 210 | tg128 | 45.08 | 45.07 | 45.08 | 0.0% | 40 °C |
| bielik-11b | thermal | 210 | pp4096 | 2694.49 | 2692.06 | 2699.32 | 0.3% | 57 °C |
| bielik-11b | thermal | 210 | tg1024 | 44.75 | 44.73 | 44.77 | 0.1% | 57 °C |

## How to read this

**Spread matters more than the mean.** Decode throughput on this class of card is documented as *bimodal* — the card settles into one of two modes and stays there for the life of the process. That is why every run here is a separate process rather than `-r N`.

- **Spread below ~2 %** → bimodality did not appear; the mean is a sensible number to quote.
- **Spread above ~10 %, results in two clusters** → that is the bimodality. **Report both modes, not their mean** — the mean describes a state the card is never in.

**Peak junction near 90 °C** means you are measuring case airflow, not the model. Lower the power cap and repeat.

**Peak power below ~60 W on an otherwise valid result does not mean the GPU idled.** It means the compute window was shorter than the telemetry sampling period. For a fast MoE model at `-p 512 -n 128` the whole computation can take under a second, with the rest of the wall time spent loading weights — so the **thermal figures are meaningless while the throughput stays valid.** Use a longer test (e.g. `-p 4096 -n 2048`) for thermal work.

**The PCIe link** is read under load, because at idle it downtrains to save power. If you add a second GPU and the first drops from x16 to x8, the same config file gives you a directly comparable number.

