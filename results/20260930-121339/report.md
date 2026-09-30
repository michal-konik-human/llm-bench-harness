# Benchmark report — 20260930-121339

Config: **qwen3-235b-a22b-q3-4gpu**  

Run at: 2026-09-30 12:13:39 CEST


## Verdict

- ✅ **qwen3-235b-a22b-q3 / throughput** (250): pp512 **619.4** tok/s, tg128 **32.7** tok/s
- ✅ **qwen3-235b-a22b-q3 / thermal** (250): pp4096 **583.3** tok/s, tg1024 **32.6** tok/s

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
| GPU 2 | AMD Radeon AI PRO R9700 — gfx1201, 64 CU, 0000:09:00.0 |
| GPU 3 | AMD Radeon AI PRO R9700 — gfx1201, 64 CU, 0000:19:00.0 |
| GPU 4 | AMD Radeon Graphics — gfx1036, 2 CU, 0000:1a:00.0 |
| GPU under test | index 0 (0000:03:00.0) |
| VRAM | 32624 MiB |
| Power cap | 250 |

## Software state at run time

| Parameter | Value |
|---|---|
| Kernel | 7.0.0-34-generic |
| Kernel cmdline | `BOOT_IMAGE=/boot/vmlinuz-7.0.0-34-generic root=UUID=2177f55b-d49b-4b63-b447-4b5d00d2e414 ro quiet splash amdgpu.ppfeaturemask=0xfff7ffff vt.handoff=7` |
| GPU driver | amdgpu |
| ROCm | 7.2.4 |
| llama.cpp | `680a036` (2026-09-28) |
| llama.cpp build flags | `CMAKE_BUILD_TYPE=Release`, `GGML_CUDA=OFF`, `GGML_HIP=ON`, `GGML_VULKAN=OFF`, `GPU_TARGETS=gfx1201` |

## Measurement parameters

| Parameter | Value |
|---|---|
| Runs per cell | 3 — **separate OS processes** |
| Cooldown between runs | 60 s |
| Thermal guard | abort at junction ≥ 95 °C |

**Configurations used** — env vars and flags affect the result, so they are recorded verbatim:

| Config | Environment | llama-bench flags |
|---|---|---|
| throughput | `HIP_VISIBLE_DEVICES=0,1,2,3`, `NCCL_PROTO=Simple`, `NCCL_P2P_DISABLE=1` | `-ngl 999 -sm layer -ts 1/1/1/1 --load-mode dio -p 512 -n 128` |
| thermal | `HIP_VISIBLE_DEVICES=0,1,2,3`, `NCCL_PROTO=Simple`, `NCCL_P2P_DISABLE=1` | `-ngl 999 -sm layer -ts 1/1/1/1 --load-mode dio -p 4096 -n 1024` |

## Results

| Model | Config | Cap | Test | Mean tok/s | min | max | Spread | Peak junction |
|---|---|---|---|---:|---:|---:|---:|---:|
| qwen3-235b-a22b-q3 | throughput | 250 | pp512 | 619.39 | 618.39 | 620.18 | 0.3% | 53 °C |
| qwen3-235b-a22b-q3 | throughput | 250 | tg128 | 32.74 | 32.73 | 32.75 | 0.1% | 53 °C |
| qwen3-235b-a22b-q3 | thermal | 250 | pp4096 | 583.27 | 580.41 | 586.21 | 1.0% | 60 °C |
| qwen3-235b-a22b-q3 | thermal | 250 | tg1024 | 32.57 | 32.56 | 32.59 | 0.1% | 60 °C |

## How to read this

**Spread matters more than the mean.** Decode throughput on this class of card is documented as *bimodal* — the card settles into one of two modes and stays there for the life of the process. That is why every run here is a separate process rather than `-r N`.

- **Spread below ~2 %** → bimodality did not appear; the mean is a sensible number to quote.
- **Spread above ~10 %, results in two clusters** → that is the bimodality. **Report both modes, not their mean** — the mean describes a state the card is never in.

**Peak junction near 90 °C** means you are measuring case airflow, not the model. Lower the power cap and repeat.

**Peak power below ~60 W on an otherwise valid result does not mean the GPU idled.** It means the compute window was shorter than the telemetry sampling period. For a fast MoE model at `-p 512 -n 128` the whole computation can take under a second, with the rest of the wall time spent loading weights — so the **thermal figures are meaningless while the throughput stays valid.** Use a longer test (e.g. `-p 4096 -n 2048`) for thermal work.

**The PCIe link** is read under load, because at idle it downtrains to save power. If you add a second GPU and the first drops from x16 to x8, the same config file gives you a directly comparable number.

