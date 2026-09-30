# Benchmark report — 20260930-221230

Config: **deepseek-v4-flash-0731-4gpu** (`~/Desktop/github_repos/llm-bench-harness/configs/deepseek-v4-flash-0731-4gpu.json`)  
Run at: 2026-09-30 22:12:30 CEST

## Verdict

- ✅ **deepseek-v4-flash-0731 / throughput** (250 W): pp512 **559.7** tok/s, tg128 **24.7** tok/s
- ✅ **deepseek-v4-flash-0731 / thermal** (250 W): pp4096 **1089.2** tok/s, tg1024 **23.9** tok/s

✅ Hardware error counters (PCIe AER along each GPU's path, GDDR6 ECC) unchanged on every GPU used — results are not contaminated by bus or memory errors.

## Model

| Model | Architecture | Parameters | Quant (llama.cpp type) | File | Layers | Native context |
|---|---|---|---|---|---:|---:|
| deepseek-v4-flash-0731 | deepseek4 | 256x8.4B (MoE 256/6) | deepseek4 ?B IQ3_XXS - 3.0625 bpw | 97.05 GiB in 4 file(s) | 43 | 1048576 |

## Results

| Model | Config | Test | Mean tok/s | min | max | Spread | n |
|---|---|---|---:|---:|---:|---:|---:|
| deepseek-v4-flash-0731 | throughput | pp512 | 559.75 | 557.56 | 563.06 | 1.0% | 3 |
| deepseek-v4-flash-0731 | throughput | tg128 | 24.71 | 24.01 | 26.07 | 8.3% | 3 |
| deepseek-v4-flash-0731 | thermal | pp4096 | 1089.20 | 1085.46 | 1093.81 | 0.8% | 3 |
| deepseek-v4-flash-0731 | thermal | tg1024 | 23.85 | 23.84 | 23.86 | 0.1% | 3 |

## Per-GPU telemetry (peaks over all runs of the cell)

| Model / config | GPU | PCI | Link under load | Cap | Fan curve | Peak junction | Peak VRAM | Peak power | Peak fan |
|---|---|---|---|---:|---|---:|---:|---:|---:|
| deepseek-v4-flash-0731 / throughput | gpu0 | 0000:03:00.0 | x8 Gen5 | 250 W | 55:20 65:30 75:55 82:80 90:100 | 40 °C | 34 °C | 166 W | 1810 rpm |
| deepseek-v4-flash-0731 / throughput | gpu1 | 0000:06:00.0 | x4 Gen5 | 250 W | 55:15 65:30 75:55 82:80 90:100 | 62 °C | 36 °C | 205 W | 893 rpm |
| deepseek-v4-flash-0731 / throughput | gpu2 | 0000:09:00.0 | x8 Gen5 | 250 W | 55:15 65:30 75:55 82:80 90:100 | 56 °C | 38 °C | 258 W | 891 rpm |
| deepseek-v4-flash-0731 / throughput | gpu3 | 0000:19:00.0 | x4 Gen5 | 250 W | 55:15 65:30 75:55 82:80 90:100 | 43 °C | 36 °C | 144 W | 894 rpm |
| deepseek-v4-flash-0731 / throughput | **all** | | | | | **62 °C** | **38 °C** | **514 W** peak sum, 127.8 W mean | |
| deepseek-v4-flash-0731 / thermal | gpu0 | 0000:03:00.0 | x8 Gen5 | 250 W | 55:20 65:30 75:55 82:80 90:100 | 57 °C | 42 °C | 250 W | 1816 rpm |
| deepseek-v4-flash-0731 / thermal | gpu1 | 0000:06:00.0 | x4 Gen5 | 250 W | 55:15 65:30 75:55 82:80 90:100 | 75 °C | 54 °C | 255 W | 893 rpm |
| deepseek-v4-flash-0731 / thermal | gpu2 | 0000:09:00.0 | x8 Gen5 | 250 W | 55:15 65:30 75:55 82:80 90:100 | 70 °C | 58 °C | 254 W | 891 rpm |
| deepseek-v4-flash-0731 / thermal | gpu3 | 0000:19:00.0 | x4 Gen5 | 250 W | 55:15 65:30 75:55 82:80 90:100 | 72 °C | 64 °C | 251 W | 1792 rpm |
| deepseek-v4-flash-0731 / thermal | **all** | | | | | **75 °C** | **64 °C** | **943 W** peak sum, 288.3 W mean | |

## Effective llama-bench parameters

Read back from llama-bench's own JSON output, i.e. what actually ran — not what the config intended.

| Model / config | Split | Tensor split | FA | batch / ubatch | KV cache | Load mode | Threads | Build |
|---|---|---|---|---|---|---|---:|---|
| deepseek-v4-flash-0731 / throughput | layer | 1.00/1.00/1.00/1.00 | -1 | 2048 / 512 | f16/f16 | dio | 8 | 680a036 (ROCm) |
| deepseek-v4-flash-0731 / thermal | layer | 1.00/1.00/1.00/1.00 | -1 | 2048 / 512 | f16/f16 | dio | 8 | 680a036 (ROCm) |

## Hardware state at run time

| Parameter | Value |
|---|---|
| Board / BIOS | B850 AI TOP / F13c |
| CPU | AMD Ryzen 7 9700X 8-Core Processor — profile `balanced`, governor `powersave`, EPP `balance_performance` |
| RAM | 60.5 GB |
| gpu0 (HIP 0) | AMD Radeon AI PRO R9700 — gfx1201, 64 CU, 32624 MiB, 0000:03:00.0 — **used**, VBIOS `113-EXT119250-100`, subsystem `0x148c:0x2443`, idle link x8 Gen5 |
| gpu1 (HIP 1) | AMD Radeon AI PRO R9700 — gfx1201, 64 CU, 32624 MiB, 0000:06:00.0 — **used**, VBIOS `113-48WD6SHD1-P02`, subsystem `0x1eae:0x9801`, idle link x4 Gen5 |
| gpu2 (HIP 2) | AMD Radeon AI PRO R9700 — gfx1201, 64 CU, 32624 MiB, 0000:09:00.0 — **used**, VBIOS `113-48WD6SHD1-P02`, subsystem `0x1eae:0x9801`, idle link x8 Gen5 |
| gpu3 (HIP 3) | AMD Radeon AI PRO R9700 — gfx1201, 64 CU, 32624 MiB, 0000:19:00.0 — **used**, VBIOS `113-48WD6SHD1-P02`, subsystem `0x1eae:0x9801`, idle link x4 Gen5 |
| gpu4 (HIP 4) | AMD Radeon Graphics — gfx1036, 2 CU, 2048 MiB, 0000:1a:00.0 |
| GPU tuning service | active |
| llama-swap models loaded at start | [] |

## Software state at run time

| Parameter | Value |
|---|---|
| Kernel | 7.0.0-34-generic |
| Kernel cmdline | `BOOT_IMAGE=/boot/vmlinuz-7.0.0-34-generic root=UUID=2177f55b-d49b-4b63-b447-4b5d00d2e414 ro quiet splash amdgpu.ppfeaturemask=0xfff7ffff vt.handoff=7` |
| amdgpu ppfeaturemask | `0xfff7ffff` |
| ROCm | 7.2.4 |
| llama.cpp | `680a036` (2026-09-28) |
| llama.cpp build flags | `CMAKE_BUILD_TYPE=Release`, `GGML_CUDA=OFF`, `GGML_HIP=ON`, `GGML_VULKAN=OFF`, `GPU_TARGETS=gfx1201` |

## Measurement parameters

| Parameter | Value |
|---|---|
| Runs per cell | 3 — **separate OS processes** |
| Cooldown between runs | ≥ 60 s and until every GPU ≤ 50 °C (max 300 s) |
| Thermal guard | abort at junction ≥ 95 °C or VRAM ≥ 95 °C on any GPU used |
| Telemetry | amd-smi every 1.0 s, all GPUs used |

**Configurations used** — recorded verbatim:

| Config | Environment | llama-bench flags | GPUs |
|---|---|---|---|
| throughput | `HIP_VISIBLE_DEVICES=0,1,2,3`, `NCCL_PROTO=Simple`, `NCCL_P2P_DISABLE=1` | `-ngl 999 -sm layer -ts 1/1/1/1 --load-mode dio -p 512 -n 128` | [0, 1, 2, 3] |
| thermal | `HIP_VISIBLE_DEVICES=0,1,2,3`, `NCCL_PROTO=Simple`, `NCCL_P2P_DISABLE=1` | `-ngl 999 -sm layer -ts 1/1/1/1 --load-mode dio -p 4096 -n 1024` | [0, 1, 2, 3] |

## How to read this

**Spread matters more than the mean.** Decode throughput on this class of card is documented as *bimodal* — the card settles into one of two modes and stays there for the life of the process. That is why every run is a separate process, not `-r N`.

- **Spread below ~2 %** → bimodality did not appear; the mean is a sensible number.
- **Spread above ~10 %, two clusters** → report both modes, not their mean.

**Peak junction near 90 °C** means you are measuring case airflow, not the model.

**Low peak power on a valid result does not mean the GPU idled** — the compute window was shorter than the sampling period. With `-sm layer` each card also works only on its share of the layers in turn, so per-card power on a multi-GPU run is naturally far below the cap; look at the summed power.

**PCIe** is the bottleneck link of each GPU's path (root port → switch on the card → GPU), read under load. The GPU endpoint itself always reports x16.

