# Benchmark report — 20260920-134642

Config: **qwen3-32b-2gpu-layer** (TECHNIKA §9.4, three separate OS processes)

Run at: 2026-09-20 13:46:42 CEST

## Verdict

- ✅ **qwen3-32b / 2gpu-layer** (210): pp512 **950.3** tok/s, tg128 **27.76** tok/s
- Hardware: **2× R9700** (`HIP_VISIBLE_DEVICES=0,1`), `--split-mode layer --tensor-split 1/1`
- PCIe under load (CPU parent ports): **32.0 GT/s x8** on both
- AER/ECC unchanged
- Junction/power were **not** sampled (manual `llama-bench`, not `bench.py`)

## Comparison with 1-card baseline `20260919-225407` (same 210 W, same commit)

| | pp512 | tg128 |
|---|---:|---:|
| 1-card `-sm none` | 910.75 | 27.07 |
| 2-card layer | 950.28 | 27.76 |
| delta | +4.3% | +2.5% |

Decode stays in the same class. Splitting a model that already fits on one card does not buy useful decode speed.

## Runs

| run | pp512 | tg128 |
|---|---:|---:|
| 1 | 946.82 | 27.76 |
| 2 | 953.22 | 27.76 |
| 3 | 950.80 | 27.77 |
