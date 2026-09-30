# All benchmark results — every model measured, with its conditions

Appended automatically by `bench.py` after every run (so also by `bench-model`).
Rebuild from all `results/*/results.json` with `./bench.py --reindex` — rows without a
results directory (manual measurements) and the notes below the table are preserved.
Newest at the bottom. Detailed report of each run: `results/<run_id>/report.md`.

How to read the columns:

- `tok/s` is the mean of `n` **separate processes**; `spread` is best-vs-worst. Above 10 %
  indicates bimodality — the mean is then misleading; open the run's `report.md`.
- `arch` / `params` / `quant` / `GiB` describe the model file: `235B-A22B (MoE 128/8)` =
  235 B total, 22 B active, 128 experts with 8 used per token.
- `GPUs` is what the run actually used (`HIP_VISIBLE_DEVICES`), `split` the llama.cpp split.
  `PCIe` is each GPU's **bottleneck** link on the path to the CPU, sampled under load.
- `cap W` = power cap per card, `fan curve` = amdgpu overdrive curve (`°C:%`, `default` =
  firmware curve). `peak junction` / `peak VRAM` = hottest reading on ANY used GPU.
  `GPU W peak / mean` = sum over the used GPUs (peak sample / mean over the runs).
- `FA` flash attention, `batch/ubatch`, `KV` cache type, `load` model load mode, as
  llama-bench itself reported them.

**Only compare rows with the same power cap, llama.cpp commit, and GPUs/split.**
Otherwise you are comparing conditions, not models. Rows from before 30 Sep 2026
(schema 1) lack some columns (`—`) and their `PCIe` is the card's endpoint link (always x16).

| run_id | model | config | test | tok/s | spread | n | arch | params | quant | GiB | GPUs | split | PCIe (under load) | cap W | fan curve | peak junction | peak VRAM | GPU W peak / mean | FA | batch/ubatch | KV | load | CPU profile | llama.cpp | ROCm | verdict |
|---|---|---|---|---:|---:|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260915-164813 | gpt-oss-20b | throughput | pp512 | 6038.74 | 0.6% | 3 | gpt-oss | 32x2.4B (MoE 32/4) | MXFP4 | 11.3 | 1×R9700 (0) | none | x16 (endpoint) | 300 | — | 30 °C | 30 °C | 19 | — | — | — | — | — | `987498f` | 7.2.4 | WITH_CAVEAT |
| 20260915-164813 | gpt-oss-20b | throughput | tg128 | 149.72 | 0.2% | 3 | gpt-oss | 32x2.4B (MoE 32/4) | MXFP4 | 11.3 | 1×R9700 (0) | none | x16 (endpoint) | 300 | — | 30 °C | 30 °C | 19 | — | — | — | — | — | `987498f` | 7.2.4 | WITH_CAVEAT |
| 20260919-225407 | qwen3-32b | throughput | pp512 | 910.75 | 0.4% | 3 | qwen3 | 32B (dense) | Q4_K_M | 18.4 | 1×R9700 (0) | none | x16 (endpoint) | 210 | — | 47 °C | 42 °C | 210 | — | — | — | — | — | `987498f` | 7.2.4 | OK |
| 20260919-225407 | qwen3-32b | throughput | tg128 | 27.07 | 0.2% | 3 | qwen3 | 32B (dense) | Q4_K_M | 18.4 | 1×R9700 (0) | none | x16 (endpoint) | 210 | — | 47 °C | 42 °C | 210 | — | — | — | — | — | `987498f` | 7.2.4 | OK |
| 20260919-225407 | qwen3-32b | thermal | pp4096 | 844.32 | 0.5% | 3 | qwen3 | 32B (dense) | Q4_K_M | 18.4 | 1×R9700 (0) | none | x16 (endpoint) | 210 | — | 66 °C | 68 °C | 214 | — | — | — | — | — | `987498f` | 7.2.4 | OK |
| 20260919-225407 | qwen3-32b | thermal | tg1024 | 26.70 | 0.7% | 3 | qwen3 | 32B (dense) | Q4_K_M | 18.4 | 1×R9700 (0) | none | x16 (endpoint) | 210 | — | 66 °C | 68 °C | 214 | — | — | — | — | — | `987498f` | 7.2.4 | OK |
| 20260919-231520 | bielik-11b | throughput | pp512 | 2888.34 | 0.6% | 3 | llama | 11B (dense) | Q8_0 | 11.1 | 1×R9700 (0) | none | x16 (endpoint) | 210 | — | 40 °C | 38 °C | 210 | — | — | — | — | — | `987498f` | 7.2.4 | OK |
| 20260919-231520 | bielik-11b | throughput | tg128 | 45.08 | 0.0% | 3 | llama | 11B (dense) | Q8_0 | 11.1 | 1×R9700 (0) | none | x16 (endpoint) | 210 | — | 40 °C | 38 °C | 210 | — | — | — | — | — | `987498f` | 7.2.4 | OK |
| 20260919-231520 | bielik-11b | thermal | pp4096 | 2694.49 | 0.3% | 3 | llama | 11B (dense) | Q8_0 | 11.1 | 1×R9700 (0) | none | x16 (endpoint) | 210 | — | 57 °C | 58 °C | 211 | — | — | — | — | — | `987498f` | 7.2.4 | OK |
| 20260919-231520 | bielik-11b | thermal | tg1024 | 44.75 | 0.1% | 3 | llama | 11B (dense) | Q8_0 | 11.1 | 1×R9700 (0) | none | x16 (endpoint) | 210 | — | 57 °C | 58 °C | 211 | — | — | — | — | — | `987498f` | 7.2.4 | OK |
| 20260920-134642 | qwen3-32b | 2gpu-layer | pp512 | 950.28 | 0.7% | 3 | qwen3 | 32B (dense) | Q4_K_M | 18.4 | 2×R9700 (0,1) | layer 1/1 | x8 (endpoint) | 210 | — | — | — | — | — | — | — | — | — | `987498f` | 7.2.4 | WITH_CAVEAT |
| 20260920-134642 | qwen3-32b | 2gpu-layer | tg128 | 27.76 | 0.0% | 3 | qwen3 | 32B (dense) | Q4_K_M | 18.4 | 2×R9700 (0,1) | layer 1/1 | x8 (endpoint) | 210 | — | — | — | — | — | — | — | — | — | `987498f` | 7.2.4 | WITH_CAVEAT |
| 20260920-1525 | qwen3-next-80b | 2gpu-manual | pp512 | 2240.52 | 4.2%* | — | — | — | — | — | 2×R9700 (0,1) | layer auto | x8 32.0 GT/s PCIe | 210 | — | — | — | — | — | — | — | — | — | `987498f` | 7.2.4 | — |
| 20260920-1525 | qwen3-next-80b | 2gpu-manual | tg128 | 71.23 | 0.4%* | — | — | — | — | — | 2×R9700 (0,1) | layer auto | x8 32.0 GT/s PCIe | 210 | — | — | — | — | — | — | — | — | — | `987498f` | 7.2.4 | — |
| 20260920-1540 | gpt-oss-20b | 2gpu-manual | pp512 | 5945.78 | 1.6%* | — | — | — | — | — | 2×R9700 (0,1) | layer auto | x8 32.0 GT/s PCIe | 210 | — | — | — | — | — | — | — | — | — | `987498f` | 7.2.4 | — |
| 20260920-1540 | gpt-oss-20b | 2gpu-manual | tg128 | 145.28 | 0.3%* | — | — | — | — | — | 2×R9700 (0,1) | layer auto | x8 32.0 GT/s PCIe | 210 | — | — | — | — | — | — | — | — | — | `987498f` | 7.2.4 | — |
| 20260928-2300 | gpt-oss-120b | 3gpu-manual | pp512 | 194.13 | 1.1%* | — | — | — | — | — | 3×R9700 (0,1,2) | layer 1/1/1 | x8 32.0 GT/s PCIe | 300 | — | not sampled | — | not sampled | — | — | — | — | — | `680a036` | 7.2.4 | — |
| 20260928-2300 | gpt-oss-120b | 3gpu-manual | tg128 | 94.30 | 4.3%* | — | — | — | — | — | 3×R9700 (0,1,2) | layer 1/1/1 | x8 32.0 GT/s PCIe | 300 | — | not sampled | — | not sampled | — | — | — | — | — | `680a036` | 7.2.4 | — |
| 20260929-0930 | laguna-s-2.1 | 3gpu-manual | pp512 | 125.27 | 0.8%* | — | — | — | — | — | 3×R9700 (0,1,2) | layer 1/1/1 | x8 32.0 GT/s PCIe | 300 | — | 51 °C† | — | 155† | — | — | — | — | — | `680a036` | 7.2.4 | — |
| 20260929-0930 | laguna-s-2.1 | 3gpu-manual | tg128 | 49.07 | 0.2%* | — | — | — | — | — | 3×R9700 (0,1,2) | layer 1/1/1 | x8 32.0 GT/s PCIe | 300 | — | 51 °C† | — | 155† | — | — | — | — | — | `680a036` | 7.2.4 | — |
| 20260929-0930 | laguna-s-2.1 | 3gpu-manual | thermal (pp4096/tg1024) | 128.4 / 44.1 | n/a (1 run) | — | — | — | — | — | 3×R9700 (0,1,2) | layer 1/1/1 | x8 32.0 GT/s PCIe | 300 | — | 51 °C† | — | 155† | — | — | — | — | — | `680a036` | 7.2.4 | — |
| 20260929-0945 | qwen3-235b-a22b | 3gpu-manual | pp512 | 41.70 | 2.4%* | — | — | — | — | — | 3×R9700 (0,1,2) | layer 1/1/1 | x8 32.0 GT/s PCIe | 300 | — | 63 °C† | — | 172† | — | — | — | — | — | `680a036` | 7.2.4 | — |
| 20260929-0945 | qwen3-235b-a22b | 3gpu-manual | tg128 | 31.40 | 0.0%* | — | — | — | — | — | 3×R9700 (0,1,2) | layer 1/1/1 | x8 32.0 GT/s PCIe | 300 | — | 63 °C† | — | 172† | — | — | — | — | — | `680a036` | 7.2.4 | — |
| 20260929-0945 | qwen3-235b-a22b | 3gpu-manual | thermal (pp4096/tg1024) | 50.7 / 31.3 | n/a (1 run) | — | — | — | — | — | 3×R9700 (0,1,2) | layer 1/1/1 | x8 32.0 GT/s PCIe | 300 | — | 63 °C† | — | 172† | — | — | — | — | — | `680a036` | 7.2.4 | — |
| 20260930-121339 | qwen3-235b-a22b-q3 | throughput | pp512 | 619.39 | 0.3% | 3 | qwen3moe | 235B-A22B (MoE 128/8) | UD-Q3_K_XL | 96.6 | 4×R9700 (0,1,2,3) | layer 1/1/1/1 | x16 (endpoint) | 250 | — | 53 °C | 38 °C | 244 | — | — | — | dio | — | `680a036` | 7.2.4 | OK |
| 20260930-121339 | qwen3-235b-a22b-q3 | throughput | tg128 | 32.74 | 0.1% | 3 | qwen3moe | 235B-A22B (MoE 128/8) | UD-Q3_K_XL | 96.6 | 4×R9700 (0,1,2,3) | layer 1/1/1/1 | x16 (endpoint) | 250 | — | 53 °C | 38 °C | 244 | — | — | — | dio | — | `680a036` | 7.2.4 | OK |
| 20260930-121339 | qwen3-235b-a22b-q3 | thermal | pp4096 | 583.27 | 1.0% | 3 | qwen3moe | 235B-A22B (MoE 128/8) | UD-Q3_K_XL | 96.6 | 4×R9700 (0,1,2,3) | layer 1/1/1/1 | x16 (endpoint) | 250 | — | 60 °C | 48 °C | 294 | — | — | — | dio | — | `680a036` | 7.2.4 | OK |
| 20260930-121339 | qwen3-235b-a22b-q3 | thermal | tg1024 | 32.57 | 0.1% | 3 | qwen3moe | 235B-A22B (MoE 128/8) | UD-Q3_K_XL | 96.6 | 4×R9700 (0,1,2,3) | layer 1/1/1/1 | x16 (endpoint) | 250 | — | 60 °C | 48 °C | 294 | — | — | — | dio | — | `680a036` | 7.2.4 | OK |
| 20260930-122320 | deepseek-v4-flash | throughput | pp512 | 517.42 | 1.3% | 3 | deepseek4 | 256x8.4B (MoE 256/6) | UD-IQ3_XXS | 95.9 | 4×R9700 (0,1,2,3) | layer 1/1/1/1 | x16 (endpoint) | 250 | — | 55 °C | 46 °C | 103 | — | — | — | dio | — | `680a036` | 7.2.4 | OK |
| 20260930-122320 | deepseek-v4-flash | throughput | tg128 | 23.97 | 1.3% | 3 | deepseek4 | 256x8.4B (MoE 256/6) | UD-IQ3_XXS | 95.9 | 4×R9700 (0,1,2,3) | layer 1/1/1/1 | x16 (endpoint) | 250 | — | 55 °C | 46 °C | 103 | — | — | — | dio | — | `680a036` | 7.2.4 | OK |
| 20260930-122320 | deepseek-v4-flash | thermal | pp4096 | 1019.51 | 0.5% | 3 | deepseek4 | 256x8.4B (MoE 256/6) | UD-IQ3_XXS | 95.9 | 4×R9700 (0,1,2,3) | layer 1/1/1/1 | x16 (endpoint) | 250 | — | 62 °C | 46 °C | 256 | — | — | — | dio | — | `680a036` | 7.2.4 | OK |
| 20260930-122320 | deepseek-v4-flash | thermal | tg1024 | 24.07 | 2.7% | 3 | deepseek4 | 256x8.4B (MoE 256/6) | UD-IQ3_XXS | 95.9 | 4×R9700 (0,1,2,3) | layer 1/1/1/1 | x16 (endpoint) | 250 | — | 62 °C | 46 °C | 256 | — | — | — | dio | — | `680a036` | 7.2.4 | OK |
| 20260930-125802 | gpt-oss-20b | throughput | pp512 | 5726.01 | 0.7% | 3 | gpt-oss | 32x2.4B (MoE 32/4) | MXFP4 | 11.3 | 1×R9700 (0) | none | x8 Gen5 | 250 | 55:20 65:30 75:55 82:80 90:100 | 52 °C | 42 °C | 246 / 105 | on | 2048/512 | f16/f16 | auto | balanced | `680a036` | 7.2.4 | OK |
| 20260930-125802 | gpt-oss-20b | throughput | tg128 | 149.53 | 0.1% | 3 | gpt-oss | 32x2.4B (MoE 32/4) | MXFP4 | 11.3 | 1×R9700 (0) | none | x8 Gen5 | 250 | 55:20 65:30 75:55 82:80 90:100 | 52 °C | 42 °C | 246 / 105 | on | 2048/512 | f16/f16 | auto | balanced | `680a036` | 7.2.4 | OK |
| 20260930-125943 | qwen3-next-80b | throughput | pp512 | 2384.99 | 0.2% | 3 | qwen3next | 80B-A3B (MoE 512/10) | Q4_1 | 46.6 | 2×R9700 (0,1) | layer 1.00/1.00 | x8/x4 Gen5 | 250 | mixed | 52 °C | 40 °C | 247 / 71 | on | 2048/512 | f16/f16 | dio | balanced | `680a036` | 7.2.4 | OK |
| 20260930-125943 | qwen3-next-80b | throughput | tg128 | 71.59 | 0.4% | 3 | qwen3next | 80B-A3B (MoE 512/10) | Q4_1 | 46.6 | 2×R9700 (0,1) | layer 1.00/1.00 | x8/x4 Gen5 | 250 | mixed | 52 °C | 40 °C | 247 / 71 | on | 2048/512 | f16/f16 | dio | balanced | `680a036` | 7.2.4 | OK |
| 20260930-221230 | deepseek-v4-flash-0731 | throughput | pp512 | 559.75 | 1.0% | 3 | deepseek4 | 256x8.4B (MoE 256/6) | UD-IQ3_XXS | 97.0 | 4×R9700 (0,1,2,3) | layer 1.00/1.00/1.00/1.00 | x8/x4/x8/x4 Gen5 | 250 | mixed | 62 °C | 38 °C | 514 / 128 | -1 | 2048/512 | f16/f16 | dio | balanced | `680a036` | 7.2.4 | OK |
| 20260930-221230 | deepseek-v4-flash-0731 | throughput | tg128 | 24.71 | 8.3% | 3 | deepseek4 | 256x8.4B (MoE 256/6) | UD-IQ3_XXS | 97.0 | 4×R9700 (0,1,2,3) | layer 1.00/1.00/1.00/1.00 | x8/x4/x8/x4 Gen5 | 250 | mixed | 62 °C | 38 °C | 514 / 128 | -1 | 2048/512 | f16/f16 | dio | balanced | `680a036` | 7.2.4 | OK |
| 20260930-221230 | deepseek-v4-flash-0731 | thermal | pp4096 | 1089.20 | 0.8% | 3 | deepseek4 | 256x8.4B (MoE 256/6) | UD-IQ3_XXS | 97.0 | 4×R9700 (0,1,2,3) | layer 1.00/1.00/1.00/1.00 | x8/x4/x8/x4 Gen5 | 250 | mixed | 75 °C | 64 °C | 943 / 288 | -1 | 2048/512 | f16/f16 | dio | balanced | `680a036` | 7.2.4 | OK |
| 20260930-221230 | deepseek-v4-flash-0731 | thermal | tg1024 | 23.85 | 0.1% | 3 | deepseek4 | 256x8.4B (MoE 256/6) | UD-IQ3_XXS | 97.0 | 4×R9700 (0,1,2,3) | layer 1.00/1.00/1.00/1.00 | x8/x4/x8/x4 Gen5 | 250 | mixed | 75 °C | 64 °C | 943 / 288 | -1 | 2048/512 | f16/f16 | dio | balanced | `680a036` | 7.2.4 | OK |

\* Rows marked `2gpu-manual` / `3gpu-manual` came from a direct `llama-cli`/`llama-bench` call,
not from `bench-model`. They use separate OS processes (three runs, `pp512`/`tg128` prompt
lengths) but no `bench.py` telemetry sampling, so junction/power are not recorded (unless
marked † — see below). Treat them as indicative; re-run through `bench.py` once its
`--load-mode` bug (see below) is fixed upstream, before quoting them anywhere more formal.

† For `laguna-s-2.1` and `qwen3-235b-a22b`, junction/power were manually sampled via
`amd-smi metric --temperature --power` at ~10 s intervals during the separate thermal run
(one process, `-p 4096 -n 1024`), not continuously instrumented like `bench.py`'s built-in
sampling — treat as a coarse peak, not an exact maximum. Both stayed far below the 95 °C
guard and the 300 W cap.

⚠️ **`qwen3-235b-a22b` is the tightest VRAM fit run on this rig to date.** UD-Q2_K_XL
weights alone are ~83 GiB of the 96 GiB total across 3 cards, leaving only ~10-13 GiB for
KV-cache and compute buffers. It loaded and ran without OOM at the modest context implied
by `-p 512/4096` (≤ ~5.1K tokens), but a much longer context has not been tested and may
not fit — see the `--ctx-size 4096` note in `llama-swap`'s config for this model.

## Models that did NOT produce a result

| date | model | GPUs | what happened |
|---|---|---|---|
| 2026-09-19/20 | **gpt-oss-120b** MXFP4 (59.0 GiB) | 2×R9700 | **RESOLVED 2026-09-28 — the original conclusion below was wrong.** Loads, but doesn't compute: GPU activity sits at 4-6%, power ~15W, one CPU core flat out, zero tokens for 4+ minutes. This is **not** a VRAM/KV-cache capacity problem — it's a known ROCm HSA-runtime bug registering very large (≳50 GiB) mmap'd host buffers (llama.cpp issue [#19482](https://github.com/ggml-org/llama.cpp/issues/19482), confirmed by multiple independent users including another 4×R9700 report). The exact same hang reproduced on 3 cards after adding a third R9700 — proving the "needs a third card" theory false. Fixed with `--load-mode dio` (this llama.cpp's replacement for the older `--no-mmap` flag) — see the `3gpu-manual` rows above and `radeon-r9700-rocm-notes/findings/rocm-mmap-load-hang.md` for the full writeup. Original (now superseded) analysis: *"Loads, but cannot compute on GPU. Weights go resident (≈30 GiB per card), then GPU activity sits at 4-6% and power at ~15W while one CPU core runs flat out. No tokens produced in >4 min. Four attempts, incl. rebalanced `-ts 0.48/0.52` and batch reduced to 128. Cause: 59.0 GiB of weights in 63.7 GiB of VRAM leaves ~4.7 GiB for KV-cache and compute buffers across both cards, which is not enough — the compute buffer falls back to host memory. Conclusion: this model needs a third card, not a different flag."* — the "not a different flag" claim was the part that was wrong. |

## Known bug: `bench.py`/`llama-bench` + `--load-mode dio` + multi-GPU

> ⚠️ **CORRECTION, 30 Sep 2026 — this bug does not exist.** The failing configs wrote the
> tensor split as `-ts 1,1,1`. In `llama-bench` the separator is `/`; a comma means "run the
> test once per value", so `-ts 1,1,1` was three tests with split `1` = the whole model on
> GPU 0, which cannot load a model bigger than one card. With `-ts 1/1/1/1` the same model,
> same `--load-mode dio`, same 4 GPUs loads in ~30 s (diagnostic 30 Sep 12:12). `bench.py` now
> converts a comma `-ts` automatically and rejects other sweeps. Consequence for the table:
> the `3gpu-manual` rows (gpt-oss-120b, laguna-s-2.1, qwen3-235b-a22b) came from a manual
> `llama-cli` workaround that **understated prompt processing heavily** (qwen3-235b: 41.7 tok/s
> manual on 3 cards vs 619 measured on 4) — treat their `pp` values as unreliable; their `tg`
> values are in line with later measurements. The original text below is kept as a record.


`llama-bench --load-mode dio` works fine single-GPU (confirmed on `gpt-oss-20b`), but fails
immediately with `error: failed to load model` on this exact `gpt-oss-120b` file as soon as
more than one GPU is visible (`HIP_VISIBLE_DEVICES=0,1` or `0,1,2`), regardless of
`--tensor-split`. `llama-cli` with the identical flags works correctly. Root cause not yet
isolated — likely a bug specific to `llama-bench`'s own load path, not `llama_model_load()`
itself. Until fixed, use `llama-cli` directly for multi-GPU + `dio` + large-model
measurements (see the `3gpu-manual` rows above) rather than `bench.py`.

**Confirmed again 2026-09-29** on two more, unrelated models (`laguna-s-2.1`,
`qwen3-235b-a22b`) — same instant `error: failed to load model`, same 3-GPU + `dio`
combination, still nothing when only one GPU is visible. This is not specific to
`gpt-oss-120b` or its architecture; it looks like a general property of `llama-bench` +
`--load-mode dio` + `HIP_VISIBLE_DEVICES` with more than one entry, independent of which
model is being loaded. Still not reported upstream as of this date — worth filing.
