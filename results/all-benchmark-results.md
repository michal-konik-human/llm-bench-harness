# All benchmark results — every model measured, with its conditions

Appended automatically by `bench-model` (or rebuilt with `--reindex`). Newest at the bottom.

`spread` is the difference between the best and worst independent run.
Above 10 % indicates bimodality — the mean in this table is then misleading
and you should open that run's `report.md`.

`GPUs` is what the run actually used (`HIP_VISIBLE_DEVICES` + split), not the inventory of the machine. `PCIe` is the link sampled under load.

**Only compare rows with the same power cap, the same llama.cpp commit, and the same GPU/split.** Otherwise you are comparing conditions, not models.

| run_id | model | config | test | tok/s | spread | junction | power | cap | GPUs | split | PCIe | llama.cpp | ROCm |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|---|---|---|
| 20260915-164813 | gpt-oss-20b | throughput | pp512 | 6038.74 | 0.6% | 30 °C | 19 W | 300 | 1×R9700 (0) | none | x16 32.0 GT/s PCIe | `987498f` | 7.2.4 |
| 20260915-164813 | gpt-oss-20b | throughput | tg128 | 149.72 | 0.2% | 30 °C | 19 W | 300 | 1×R9700 (0) | none | x16 32.0 GT/s PCIe | `987498f` | 7.2.4 |
| 20260919-225407 | qwen3-32b | throughput | pp512 | 910.75 | 0.4% | 47 °C | 210 W | 210 | 1×R9700 (0) | none | x16 32.0 GT/s PCIe | `987498f` | 7.2.4 |
| 20260919-225407 | qwen3-32b | throughput | tg128 | 27.07 | 0.2% | 47 °C | 210 W | 210 | 1×R9700 (0) | none | x16 32.0 GT/s PCIe | `987498f` | 7.2.4 |
| 20260919-225407 | qwen3-32b | thermal | pp4096 | 844.32 | 0.5% | 66 °C | 214 W | 210 | 1×R9700 (0) | none | x16 32.0 GT/s PCIe | `987498f` | 7.2.4 |
| 20260919-225407 | qwen3-32b | thermal | tg1024 | 26.70 | 0.7% | 66 °C | 214 W | 210 | 1×R9700 (0) | none | x16 32.0 GT/s PCIe | `987498f` | 7.2.4 |
| 20260919-231520 | bielik-11b | throughput | pp512 | 2888.34 | 0.6% | 40 °C | 210 W | 210 | 1×R9700 (0) | none | x16 32.0 GT/s PCIe | `987498f` | 7.2.4 |
| 20260919-231520 | bielik-11b | throughput | tg128 | 45.08 | 0.0% | 40 °C | 210 W | 210 | 1×R9700 (0) | none | x16 32.0 GT/s PCIe | `987498f` | 7.2.4 |
| 20260919-231520 | bielik-11b | thermal | pp4096 | 2694.49 | 0.3% | 57 °C | 211 W | 210 | 1×R9700 (0) | none | x16 32.0 GT/s PCIe | `987498f` | 7.2.4 |
| 20260919-231520 | bielik-11b | thermal | tg1024 | 44.75 | 0.1% | 57 °C | 211 W | 210 | 1×R9700 (0) | none | x16 32.0 GT/s PCIe | `987498f` | 7.2.4 |
| 20260920-134642 | qwen3-32b | 2gpu-layer | pp512 | 950.28 | 0.7% | — | — | 210 | 2×R9700 (0,1) | layer 1/1 | x8 32.0 GT/s PCIe | `987498f` | 7.2.4 |
| 20260920-134642 | qwen3-32b | 2gpu-layer | tg128 | 27.76 | 0.0% | — | — | 210 | 2×R9700 (0,1) | layer 1/1 | x8 32.0 GT/s PCIe | `987498f` | 7.2.4 |
| 20260920-1525 | qwen3-next-80b | 2gpu-manual | pp512 | 2240.52 | 4.2%* | — | — | 210 | 2×R9700 (0,1) | layer auto | x8 32.0 GT/s PCIe | `987498f` | 7.2.4 |
| 20260920-1525 | qwen3-next-80b | 2gpu-manual | tg128 | 71.23 | 0.4%* | — | — | 210 | 2×R9700 (0,1) | layer auto | x8 32.0 GT/s PCIe | `987498f` | 7.2.4 |
| 20260920-1540 | gpt-oss-20b | 2gpu-manual | pp512 | 5945.78 | 1.6%* | — | — | 210 | 2×R9700 (0,1) | layer auto | x8 32.0 GT/s PCIe | `987498f` | 7.2.4 |
| 20260920-1540 | gpt-oss-20b | 2gpu-manual | tg128 | 145.28 | 0.3%* | — | — | 210 | 2×R9700 (0,1) | layer auto | x8 32.0 GT/s PCIe | `987498f` | 7.2.4 |

\* Rows marked `2gpu-manual` came from a direct `llama-bench` call, not from `bench-model`.
They use llama-bench's own repetitions **inside one process**, so the figure after ± is a
standard deviation, not the best-to-worst spread of independent processes. Treat them as
indicative; re-run through `bench-model` before quoting them anywhere.

## Models that did NOT produce a result

| date | model | GPUs | what happened |
|---|---|---|---|
| 2026-09-19/20 | **gpt-oss-120b** MXFP4 (59.0 GiB) | 2×R9700 | **Loads, but cannot compute on GPU.** Weights go resident (≈30 GiB per card), then GPU activity sits at 4-6 % and power at ~15 W while one CPU core runs flat out. No tokens produced in >4 min. Four attempts, incl. rebalanced `-ts 0.48/0.52` and batch reduced to 128. Cause: 59.0 GiB of weights in 63.7 GiB of VRAM leaves ~4.7 GiB for KV-cache and compute buffers across both cards, which is not enough — the compute buffer falls back to host memory. Proven not to be a 2-GPU or MoE problem: `gpt-oss-20b` (same family, same MXFP4, same `-sm layer`) runs at 145 tok/s on the same two cards, and `qwen3-next-80b` (46.6 GiB) runs at 71 tok/s. **Conclusion: this model needs a third card, not a different flag.** |
