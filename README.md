# llm-bench-harness

A benchmark harness for local LLM inference that records **the conditions, not just the
numbers** — and tells you when a result is not trustworthy.

Built for a single-node AMD/ROCm box running `llama.cpp` — one GPU or several (developed on
4× Radeon AI PRO R9700). Driven either by you or by a local agent.

```bash
bench-model --list-new       # which models have no result yet
bench-model qwen3-32b        # measure one, end to end
cat results/all-benchmark-results.md         # every model ever measured, side by side
```

---

## Why this exists

I kept benchmarking models by hand, and kept producing numbers I couldn't compare a week
later — because something different was set each time and I hadn't written down what.

The problem isn't that hand-run benchmarks are wrong. It's that they're **plausible**.
A number with no recorded conditions looks exactly as credible as one with them, right up
until you try to use it for a decision.

So this harness makes a measurement a *declaration*: you describe the matrix in JSON, and
it records everything needed to reproduce or invalidate the result.

## What it records with every single run

| | Why it has to be recorded |
|---|---|
| Model: architecture, params (total/active, experts), quant, size, native context | Read from the GGUF header — a "235B" dense and a "235B-A22B" MoE are different animals |
| `llama.cpp` commit **and build flags** | Results are not comparable across builds |
| llama-bench's **effective** parameters | Flash attention, batch/ubatch, KV type, split, load mode — read back from its JSON, not assumed |
| ROCm version, kernel, kernel cmdline | `amdgpu.ppfeaturemask` and friends live here |
| Per-GPU power cap and fan curve | 300 W and 250 W are two different measurements of the same model |
| PCIe link, **bottleneck of the path**, under load | The R9700's endpoint always says x16 (it talks to a switch on the card); the real slot link is x8 or x4 |
| Peak junction / VRAM temp / power **per GPU**, plus summed power | If a card throttled, the number is about cooling, not the model |
| Env vars and flags, **verbatim** | `HIP_VISIBLE_DEVICES`, `GGML_*`, `NCCL_*` change everything |
| ECC + AER counters **before and after**, every GPU and link | If they moved, the hardware misbehaved — discard, don't adjust |
| Other processes holding the GPUs, models loaded in llama-swap | A second consumer makes the run measure contention |
| CPU, CPU power profile, RAM, BIOS | Cheap to record, annoying to reconstruct later |

## What makes it different from `llama-bench` in a loop

**1. One OS process per run — never `-r N`.**
Decode throughput on some cards is *bimodal*: the card settles into one of two modes and
stays there for the lifetime of the process. `llama-bench -r 3` therefore averages three
samples **inside a single mode** and hands you tiny variance and false confidence. Only
separate processes can land in both modes.

**2. A thermal guard.**
Runs abort at junction ≥ 95 °C. The card itself only throttles at 110 °C — but GDDR6
degrades *permanently* above 100 °C, so waiting for the firmware is waiting too long.

**3. Two tests per model, not one.**
A short test (`-p 512 -n 128`) measures throughput. A long one (`-p 4096 -n 1024`)
measures thermals. You need both, for a reason worth spelling out:

> A fast MoE model finishes `-p 512 -n 128` in **under a second** — prefill ~0.08 s,
> decode ~0.85 s — and the rest of the wall time is loading weights from disk. Telemetry
> sampled every 2 s never sees the load. My first report claimed gpt-oss-20b peaked at
> **19 W**, which is nonsense. Under the long test the same model draws **288 W at 57 °C**.
>
> The harness now flags any run whose peak power is implausibly low, instead of letting
> me believe it.

**4. The report starts with a verdict.**

```
## Verdict
- ✅ qwen3-32b / throughput (300): pp512 1030.5 tok/s, tg128 28.0 tok/s
- ⚠️ gpt-oss-20b / throughput (300): pp512 6049.8 tok/s, tg128 149.9 tok/s
     — peak power 19 W — TELEMETRY UNRELIABLE: compute window shorter than the 2.0 s
     sampling period; do not draw thermal conclusions, use a longer test
✅ Hardware error counters (AER, ECC) unchanged
```

You shouldn't have to infer from a table whether a measurement was valid.

**5. Spread is reported as prominently as the mean** — because with bimodal hardware the
mean describes a state the card is never in.

**6. Multi-GPU aware.** GPUs come from the config's `HIP_VISIBLE_DEVICES`, mapped through
the PCI address to `amd-smi` (never assumed). Telemetry, the thermal guard (junction **and**
VRAM), error counters and power caps cover **every** GPU used; power caps set by a config
are restored afterwards. `bench-model` picks the smallest number of cards a model fits on.

**7. It validates the config before running anything.** The most expensive mistake it
catches: in `llama-bench` a comma means *sweep* — `-ts 1,1,1,1` is four tests with split
`1`, i.e. everything on GPU 0, which fails with a bare `failed to load model` on any model
bigger than one card. That cost two days of chasing a "llama-bench + dio multi-GPU bug"
that did not exist. The harness converts a comma `-ts` to the slash form and rejects every
other sweep, `-r`, a missing model file, and a config that would run on the integrated GPU.

---

## Install

Requirements: Linux, ROCm with `amd-smi` and `rocminfo`, a built `llama-bench`, Python 3.10+.
No third-party Python packages.

```bash
git clone https://github.com/michal-konik-human/llm-bench-harness.git
cd llm-bench-harness
sudo ln -s "$PWD/bench-model" /usr/local/bin/bench-model   # optional, for PATH
```

Everything hardware-specific is auto-detected. Overrides if you need them:

| Variable | Default |
|---|---|
| `RIG_GPU_INDEX` | `0` — GPU for configs without `HIP_VISIBLE_DEVICES` |
| `RIG_MODELS_DIR` | `~/models` |
| `LLAMA_CPP_DIR` | `~/llama.cpp` |

---

## Usage

### Measure a new model — one command

```bash
bench-model qwen3-32b
```

It sizes the model from all its shards, picks the smallest number of GPUs it fits on,
**refuses to start** on a hot card, non-zero ECC/AER counters, a GPU held by another
process or a model left loaded in llama-swap, runs both tests × 3 independent processes,
writes a detailed report, and appends to the shared table. It ends with one machine-readable line:

```
VERDICT: OK | WITH_CAVEAT | INVALID | SKIPPED
```

Adding a model is one line of JSON — or nothing at all, if you drop it in `~/models/<name>/`
and run `bench-model --list-new`.

### Custom matrices

```bash
./bench.py configs/power-cap.json --dry-run   # always check the plan first
./bench.py configs/power-cap.json
./bench.py --compare results/A results/B
```

| Config | Question it answers |
|---|---|
| `single-gpu.json` | Baseline on one GPU |
| `power-cap.json` | What does a lower power cap cost in tokens? |
| `context-scaling.json` | How does throughput fall off with context length? |
| `multi-gpu-EXAMPLE.json` | Template for multiple GPUs (read the comment first) |

`--dry-run` prints how many measurement processes the matrix implies. Check it — matrices
multiply, and it's easy to accidentally order three hours of work.

### Output

```
results/<timestamp>/
├── report.md      human-readable, verdict first
├── results.json   full machine-readable record
├── results.csv    spreadsheet
└── raw.log        raw llama-bench output (JSON + stderr) of every process
results/all-benchmark-results.md   growing comparison table across ALL runs
```

**Every run** — through `bench-model` or straight through `bench.py` — appends its rows
to `all-benchmark-results.md`: model (arch, params, quant, size), result (mean, spread, n),
hardware used (GPUs, split, PCIe per GPU), hardware settings (cap, fan curve, CPU profile),
software settings (FA, batch/ubatch, KV, load mode, llama.cpp, ROCm) and the verdict.
`./bench.py --reindex` rebuilds it from every `results.json`, keeping manual rows and notes.

`all-benchmark-results.md` is the point of the whole thing: one table where every model you've ever
measured sits beside the others **with its conditions**. Only compare rows with the same
power cap and the same `llama.cpp` commit — otherwise you're comparing conditions, not
models.

---

## Driving it with a local agent

`docs/AGENT-PLAYBOOK.md` is a system prompt for a **small** local model: numbered
procedure, decision tables instead of open-ended judgement, and an explicit list of
prohibitions. The agent doesn't evaluate the result — the harness does, and the agent
reads the verdict and reports it.

```bash
AGENT_MODEL=~/models/gpt-oss-20b/gpt-oss-20b-MXFP4.gguf ./agent-server.sh
./agent-server.sh --status      # must say "mode: CPU"
```

**The agent runs on the CPU, and that's deliberate.** The agent is itself a model and needs
VRAM. On one GPU it would take ~12 GB, leaving too little for an 18 GB model under test —
and every result would be depressed because the card is doing two jobs. Orchestration is a
dozen tool calls, so 5–10 tok/s from the CPU is plenty. **The measurement is the expensive
part; talking to the agent isn't.**

The harness also refuses to measure while anything else holds the GPU, so this mistake
fails loudly rather than quietly costing you 20 %.

---

## Example results

Radeon AI PRO R9700 (RDNA4, gfx1201, 32 GB), one card, stock 300 W cap, x16 Gen5,
`llama.cpp 987498f`, ROCm 7.2.4, 3 independent runs each:

| Model | prefill `pp512` | decode `tg128` | Spread |
|---|---|---|---|
| gpt-oss-20b MXFP4 (MoE, 11.3 GiB) | **6050** tok/s | **149.9** tok/s | 0.2 % |
| Qwen3-32B Q4_K_M (dense, 18.4 GiB) | **1030** tok/s | **27.98** tok/s | 0.0 % |

MoE decodes **5.4× faster** — not because it's a better model, but because it reads a
fraction of the weights per token, and decode is memory-bandwidth bound.

Under a long test, gpt-oss-20b reaches **288 W / 57 °C**, and the dense model pins **300 W**
with junction still climbing past 83 °C at the 60-second mark. Sustained real inference is
a heavier load than a synthetic stress test.

Spread across independent processes stayed under 0.25 %, so the documented bimodality
**did not appear** in this configuration. The method stays anyway — it costs nothing, and
the alternative is variance you never notice.

### Four cards, 128 GB VRAM — the big MoE models (30 Sep 2026)

4× R9700, `-sm layer`, 250 W cap per card, PCIe x8/x4/x8/x4 Gen5, `llama.cpp 680a036`,
`--load-mode dio`, llama.cpp default batch sizes, 3 independent runs each:

| Model | Size | prefill `pp512` | prefill `pp4096` | decode `tg128` | Spread |
|---|---|---:|---:|---:|---:|
| Qwen3-235B-A22B UD-Q3_K_XL (MoE 128/8) | 96.6 GiB | 619 | 583 | **32.7** | ≤ 1.0 % |
| DeepSeek-V4-Flash UD-IQ3_XXS (MoE 256/6) | 95.9 GiB | 517 | **1020** | **24.0** | ≤ 2.7 % |

DeepSeek processes a 4k prompt **twice as fast** as a 512-token one; Qwen3 gets slightly
slower. The prompt-batch size (`--ubatch-size`) is worth up to **+57 %** prefill for one
model and **−58 %** for another — see `results/all-benchmark-results.md`.

More hardware-specific findings: [radeon-r9700-rocm-notes](https://github.com/michal-konik-human/radeon-r9700-rocm-notes).

---

## Documentation

| Doc | For |
|---|---|
| [`docs/NEW-MODEL.md`](docs/NEW-MODEL.md) | The runbook: from download to report |
| [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) | What is actually being measured, and why |
| [`docs/MANUAL.md`](docs/MANUAL.md) | The same measurement by hand, every flag explained |
| [`docs/AGENT-PLAYBOOK.md`](docs/AGENT-PLAYBOOK.md) | System prompt for an orchestrating agent |

---

## What this does not measure yet

Stated plainly, because it matters:

- **TTFT (time to first token)** — arguably *the* metric for interactive agents.
  `llama-bench` measures throughput. This is the most important missing piece.
- **Answer quality** — use `lm-evaluation-harness` against `llama-server`.
- **Concurrent request behaviour** — matters as soon as more than one person uses the box.
- **Tensor-parallel / vLLM.** The harness drives `llama-bench` (layer split). vLLM TP on
  RDNA4 needs `NCCL_PROTO=Simple` and more — see the notes repo.
- **Non-AMD hardware.** Telemetry goes through `amd-smi`. The structure would port to
  `nvidia-smi` without much trouble; I don't have the hardware to test it.

PRs and corrections welcome — especially if you have numbers that contradict mine.

## License

MIT
