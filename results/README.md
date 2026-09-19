# llm-bench-harness

A benchmark harness for local LLM inference that records **the conditions, not just the
numbers** — and tells you when a result is not trustworthy.

Built for a single-node AMD/ROCm box running `llama.cpp`. Driven either by you or by a
local agent.

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
| `llama.cpp` commit **and build flags** | Results are not comparable across builds |
| ROCm version, kernel, kernel cmdline | `iommu=pt` and friends live here |
| GPU power cap | 300 W and 210 W are two different measurements of the same model |
| PCIe link width/speed, read **under load** | Links downtrain at idle; add a 2nd GPU and the 1st drops x16→x8 |
| Peak junction / VRAM temp / power | If the card throttled, the number is about cooling, not the model |
| Env vars and flags, **verbatim** | `HIP_VISIBLE_DEVICES`, `GGML_*`, `NCCL_*` change everything |
| ECC + AER counters **before and after** | If they moved, the hardware misbehaved — discard, don't adjust |
| Other processes holding the GPU | A second consumer makes the run measure contention |
| CPU, RAM, BIOS version | Cheap to record, annoying to reconstruct later |

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

---

## Install

Requirements: Linux, ROCm with `amd-smi`, a built `llama-bench`, Python 3.10+.
No third-party Python packages. `lm-sensors` optional.

```bash
git clone https://github.com/<you>/llm-bench-harness.git
cd llm-bench-harness
sudo ln -s "$PWD/bench-model" /usr/local/bin/bench-model   # optional, for PATH
```

Everything hardware-specific is auto-detected. Overrides if you need them:

| Variable | Default |
|---|---|
| `RIG_GPU_INDEX` | `0` |
| `RIG_GPU_BDF` | auto-detected from `amd-smi` |
| `RIG_VRAM_MB` | auto-detected |
| `RIG_MODELS_DIR` | `~/models` |
| `LLAMA_CPP_DIR` | `~/llama.cpp` |

---

## Usage

### Measure a new model — one command

```bash
bench-model qwen3-32b
```

It checks the model fits in VRAM, **refuses to start** on a hot card or with non-zero
ECC/AER counters, runs both tests × 3 independent processes, writes a report, and appends
to the index. It ends with one machine-readable line:

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
└── raw.log        raw llama-bench output
results/all-benchmark-results.md   growing comparison table across ALL runs
```

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

More hardware-specific findings: [radeon-r9700-rocm-notes](https://github.com/<you>/radeon-r9700-rocm-notes).

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
- **Non-AMD hardware.** Telemetry goes through `amd-smi`. The structure would port to
  `nvidia-smi` without much trouble; I don't have the hardware to test it.

PRs and corrections welcome — especially if you have numbers that contradict mine.

## License

MIT
