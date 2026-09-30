# Methodology — what is actually being measured, and why

Read this once, slowly. Then work from [`NEW-MODEL.md`](NEW-MODEL.md).

The point of this document is to make the numbers in a report *mean something*.

---

## Two phases with opposite hardware demands

A language model works in two phases that stress completely different parts of the
machine, which is why they are measured separately.

**Prefill** (`pp512` = prefill 512 tokens) — the model reads your input: the question,
the document, the context. It can process all tokens **in parallel**, so it saturates
the GPU's compute units. It is **compute bound**. Numbers are large — thousands of
tokens/second.

**Decode** (`tg128` = generate 128 tokens) — the model writes the answer, one token at a
time. Each token depends on the previous one, so this **cannot be parallelised**. For
every token the GPU must read the weights out of memory. It is **memory-bandwidth bound**,
not compute bound. Numbers are small — tens of tokens/second.

**Why this matters in practice:** a long document on the input stresses prefill; a long
answer stresses decode. If your application reads 30 pages and replies in two sentences,
prefill is what you're paying for. That's why `configs/context-scaling.json` exists
separately.

---

## Why MoE wins so clearly on decode

Measured on one 32 GB card:

| Model | Prefill | Decode |
|---|---|---|
| gpt-oss-20b (MoE, 20.9 B params) | ~6050 tok/s | ~150 tok/s |
| Qwen3-32B (dense, 32.8 B params) | ~1030 tok/s | ~28 tok/s |

The **5.4× difference on decode** isn't because one model is "better engineered". It
follows from the architecture:

- **Dense** — for **every** token the model reads **all** the weights. Qwen3-32B is
  18.4 GiB that must move through the memory controller per token.
- **MoE (mixture of experts)** — the model has many experts but activates only a
  **fraction** of them per token. It might read 3–4 GB instead of 12.

Since decode is bandwidth bound, and MoE reads several times less memory per token, it is
several times faster. The corollary: **MoE needs memory *capacity* more than memory
*bandwidth*** — and capacity is comparatively cheap.

---

## Why every run is a separate OS process

Decode throughput on this class of card is documented as **bimodal**: the card settles
into one of two operating modes and **stays there for the lifetime of the process**.

The consequence: `llama-bench -r 3` runs three measurements **in one process**, i.e.
inside one mode. You get three nearly identical numbers and a false sense of precision.
Only **three separate processes** can land in different modes.

So the harness spawns one process per run and reports the **spread** — which is why
spread is more important than the mean here.

> **Result from my own runs:** across 3 independent processes the spread was **under
> 0.25 %** for both models. The bimodality **did not appear** in this configuration.
> That doesn't mean it isn't real — it may depend on model, driver or workload. The
> method stays, because it costs nothing and the alternative is variance you never notice.

---

## Why the power cap is a condition, not a detail

The card here has a stock limit of **300 W** and a firmware **floor of 210 W** — 200 W is
simply not settable, `amd-smi` rejects it. With amdgpu *overdrive* enabled
(`amdgpu.ppfeaturemask` bit 0x4000, needed for fan-curve control) the ceiling rises to
**330 W**. The machine's standing setting since 30 Sep 2026 is **250 W** per card
(`gpu-tuning.service`); a config's `power_cap_w` overrides it for that config only and is
restored afterwards.

Measurement showed that **a dense model holds the card at 300 W continuously**, with
junction temperature climbing the whole time: 64 °C → 73 °C → 79 °C → 83 °C over 60
seconds, and **still rising** when the run ended. A synthetic FMA stress test only drew
279 W in bursts — so **real inference is a heavier load than the stress test** used to
validate the card.

Conclusion: the same model at 300 W and at 210 W are **two different measurements**. If
you don't record which was which, the results are not comparable. The harness records the
cap with every number, and `configs/power-cap.json` measures the difference directly.

---

## Why PCIe link width is in the report — and where to read it

With one card in the top slot it gets **x16 Gen5**. Add a second card and the board splits
the lanes automatically (x8 + x8); cards on M.2 adapters get **x4**. On the 4-card build
the measured links are **x8 / x4 / x8 / x4, all Gen5**.

Link width barely affects decode (weights are loaded into VRAM once and stay there), but
it **does** affect model load time, prefill when anything lives in host RAM, and every
hand-off between cards in a multi-GPU split. So it is recorded with every result — the
cost of x16 → x8 → x4 is **measured rather than assumed**.

**Trap 1 — the endpoint lies.** A Radeon AI PRO R9700 carries its **own PCIe switch**:
root port → switch upstream port → switch downstream port → GPU. The GPU endpoint always
reports **x16** at full speed, because it talks to that switch. The slot link is further
up. The harness walks the whole path in sysfs and records the **bottleneck** (narrowest,
slowest link) for each GPU. Results from before 30 Sep 2026 recorded the endpoint and say
x16 for every card; the table marks them `(endpoint)`.

**Trap 2 — idle downtraining.** At idle the link downtrains to save power, so the reading
is only meaningful **under load**; the harness samples it while each GPU is busy and marks
an idle reading `(idle)`.

---

## Multiple GPUs — what changes in the measurement

- **Which cards:** taken from the config's `HIP_VISIBLE_DEVICES` and mapped HIP index →
  PCI address → `amd-smi` index (HIP numbering comes from `rocminfo`; nothing guarantees
  it equals `amd-smi`'s). The integrated GPU is refused — it would produce a plausible
  number ~10× too low.
- **Everything covers every card:** telemetry, the thermal guard (junction **and** VRAM),
  ECC/AER before and after (including the links above each card), exclusivity, power caps.
  The old harness watched only GPU 0 — on a 4-card run a hot GPU 3 could not stop it.
- **Per-card power is low, and that is correct.** With `-sm layer` the model is cut into
  consecutive slices; for each token the cards work **in turn**, so each is busy a fraction
  of the time. A 4-card decode draws far less than 4 × the cap. Read the **summed** power.
- **Fewer cards is faster for decode.** Each extra card adds a hand-off per token.
  `bench-model` picks the smallest number of cards the model fits on (weights ≤ 75 % of
  their VRAM).

---

## The tensor-split trap (`,` vs `/`)

`llama-server` and `llama-cli` take `--tensor-split 1,1,1,1`. **`llama-bench` takes
`-ts 1/1/1/1`** — in `llama-bench` a comma means "run the test once per value". So
`-ts 1,1,1,1` there is four tests with split `1` = **everything on GPU 0**, and any model
bigger than one card fails with a bare `failed to load model`.

On this rig that looked for two days like a "llama-bench + `--load-mode dio` + multi-GPU
bug" and pushed three models onto a manual `llama-cli` workaround whose prefill numbers
were ~15× lower than the real ones. The harness now converts a comma `-ts` to slashes and
rejects every other comma (sweeps would silently overwrite each other's results).

---

## Prompt batch size is a condition too

`--ubatch-size` (physical batch) decides how many prompt tokens go through the GPU at once.
llama.cpp's default is 512. Measured on 30 Sep 2026 (prefill of 4096 tokens, 256…4096):

| Model | Best ubatch | vs 512 | Worst case |
|---|---|---|---|
| Qwen3-235B-A22B | 4096 | **+57 %** | rises all the way |
| DeepSeek-V4-Flash | 1024 | **+35 %** | 4096 is **−58 %** vs 1024 |
| gpt-oss-120b | 1024 | +18 % | 4096 −35 % |
| Qwen3-32B (dense) | 1024–4096 | +3 % | flat |

Decode does not depend on it. Because it moves prefill by up to ±50 %, the harness records
the **effective** batch/ubatch (read back from llama-bench's JSON) in every row. Standard
`bench-model` runs use llama.cpp defaults so rows stay comparable; served models use the
measured optimum (see the llama-swap config).

---

## Why ECC and AER are checked before and after

**ECC** counts GPU memory errors. **AER** counts PCIe bus errors. If either increases
**during** a run, the hardware was misbehaving and the result must be **discarded** — not
corrected, not asterisked. Discarded.

Without this check you'd get a number that looks entirely normal but describes a card with
failing memory or a marginal seat in its slot. The harness checks automatically and shouts
in the report's verdict.

---

## The thermal guard

The card throttles itself around **110 °C** and shuts down around **115 °C**, so in theory
nothing is at risk. But **GDDR6 degrades permanently above 100 °C** — it doesn't fail, it
quietly loses its characteristics. So the harness aborts a run at **junction ≥ 95 °C**
(configurable via `temp_limit_c`) rather than trusting the firmware to save the hardware.

An aborted run is marked in the report and **excluded** from the statistics.

---

## The measurement artifact worth knowing about

The most instructive mistake in this project. My first report stated that gpt-oss-20b
peaked at **19 W** — a 6000 tok/s model barely warming the card.

That was false, and the reason is mundane: at `-p 512 -n 128` the model's entire
computation takes **under a second** (prefill ~0.08 s, decode ~0.85 s). The rest of the
wall-clock time is loading 12 GB of weights from disk. Telemetry sampled every 2 s simply
never observed the load. (The harness now samples every 1 s — one `amd-smi` call for four
GPUs takes ~0.1 s — but the lesson stands.)

Under a long test the same model draws **288 W at 57 °C** — nearly identical to the dense
model.

**So:** throughput from a short test is valid; **thermal numbers from a short test are
not.** The harness now flags any run whose peak power is implausibly low, and
`bench-model` always runs a long test alongside the short one.
