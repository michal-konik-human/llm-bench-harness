# New model — from download to finished report

The procedure you follow **every time** you want to test a new model. Three commands and
about twenty minutes of waiting.

If you want an agent to do it for you: [`AGENT-PLAYBOOK.md`](AGENT-PLAYBOOK.md).
If you want to know *what* and *why* is measured: [`METHODOLOGY.md`](METHODOLOGY.md).
If you want to do it by hand: [`MANUAL.md`](MANUAL.md).

---

## 1. Download the model

It has to fit in VRAM — and the weights aren't the only occupant, since KV cache grows
with context length. **Practical ceiling: ~75 % of VRAM for weights.** On a 32 GB card
that's roughly 24 GB.

| Weights | Verdict |
|---|---|
| up to ~14 GB | Lots of headroom: long context, several concurrent requests |
| 14–24 GB | Fits, context to ~32k. The quality ceiling for a single 32 GB card |
| 24–32 GB | Fits, but KV cache competes for space |
| over 32 GB | **Don't, on one card.** Part spills to host RAM and you measure PCIe |

```bash
pip install -U "huggingface_hub[cli]"
hf download <repo> <file.gguf> --local-dir ~/models/<name>
```

**Convention, no exceptions: one model per directory under `~/models/`.** The directory
name becomes the model name in reports and in `INDEX.md`. `bielik-11b` is good;
`new`, `test2`, `downloaded` are not — in a month you won't know what they were.

Check the file arrived complete:

```bash
ls -la ~/models/<name>/
```

---

## 2. Measure — one command

```bash
bench-model --list-new       # confirm it sees the new model
bench-model <name>           # measure
```

`bench-model` handles everything:

1. checks the model fits in VRAM (and reports the KV-cache headroom);
2. **refuses to start** if the card is hot (> 45 °C), if ECC/AER counters are non-zero,
   or if **another process is holding the GPU** — in any of those cases the result would
   be void anyway;
3. runs **two** tests, each as **three independent processes**:
   - `throughput` (`-p 512 -n 128`) — tokens per second,
   - `thermal` (`-p 4096 -n 1024`) — real temperature and power draw;
4. records the full hardware and software state (see METHODOLOGY);
5. writes `report.md` and **appends a row to `results/INDEX.md`**.

**Why two tests and not one.** A fast MoE model finishes the short test in **under a
second** — telemetry sampled every 2 s has nothing to observe, and the report would show
a bogus "19 W, 30 °C". The long test exists so the thermal numbers are real. Take
throughput from the short test, thermals from the long one.

---

## 3. Read the verdict

Last line:

```
VERDICT: OK (bielik-11b: report in 20260916-101500)
```

| Verdict | Meaning | What to do |
|---|---|---|
| `OK` | Valid result | Report the numbers |
| `WITH_CAVEAT` | Usable, conditionally | **Quote the number together with the caveat**, never alone |
| `INVALID` | Discard | Remove the cause (usually hardware) and repeat |
| `SKIPPED` | Doesn't fit on the card | Use a smaller quantisation |

Then:

```bash
cat results/INDEX.md                    # growing table of ALL measurements
less results/<id>/report.md             # full report for one run
```

`INDEX.md` is the reason this harness exists: **one table where every model you've ever
measured sits beside the others, with its conditions.** Only compare rows with the same
**power cap** and the same **llama.cpp commit** — otherwise you're comparing conditions,
not models.

---

## 4. Hand it to a local agent

The point: you don't want to remember this procedure by the twentieth model.

### 4.1 Start the agent — **on the CPU**

```bash
AGENT_MODEL=~/models/gpt-oss-20b/gpt-oss-20b-MXFP4.gguf ./agent-server.sh
./agent-server.sh --status     # must say "mode: CPU"
```

**Why CPU and not the GPU.** The agent is itself a model and needs memory. On a single
card an agent on the GPU would take ~12 GB — an 18 GB model under test would no longer fit
alongside its KV cache, and every measurement would be depressed because the card is doing
two things at once. Orchestration is a dozen tool calls, so 5–10 tok/s from the CPU is
plenty. **The measurement is the expensive part; talking to the agent isn't.**

With more than one GPU the agent can have its own — see the note at the end of
`agent-server.sh`.

### 4.2 Connect it and give it the playbook

The endpoint is OpenAI-compatible:

```
http://127.0.0.1:8080/v1
```

Give it [`AGENT-PLAYBOOK.md`](AGENT-PLAYBOOK.md) as its **system prompt**. That file is
written for a small model: step-by-step procedure, decision tables instead of open-ended
judgement, explicit prohibitions.

The agent needs shell access to exactly three commands: `bench-model`, `bench.py`, and
whatever you use to check hardware state.

### 4.3 Tell it what to do

> Measure all new models and report the results.

It will run `bench-model --list-new` → `bench-model <each>` → read the verdict from its
decision table → report in the agreed format.

### 4.4 Verify the agent before trusting it

Before letting it loose on a new model, have it measure a model **you have already
measured** and compare against `INDEX.md`. Agreement within a few percent means the agent
works. That costs one run and buys confidence in the next twenty.

---

## 5. When the default isn't enough

`bench-model` is the fast path. For non-standard measurements use `bench.py` with your own
config — syntax and worked examples in the repo README and `configs/`.

| You want | Config |
|---|---|
| What a 210 W cap costs | `configs/power-cap.json` |
| How performance falls off with context | `configs/context-scaling.json` |
| Multiple GPUs | `configs/multi-gpu-EXAMPLE.json` |
