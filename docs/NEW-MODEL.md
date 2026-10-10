# New model — from download to finished report

The procedure you follow **every time** you want to test a new model. Three commands and
about twenty minutes of waiting.

If you want an agent to do it for you: [`AGENT-PLAYBOOK.md`](AGENT-PLAYBOOK.md).
If you want to know *what* and *why* is measured: [`METHODOLOGY.md`](METHODOLOGY.md).
If you want to do it by hand: [`MANUAL.md`](MANUAL.md).

---

## 1. Download the model

It has to fit in VRAM — and the weights aren't the only occupant, since KV cache grows
with context length. **Practical ceiling: ~75 % of the VRAM of the GPUs used for weights.**
`bench-model` sums every shard of a split GGUF and picks the **smallest number of GPUs** on
which the weights stay under that ceiling (fewer cards = fewer hand-offs between cards =
faster decode). On 32 GB cards:

| Weights | GPUs `bench-model` picks |
|---|---|
| up to ~24 GB | 1 (up to ~14 GB leaves room for long context and concurrent requests) |
| ~24–48 GB | 2 |
| ~48–72 GB | 3 |
| ~72–96 GB | 4 |
| ~96–118 GB | 4, reported as `TIGHT` (it still runs; little room is left for KV cache) |
| more than all cards | `SKIPPED` — part would spill to host RAM and you'd measure PCIe |

```bash
pip install -U "huggingface_hub[cli]"
hf download <repo> <file.gguf> --local-dir ~/models/<name>
```

**Convention, no exceptions: one model per directory under `~/models/`.** The directory
name becomes the model name in reports and in `all-benchmark-results.md`. `bielik-11b` is good;
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

1. checks the model fits and picks the number of GPUs (override with `--gpus N`); models
   over 40 GiB load with `--load-mode dio` and multi-GPU runs get the required `NCCL_*` env;
2. **refuses to start** if any chosen card is hot (junction > 45 °C), if ECC/AER counters
   are non-zero anywhere on its PCIe path, if **another process is holding a GPU**, or if
   **llama-swap still has a model loaded** — in any of those cases the result would be void
   anyway (it prints the `curl … /api/models/unload` command to free them);
3. runs **two** tests, each as **three independent processes**:
   - `throughput` (`-p 512 -n 128`) — tokens per second,
   - `thermal` (`-p 4096 -n 1024`) — real temperature and power draw;
4. records the full hardware and software state (see METHODOLOGY);
5. writes `report.md` and **appends a row to `results/all-benchmark-results.md`**.

**Why two tests and not one.** A fast MoE model finishes the short test in **under a
second** — telemetry (sampled every 1 s) has almost nothing to observe, and the report
would show a bogus "19 W, 30 °C". The long test exists so the thermal numbers are real. Take
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
cat results/all-benchmark-results.md                    # growing table of ALL measurements
less results/<id>/report.md             # full report for one run
```

`all-benchmark-results.md` is the reason this harness exists: **one table where every model you've ever
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
measured** and compare against `all-benchmark-results.md`. Agreement within a few percent means the agent
works. That costs one run and buys confidence in the next twenty.

---

## 5. When the default isn't enough

`bench-model` is the fast path. For non-standard measurements use `bench.py` with your own
config — syntax and worked examples in the repo README and `configs/`.

| You want | Config |
|---|---|
| What a 210 W cap costs | `configs/power-cap.json` |
| How performance falls off with context | `configs/context-scaling.json` |
| Multiple GPUs, your own split | `configs/multi-gpu-EXAMPLE.json` (in `llama-bench`, `-ts` uses slashes: `1/1/1/1`) |

---

## 6. Then measure how good it is

Speed tells you whether a model is usable; it doesn't tell you whether it is any good at your
job. The sibling repo [local-llm-quality-bench](https://github.com/michal-konik-human/local-llm-quality-bench)
grades the same model, served through the same llama-swap, on agent and assistant tasks in
Polish and English. Its leaderboard links back to the decode speed measured here.
