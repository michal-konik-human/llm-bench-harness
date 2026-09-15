# Measuring by hand — without the scripts

Exactly what `bench-model` and `bench.py` do, broken out into individual commands.
Useful for verifying the harness, for a one-off measurement, and for understanding what
is actually happening.

For anything that will end up in a report, use `bench-model` — by hand it is easy to skip
recording the conditions, which is the whole point.

---

## Step 1 — state before the measurement

Junction should be **below 45 °C** (card has cooled), error counters at zero. If not,
don't start.

```bash
amd-smi metric -g 0 | grep -A3 "TEMPERATURE"
amd-smi metric -g 0 | grep -A4 "    ECC:"
```

---

## Step 2 — record the conditions. **Before** you measure

Without this, the number from step 4 is worthless.

```bash
# llama.cpp commit - results are not comparable across builds
git -C ~/llama.cpp rev-parse --short HEAD

# build flags - GGML_HIP / GGML_VULKAN / GPU_TARGETS change everything
grep -E "GGML_HIP|GGML_VULKAN|GPU_TARGETS|CMAKE_BUILD_TYPE" ~/llama.cpp/build/CMakeCache.txt

# ROCm version
cat /opt/rocm/.info/version

# CURRENT power cap
amd-smi static -g 0 | grep -A3 PPT0

# error counters BEFORE - write these down
BDF=$(amd-smi static -g 0 --json | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d["gpu_data"][0]["bus"]["bdf"])')
cat /sys/bus/pci/devices/$BDF/aer_dev_correctable
```

---

## Step 3 — set the power cap, if the run is meant to be at a specific one

```bash
sudo amd-smi set -g 0 --power-cap 210     # 210 W is the FLOOR on R9700; 200 W is rejected
amd-smi static -g 0 | grep SOCKET_POWER_LIMIT
```

Back to stock: `sudo amd-smi set -g 0 --power-cap 300`.

---

## Step 4 — the measurement: three **separate** processes

This is the crux. The same command three times, **each as a new process**, with cooldown
in between.

Mind `HIP_VISIBLE_DEVICES` — on a system with an integrated GPU, **check which index is
your discrete card**. Don't assume.

```bash
export HIP_VISIBLE_DEVICES=0
cd ~/llama.cpp/build/bin
M=~/models/gpt-oss-20b/gpt-oss-20b-MXFP4.gguf

for i in 1 2 3; do
  echo "=== run $i ==="
  ./llama-bench -m "$M" -ngl 999 -sm none -p 512 -n 128 -r 1
  sleep 45          # cooldown: each run should start from a similar temperature
done
```

| Flag | Meaning |
|---|---|
| `-ngl 999` | Put **all** layers on the GPU. If some stayed on the CPU you'd be measuring the CPU |
| `-sm none` | No split across GPUs. Correct for a single card |
| `-p 512` | Prefill 512 tokens |
| `-n 128` | Generate 128 tokens |
| `-r 1` | **One** measurement per process. Repeatability comes from separate processes, not this flag |

For **thermal** numbers use a much longer test, e.g. `-p 4096 -n 1024`. A short test on a
fast MoE model completes in under a second and telemetry never sees the load.

---

## Step 5 — telemetry **during** the run, in a second terminal

Without this you don't know whether you measured the model or the case airflow.

```bash
while true; do
  amd-smi metric -g 0 | grep -E "GFX_ACTIVITY|SOCKET_POWER" | tr '\n' ' '
  amd-smi metric -g 0 | grep -A3 "TEMPERATURE" | grep -i hotspot | tr '\n' ' '
  echo
  sleep 3
done
```

You want: `GFX_ACTIVITY` near 100 % (the GPU really is working) and junction below 85 °C.

**Caveat:** this card reports `THROTTLED` even at idle, at 16 W and 33 °C. It's a known
bogus reading — don't interpret it as throttling.

PCIe link — **under load only**, because it downtrains at idle:

```bash
cat /sys/bus/pci/devices/$BDF/current_link_width   # expect 16 (or 8 with a 2nd card)
cat /sys/bus/pci/devices/$BDF/current_link_speed   # under load: 32.0 GT/s
```

---

## Step 6 — error counters **after**

```bash
cat /sys/bus/pci/devices/$BDF/aer_dev_correctable
amd-smi metric -g 0 | grep -A4 "    ECC:"
```

**Different from step 2? Discard the result and find the hardware cause.**

---

## Step 7 — judgement

- Spread across the three runs **below ~2 %** → the mean is a sensible number.
- Spread **above ~10 %**, results in two clusters → bimodality. **Report both modes, not
  the mean.** Do 5–7 runs.
- Peak junction **≥ 85 °C** → lower the power cap and repeat; you're measuring cooling.
- Peak power implausibly low on a fast model → your test was shorter than the sampling
  period. Thermals are meaningless; throughput is fine.

---

## When something looks wrong

| Symptom | Cause and fix |
|---|---|
| Result ~10× too low | You're measuring the integrated GPU. Check `HIP_VISIBLE_DEVICES` |
| `rocm-smi` shows `N/A` for the card | The card is runtime-suspended (D3hot) and sysfs returns `EBUSY`. **Normal.** Use `amd-smi` |
| `amd-smi set --power-cap 200` rejected | Firmware floor is 210 W on this card |
| Model won't fit in VRAM | Weights **plus KV cache** must fit. Practical ceiling is ~75 % of VRAM for weights |
| First run slower than the rest | Weights loading from disk. Normal — which is why you do several runs |
| `sudo` prompts and the run stalls | Setting the power cap needs root. Set it manually beforehand, or add a `NOPASSWD` entry for `amd-smi` |
