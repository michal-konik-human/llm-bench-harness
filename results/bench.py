#!/usr/bin/env python3
"""
bench.py — a declarative benchmark harness for local LLM inference.

WHY THIS EXISTS

Benchmarking models across hardware and software configurations is repetitive work.
Done by hand each time, it produces numbers that cannot be compared with each other,
because something different was set each time — and usually nobody wrote down what.

Worse: such a number looks credible and ends up in a report.

This turns a measurement into a **declaration**. You describe what you want measured
in a JSON file; the harness executes the matrix and records the *conditions* alongside
the results. Two runs of the same config file are comparable by construction rather
than by good memory.

Designed to be driven by an agent — see docs/AGENT-PLAYBOOK.md.

WHAT IT RECORDS WITH EVERY RESULT (and why each one is necessary)

  llama.cpp commit + build flags   results are not comparable across builds
  ROCm version                     same
  GPU power cap                    300 W and 210 W are two different measurements
  PCIe link width/speed            read *under load*; links downtrain at idle
  peak junction / VRAM temp        if the card throttled, the number is about
                                   cooling, not about the model
  environment variables            HIP_VISIBLE_DEVICES, GGML_*, NCCL_* change everything
  ECC + AER counters before/after  if they moved, the hardware misbehaved and the
                                   result must be discarded, not adjusted
  other processes holding the GPU   a second consumer makes the run measure contention

USAGE

  ./bench.py configs/single-gpu.json              run the matrix
  ./bench.py configs/single-gpu.json --dry-run    print the plan, run nothing
  ./bench.py --compare results/A results/B        diff two runs
  ./bench.py --list                               available models and configs

Results land in results/<timestamp>/ as:
  report.md     human-readable, leads with a verdict
  results.json  full machine-readable data
  results.csv   for a spreadsheet
  raw.log       raw llama-bench output

CONFIGURATION

Everything hardware-specific is auto-detected. Overrides, if you need them:

  RIG_GPU_INDEX   GPU index to measure and read telemetry from   (default 0)
  RIG_GPU_BDF     PCI address, e.g. 0000:03:00.0                 (default: auto)
  RIG_VRAM_MB     total VRAM in MiB                              (default: auto)

Requirements: ROCm with `amd-smi`, a built `llama-bench`, `lm-sensors` (optional),
Python 3.10+. No third-party Python packages.

License: MIT
"""

import argparse
import csv
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"

GPU_INDEX = int(os.environ.get("RIG_GPU_INDEX", "0"))

# Junction temperature at which a run is aborted. The card itself throttles around
# 110 C and shuts down around 115 C, but GDDR6 degrades *permanently* above 100 C,
# so we stop well before the firmware would ever intervene.
DEFAULT_TEMP_LIMIT = 95

# Telemetry sampling period. Keep this well below the length of your shortest test,
# or peak power/temperature readings will be meaningless — see the note in
# docs/METHODOLOGY.md about short compute windows.
TELEMETRY_INTERVAL = 2.0


# --------------------------------------------------------------------- helpers

def sh(cmd, timeout=60, env=None):
    """Run a command; return (rc, stdout, stderr). Never raises."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, env=env)
        return p.returncode, p.stdout, p.stderr
    except Exception as exc:
        return -1, "", str(exc)


def sh_json(cmd, timeout=60):
    rc, out, _ = sh(cmd, timeout)
    if not out.strip():
        return None
    try:
        return json.loads(out)
    except Exception:
        return None


def dsec(node, name):
    """Fetch a metric sub-section as a dict.

    amd-smi is not uniformly shaped across GPUs: for an integrated GPU several
    sections (e.g. "power") come back as the bare string "N/A" instead of a dict.
    A plain `.get(name, {}) or {}` keeps that string, and the next `.get()` raises
    AttributeError — which kills the collection loop mid-cycle and silently drops
    every chart defined after it.
    """
    if not isinstance(node, dict):
        return {}
    v = node.get(name)
    return v if isinstance(v, dict) else {}


def leaf(x, default=None):
    """amd-smi leaves are {"value": x, "unit": "..."} or a bare scalar."""
    if isinstance(x, dict):
        x = x.get("value")
    return default if x in (None, "N/A") else x


def read_file(path, default=None):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return default


# ------------------------------------------------------------ device discovery

def _gpu_static():
    return sh_json(["amd-smi", "static", "-g", str(GPU_INDEX), "--json"], timeout=25)


def detect_bdf():
    """PCI address of the GPU under test.

    Auto-detected so this harness works on any machine. Override with RIG_GPU_BDF.
    """
    override = os.environ.get("RIG_GPU_BDF")
    if override:
        return override
    data = _gpu_static()
    if data and data.get("gpu_data"):
        bdf = leaf(dsec(data["gpu_data"][0], "bus").get("bdf"))
        if bdf:
            return bdf
    return None


def detect_vram_mb():
    override = os.environ.get("RIG_VRAM_MB")
    if override:
        try:
            return int(override)
        except ValueError:
            pass
    data = sh_json(["amd-smi", "metric", "-g", str(GPU_INDEX), "--json"], timeout=25)
    if data and data.get("gpu_data"):
        total = leaf(dsec(data["gpu_data"][0], "mem_usage").get("total_vram"))
        if total:
            try:
                return int(total)
            except (TypeError, ValueError):
                pass
    return None


CARD_BDF = detect_bdf()


def sysfs(*parts):
    if not CARD_BDF:
        return None
    return Path("/sys/bus/pci/devices") / CARD_BDF / Path(*parts)


# ------------------------------------------------------------------- telemetry

def gpu_metrics():
    """Read telemetry through amd-smi, NOT sysfs.

    A discrete GPU with no display attached gets parked in D3hot by runtime power
    management, and sysfs hwmon reads then return EBUSY. amd-smi wakes the card by a
    different path and always returns real values. This is the single most confusing
    thing about monitoring an idle discrete AMD GPU.
    """
    data = sh_json(["amd-smi", "metric", "-g", str(GPU_INDEX), "--json"], timeout=20)
    if not data or not data.get("gpu_data"):
        return {}
    g = data["gpu_data"][0]
    t, p, f, u = (dsec(g, "temperature"), dsec(g, "power"),
                  dsec(g, "fan"), dsec(g, "usage"))
    return {
        "junction_c": leaf(t.get("hotspot")),
        "edge_c": leaf(t.get("edge")),
        "vram_c": leaf(t.get("mem")),
        "power_w": leaf(p.get("socket_power")),
        "fan_rpm": leaf(f.get("rpm")),
        "gfx_pct": leaf(u.get("gfx_activity")),
        "throttle": leaf(p.get("throttle_status")),
    }


def power_cap_now():
    data = _gpu_static()
    if not data or not data.get("gpu_data"):
        return None
    ppt = dsec(dsec(data["gpu_data"][0], "limit"), "ppt0")
    return leaf(ppt.get("socket_power_limit"))


def link_state():
    """PCIe link width and speed.

    At idle the link downtrains (often to 2.5 GT/s) to save power, so this reading
    only means something *under load* — which is why we sample it while the GPU is busy.
    """
    w = read_file(sysfs("current_link_width") or "", "?")
    s = read_file(sysfs("current_link_speed") or "", "?")
    return {"width": w, "speed": s}


def aer_totals():
    """PCIe Advanced Error Reporting counters, straight from sysfs. No root needed.

    If these move during a run, the link misbehaved and the result is suspect.
    """
    out = {}
    for kind in ("correctable", "nonfatal", "fatal"):
        path = sysfs(f"aer_dev_{kind}")
        raw = read_file(path, None) if path else None
        total = None
        if raw:
            total = 0
            for line in raw.splitlines():
                parts = line.split()
                if len(parts) == 2:
                    try:
                        total += int(parts[1])
                    except ValueError:
                        pass
        out[kind] = total
    return out


def ecc_totals():
    data = sh_json(["amd-smi", "metric", "-g", str(GPU_INDEX), "--json"], timeout=20)
    if not data or not data.get("gpu_data"):
        return {}
    e = dsec(data["gpu_data"][0], "ecc")
    return {
        "correctable": leaf(e.get("total_correctable_count")),
        "uncorrectable": leaf(e.get("total_uncorrectable_count")),
    }


def gpu_consumers():
    """Processes currently holding the GPU.

    Anything besides our own measurement invalidates the result. The realistic cause
    is a forgotten llama-server, or an agent's own model left resident on the GPU —
    in which case the run measures contention for the card, not the model.
    """
    data = sh_json(["amd-smi", "process", "--json"], timeout=25)
    out = []
    if not isinstance(data, list):
        return out
    for entry in data:
        if entry.get("gpu") != GPU_INDEX:
            continue
        plist = entry.get("process_list")
        if not isinstance(plist, list):
            continue
        for item in plist:
            info = item.get("process_info") if isinstance(item, dict) else None
            if isinstance(info, dict) and info.get("name"):
                out.append({"name": info["name"], "pid": info.get("pid")})
    return out


# --------------------------------------------------------------- configuration

def hardware_config():
    cpu = "?"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    ram_kb = 0
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                ram_kb = int(line.split()[1])
                break
    except OSError:
        pass
    # dmidecode needs root; without a NOPASSWD entry these simply stay unset
    _, bios, _ = sh(["sudo", "-n", "dmidecode", "-s", "bios-version"], timeout=20)
    _, memspeed, _ = sh(["sudo", "-n", "sh", "-c",
                         "dmidecode -t memory | grep -m1 'Configured Memory Speed'"],
                        timeout=20)
    return {
        "cpu": cpu,
        "ram_gb": round(ram_kb / 1024 / 1024, 1) if ram_kb else None,
        "ram_configured_speed": (memspeed.split(":")[-1].strip() or None) if memspeed else None,
        "bios": bios.strip() or None,
        "vram_mb": detect_vram_mb(),
    }


def software_config():
    llama = Path(os.environ.get("LLAMA_CPP_DIR", Path.home() / "llama.cpp"))
    _, commit, _ = sh(["git", "-C", str(llama), "rev-parse", "--short", "HEAD"])
    _, cdate, _ = sh(["git", "-C", str(llama), "show", "-s", "--format=%cs", "HEAD"])
    flags = {}
    cache = llama / "build" / "CMakeCache.txt"
    if cache.exists():
        for line in cache.read_text(errors="ignore").splitlines():
            for key in ("GGML_HIP:", "GGML_CUDA:", "GGML_VULKAN:",
                        "GPU_TARGETS:", "CMAKE_BUILD_TYPE:"):
                if line.startswith(key):
                    k, _, v = line.partition("=")
                    flags[k.split(":")[0]] = v.strip()
    drv = None
    if CARD_BDF:
        _, out, _ = sh(["sh", "-c",
                        f"lspci -nnk -s {CARD_BDF.split(':', 1)[1]} "
                        f"| grep -i 'Kernel driver in use'"], timeout=15)
        drv = out.split(":")[-1].strip() or None
    return {
        "kernel": (sh(["uname", "-r"])[1] or "").strip(),
        "kernel_cmdline": read_file("/proc/cmdline", "?"),
        "rocm": read_file("/opt/rocm/.info/version", "?"),
        "llama_cpp_commit": commit.strip() or "?",
        "llama_cpp_commit_date": cdate.strip() or None,
        "llama_cpp_build": flags,
        "gpu_driver": drv,
    }


def environment():
    gpus = []
    data = sh_json(["amd-smi", "static", "--json"], timeout=25)
    for e in (data or {}).get("gpu_data", []):
        asic, bus = dsec(e, "asic"), dsec(e, "bus")
        gpus.append({
            "index": e.get("gpu"),
            "name": leaf(asic.get("market_name")),
            "gfx": leaf(asic.get("target_graphics_version")),
            "bdf": leaf(bus.get("bdf")),
            "cus": leaf(asic.get("num_compute_units")),
        })
    sw = software_config()
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "gpu_index": GPU_INDEX,
        "gpu_bdf": CARD_BDF,
        "kernel": sw["kernel"],
        "rocm": sw["rocm"],
        "llama_cpp_commit": sw["llama_cpp_commit"],
        "power_cap": power_cap_now(),
        "gpus": gpus,
        "hardware": hardware_config(),
        "software": sw,
        "gpu_consumers_at_start": gpu_consumers(),
        "aer_before": aer_totals(),
        "ecc_before": ecc_totals(),
    }


def set_power_cap(watts):
    """Set the GPU power cap. Returns (ok, message).

    Note: cards have a firmware *floor*. On the Radeon AI PRO R9700 it is 210 W —
    anything lower is rejected outright.
    """
    rc, out, err = sh(["sudo", "-n", "amd-smi", "set", "-g", str(GPU_INDEX),
                       "--power-cap", str(watts)], timeout=40)
    if rc != 0:
        msg = (err or out or "unknown error").strip().splitlines()
        return False, (msg[-1][:200] if msg else "unknown error")
    time.sleep(2)
    return True, f"set to {power_cap_now()}"


# ----------------------------------------------------------------- measurement

BENCH_ROW = re.compile(
    r"\|\s*(?P<test>(?:pp|tg)\d+)\s*\|\s*(?P<val>[\d.]+)\s*(?:±\s*(?P<sd>[\d.]+))?\s*\|"
)


def parse_bench(stdout):
    res = {}
    for line in stdout.splitlines():
        m = BENCH_ROW.search(line)
        if m:
            res[m.group("test")] = {
                "tok_s": float(m.group("val")),
                "sd": float(m.group("sd")) if m.group("sd") else None,
            }
    return res


def run_once(model_path, cfg, temp_limit, log_fh):
    """One INDEPENDENT measurement process, with telemetry sampling and a thermal guard.

    A separate OS process per run is deliberate. Decode throughput on some cards is
    bimodal: the card settles into one of two modes and stays there for the lifetime of
    the process. `llama-bench -r N` therefore averages N samples *inside a single mode*
    and reports falsely tiny variance. Only separate processes can land in both.
    """
    binary = os.path.expanduser(
        cfg.get("binary", str(Path.home() / "llama.cpp/build/bin/llama-bench")))
    env = dict(os.environ)
    env.update({k: str(v) for k, v in cfg.get("env", {}).items()})

    cmd = [binary, "-m", str(model_path), "-r", "1"]
    cmd += [str(a) for a in cfg.get("args", ["-ngl", "999", "-sm", "none",
                                             "-p", "512", "-n", "128"])]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=env, start_new_session=True)

    peak = {"junction_c": 0, "vram_c": 0, "power_w": 0, "fan_rpm": 0}
    samples, aborted, link_under_load = 0, None, None

    while proc.poll() is None:
        m = gpu_metrics()
        if m:
            samples += 1
            for k in peak:
                v = m.get(k)
                if isinstance(v, (int, float)) and v > peak[k]:
                    peak[k] = v
            # Only meaningful while the GPU is actually busy — see link_state().
            if isinstance(m.get("gfx_pct"), (int, float)) and m["gfx_pct"] > 50:
                link_under_load = link_state()
            j = m.get("junction_c")
            if isinstance(j, (int, float)) and j >= temp_limit:
                aborted = j
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:
                    proc.terminate()
                break
        time.sleep(TELEMETRY_INTERVAL)

    try:
        stdout = proc.communicate(timeout=30)[0] or ""
    except Exception:
        stdout = ""
    log_fh.write(stdout + "\n" + "-" * 70 + "\n")

    return {
        "results": parse_bench(stdout) if not aborted else {},
        "peak": peak,
        "samples": samples,
        "aborted_at_c": aborted,
        "link_under_load": link_under_load or link_state(),
        "ok": aborted is None and bool(parse_bench(stdout)),
    }


def execute(config, dry_run=False):
    models = config["models"]
    configs = config["configs"]
    runs = config.get("runs", 3)
    cooldown = config.get("cooldown_s", 45)
    temp_limit = config.get("temp_limit_c", DEFAULT_TEMP_LIMIT)

    cells = [(m, c) for c in configs for m in models]
    print(f"\nPlan: {len(models)} model(s) x {len(configs)} config(s) x {runs} run(s) "
          f"= {len(cells) * runs} measurement processes")
    for m in models:
        print(f"  model   {m['name']:<20} {m['path']}")
    for c in configs:
        cap = f", cap {c['power_cap_w']} W" if c.get("power_cap_w") else ""
        print(f"  config  {c['name']:<20} "
              f"{' '.join(str(a) for a in c.get('args', []))}{cap}")
    print(f"  thermal guard : abort at junction >= {temp_limit} C")
    print(f"  cooldown      : {cooldown} s between runs")
    print(f"  GPU under test: index {GPU_INDEX}, {CARD_BDF or 'BDF unknown'}\n")

    if dry_run:
        print("--dry-run: nothing executed.")
        return None

    run_id = time.strftime("%Y%m%d-%H%M%S")
    outdir = RESULTS / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    record = {
        "run_id": run_id,
        "config_name": config.get("name", "?"),
        "config_file": config.get("_source"),
        "environment": environment(),
        "settings": {"runs": runs, "cooldown_s": cooldown, "temp_limit_c": temp_limit},
        # The exact configs used. Env vars and flags change results, so the report
        # must show them verbatim rather than describing them.
        "configs_used": [{"name": c["name"], "env": c.get("env", {}),
                          "args": c.get("args", []),
                          "power_cap_w": c.get("power_cap_w")} for c in configs],
        "cells": [],
    }

    raw_log = open(outdir / "raw.log", "w")

    for cfg in configs:
        # The power cap is a property of the CONFIG, not of a single run — set it once
        # per group so every result in that group shares one condition.
        if cfg.get("power_cap_w"):
            ok, msg = set_power_cap(cfg["power_cap_w"])
            print(f"[{cfg['name']}] power cap: {msg}" if ok
                  else f"[{cfg['name']}] FAILED to set power cap: {msg}")
            if not ok:
                print("   -> measuring at the CURRENT cap; recorded in the result")

        actual_cap = power_cap_now()

        for model in cfg.get("models", models):
            if isinstance(model, str):
                model = next((m for m in models if m["name"] == model), None)
                if not model:
                    continue
            path = Path(os.path.expanduser(model["path"]))
            # Store the path home-relative. Results are meant to be publishable, and an
            # absolute path leaks the username for no analytical benefit.
            try:
                shown = "~/" + str(path.relative_to(Path.home()))
            except ValueError:
                shown = str(path)
            cell = {"model": model["name"], "config": cfg["name"],
                    "model_path": shown, "power_cap_w": actual_cap, "runs": []}
            if not path.exists():
                cell["error"] = f"file not found: {path}"
                print(f"[{cfg['name']}/{model['name']}] SKIPPED - file not found")
                record["cells"].append(cell)
                continue

            print(f"\n[{cfg['name']}/{model['name']}] cap {actual_cap}")
            for r in range(1, runs + 1):
                res = run_once(path, cfg, temp_limit, raw_log)
                cell["runs"].append(res)
                if res["aborted_at_c"]:
                    print(f"  {r}/{runs}: ABORTED by thermal guard "
                          f"({res['aborted_at_c']} C)")
                else:
                    pretty = "  ".join(f"{k} {v['tok_s']:.2f} tok/s"
                                       for k, v in res["results"].items())
                    print(f"  {r}/{runs}: {pretty or 'no result'}  | peak "
                          f"junction {res['peak']['junction_c']}C "
                          f"vram {res['peak']['vram_c']}C "
                          f"{res['peak']['power_w']}W | link x"
                          f"{res['link_under_load']['width']} @ "
                          f"{res['link_under_load']['speed']}")
                if r < runs:
                    time.sleep(cooldown)
            record["cells"].append(cell)

    raw_log.close()
    record["environment"]["aer_after"] = aer_totals()
    record["environment"]["ecc_after"] = ecc_totals()

    (outdir / "results.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False))
    write_csv(record, outdir / "results.csv")
    (outdir / "report.md").write_text(build_report(record))

    print(f"\nResults: {outdir}")
    print("  report.md    <- read this")
    print("  results.json <- machine-readable")
    print("  results.csv  <- spreadsheet")
    return outdir


# ------------------------------------------------------------------- reporting

def summarise(cell):
    per_test = {}
    for run in cell["runs"]:
        for test, v in run.get("results", {}).items():
            per_test.setdefault(test, []).append(v["tok_s"])
    out = {}
    for test, vals in per_test.items():
        if not vals:
            continue
        mean = statistics.fmean(vals)
        out[test] = {
            "n": len(vals), "mean": mean, "min": min(vals), "max": max(vals),
            # Spread matters more than the mean here: a wide spread is the signal
            # for bimodality, and then the mean is actively misleading.
            "spread_pct": (max(vals) - min(vals)) / mean * 100 if mean else 0,
            "values": vals,
        }
    return out


def write_csv(record, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "config", "power_cap_w", "test", "n", "mean_tok_s",
                    "min_tok_s", "max_tok_s", "spread_pct", "peak_junction_c",
                    "peak_vram_c", "peak_power_w", "link_width", "link_speed",
                    "llama_commit", "rocm"])
        env = record["environment"]
        for cell in record["cells"]:
            stats = summarise(cell)
            pj = max((r["peak"]["junction_c"] for r in cell["runs"]), default="")
            pv = max((r["peak"]["vram_c"] for r in cell["runs"]), default="")
            pw = max((r["peak"]["power_w"] for r in cell["runs"]), default="")
            link = cell["runs"][0]["link_under_load"] if cell["runs"] else {}
            for test, s in stats.items():
                w.writerow([cell["model"], cell["config"], cell["power_cap_w"], test,
                            s["n"], f"{s['mean']:.2f}", f"{s['min']:.2f}",
                            f"{s['max']:.2f}", f"{s['spread_pct']:.2f}", pj, pv, pw,
                            link.get("width", ""), link.get("speed", ""),
                            env["llama_cpp_commit"], env["rocm"]])


def build_report(record):
    env = record["environment"]
    hw = env.get("hardware", {}) or {}
    sw = env.get("software", {}) or {}
    L = []
    A = L.append

    A(f"# Benchmark report — {record['run_id']}\n")
    A(f"Config: **{record['config_name']}**  \n")
    A(f"Run at: {env['timestamp']}\n")

    # Verdict first, deliberately before any table: the report should answer
    # "is this result usable?" without the reader having to work it out.
    A("\n## Verdict\n")
    verdict = []
    for cell in record["cells"]:
        if cell.get("error"):
            verdict.append(f"❌ **{cell['model']} / {cell['config']}** — {cell['error']}")
            continue
        n_ab = sum(1 for r in cell["runs"] if r["aborted_at_c"])
        stats = summarise(cell)
        if not stats:
            verdict.append(f"❌ **{cell['model']} / {cell['config']}** — no result "
                           f"({n_ab} run(s) aborted by the thermal guard)")
            continue
        pj = max((r["peak"]["junction_c"] for r in cell["runs"]), default=0)
        pw = max((r["peak"]["power_w"] for r in cell["runs"]), default=0)
        worst = max(s["spread_pct"] for s in stats.values())
        flags = []
        # An implausibly low peak does not mean the GPU idled — it means the compute
        # window was shorter than the telemetry sampling period. Reporting the number
        # without this caveat invites a completely wrong conclusion.
        if isinstance(pw, (int, float)) and pw < 60:
            flags.append(f"peak power {pw} W — TELEMETRY UNRELIABLE: compute window "
                         f"shorter than the {TELEMETRY_INTERVAL} s sampling period; "
                         f"do not draw thermal conclusions, use a longer test")
        if n_ab:
            flags.append(f"{n_ab} run(s) aborted thermally")
        if worst > 10:
            flags.append(f"spread {worst:.0f}% — likely bimodal, do NOT report the mean")
        if isinstance(pj, (int, float)) and pj >= 85:
            flags.append(f"junction {pj} °C — you are measuring cooling, not the model")
        mark = "✅" if not flags else "⚠️"
        bits = ", ".join(f"{k} **{s['mean']:.1f}** tok/s" for k, s in stats.items())
        verdict.append(f"{mark} **{cell['model']} / {cell['config']}** "
                       f"({cell['power_cap_w']}): {bits}"
                       + (f" — {'; '.join(flags)}" if flags else ""))
    for v in verdict:
        A(f"- {v}")

    if env.get("gpu_consumers_at_start"):
        names = ", ".join(c.get("name", "?").split("/")[-1]
                          for c in env["gpu_consumers_at_start"])
        A(f"\n❌ **The GPU was not exclusive at start** ({names}) — this result "
          f"describes contention for the card, not model performance.")

    drift = [k for k in env.get("aer_before", {})
             if env.get("aer_before", {}).get(k) != env.get("aer_after", {}).get(k)]
    drift += [k for k in env.get("ecc_before", {})
              if env.get("ecc_before", {}).get(k) != env.get("ecc_after", {}).get(k)]
    A("")
    if drift:
        A("❌ **Hardware error counters increased during the run — discard these "
          "results and repeat.** Details below.")
    else:
        A("✅ Hardware error counters (AER, ECC) unchanged — results are not "
          "contaminated by bus or memory errors.")

    A("\n## Hardware state at run time\n")
    A("Recorded automatically. Without these values the results below are not "
      "comparable with anything.\n")
    A("\n| Parameter | Value |\n|---|---|")
    A(f"| CPU | {hw.get('cpu', '?')} |")
    A(f"| RAM | {hw.get('ram_gb', '?')} GB"
      + (f", {hw['ram_configured_speed']}" if hw.get("ram_configured_speed") else "") + " |")
    A(f"| BIOS | {hw.get('bios') or 'not read (needs root)'} |")
    for g in env["gpus"]:
        A(f"| GPU {g['index']} | {g['name']} — {g['gfx']}, {g['cus']} CU, {g['bdf']} |")
    A(f"| GPU under test | index {env.get('gpu_index')} ({env.get('gpu_bdf')}) |")
    A(f"| VRAM | {hw.get('vram_mb', '?')} MiB |")
    A(f"| Power cap | {env['power_cap']} |")

    A("\n## Software state at run time\n")
    A("| Parameter | Value |\n|---|---|")
    A(f"| Kernel | {sw.get('kernel', '?')} |")
    A(f"| Kernel cmdline | `{sw.get('kernel_cmdline', '?')}` |")
    A(f"| GPU driver | {sw.get('gpu_driver') or '?'} |")
    A(f"| ROCm | {sw.get('rocm', '?')} |")
    commit, cdate = sw.get("llama_cpp_commit", "?"), sw.get("llama_cpp_commit_date")
    A(f"| llama.cpp | `{commit}`" + (f" ({cdate})" if cdate else "") + " |")
    if sw.get("llama_cpp_build"):
        A("| llama.cpp build flags | " +
          ", ".join(f"`{k}={v}`" for k, v in sorted(sw["llama_cpp_build"].items())) + " |")

    A("\n## Measurement parameters\n")
    A("| Parameter | Value |\n|---|---|")
    A(f"| Runs per cell | {record['settings']['runs']} — **separate OS processes** |")
    A(f"| Cooldown between runs | {record['settings']['cooldown_s']} s |")
    A(f"| Thermal guard | abort at junction ≥ {record['settings']['temp_limit_c']} °C |")
    A("\n**Configurations used** — env vars and flags affect the result, so they are "
      "recorded verbatim:\n")
    A("| Config | Environment | llama-bench flags |")
    A("|---|---|---|")
    for cfg in (record.get("configs_used") or []):
        envs = ", ".join(f"`{k}={v}`" for k, v in (cfg.get("env") or {}).items()) or "—"
        args = " ".join(str(a) for a in (cfg.get("args") or [])) or "—"
        A(f"| {cfg['name']} | {envs} | `{args}` |")

    A("\n## Results\n")
    A("| Model | Config | Cap | Test | Mean tok/s | min | max | Spread | Peak junction |")
    A("|---|---|---|---|---:|---:|---:|---:|---:|")
    for cell in record["cells"]:
        if cell.get("error"):
            A(f"| {cell['model']} | {cell['config']} | — | — | — | — | — | — | "
              f"{cell['error']} |")
            continue
        stats = summarise(cell)
        pj = max((r["peak"]["junction_c"] for r in cell["runs"]), default="—")
        for test, s in stats.items():
            A(f"| {cell['model']} | {cell['config']} | {cell['power_cap_w']} | {test} "
              f"| {s['mean']:.2f} | {s['min']:.2f} | {s['max']:.2f} "
              f"| {s['spread_pct']:.1f}% | {pj} °C |")

    A("\n## How to read this\n")
    A("**Spread matters more than the mean.** Decode throughput on this class of card "
      "is documented as *bimodal* — the card settles into one of two modes and stays "
      "there for the life of the process. That is why every run here is a separate "
      "process rather than `-r N`.\n")
    A("- **Spread below ~2 %** → bimodality did not appear; the mean is a sensible "
      "number to quote.")
    A("- **Spread above ~10 %, results in two clusters** → that is the bimodality. "
      "**Report both modes, not their mean** — the mean describes a state the card "
      "is never in.\n")
    A("**Peak junction near 90 °C** means you are measuring case airflow, not the "
      "model. Lower the power cap and repeat.\n")
    A("**Peak power below ~60 W on an otherwise valid result does not mean the GPU "
      "idled.** It means the compute window was shorter than the telemetry sampling "
      "period. For a fast MoE model at `-p 512 -n 128` the whole computation can take "
      "under a second, with the rest of the wall time spent loading weights — so the "
      "**thermal figures are meaningless while the throughput stays valid.** Use a "
      "longer test (e.g. `-p 4096 -n 2048`) for thermal work.\n")
    A("**The PCIe link** is read under load, because at idle it downtrains to save "
      "power. If you add a second GPU and the first drops from x16 to x8, the same "
      "config file gives you a directly comparable number.\n")
    return "\n".join(L) + "\n"


def compare(dir_a, dir_b):
    a = json.loads((Path(dir_a) / "results.json").read_text())
    b = json.loads((Path(dir_b) / "results.json").read_text())

    def flat(rec):
        out = {}
        for cell in rec["cells"]:
            for test, s in summarise(cell).items():
                out[(cell["model"], cell["config"], test)] = s["mean"]
        return out

    fa, fb = flat(a), flat(b)
    print(f"\nA = {a['run_id']}  ({a['environment']['llama_cpp_commit']}, "
          f"cap {a['environment']['power_cap']})")
    print(f"B = {b['run_id']}  ({b['environment']['llama_cpp_commit']}, "
          f"cap {b['environment']['power_cap']})\n")
    if a['environment']['llama_cpp_commit'] != b['environment']['llama_cpp_commit']:
        print("  NOTE: different llama.cpp commits — do not attribute the delta to "
              "hardware.\n")
    print(f"{'model/config/test':<46} {'A':>11} {'B':>11} {'change':>9}")
    print("-" * 80)
    for key in sorted(set(fa) | set(fb)):
        va, vb = fa.get(key), fb.get(key)
        label = "/".join(key)
        if va and vb:
            print(f"{label:<46} {va:>11.2f} {vb:>11.2f} {(vb - va) / va * 100:>+8.1f}%")
        else:
            print(f"{label:<46} {va or '—':>11} {vb or '—':>11} {'—':>9}")
    print()


def main():
    ap = argparse.ArgumentParser(
        description="Declarative benchmark harness for local LLM inference")
    ap.add_argument("config", nargs="?", help="JSON file describing the matrix")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="compare two result directories")
    ap.add_argument("--list", action="store_true", help="models and configs")
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    if args.list:
        models_dir = Path(os.environ.get("RIG_MODELS_DIR", Path.home() / "models"))
        print(f"\nModels in {models_dir}:")
        for p in sorted(models_dir.rglob("*.gguf")):
            print(f"  {p.stat().st_size / 1e9:6.1f} GB  {p}")
        print("\nConfig files:")
        for p in sorted((BASE / "configs").glob("*.json")):
            try:
                c = json.loads(p.read_text())
                print(f"  {p.name:<28} {c.get('name', '')} "
                      f"({len(c.get('models', []))} models, "
                      f"{len(c.get('configs', []))} configs)")
            except Exception:
                print(f"  {p.name:<28} (parse error)")
        print("\nResults:")
        for p in sorted(RESULTS.glob("*")):
            if (p / "results.json").exists():
                print(f"  {p.name}")
        print()
        return

    if not args.config:
        ap.error("pass a config file, or --list / --compare")

    if not shutil.which("amd-smi"):
        print("ERROR: amd-smi not found — telemetry and power cap will not work.")
        sys.exit(1)

    cfg = json.loads(Path(args.config).read_text())
    cfg["_source"] = str(Path(args.config).resolve())
    execute(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
