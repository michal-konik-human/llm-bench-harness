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

  model: architecture, parameters,   a "70B" and a "70B-A3B" are different animals;
    quantisation, size                 the quant changes both speed and quality
  llama.cpp commit + build flags     results are not comparable across builds
  llama-bench's EFFECTIVE params     flash attention, batch/ubatch, split, load mode —
                                     read back from llama-bench's own JSON, not assumed
  ROCm version, kernel cmdline       same
  per-GPU power cap and fan curve    300 W and 210 W are two different measurements
  PCIe link, bottleneck of the path  read *under load*; the card's own endpoint always
                                     says x16 (it talks to a switch on the card) — the
                                     real width is further up, at the root port
  peak junction / VRAM temp, power   per GPU, for EVERY GPU the config uses; the thermal
                                     guard watches all of them
  environment variables              HIP_VISIBLE_DEVICES, GGML_*, NCCL_* change everything
  ECC + AER counters before/after    per GPU; if they moved, the hardware misbehaved and
                                     the result must be discarded, not adjusted
  other processes holding the GPUs   a second consumer makes the run measure contention
  CPU power profile                  a 'powersave' CPU slows prompt feeding and loading

USAGE

  ./bench.py configs/single-gpu.json              run the matrix
  ./bench.py configs/single-gpu.json --dry-run    print the plan, run nothing
  ./bench.py --compare results/A results/B        diff two runs
  ./bench.py --list                               available models and configs
  ./bench.py --reindex                            rebuild results/all-benchmark-results.md

Every run writes results/<timestamp>/ with:
  report.md     detailed human-readable report, leads with a verdict
  results.json  full machine-readable data
  results.csv   one row per cell/test, for a spreadsheet
  raw.log       raw llama-bench output (stdout JSON + stderr log) of every process
and appends one row per result to the shared table results/all-benchmark-results.md.

CONFIG FILE RULES

  * One config = one parameter set. llama-bench treats a COMMA in a value as "sweep
    over these values" (-p 512,1024 runs two tests). The harness keys results by test
    name, so sweeps are rejected — write one config per value instead.
  * The one exception is the tensor split: llama-bench's separator is '/', not ','.
    `-ts 1,1,1,1` means four sweep values of "1" = everything on GPU 0, which fails
    with a bare "failed to load model" on anything larger than one card. A comma-form
    `-ts` is converted to the slash form automatically, with a warning.
  * GPUs are taken from the config's env HIP_VISIBLE_DEVICES. Telemetry, the thermal
    guard, error counters and power caps apply to exactly those GPUs. If it is unset,
    RIG_GPU_INDEX (default 0) is measured and HIP_VISIBLE_DEVICES is set to it, so a
    run can never silently use the integrated GPU.

Optional per-file settings: runs (3), cooldown_s (45), cooldown_max_s (300),
start_temp_c (50: next run waits until every GPU is below this), temp_limit_c (95),
vram_temp_limit_c (95).

Overrides: RIG_GPU_INDEX (default GPU when a config has no HIP_VISIBLE_DEVICES),
LLAMA_CPP_DIR (~/llama.cpp), RIG_MODELS_DIR (~/models).

Requirements: ROCm with `amd-smi` and `rocminfo`, a built `llama-bench`, Python 3.10+.
No third-party Python packages.

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
import struct
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
INDEX = RESULTS / "all-benchmark-results.md"

DEFAULT_GPU_INDEX = int(os.environ.get("RIG_GPU_INDEX", "0"))

# Junction temperature at which a run is aborted. The card itself throttles around
# 110 C and shuts down around 115 C, but GDDR6 degrades *permanently* above 100 C,
# so we stop well before the firmware would ever intervene.
DEFAULT_TEMP_LIMIT = 95
DEFAULT_VRAM_TEMP_LIMIT = 95

# Telemetry sampling period. One amd-smi call for four GPUs takes ~0.1 s, so 1 s is
# cheap. Keep it well below the length of your shortest test, or peak readings are
# meaningless — see docs/METHODOLOGY.md about short compute windows.
TELEMETRY_INTERVAL = 1.0

# Flags whose comma form is a sweep in llama-bench (all value flags are, in fact).
TS_FLAGS = {"-ts", "--tensor-split"}

BDF_RE = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")


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
    AttributeError — which kills the collection loop mid-cycle.
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
    except (OSError, TypeError):
        return default


def amd_smi_list(data):
    """amd-smi JSON is either a list or {"gpu_data": [...]} depending on version."""
    if isinstance(data, dict):
        data = data.get("gpu_data")
    return data if isinstance(data, list) else []


def scrub(text):
    """Replace the home directory with '~' in anything written to results/.

    Results are meant to be publishable; raw llama-bench output and error messages carry
    absolute paths that would leak the username for no analytical benefit.
    """
    home = str(Path.home())
    return text.replace(home + "/", "~/").replace(home, "~") if home not in ("/", "") else text


def home_rel(path):
    """Home-relative path: results are meant to be publishable, and an absolute path
    leaks the username for no analytical benefit."""
    try:
        return "~/" + str(Path(path).resolve().relative_to(Path.home().resolve()))
    except (ValueError, OSError):
        return str(path)


# ------------------------------------------------------------ device discovery

def amd_inventory():
    """amd-smi index -> {name, gfx, bdf, cus, vram_mb}, for every GPU incl. the iGPU."""
    out = {}
    for e in amd_smi_list(sh_json(["amd-smi", "static", "--json"], timeout=30)):
        asic, bus, vram = dsec(e, "asic"), dsec(e, "bus"), dsec(e, "vram")
        idx = e.get("gpu")
        if not isinstance(idx, int):
            continue
        size = leaf(vram.get("size"))
        out[idx] = {
            "amd_index": idx,
            "name": leaf(asic.get("market_name")),
            "gfx": leaf(asic.get("target_graphics_version")),
            "bdf": (leaf(bus.get("bdf")) or "").lower() or None,
            "cus": leaf(asic.get("num_compute_units")),
            "vram_mb": size if isinstance(size, int) else None,
        }
    return out


def hip_order():
    """BDFs of GPU agents in HIP enumeration order (what HIP_VISIBLE_DEVICES indexes).

    Read from rocminfo, whose GPU agents appear in the same order HIP numbers them.
    On this machine that matches amd-smi's order, but nothing guarantees it — so the
    harness maps through the PCI address instead of assuming.
    """
    rc, out, _ = sh(["rocminfo"], timeout=30)
    bdfs = []
    if rc != 0:
        return bdfs
    for block in out.split("*******"):
        if "Device Type:" not in block or not re.search(r"Device Type:\s+GPU", block):
            continue
        m = re.search(r"BDFID:\s+(\d+)", block)
        if m:
            v = int(m.group(1))
            bdfs.append(f"0000:{(v >> 8) & 0xff:02x}:{(v >> 3) & 0x1f:02x}.{v & 0x7}")
    return bdfs


INVENTORY = None
HIP_BDFS = None


def discover():
    global INVENTORY, HIP_BDFS
    if INVENTORY is None:
        INVENTORY = amd_inventory()
        HIP_BDFS = hip_order()


def is_igpu(g):
    """An integrated GPU: 'Radeon Graphics' / gfx103x (RDNA2 iGPU) / small carve-out."""
    name, gfx = str(g.get("name") or ""), str(g.get("gfx") or "")
    return ("Radeon Graphics" in name and "PRO" not in name) or gfx.startswith("gfx103") \
        or (isinstance(g.get("vram_mb"), int) and g["vram_mb"] < 8192)


def gpus_for(cfg):
    """The GPUs a config runs on: list of inventory dicts (with 'hip_index').

    Parsed from the config's HIP_VISIBLE_DEVICES and mapped HIP index -> PCI address ->
    amd-smi index, so telemetry and power caps land on the cards actually used.
    """
    discover()
    hip = str((cfg.get("env") or {}).get("HIP_VISIBLE_DEVICES", "")).strip()
    if not hip:
        idxs = [DEFAULT_GPU_INDEX]
    else:
        idxs = [int(p) for p in hip.split(",") if p.strip().isdigit()]
    by_bdf = {g["bdf"]: g for g in INVENTORY.values() if g.get("bdf")}
    out = []
    for i in idxs:
        g = None
        if HIP_BDFS and i < len(HIP_BDFS):
            g = by_bdf.get(HIP_BDFS[i])
        if g is None:                       # fall back to identical numbering
            g = INVENTORY.get(i)
        if g is None:
            out.append({"hip_index": i, "amd_index": None, "name": "?", "bdf": None})
        else:
            out.append(dict(g, hip_index=i))
    return out


def pci_path(bdf):
    """Every PCI function from the root port down to the device."""
    try:
        real = os.path.realpath(f"/sys/bus/pci/devices/{bdf}")
    except OSError:
        return []
    return [p for p in real.split("/") if BDF_RE.match(p)]


def _speed(s):
    m = re.match(r"([\d.]+)", s or "")
    return float(m.group(1)) if m else None


def link_bottleneck(bdf):
    """Narrowest/slowest link on the path from the root port to the GPU.

    A Radeon AI PRO R9700 carries its own PCIe switch: the GPU endpoint always reports
    x16 at full speed (to that switch), while the slot link above it may be x8 or x4.
    Reading only the endpoint — as this harness used to — reports x16 for every card.
    At idle the link also downtrains, which is why this is sampled under load.
    """
    worst_w, worst_s, where = None, None, None
    for dev in pci_path(bdf):
        base = f"/sys/bus/pci/devices/{dev}"
        w = read_file(f"{base}/current_link_width")
        s = read_file(f"{base}/current_link_speed")
        try:
            w = int(w)
        except (TypeError, ValueError):
            continue
        if w <= 0:
            continue
        sp = _speed(s)
        if worst_w is None or w < worst_w or (w == worst_w and sp and worst_s and sp < worst_s):
            worst_w, worst_s, where = w, sp, dev
        elif sp and worst_s and sp < worst_s:
            worst_s = sp
    if worst_w is None:
        return {"width": None, "speed_gts": None, "at": None}
    return {"width": worst_w, "speed_gts": worst_s, "at": where}


def fmt_link(l):
    if not l or not l.get("width"):
        return "?"
    gen = {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}.get(l.get("speed_gts"))
    return f"x{l['width']}" + (f" Gen{gen}" if gen else (f" {l['speed_gts']} GT/s" if l.get("speed_gts") else ""))


def hwmon_dir(bdf):
    try:
        return next(Path(f"/sys/bus/pci/devices/{bdf}/hwmon").glob("hwmon*"))
    except (StopIteration, OSError):
        return None


def power_cap_w(g):
    """Current power cap of one GPU, in W (sysfs, no root, no wake-up needed)."""
    h = hwmon_dir(g.get("bdf")) if g.get("bdf") else None
    v = read_file(h / "power1_cap") if h else None
    try:
        return round(int(v) / 1_000_000)
    except (TypeError, ValueError):
        return None


def fan_curve(g):
    """User fan curve from the amdgpu overdrive interface, e.g. '55:15 65:30 ...'.

    'default' when overdrive is off (no gpu_od directory) or the curve is unset (all 0).
    """
    txt = read_file(f"/sys/bus/pci/devices/{g.get('bdf')}/gpu_od/fan_ctrl/fan_curve")
    if not txt:
        return "default"
    pts = re.findall(r"^\s*\d+:\s*(\d+)C\s+(\d+)%", txt, re.M)
    if not pts or all(t == "0" and p == "0" for t, p in pts):
        return "default"
    return " ".join(f"{t}:{p}" for t, p in pts)


def aer_totals(bdf):
    """PCIe AER counters of the GPU and of every link above it (sysfs, no root)."""
    out = {"correctable": 0, "nonfatal": 0, "fatal": 0}
    seen = False
    for dev in pci_path(bdf) if bdf else []:
        for kind in out:
            raw = read_file(f"/sys/bus/pci/devices/{dev}/aer_dev_{kind}")
            if not raw:
                continue
            seen = True
            for line in raw.splitlines():
                parts = line.split()
                if len(parts) == 2 and parts[0].startswith("TOTAL"):
                    try:
                        out[kind] += int(parts[1])
                    except ValueError:
                        pass
    return out if seen else {k: None for k in out}


def ecc_totals(amd_indices):
    out = {}
    data = sh_json(["amd-smi", "metric", "--ecc", "-g", *map(str, amd_indices), "--json"],
                   timeout=30)
    for e in amd_smi_list(data):
        ecc = dsec(e, "ecc")
        out[str(e.get("gpu"))] = {
            "correctable": leaf(ecc.get("total_correctable_count")),
            "uncorrectable": leaf(ecc.get("total_uncorrectable_count")),
        }
    return out


def error_snapshot(gpus):
    return {"aer": {g["bdf"]: aer_totals(g["bdf"]) for g in gpus if g.get("bdf")},
            "ecc": ecc_totals([g["amd_index"] for g in gpus if g.get("amd_index") is not None])}


# ------------------------------------------------------------------- telemetry

def gpu_metrics(amd_indices):
    """One amd-smi call for all GPUs -> {amd_index: metrics}.

    Read through amd-smi, NOT sysfs: a discrete GPU with no display attached gets
    parked in D3hot by runtime power management, and sysfs hwmon reads then return
    EBUSY. amd-smi wakes the card by a different path and always returns real values.
    """
    if not amd_indices:
        return {}
    data = sh_json(["amd-smi", "metric", "-g", *map(str, amd_indices), "--json"], timeout=20)
    out = {}
    for g in amd_smi_list(data):
        t, p, f, u = (dsec(g, "temperature"), dsec(g, "power"),
                      dsec(g, "fan"), dsec(g, "usage"))
        out[g.get("gpu")] = {
            "junction_c": leaf(t.get("hotspot")),
            "edge_c": leaf(t.get("edge")),
            "vram_c": leaf(t.get("mem")),
            "power_w": leaf(p.get("socket_power")),
            "fan_rpm": leaf(f.get("rpm")),
            "gfx_pct": leaf(u.get("gfx_activity")),
        }
    return out


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def gpu_consumers(amd_indices):
    """Compute processes currently holding any of these GPUs.

    Anything besides our own measurement invalidates the result. The realistic causes
    are a forgotten llama-server, a model left loaded by llama-swap, or an agent's own
    model on the GPU. (Desktop graphics — Xwayland, a browser — do not appear here and
    do not count; they use the graphics queue, not compute.)
    """
    data = sh_json(["amd-smi", "process", "--json"], timeout=25)
    out = []
    for entry in data if isinstance(data, list) else []:
        if not isinstance(entry, dict) or entry.get("gpu") not in amd_indices:
            continue
        plist = entry.get("process_list")
        for item in plist if isinstance(plist, list) else []:
            info = item.get("process_info") if isinstance(item, dict) else None
            if isinstance(info, dict) and info.get("name"):
                out.append({"gpu": entry.get("gpu"), "name": info["name"],
                            "pid": info.get("pid")})
    return out


def llama_swap_running():
    """Models llama-swap currently has loaded (they hold VRAM), or None if not running."""
    try:
        with urllib.request.urlopen("http://127.0.0.1:8080/running", timeout=2) as r:
            data = json.loads(r.read().decode())
        return [m.get("model") for m in data.get("running", []) if isinstance(m, dict)]
    except Exception:
        return None


# ----------------------------------------------------------------- model facts

GGUF_TYPES = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
              10: "<Q", 11: "<q", 12: "<d"}
WANTED_KEYS = ("general.architecture", "general.name", "general.size_label",
               "general.file_type", "general.quantized_by", "split.count")
WANTED_SUFFIX = (".expert_count", ".expert_used_count", ".context_length", ".block_count")


def gguf_metadata(path):
    """Scalar metadata from a GGUF header (architecture, size label, experts, ...).

    Minimal pure-Python reader: walks the key/value section, skipping arrays (the
    tokenizer vocabulary) without keeping them. Only the first shard carries metadata.
    """
    out = {}
    try:
        with open(path, "rb") as fh:
            if fh.read(4) != b"GGUF":
                return out
            version = struct.unpack("<I", fh.read(4))[0]
            if version < 2:
                return out
            _, n_kv = struct.unpack("<QQ", fh.read(16))

            def rstr():
                (n,) = struct.unpack("<Q", fh.read(8))
                return fh.read(n).decode("utf-8", errors="replace")

            def rval(t):
                if t == 8:
                    return rstr()
                if t == 9:
                    at, n = struct.unpack("<IQ", fh.read(12))
                    for _ in range(n):
                        rval(at)
                    return None
                fmt = GGUF_TYPES[t]
                return struct.unpack(fmt, fh.read(struct.calcsize(fmt)))[0]

            for _ in range(n_kv):
                key = rstr()
                (t,) = struct.unpack("<I", fh.read(4))
                val = rval(t)
                if key in WANTED_KEYS or key.endswith(WANTED_SUFFIX):
                    out[key] = val
    except Exception:
        pass
    return out


QUANT_RE = re.compile(r"((?:UD-)?(?:I?Q\d(?:_[A-Z0-9]+)*|MXFP4|BF16|F16|F32))(?=[.\-]|$)", re.I)


def model_files(path):
    """All shards of a (possibly split) model, e.g. X-00001-of-00004.gguf."""
    p = Path(path)
    m = re.match(r"(.*)-(\d{5})-of-(\d{5})\.gguf$", p.name)
    if not m:
        return [p]
    return sorted(p.parent.glob(f"{m.group(1)}-*-of-{m.group(3)}.gguf"))


SKIP_FILE_RE = re.compile(r"mmproj|eagle|draft|-(?!00001)\d{5}-of-\d{5}\.gguf$", re.I)


def model_catalog(models_dir=None):
    """Every model under the models dir: {name: {"path", "files", "bytes", "quant"}}.

    One model = one directory (the harness convention). Names:
      ~/models/qwen3-32b/X.gguf                 -> qwen3-32b
      ~/models/qwen3-next-80b/Q4_1/X-00001...   -> qwen3-next-80b   (quant sub-folder)
      ~/models/DeepSeek-V4-Flash-GGUF/UD-IQ3_XXS/...  -> deepseek-v4-flash
    and if two folders would get the same name (two quants of one model), the quant is
    appended: qwen3-235b-a22b-ud-q2_k_xl / qwen3-235b-a22b-ud-q3_k_xl.
    Size counts ALL shards — the first shard of a split model can be a 5 MB header.
    Vision projectors (mmproj) and speculative-decoding drafts (eagle/draft) are skipped.
    """
    root = Path(models_dir or os.environ.get("RIG_MODELS_DIR", Path.home() / "models"))
    found = {}
    for p in sorted(root.rglob("*.gguf")) if root.exists() else []:
        if SKIP_FILE_RE.search(p.name):
            continue
        d = p.parent
        quant_dir = bool(QUANT_RE.fullmatch(d.name))
        base_dir = d.parent if quant_dir and d.parent != root else d
        base = re.sub(r"-gguf$", "", base_dir.name, flags=re.I).lower() if base_dir != root \
            else p.stem.lower()
        files = model_files(p)
        size = sum(f.stat().st_size for f in files if f.exists())
        q = QUANT_RE.search(p.name)
        key = (base, str(d))
        if key not in found or size > found[key]["bytes"]:
            found[key] = {"base": base, "path": p, "files": files, "bytes": size,
                          "quant": q.group(1) if q else None}
    by_base = {}
    for (base, _), m in found.items():
        by_base.setdefault(base, []).append(m)
    out = {}
    for base, ms in by_base.items():
        for m in ms:
            name = base if len(ms) == 1 else f"{base}-{(m['quant'] or 'x').lower()}"
            out[name] = m
    return dict(sorted(out.items()))


def model_facts(path, bench_params=None):
    """What the model IS: architecture, size, experts, quant — for every report row."""
    meta = gguf_metadata(path)
    arch = meta.get("general.architecture")
    files = model_files(path)
    size = sum(f.stat().st_size for f in files if f.exists())
    bp = bench_params or {}
    facts = {
        "architecture": arch,
        "name": meta.get("general.name"),
        "size_label": meta.get("general.size_label"),
        "experts": meta.get(f"{arch}.expert_count") if arch else None,
        "experts_used": meta.get(f"{arch}.expert_used_count") if arch else None,
        "context_length": meta.get(f"{arch}.context_length") if arch else None,
        "layers": meta.get(f"{arch}.block_count") if arch else None,
        "shards": len(files),
        "file_size_gib": round(size / 2**30, 2) if size else None,
        # llama-bench's own description, e.g. "qwen3moe 235B.A22B Q3_K - Medium"
        "llama_type": bp.get("model_type"),
        "n_params": bp.get("model_n_params"),
        "loaded_size_gib": round(bp["model_size"] / 2**30, 2) if bp.get("model_size") else None,
    }
    # The quant as its publisher names it (UD-Q3_K_XL, Q4_K_M, MXFP4...) is in the file
    # name; llama.cpp's own type string only names the dominant tensor type.
    q = QUANT_RE.search(Path(path).name) or QUANT_RE.search(facts["llama_type"] or "")
    facts["quant"] = q.group(1) if q else None
    return facts


def params_label(f):
    """'235B-A22B (MoE 128/8)' or '32.8B (dense)'."""
    label = f.get("size_label")
    if not label and f.get("n_params"):
        label = f"{f['n_params'] / 1e9:.1f}B"
    if f.get("experts"):
        return f"{label or '?'} (MoE {f['experts']}/{f.get('experts_used') or '?'})"
    return f"{label or '?'} (dense)" if label else "?"


# --------------------------------------------------------------- configuration

def cpu_state():
    gov = read_file("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    epp = read_file("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference")
    rc, prof, _ = sh(["powerprofilesctl", "get"], timeout=10)
    return {"profile": prof.strip() if rc == 0 else None, "governor": gov, "epp": epp}


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
    board = read_file("/sys/class/dmi/id/board_name")
    bios = bios.strip() or read_file("/sys/class/dmi/id/bios_version")
    return {
        "cpu": cpu,
        "cpu_state": cpu_state(),
        "board": board,
        "ram_gb": round(ram_kb / 1024 / 1024, 1) if ram_kb else None,
        "bios": bios or None,
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
    rc, tuning, _ = sh(["systemctl", "is-active", "gpu-tuning"], timeout=10)
    return {
        "kernel": (sh(["uname", "-r"])[1] or "").strip(),
        "kernel_cmdline": read_file("/proc/cmdline", "?"),
        "amdgpu_ppfeaturemask": read_file("/sys/module/amdgpu/parameters/ppfeaturemask"),
        "rocm": read_file("/opt/rocm/.info/version", "?"),
        "llama_cpp_commit": commit.strip() or "?",
        "llama_cpp_commit_date": cdate.strip() or None,
        "llama_cpp_build": flags,
        "gpu_tuning_service": tuning.strip() or None,
    }


def gpu_state(g):
    """Static + setting state of one GPU at run time."""
    return {
        "hip_index": g.get("hip_index"), "amd_index": g.get("amd_index"),
        "name": g.get("name"), "gfx": g.get("gfx"), "bdf": g.get("bdf"),
        "cus": g.get("cus"), "vram_mb": g.get("vram_mb"),
        "vbios": read_file(f"/sys/bus/pci/devices/{g.get('bdf')}/vbios_version"),
        "subsystem": "{}:{}".format(
            read_file(f"/sys/bus/pci/devices/{g.get('bdf')}/subsystem_vendor", "?"),
            read_file(f"/sys/bus/pci/devices/{g.get('bdf')}/subsystem_device", "?")),
        "power_cap_w": power_cap_w(g),
        "fan_curve": fan_curve(g),
        "link_idle": link_bottleneck(g["bdf"]) if g.get("bdf") else None,
    }


def environment(all_gpus):
    discover()
    sw = software_config()
    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "kernel": sw["kernel"],
        "rocm": sw["rocm"],
        "llama_cpp_commit": sw["llama_cpp_commit"],
        "inventory": [dict(g, hip_index=(HIP_BDFS.index(g["bdf"])
                                         if HIP_BDFS and g.get("bdf") in HIP_BDFS else None))
                      for g in INVENTORY.values()],
        "gpus_used": [gpu_state(g) for g in all_gpus],
        "hardware": hardware_config(),
        "software": sw,
        "gpu_consumers_at_start": gpu_consumers([g["amd_index"] for g in all_gpus
                                                 if g.get("amd_index") is not None]),
        "llama_swap_loaded_at_start": llama_swap_running(),
        "errors_before": error_snapshot(all_gpus),
    }


def set_power_caps(gpus, watts):
    """Set the power cap on every GPU of a config. Returns (ok, message).

    Cards have a firmware *floor*: on the Radeon AI PRO R9700 it is 210 W; with amdgpu
    overdrive enabled the ceiling rises from 300 to 330 W.
    """
    msgs, ok = [], True
    for g in gpus:
        rc, out, err = sh(["sudo", "-n", "amd-smi", "set", "-g", str(g["amd_index"]),
                           "--power-cap", str(watts)], timeout=40)
        if rc != 0:
            ok = False
            m = (err or out or "unknown error").strip().splitlines()
            msgs.append(f"gpu{g['amd_index']}: {m[-1][:160] if m else 'error'}")
    time.sleep(2)
    now = ", ".join(f"gpu{g['amd_index']}={power_cap_w(g)} W" for g in gpus)
    return ok, (now if ok else "; ".join(msgs) + f" (now: {now})")


# ------------------------------------------------------------- config checks

def normalise_args(args):
    """Validate llama-bench args. Returns (args, warnings, errors)."""
    args, warns, errs = [str(a) for a in args], [], []
    for i, a in enumerate(args[:-1]):
        v = args[i + 1]
        if not a.startswith("-") or v.startswith("-"):
            continue
        if "," in v:
            if a in TS_FLAGS:
                args[i + 1] = v.replace(",", "/")
                warns.append(f"{a} {v} -> {args[i + 1]} (llama-bench's tensor-split "
                             f"separator is '/'; ',' would make it a sweep)")
            else:
                errs.append(f"{a} {v}: a comma is a sweep in llama-bench; write one "
                            f"config per value")
    if "-o" in args or "--output" in args:
        errs.append("do not pass -o/--output; the harness sets -o json itself")
    if "-r" in args or "--repetitions" in args:
        errs.append("do not pass -r; the harness runs separate processes instead "
                    "(set 'runs' in the config file)")
    return args, warns, errs


def check_config(config):
    """Everything wrong with a config file, before anything runs."""
    problems, warnings = [], []
    for c in config.get("configs", []):
        c["args"], w, e = normalise_args(c.get("args", ["-ngl", "999", "-sm", "none",
                                                        "-p", "512", "-n", "128"]))
        warnings += [f"[{c['name']}] {x}" for x in w]
        problems += [f"[{c['name']}] {x}" for x in e]
        env = c.setdefault("env", {})
        if not str(env.get("HIP_VISIBLE_DEVICES", "")).strip():
            env["HIP_VISIBLE_DEVICES"] = str(DEFAULT_GPU_INDEX)
            warnings.append(f"[{c['name']}] no HIP_VISIBLE_DEVICES — using GPU "
                            f"{DEFAULT_GPU_INDEX} (RIG_GPU_INDEX)")
        gs = gpus_for(c)
        for g in gs:
            if g.get("amd_index") is None:
                problems.append(f"[{c['name']}] HIP device {g['hip_index']} does not exist")
            elif is_igpu(g) and not c.get("allow_igpu"):
                problems.append(f"[{c['name']}] HIP device {g['hip_index']} is the "
                                f"INTEGRATED GPU ({g.get('name')}) — results would be ~10x "
                                f"too low. Fix HIP_VISIBLE_DEVICES (or set allow_igpu)")
        ts = next((c["args"][i + 1] for i, a in enumerate(c["args"][:-1]) if a in TS_FLAGS), None)
        if ts and len(ts.split("/")) != len(gs):
            warnings.append(f"[{c['name']}] tensor split {ts} has {len(ts.split('/'))} "
                            f"parts but {len(gs)} GPU(s) are visible")
    for m in config.get("models", []):
        if not Path(os.path.expanduser(m["path"])).exists():
            problems.append(f"model {m['name']}: file not found: {m['path']}")
    return problems, warnings


# ----------------------------------------------------------------- measurement

BENCH_ROW = re.compile(
    r"\|\s*(?P<test>(?:pp|tg)\d+(?:\+tg\d+)?)\s*\|\s*(?P<val>[\d.]+)\s*(?:±\s*(?P<sd>[\d.]+))?\s*\|")


def test_name(e):
    npp, ntg = e.get("n_prompt", 0), e.get("n_gen", 0)
    name = (f"pp{npp}" if npp else "") + ("+" if npp and ntg else "") + (f"tg{ntg}" if ntg else "")
    if e.get("n_depth"):
        name += f"@d{e['n_depth']}"
    return name or "?"


def parse_bench(stdout):
    """(results, bench_params). Prefers llama-bench's JSON; falls back to the md table."""
    try:
        start = stdout.index("[")
        data = json.loads(stdout[start:stdout.rindex("]") + 1])
    except (ValueError, json.JSONDecodeError):
        data = None
    if isinstance(data, list) and data:
        res, dup = {}, False
        for e in data:
            t = test_name(e)
            dup = dup or t in res
            res[t] = {"tok_s": float(e.get("avg_ts", 0)), "sd": e.get("stddev_ts")}
        params = {k: v for k, v in data[0].items()
                  if not isinstance(v, (list, dict)) and k not in
                  ("avg_ts", "stddev_ts", "avg_ns", "stddev_ns", "test_time",
                   "n_prompt", "n_gen", "n_depth", "model_filename")}
        if dup:
            params["_warning"] = "duplicate test names — config produced a sweep"
        return res, params
    res = {}
    for line in stdout.splitlines():
        m = BENCH_ROW.search(line)
        if m:
            res[m.group("test")] = {"tok_s": float(m.group("val")),
                                    "sd": float(m.group("sd")) if m.group("sd") else None}
    return res, {}


def error_lines(stderr, n=4):
    keep = [l.strip() for l in stderr.splitlines()
            if re.search(r"error|failed|abort|out of memory|oom|cannot|unable", l, re.I)]
    return keep[-n:]


def run_once(model_path, cfg, gpus, limits, log_fh):
    """One INDEPENDENT measurement process, with telemetry and a thermal guard over
    EVERY GPU the config uses.

    A separate OS process per run is deliberate. Decode throughput on some cards is
    bimodal: the card settles into one of two modes and stays there for the lifetime of
    the process. `llama-bench -r N` averages N samples *inside a single mode* and
    reports falsely tiny variance. Only separate processes can land in both.
    """
    binary = os.path.expanduser(
        cfg.get("binary", str(Path.home() / "llama.cpp/build/bin/llama-bench")))
    env = dict(os.environ)
    env.update({k: str(v) for k, v in cfg.get("env", {}).items()})
    cmd = [binary, "-m", str(model_path), "-r", "1", "-o", "json"] + cfg["args"]

    idx = [g["amd_index"] for g in gpus if g.get("amd_index") is not None]
    t0 = time.time()
    # stdout/stderr go to temporary FILES, not pipes: a large model load writes
    # hundreds of loader lines to stderr, and an undrained pipe (64 KiB) would block
    # llama-bench mid-load while this loop waits for it — a silent hang.
    out_f = tempfile.TemporaryFile(mode="w+")
    err_f = tempfile.TemporaryFile(mode="w+")
    proc = subprocess.Popen(cmd, stdout=out_f, stderr=err_f,
                            text=True, env=env, start_new_session=True)
    peak = {i: {"junction_c": 0, "vram_c": 0, "power_w": 0, "fan_rpm": 0} for i in idx}
    link_load = {}
    samples, aborted, total_pw, total_peak = 0, None, [], 0

    while proc.poll() is None:
        ms = gpu_metrics(idx)
        if ms:
            samples += 1
            tot = 0
            for i, m in ms.items():
                if i not in peak:
                    continue
                for k in peak[i]:
                    if _num(m.get(k)) and m[k] > peak[i][k]:
                        peak[i][k] = m[k]
                if _num(m.get("power_w")):
                    tot += m["power_w"]
                # Only meaningful while the GPU is busy — links downtrain at idle.
                if _num(m.get("gfx_pct")) and m["gfx_pct"] > 10:
                    g = next(x for x in gpus if x.get("amd_index") == i)
                    link_load[i] = link_bottleneck(g["bdf"])
                j, v = m.get("junction_c"), m.get("vram_c")
                if (_num(j) and j >= limits["junction"]) or (_num(v) and v >= limits["vram"]):
                    aborted = {"gpu": i, "junction_c": j, "vram_c": v}
            total_pw.append(tot)
            total_peak = max(total_peak, tot)
            if aborted:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                except Exception:
                    proc.terminate()
                break
        time.sleep(TELEMETRY_INTERVAL)

    try:
        proc.wait(timeout=60)
    except Exception:
        proc.kill()
    out_f.seek(0), err_f.seek(0)
    stdout, stderr = out_f.read(), err_f.read()
    out_f.close(), err_f.close()
    wall = time.time() - t0
    log_fh.write(scrub(f"$ {' '.join(cmd)}\n# env HIP_VISIBLE_DEVICES="
                       f"{env.get('HIP_VISIBLE_DEVICES', '')}\n{stdout}\n--- stderr ---\n"
                       f"{stderr}\n" + "-" * 70 + "\n"))
    log_fh.flush()

    results, params = parse_bench(stdout) if not aborted else ({}, {})
    for g in gpus:                       # a GPU never busy enough: record the idle link
        i = g.get("amd_index")
        if i is not None and i not in link_load and g.get("bdf"):
            link_load[i] = dict(link_bottleneck(g["bdf"]), idle=True)
    err = None
    if not results and not aborted:
        err = scrub("; ".join(error_lines(stderr)) or f"llama-bench exit code {proc.returncode}")
    return {
        "results": results,
        "bench_params": params,
        "peak": peak,                                  # per GPU (amd index)
        "peak_total_power_w": round(total_peak, 1),
        "mean_total_power_w": round(statistics.fmean(total_pw), 1) if total_pw else None,
        "wall_s": round(wall, 1),
        "samples": samples,
        "aborted": aborted,
        "link_under_load": {str(k): v for k, v in link_load.items()},
        "error": err,
        "ok": aborted is None and bool(results),
    }


def wait_cool(gpus, min_s, max_s, start_temp):
    """Sleep at least min_s, then until every GPU's junction <= start_temp (max max_s).

    A run that starts on a warm card measures low; a fixed sleep either wastes time or
    is too short, depending on the model.
    """
    idx = [g["amd_index"] for g in gpus if g.get("amd_index") is not None]
    time.sleep(min_s)
    waited = min_s
    while waited < max_s:
        ms = gpu_metrics(idx)
        hot = [m.get("junction_c") for m in ms.values() if _num(m.get("junction_c"))]
        if not hot or max(hot) <= start_temp:
            break
        time.sleep(5)
        waited += 5
    return waited


def execute(config, dry_run=False):
    models = config["models"]
    configs = config["configs"]
    runs = config.get("runs", 3)
    cooldown = config.get("cooldown_s", 45)
    cooldown_max = config.get("cooldown_max_s", 300)
    start_temp = config.get("start_temp_c", 50)
    limits = {"junction": config.get("temp_limit_c", DEFAULT_TEMP_LIMIT),
              "vram": config.get("vram_temp_limit_c", DEFAULT_VRAM_TEMP_LIMIT)}

    problems, warnings = check_config(config)
    cells = [(m, c) for c in configs for m in models]
    print(f"\nPlan: {len(models)} model(s) x {len(configs)} config(s) x {runs} run(s) "
          f"= {len(cells) * runs} measurement processes")
    for m in models:
        print(f"  model   {m['name']:<22} {m['path']}")
    for c in configs:
        gs = gpus_for(c)
        cap = f", cap {c['power_cap_w']} W" if c.get("power_cap_w") else ""
        print(f"  config  {c['name']:<22} {' '.join(c['args'])}{cap}")
        print(f"          GPUs: " + ", ".join(
            f"HIP {g['hip_index']} = gpu{g.get('amd_index')} {g.get('bdf')} "
            f"({g.get('name')})" for g in gs))
    print(f"  thermal guard : abort at junction >= {limits['junction']} C or VRAM >= "
          f"{limits['vram']} C, on ANY GPU used")
    print(f"  cooldown      : >= {cooldown} s and until every GPU <= {start_temp} C "
          f"(max {cooldown_max} s)")
    for w in warnings:
        print(f"  WARNING: {w}")
    for p in problems:
        print(f"  ERROR:   {p}")
    if problems:
        print("\nConfig has errors — nothing executed.")
        return None
    if dry_run:
        print("\n--dry-run: nothing executed.")
        return None

    all_gpus, seen = [], set()
    for c in configs:
        for g in gpus_for(c):
            if g.get("amd_index") is not None and g["amd_index"] not in seen:
                seen.add(g["amd_index"])
                all_gpus.append(g)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    outdir = RESULTS / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    env_rec = environment(all_gpus)
    if env_rec["gpu_consumers_at_start"]:
        print("\nWARNING: GPU(s) NOT exclusive at start: " + ", ".join(
            f"gpu{c['gpu']} {Path(str(c['name'])).name}" for c in env_rec["gpu_consumers_at_start"]))
    if env_rec["llama_swap_loaded_at_start"]:
        print(f"WARNING: llama-swap has {env_rec['llama_swap_loaded_at_start']} loaded — "
              f"curl -X POST http://127.0.0.1:8080/api/models/unload")

    record = {
        "schema": 2,
        "run_id": run_id,
        "config_name": config.get("name", "?"),
        "config_file": home_rel(config["_source"]) if config.get("_source") else None,
        "environment": env_rec,
        "settings": {"runs": runs, "cooldown_s": cooldown, "cooldown_max_s": cooldown_max,
                     "start_temp_c": start_temp, "temp_limit_c": limits["junction"],
                     "vram_temp_limit_c": limits["vram"],
                     "telemetry_interval_s": TELEMETRY_INTERVAL},
        "warnings": warnings,
        # The exact configs used. Env vars and flags change results, so the report
        # must show them verbatim rather than describing them.
        "configs_used": [{"name": c["name"], "env": c.get("env", {}), "args": c["args"],
                          "power_cap_w": c.get("power_cap_w"),
                          "gpus": [g.get("amd_index") for g in gpus_for(c)]}
                         for c in configs],
        "models": {},
        "cells": [],
    }

    raw_log = open(outdir / "raw.log", "w")
    for cfg in configs:
        gpus = gpus_for(cfg)
        prev_caps = {g["amd_index"]: power_cap_w(g) for g in gpus}
        try:
            # The power cap is a property of the CONFIG, not of a single run — set it
            # once per group so every result in that group shares one condition, and
            # restore it afterwards so the machine's standing setting is not lost.
            if cfg.get("power_cap_w"):
                ok, msg = set_power_caps(gpus, cfg["power_cap_w"])
                print(f"[{cfg['name']}] power cap: {msg}" if ok
                      else f"[{cfg['name']}] FAILED to set power cap: {msg}\n"
                           f"   -> measuring at the CURRENT cap; recorded in the result")
            caps = {str(g["amd_index"]): power_cap_w(g) for g in gpus}

            for model in cfg.get("models", models):
                if isinstance(model, str):
                    model = next((m for m in models if m["name"] == model), None)
                    if not model:
                        continue
                path = Path(os.path.expanduser(model["path"]))
                cell = {"model": model["name"], "config": cfg["name"],
                        "model_path": home_rel(path), "gpus": [g["amd_index"] for g in gpus],
                        "power_caps_w": caps,
                        "power_cap_w": (next(iter(set(caps.values())))
                                        if len(set(caps.values())) == 1 else
                                        "/".join(str(v) for v in caps.values())),
                        "runs": []}
                print(f"\n[{cfg['name']}/{model['name']}] cap {cell['power_cap_w']} W, "
                      f"GPUs {cell['gpus']}")
                for r in range(1, runs + 1):
                    res = run_once(path, cfg, gpus, limits, raw_log)
                    cell["runs"].append(res)
                    if model["name"] not in record["models"] and res["bench_params"]:
                        record["models"][model["name"]] = model_facts(path, res["bench_params"])
                    if res["aborted"]:
                        a = res["aborted"]
                        print(f"  {r}/{runs}: ABORTED by thermal guard on gpu{a['gpu']} "
                              f"(junction {a['junction_c']} C, VRAM {a['vram_c']} C)")
                    elif not res["results"]:
                        print(f"  {r}/{runs}: NO RESULT — {res['error']}")
                    else:
                        pretty = "  ".join(f"{k} {v['tok_s']:.2f} tok/s"
                                           for k, v in res["results"].items())
                        pk = res["peak"]
                        print(f"  {r}/{runs}: {pretty}  | peak J "
                              + "/".join(f"{pk[i]['junction_c']}" for i in pk) + " C, "
                              + "/".join(f"{pk[i]['power_w']}" for i in pk) + " W "
                              f"(sum {res['peak_total_power_w']} W) | links "
                              + "/".join(fmt_link(res['link_under_load'].get(str(i)))
                                         for i in pk))
                    if r < runs:
                        wait_cool(gpus, cooldown, cooldown_max, start_temp)
                if model["name"] not in record["models"]:
                    record["models"][model["name"]] = model_facts(path)
                record["cells"].append(cell)
        finally:
            if cfg.get("power_cap_w"):
                for g in gpus:
                    old = prev_caps.get(g["amd_index"])
                    if old:
                        sh(["sudo", "-n", "amd-smi", "set", "-g", str(g["amd_index"]),
                            "--power-cap", str(old)], timeout=40)
                print(f"[{cfg['name']}] power caps restored: " + ", ".join(
                    f"gpu{g['amd_index']}={power_cap_w(g)} W" for g in gpus))
    raw_log.close()
    record["environment"]["errors_after"] = error_snapshot(all_gpus)

    (outdir / "results.json").write_text(scrub(json.dumps(record, indent=2, ensure_ascii=False)))
    write_csv(record, outdir / "results.csv")
    (outdir / "report.md").write_text(scrub(build_report(record)))
    n = update_index(record)

    print(f"\nResults: {outdir}")
    print("  report.md    <- read this")
    print("  results.json <- machine-readable")
    print("  results.csv  <- spreadsheet")
    print(f"  {INDEX.name}: {n} row(s) added")
    return outdir


# ------------------------------------------------------------------- analysis

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


def cell_peaks(cell):
    """Max over runs: per-GPU peaks, plus the maxima across GPUs."""
    per = {}
    for r in cell["runs"]:
        pk = r.get("peak") or {}
        if "junction_c" in pk:                    # schema 1: a single GPU
            pk = {"0": pk}
        for i, v in pk.items():
            d = per.setdefault(str(i), {"junction_c": 0, "vram_c": 0, "power_w": 0, "fan_rpm": 0})
            for k in d:
                if _num(v.get(k)) and v[k] > d[k]:
                    d[k] = v[k]
    tot = max((r.get("peak_total_power_w") or 0 for r in cell["runs"]), default=0)
    mean_tot = [r["mean_total_power_w"] for r in cell["runs"] if _num(r.get("mean_total_power_w"))]
    return {
        "per_gpu": per,
        "junction_c": max((v["junction_c"] for v in per.values()), default=None),
        "vram_c": max((v["vram_c"] for v in per.values()), default=None),
        "power_w_card": max((v["power_w"] for v in per.values()), default=None),
        "power_w_total": tot or None,
        "mean_power_w_total": round(statistics.fmean(mean_tot), 1) if mean_tot else None,
    }


def cell_links(cell):
    """'x8/x4/x8/x4 Gen5' — the bottleneck link of each GPU, under load."""
    for r in cell["runs"]:
        lu = r.get("link_under_load") or {}
        if "width" in lu:                         # schema 1: endpoint reading
            return f"x{lu.get('width')} (endpoint)" if lu.get("width") else "—"
        if lu:
            links = [lu[k] for k in sorted(lu, key=lambda x: int(x))]
            ws = "/".join(f"x{l.get('width')}" for l in links)
            gens = {fmt_link(l).split(" ", 1)[-1] for l in links if l.get("width")}
            idle = any(l.get("idle") for l in links)
            return ws + (f" {gens.pop()}" if len(gens) == 1 else "") + (" (idle)" if idle else "")
    return "—"


def verdict_for(record, cell):
    """('OK'|'WITH_CAVEAT'|'INVALID', [flags])."""
    if cell.get("error"):
        return "INVALID", [cell["error"]]
    stats = summarise(cell)
    n_ab = sum(1 for r in cell["runs"] if r.get("aborted") or r.get("aborted_at_c"))
    if not stats:
        errs = {r.get("error") for r in cell["runs"] if r.get("error")}
        return "INVALID", [f"no result ({n_ab} aborted thermally)" if n_ab else
                           f"no result: {'; '.join(e for e in errs if e) or 'unknown'}"]
    pk = cell_peaks(cell)
    flags, worst = [], max(s["spread_pct"] for s in stats.values())
    if _num(pk["power_w_card"]) and pk["power_w_card"] < 60:
        flags.append(f"peak power {pk['power_w_card']} W — TELEMETRY UNRELIABLE: compute "
                     f"window shorter than the sampling period; no thermal conclusions")
    if n_ab:
        flags.append(f"{n_ab} run(s) aborted thermally")
    if len(stats) and min(s["n"] for s in stats.values()) < len(cell["runs"]):
        flags.append("some runs produced no result")
    if worst > 10:
        flags.append(f"spread {worst:.0f}% — likely bimodal, do NOT report the mean")
    if _num(pk["junction_c"]) and pk["junction_c"] >= 85:
        flags.append(f"junction {pk['junction_c']} °C — you are measuring cooling, not the model")
    env = record.get("environment", {})
    if env.get("gpu_consumers_at_start"):
        return "INVALID", flags + ["GPU not exclusive at start"]
    if errors_moved(env):
        return "INVALID", flags + ["hardware error counters increased"]
    return ("WITH_CAVEAT" if flags else "OK"), flags


def errors_moved(env):
    b, a = env.get("errors_before"), env.get("errors_after")
    if b is None:                                  # schema 1
        drift = [k for k in env.get("aer_before", {})
                 if env.get("aer_before", {}).get(k) != env.get("aer_after", {}).get(k)]
        drift += [k for k in env.get("ecc_before", {})
                  if env.get("ecc_before", {}).get(k) != env.get("ecc_after", {}).get(k)]
        return drift
    moved = []
    for kind in ("aer", "ecc"):
        for dev, vals in (b.get(kind) or {}).items():
            after = (a or {}).get(kind, {}).get(dev, {})
            for k, v in vals.items():
                if v != after.get(k):
                    moved.append(f"{kind} {dev} {k}: {v} -> {after.get(k)}")
    return moved


# -------------------------------------------------------------- shared index

INDEX_COLUMNS = ["run_id", "model", "config", "test", "tok/s", "spread", "n",
                 "arch", "params", "quant", "GiB", "GPUs", "split", "PCIe (under load)",
                 "cap W", "fan curve", "peak junction", "peak VRAM", "GPU W peak / mean",
                 "FA", "batch/ubatch", "KV", "load", "CPU profile", "llama.cpp", "ROCm",
                 "verdict"]

INDEX_INTRO = """# All benchmark results — every model measured, with its conditions

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
"""


def _gpu_label(record, gpu_ids):
    inv = {str(g.get("amd_index", g.get("index"))): g
           for g in (record["environment"].get("inventory") or record["environment"].get("gpus") or [])}
    labels = []
    for i in gpu_ids:
        g = inv.get(str(i), {})
        n = str(g.get("name") or "")
        labels.append("R9700" if "R9700" in n else ("iGPU" if is_igpu(g) else (n or "?")))
    if not labels:
        return "—"
    hip = ",".join(str(i) for i in gpu_ids)
    return (f"{len(labels)}×{labels[0]}" if len(set(labels)) == 1 else "+".join(labels)) + f" ({hip})"


def _arg(args, names):
    for i, a in enumerate(args[:-1]):
        if a in names:
            return str(args[i + 1])
    return None


_facts_cache = {}


def index_rows(record):
    """Rows of the shared table for one results.json (schema 1 or 2)."""
    env = record.get("environment", {})
    sw = env.get("software", {}) or {}
    cpu = ((env.get("hardware") or {}).get("cpu_state") or {}).get("profile") or "—"
    cfgs = {c["name"]: c for c in record.get("configs_used") or []}
    rows = []
    for cell in record.get("cells", []):
        stats = summarise(cell) if cell.get("runs") else {}
        if not stats:
            continue
        verdict, _ = verdict_for(record, cell)
        cfg = cfgs.get(cell["config"], {})
        args = [str(a) for a in cfg.get("args") or []]
        gpu_ids = cell.get("gpus")
        if gpu_ids is None:                                   # schema 1
            hip = str((cfg.get("env") or {}).get("HIP_VISIBLE_DEVICES", env.get("gpu_index", 0)))
            gpu_ids = [int(x) for x in hip.split(",") if x.strip().isdigit()] or [0]
        bp = next((r.get("bench_params") for r in cell["runs"] if r.get("bench_params")), {}) or {}
        mf = (record.get("models") or {}).get(cell["model"], {})
        if not mf and cell.get("model_path"):
            # Older results (schema 1) did not record the model's facts; the file is
            # usually still there, so read them from its GGUF header now.
            mp = Path(os.path.expanduser(cell["model_path"]))
            if mp.exists():
                mf = _facts_cache.setdefault(str(mp), model_facts(mp, bp))
        pk = cell_peaks(cell)
        used = {str(g.get("amd_index")): g for g in env.get("gpus_used") or []}
        curves = {used[str(i)].get("fan_curve") for i in gpu_ids if str(i) in used}
        sm = bp.get("split_mode") or _arg(args, ("-sm", "--split-mode")) or "none"
        ts = bp.get("tensor_split") if bp.get("tensor_split") not in (None, "0.00") \
            else _arg(args, ("-ts", "--tensor-split"))
        fa = bp.get("flash_attn", _arg(args, ("-fa", "--flash-attn")))
        row = {
            "run_id": record.get("run_id", "?"), "model": cell["model"], "config": cell["config"],
            "arch": mf.get("architecture") or (bp.get("model_type") or "—").split(" ")[0],
            "params": params_label(mf) if mf else "—",
            "quant": mf.get("quant") or "—",
            "GiB": f"{mf['file_size_gib']:.1f}" if mf.get("file_size_gib") else "—",
            "GPUs": _gpu_label(record, gpu_ids),
            "split": f"{sm} {ts}" if ts and sm != "none" else sm,
            "PCIe (under load)": cell_links(cell),
            "cap W": str(cell.get("power_cap_w", env.get("power_cap", "—"))),
            "fan curve": (curves.pop() if len(curves) == 1 else ("mixed" if curves else "—")),
            "peak junction": f"{pk['junction_c']} °C" if _num(pk["junction_c"]) and pk["junction_c"] else "—",
            "peak VRAM": f"{pk['vram_c']} °C" if _num(pk["vram_c"]) and pk["vram_c"] else "—",
            "GPU W peak / mean": (f"{pk['power_w_total']:.0f} / {pk['mean_power_w_total']:.0f}"
                                  if pk["power_w_total"] and pk["mean_power_w_total"] else
                                  (f"{pk['power_w_card']}" if pk["power_w_card"] else "—")),
            "FA": {1: "on", 0: "off", True: "on", False: "off"}.get(fa, str(fa) if fa is not None else "—"),
            "batch/ubatch": (f"{bp['n_batch']}/{bp['n_ubatch']}" if bp.get("n_batch") else "—"),
            "KV": (f"{bp['type_k']}/{bp['type_v']}" if bp.get("type_k") else "—"),
            "load": bp.get("load_mode") or _arg(args, ("-lm", "--load-mode")) or "—",
            "CPU profile": cpu,
            "llama.cpp": f"`{env.get('llama_cpp_commit') or sw.get('llama_cpp_commit') or '?'}`",
            "ROCm": env.get("rocm") or "?",
            "verdict": verdict,
        }
        for test, s in stats.items():
            rows.append(dict(row, test=test, **{"tok/s": f"{s['mean']:.2f}",
                                                "spread": f"{s['spread_pct']:.1f}%",
                                                "n": str(s["n"])}))
    return rows


def _row_line(r):
    return "| " + " | ".join(str(r.get(c, "—")).replace("|", "/") for c in INDEX_COLUMNS) + " |"


def _parse_index(text):
    """-> (table rows as dicts keyed by the file's own header, notes after the table)."""
    lines = text.splitlines()
    try:
        h = next(i for i, l in enumerate(lines) if l.startswith("| run_id |"))
    except StopIteration:
        return [], "\n".join(lines)
    header = [c.strip() for c in lines[h].strip().strip("|").split("|")]
    rows, i = [], h + 2
    while i < len(lines) and lines[i].startswith("|"):
        cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
        rows.append(dict(zip(header, cells)))
        i += 1
    return rows, "\n".join(lines[i:])


def _migrate(row):
    """Map a row from the schema-1 table (14 columns) onto the current columns."""
    out = {c: row.get(c, "—") for c in INDEX_COLUMNS}
    out["peak junction"] = row.get("peak junction", row.get("junction", "—"))
    if "power" in row and out["GPU W peak / mean"] == "—":
        out["GPU W peak / mean"] = row["power"].replace(" W", "")
    if "cap" in row and out["cap W"] == "—":
        out["cap W"] = row["cap"]
    if "PCIe" in row and out["PCIe (under load)"] == "—":
        out["PCIe (under load)"] = row["PCIe"] + (" (endpoint)" if "x16" in row["PCIe"] else "")
    return out


def write_index(rows, notes):
    body = [INDEX_INTRO, "| " + " | ".join(INDEX_COLUMNS) + " |",
            "|" + "|".join("---:" if c in ("tok/s", "spread", "n") else "---"
                           for c in INDEX_COLUMNS) + "|"]
    body += [_row_line(r) for r in rows]
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    INDEX.write_text("\n".join(body) + "\n" + (notes.rstrip() + "\n" if notes.strip() else ""),
                     encoding="utf-8")


def update_index(record):
    """Append this run's rows to the shared table (idempotent, keeps the notes)."""
    rows, notes = _parse_index(INDEX.read_text(encoding="utf-8")) if INDEX.exists() else ([], "")
    rows = [_migrate(r) for r in rows]
    have = {(r["run_id"], r["model"], r["config"], r["test"]) for r in rows}
    new = [r for r in index_rows(record)
           if (r["run_id"], r["model"], r["config"], r["test"]) not in have]
    write_index(rows + new, notes)
    return len(new)


def rebuild_index():
    """Regenerate every row that has a results.json; keep manual rows and the notes."""
    old_rows, notes = _parse_index(INDEX.read_text(encoding="utf-8")) if INDEX.exists() else ([], "")
    if INDEX.exists():
        shutil.copy2(INDEX, INDEX.with_suffix(".md.bak"))
    auto, ids = [], set()
    for rj in sorted(RESULTS.glob("*/results.json")):
        try:
            rec = json.loads(rj.read_text())
        except Exception:
            continue
        ids.add(rec.get("run_id"))
        auto += index_rows(rec)
    manual = [_migrate(r) for r in old_rows if r.get("run_id") not in ids]
    rows = sorted(auto + manual, key=lambda r: r["run_id"])
    write_index(rows, notes)
    print(f"Index rebuilt: {len(auto)} rows from {len(ids)} runs + {len(manual)} manual rows "
          f"kept\n  {INDEX} (previous version: {INDEX.with_suffix('.md.bak').name})")


# ------------------------------------------------------------------- reporting

def write_csv(record, path):
    rows = index_rows(record)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(INDEX_COLUMNS)
        for r in rows:
            w.writerow([str(r.get(c, "")).strip("`") for c in INDEX_COLUMNS])


def build_report(record):
    env = record["environment"]
    hw = env.get("hardware", {}) or {}
    sw = env.get("software", {}) or {}
    L = []
    A = L.append

    A(f"# Benchmark report — {record['run_id']}\n")
    A(f"Config: **{record['config_name']}** (`{record.get('config_file') or '?'}`)  ")
    A(f"Run at: {env['timestamp']}\n")

    # Verdict first, deliberately before any table: the report should answer
    # "is this result usable?" without the reader having to work it out.
    A("## Verdict\n")
    for cell in record["cells"]:
        v, flags = verdict_for(record, cell)
        mark = {"OK": "✅", "WITH_CAVEAT": "⚠️", "INVALID": "❌"}[v]
        stats = summarise(cell)
        bits = ", ".join(f"{k} **{s['mean']:.1f}** tok/s" for k, s in stats.items())
        A(f"- {mark} **{cell['model']} / {cell['config']}** ({cell.get('power_cap_w')} W): "
          f"{bits or 'no result'}" + (f" — {'; '.join(flags)}" if flags else ""))
    if env.get("gpu_consumers_at_start"):
        A("\n❌ **GPU(s) not exclusive at start** (" + ", ".join(
            f"gpu{c['gpu']} {Path(str(c['name'])).name}" for c in env["gpu_consumers_at_start"])
          + ") — the result describes contention for the card, not model performance.")
    moved = errors_moved(env)
    A("")
    A("❌ **Hardware error counters increased during the run — discard these results "
      "and repeat.** " + "; ".join(moved) if moved else
      "✅ Hardware error counters (PCIe AER along each GPU's path, GDDR6 ECC) unchanged "
      "on every GPU used — results are not contaminated by bus or memory errors.")
    for w in record.get("warnings") or []:
        A(f"\n⚠️ Config adjusted: {w}")

    A("\n## Model\n")
    A("| Model | Architecture | Parameters | Quant (llama.cpp type) | File | Layers | "
      "Native context |")
    A("|---|---|---|---|---|---:|---:|")
    for name, f in (record.get("models") or {}).items():
        A(f"| {name} | {f.get('architecture') or '?'} | {params_label(f)} | "
          f"{f.get('llama_type') or f.get('quant') or '?'} | "
          f"{f.get('file_size_gib') or '?'} GiB in {f.get('shards')} file(s) | "
          f"{f.get('layers') or '?'} | {f.get('context_length') or '?'} |")

    A("\n## Results\n")
    A("| Model | Config | Test | Mean tok/s | min | max | Spread | n |")
    A("|---|---|---|---:|---:|---:|---:|---:|")
    for cell in record["cells"]:
        for test, s in summarise(cell).items():
            A(f"| {cell['model']} | {cell['config']} | {test} | {s['mean']:.2f} | "
              f"{s['min']:.2f} | {s['max']:.2f} | {s['spread_pct']:.1f}% | {s['n']} |")
        for r in cell["runs"]:
            if r.get("error"):
                A(f"| {cell['model']} | {cell['config']} | — | failed run: {r['error']} "
                  f"| | | | |")

    A("\n## Per-GPU telemetry (peaks over all runs of the cell)\n")
    A("| Model / config | GPU | PCI | Link under load | Cap | Fan curve | Peak junction | "
      "Peak VRAM | Peak power | Peak fan |")
    A("|---|---|---|---|---:|---|---:|---:|---:|---:|")
    used = {str(g.get("amd_index")): g for g in env.get("gpus_used") or []}
    for cell in record["cells"]:
        pk = cell_peaks(cell)
        lu = next((r.get("link_under_load") for r in cell["runs"] if r.get("link_under_load")), {})
        for i, v in sorted(pk["per_gpu"].items()):
            g = used.get(str(i), {})
            A(f"| {cell['model']} / {cell['config']} | gpu{i} | {g.get('bdf', '?')} | "
              f"{fmt_link(lu.get(str(i)))} | {cell.get('power_caps_w', {}).get(str(i), '?')} W | "
              f"{g.get('fan_curve', '?')} | {v['junction_c']} °C | {v['vram_c']} °C | "
              f"{v['power_w']} W | {v['fan_rpm']} rpm |")
        A(f"| {cell['model']} / {cell['config']} | **all** | | | | | "
          f"**{pk['junction_c']} °C** | **{pk['vram_c']} °C** | "
          f"**{pk['power_w_total']} W** peak sum, {pk['mean_power_w_total']} W mean | |")

    A("\n## Effective llama-bench parameters\n")
    A("Read back from llama-bench's own JSON output, i.e. what actually ran — not what the "
      "config intended.\n")
    A("| Model / config | Split | Tensor split | FA | batch / ubatch | KV cache | Load mode | "
      "Threads | Build |")
    A("|---|---|---|---|---|---|---|---:|---|")
    for cell in record["cells"]:
        bp = next((r.get("bench_params") for r in cell["runs"] if r.get("bench_params")), {}) or {}
        A(f"| {cell['model']} / {cell['config']} | {bp.get('split_mode', '?')} | "
          f"{bp.get('tensor_split', '?')} | {bp.get('flash_attn', '?')} | "
          f"{bp.get('n_batch', '?')} / {bp.get('n_ubatch', '?')} | "
          f"{bp.get('type_k', '?')}/{bp.get('type_v', '?')} | {bp.get('load_mode', '?')} | "
          f"{bp.get('n_threads', '?')} | {bp.get('build_commit', '?')} "
          f"({bp.get('backends', '?')}) |")

    A("\n## Hardware state at run time\n")
    A("| Parameter | Value |\n|---|---|")
    A(f"| Board / BIOS | {hw.get('board') or '?'} / {hw.get('bios') or 'not read'} |")
    cs = hw.get("cpu_state") or {}
    A(f"| CPU | {hw.get('cpu', '?')} — profile `{cs.get('profile')}`, governor "
      f"`{cs.get('governor')}`, EPP `{cs.get('epp')}` |")
    A(f"| RAM | {hw.get('ram_gb', '?')} GB |")
    for g in env.get("inventory") or []:
        u = used.get(str(g.get("amd_index")))
        A(f"| gpu{g.get('amd_index')} (HIP {g.get('hip_index')}) | {g.get('name')} — "
          f"{g.get('gfx')}, {g.get('cus')} CU, {g.get('vram_mb')} MiB, {g.get('bdf')}"
          + (f" — **used**, VBIOS `{u.get('vbios')}`, subsystem `{u.get('subsystem')}`, "
             f"idle link {fmt_link(u.get('link_idle'))}" if u else "") + " |")
    A(f"| GPU tuning service | {sw.get('gpu_tuning_service') or 'n/a'} |")
    A(f"| llama-swap models loaded at start | {env.get('llama_swap_loaded_at_start')} |")

    A("\n## Software state at run time\n")
    A("| Parameter | Value |\n|---|---|")
    A(f"| Kernel | {sw.get('kernel', '?')} |")
    A(f"| Kernel cmdline | `{sw.get('kernel_cmdline', '?')}` |")
    A(f"| amdgpu ppfeaturemask | `{sw.get('amdgpu_ppfeaturemask')}` |")
    A(f"| ROCm | {sw.get('rocm', '?')} |")
    commit, cdate = sw.get("llama_cpp_commit", "?"), sw.get("llama_cpp_commit_date")
    A(f"| llama.cpp | `{commit}`" + (f" ({cdate})" if cdate else "") + " |")
    if sw.get("llama_cpp_build"):
        A("| llama.cpp build flags | " +
          ", ".join(f"`{k}={v}`" for k, v in sorted(sw["llama_cpp_build"].items())) + " |")

    st = record["settings"]
    A("\n## Measurement parameters\n")
    A("| Parameter | Value |\n|---|---|")
    A(f"| Runs per cell | {st['runs']} — **separate OS processes** |")
    A(f"| Cooldown between runs | ≥ {st['cooldown_s']} s and until every GPU ≤ "
      f"{st.get('start_temp_c')} °C (max {st.get('cooldown_max_s')} s) |")
    A(f"| Thermal guard | abort at junction ≥ {st['temp_limit_c']} °C or VRAM ≥ "
      f"{st.get('vram_temp_limit_c')} °C on any GPU used |")
    A(f"| Telemetry | amd-smi every {st.get('telemetry_interval_s')} s, all GPUs used |")
    A("\n**Configurations used** — recorded verbatim:\n")
    A("| Config | Environment | llama-bench flags | GPUs |")
    A("|---|---|---|---|")
    for cfg in record.get("configs_used") or []:
        envs = ", ".join(f"`{k}={v}`" for k, v in (cfg.get("env") or {}).items()) or "—"
        A(f"| {cfg['name']} | {envs} | `{' '.join(cfg.get('args') or [])}` | "
          f"{cfg.get('gpus')} |")

    A("\n## How to read this\n")
    A("**Spread matters more than the mean.** Decode throughput on this class of card is "
      "documented as *bimodal* — the card settles into one of two modes and stays there for "
      "the life of the process. That is why every run is a separate process, not `-r N`.\n")
    A("- **Spread below ~2 %** → bimodality did not appear; the mean is a sensible number.")
    A("- **Spread above ~10 %, two clusters** → report both modes, not their mean.\n")
    A("**Peak junction near 90 °C** means you are measuring case airflow, not the model.\n")
    A("**Low peak power on a valid result does not mean the GPU idled** — the compute window "
      "was shorter than the sampling period. With `-sm layer` each card also works only on "
      "its share of the layers in turn, so per-card power on a multi-GPU run is naturally "
      "far below the cap; look at the summed power.\n")
    A("**PCIe** is the bottleneck link of each GPU's path (root port → switch on the card → "
      "GPU), read under load. The GPU endpoint itself always reports x16.\n")
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

    def cond(rec):
        e = rec["environment"]
        caps = {str(c.get("power_cap_w")) for c in rec["cells"]} or {str(e.get("power_cap"))}
        return e["llama_cpp_commit"], "/".join(sorted(caps))

    fa, fb = flat(a), flat(b)
    (ca, pa), (cb, pb) = cond(a), cond(b)
    print(f"\nA = {a['run_id']}  (llama.cpp {ca}, cap {pa})")
    print(f"B = {b['run_id']}  (llama.cpp {cb}, cap {pb})\n")
    if ca != cb:
        print("  NOTE: different llama.cpp commits — do not attribute the delta to hardware.")
    if pa != pb:
        print("  NOTE: different power caps — the delta includes the cap.")
    print(f"\n{'model/config/test':<50} {'A':>11} {'B':>11} {'change':>9}")
    print("-" * 84)
    for key in sorted(set(fa) | set(fb)):
        va, vb = fa.get(key), fb.get(key)
        label = "/".join(key)
        if va and vb:
            print(f"{label:<50} {va:>11.2f} {vb:>11.2f} {(vb - va) / va * 100:>+8.1f}%")
        else:
            print(f"{label:<50} {va or '—':>11} {vb or '—':>11} {'—':>9}")
    print()


def main():
    ap = argparse.ArgumentParser(
        description="Declarative benchmark harness for local LLM inference")
    ap.add_argument("config", nargs="?", help="JSON file describing the matrix")
    ap.add_argument("--dry-run", action="store_true", help="validate + print the plan, run nothing")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="compare two result directories")
    ap.add_argument("--list", action="store_true", help="models and configs")
    ap.add_argument("--reindex", action="store_true",
                    help="rebuild results/all-benchmark-results.md from all results")
    args = ap.parse_args()

    if args.compare:
        compare(*args.compare)
        return
    if args.reindex:
        rebuild_index()
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
                print(f"  {p.name:<30} {c.get('name', '')} "
                      f"({len(c.get('models', []))} models, "
                      f"{len(c.get('configs', []))} configs)")
            except Exception:
                print(f"  {p.name:<30} (parse error)")
        print("\nResults:")
        for p in sorted(RESULTS.glob("*")):
            if (p / "results.json").exists():
                print(f"  {p.name}")
        print()
        return

    if not args.config:
        ap.error("pass a config file, or --list / --compare / --reindex")
    for tool in ("amd-smi", "rocminfo"):
        if not shutil.which(tool):
            print(f"ERROR: {tool} not found — GPU mapping/telemetry will not work.")
            sys.exit(1)

    cfg = json.loads(Path(args.config).read_text())
    cfg["_source"] = str(Path(args.config).resolve())
    out = execute(cfg, dry_run=args.dry_run)
    if out is None and not args.dry_run:
        sys.exit(2)


if __name__ == "__main__":
    main()
