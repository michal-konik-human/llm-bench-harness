# Contributing to local-llm-perf-bench

Thanks for considering it! The harness exists to produce numbers you can trust and compare, so
the most valuable contributions are the ones that test that promise on hardware I don't have.

## Good first contributions

| Contribution | How |
|---|---|
| **Run it on your GPU** and share the report | `bench-model <model>`, then open a "Share results" issue with `report.md` and `results.json` |
| **Numbers that contradict mine** | Same as above. Note your llama.cpp commit and power cap, which are the usual culprits |
| **A test config** for a question the existing ones don't answer | Add `configs/<name>.json` with a `_comment` explaining the question; run `./bench.py <config> --dry-run` |
| **Telemetry for NVIDIA or Apple Silicon** | The AMD path lives in `bench.py` (`amd-smi`, sysfs). Open an issue first to agree on the shape |
| **Docs fixes** | Anything that confused you is worth a PR |

## Ground rules

1. **Don't trade validity for convenience.** One OS process per run, conditions recorded, error
   counters checked before and after, the thermal guard on. A change that makes a number easier
   to get but harder to trust will not be merged.
2. **Keep it dependency-free.** The harness is stdlib Python plus the system tools it measures
   with (`llama-bench`, `amd-smi`, `rocminfo`).
3. **No personal data in the repo.** Results go to `$LLM_PERF_RESULTS` (default `./results`, which is
   git-ignored). Paths written to results are home-relative (`~`).
4. **Explain the why.** A measurement trap that cost you a day is worth a paragraph in
   `docs/METHODOLOGY.md`, as are the ones already there.

## Development

```bash
git clone https://github.com/michal-konik-human/local-llm-perf-bench.git && cd local-llm-perf-bench
./bench.py configs/single-gpu.json --dry-run     # validates the config and prints the plan
```

## Code of conduct

Be kind, assume good faith, critique the work rather than the person. Harassment of any kind is
not tolerated.
