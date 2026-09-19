# All benchmark results — every model measured, with its conditions

Appended automatically by `bench-model`. Newest at the bottom.

`spread` is the difference between the best and worst independent run.
Above 10 % indicates bimodality — the mean in this table is then misleading
and you should open that run's `report.md`.

**Only compare rows with the same power cap and the same llama.cpp commit.**
Otherwise you are comparing conditions, not models.

| run_id | model | config | test | tok/s | spread | junction | power | cap | llama.cpp | ROCm |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| 20260915-164813 | gpt-oss-20b | throughput | pp512 | 6038.74 | 0.6% | 30 °C | 19 W | 300 | `987498f` | 7.2.4 |
| 20260915-164813 | gpt-oss-20b | throughput | tg128 | 149.72 | 0.2% | 30 °C | 19 W | 300 | `987498f` | 7.2.4 |
