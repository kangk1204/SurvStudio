"""A few replicates of 06_simulation.py's scenarios, with timings, and the summary of 06_simulation_summary.py over
them (nothing is written to results/).
Usage: python simulation_smoke.py [replicates] [scenario ...]   (default: one replicate of null_filter and of
alt_0.30_filter)
"""

import importlib
import sys
import time

import pandas as pd

replicates = int(sys.argv[1]) if len(sys.argv) > 1 else 1
began = time.time()
simulation = importlib.import_module("06_simulation")
summary = importlib.import_module("06_simulation_summary")
scenarios = sys.argv[2:] or ["null_filter", "alt_0.30_filter"]
unknown = [name for name in scenarios if name not in simulation.SCENARIOS]
if unknown:
    raise SystemExit(f"unknown scenarios {unknown}; the design has {list(simulation.SCENARIOS)}")
print(f"setup {time.time() - began:.0f}s; patients {simulation.N}; genes {simulation.GENES.size}; censoring rate {simulation.CENSOR_RATE:.4f}")
print("clinical log HR:", dict(zip(simulation.DESIGN.columns, simulation.CLINICAL_FIT.beta.round(3))))
records = []
for name in scenarios:
    for index in range(replicates):
        began = time.time()
        records.append(simulation.replicate((name, index)))
        print((name, index), records[-1], f"{time.time() - began:.0f}s", flush=True)
table = pd.DataFrame(records).reindex(columns=simulation.COLUMNS)
settings = {name: {**simulation.SCENARIOS[name]._asdict(), "replicates": replicates} for name in scenarios}
with pd.option_context("display.width", 250, "display.max_columns", 60):
    print(pd.DataFrame(summary.summarise(table, {"scenarios": settings})).round(4).to_string(index=False))
