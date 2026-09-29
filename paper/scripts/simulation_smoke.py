"""One replicate of a null and an alternative scenario of 06_simulation.py, with timings."""

import importlib
import time

began = time.time()
simulation = importlib.import_module("06_simulation")
print(f"setup {time.time() - began:.0f}s; patients {simulation.N}; genes {simulation.GENES.size}; censoring rate {simulation.CENSOR_RATE:.4f}")
print("clinical log HR:", dict(zip(simulation.DESIGN.columns, simulation.CLINICAL_FIT.beta.round(3))))
for task in (("null_filter", 0), ("alt_0.30_filter", 0)):
    began = time.time()
    print(task, simulation.replicate(task), f"{time.time() - began:.0f}s", flush=True)
