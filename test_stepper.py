import time
import numpy as np
from lif_stepper import LIFBrain
from stimuli import STIMULI

brain = LIFBrain(dt=1.0)
sugar_ids = [i for i in STIMULI["sugar"]["ids"] if i in brain.id_to_idx]
print(f"{len(sugar_ids)} of {len(STIMULI['sugar']['ids'])} sugar neurons found in subgraph")

poisson_drive = {nid: 200.0 for nid in sugar_ids}

N_STEPS = 1000  # 1000ms = 1 simulated second at dt=1ms
total_spikes = 0
start = time.time()
for _ in range(N_STEPS):
    spiked = brain.step(poisson_drive)
    total_spikes += spiked.sum()
elapsed = time.time() - start

print(f"Simulated 1.0s in {elapsed:.3f}s wall-clock ({1.0/elapsed:.1f}x realtime)")
print(f"Total spikes: {total_spikes}")