import time
from lif_stepper import LIFBrain
from stimuli import STIMULI

for name in ["sugar", "bitter", "fear"]:
    brain = LIFBrain(dt=1.0)
    ids = [i for i in STIMULI[name]["ids"] if i in brain.id_to_idx]
    drive = {nid: STIMULI[name]["hz"] for nid in ids}
    total = 0
    for _ in range(1000):
        total += brain.step(drive).sum()
    print(f"{name}: {len(ids)} drivers found, {total} spikes/sec, {total/max(len(ids),1):.1f} spikes/driver")