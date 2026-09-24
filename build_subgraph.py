import pandas as pd
import numpy as np
import scipy.sparse as sp
from stimuli import STIMULI

# --- knobs -------------------------------------------------------------
HOPS = 2            # synaptic hops to expand outward from each stimulus
PER_HOP_CAP = 350    # neurons a category may newly claim at each hop
MIN_WEIGHT = 1       # drop near-zero synapses before ranking candidates

conn = pd.read_parquet("data/2025_Connectivity_783.parquet")
conn = conn[conn["Excitatory x Connectivity"].abs() >= MIN_WEIGHT]

# Categories are expanded smallest-seed-pool first. A fixed hand-picked order
# (e.g. "taste before vision") isn't enough: ranking is by total synaptic
# weight from a category's own seed neurons, so a category with MORE seed
# neurons naturally outweighs one with fewer for any hub they both feed —
# sugar (23 seeds) and bitter (42) were dominating the shared taste-processing
# hubs water/salt (18 each) also feed, no matter what order they went in
# relative to fear/motion/vision. Going scarcest-first means water/salt claim
# their own best-connected partners before sugar/bitter get the chance to.
PROCESS_ORDER = sorted(STIMULI, key=lambda name: len(STIMULI[name]["ids"]))

category_ids = {name: set(cfg["ids"]) for name, cfg in STIMULI.items()}

all_ids = set()
claimed_by = {}  # neuron id -> the one category that "owns" it for coloring
for name in PROCESS_ORDER:
    ids = category_ids[name]
    all_ids.update(ids)
    for nid in ids:
        claimed_by.setdefault(nid, name)

for name in PROCESS_ORDER:
    frontier = set(category_ids[name])
    for _hop in range(HOPS):
        edges = conn[conn["Presynaptic_ID"].isin(frontier)]
        if edges.empty:
            break
        # Rank candidates by total synaptic weight this category drives them
        # with (not just how many other things also happen to connect to
        # them) — that's what actually decides how hard they'd fire.
        weight = (
            edges.groupby("Postsynaptic_ID")["Excitatory x Connectivity"]
            .apply(lambda s: s.abs().sum())
            .sort_values(ascending=False)
        )
        picked, new_frontier = 0, set()
        for nid, _w in weight.items():
            if nid in claimed_by:
                continue  # a stronger/earlier category already owns this hub
            claimed_by[nid] = name
            all_ids.add(nid)
            new_frontier.add(nid)
            picked += 1
            if picked >= PER_HOP_CAP:
                break
        if not new_frontier:
            break
        frontier = new_frontier

all_ids = sorted(all_ids)
id_to_idx = {nid: i for i, nid in enumerate(all_ids)}
N = len(all_ids)
print(f"Subgraph size: {N} neurons ({HOPS}-hop, {PER_HOP_CAP}/hop/category cap)")
for name in PROCESS_ORDER:
    print(f"  {name:<16} owns {sum(1 for v in claimed_by.values() if v == name):5d} neurons")

sub = conn[conn["Presynaptic_ID"].isin(all_ids) & conn["Postsynaptic_ID"].isin(all_ids)]
print(f"{len(sub)} real synapses among them")

rows = sub["Postsynaptic_ID"].map(id_to_idx).values  # post = row (who receives)
cols = sub["Presynaptic_ID"].map(id_to_idx).values    # pre = col (who sends)
weights = sub["Excitatory x Connectivity"].values.astype(np.float32)
W = sp.csr_matrix((weights, (rows, cols)), shape=(N, N))

np.save("data/subgraph_ids.npy", np.array(all_ids))
sp.save_npz("data/subgraph_weights.npz", W)

# Which category "owns" each neuron in the subgraph (empty string if somehow
# unclaimed). fly_game.py can use this to color every propagated neuron by
# its primary behavior, not just the directly-stimulated sensory/command
# populations it already tags from stimuli.py.
owner = np.array([claimed_by.get(nid, "") for nid in all_ids])
np.save("data/subgraph_owner.npy", owner)

print("Saved subgraph_ids.npy, subgraph_weights.npz, subgraph_owner.npy")
