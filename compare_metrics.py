import pandas as pd
import networkx as nx
from stimuli import STIMULI

conn = pd.read_parquet("data/2025_Connectivity_783.parquet")

def analyze(spike_file, driven_ids, label, top_n=300):
    driven_ids = set(driven_ids)
    spikes = pd.read_parquet(spike_file)
    counts = spikes.groupby("flywire_id").size()

    total_spikes = len(spikes)
    driven_spikes = counts.reindex(driven_ids).fillna(0).sum()
    downstream = counts.drop(labels=driven_ids, errors="ignore")
    downstream_spikes = downstream.sum()
    downstream_active = downstream.shape[0]

    active_set = set(downstream.sort_values(ascending=False).head(top_n).index)
    sub = conn[conn["Presynaptic_ID"].isin(active_set) & conn["Postsynaptic_ID"].isin(active_set)]
    G = nx.DiGraph()
    for nid in active_set:
        G.add_node(nid)
    G.add_weighted_edges_from(zip(sub["Presynaptic_ID"], sub["Postsynaptic_ID"], sub["Connectivity"]))

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)
    avg_degree = (2 * n_edges / n_nodes) if n_nodes else 0

    print(f"\n=== {label} ===")
    print(f"Driven neurons:              {len(driven_ids)}")
    print(f"Total spikes (all neurons):  {total_spikes}")
    print(f"Spikes from driven neurons:  {int(driven_spikes)}")
    print(f"Downstream spikes:           {int(downstream_spikes)}")
    print(f"Downstream spikes/driver:    {downstream_spikes / len(driven_ids):.1f}")
    print(f"Unique downstream neurons:   {downstream_active}")
    print(f"Top-{top_n} subgraph edges:   {n_edges}")
    print(f"Top-{top_n} subgraph density: {density:.4f}")
    print(f"Avg node degree (top {top_n}): {avg_degree:.1f}")

    return {
        "label": label, "downstream_active": downstream_active, "edges": n_edges,
        "density": density, "avg_degree": avg_degree, "spikes_per_driver": downstream_spikes / len(driven_ids)
    }

r_sugar = analyze("data/results/brian2cpp_t10.0s_n1.parquet", STIMULI["sugar"]["ids"], "SUGAR")
r_bitter = analyze("data/results/Bitter_200Hz.parquet", STIMULI["bitter"]["ids"], "BITTER")
r_fear = analyze("data/results/Fear_150Hz.parquet", STIMULI["fear"]["ids"], "FEAR")

results = [r_sugar, r_bitter, r_fear]
print("\n=== SIDE BY SIDE ===")
print(f"{'Metric':<28}" + "".join(f"{r['label']:>12}" for r in results))
print(f"{'Downstream spikes/driver':<28}" + "".join(f"{r['spikes_per_driver']:>12.1f}" for r in results))
print(f"{'Unique downstream neurons':<28}" + "".join(f"{r['downstream_active']:>12}" for r in results))
print(f"{'Subgraph edges':<28}" + "".join(f"{r['edges']:>12}" for r in results))
print(f"{'Subgraph density':<28}" + "".join(f"{r['density']:>12.4f}" for r in results))
print(f"{'Avg node degree':<28}" + "".join(f"{r['avg_degree']:>12.1f}" for r in results))