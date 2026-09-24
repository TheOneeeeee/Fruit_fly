import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

spikes = pd.read_parquet("data/results/brian2cpp_t1.0s_n1.parquet")
spike_counts = spikes.groupby("flywire_id").size()

TOP_N = 300
active_set = set(spike_counts.sort_values(ascending=False).head(TOP_N).index)
print(f"Using top {len(active_set)} most active neurons (out of {spike_counts.shape[0]} total)")

conn = pd.read_parquet("data/2025_Connectivity_783.parquet")
sub = conn[conn["Presynaptic_ID"].isin(active_set) & conn["Postsynaptic_ID"].isin(active_set)]
print(f"{len(sub)} real synapses among them")

G = nx.DiGraph()
for nid in active_set:
    G.add_node(nid, spikes=int(spike_counts.get(nid, 0)))
G.add_weighted_edges_from(zip(sub["Presynaptic_ID"], sub["Postsynaptic_ID"], sub["Connectivity"]))
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

pos = nx.spring_layout(G, seed=42, k=0.3)
node_colors = [G.nodes[n]["spikes"] for n in G.nodes]

plt.figure(figsize=(12, 12))
nx.draw_networkx_edges(G, pos, alpha=0.15, arrows=False, width=0.5)
nodes = nx.draw_networkx_nodes(G, pos, node_size=40, node_color=node_colors, cmap="plasma")
plt.colorbar(nodes, label="Spike count")
plt.title(f"Real fly brain subnetwork: top {TOP_N} most active neurons")
plt.axis("off")
plt.tight_layout()
plt.savefig("brain_network_static.png", dpi=150)
print("Saved to brain_network_static.png")