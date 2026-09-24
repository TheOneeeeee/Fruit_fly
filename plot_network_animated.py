import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

T_RUN = 10.0
spikes = pd.read_parquet(f"data/results/brian2cpp_t{T_RUN}s_n1.parquet")
spike_counts = spikes.groupby("flywire_id").size()

TOP_N = 300
ranked = spike_counts.sort_values(ascending=False)
active_set = set(ranked.iloc[21:21+TOP_N].index)  # skip the 21 most active (the direct drivers)

conn = pd.read_parquet("data/2025_Connectivity_783.parquet")
sub = conn[conn["Presynaptic_ID"].isin(active_set) & conn["Postsynaptic_ID"].isin(active_set)]

G = nx.DiGraph()
for nid in active_set:
    G.add_node(nid)
G.add_weighted_edges_from(zip(sub["Presynaptic_ID"], sub["Postsynaptic_ID"], sub["Connectivity"]))
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

pos = nx.spring_layout(G, seed=42, k=0.3)  # same seed as your static plot, same shape
node_list = list(G.nodes)
node_index = {nid: i for i, nid in enumerate(node_list)}

spikes_sub = spikes[spikes["flywire_id"].isin(active_set)].sort_values("t")

FPS = 20
N_FRAMES = int(T_RUN * FPS)
DECAY = 0.6  # how fast a neuron fades back to dark after spiking

frame_spikes = [[] for _ in range(N_FRAMES)]
for nid, t in zip(spikes_sub["flywire_id"], spikes_sub["t"]):
    idx = min(int(t / T_RUN * N_FRAMES), N_FRAMES - 1)
    frame_spikes[idx].append(node_index[nid])

brightness = np.zeros(len(node_list))
xs = [pos[n][0] for n in node_list]
ys = [pos[n][1] for n in node_list]

fig, ax = plt.subplots(figsize=(10, 10))
nx.draw_networkx_edges(G, pos, alpha=0.1, arrows=False, width=0.5, ax=ax)
scat = ax.scatter(xs, ys, s=40, c=brightness, cmap="plasma", vmin=0, vmax=1)
title = ax.set_title("t = 0.00s")
ax.axis("off")

def update(frame):
    brightness[:] *= DECAY
    for idx in frame_spikes[frame]:
        brightness[idx] = 1.0
    scat.set_array(brightness.copy())
    title.set_text(f"t = {frame / FPS:.2f}s")
    return scat, title

ani = animation.FuncAnimation(fig, update, frames=N_FRAMES, interval=1000/FPS)
ani.save("brain_activity.mp4", writer="ffmpeg", fps=FPS, dpi=120)
print("Saved to brain_activity.mp4")