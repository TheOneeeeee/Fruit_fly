import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

T_RUN = 10.0
DRIVEN_IDS = {720575940619072513,720575940646212996,720575940622298631,720575940642088333,720575940627692048,720575940617239197,720575940618682526,
               720575940604714528,720575940603266592,720575940604027168,720575940619197093,720575940610259370,720575940627578156,720575940629481516,
               720575940618887217,720575940614281266,720575940634859188,720575940645743412,720575940637742911,720575940617094208,720575940629416318,
               720575940630195909,720575940615641798,720575940638312262,720575940624310345,720575940621778381,720575940619659861,720575940629146711,
               720575940625750105,720575940610483162,720575940610481370,720575940602353632,720575940610773090,720575940617433830,720575940628962407,
               720575940626287336,720575940623183083,720575940618025199,720575940619028208,720575940621864060,720575940613061118,720575940621008895}

spikes = pd.read_parquet("data/results/Bitter_200Hz.parquet")
spike_counts = spikes.groupby("flywire_id").size()
print(f"{spike_counts.shape[0]} total active neurons, {len(DRIVEN_IDS)} directly driven")

TOP_N = 300
downstream = spike_counts.drop(labels=DRIVEN_IDS, errors="ignore")
active_set = set(downstream.sort_values(ascending=False).head(TOP_N).index)
print(f"Plotting {len(active_set)} downstream responders")

conn = pd.read_parquet("data/2025_Connectivity_783.parquet")
sub = conn[conn["Presynaptic_ID"].isin(active_set) & conn["Postsynaptic_ID"].isin(active_set)]

G = nx.DiGraph()
for nid in active_set:
    G.add_node(nid)
G.add_weighted_edges_from(zip(sub["Presynaptic_ID"], sub["Postsynaptic_ID"], sub["Connectivity"]))
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

pos = nx.spring_layout(G, seed=42, k=0.3)
node_list = list(G.nodes)
node_index = {nid: i for i, nid in enumerate(node_list)}

spikes_sub = spikes[spikes["flywire_id"].isin(active_set)].sort_values("t")

FPS = 20
N_FRAMES = int(T_RUN * FPS)
DECAY = 0.6

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
title = ax.set_title("BITTER — t = 0.00s")
ax.axis("off")

def update(frame):
    brightness[:] *= DECAY
    for idx in frame_spikes[frame]:
        brightness[idx] = 1.0
    scat.set_array(brightness.copy())
    title.set_text(f"BITTER — t = {frame / FPS:.2f}s")
    return scat, title

ani = animation.FuncAnimation(fig, update, frames=N_FRAMES, interval=1000/FPS)
ani.save("brain_activity_bitter.mp4", writer="ffmpeg", fps=FPS, dpi=120)
print("Saved to brain_activity_bitter.mp4")