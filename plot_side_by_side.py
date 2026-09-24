import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

T_RUN = 10.0
FPS = 20
N_FRAMES = int(T_RUN * FPS)
DECAY = 0.6
TOP_N = 300

SUGAR_DRIVEN = {720575940616885538,720575940630233916,720575940639332736,720575940632889389,720575940617000768,720575940632425919,720575940637568838,
                720575940629176663,720575940621502051,720575940638202345,720575940612670570,720575940611875570,720575940621754367,720575940633143833,
                720575940613601698,720575940630797113,720575940639198653,720575940639259967,720575940624963786,720575940640649691,720575940610788069,
                720575940623172843,720575940628853239}

BITTER_DRIVEN = {720575940619072513,720575940646212996,720575940622298631,720575940642088333,720575940627692048,720575940617239197,720575940618682526,
               720575940604714528,720575940603266592,720575940604027168,720575940619197093,720575940610259370,720575940627578156,720575940629481516,
               720575940618887217,720575940614281266,720575940634859188,720575940645743412,720575940637742911,720575940617094208,720575940629416318,
               720575940630195909,720575940615641798,720575940638312262,720575940624310345,720575940621778381,720575940619659861,720575940629146711,
               720575940625750105,720575940610483162,720575940610481370,720575940602353632,720575940610773090,720575940617433830,720575940628962407,
               720575940626287336,720575940623183083,720575940618025199,720575940619028208,720575940621864060,720575940613061118,720575940621008895}

conn = pd.read_parquet("data/2025_Connectivity_783.parquet")

def build_panel(spike_file, driven_ids, label):
    spikes = pd.read_parquet(spike_file)
    counts = spikes.groupby("flywire_id").size()
    downstream = counts.drop(labels=driven_ids, errors="ignore")
    active_set = set(downstream.sort_values(ascending=False).head(TOP_N).index)

    sub = conn[conn["Presynaptic_ID"].isin(active_set) & conn["Postsynaptic_ID"].isin(active_set)]
    G = nx.DiGraph()
    for nid in active_set:
        G.add_node(nid)
    G.add_weighted_edges_from(zip(sub["Presynaptic_ID"], sub["Postsynaptic_ID"], sub["Connectivity"]))
    print(f"{label}: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    pos = nx.spring_layout(G, seed=42, k=0.3)
    node_list = list(G.nodes)
    node_index = {nid: i for i, nid in enumerate(node_list)}

    spikes_sub = spikes[spikes["flywire_id"].isin(active_set)].sort_values("t")
    frame_spikes = [[] for _ in range(N_FRAMES)]
    for nid, t in zip(spikes_sub["flywire_id"], spikes_sub["t"]):
        idx = min(int(t / T_RUN * N_FRAMES), N_FRAMES - 1)
        frame_spikes[idx].append(node_index[nid])

    return G, pos, node_list, frame_spikes

G_sugar, pos_sugar, nodes_sugar, fs_sugar = build_panel("data/results/brian2cpp_t10.0s_n1.parquet", SUGAR_DRIVEN, "Sugar")
G_bitter, pos_bitter, nodes_bitter, fs_bitter = build_panel("data/results/Bitter_200Hz.parquet", BITTER_DRIVEN, "Bitter")

bright_sugar = np.zeros(len(nodes_sugar))
bright_bitter = np.zeros(len(nodes_bitter))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

for ax, G, pos, label in [(ax1, G_sugar, pos_sugar, "SUGAR"), (ax2, G_bitter, pos_bitter, "BITTER")]:
    nx.draw_networkx_edges(G, pos, alpha=0.1, arrows=False, width=0.5, ax=ax)
    ax.axis("off")

xs1 = [pos_sugar[n][0] for n in nodes_sugar]; ys1 = [pos_sugar[n][1] for n in nodes_sugar]
xs2 = [pos_bitter[n][0] for n in nodes_bitter]; ys2 = [pos_bitter[n][1] for n in nodes_bitter]

scat1 = ax1.scatter(xs1, ys1, s=40, c=bright_sugar, cmap="plasma", vmin=0, vmax=1)
scat2 = ax2.scatter(xs2, ys2, s=40, c=bright_bitter, cmap="plasma", vmin=0, vmax=1)
ax1.set_title("SUGAR — t = 0.00s")
ax2.set_title("BITTER — t = 0.00s")

def update(frame):
    bright_sugar[:] *= DECAY
    for idx in fs_sugar[frame]:
        bright_sugar[idx] = 1.0
    bright_bitter[:] *= DECAY
    for idx in fs_bitter[frame]:
        bright_bitter[idx] = 1.0

    scat1.set_array(bright_sugar.copy())
    scat2.set_array(bright_bitter.copy())
    t = frame / FPS
    ax1.set_title(f"SUGAR — t = {t:.2f}s")
    ax2.set_title(f"BITTER — t = {t:.2f}s")
    return scat1, scat2

ani = animation.FuncAnimation(fig, update, frames=N_FRAMES, interval=1000/FPS)
ani.save("brain_comparison.mp4", writer="ffmpeg", fps=FPS, dpi=120)
print("Saved to brain_comparison.mp4")