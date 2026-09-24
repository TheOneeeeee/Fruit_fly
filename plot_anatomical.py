import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

T_RUN = 10.0
FPS = 20
N_FRAMES = int(T_RUN * FPS)
DECAY = 0.6

ann = pd.read_csv("data/flywire_annotations.tsv", sep="\t", low_memory=False)
ann = ann.dropna(subset=["soma_x", "soma_y"])[["root_id", "soma_x", "soma_y"]]
print(f"{len(ann)} neurons have soma position data")

# ALL neuron positions -> the dim background "brain shape"
all_x = ann["soma_x"].values
all_y = ann["soma_y"].values

spikes = pd.read_parquet("data/results/brian2cpp_t10.0s_n1.parquet")
spikes = spikes.merge(ann, left_on="flywire_id", right_on="root_id", how="inner")
print(f"{spikes['flywire_id'].nunique()} active neurons matched to a soma position")

active_ids = spikes["flywire_id"].unique()
active_pos = ann[ann["root_id"].isin(active_ids)].set_index("root_id")
node_index = {nid: i for i, nid in enumerate(active_pos.index)}
xs = active_pos["soma_x"].values
ys = active_pos["soma_y"].values

frame_spikes = [[] for _ in range(N_FRAMES)]
for nid, t in zip(spikes["flywire_id"], spikes["t"]):
    if nid in node_index:
        idx = min(int(t / T_RUN * N_FRAMES), N_FRAMES - 1)
        frame_spikes[idx].append(node_index[nid])

brightness = np.zeros(len(xs))

fig, ax = plt.subplots(figsize=(10, 6), facecolor="black")
ax.set_facecolor("black")
ax.scatter(all_x, all_y, s=0.3, color="#333355", alpha=0.5)  # dim full-brain silhouette
scat = ax.scatter(xs, ys, s=15, c=brightness, cmap="plasma", vmin=0, vmax=1)
title = ax.set_title("t = 0.00s", color="white")
ax.invert_yaxis()  # FlyWire's y-axis convention often renders upside-down otherwise
ax.set_aspect("equal")
ax.axis("off")

def update(frame):
    brightness[:] *= DECAY
    for idx in frame_spikes[frame]:
        brightness[idx] = 1.0
    scat.set_array(brightness.copy())
    title.set_text(f"t = {frame / FPS:.2f}s")
    return scat, title

ani = animation.FuncAnimation(fig, update, frames=N_FRAMES, interval=1000/FPS)
ani.save("brain_activity_anatomical.mp4", writer="ffmpeg", fps=FPS, dpi=120, savefig_kwargs={"facecolor": "black"})
print("Saved to brain_activity_anatomical.mp4")