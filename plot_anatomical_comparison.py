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
all_x = ann["soma_x"].values
all_y = ann["soma_y"].values

def build_panel(spike_file):
    spikes = pd.read_parquet(spike_file)
    spikes = spikes.merge(ann, left_on="flywire_id", right_on="root_id", how="inner")
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

    return xs, ys, frame_spikes

xs_sugar, ys_sugar, fs_sugar = build_panel("data/results/brian2cpp_t10.0s_n1.parquet")
xs_bitter, ys_bitter, fs_bitter = build_panel("data/results/Bitter_200Hz.parquet")

bright_sugar = np.zeros(len(xs_sugar))
bright_bitter = np.zeros(len(xs_bitter))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), facecolor="black")
for ax in (ax1, ax2):
    ax.set_facecolor("black")
    ax.scatter(all_x, all_y, s=0.3, color="#333355", alpha=0.5)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")

scat1 = ax1.scatter(xs_sugar, ys_sugar, s=15, c=bright_sugar, cmap="plasma", vmin=0, vmax=1)
scat2 = ax2.scatter(xs_bitter, ys_bitter, s=15, c=bright_bitter, cmap="plasma", vmin=0, vmax=1)
title1 = ax1.set_title("SUGAR — t = 0.00s", color="white")
title2 = ax2.set_title("BITTER — t = 0.00s", color="white")

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
    title1.set_text(f"SUGAR — t = {t:.2f}s")
    title2.set_text(f"BITTER — t = {t:.2f}s")
    return scat1, scat2

ani = animation.FuncAnimation(fig, update, frames=N_FRAMES, interval=1000/FPS)
ani.save("brain_comparison_anatomical.mp4", writer="ffmpeg", fps=FPS, dpi=120, savefig_kwargs={"facecolor": "black"})
print("Saved to brain_comparison_anatomical.mp4")