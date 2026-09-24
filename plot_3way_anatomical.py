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

    print(f"{spike_file}: {len(xs)} active neurons matched")
    return xs, ys, frame_spikes

panels_data = [
    ("SUGAR", "data/results/brian2cpp_t10.0s_n1.parquet"),
    ("BITTER", "data/results/Bitter_200Hz.parquet"),
    ("FEAR", "data/results/Fear_150Hz.parquet"),
]

panels = [(label, *build_panel(f)) for label, f in panels_data]
brightness = [np.zeros(len(xs)) for _, xs, ys, fs in panels]

fig, axes = plt.subplots(1, 3, figsize=(24, 6), facecolor="black")
scats, titles = [], []
for ax, (label, xs, ys, fs) in zip(axes, panels):
    ax.set_facecolor("black")
    ax.scatter(all_x, all_y, s=0.15, color="#333355", alpha=0.5)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.axis("off")
    scat = ax.scatter(xs, ys, s=15, c=np.zeros(len(xs)), cmap="plasma", vmin=0, vmax=1)
    title = ax.set_title(f"{label} — t = 0.00s", color="white")
    scats.append(scat)
    titles.append(title)

def update(frame):
    t = frame / FPS
    for i, (label, xs, ys, fs) in enumerate(panels):
        brightness[i][:] *= DECAY
        for idx in fs[frame]:
            brightness[i][idx] = 1.0
        scats[i].set_array(brightness[i].copy())
        titles[i].set_text(f"{label} — t = {t:.2f}s")
    return scats + titles

ani = animation.FuncAnimation(fig, update, frames=N_FRAMES, interval=1000/FPS)
ani.save("brain_comparison_3way.mp4", writer="ffmpeg", fps=FPS, dpi=120, savefig_kwargs={"facecolor": "black"})
print("Saved to brain_comparison_3way.mp4")