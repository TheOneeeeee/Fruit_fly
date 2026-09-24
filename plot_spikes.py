import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet("data/results/brian2cpp_t1.0s_n1.parquet")
print(df.columns.tolist())

# Compress the huge real FlyWire IDs into a compact 0..N index, just for plotting
df["neuron_row"] = pd.factorize(df["flywire_id"])[0]

plt.figure(figsize=(10, 6))
plt.scatter(df["t"], df["neuron_row"], s=2, color="black")
plt.xlabel("Time (s)")
plt.ylabel("Neuron (compact index, not real ID)")
plt.title("Spike raster: real fly brain response to sugar-neuron stimulation")
plt.tight_layout()
plt.savefig("spike_raster.png", dpi=150)
print("Saved to spike_raster.png")