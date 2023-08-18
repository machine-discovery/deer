# %%
import numpy as np
import matplotlib.pyplot as plt

speedups = np.full((6, 7, 5), np.nan)
batch_size = 16
fname = "fwd-speedup-report.txt"
nhs = [1, 2, 4, 8, 16, 32]
nsequences = [1000, 3000, 10000, 30000, 100000, 300000, 1000000]
nsequences_labels = ["1k", "3k", "10k", "30k", "100k", "300k", "1M"]
with open(fname, "r") as f:
    for line in f:
        if line.startswith("nh:"):
            nh, nsequence, seed = [int(c) for c in line.split()[1::2]]
            nh_idx = nhs.index(nh)
            nsequence_idx = nsequences.index(nsequence)
        elif line.startswith("Speedup of DEER GRU over Sequential GRU: "):
            speedups[nh_idx, nsequence_idx, seed] = float(line.split()[-1])

mean_speedups = np.mean(speedups, axis=-1)  # (nhs, nsequences)
width = 1 / (len(nhs) + 2)
print(mean_speedups.shape)

# create a bar plot with nsequences as the x-axis and nh as the hue
plt.figure(figsize=(12, 4))
for i in range(len(nhs)):
    plt.bar(np.arange(len(nsequences)) + i * width, mean_speedups[i], width=width, label=f"dim = {nhs[i]}")

plt.gca().hlines(1.0, -width, len(nsequences) - 1 + (len(nhs) - 1) * width, color="k", linestyle="--")
plt.xticks(np.arange(len(nsequences)) + ((len(nhs) - 1) / 2 * width), nsequences_labels, fontsize=12)
plt.xlabel("Sequence length", fontsize=14)
plt.ylabel("Speed up", fontsize=14)
plt.title(f"Speed up of DEER GRU over sequential GRU with batch size = {batch_size}", fontsize=16)
plt.gca().set_yscale("log")
plt.grid()
plt.yticks(fontsize=12)
plt.legend()
plt.tight_layout()
plt.savefig("test.png")
plt.close()
