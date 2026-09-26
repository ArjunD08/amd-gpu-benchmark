import csv

import matplotlib.pyplot as plt


# -------------------------
# File paths
# -------------------------

csv_path = "results/benchmark.csv"
plot_path = "plots/speedup_comparison.png"


# -------------------------
# Read benchmark data
# -------------------------

matrix_sizes = []
speedups = []

with open(csv_path, "r") as csv_file:
    reader = csv.DictReader(csv_file)

    for row in reader:
        matrix_size = int(row["matrix_size"])

        cpu_time = float(row["cpu_average_ms"])
        gpu_time = float(row["gpu_average_ms"])

        speedup = cpu_time / gpu_time

        matrix_sizes.append(matrix_size)
        speedups.append(speedup)


# -------------------------
# Print speedup values
# -------------------------

print("\nGPU Speedup")
print("-----------")

for size, speedup in zip(matrix_sizes, speedups):
    print(f"{size} x {size}: {speedup:.2f}x")


# -------------------------
# Create plot
# -------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    matrix_sizes,
    speedups,
    marker="o",
    label="GPU Speedup"
)


# -------------------------
# Axis labels
# -------------------------

plt.xlabel("Matrix Size (N × N)")
plt.ylabel("Speedup (×)")

plt.title("AMD Radeon 8060S GPU Speedup vs CPU")


# -------------------------
# X-axis
# -------------------------

plt.xscale("log", base=2)

plt.xticks(
    matrix_sizes,
    ["1024", "2048", "4096", "8192"]
)


# -------------------------
# Grid and legend
# -------------------------

plt.grid(
    True,
    which="both",
    linestyle="--",
    alpha=0.5
)

plt.legend()


# -------------------------
# Layout
# -------------------------

plt.tight_layout()


# -------------------------
# Save plot
# -------------------------

plt.savefig(
    plot_path,
    dpi=300
)

plt.close()

print(f"\nPlot saved to: {plot_path}")
