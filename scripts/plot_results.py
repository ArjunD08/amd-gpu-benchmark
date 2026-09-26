import csv

import matplotlib.pyplot as plt


# -------------------------
# File paths
# -------------------------

csv_path = "results/benchmark.csv"
plot_path = "plots/performance_comparison.png"


# -------------------------
# Read benchmark data
# -------------------------

matrix_sizes = []
cpu_times = []
gpu_times = []

with open(csv_path, "r") as csv_file:
    reader = csv.DictReader(csv_file)

    for row in reader:
        matrix_sizes.append(int(row["matrix_size"]))
        cpu_times.append(float(row["cpu_average_ms"]))
        gpu_times.append(float(row["gpu_average_ms"]))


# -------------------------
# Create plot
# -------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    matrix_sizes,
    cpu_times,
    marker="o",
    label="CPU"
)

plt.plot(
    matrix_sizes,
    gpu_times,
    marker="o",
    label="AMD Radeon 8060S GPU"
)


# -------------------------
# Axis labels
# -------------------------

plt.xlabel("Matrix Size (N × N)")
plt.ylabel("Average Execution Time (ms)")

plt.title("CPU vs AMD GPU Matrix Multiplication")


# -------------------------
# Logarithmic axes
# -------------------------

plt.xscale("log", base=2)
plt.yscale("log")


# Show actual matrix sizes
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

print(f"Plot saved to: {plot_path}")
