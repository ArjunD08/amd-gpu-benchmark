import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# Load benchmark results
# ------------------------------------------------------------

input_file = "results/benchmark.csv"
output_file = "plots/speedup_comparison.png"

df = pd.read_csv(input_file)


# ------------------------------------------------------------
# Display speedup values
# ------------------------------------------------------------

print("\nGPU Speedup")
print("-----------")

for _, row in df.iterrows():

    print(
        f"{int(row['matrix_size'])} x "
        f"{int(row['matrix_size'])}: "
        f"{row['gpu_speedup']:.2f}x"
    )


# ------------------------------------------------------------
# Create plot
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    df["matrix_size"],
    df["gpu_speedup"],
    marker="o",
    linewidth=2
)

plt.xscale("log", base=2)

plt.xlabel("Matrix Size (N × N)")
plt.ylabel("GPU Speedup (×)")

plt.title("AMD Radeon 8060S GPU Speedup vs CPU")

plt.grid(True, which="both", linestyle="--", alpha=0.5)

plt.xticks(
    df["matrix_size"],
    [f"{int(size)}²" for size in df["matrix_size"]]
)

plt.tight_layout()


# ------------------------------------------------------------
# Save plot
# ------------------------------------------------------------

plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(f"\nPlot saved to: {output_file}")
