import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_results():
    df = pd.read_csv("results/density.csv")
    outdir = "results/figures"
    os.makedirs(outdir, exist_ok=True)

    # vehicle count
    plt.figure()
    plt.plot(df["frame"], df["vehicle_count"])
    plt.xlabel("Frame")
    plt.ylabel("Vehicle Count")
    plt.title("Traffic Density")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "vehicle_count.png"), dpi=300)

    # congestion score
    plt.figure()
    plt.plot(df["frame"], df["congestion_score"])
    plt.xlabel("Frame")
    plt.ylabel("Congestion Score")
    plt.title("Congestion Score Over Time")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "congestion_score.png"), dpi=300)

    print(f"Saved plots in {outdir}")

if __name__ == "__main__":
    plot_results()
