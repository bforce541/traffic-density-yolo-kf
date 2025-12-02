import pandas as pd
import os

def compute_density():
    tracks_path = "results/tracks.csv"
    out_path = "results/density.csv"
    os.makedirs("results", exist_ok=True)

    df = pd.read_csv(tracks_path)

    frame_counts = df.groupby("frame")["track_id"].nunique().reset_index()
    frame_counts.rename(columns={"track_id": "vehicle_count"}, inplace=True)

    maxv = frame_counts["vehicle_count"].max()
    frame_counts["congestion_score"] = frame_counts["vehicle_count"] / maxv

    frame_counts.to_csv(out_path, index=False)
    print(f"Saved density metrics to {out_path}")

if __name__ == "__main__":
    compute_density()
