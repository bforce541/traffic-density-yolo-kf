from ultralytics import YOLO
import cv2
import pandas as pd
import os

def run_detection():
    video_path = "data/traffic.mp4"
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    detections_csv = os.path.join(results_dir, "detections.csv")

    model = YOLO("yolov8s.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_idx = 0
    rows = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            if cls_id not in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())

            rows.append({
                "frame": frame_idx,
                "cls_id": cls_id,
                "conf": conf,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx} frames...")

    cap.release()

    df = pd.DataFrame(rows)
    df.to_csv(detections_csv, index=False)
    print(f"Saved detections to {detections_csv}")
    print(f"Total frames processed: {frame_idx}")

if __name__ == "__main__":
    run_detection()

