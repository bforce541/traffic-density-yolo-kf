import numpy as np
import pandas as pd
import os
from filterpy.kalman import KalmanFilter

IOU_THRESHOLD = 0.3
MAX_AGE = 30
MIN_HITS = 3

def iou(bb_test, bb_gt):
    xx1 = max(bb_test[0], bb_gt[0])
    yy1 = max(bb_test[1], bb_gt[1])
    xx2 = min(bb_test[2], bb_gt[2])
    yy2 = min(bb_test[3], bb_gt[3])
    w = max(0., xx2 - xx1)
    h = max(0., yy2 - yy1)
    inter = w * h
    area1 = (bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
    area2 = (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def create_kf(cx, cy, w, h):
    kf = KalmanFilter(dim_x=8, dim_z=4)
    dt = 1.

    kf.F = np.array([
        [1,0,0,0,dt,0,0,0],
        [0,1,0,0,0,dt,0,0],
        [0,0,1,0,0,0,dt,0],
        [0,0,0,1,0,0,0,dt],
        [0,0,0,0,1,0,0,0],
        [0,0,0,0,0,1,0,0],
        [0,0,0,0,0,0,1,0],
        [0,0,0,0,0,0,0,1],
    ])

    kf.H = np.array([
        [1,0,0,0,0,0,0,0],
        [0,1,0,0,0,0,0,0],
        [0,0,1,0,0,0,0,0],
        [0,0,0,1,0,0,0,0],
    ])

    kf.R *= 10.
    kf.P *= 100.
    kf.Q *= 0.01

    kf.x[:4] = np.array([[cx],[cy],[w],[h]])
    return kf

class Track:
    def __init__(self, tid, bbox, frame, cls_id):
        x1,y1,x2,y2 = bbox
        w, h = x2-x1, y2-y1
        cx, cy = x1 + w/2, y1 + h/2

        self.id = tid
        self.cls_id = cls_id
        self.kf = create_kf(cx,cy,w,h)
        self.time_since_update = 0
        self.hits = 1
        self.age = 0
        self.history = [(frame,x1,y1,x2,y2)]

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        self.age += 1

    def update(self, bbox, frame):
        x1,y1,x2,y2 = bbox
        w, h = x2-x1, y2-y1
        cx, cy = x1 + w/2, y1 + h/2
        z = np.array([[cx],[cy],[w],[h]])
        self.kf.update(z)
        self.time_since_update = 0
        self.hits += 1
        self.history.append((frame,x1,y1,x2,y2))

    def get_state(self):
        x = self.kf.x
        cx,cy,w,h = x[0],x[1],x[2],x[3]
        return float(cx-w/2), float(cy-h/2), float(cx+w/2), float(cy+h/2)

def run_tracker():
    det_path = "results/detections.csv"
    out_path = "results/tracks.csv"
    os.makedirs("results", exist_ok=True)

    df = pd.read_csv(det_path)
    max_frame = df["frame"].max()

    tracks = []
    next_id = 1
    rows = []

    for frame in range(max_frame + 1):
        fdf = df[df.frame == frame]
        dets = fdf[["x1","y1","x2","y2"]].to_numpy()
        det_cls = fdf["cls_id"].to_numpy()

        # predict existing tracks
        for t in tracks: t.predict()

        used = set()

        # match detections → tracks
        for t in tracks:
            best_iou, best_idx = 0, -1
            tbox = t.get_state()
            for i,b in enumerate(dets):
                if i in used: continue
                iouv = iou(tbox,b)
                if iouv > best_iou:
                    best_iou, best_idx = iouv, i

            if best_idx >= 0 and best_iou >= IOU_THRESHOLD:
                bbox = dets[best_idx]
                t.update(bbox, frame)
                used.add(best_idx)
                x1,y1,x2,y2 = bbox
                rows.append({
                    "frame": frame,
                    "track_id": t.id,
                    "cls_id": t.cls_id,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2
                })

        # new tracks for unmatched detections
        for i,bbox in enumerate(dets):
            if i in used: continue
            cls_id = int(det_cls[i])
            new = Track(next_id,bbox,frame,cls_id)
            x1,y1,x2,y2 = bbox
            rows.append({
                "frame": frame,
                "track_id": next_id,
                "cls_id": cls_id,
                "x1": x1,"y1": y1,"x2": x2,"y2": y2
            })
            tracks.append(new)
            next_id += 1

        # prune old tracks
        tracks = [t for t in tracks if t.time_since_update <= MAX_AGE]

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved tracks to {out_path}")

if __name__ == "__main__":
    run_tracker()

