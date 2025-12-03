# Real-Time Traffic Monitoring Using YOLOv8 and Kalman Filter Tracking

This repository contains the full implementation, figures, and data outputs for the paper:

**Real-Time Traffic Monitoring Using Deep Learning and Multi-Object Tracking**
Yoshua Alexander, The University of Texas at Dallas

This project demonstrates a lightweight vision-based system that detects, tracks, and analyzes traffic flow using only a single fixed video camera. Vehicle detections are generated using **YOLOv8**, and multi-object tracking is achieved via a **Kalman filter** with IoU-based association. From these tracks, the system computes frame-level **vehicle density** and a **normalized congestion score**. All results, figures, and code are included for full reproducibility.

---

## Features

* **YOLOv8-based** vehicle detection on each video frame.
* **Multi-object tracking** using a Kalman filter.
* Generation of unique, stable **track IDs**.
* Frame-level **vehicle count** extraction.
* **Normalized congestion metric** mapping traffic level to $[0,1]$.
* Automatic **CSV export** for both metrics.
* Plot generation for density and congestion curves.
* Fully reproducible pipeline requiring **no GPU**.

---

## Installation

### Clone the repository:

```bash
git clone [https://github.com/bforce541/traffic-density-yolo-kf.git](https://github.com/bforce541/traffic-density-yolo-kf.git)
cd traffic-density-yolo-kf