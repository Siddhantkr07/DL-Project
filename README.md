# 🛡️ Sentinel AI - Cognitive Multi-Camera Vision System

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-latest-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-blue.svg)

## Project Description

Sentinel AI is a cognitive multi-camera surveillance system designed for dynamic real-time threat detection, tracking, and resource allocation. It moves beyond traditional passive surveillance by actively analyzing feeds from multiple cameras to identify emergencies, track individuals across feeds, and intelligently prioritize critical streams. By utilizing a Multi-Camera Cross-View Embedder (CCVE) and an Event Prioritization & Resource Allocator (EPRA) module, Sentinel AI dramatically reduces the cognitive load on human operators while ensuring optimal utilization of computational resources.

## Architecture Diagram

```text
+-------------------------------------------------------------------------------------------------+
|                                     Sentinel AI Architecture                                    |
+-------------------------------------------------------------------------------------------------+
|   Camera 1   | |   Camera 2   | |   Camera 3   | ... |   Camera N   | (Input Streams)           |
+------+-------+ +------+-------+ +------+-------+     +------+-------+                           |
       |                |                |                    |                                   |
+------v----------------v----------------v--------------------v-----------------------------------+
|                            Dynamic Feature Extraction Pipeline                                  |
|  [ YOLOv8 Object Detection ] + [ DeepSORT Object Tracking ] + [ Spatial-Temporal Encoding ]     |
+----------------------------------------+--------------------------------------------------------+
                                         |
+----------------------------------------v--------------------------------------------------------+
|                      Cross-Camera View Embedder (CCVE) / Data Fusion                            |
|       Integrates Multi-View Features -> Attention Mechanism -> Unified Spatial Context          |
+----------------------------------------+--------------------------------------------------------+
                                         |
+----------------------------------------v--------------------------------------------------------+
|                  Event Prioritization & Resource Allocator (EPRA) Module                        |
|  [ Compute Threat Severity ] -> [ Estimate Proximity/Context ] -> [ Stream Priority Score ]     |
+----------------------------------------+--------------------------------------------------------+
                                         |
+----------------------------------------v--------------------------------------------------------+
|                                Command & Control Dashboard                                      |
|    - High-Priority Feeds Enlarged               - Real-Time Alerts & Bounding Boxes             |
|    - Cross-Camera Tracking Visualization        - System Health & Resource Metrics              |
+-------------------------------------------------------------------------------------------------+
```

## Features

- **Multi-Camera Synthesis:** Fuses multiple video streams into a unified understanding of the physical space.
- **Dynamic Threat Detection:** Identifies anomalies (e.g., weapons, fights, fires, anomalous crowd behavior) in real-time.
- **Cross-Camera Tracking:** Tracks individuals consistently as they move across different camera fields of view.
- **Intelligent Resource Allocation:** Dynamically reallocates CPU/GPU resources and dashboard screen space to the most critical events.
- **Operator-Centric Dashboard:** Automatically surfaces high-priority feeds, reducing fatigue and improving response times.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/Sentinel-AI.git
cd Sentinel-AI

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Run the system in simulation mode to see a demonstration without external hardware:

```bash
python main.py --mode demo --cameras 4 --dashboard True
```

This will spin up 4 simulated cameras, start the AI pipeline, and launch the web dashboard at `http://localhost:5000`.

## Dataset Setup

We use the UCF-Crime dataset from Kaggle for training and testing anomaly detection scenarios.

1. Ensure you have a Kaggle account and an API key (`kaggle.json`).
2. Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<User>\.kaggle\` (Windows).
3. Run the download script:

```bash
python data/download_dataset.py
```

## Project Structure

```text
Sentinel-AI/
├── config/
│   └── config.yaml             # System configuration parameters
├── data/
│   ├── download_dataset.py     # Script to fetch UCF-Crime/MOT17
│   ├── dataset_loader.py       # Data loading and preprocessing pipeline
│   └── README.md
├── models/
│   └── README.md               # Model weights documentation
├── notebooks/
│   └── sentinel_ai_demo.py     # Interactive demo script
├── simulation/
│   ├── __init__.py
│   ├── camera_simulator.py     # Multi-camera feed simulation
│   └── scenario_generator.py   # Synthetic event scenario generation
├── tests/
│   ├── __init__.py
│   ├── test_detector.py
│   └── test_prioritizer.py
├── .gitignore
├── main.py                     # Main application entry point
├── README.md                   # Project documentation (You are here)
└── requirements.txt            # Python dependencies
```

## How It Works

1. **Feature Extraction Pipeline:** Captures video streams and runs YOLOv8 for rapid object detection, followed by DeepSORT for reliable frame-to-frame object tracking.
2. **Cross-Camera View Embedder (CCVE):** A custom neural network layer that takes features from multiple cameras and aligns them spatially to map how objects in one camera relate to another.
3. **Event Prioritization Module (EPRA):** Computes a dynamic Priority Score for each camera feed based on the severity of detections, the proximity to critical areas, and historical context.
4. **Command Dashboard:** A Flask-based web interface that dynamically adjusts its layout to emphasize high-priority camera feeds and logs critical alerts.

## API Documentation

- `GET /api/status`: Returns current system health and camera status.
- `GET /api/alerts`: Returns recent high-priority alerts.
- `WS /stream`: WebSocket connection for real-time video stream frames and metadata.

## Citation

If you find this project useful, please consider citing it.
