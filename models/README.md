# Model Weights Documentation

This directory is intended to store pre-trained and fine-tuned model weights for the Sentinel AI system.

## YOLOv8 Weights

The ultralytics library will automatically download standard YOLOv8 weights (e.g., `yolov8n.pt`, `yolov8s.pt`) to this directory or the root directory when you first run the system. 

If you want to manually download them or use custom weights:
1. Place the `.pt` files in this directory.
2. Update the `config/config.yaml` to point to the correct model file:
   ```yaml
   detection:
     model: models/custom_yolov8_weights.pt
   ```

## DeepSORT Checkpoints

If you are using a custom feature extractor for DeepSORT, place the PyTorch checkpoint (`.pth`) here.

## CCVE (Cross-Camera View Embedder) Weights

Once trained, the weights for the CCVE spatial-temporal encoding layers should be saved here as `ccve_model_v1.pth`.

## Pretrained Model Performance Metrics (Placeholder)

| Model | Resolution | mAP50-95 | Inference Time (RTX 3060) |
|-------|------------|----------|---------------------------|
| YOLOv8n | 640x640  | 37.3     | ~4ms                      |
| YOLOv8s | 640x640  | 44.9     | ~9ms                      |
| CCVE-v1 | Feature-level | N/A | ~2ms (per fusion step)    |
