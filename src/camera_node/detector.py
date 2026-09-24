import logging
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None
    logging.warning("ultralytics not installed. YOLODetector will not work.")

@dataclass
class Detection:
    """Dataclass representing a single detection."""
    bbox: List[float]  # [x1, y1, x2, y2]
    class_id: int
    class_name: str
    confidence: float

class YOLODetector:
    """
    YOLOv8 detector class using ultralytics.
    Provides object detection capabilities for Sentinel AI.
    """
    
    # Common classes to filter for surveillance (COCO dataset)
    TARGET_CLASSES = {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 
        7: 'truck', 8: 'boat'
    }

    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5, device: str = None):
        """
        Initializes the YOLOv8 detector.
        
        Args:
            model_path: Path to the YOLOv8 model weights.
            confidence_threshold: Minimum confidence score for a detection to be kept.
            device: Target device ('cuda', 'cpu', etc.). If None, auto-detects.
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
        if YOLO is None:
            raise RuntimeError("ultralytics library is required but not found. Install it via 'pip install ultralytics'.")
            
        try:
            self.model = YOLO(model_path)
            # Default to CPU if no device specified to ensure fallback, but Ultralytics auto-selects well
            if device:
                self.model.to(device)
            logging.info(f"Initialized YOLODetector with {model_path} on {device or 'auto'}.")
        except Exception as e:
            logging.error(f"Failed to load YOLO model: {e}")
            raise

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Detect objects in a given frame.
        
        Args:
            frame: BGR image as a numpy array.
            
        Returns:
            List of Detection objects filtered by target classes and confidence.
        """
        detections = []
        try:
            # Run inference
            results = self.model(frame, verbose=False)[0]
            
            # Parse results
            for box in results.boxes:
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue
                    
                cls_id = int(box.cls[0])
                # Filter for target classes (e.g., person, car)
                if cls_id in self.TARGET_CLASSES:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append(Detection(
                        bbox=[x1, y1, x2, y2],
                        class_id=cls_id,
                        class_name=self.TARGET_CLASSES[cls_id],
                        confidence=conf
                    ))
        except Exception as e:
            logging.error(f"Error during detection: {e}")
            
        return detections

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        detector = YOLODetector()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        dets = detector.detect(dummy_frame)
        print(f"Detected {len(dets)} objects in dummy frame.")
    except Exception as e:
        print(f"Demo failed: {e}")
