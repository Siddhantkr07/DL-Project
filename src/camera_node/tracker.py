import logging
from dataclasses import dataclass
from typing import List, Any
import numpy as np
from .detector import Detection

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    DeepSort = None
    logging.warning("deep_sort_realtime not installed. DeepSORTTracker will not work.")

@dataclass
class Track:
    """Dataclass representing a tracked object."""
    track_id: str
    bbox: List[float]  # [x1, y1, x2, y2]
    class_name: str
    confidence: float
    state: str  # e.g., 'confirmed', 'tentative'

class DeepSORTTracker:
    """
    DeepSORT multi-object tracker for tracking detected entities across frames.
    """
    def __init__(self, max_age: int = 30, n_init: int = 3):
        """
        Initializes the DeepSORT tracker.
        
        Args:
            max_age: Maximum number of missed misses before a track is deleted.
            n_init: Number of consecutive detections before the track is confirmed.
        """
        if DeepSort is None:
            raise RuntimeError("deep_sort_realtime library is required. Install via 'pip install deep-sort-realtime'.")
            
        self.tracker = DeepSort(max_age=max_age, n_init=n_init, nms_max_overlap=1.0)
        logging.info("Initialized DeepSORTTracker.")

    def update(self, detections: List[Detection], frame: np.ndarray) -> List[Track]:
        """
        Update the tracker with the latest detections.
        
        Args:
            detections: List of Detection objects from the detector.
            frame: The current video frame.
            
        Returns:
            List of active Track objects.
        """
        active_tracks = []
        try:
            # Format detections for DeepSORT: [ [x,y,w,h], confidence, class_name ]
            bbs = []
            for det in detections:
                x1, y1, x2, y2 = det.bbox
                w = x2 - x1
                h = y2 - y1
                bbs.append(([x1, y1, w, h], det.confidence, det.class_name))
                
            # Update tracker
            tracks = self.tracker.update_tracks(bbs, frame=frame)
            
            for track in tracks:
                if not track.is_confirmed():
                    continue
                    
                ltrb = track.to_ltrb() # Left, Top, Right, Bottom
                active_tracks.append(Track(
                    track_id=str(track.track_id),
                    bbox=list(ltrb),
                    class_name=track.det_class if track.det_class else "unknown",
                    confidence=track.det_conf if track.det_conf else 0.0,
                    state="confirmed" if track.is_confirmed() else "tentative"
                ))
        except Exception as e:
            logging.error(f"Error during tracking update: {e}")
            
        return active_tracks

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        tracker = DeepSORTTracker()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        tracks = tracker.update([], dummy_frame)
        print(f"Active tracks: {len(tracks)}")
    except Exception as e:
        print(f"Demo failed: {e}")
