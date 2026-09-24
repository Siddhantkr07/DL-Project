import logging
import threading
import time
import cv2
from typing import Callable, Optional

from .detector import YOLODetector
from .tracker import DeepSORTTracker
from .feature_extractor import FeatureExtractor, SceneFeature

class CameraNode:
    """
    Main camera node orchestrator.
    Combines detection, tracking, and feature extraction.
    Runs in its own thread to process video continuously.
    """
    def __init__(self, camera_id: str, video_source: str, callback: Optional[Callable[[SceneFeature], None]] = None):
        """
        Initializes the Camera Node.
        
        Args:
            camera_id: Unique string identifier for this camera.
            video_source: Path to video file, RTSP URL, or integer for webcam.
            callback: Function to call with the SceneFeature output per frame.
        """
        self.camera_id = camera_id
        # Convert integer strings to actual integers for webcams
        self.video_source = int(video_source) if str(video_source).isdigit() else video_source
        self.callback = callback
        
        self.detector = YOLODetector()
        self.tracker = DeepSORTTracker()
        self.feature_extractor = FeatureExtractor()
        
        self.running = False
        self.thread = None
        self.latest_scene_feature: Optional[SceneFeature] = None

    def start(self):
        """Starts the video processing thread."""
        if self.running:
            logging.warning(f"CameraNode {self.camera_id} is already running.")
            return
            
        self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logging.info(f"Started CameraNode {self.camera_id}")

    def stop(self):
        """Stops the video processing thread."""
        self.running = False
        if self.thread:
            self.thread.join()
        logging.info(f"Stopped CameraNode {self.camera_id}")

    def process_frame(self, frame) -> SceneFeature:
        """
        Process a single frame through the entire pipeline.
        """
        timestamp = time.time()
        
        # 1. Detect
        detections = self.detector.detect(frame)
        
        # 2. Track
        tracks = self.tracker.update(detections, frame)
        
        # 3. Extract Features
        scene_feature = self.feature_extractor.extract(frame, tracks, timestamp, self.camera_id)
        
        self.latest_scene_feature = scene_feature
        return scene_feature

    def _run_loop(self):
        """Internal loop for reading frames and processing them."""
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            logging.error(f"Failed to open video source {self.video_source} for camera {self.camera_id}")
            self.running = False
            return
            
        while self.running:
            ret, frame = cap.read()
            if not ret:
                logging.warning(f"Failed to read frame from camera {self.camera_id}. Re-initializing...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            try:
                scene_feature = self.process_frame(frame)
                
                # Emit event via callback if provided
                if self.callback:
                    self.callback(scene_feature)
                    
            except Exception as e:
                logging.error(f"Error processing frame in camera {self.camera_id}: {e}")
                
            # Sleep slightly to avoid hogging CPU if reading too fast
            time.sleep(0.01)
            
        cap.release()

    def get_scene_feature(self) -> Optional[SceneFeature]:
        """Returns the most recently computed scene feature."""
        return self.latest_scene_feature

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    node = CameraNode("test_cam", "0", callback=lambda sf: print(f"Got features with {len(sf.object_features)} objects."))
    try:
        node.start()
        time.sleep(5)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
