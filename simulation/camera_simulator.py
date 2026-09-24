import cv2
import numpy as np
import threading
import time
from typing import Dict, List, Optional
import math

class MultiCameraSimulator:
    """Simulates N camera feeds using video files or generated synthetic frames"""
    
    def __init__(self, num_cameras: int = 4, video_sources: Optional[Dict[str, str]] = None):
        self.num_cameras = num_cameras
        self.video_sources = video_sources or {}
        self.cameras = {}  # cam_id -> VideoCapture or synthetic state
        self.current_frames = {}
        self.running = False
        self.lock = threading.Lock()
        self.threads = []
        
        # For synthetic frames
        self.synthetic_state = {f"cam_{i+1}": self._init_synthetic_state() for i in range(num_cameras)}

    def _init_synthetic_state(self):
        """Initialize state for synthetic camera feed generation"""
        return {
            "objects": [{"x": np.random.randint(50, 750), 
                         "y": np.random.randint(50, 550), 
                         "dx": np.random.uniform(-5, 5), 
                         "dy": np.random.uniform(-5, 5),
                         "color": (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))}
                        for _ in range(5)],
            "anomaly": None  # 'fire', 'fight', etc.
        }

    def start(self):
        """Starts all camera simulation threads"""
        self.running = True
        for i in range(self.num_cameras):
            cam_id = f"cam_{i+1}"
            thread = threading.Thread(target=self._camera_loop, args=(cam_id,))
            thread.daemon = True
            self.threads.append(thread)
            thread.start()
            
    def _camera_loop(self, cam_id: str):
        """Inner loop for each camera thread to generate or read frames"""
        if cam_id in self.video_sources and self.video_sources[cam_id]:
            cap = cv2.VideoCapture(self.video_sources[cam_id])
            use_video = cap.isOpened()
        else:
            use_video = False
            
        while self.running:
            start_time = time.time()
            
            if use_video:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
            else:
                frame = self._generate_synthetic_frame(cam_id)
                
            if frame is not None:
                # Add timestamp and cam_id
                cv2.putText(frame, f"{cam_id} - {time.strftime('%H:%M:%S')}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                with self.lock:
                    self.current_frames[cam_id] = frame
                    
            # Try to maintain ~30fps
            elapsed = time.time() - start_time
            time.sleep(max(0.01, 0.033 - elapsed))
            
        if use_video:
            cap.release()

    def _generate_synthetic_frame(self, cam_id: str) -> np.ndarray:
        """Generates a synthetic frame with moving shapes (simulating people/objects)"""
        frame = np.zeros((600, 800, 3), dtype=np.uint8)
        
        # Add a subtle grid/background
        frame[::50, :] = (40, 40, 40)
        frame[:, ::50] = (40, 40, 40)
        
        state = self.synthetic_state[cam_id]
        
        # Update and draw objects
        for obj in state["objects"]:
            obj["x"] += obj["dx"]
            obj["y"] += obj["dy"]
            
            # Bounce off walls
            if obj["x"] < 20 or obj["x"] > 780: obj["dx"] *= -1
            if obj["y"] < 20 or obj["y"] > 580: obj["dy"] *= -1
            
            cv2.circle(frame, (int(obj["x"]), int(obj["y"])), 15, obj["color"], -1)
            
        # Draw anomaly if present
        if state["anomaly"] == 'fire':
            cx, cy = 400, 300
            radius = np.random.randint(40, 60)
            cv2.circle(frame, (cx, cy), radius, (0, 0, 255), -1)  # Red core
            cv2.circle(frame, (cx, cy), radius + 20, (0, 165, 255), 10)  # Orange flame
            cv2.putText(frame, "FIRE DETECTED", (300, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        return frame

    def get_frames(self) -> Dict[str, np.ndarray]:
        """Returns the latest frames from all cameras"""
        with self.lock:
            return self.current_frames.copy()
            
    def stop(self):
        """Stops all threads"""
        self.running = False
        for thread in self.threads:
            thread.join(timeout=1.0)
            
    def trigger_anomaly(self, cam_id: str, anomaly_type: str):
        """Triggers a synthetic anomaly in a specific camera"""
        if cam_id in self.synthetic_state:
            self.synthetic_state[cam_id]["anomaly"] = anomaly_type
