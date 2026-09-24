import os
import cv2
import numpy as np
import random
from typing import List, Generator, Dict
from pathlib import Path

class UCFCrimeLoader:
    """Loads and preprocesses video clips from the UCF-Crime dataset"""
    
    CATEGORIES = [
        "Abuse", "Arson", "Assault", "Burglary", "Explosion", 
        "Fighting", "RoadAccidents", "Robbery", "Shooting", 
        "Shoplifting", "Stealing", "Vandalism", "Normal"
    ]
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        
    def get_video_paths(self, category: str = None) -> List[str]:
        """Returns a list of video paths, optionally filtered by category"""
        if not self.dataset_path.exists():
            print(f"Dataset path not found: {self.dataset_path}")
            return []
            
        video_paths = []
        
        if category and category in self.CATEGORIES:
            cat_dir = self.dataset_path / category
            if cat_dir.exists():
                video_paths.extend([str(p) for p in cat_dir.glob("*.mp4")])
        else:
            for cat in self.CATEGORIES:
                cat_dir = self.dataset_path / cat
                if cat_dir.exists():
                    video_paths.extend([str(p) for p in cat_dir.glob("*.mp4")])
                    
        return video_paths
        
    def get_frame_generator(self, video_path: str, fps: int = 10) -> Generator[np.ndarray, None, None]:
        """Generator that yields frames from a video file at a specific FPS"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
            
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps == 0 or np.isnan(original_fps):
            original_fps = 30.0  # fallback
            
        frame_skip = max(1, int(original_fps / fps))
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_skip == 0:
                yield frame
                
            frame_idx += 1
            
        cap.release()
        
    def create_multi_camera_scenario(self, num_cameras: int = 4) -> Dict[str, str]:
        """Creates a synthetic multi-camera scenario mapping camera_id to video_path"""
        all_videos = self.get_video_paths()
        
        if not all_videos:
            # Fallback to dummy paths if dataset doesn't exist
            return {f"cam_{i+1}": f"dummy_path_{i+1}.mp4" for i in range(num_cameras)}
            
        scenario = {}
        for i in range(num_cameras):
            scenario[f"cam_{i+1}"] = random.choice(all_videos)
            
        return scenario

class VideoStreamSimulator:
    """Simulates multiple camera feeds from video files, synchronizing playback"""
    
    def __init__(self, camera_mapping: Dict[str, str], target_fps: int = 30):
        self.camera_mapping = camera_mapping
        self.target_fps = target_fps
        self.caps = {}
        
    def start(self):
        """Initializes VideoCapture objects for all cameras"""
        for cam_id, path in self.camera_mapping.items():
            if os.path.exists(path):
                self.caps[cam_id] = cv2.VideoCapture(path)
            else:
                print(f"Warning: Path {path} for {cam_id} does not exist.")
                
    def get_synced_frames(self) -> Dict[str, np.ndarray]:
        """Reads one frame from each active camera and adds metadata overlay"""
        frames = {}
        for cam_id, cap in self.caps.items():
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Add camera metadata overlay
                    cv2.putText(frame, f"CAM: {cam_id}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    frames[cam_id] = frame
                else:
                    # Loop video if it ends
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if ret:
                        cv2.putText(frame, f"CAM: {cam_id}", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        frames[cam_id] = frame
        return frames
        
    def stop(self):
        """Releases all VideoCapture objects"""
        for cap in self.caps.values():
            if cap:
                cap.release()
