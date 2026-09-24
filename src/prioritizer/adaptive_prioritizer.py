import logging
from dataclasses import dataclass
from typing import Dict, List

logger = logging.getLogger(__name__)

@dataclass
class CameraScore:
    """Dataclass holding the scoring and allocation for a camera."""
    camera_id: str
    priority_score: float
    allocated_bandwidth: float
    processing_fps: float


class AdaptivePrioritizer:
    """
    Adaptive Camera Prioritization logic.
    Scores each camera feed based on: severity, proximity, contextual relevance, and historical activity.
    """

    def __init__(self, num_cameras: int):
        """
        Initialize the prioritizer.
        
        Args:
            num_cameras (int): The total number of cameras in the network.
        """
        self.num_cameras = num_cameras
        self.camera_scores: Dict[str, CameraScore] = {}
        self.historical_activity: Dict[str, float] = {}
        self.smoothing_alpha = 0.3

    def update_scores(self, camera_data: Dict[str, dict]) -> Dict[str, float]:
        """
        Update the priority scores for each camera based on incoming data.
        
        Args:
            camera_data: Dictionary mapping camera_id to its metrics (severity, proximity, active_tracks, anomaly_score).
            
        Returns:
            Dict[str, float]: Mapping of camera_id to its new priority score.
        """
        scores = {}
        for cam_id, data in camera_data.items():
            try:
                severity = data.get('severity', 0.0)
                proximity = data.get('proximity', 0.0)
                
                active_tracks = data.get('active_tracks', 0)
                anomaly_score = data.get('anomaly_score', 0.0)
                
                # Contextual relevance: bounded active tracks contribution + anomaly
                context = min(1.0, (active_tracks / 20.0)) * 0.5 + anomaly_score * 0.5

                # Historical activity EMA
                current_activity = context
                hist_act = self.historical_activity.get(cam_id, current_activity)
                activity = self.smoothing_alpha * current_activity + (1 - self.smoothing_alpha) * hist_act
                self.historical_activity[cam_id] = activity
                
                # Priority Score Formula
                priority = 0.4 * severity + 0.3 * proximity + 0.2 * activity + 0.1 * context
                
                # Prevent rapid thrashing via smoothing with previous priority score
                if cam_id in self.camera_scores:
                    prev_priority = self.camera_scores[cam_id].priority_score
                    priority = self.smoothing_alpha * priority + (1 - self.smoothing_alpha) * prev_priority
                    
                scores[cam_id] = priority
                
                if cam_id not in self.camera_scores:
                    self.camera_scores[cam_id] = CameraScore(
                        camera_id=cam_id,
                        priority_score=priority,
                        allocated_bandwidth=0.0,
                        processing_fps=0.0
                    )
                else:
                    self.camera_scores[cam_id].priority_score = priority

            except Exception as e:
                logger.error(f"Failed to update score for camera {cam_id}: {e}")
                
        return scores

    def get_priority_order(self) -> List[str]:
        """
        Returns camera IDs sorted by priority score, highest first.
        """
        sorted_cams = sorted(self.camera_scores.values(), key=lambda x: x.priority_score, reverse=True)
        return [cam.camera_id for cam in sorted_cams]

    def allocate_resources(self, total_budget: float) -> Dict[str, float]:
        """
        Allocates processing resources (e.g., FPS) among cameras based on priority.
        
        Args:
            total_budget (float): Total available budget (e.g., total FPS capacity).
            
        Returns:
            Dict[str, float]: Mapping of camera_id to allocated FPS/budget.
        """
        total_priority = sum(cam.priority_score for cam in self.camera_scores.values())
        allocations = {}
        
        if total_priority == 0:
            # If no activity, distribute equally
            budget_per_cam = total_budget / len(self.camera_scores) if self.camera_scores else 0
            for cam_id, cam in self.camera_scores.items():
                cam.allocated_bandwidth = budget_per_cam
                cam.processing_fps = budget_per_cam
                allocations[cam_id] = budget_per_cam
            return allocations

        for cam_id, cam in self.camera_scores.items():
            allocation = (cam.priority_score / total_priority) * total_budget
            cam.allocated_bandwidth = allocation
            cam.processing_fps = min(30.0, allocation)  # Example cap at 30 FPS
            allocations[cam_id] = allocation
        
        return allocations
