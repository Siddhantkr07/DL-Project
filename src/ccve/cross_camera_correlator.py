import logging
from typing import Dict, List
import numpy as np

# Absolute import as instructed
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from camera_node.feature_extractor import SceneFeature
except ImportError:
    pass

class CrossCameraCorrelator:
    """
    Performs cross-camera entity correlation.
    Matches tracked entities across cameras using feature cosine similarity.
    (Re-identification).
    """
    def __init__(self, similarity_threshold: float = 0.75):
        """
        Args:
            similarity_threshold: Cosine similarity threshold to consider two features as the same entity.
        """
        self.similarity_threshold = similarity_threshold
        logging.info(f"Initialized CrossCameraCorrelator with threshold {similarity_threshold}")

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Computes cosine similarity between two vectors."""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(v1, v2) / (norm1 * norm2)

    def correlate(self, scene_features: Dict[str, 'SceneFeature']) -> Dict[str, List[str]]:
        """
        Correlates entities across multiple cameras.
        
        Args:
            scene_features: Dictionary mapping camera_id -> SceneFeature
            
        Returns:
            Dictionary mapping a generic entity ID to a list of camera IDs where it is seen.
        """
        # A simple clustering approach:
        # 1. Collect all objects from all cameras.
        # 2. Iterate and group objects that are highly similar.
        
        all_objects = []
        for cam_id, sf in scene_features.items():
            for track_id, feature in sf.object_features.items():
                all_objects.append({
                    "cam_id": cam_id,
                    "local_track_id": track_id,
                    "feature": feature
                })
                
        # entity_id (int) -> list of cam_ids
        correlations: Dict[str, List[str]] = {}
        # entity_id -> aggregated_feature
        entity_features = {}
        
        entity_counter = 0
        
        for obj in all_objects:
            best_match_id = None
            best_sim = -1.0
            
            # Find best matching existing entity
            for ent_id, ent_feat in entity_features.items():
                sim = self._cosine_similarity(obj["feature"], ent_feat)
                if sim > self.similarity_threshold and sim > best_sim:
                    best_sim = sim
                    best_match_id = ent_id
                    
            if best_match_id is not None:
                # Add to existing entity
                if obj["cam_id"] not in correlations[str(best_match_id)]:
                    correlations[str(best_match_id)].append(obj["cam_id"])
                # Update entity feature (running average)
                entity_features[best_match_id] = 0.5 * entity_features[best_match_id] + 0.5 * obj["feature"]
            else:
                # Create new entity
                ent_id = str(entity_counter)
                entity_counter += 1
                correlations[ent_id] = [obj["cam_id"]]
                entity_features[ent_id] = obj["feature"]
                
        return correlations

if __name__ == "__main__":
    from dataclasses import dataclass
    @dataclass
    class MockSceneFeature:
        object_features: Dict[str, np.ndarray]

    correlator = CrossCameraCorrelator()
    # Create mock features: obj1 in cam1 is very similar to objA in cam2
    f1 = np.array([1.0, 0.0, 0.0])
    f2 = np.array([0.99, 0.1, 0.0])
    f3 = np.array([0.0, 1.0, 0.0])
    
    sf1 = MockSceneFeature({"obj1": f1})
    sf2 = MockSceneFeature({"objA": f2, "objB": f3})
    
    result = correlator.correlate({"cam1": sf1, "cam2": sf2})
    print(f"Correlations: {result}")
