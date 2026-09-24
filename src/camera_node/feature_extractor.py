import logging
from dataclasses import dataclass
from typing import List, Dict
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
except ImportError:
    torch = None
    logging.warning("PyTorch/torchvision not installed. FeatureExtractor will not work.")

from .tracker import Track

@dataclass
class SceneFeature:
    """Dataclass containing feature embeddings for a scene and its objects."""
    object_features: Dict[str, np.ndarray]  # track_id -> 256-dim embedding
    scene_context: np.ndarray               # 256-dim global scene embedding
    timestamp: float
    camera_id: str

class FeatureExtractor:
    """
    Extracts compact semantic feature vectors from video frames and tracked objects
    using a lightweight MobileNetV3 CNN.
    """
    def __init__(self, device: str = None):
        """
        Initializes the MobileNetV3 feature extractor.
        """
        if torch is None:
            raise RuntimeError("PyTorch is required. Install via 'pip install torch torchvision'.")
            
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Use MobileNetV3 Small for fast, lightweight extraction
            weights = MobileNet_V3_Small_Weights.DEFAULT
            base_model = mobilenet_v3_small(weights=weights)
            
            # Remove classification head, replace with a projection layer to 256-dim
            # The features output of mobilenet_v3_small is usually 576-dim before classifier
            self.feature_dim = 256
            self.model = nn.Sequential(
                base_model.features,
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(576, self.feature_dim) # Project to 256 dims
            )
            self.model.to(self.device)
            self.model.eval()
            
            self.transform = weights.transforms()
            logging.info(f"Initialized FeatureExtractor on {self.device}")
        except Exception as e:
            logging.error(f"Failed to initialize FeatureExtractor: {e}")
            raise

    def _extract_tensor(self, img_tensor: 'torch.Tensor') -> np.ndarray:
        """Helper to run the model on a tensor and return numpy array."""
        with torch.no_grad():
            try:
                features = self.model(img_tensor)
                # L2 normalization for cosine similarity compatibility
                features = nn.functional.normalize(features, p=2, dim=1)
                return features.cpu().numpy().squeeze()
            except Exception as e:
                logging.error(f"Extraction failed: {e}")
                return np.zeros(self.feature_dim, dtype=np.float32)

    def extract(self, frame: np.ndarray, tracks: List[Track], timestamp: float, camera_id: str) -> SceneFeature:
        """
        Extract features for the global scene and individual tracked objects.
        
        Args:
            frame: The current video frame (BGR numpy array)
            tracks: List of active tracks
            timestamp: Current timestamp
            camera_id: Identifier for the camera
            
        Returns:
            SceneFeature object
        """
        if torch is None:
            return SceneFeature({}, np.zeros(256), timestamp, camera_id)

        try:
            import cv2 # Ensure cv2 is available for image conversion
            # Convert BGR to RGB for PyTorch vision models
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Global scene feature
            scene_tensor = self.transform(torch.from_numpy(frame_rgb).permute(2, 0, 1)).unsqueeze(0).to(self.device)
            scene_context = self._extract_tensor(scene_tensor)
            
            # Object features
            object_features = {}
            for track in tracks:
                x1, y1, x2, y2 = [int(v) for v in track.bbox]
                # Boundary checks
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                crop = frame_rgb[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                    
                crop_tensor = self.transform(torch.from_numpy(crop).permute(2, 0, 1)).unsqueeze(0).to(self.device)
                obj_feat = self._extract_tensor(crop_tensor)
                object_features[track.track_id] = obj_feat
                
            return SceneFeature(
                object_features=object_features,
                scene_context=scene_context,
                timestamp=timestamp,
                camera_id=camera_id
            )
        except Exception as e:
            logging.error(f"Error in feature extraction: {e}")
            return SceneFeature({}, np.zeros(256, dtype=np.float32), timestamp, camera_id)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        extractor = FeatureExtractor(device='cpu')
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        sf = extractor.extract(dummy_frame, [], 0.0, "cam_1")
        print(f"Scene context shape: {sf.scene_context.shape}")
    except Exception as e:
        print(f"Demo failed: {e}")
