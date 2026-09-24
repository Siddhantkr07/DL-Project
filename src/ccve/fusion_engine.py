import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
import time

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None
    logging.warning("PyTorch not installed. CognitiveFusionEngine will not work.")

# Absolute import as instructed or relative based on directory structure.
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from camera_node.feature_extractor import SceneFeature
except ImportError:
    # Dummy definition if not found
    @dataclass
    class SceneFeature:
        object_features: Dict[str, 'np.ndarray']
        scene_context: 'np.ndarray'
        timestamp: float
        camera_id: str

@dataclass
class GlobalSceneModel:
    """Dataclass representing the fused understanding of the entire multi-camera environment."""
    timestamp: float
    incident_confidence: float
    incident_type: str
    propagation_path: List[str]
    severity_score: float
    camera_contributions: Dict[str, float]

class MultiCameraTransformer(nn.Module) if nn is not None else object:
    """Lightweight Transformer to fuse features from multiple cameras."""
    def __init__(self, feature_dim: int = 256, num_layers: int = 4, num_heads: int = 4):
        super().__init__()
        if nn is not None:
            self.feature_dim = feature_dim
            # A simple learned token to represent the global 'incident' state (like [CLS] token)
            self.global_token = nn.Parameter(torch.randn(1, 1, feature_dim))
            
            encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            
            # Classification heads
            self.incident_classifier = nn.Linear(feature_dim, 1) # Probability of incident
            self.severity_regressor = nn.Linear(feature_dim, 1)  # Severity score

    def forward(self, camera_features: 'torch.Tensor') -> 'torch.Tensor':
        # camera_features shape: [batch, num_cameras, feature_dim]
        batch_size = camera_features.size(0)
        
        # Prepend global token
        cls_tokens = self.global_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, camera_features), dim=1) # [batch, num_cameras + 1, feature_dim]
        
        # Pass through transformer
        out = self.transformer(x)
        
        # Use the output of the global token
        global_repr = out[:, 0, :]
        
        incident_logit = self.incident_classifier(global_repr)
        severity = self.severity_regressor(global_repr)
        
        return incident_logit, severity

class CognitiveFusionEngine:
    """
    Central Cognitive Vision Engine (CCVE).
    Accepts SceneFeature from multiple cameras and uses a Transformer to fuse multi-camera features.
    """
    def __init__(self, num_cameras: int, feature_dim: int = 256, memory_size: int = 30):
        self.num_cameras = num_cameras
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        
        # Incident memory (recent N frames)
        self.history = []
        
        if torch is not None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.transformer = MultiCameraTransformer(feature_dim=feature_dim)
            self.transformer.to(self.device)
            self.transformer.eval()
        else:
            self.transformer = None
            
        logging.info("Initialized CognitiveFusionEngine.")

    def fuse(self, scene_features: Dict[str, SceneFeature]) -> GlobalSceneModel:
        """
        Fuses multi-camera features into a global model.
        
        Args:
            scene_features: Dictionary mapping camera_id to its latest SceneFeature.
            
        Returns:
            GlobalSceneModel containing system-wide analysis.
        """
        timestamp = time.time()
        
        if self.transformer is None or torch is None:
            return GlobalSceneModel(timestamp, 0.0, "unknown", [], 0.0, {})

        try:
            # Collect scene contexts
            features_list = []
            cam_ids = []
            
            for cid, sf in scene_features.items():
                features_list.append(torch.from_numpy(sf.scene_context).float())
                cam_ids.append(cid)
                
            if not features_list:
                return GlobalSceneModel(timestamp, 0.0, "none", [], 0.0, {})
                
            # Stack features: [1, num_cams, feature_dim]
            features_tensor = torch.stack(features_list).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                incident_logit, severity = self.transformer(features_tensor)
                
                prob = torch.sigmoid(incident_logit).item()
                sev = severity.item()
                
            # Dummy attention/contribution (in a real scenario, we'd extract transformer attention weights)
            contributions = {cid: 1.0/len(cam_ids) for cid in cam_ids}
            
            # Simple threshold for incident detection
            incident_type = "anomaly_detected" if prob > 0.7 else "normal"
            
            model = GlobalSceneModel(
                timestamp=timestamp,
                incident_confidence=prob,
                incident_type=incident_type,
                propagation_path=cam_ids, # Default to all active cams for now
                severity_score=sev,
                camera_contributions=contributions
            )
            
            # Maintain memory
            self.history.append(model)
            if len(self.history) > self.memory_size:
                self.history.pop(0)
                
            return model
            
        except Exception as e:
            logging.error(f"Error in Cognitive Fusion: {e}")
            return GlobalSceneModel(timestamp, 0.0, "error", [], 0.0, {})

if __name__ == "__main__":
    import numpy as np
    logging.basicConfig(level=logging.INFO)
    try:
        engine = CognitiveFusionEngine(num_cameras=2)
        sf1 = SceneFeature({}, np.random.randn(256).astype(np.float32), time.time(), "cam1")
        sf2 = SceneFeature({}, np.random.randn(256).astype(np.float32), time.time(), "cam2")
        global_model = engine.fuse({"cam1": sf1, "cam2": sf2})
        print(f"Fusion result: Confidence={global_model.incident_confidence:.2f}, Type={global_model.incident_type}")
    except Exception as e:
        print(f"Demo failed: {e}")
