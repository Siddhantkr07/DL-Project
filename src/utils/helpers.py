import uuid
import json
import logging
import math
import numpy as np
from datetime import datetime

def get_timestamp() -> str:
    """Returns the current UTC timestamp as an ISO formatted string."""
    return datetime.utcnow().isoformat()

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Calculates cosine similarity between two vectors."""
    try:
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))
    except Exception as e:
        logging.error(f"Failed to calculate cosine similarity: {e}")
        return 0.0

def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalizes a vector to unit length."""
    try:
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm
    except Exception as e:
        logging.error(f"Failed to normalize vector: {e}")
        return v

def exponential_moving_average(current: float, previous: float, alpha: float = 0.3) -> float:
    """Calculates the exponential moving average."""
    return alpha * current + (1 - alpha) * previous

def generate_incident_id() -> str:
    """Generates a unique UUID string for an incident."""
    return str(uuid.uuid4())

def load_config(config_path: str) -> dict:
    """Loads a JSON configuration file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Failed to load config at {config_path}: {e}")
        return {}

def setup_logging(level: str = 'INFO') -> logging.Logger:
    """Sets up and configures the main logger."""
    logger = logging.getLogger('SentinelAI')
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not logger.handlers:
        ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger

def euclidean_distance(p1: tuple, p2: tuple) -> float:
    """Calculates the Euclidean distance between two points."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def iou(bbox1: tuple, bbox2: tuple) -> float:
    """
    Calculates the Intersection over Union (IoU) of two bounding boxes.
    Bboxes should be in (x1, y1, x2, y2) format.
    """
    try:
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0
    except Exception as e:
        logging.error(f"Failed to calculate IoU: {e}")
        return 0.0
