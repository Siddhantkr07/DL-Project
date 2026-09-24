import cv2
import base64
import numpy as np
import logging

logger = logging.getLogger(__name__)

def draw_detections(frame: np.ndarray, tracks: list) -> np.ndarray:
    """
    Draw bounding boxes, labels, and track IDs on the frame.
    
    Args:
        frame (np.ndarray): The image frame.
        tracks (list): A list of dicts, each with 'bbox', 'label', and 'track_id'.
        
    Returns:
        np.ndarray: The frame with detections drawn.
    """
    try:
        result = frame.copy()
        for track in tracks:
            bbox = track.get('bbox')
            label = track.get('label', 'Unknown')
            track_id = track.get('track_id', -1)
            
            if bbox is not None and len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    result, 
                    f"{label} ID:{track_id}", 
                    (x1, max(0, y1 - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    2
                )
        return result
    except Exception as e:
        logger.error(f"Failed to draw detections: {e}")
        return frame

def draw_priority_overlay(frame: np.ndarray, priority_score: float, camera_id: str) -> np.ndarray:
    """
    Draws a priority overlay on the frame.
    
    Args:
        frame (np.ndarray): The image frame.
        priority_score (float): The current priority score (0-1).
        camera_id (str): The camera identifier.
        
    Returns:
        np.ndarray: Frame with priority overlay.
    """
    try:
        result = frame.copy()
        text = f"Cam: {camera_id} | Prio: {priority_score:.2f}"
        
        # Color coding: Red for high, Orange for medium, Green for low priority
        if priority_score > 0.7:
            color = (0, 0, 255) # BGR Red
        elif priority_score > 0.4:
            color = (0, 165, 255) # BGR Orange
        else:
            color = (0, 255, 0) # BGR Green
            
        cv2.putText(result, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return result
    except Exception as e:
        logger.error(f"Failed to draw priority overlay: {e}")
        return frame

def resize_frame(frame: np.ndarray, target_size: tuple) -> np.ndarray:
    """Resizes a frame to the target size."""
    try:
        return cv2.resize(frame, target_size)
    except Exception as e:
        logger.error(f"Failed to resize frame: {e}")
        return frame

def frame_to_base64(frame: np.ndarray) -> str:
    """Encodes a numpy frame as a base64 string for web streaming."""
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to convert frame to base64: {e}")
        return ""

def extract_roi(frame: np.ndarray, bbox: tuple) -> np.ndarray:
    """
    Extracts a region of interest from the frame given a bounding box.
    """
    try:
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        return frame[y1:y2, x1:x2]
    except Exception as e:
        logger.error(f"Failed to extract ROI: {e}")
        return np.array([])

def calculate_optical_flow(prev_frame: np.ndarray, curr_frame: np.ndarray) -> tuple:
    """
    Calculates dense optical flow between two frames.
    
    Returns:
        tuple: (magnitude, angle) arrays.
    """
    try:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return mag, ang
    except Exception as e:
        logger.error(f"Failed to calculate optical flow: {e}")
        return np.array([]), np.array([])
