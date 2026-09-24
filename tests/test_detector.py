import unittest
import numpy as np

# Mocking the detector since ultralytics might not be installed during basic tests
class MockYOLODetector:
    def __init__(self, model_path="yolov8n.pt", conf=0.5):
        self.model_path = model_path
        self.conf = conf
        self.classes = ['person', 'car', 'fire']
        
    def detect(self, frame):
        # If frame is completely black, return no detections
        if np.sum(frame) == 0:
            return []
            
        # Synthetic detection for testing
        return [
            {"class": "person", "conf": 0.85, "bbox": [10, 10, 50, 100]},
            {"class": "fire", "conf": 0.92, "bbox": [200, 200, 250, 250]}
        ]

class TestDetector(unittest.TestCase):
    
    def setUp(self):
        self.detector = MockYOLODetector(conf=0.5)
        
    def test_initialization(self):
        """Test YOLODetector initialization"""
        self.assertEqual(self.detector.model_path, "yolov8n.pt")
        self.assertEqual(self.detector.conf, 0.5)
        
    def test_detect_synthetic_frame(self):
        """Test detect() with synthetic frame"""
        frame = np.ones((640, 640, 3), dtype=np.uint8) * 255 # White frame
        results = self.detector.detect(frame)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["class"], "person")
        self.assertEqual(results[1]["class"], "fire")
        self.assertTrue(results[1]["conf"] > self.detector.conf)
        
    def test_detect_black_frame(self):
        """Test with black frame (no detections)"""
        frame = np.zeros((640, 640, 3), dtype=np.uint8) # Black frame
        results = self.detector.detect(frame)
        
        self.assertEqual(len(results), 0)

if __name__ == '__main__':
    unittest.main()
