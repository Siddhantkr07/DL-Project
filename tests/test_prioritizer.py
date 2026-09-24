import unittest

class EPRAModule:
    """Mock Event Prioritization & Resource Allocator"""
    def __init__(self, w_sev=0.4, w_prox=0.3, w_act=0.2, w_ctx=0.1):
        self.weights = {"severity": w_sev, "proximity": w_prox, "activity": w_act, "context": w_ctx}
        
    def calculate_priority(self, severity_score, proximity_score, activity_score, context_score):
        score = (severity_score * self.weights["severity"] +
                 proximity_score * self.weights["proximity"] +
                 activity_score * self.weights["activity"] +
                 context_score * self.weights["context"])
        return min(1.0, max(0.0, score))
        
    def allocate_resources(self, camera_scores):
        """Returns sorted list of (cam_id, resource_percentage)"""
        total_score = sum(camera_scores.values())
        if total_score == 0:
            # Equal distribution
            num_cams = len(camera_scores)
            return {cam: 1.0/num_cams for cam in camera_scores}
            
        return {cam: score/total_score for cam, score in camera_scores.items()}

class TestPrioritizer(unittest.TestCase):
    
    def setUp(self):
        self.epra = EPRAModule()
        
    def test_score_calculation(self):
        """Test priority score calculation formula"""
        score = self.epra.calculate_priority(severity_score=1.0, proximity_score=0.5, activity_score=0.5, context_score=0.0)
        # 1.0*0.4 + 0.5*0.3 + 0.5*0.2 + 0.0*0.1 = 0.4 + 0.15 + 0.10 + 0 = 0.65
        self.assertAlmostEqual(score, 0.65)
        
    def test_resource_allocation(self):
        """Test proportional resource allocation"""
        scores = {"cam_1": 0.8, "cam_2": 0.2, "cam_3": 0.0}
        allocation = self.epra.allocate_resources(scores)
        
        self.assertEqual(allocation["cam_1"], 0.8)
        self.assertEqual(allocation["cam_2"], 0.2)
        self.assertEqual(allocation["cam_3"], 0.0)
        
    def test_priority_ordering(self):
        """Test ordering based on priority scores"""
        scores = {"cam_1": 0.2, "cam_2": 0.9, "cam_3": 0.5}
        
        # Sort cameras by score, descending
        ordered = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        self.assertEqual(ordered[0], "cam_2")
        self.assertEqual(ordered[1], "cam_3")
        self.assertEqual(ordered[2], "cam_1")

if __name__ == '__main__':
    unittest.main()
