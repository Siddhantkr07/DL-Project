import numpy as np
from typing import Dict, Any

class ScenarioGenerator:
    """Generates test scenarios for the Sentinel AI system evaluation"""
    
    def __init__(self, simulator):
        self.simulator = simulator
        
    def scenario_normal(self) -> Dict[str, Any]:
        """Sets up a normal pedestrian traffic scenario"""
        # Clear any anomalies
        for cam_id in self.simulator.synthetic_state.keys():
            self.simulator.synthetic_state[cam_id]["anomaly"] = None
            
        return {
            "name": "Normal Traffic",
            "description": "Standard monitoring with no critical events.",
            "expected_priority": "low"
        }
        
    def scenario_fire(self) -> Dict[str, Any]:
        """Sets up a fire incident scenario in camera 1"""
        self.scenario_normal() # Reset
        
        # Trigger fire in cam_1
        self.simulator.trigger_anomaly("cam_1", "fire")
        
        # Accelerate people in other cameras (simulating fleeing)
        for cam_id in ["cam_2", "cam_3", "cam_4"]:
            if cam_id in self.simulator.synthetic_state:
                for obj in self.simulator.synthetic_state[cam_id]["objects"]:
                    obj["dx"] *= 2.0
                    obj["dy"] *= 2.0
                    
        return {
            "name": "Fire Incident",
            "description": "Fire in Camera 1, people fleeing in other cameras.",
            "expected_priority": "high",
            "critical_camera": "cam_1"
        }
        
    def scenario_crowd_surge(self) -> Dict[str, Any]:
        """Sets up a progressive crowd density increase scenario"""
        self.scenario_normal() # Reset
        
        # Add many objects to cam_2
        if "cam_2" in self.simulator.synthetic_state:
            for _ in range(20): # Add 20 more people
                self.simulator.synthetic_state["cam_2"]["objects"].append({
                    "x": np.random.randint(100, 700), 
                    "y": np.random.randint(100, 500), 
                    "dx": np.random.uniform(-2, 2), 
                    "dy": np.random.uniform(-2, 2),
                    "color": (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255))
                })
                
        return {
            "name": "Crowd Surge",
            "description": "High density of objects in Camera 2.",
            "expected_priority": "medium",
            "critical_camera": "cam_2"
        }
        
    def scenario_suspicious(self) -> Dict[str, Any]:
        """Sets up a single suspicious individual moving across cameras"""
        self.scenario_normal()
        # In a real implementation, this would orchestrate a specific object moving
        # from the FoV of cam_1 to cam_2 etc.
        return {
            "name": "Suspicious Movement",
            "description": "Tracking a specific entity across multiple cameras.",
            "expected_priority": "medium"
        }
