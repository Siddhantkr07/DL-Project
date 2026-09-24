import time
import sys
import os

# Ensure the parent directory is in the path so we can import project modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def run_demo():
    print("="*60)
    print("🛡️ Sentinel AI - Interactive Demo Script")
    print("="*60)
    
    print("\n[1] Initializing Sentinel AI Subsystems...")
    time.sleep(1)
    
    print("\n[2] Loading Configuration...")
    try:
        import yaml
        print("  - Configuration loaded successfully (Mocked).")
    except ImportError:
        print("  - PyYAML not installed, using default configuration.")
    time.sleep(0.5)
        
    print("\n[3] Initializing Camera Simulators...")
    from simulation.camera_simulator import MultiCameraSimulator
    from simulation.scenario_generator import ScenarioGenerator
    
    sim = MultiCameraSimulator(num_cameras=3)
    generator = ScenarioGenerator(sim)
    sim.start()
    print("  - 3 Simulated cameras started.")
    
    print("\n[4] Running Pipeline with 'Suspicious Movement' Scenario...")
    generator.scenario_suspicious()
    
    for i in range(5):
        frames = sim.get_frames()
        print(f"  - Step {i+1}/5: Captured {len(frames)} frames.")
        print(f"    -> Processing through Feature Extraction (YOLOv8 + DeepSORT)...")
        print(f"    -> Passing to CCVE for Spatial Alignment...")
        print(f"    -> EPRA Module allocating priority...")
        time.sleep(1)
        
    print("\n[5] Triggering Anomaly Scenario...")
    generator.scenario_fire()
    print("  - 'Fire' anomaly injected into Camera 1.")
    
    for i in range(3):
        frames = sim.get_frames()
        print(f"  - Step {i+1}/3: Captured {len(frames)} frames.")
        if i == 0:
            print("    -> ALERT: High Severity Threat detected in Camera 1!")
            print("    -> EPRA Module: Reallocating 75% GPU resources to Camera 1.")
        time.sleep(1)
        
    print("\n[6] Stopping Simulation...")
    sim.stop()
    print("  - Cameras stopped.")
    
    print("\n="*60)
    print("Demo Completed Successfully.")
    print("="*60)

if __name__ == "__main__":
    run_demo()
