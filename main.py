import argparse
import sys
import time
import logging
import signal
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SentinelAI")

def signal_handler(sig, frame):
    logger.info("Graceful shutdown initiated (Ctrl+C).")
    sys.exit(0)

def print_system_info():
    """Prints system and environment info on startup"""
    logger.info("=========================================")
    logger.info("   Sentinel AI - System Initialization   ")
    logger.info("=========================================")
    logger.info(f"Python Version: {sys.version.split()[0]}")
    
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"PyTorch Version: {torch.__version__}")
        logger.info(f"Compute Device: {device.upper()}")
    except ImportError:
        logger.info("PyTorch not installed. Running in limited mode.")
        
    try:
        import cv2
        logger.info(f"OpenCV Version: {cv2.__version__}")
    except ImportError:
        pass
        
    logger.info("=========================================")

def run_simulation(num_cameras, launch_dashboard, config_path):
    """Runs the system in simulation mode"""
    logger.info(f"Starting simulation mode with {num_cameras} cameras.")
    
    from simulation.camera_simulator import MultiCameraSimulator
    from simulation.scenario_generator import ScenarioGenerator
    
    sim = MultiCameraSimulator(num_cameras=num_cameras)
    generator = ScenarioGenerator(sim)
    
    logger.info("Starting camera threads...")
    sim.start()
    
    logger.info("Setting up 'Fire Incident' scenario...")
    generator.scenario_fire()
    
    if launch_dashboard:
        logger.info("Dashboard would launch here (mocked). Go to http://localhost:5000")
        
    try:
        logger.info("Simulation running. Press Ctrl+C to stop.")
        while True:
            frames = sim.get_frames()
            if frames:
                logger.debug(f"Received frames from {len(frames)} cameras.")
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Stopping simulation...")
        sim.stop()

def run_live(video_path, config_path):
    """Runs the system on a live video or specific video file"""
    logger.info(f"Starting live mode on source: {video_path}")
    logger.info("Live mode pipeline initialization... (Placeholder)")
    # Real pipeline initialization goes here

def run_demo(num_cameras, launch_dashboard, config_path):
    """Runs a full demonstration of the pipeline"""
    logger.info("Starting full Sentinel AI Demo.")
    run_simulation(num_cameras, launch_dashboard, config_path)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser(description="Sentinel AI - Cognitive Multi-Camera Vision System")
    parser.add_argument("--mode", type=str, choices=["simulation", "live", "demo"], default="demo", help="Operating mode")
    parser.add_argument("--cameras", type=int, default=4, help="Number of cameras (simulation/demo mode)")
    parser.add_argument("--video", type=str, default="", help="Path to video file (live mode)")
    parser.add_argument("--dashboard", type=bool, default=True, help="Launch web dashboard")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    
    args = parser.parse_args()
    
    print_system_info()
    
    if args.mode == "simulation":
        run_simulation(args.cameras, args.dashboard, args.config)
    elif args.mode == "live":
        if not args.video:
            logger.error("Live mode requires a --video source.")
            sys.exit(1)
        run_live(args.video, args.config)
    elif args.mode == "demo":
        run_demo(args.cameras, args.dashboard, args.config)
