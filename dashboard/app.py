from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
from api.routes import api
import threading
import time
import random
import uuid

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sentinel-ai-secret!'
app.register_blueprint(api, url_prefix='/api')
socketio = SocketIO(app, cors_allowed_origins="*")

# In-memory data for simulation
cameras = [
    {"id": "CAM-01", "location": "Main Entrance", "status": "active", "priority": 0.2, "fps": 30.0, "detections": 0},
    {"id": "CAM-02", "location": "Lobby", "status": "active", "priority": 0.1, "fps": 29.5, "detections": 0},
    {"id": "CAM-03", "location": "Perimeter North", "status": "active", "priority": 0.5, "fps": 28.0, "detections": 0},
    {"id": "CAM-04", "location": "Server Room", "status": "active", "priority": 0.8, "fps": 30.0, "detections": 0}
]
alerts = []
stats = {"active_cameras": 4, "incident_count": 0, "avg_fps": 29.3}

def simulate_system():
    """Background thread to simulate the Sentinel AI system."""
    while True:
        time.sleep(1)
        # Update camera states
        total_fps = 0
        for cam in cameras:
            cam['fps'] = round(random.uniform(25.0, 30.0), 1)
            # Randomly fluctuate priority based on mock scene activity
            cam['priority'] += random.uniform(-0.05, 0.05)
            cam['priority'] = max(0.0, min(1.0, cam['priority']))
            total_fps += cam['fps']
        
        stats['avg_fps'] = round(total_fps / len(cameras), 1)
        socketio.emit('camera_update', {'cameras': cameras, 'stats': stats})
        
        # Randomly generate alerts
        if random.random() < 0.15: # 15% chance every second to generate an alert
            severity = random.choices(['INFO', 'WARNING', 'CRITICAL'], weights=[0.6, 0.3, 0.1])[0]
            alert_types = {
                'INFO': ['Person detected', 'Vehicle detected', 'Routine patrol completed'],
                'WARNING': ['Loitering detected', 'Unrecognized vehicle', 'Access denied'],
                'CRITICAL': ['Intrusion detected', 'Weapon detected', 'Fire detected']
            }
            cam = random.choice(cameras)
            alert = {
                'id': str(uuid.uuid4()),
                'timestamp': time.time() * 1000,
                'type': random.choice(alert_types[severity]),
                'severity': severity,
                'camera_ids': [cam['id']],
                'description': f"{severity} event logged at {cam['location']}"
            }
            alerts.insert(0, alert)
            if len(alerts) > 50:
                alerts.pop()
            
            if severity in ['WARNING', 'CRITICAL']:
                stats['incident_count'] += 1
                cam['detections'] += 1
            
            socketio.emit('alert', alert)
            
        # Scene update
        if random.random() < 0.2:
            socketio.emit('scene_update', {'timestamp': time.time(), 'global_entities': random.randint(1, 15)})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    simulator_thread = threading.Thread(target=simulate_system, daemon=True)
    simulator_thread.start()
    socketio.run(app, debug=True, port=5000, use_reloader=False)
