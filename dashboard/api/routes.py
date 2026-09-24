from flask import Blueprint, jsonify, request

api = Blueprint('api', __name__)

@api.route('/status')
def get_status():
    return jsonify({"status": "operational", "version": "1.0.0", "system": "Sentinel AI"})

@api.route('/cameras')
def get_cameras():
    # Return mock cameras since actual real-time state is sent via Socket.IO
    return jsonify([
        {"id": "CAM-01", "location": "Main Entrance", "status": "active", "priority": 0.2},
        {"id": "CAM-02", "location": "Lobby", "status": "active", "priority": 0.1},
        {"id": "CAM-03", "location": "Perimeter North", "status": "active", "priority": 0.5},
        {"id": "CAM-04", "location": "Server Room", "status": "active", "priority": 0.8}
    ])

@api.route('/alerts')
def get_alerts():
    return jsonify([])

@api.route('/incident/<incident_id>')
def get_incident(incident_id):
    return jsonify({
        "id": incident_id, 
        "status": "investigating", 
        "description": "Details for incident " + incident_id,
        "severity": "CRITICAL"
    })

@api.route('/cameras/<camera_id>/priority', methods=['POST'])
def override_priority(camera_id):
    data = request.json or {}
    new_priority = data.get('priority', 0.5)
    # In a real system, this would update the camera's priority in the database/system
    return jsonify({
        "status": "success", 
        "message": f"Camera {camera_id} priority overridden", 
        "new_priority": new_priority
    })
