"""
Sentinel AI – Dashboard Backend (Multi-Camera + YOLOv8s)
Real webcam + IP Camera support + Threat Intelligence Engine
"""

from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO
from api.routes import api
import threading, time, random, uuid, logging, cv2, numpy as np
from collections import deque, defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("SentinelAI")

# ── YOLO ──────────────────────────────────────────────────────────────────────
YOLO_AVAILABLE = False
_yolo_model = None
try:
    from ultralytics import YOLO
    # TENSORRT ACTIVATED! Maximum performance on RTX 4050
    _yolo_model = YOLO("yolov8m.engine", task="detect")
    YOLO_AVAILABLE = True
    log.info("YOLOv8m TensorRT Engine loaded ✅")
except Exception as e:
    log.warning(f"YOLOv8 not available: {e}")

# ── Threat Intelligence Engine ─────────────────────────────────────────────────
class ThreatEngine:
    WEAPON_CLASSES   = {"knife", "gun", "pistol", "rifle", "scissors"}
    HAZARD_CLASSES   = {"fire", "smoke"}
    VEHICLE_CLASSES  = {"car", "truck", "motorcycle", "bus"}
    PERSON_CLASS     = "person"

    LEVEL_RANK = {"SAFE": 0, "INFO": 1, "WARNING": 2, "CRITICAL": 3}
    LEVEL_COLOR = {
        "SAFE":     "#34d399",
        "INFO":     "#38bdf8",
        "WARNING":  "#fbbf24",
        "CRITICAL": "#f43f5e",
    }

    def _proximity_ratio(self, x1, y1, x2, y2, W, H) -> float:
        return ((x2 - x1) * (y2 - y1)) / max(W * H, 1)

    def analyse(self, results, frame_shape) -> dict:
        H, W = frame_shape[:2]
        now = time.time()
        persons, weapons, hazards, vehicles, other = [], [], [], [], []

        if YOLO_AVAILABLE and results and len(results[0].boxes):
            for box in results[0].boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = results[0].names[cls_id].lower()

                # STRICTER CONFIDENCE: Weapons need > 55% to avoid toothbrush=knife
                if cls_name in self.WEAPON_CLASSES and conf < 0.55:
                    continue
                # Base confidence for others: 45% (better range than before)
                elif conf < 0.45:
                    continue

                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                prox = self._proximity_ratio(x1,y1,x2,y2, W, H)
                obj = dict(cls=cls_name, conf=round(conf,2), prox=round(prox,3), bbox=[x1,y1,x2,y2])
                
                track_id = int(box.id[0]) if box.id is not None else None
                obj["track_id"] = track_id

                if cls_name == self.PERSON_CLASS:
                    # Fall Detection — only trigger if bbox is VERY wide vs tall (clearly lying down)
                    # Ratio 1.8 prevents false positives for sitting/leaning people
                    w, h = x2 - x1, y2 - y1
                    if w > h * 1.8:
                        obj["fallen"] = True
                    persons.append(obj)
                elif cls_name in self.WEAPON_CLASSES:  weapons.append(obj)
                elif cls_name in self.HAZARD_CLASSES:  hazards.append(obj)
                elif cls_name in self.VEHICLE_CLASSES: vehicles.append(obj)
                else:                                  other.append(obj)

        level, reasons, tags = "SAFE", [], []
        
        fallen_count = sum(1 for p in persons if p.get("fallen"))
        if fallen_count > 0:
            level = "CRITICAL"
            reasons.append(f"FALL DETECTED: {fallen_count} person(s) down")
            tags.append("fall")

        if weapons:
            level = "CRITICAL"
            reasons.append(f"WEAPON DETECTED: {weapons[0]['cls'].upper()}")
            tags.append("weapon")
        if hazards:
            level = "CRITICAL"
            reasons.append(f"HAZARD: {hazards[0]['cls'].upper()}")
            tags.append("hazard")

        n = len(persons)
        if n >= 10:
            level = "CRITICAL"
            reasons.append(f"CROWD SURGE — {n} persons")
            tags.append("crowd")
        elif n >= 5:
            if self.LEVEL_RANK[level] < self.LEVEL_RANK["WARNING"]: level = "WARNING"
            reasons.append(f"HIGH OCCUPANCY — {n} persons")
            tags.append("crowd")

        # Proximity Breach — only trigger if subject fills >55% of frame (very close/intruding)
        close = [p for p in persons if p["prox"] > 0.55]
        if close:
            if self.LEVEL_RANK[level] < self.LEVEL_RANK["WARNING"]: level = "WARNING"
            reasons.append("PROXIMITY BREACH — subject too close")
            tags.append("proximity")

        if not reasons:
            if persons:
                level = "INFO"
                reasons.append(f"{n} person{'s' if n>1 else ''} monitored")
                tags.append("normal")
            elif vehicles:
                level = "INFO"
                reasons.append(f"{len(vehicles)} vehicle(s) detected")
                tags.append("vehicle")
            else:
                level = "SAFE"
                reasons.append("Area clear")

        assessment = {
            "level": level, "color": self.LEVEL_COLOR[level], "reasons": reasons, "tags": tags,
            "counts": {"persons": len(persons), "weapons": len(weapons), "hazards": len(hazards), "vehicles": len(vehicles), "other": len(other)},
            "objects": persons + weapons + hazards + vehicles + other,
            "ts": now,
        }
        return assessment

_threat_engine = ThreatEngine()

# ── Multi-Camera Manager ───────────────────────────────────────────────────────
CLS_COLORS = {"person": (0, 230, 120), "car": (0, 190, 255), "truck": (0, 100, 255), "knife": (0, 0, 255), "fire": (0, 80, 255), "default": (180, 180, 0)}

def _draw(frame, results, threat, cam_id, fps):
    h, w = frame.shape[:2]
    if YOLO_AVAILABLE and results and len(results[0].boxes):
        for box in results[0].boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = results[0].names[cls_id].lower()
            if cls_name in ThreatEngine.WEAPON_CLASSES and conf < 0.55: continue
            elif conf < 0.45: continue
            
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            color = CLS_COLORS.get(cls_name, CLS_COLORS["default"])
            
            # Draw tracking ID if available
            track_label = ""
            if box.id is not None:
                track_id = int(box.id[0])
                track_label = f" #{track_id}"
                
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            lbl = f"{cls_name}{track_label} {conf:.0%}"
            (tw,th),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.rectangle(frame, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
            cv2.putText(frame, lbl, (x1+2, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,0,0), 1)

    cv2.rectangle(frame, (0,0), (w,46), (8,12,24), -1)
    cv2.putText(frame, "SENTINEL AI", (12,30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,212,255), 2)
    cv2.putText(frame, f"{cam_id} | LIVE", (w//2-70,30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200,200,200), 1)
    cv2.putText(frame, f"FPS:{fps:.0f}", (w-110,30), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0,220,128), 2)

    tlvl = threat.get("level","SAFE")
    tc_bgr = {"SAFE":(52,211,153),"INFO":(56,189,248),"WARNING":(251,191,36),"CRITICAL":(244,63,94)}.get(tlvl,(52,211,153))
    cv2.rectangle(frame,(0,h-40),(200,h),(8,12,24),-1)
    cv2.putText(frame,f"THREAT: {tlvl}",(8,h-12), cv2.FONT_HERSHEY_SIMPLEX,0.60,tc_bgr,2)
    cv2.rectangle(frame,(200,h-40),(330,h),(8,12,24),-1)
    cv2.putText(frame,f"OBJ: {len(threat.get('objects',[]))}",(208,h-12), cv2.FONT_HERSHEY_SIMPLEX,0.60,(255,220,0),2)
    return frame

class CameraStream:
    def __init__(self, cam_id, source, location):
        self.cam_id = cam_id
        self.source = source
        self.location = location
        self.running = False
        self.latest_frame = None
        self.latest_threat = {"level":"SAFE", "reasons":["Connecting..."], "counts":{}, "objects":[]}
        self.fps = 0.0
        self.lock = threading.Lock()
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _run(self):
        src = int(self.source) if str(self.source).isdigit() else self.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            log.error(f"Cannot open camera {self.cam_id} at {self.source}")
            with self.lock: self.latest_threat["reasons"] = ["Camera Offline"]
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        prev = time.time()

        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            now = time.time()
            fps = round(1.0/max(now-prev, 1e-9), 1)
            prev = now

            results = None
            if YOLO_AVAILABLE:
                try: 
                    # Enable ByteTrack for Object Tracking across frames
                    results = _yolo_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
                except: pass

            threat = _threat_engine.analyse(results or [], frame.shape)
            annotated = _draw(frame.copy(), results, threat, self.cam_id, fps)
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
            
            with self.lock:
                self.fps = fps
                self.latest_threat = threat
                self.latest_frame = buf.tobytes()

        cap.release()

cam_streams = {} # cam_id -> CameraStream
cameras_info = [
    {"id":"CAM-01","location":"Laptop Webcam","status":"active", "source": 0, "priority":0.9,"fps":0.0,"detections":0,"live":True,"threat":"SAFE"},
    {"id":"CAM-02","location":"North Corridor","status":"inactive", "source": "", "priority":0.2,"fps":0.0,"detections":0,"live":False,"threat":"SAFE"},
    {"id":"CAM-03","location":"Server Room","status":"inactive", "source": "", "priority":0.3,"fps":0.0,"detections":0,"live":False,"threat":"SAFE"},
    {"id":"CAM-04","location":"Parking Lot","status":"inactive", "source": "", "priority":0.15,"fps":0.0,"detections":0,"live":False,"threat":"SAFE"},
]

def get_cam_stream(cam_id):
    if cam_id not in cam_streams: return None
    return cam_streams[cam_id]

def start_camera(cam_id, source, location):
    if cam_id in cam_streams:
        cam_streams[cam_id].stop()
    stream = CameraStream(cam_id, source, location)
    cam_streams[cam_id] = stream
    stream.start()
    for c in cameras_info:
        if c["id"] == cam_id:
            c["status"] = "active"
            c["live"] = True
            c["source"] = source
            c["location"] = location

# Start default webcam
start_camera("CAM-01", 0, "Laptop Webcam")

def get_blank_frame(cam_id, message="Connecting..."):
    blank = np.zeros((720, 1280, 3), dtype=np.uint8)
    blank[:] = (20, 24, 34) # dark background
    cv2.putText(blank, message, (450, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 255), 2)
    cv2.putText(blank, cam_id, (450, 410), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
    _, buf = cv2.imencode('.jpg', blank)
    return buf.tobytes()

def _generate_mjpeg(cam_id):
    while True:
        stream = get_cam_stream(cam_id)
        f = None
        if stream:
            with stream.lock: f = stream.latest_frame
        
        if f is None:
            f = get_blank_frame(cam_id, "Connecting or Offline...")
            
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
        time.sleep(0.1)

# ── Shared state for emitter ──────────────────────────────────────────────────
alerts         = []
stats          = {"active_cameras":1,"incident_count":0,"avg_fps":0.0,"threat_level":"SAFE"}
detection_log  = defaultdict(int)
timeline_data  = deque(maxlen=30)

def _emitter_thread():
    prev_global_level = "SAFE"
    while True:
        time.sleep(1)
        
        global_level = "SAFE"
        total_fps = 0.0
        active_count = 0
        total_counts = defaultdict(int)
        agg_reasons = []

        for cam_id, stream in cam_streams.items():
            with stream.lock:
                threat = dict(stream.latest_threat)
                fps = stream.fps

            level = threat.get("level","SAFE")
            nobj = len(threat.get("objects",[]))

            # Update camera info
            for c in cameras_info:
                if c["id"] == cam_id:
                    c["fps"] = fps
                    c["detections"] = nobj
                    c["threat"] = level
            
            if ThreatEngine.LEVEL_RANK.get(level,0) > ThreatEngine.LEVEL_RANK.get(global_level,0):
                global_level = level
            
            total_fps += fps
            active_count += 1
            for k,v in threat.get("counts",{}).items(): total_counts[k] += v
            if level != "SAFE" and threat.get("reasons"):
                agg_reasons.append(f"{cam_id}: {threat['reasons'][0]}")

            for obj in threat.get("objects",[]):
                detection_log[obj["cls"]] += 1

        avg_fps = total_fps / max(1, active_count)
        stats["avg_fps"] = avg_fps
        stats["threat_level"] = global_level
        stats["active_cameras"] = active_count

        score_map = {"SAFE":0,"INFO":1,"WARNING":2,"CRITICAL":3}
        timeline_data.append({"ts": time.time(), "score": score_map.get(global_level,0)})

        socketio.emit("camera_update", {"cameras": cameras_info, "stats": stats})
        socketio.emit("threat_update", {
            "level":   global_level,
            "color":   ThreatEngine.LEVEL_COLOR.get(global_level,"#34d399"),
            "reasons": agg_reasons if agg_reasons else ["Area clear"],
            "counts":  dict(total_counts),
            "fps":     avg_fps,
            "timeline":[{"ts":d["ts"],"score":d["score"]} for d in timeline_data],
            "detection_totals": dict(detection_log),
        })

        emit_alert = (global_level != prev_global_level) or (global_level in ("CRITICAL","WARNING"))
        if emit_alert and global_level != "SAFE":
            alert = {
                "id":          str(uuid.uuid4()),
                "timestamp":   time.time() * 1000,
                "type":        "Activity Detected",
                "severity":    global_level,
                "camera_ids":  list(cam_streams.keys()),
                "description": " | ".join(agg_reasons),
                "counts":      dict(total_counts),
            }
            alerts.insert(0, alert)
            if len(alerts) > 100: alerts.pop()
            if global_level in ("WARNING","CRITICAL"): stats["incident_count"] += 1
            socketio.emit("alert", alert)

        prev_global_level = global_level


# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = "sentinel-ai-secret!"
app.register_blueprint(api, url_prefix="/api")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed/<cam_id>")
def video_feed(cam_id):
    return Response(_generate_mjpeg(cam_id), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/threat")
def get_threat():
    return jsonify({"level": stats["threat_level"]})

@app.route("/api/alerts")
def get_alerts():
    return jsonify(alerts[:50])

@app.route("/api/stats")
def get_stats():
    return jsonify(stats)

@app.route("/api/cameras")
def get_cameras():
    return jsonify(cameras_info)

@app.route("/api/cameras/add", methods=["POST"])
def add_camera_api():
    data = request.json
    cam_id = data.get("id")
    source = data.get("source")
    location = data.get("location", "New Camera")
    
    if not cam_id or source is None:
        return jsonify({"success": False, "error": "Missing id or source"}), 400
        
    start_camera(cam_id, source, location)
    return jsonify({"success": True, "cam_id": cam_id})

if __name__ == "__main__":
    threading.Thread(target=_emitter_thread, daemon=True).start()
    print("\n🛡️  Sentinel AI  →  http://127.0.0.1:5000\n")
    socketio.run(app, debug=False, port=5000, use_reloader=False, allow_unsafe_werkzeug=True)
