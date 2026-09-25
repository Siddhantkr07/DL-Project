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
_yolo_model = None       # General object detection (knife, car, etc.)
_pose_model  = None      # Person detection via skeleton keypoints (zero false positives)
try:
    from ultralytics import YOLO
    # Pose model: detects 17 human skeleton keypoints — much more accurate for person detection
    _pose_model  = YOLO("yolo11x-pose.pt")
    # Detection model: for everything else (weapons, vehicles, hazards)
    _yolo_model  = YOLO("yolo11x.pt")
    YOLO_AVAILABLE = True
    log.info("YOLO11x + YOLO11x-Pose dual model loaded ✅")
except Exception as e:
    log.warning(f"YOLO not available: {e}")

# ── Zero-Shot Verification (CLIP) ─────────────────────────────────────────────
CLIP_AVAILABLE = False
_clip_model = None
_clip_processor = None
try:
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image
    import torch
    # ViT-B/32 is very fast and highly accurate for zero-shot image classification
    _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
    _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    CLIP_AVAILABLE = True
    log.info("OpenAI CLIP (ViT-B/32) Verification Engine loaded on GPU ✅")
except Exception as e:
    log.warning(f"CLIP Verification Engine not available: {e}")

# ── Threat Intelligence Engine (v2 — Temporal Smoothing) ─────────────────────
class ThreatEngine:
    WEAPON_CLASSES  = {"knife", "scissors"}          # Only real COCO weapon classes
    HAZARD_CLASSES  = {"fire", "smoke"}
    VEHICLE_CLASSES = {"car", "truck", "motorcycle", "bus"}
    PERSON_CLASS    = "person"

    # Confidence thresholds — raised significantly to kill false positives
    CONF_PERSON  = 0.70   # Person must be 70% confident (strict to avoid object=person)
    CONF_WEAPON  = 0.75   # Weapon must be 75% confident (very strict)
    CONF_VEHICLE = 0.55
    CONF_DEFAULT = 0.55

    # Temporal buffer: threat must be seen in N consecutive frames before alerting
    SMOOTHING_FRAMES = 5   # ~5 frames at 15fps ≈ 0.3 seconds of consistent detection

    LEVEL_RANK = {"SAFE": 0, "INFO": 1, "WARNING": 2, "CRITICAL": 3}
    LEVEL_COLOR = {
        "SAFE":     "#34d399",
        "INFO":     "#38bdf8",
        "WARNING":  "#fbbf24",
        "CRITICAL": "#f43f5e",
    }

    def __init__(self):
        # Circular buffer tracking how many of last N frames had each threat tag
        self._buffers = defaultdict(lambda: deque(maxlen=self.SMOOTHING_FRAMES))
        # Verification cache for CLIP: track_id -> is_threat (bool)
        self.verifications = {}

    def _proximity_ratio(self, x1, y1, x2, y2, W, H) -> float:
        return ((x2 - x1) * (y2 - y1)) / max(W * H, 1)

    def _verify_threat(self, frame_bgr, bbox, category) -> bool:
        """Uses OpenAI CLIP to verify the cropped region zero-shot."""
        if not CLIP_AVAILABLE:
            return True # fallback to YOLO if CLIP fails
            
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame_bgr[max(0,y1):y2, max(0,x1):x2]
        if crop.size == 0: return False
        
        # Convert BGR to RGB for PIL
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)

        if category == "weapon":
            # Very precise labels to force the model to differentiate
            labels = ["a photo of a deadly knife weapon", "a photo of a gun or pistol", 
                      "a photo of a smartphone", "a photo of a pen or pencil", 
                      "a photo of keys", "a photo of a hand", "a photo of a person"]
            threat_labels = {"a photo of a deadly knife weapon", "a photo of a gun or pistol"}
        elif category == "hazard":
            labels = ["a photo of dangerous fire and flames", "a photo of thick smoke", 
                      "a photo of a bright light or reflection", "a photo of fog or clouds"]
            threat_labels = {"a photo of dangerous fire and flames", "a photo of thick smoke"}
        else:
            return True
            
        import torch
        with torch.no_grad():
            inputs = _clip_processor(text=labels, images=pil_img, return_tensors="pt", padding=True).to("cuda")
            outputs = _clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]
            best_idx = probs.argmax().item()
            best_label = labels[best_idx]
            
        return best_label in threat_labels

    def _smooth(self, tag: str, detected: bool) -> bool:
        """Push detection result into rolling buffer. Return True only if
        the tag was detected in ALL of the last SMOOTHING_FRAMES frames."""
        self._buffers[tag].append(1 if detected else 0)
        buf = self._buffers[tag]
        return len(buf) == self.SMOOTHING_FRAMES and sum(buf) == self.SMOOTHING_FRAMES

    def analyse(self, results, frame_shape, pose_results=None, frame_bgr=None) -> dict:
        H, W = frame_shape[:2]
        now = time.time()
        persons, weapons, hazards, vehicles, other = [], [], [], [], []

        # ── PERSONS: from Pose model (skeleton keypoints — zero false positives) ──
        if YOLO_AVAILABLE and pose_results and len(pose_results[0].boxes):
            for i, box in enumerate(pose_results[0].boxes):
                conf = float(box.conf[0])
                if conf < self.CONF_PERSON:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                bw, bh = x2 - x1, y2 - y1
                if bh < 40: continue  # too tiny

                prox = self._proximity_ratio(x1, y1, x2, y2, W, H)
                track_id = int(box.id[0]) if box.id is not None else None
                obj = dict(cls="person", conf=round(conf, 2), prox=round(prox, 3),
                           bbox=[x1, y1, x2, y2], track_id=track_id)

                # ── Pose-based Fall Detection ─────────────────────────────────
                fallen = False
                try:
                    kpts      = pose_results[0].keypoints.xy[i]    # (17, 2) positions
                    kpts_conf = pose_results[0].keypoints.conf[i]  # (17,) confidence scores

                    # Only use keypoints the model is confident about (>50%)
                    KCONF = 0.5
                    sh_conf  = kpts_conf[[5, 6]]
                    an_conf  = kpts_conf[[15, 16]]
                    shoulders = kpts[[5, 6]]
                    ankles    = kpts[[15, 16]]

                    # Filter: keep only high-confidence keypoints
                    valid_sh = shoulders[sh_conf > KCONF]
                    valid_an = ankles[an_conf > KCONF]

                    # STRICT: BOTH ankles must be visible and confident
                    # If ankles aren't visible (sitting at desk, webcam), skip entirely
                    if len(valid_sh) >= 1 and len(valid_an) >= 2:
                        sh_y = float(valid_sh[:, 1].mean())
                        an_y = float(valid_an[:, 1].mean())
                        # Ankles must be below shoulders by at least 10% of frame height
                        # to confirm this is a full-body visible shot
                        if an_y > sh_y + (H * 0.10):
                            # THEN check if they're at roughly the same level (fallen)
                            if abs(an_y - sh_y) < (H * 0.12):
                                fallen = True
                except Exception:
                    pass

                if fallen:
                    obj["fallen"] = True
                persons.append(obj)

        # ── OBJECTS: from Detection model (weapons, vehicles, hazards) ────────────
        if YOLO_AVAILABLE and results and len(results[0].boxes):
            for box in results[0].boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = results[0].names[cls_id].lower()

                # Skip persons — handled by pose model above
                if cls_name == self.PERSON_CLASS:
                    continue

                if cls_name in self.WEAPON_CLASSES and conf < self.CONF_WEAPON:
                    continue
                elif cls_name in self.VEHICLE_CLASSES and conf < self.CONF_VEHICLE:
                    continue
                elif cls_name not in (self.WEAPON_CLASSES | self.HAZARD_CLASSES | self.VEHICLE_CLASSES):
                    if conf < self.CONF_DEFAULT:
                        continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                prox = self._proximity_ratio(x1, y1, x2, y2, W, H)
                track_id = int(box.id[0]) if box.id is not None else None
                obj = dict(cls=cls_name, conf=round(conf, 2), prox=round(prox, 3),
                           bbox=[x1, y1, x2, y2], track_id=track_id)

                # ── Zero-Shot Verification for Critical Threats ───────────────────
                if track_id is not None and (cls_name in self.WEAPON_CLASSES or cls_name in self.HAZARD_CLASSES):
                    cat = "weapon" if cls_name in self.WEAPON_CLASSES else "hazard"
                    if track_id not in self.verifications and frame_bgr is not None:
                        is_threat = self._verify_threat(frame_bgr, [x1, y1, x2, y2], cat)
                        self.verifications[track_id] = is_threat
                        
                    # If verified as FALSE (e.g. it's a phone, not a knife), ignore it
                    if track_id in self.verifications and not self.verifications[track_id]:
                        continue
                elif frame_bgr is not None and track_id is None and (cls_name in self.WEAPON_CLASSES or cls_name in self.HAZARD_CLASSES):
                    # No track_id, verify every frame (fallback)
                    cat = "weapon" if cls_name in self.WEAPON_CLASSES else "hazard"
                    if not self._verify_threat(frame_bgr, [x1, y1, x2, y2], cat):
                        continue

                if cls_name in self.WEAPON_CLASSES:    weapons.append(obj)
                elif cls_name in self.HAZARD_CLASSES:  hazards.append(obj)
                elif cls_name in self.VEHICLE_CLASSES: vehicles.append(obj)
                else:                                  other.append(obj)

        # ── Temporal smoothing checks ─────────────────────────────────────────
        fallen_confirmed  = self._smooth("fall",      any(p.get("fallen") for p in persons))
        weapon_confirmed  = self._smooth("weapon",    len(weapons) > 0)
        hazard_confirmed  = self._smooth("hazard",    len(hazards) > 0)
        crowd_surge       = self._smooth("crowd10",   len(persons) >= 10)
        crowd_high        = self._smooth("crowd5",    5 <= len(persons) < 10)
        # Proximity: person fills >65% of frame for SMOOTHING_FRAMES in a row
        prox_confirmed    = self._smooth("prox",
                                len([p for p in persons if p["prox"] > 0.65]) > 0)

        level, reasons, tags = "SAFE", [], []

        if fallen_confirmed:
            level = "CRITICAL"
            reasons.append("FALL DETECTED — person down")
            tags.append("fall")

        if weapon_confirmed:
            level = "CRITICAL"
            reasons.append(f"WEAPON DETECTED: {weapons[0]['cls'].upper()}")
            tags.append("weapon")

        if hazard_confirmed:
            level = "CRITICAL"
            reasons.append(f"HAZARD: {hazards[0]['cls'].upper()}")
            tags.append("hazard")

        if crowd_surge:
            level = "CRITICAL"
            reasons.append(f"CROWD SURGE — {len(persons)} persons")
            tags.append("crowd")
        elif crowd_high:
            if self.LEVEL_RANK[level] < self.LEVEL_RANK["WARNING"]: level = "WARNING"
            reasons.append(f"HIGH OCCUPANCY — {len(persons)} persons")
            tags.append("crowd")

        if prox_confirmed:
            if self.LEVEL_RANK[level] < self.LEVEL_RANK["WARNING"]: level = "WARNING"
            reasons.append("PROXIMITY BREACH — subject too close")
            tags.append("proximity")

        if not reasons:
            n = len(persons)
            if persons:
                level = "INFO"
                reasons.append(f"{n} person{'s' if n > 1 else ''} monitored")
                tags.append("normal")
            elif vehicles:
                level = "INFO"
                reasons.append(f"{len(vehicles)} vehicle(s) detected")
                tags.append("vehicle")
            else:
                level = "SAFE"
                reasons.append("Area clear")

        return {
            "level": level, "color": self.LEVEL_COLOR[level],
            "reasons": reasons, "tags": tags,
            "counts": {"persons": len(persons), "weapons": len(weapons),
                       "hazards": len(hazards), "vehicles": len(vehicles), "other": len(other)},
            "objects": persons + weapons + hazards + vehicles + other,
            "ts": now,
        }

_threat_engine = ThreatEngine()


# ── Multi-Camera Manager ───────────────────────────────────────────────────────
CLS_COLORS = {"person": (0, 230, 120), "car": (0, 190, 255), "truck": (0, 100, 255), "knife": (0, 0, 255), "fire": (0, 80, 255), "default": (180, 180, 0)}

def _draw(frame, results, threat, cam_id, fps, pose_results=None):
    h, w = frame.shape[:2]

    # ── Draw persons from pose model (green skeleton boxes) ──────────────────
    if YOLO_AVAILABLE and pose_results and len(pose_results[0].boxes):
        POSE_PAIRS = [(5,6),(5,11),(6,12),(11,12),(5,7),(7,9),(6,8),(8,10),
                      (11,13),(13,15),(12,14),(14,16),(0,5),(0,6)]
        for i, box in enumerate(pose_results[0].boxes):
            conf = float(box.conf[0])
            if conf < 0.60: continue
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            track_label = f" #{int(box.id[0])}" if box.id is not None else ""
            color = (0, 230, 120)  # green for person
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            lbl = f"person{track_label} {conf:.0%}"
            (tw,th),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.rectangle(frame, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
            cv2.putText(frame, lbl, (x1+2, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,0,0), 1)
            # Draw skeleton keypoints
            try:
                kpts = pose_results[0].keypoints.xy[i]
                for kp in kpts:
                    kx, ky = int(kp[0]), int(kp[1])
                    if kx > 0 and ky > 0:
                        cv2.circle(frame, (kx, ky), 3, (0,255,255), -1)
                for a, b in POSE_PAIRS:
                    ax,ay = int(kpts[a][0]), int(kpts[a][1])
                    bx,by = int(kpts[b][0]), int(kpts[b][1])
                    if ax > 0 and ay > 0 and bx > 0 and by > 0:
                        cv2.line(frame, (ax,ay), (bx,by), (0,200,255), 2)
            except Exception:
                pass

    # ── Draw objects from detection model (weapons, vehicles etc.) ────────────
    if YOLO_AVAILABLE and results and len(results[0].boxes):
        for box in results[0].boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = results[0].names[cls_id].lower()
            if cls_name == "person": continue  # handled above
            if cls_name in ThreatEngine.WEAPON_CLASSES and conf < 0.75: continue
            elif conf < 0.55: continue
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            color = CLS_COLORS.get(cls_name, CLS_COLORS["default"])
            track_label = f" #{int(box.id[0])}" if box.id is not None else ""
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
            
            # Flip horizontally for natural mirror view (laptop webcam only)
            if str(self.source).isdigit() or self.source == 0:
                frame = cv2.flip(frame, 1)

            now = time.time()
            fps = round(1.0/max(now-prev, 1e-9), 1)
            prev = now

            results      = None   # detection results (weapons, vehicles etc.)
            pose_results = None   # pose results (persons via skeleton)
            if YOLO_AVAILABLE:
                try:
                    # Run object detection (ByteTrack) — skips persons internally
                    results = _yolo_model.track(frame, persist=True,
                                                tracker="bytetrack.yaml", verbose=False)
                except: pass
                try:
                    # Run pose model — persons detected via 17 skeleton keypoints
                    if _pose_model is not None:
                        pose_results = _pose_model.track(frame, persist=True,
                                                         tracker="bytetrack.yaml", verbose=False)
                except: pass

            threat = _threat_engine.analyse(results or [], frame.shape,
                                            pose_results=pose_results,
                                            frame_bgr=frame)
            annotated = _draw(frame.copy(), results, threat, self.cam_id, fps,
                              pose_results=pose_results)
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
