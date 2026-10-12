"""
Sentinel AI – Dashboard Backend (Multi-Camera + YOLO11x)
Real webcam + IP Camera support + Threat Intelligence Engine
GPU Beast Mode: CUDA FP16 + imgsz=1280 + ByteTrack
"""

from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO
import threading, time, random, uuid, logging, cv2, numpy as np, os
from collections import deque, defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("SentinelAI")

# ── GPU Setup ─────────────────────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    # Maximize GPU throughput: enable TF32 + benchmark cudnn kernels
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark        = True
    torch.backends.cudnn.allow_tf32       = True
    log.info(f"🔥 GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB | FP16: ON")
else:
    log.warning("No CUDA GPU found — running on CPU (slow!)")

# ── YOLO ──────────────────────────────────────────────────────────────────────
# Key GPU settings applied to every .track() / .predict() call:
#   device='cuda'   → force GPU
#   half=True       → FP16 inference (2× faster on RTX, half VRAM usage)
#   imgsz=1280      → high-res input = much better small-object detection
#   iou=0.45        → tighter NMS = fewer overlapping duplicate boxes
YOLO_AVAILABLE   = False
YOLO_DEVICE      = DEVICE
YOLO_HALF        = (DEVICE == "cuda")   # FP16 only on GPU
YOLO_IMGSZ       = 640                  # Native 640x640 standard YOLO resolution for 30+ FPS
YOLO_IOU         = 0.45                 # NMS IoU threshold

_yolo_model = None   # Object detection model (weapons, vehicles, hazards)
_pose_model = None   # Pose model (persons via 17 skeleton keypoints)

try:
    from ultralytics import YOLO

    # ── Pose Model (Ultra-fast Nano Pose for 30+ FPS) ─────────────────────────
    # yolo11n-pose runs in ~2.5ms on RTX 4050 compared to 45ms for 11x
    pose_model_name = "yolo11n-pose.pt" if os.path.exists(os.path.join(os.path.dirname(__file__), "yolo11n-pose.pt")) or os.path.exists("yolo11n-pose.pt") else "yolo11x-pose.pt"
    _pose_model = YOLO(pose_model_name)
    _pose_model.to(YOLO_DEVICE)
    if YOLO_HALF:
        _pose_model.model.half()
    log.info(f"{pose_model_name} loaded on GPU (FP16) ✅")

    # ── Threat Detection Model (Guns, Knives, Explosives) ──────────────────────
    # We downloaded a specialized YOLOv8 threat detection model
    _threat_yolo = None
    threat_model_path = os.path.join(os.path.dirname(__file__), "threat_model.pt")
    if os.path.exists(threat_model_path):
        _threat_yolo = YOLO(threat_model_path)
        _threat_yolo.to(YOLO_DEVICE)
        if YOLO_HALF: _threat_yolo.model.half()
        log.info("Specialized Threat Model (Guns/Knives/Explosives) loaded ✅")

    # ── General Object Detection Model (Vehicles, Scissors, etc) ───────────────
    # Using yolo11s (small) instead of 11x since we just need it for basic vehicles
    # and scissors now. Saves massive VRAM and boosts FPS!
    _yolo_model = YOLO("yolo11s.pt")
    _yolo_model.to(YOLO_DEVICE)
    if YOLO_HALF:
        _yolo_model.model.half()
    log.info(f"COCO Object detection model on {YOLO_DEVICE.upper()} (FP16={YOLO_HALF}, imgsz={YOLO_IMGSZ}) ✅")

    # ── GPU Warm-up (eliminate first-frame latency spike) ─────────────────────
    _dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
    _yolo_model.predict(_dummy, device=YOLO_DEVICE, half=YOLO_HALF, imgsz=YOLO_IMGSZ, verbose=False)
    _pose_model.predict(_dummy, device=YOLO_DEVICE, half=YOLO_HALF, imgsz=YOLO_IMGSZ, verbose=False)
    if _threat_yolo:
        _threat_yolo.predict(_dummy, device=YOLO_DEVICE, half=YOLO_HALF, imgsz=YOLO_IMGSZ, verbose=False)
    del _dummy
    log.info("GPU warm-up complete 🔥")

    YOLO_AVAILABLE = True

except Exception as e:
    log.warning(f"YOLO not available: {e}")

# ── Zero-Shot Verification (CLIP) ─────────────────────────────────────────────
CLIP_AVAILABLE = False
_clip_model    = None
_clip_processor = None
try:
    from transformers import CLIPProcessor, CLIPModel
    from PIL import Image

    _clip_model = (
        CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        .to(DEVICE)
        .half()          # FP16 for CLIP too — 2× faster on RTX
        .eval()
    )
    _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    CLIP_AVAILABLE = True
    log.info("OpenAI CLIP (ViT-B/32) on GPU FP16 ✅")
except Exception as e:
    log.warning(f"CLIP not available: {e}")

# ── Cross-Camera Re-Identification & Movement Tracker (MCMT) ────────────────
class CrossCameraTracker:
    """Tracks targets across multiple cameras using normalized 512-D Vision Transformer embeddings."""
    def __init__(self, similarity_threshold=0.72, memory_seconds=180):
        self.similarity_threshold = similarity_threshold
        self.memory_seconds = memory_seconds
        # global_id -> {"cls": str, "embedding": np.ndarray, "last_seen": float, "current_cam": str, "history": list}
        self.entities = {}
        self.next_id = 1
        self.lock = threading.Lock()
        self.transitions = deque(maxlen=30)  # recent cross-camera transition events

    def match_or_register(self, crop_bgr, cls_name, cam_id):
        if not CLIP_AVAILABLE or crop_bgr is None or crop_bgr.size == 0:
            return None, None
        
        now = time.time()
        try:
            h, w = crop_bgr.shape[:2]
            if h < 35 or w < 35:
                return None, None
            pil_img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
            inputs = _clip_processor(images=pil_img, return_tensors="pt").to(DEVICE)
            if YOLO_HALF:
                inputs["pixel_values"] = inputs["pixel_values"].half()
            with torch.inference_mode():
                feat = _clip_model.get_image_features(**inputs)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                emb = feat[0].cpu().numpy()
        except Exception:
            return None, None

        with self.lock:
            # 1. Purge stale tracks older than memory_seconds
            stale = [k for k, v in self.entities.items() if now - v["last_seen"] > self.memory_seconds]
            for k in stale:
                del self.entities[k]

            best_gid = None
            best_sim = -1.0

            # 2. Check cosine similarity against active entities of same category
            for gid, ent in self.entities.items():
                if ent["cls"] != cls_name:
                    continue
                sim = float(np.dot(emb, ent["embedding"]))
                if sim > best_sim:
                    best_sim = sim
                    best_gid = gid

            trans_event = None
            if best_gid is not None and best_sim >= self.similarity_threshold:
                # Match confirmed!
                ent = self.entities[best_gid]
                prev_cam = ent["current_cam"]
                
                # Update visual representation using running average
                ent["embedding"] = 0.75 * ent["embedding"] + 0.25 * emb
                ent["embedding"] /= np.linalg.norm(ent["embedding"])
                ent["last_seen"] = now

                # Did the target move to a different camera?
                if prev_cam != cam_id:
                    ent["current_cam"] = cam_id
                    ent["history"].append({"cam": cam_id, "ts": now})
                    trans_event = {
                        "global_id": best_gid,
                        "cls": cls_name,
                        "from_cam": prev_cam,
                        "to_cam": cam_id,
                        "timestamp": now,
                        "similarity": round(best_sim, 2)
                    }
                    self.transitions.appendleft(trans_event)
                    log.info(f"🔄 Cross-Camera Movement: {cls_name.upper()} [{best_gid}] moved from {prev_cam} ➔ {cam_id} (Similarity: {best_sim:.0%})")
                return best_gid, trans_event
            else:
                # New global entity
                new_gid = f"TARGET-{self.next_id:02d}"
                self.next_id += 1
                self.entities[new_gid] = {
                    "cls": cls_name,
                    "embedding": emb,
                    "last_seen": now,
                    "current_cam": cam_id,
                    "history": [{"cam": cam_id, "ts": now}]
                }
                return new_gid, None

_cross_cam_tracker = CrossCameraTracker()

# ── Threat Intelligence Engine (v2 — Temporal Smoothing) ─────────────────────
class ThreatEngine:
    # 22+ Active Security Target Classes across Weapons, Hazards, Vehicles, Bags, Devices, Animals:
    WEAPON_CLASSES   = {"knife", "scissors", "weapon", "gun", "pistol", "rifle", "grenade", "baseball bat"} 
    HAZARD_CLASSES   = {"fire", "smoke", "explosion"}
    VEHICLE_CLASSES  = {"car", "truck", "motorcycle", "bus", "bicycle"}
    SUSPICIOUS_BAGS  = {"backpack", "handbag", "suitcase"}
    RESTRICTED_DEVS  = {"cell phone", "laptop"}
    ANIMAL_CLASSES   = {"dog", "cat", "bird"}
    PERSON_CLASS     = "person"

    # Confidence thresholds — balanced for high sensitivity without noise
    CONF_PERSON  = 0.50   
    CONF_WEAPON  = 0.50   # 50% lets user detect knives/scissors/guns smoothly
    CONF_VEHICLE = 0.50
    CONF_DEFAULT = 0.45

    # Temporal buffer: length of the rolling history window
    SMOOTHING_FRAMES = 5
    # Must be seen in at least 3 out of 5 frames (filters out 1-2 frame glitches = higher precision)
    REQUIRED_DETECTIONS = 3

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
        # Re-ID Cache: (cam_id, track_id) -> {"gid": str, "ts": float, "trans": dict}
        self._reid_cache = {}

    def _proximity_ratio(self, x1, y1, x2, y2, W, H) -> float:
        return ((x2 - x1) * (y2 - y1)) / max(W * H, 1)

    def _verify_threat(self, frame_bgr, bbox, category) -> bool:
        """Uses OpenAI CLIP to verify the cropped region zero-shot (GPU FP16)."""
        if not CLIP_AVAILABLE:
            return True  # fallback: trust YOLO if CLIP not available
            
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame_bgr[max(0,y1):y2, max(0,x1):x2]
        if crop.size == 0: return False
        
        # Convert BGR → RGB for PIL
        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

        if category == "weapon":
            # Multi-weapon positive labels + realistic negative counter-classes
            labels = ["a photo of a deadly knife weapon", "a photo of a gun or pistol or firearm",
                      "a photo of scissors", "a photo of a grenade or explosive", "a photo of a baseball bat",
                      "a photo of a smartphone or mobile phone", "a photo of a pen or pencil",
                      "a photo of keys", "a photo of a human hand or finger", "a photo of a person",
                      "a photo of a remote control", "a photo of a bottle or cup", "a photo of a tool or utensil"]
            threat_labels = {"a photo of a deadly knife weapon", "a photo of a gun or pistol or firearm",
                             "a photo of scissors", "a photo of a grenade or explosive", "a photo of a baseball bat"}
        elif category == "hazard":
            labels = ["a photo of dangerous fire and flames", "a photo of thick smoke",
                      "a photo of a bright light or reflection", "a photo of fog or clouds"]
            threat_labels = {"a photo of dangerous fire and flames", "a photo of thick smoke"}
        else:
            return True

        with torch.no_grad():
            inputs = _clip_processor(text=labels, images=pil_img,
                                     return_tensors="pt", padding=True).to(DEVICE)
            # Cast pixel_values to FP16 to match the half-precision CLIP model
            if YOLO_HALF:
                inputs["pixel_values"] = inputs["pixel_values"].half()
            outputs = _clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)[0]
            best_label = labels[probs.argmax().item()]

        return best_label in threat_labels

    def _smooth(self, tag: str, detected: bool) -> bool:
        """Push detection result into rolling buffer. Return True if
        detected in at least REQUIRED_DETECTIONS out of the last SMOOTHING_FRAMES."""
        self._buffers[tag].append(1 if detected else 0)
        buf = self._buffers[tag]
        # Allow flicker: as long as it's detected REQUIRED_DETECTIONS times in the window
        return sum(buf) >= self.REQUIRED_DETECTIONS

    def analyse(self, results, frame_shape, pose_results=None, frame_bgr=None, threat_results=None, cam_id="CAM-01") -> dict:
        H, W = frame_shape[:2]
        now = time.time()
        persons, weapons, hazards, vehicles, bags, devices, animals, other = [], [], [], [], [], [], [], []

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

                # ── Cross-Camera Re-ID (MCMT Tracking) ────────────────────────
                gid, trans = None, None
                if frame_bgr is not None and track_id is not None:
                    cache_key = (cam_id, track_id)
                    cached = self._reid_cache.get(cache_key)
                    if not cached or (now - cached["ts"] > 3.0):
                        crop = frame_bgr[max(0, y1):y2, max(0, x1):x2]
                        g_id, t_ev = _cross_cam_tracker.match_or_register(crop, "person", cam_id)
                        if g_id:
                            self._reid_cache[cache_key] = {"gid": g_id, "ts": now, "trans": t_ev}
                    if cache_key in self._reid_cache:
                        gid = self._reid_cache[cache_key]["gid"]
                        trans = self._reid_cache[cache_key].get("trans")

                obj = dict(cls="person", conf=round(conf, 2), prox=round(prox, 3),
                           bbox=[x1, y1, x2, y2], track_id=track_id, global_id=gid, transition=trans)

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
        all_boxes = []
        if YOLO_AVAILABLE and results and len(results[0].boxes):
            all_boxes.extend([(box, results[0].names) for box in results[0].boxes])
        if YOLO_AVAILABLE and threat_results and len(threat_results[0].boxes):
            all_boxes.extend([(box, threat_results[0].names) for box in threat_results[0].boxes])

        for box, names in all_boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = names[cls_id].lower()

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

            # ── Cross-Camera Re-ID for Weapons / Bags ─────────────────────────
            gid, trans = None, None
            if (cls_name in self.WEAPON_CLASSES or cls_name in self.SUSPICIOUS_BAGS) and frame_bgr is not None and track_id is not None:
                cache_key = (cam_id, track_id)
                cached = self._reid_cache.get(cache_key)
                if not cached or (now - cached["ts"] > 3.0):
                    crop = frame_bgr[max(0, y1):y2, max(0, x1):x2]
                    g_id, t_ev = _cross_cam_tracker.match_or_register(crop, cls_name, cam_id)
                    if g_id:
                        self._reid_cache[cache_key] = {"gid": g_id, "ts": now, "trans": t_ev}
                if cache_key in self._reid_cache:
                    gid = self._reid_cache[cache_key]["gid"]
                    trans = self._reid_cache[cache_key].get("trans")

            obj = dict(cls=cls_name, conf=round(conf, 2), prox=round(prox, 3),
                       bbox=[x1, y1, x2, y2], track_id=track_id, global_id=gid, transition=trans)

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
            elif cls_name in self.SUSPICIOUS_BAGS: bags.append(obj)
            elif cls_name in self.RESTRICTED_DEVS: devices.append(obj)
            elif cls_name in self.ANIMAL_CLASSES:  animals.append(obj)
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

        # ── Cross-Camera Movement Alerts ──────────────────────────────────────
        for ent_obj in (persons + weapons + bags):
            if ent_obj.get("transition"):
                tr = ent_obj["transition"]
                reasons.append(f"CROSS-CAMERA TRACK: {tr['cls'].upper()} [{tr['global_id']}] transitioned from {tr['from_cam']} ➔ {tr['to_cam']}")
                tags.append("reid")
                if self.LEVEL_RANK[level] < self.LEVEL_RANK["WARNING"]: level = "WARNING"
                break

        if fallen_confirmed:
            level = "CRITICAL"
            reasons.append("FALL DETECTED — person down")
            tags.append("fall")

        if weapon_confirmed:
            level = "CRITICAL"
            cls_name = weapons[0]['cls'].upper() if weapons else "WEAPON"
            reasons.append(f"WEAPON DETECTED: {cls_name}")
            tags.append("weapon")

        if hazard_confirmed:
            level = "CRITICAL"
            cls_name = hazards[0]['cls'].upper() if hazards else "HAZARD"
            reasons.append(f"HAZARD: {cls_name}")
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
            elif weapons:
                level = "CRITICAL"
                reasons.append(f"WEAPON: {weapons[0]['cls'].upper()}")
                tags.append("weapon")
            elif hazards:
                level = "CRITICAL"
                reasons.append(f"HAZARD: {hazards[0]['cls'].upper()}")
                tags.append("hazard")
            elif vehicles:
                level = "INFO"
                reasons.append(f"{len(vehicles)} vehicle(s) detected")
                tags.append("vehicle")
            elif bags:
                level = "INFO"
                reasons.append(f"BAG: {bags[0]['cls'].upper()}")
                tags.append("bag")
            elif devices:
                level = "INFO"
                reasons.append(f"DEVICE: {devices[0]['cls'].upper()}")
                tags.append("device")
            elif animals:
                level = "INFO"
                reasons.append(f"ANIMAL: {animals[0]['cls'].upper()}")
                tags.append("animal")
            elif other:
                level = "INFO"
                reasons.append(f"{other[0]['cls'].upper()} detected")
                tags.append("object")
            else:
                level = "SAFE"
                reasons.append("Area clear")

        return {
            "level": level, "color": self.LEVEL_COLOR[level],
            "reasons": reasons, "tags": tags,
            "counts": {"persons": len(persons), "weapons": len(weapons),
                       "hazards": len(hazards), "vehicles": len(vehicles),
                       "bags": len(bags), "devices": len(devices),
                       "animals": len(animals), "other": len(other)},
            "objects": persons + weapons + hazards + vehicles + bags + devices + animals + other,
            "ts": now,
        }

_threat_engine = ThreatEngine()


# ── Multi-Camera Manager ───────────────────────────────────────────────────────
CLS_COLORS = {
    "person": (0, 230, 120),
    "car": (0, 190, 255), "truck": (0, 100, 255), "motorcycle": (0, 150, 255), "bus": (0, 120, 255), "bicycle": (0, 220, 255),
    "knife": (0, 0, 255), "gun": (0, 0, 255), "pistol": (0, 0, 255), "rifle": (0, 0, 255), "grenade": (0, 0, 255), "weapon": (0, 0, 255),
    "baseball bat": (0, 50, 255), "scissors": (0, 100, 255), "fire": (0, 80, 255), "smoke": (150, 150, 150), "explosion": (0, 60, 255),
    "backpack": (255, 140, 0), "handbag": (255, 160, 0), "suitcase": (255, 120, 0),
    "cell phone": (180, 0, 255), "laptop": (200, 0, 230),
    "dog": (0, 255, 200), "cat": (0, 230, 200), "bird": (0, 210, 220),
    "bottle": (255, 200, 0), "default": (180, 180, 0)
}

def _draw(frame, results, threat_results, threat, cam_id, fps, pose_results=None):
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
            
            # Cross-camera Re-ID label & movement badge
            gid_label = ""
            bid = int(box.id[0]) if box.id is not None else None
            trans_info = None
            for o in threat.get("objects", []):
                if o.get("track_id") == bid and o.get("cls") == "person":
                    if o.get("global_id"):
                        gid_label = f" [{o['global_id']}]"
                    if o.get("transition"):
                        trans_info = o["transition"]
                    break

            color = (0, 230, 120)  # green for person
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            lbl = f"person{gid_label}{track_label} {conf:.0%}"
            (tw,th),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.rectangle(frame, (x1, y1-th-6), (x1+tw+4, y1), color, -1)
            cv2.putText(frame, lbl, (x1+2, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0,0,0), 1)

            if trans_info:
                badge = f">> MOVED: {trans_info['from_cam']} -> {trans_info['to_cam']}"
                (bw_t, bh_t), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
                cv2.rectangle(frame, (x1, max(0, y1-th-bh_t-14)), (x1+bw_t+8, max(0, y1-th-6)), (0, 212, 255), -1)
                cv2.putText(frame, badge, (x1+4, max(0, y1-th-9)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1)

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

    # ── Draw objects from detection models (weapons, vehicles etc.) ────────────
    all_boxes = []
    if YOLO_AVAILABLE and results and len(results[0].boxes):
        all_boxes.extend([(box, results[0].names) for box in results[0].boxes])
    if YOLO_AVAILABLE and threat_results and len(threat_results[0].boxes):
        all_boxes.extend([(box, threat_results[0].names) for box in threat_results[0].boxes])

    for box, names in all_boxes:
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = names[cls_id].lower()
        if cls_name == "person": continue  # handled above
        if cls_name in ThreatEngine.WEAPON_CLASSES and conf < ThreatEngine.CONF_WEAPON: continue
        elif conf < 0.50: continue
        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
        color = CLS_COLORS.get(cls_name, CLS_COLORS["default"])
        track_label = f" #{int(box.id[0])}" if box.id is not None else ""
        
        gid_label = ""
        bid = int(box.id[0]) if box.id is not None else None
        for o in threat.get("objects", []):
            if o.get("track_id") == bid and o.get("cls") == cls_name:
                if o.get("global_id"):
                    gid_label = f" [{o['global_id']}]"
                break

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        lbl = f"{cls_name}{gid_label}{track_label} {conf:.0%}"
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
        
        # We need a separate thread purely for reading the camera buffer,
        # otherwise cv2 blocks and queues up old frames, causing massive lag!
        self.grab_thread = None
        self.raw_frame = None

    def start(self):
        self.running = True
        self.grab_thread = threading.Thread(target=self._grabber, daemon=True)
        self.grab_thread.start()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def _grabber(self):
        src = int(self.source) if str(self.source).isdigit() else self.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            log.error(f"Cannot open camera {self.cam_id} at {self.source}")
            with self.lock: self.latest_threat["reasons"] = ["Camera Offline"]
            return
            
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        while self.running:
            ret, frame = cap.read()
            if ret:
                # Flip horizontally for natural mirror view (laptop webcam only)
                if str(self.source).isdigit() or self.source == 0:
                    frame = cv2.flip(frame, 1)
                self.raw_frame = frame
            else:
                time.sleep(0.005)
        cap.release()

    def _run(self):
        prev = time.time()
        frame_idx = 0
        last_pose_results = None
        while self.running:
            frame = self.raw_frame
            if frame is None:
                time.sleep(0.01)
                continue
            
            # Make a copy so grabber doesn't overwrite it mid-inference
            frame = frame.copy()
            frame_idx += 1

            now = time.time()
            fps = round(1.0/max(now-prev, 1e-9), 1)
            prev = now

            results        = None   # detection results (weapons, vehicles etc.)
            pose_results   = last_pose_results   # pose results (persons via skeleton)
            threat_results = None   # specialized threat results (guns, grenades, knives, explosion)
            if YOLO_AVAILABLE:
                with torch.inference_mode():
                    try:
                        # 1. Specialized Threat Detection: guns, knives, explosives
                        if _threat_yolo is not None:
                            threat_results = _threat_yolo.track(
                                frame,
                                persist       = True,
                                tracker       = "bytetrack.yaml",
                                device        = YOLO_DEVICE,
                                half          = YOLO_HALF,
                                imgsz         = YOLO_IMGSZ,
                                iou           = YOLO_IOU,
                                agnostic_nms  = False,
                                verbose       = False,
                            )
                    except Exception as e:
                        log.debug(f"Threat detection error: {e}")

                    try:
                        # 2. General Object Detection: 22+ security classes (vehicles, scissors, bags, devices)
                        results = _yolo_model.track(
                            frame,
                            persist       = True,
                            tracker       = "bytetrack.yaml",
                            device        = YOLO_DEVICE,
                            half          = YOLO_HALF,
                            imgsz         = YOLO_IMGSZ,
                            iou           = YOLO_IOU,
                            agnostic_nms  = False,
                            verbose       = False,
                        )
                    except Exception as e:
                        log.debug(f"Detection error: {e}")

                    try:
                        # 3. Pose model: run on alternate frames for huge FPS boost
                        # Humans move smoothly over hundreds of milliseconds; 2-frame cadence is visually seamless
                        if _pose_model is not None and (frame_idx % 2 == 0 or last_pose_results is None):
                            pose_results = _pose_model.track(
                                frame,
                                persist       = True,
                                tracker       = "bytetrack.yaml",
                                device        = YOLO_DEVICE,
                                half          = YOLO_HALF,
                                imgsz         = YOLO_IMGSZ,
                                iou           = YOLO_IOU,
                                agnostic_nms  = False,
                                verbose       = False,
                            )
                            last_pose_results = pose_results
                    except Exception as e:
                        log.debug(f"Pose error: {e}")

            threat = _threat_engine.analyse(results or [], frame.shape,
                                            pose_results=pose_results,
                                            frame_bgr=frame, threat_results=threat_results,
                                            cam_id=self.cam_id)
            annotated = _draw(frame.copy(), results, threat_results, threat, self.cam_id, fps,
                              pose_results=pose_results)
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
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

        # Multi-Camera Re-ID Target Tracking Update
        with _cross_cam_tracker.lock:
            active_targets = [
                {
                    "global_id": gid,
                    "cls": ent["cls"],
                    "current_cam": ent["current_cam"],
                    "last_seen": ent["last_seen"],
                    "history": list(ent["history"])
                }
                for gid, ent in list(_cross_cam_tracker.entities.items())
            ]
            trans_list = list(_cross_cam_tracker.transitions)[:10]

        socketio.emit("reid_update", {
            "transitions": trans_list,
            "active_targets": active_targets
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

@app.route("/api/reid")
def get_reid():
    with _cross_cam_tracker.lock:
        return jsonify({
            "transitions": list(_cross_cam_tracker.transitions),
            "active_targets": [
                {
                    "global_id": gid,
                    "cls": ent["cls"],
                    "current_cam": ent["current_cam"],
                    "last_seen": ent["last_seen"],
                    "history": list(ent["history"])
                }
                for gid, ent in list(_cross_cam_tracker.entities.items())
            ]
        })

@app.route("/api/stats")
def get_stats():
    return jsonify(stats)

@app.route("/api/cameras")
def get_cameras():
    return jsonify(cameras_info)

@app.route("/api/cameras/add", methods=["POST"])
def add_camera_api():
    data = request.json or {}
    cam_id = data.get("id")
    source = data.get("source")
    location = data.get("location", "New Camera")
    
    if not cam_id or source is None:
        return jsonify({"success": False, "error": "Missing id or source"}), 400
        
    start_camera(cam_id, source, location)
    return jsonify({"success": True, "cam_id": cam_id})

@app.route("/api/cameras/stop", methods=["POST"])
def stop_camera_api():
    data = request.json or {}
    cam_id = data.get("id")
    if not cam_id or cam_id == "CAM-01":
        return jsonify({"success": False, "error": "Cannot disconnect primary camera"}), 400
    if cam_id in cam_streams:
        cam_streams[cam_id].stop()
        del cam_streams[cam_id]
    for c in cameras_info:
        if c["id"] == cam_id:
            c["status"] = "inactive"
            c["live"] = False
            c["source"] = ""
            c["fps"] = 0.0
            c["detections"] = 0
            c["threat"] = "SAFE"
    return jsonify({"success": True, "cam_id": cam_id})

if __name__ == "__main__":
    threading.Thread(target=_emitter_thread, daemon=True).start()
    print("\n🛡️  Sentinel AI  →  http://127.0.0.1:5000\n")
    socketio.run(app, debug=False, port=5000, use_reloader=False, allow_unsafe_werkzeug=True)
