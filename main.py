import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os
import urllib.request
import threading
import math
from collections import deque
from pynput import keyboard
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import asyncio
from fastapi import WebSocket, WebSocketDisconnect
import json

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)
templates = Jinja2Templates(directory="templates")

# ================= BACKGROUND TRACKER ================= #
class BackgroundTracker:
    def __init__(self):
        self.key_latencies = deque(maxlen=20)
        self.backspaces = 0
        self.mouse_moves = 0
        self.last_key_time = time.time()
        self.avg_lat = 0
        self.performance_drop = False
        
        self.kbd_listener = keyboard.Listener(on_release=self.on_release)
        self.kbd_listener.start()
        
        # Phase 1: Mouse Tracking
        from pynput import mouse
        self.mouse_listener = mouse.Listener(on_move=self.on_mouse_move)
        self.mouse_listener.start()
        
        threading.Thread(target=self.decay_monitor, daemon=True).start()

    def on_release(self, key):
        now = time.time()
        delay = (now - self.last_key_time) * 1000
        if delay < 2000:
            self.key_latencies.append(delay)
        self.last_key_time = now
        
        if key == keyboard.Key.backspace:
            self.backspaces += 1
            
        if len(self.key_latencies) > 5:
            self.avg_lat = sum(self.key_latencies) / len(self.key_latencies)
            if self.avg_lat > 300 or self.backspaces > 15:
                self.performance_drop = True

    def on_mouse_move(self, x, y):
        self.mouse_moves += 1

    def decay_monitor(self):
        while True:
            time.sleep(1)
            # If user hasn't typed in 4 seconds, clear latency smoothly
            if time.time() - self.last_key_time > 4:
                self.avg_lat = max(0, self.avg_lat - 15)
                self.performance_drop = False
            # Decay backspaces every 5 seconds
            if int(time.time()) % 5 == 0 and self.backspaces > 0:
                self.backspaces = max(0, self.backspaces - 1)
            # Decay mouse moves
            if int(time.time()) % 2 == 0:
                self.mouse_moves = max(0, self.mouse_moves - 10)

bg_tracker = BackgroundTracker()

# ================= GLOBAL STATE ================= #
class AppState:
    def __init__(self):
        self.fatigue_score = 0.0
        self.is_calibrating = False
        self.calibration_start_time = 0
        self.baseline_spine_angle = None
        self.baseline_eye_y = None
        self.current_posture_status = "Good Posture" 
        
        # History arrays for dynamic charts (last 30 data points)
        self.fatigue_history = [0] * 30
        self.latency_history = [0] * 30
        
        # Teams Integration Mock
        self.teams_integration_active = True
        self.teams_status = "Available"
        
        # Blink tracking
        self.ear = 0.0
        self.blinks = 0
        self.last_blink_time = time.time()
        self.frames_closed = 0

state = AppState()

# Thresholds
SLOUCH_ANGLE_THRESHOLD = 15
EYE_LEVEL_THRESHOLD = 0.05
FATIGUE_INCREMENT = 0.5
MAX_FATIGUE = 100
FATIGUE_DANGER = 75

# ================= UTILS ================= #
def ensure_models_exist():
    models = {
        'pose_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task',
        'face_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task'
    }
    for model_name, url in models.items():
        if not os.path.exists(model_name):
            print(f"Downloading {model_name}...")
            try:
                urllib.request.urlretrieve(url, model_name)
            except Exception as e:
                print(f"Error: {e}")

ensure_models_exist()

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360 - angle if angle > 180.0 else angle

def eye_aspect_ratio(eye):
    A = math.dist(eye[1], eye[5])
    B = math.dist(eye[2], eye[4])
    C = math.dist(eye[0], eye[3])
    max_C = max(C, 0.001) # Prevent DivZero edge cases
    return (A + B) / (2.0 * max_C)

# ================= VIDEO PROCESSING ================= #
def process_frame(frame, pose_detector, face_detector):
    global state
    blank_image = np.zeros(frame.shape, dtype=np.uint8)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    
    pose_results = pose_detector.detect(mp_image)
    face_results = face_detector.detect(mp_image)

    color = (0, 255, 0)
    status = "Good Posture"

    # Posture Tracking
    if pose_results and pose_results.pose_landmarks:
        landmarks = pose_results.pose_landmarks[0]
        L_S, R_S = 11, 12
        L_H, R_H = 23, 24
        L_E, R_E = 7, 8
        
        # Geometric AI Engine inspired by "SitPose" research
        # We calculate vectors to determine true spinal curvature and shoulder imbalance
        L_S, R_S = 11, 12
        L_H, R_H = 23, 24
        L_E, R_E = 7, 8
        
        def get_midpoint(l, r):
            return np.array([(landmarks[l].x + landmarks[r].x)/2, (landmarks[l].y + landmarks[r].y)/2])
            
        mid_shoulder = get_midpoint(L_S, R_S)
        mid_hip = get_midpoint(L_H, R_H)
        mid_ear = get_midpoint(L_E, R_E)

        # 1. Forward Hunching / Neck Curvature (Angle between Torso and Neck vectors)
        v_torso = mid_shoulder - mid_hip
        v_neck = mid_ear - mid_shoulder
        cosine_angle = np.dot(v_torso, v_neck) / (np.linalg.norm(v_torso) * np.linalg.norm(v_neck))
        neck_deviation = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

        # 2. Lateral Posture / Leaning (Shoulder Tilt)
        shoulder_dx = landmarks[R_S].x - landmarks[L_S].x
        shoulder_dy = landmarks[R_S].y - landmarks[L_S].y
        shoulder_tilt = np.degrees(np.arctan2(shoulder_dy, shoulder_dx))
        
        # 3. Screen Distance (Qurb al-Shasha) using shoulder width proportion (Phase 3)
        # If the shoulders take up a lot of the frame width, user is too close.
        shoulder_width = math.dist([landmarks[L_S].x, landmarks[L_S].y], [landmarks[R_S].x, landmarks[R_S].y])

        if state.is_calibrating:
            status = "Calibrating..."
            color = (0, 255, 255)
            if time.time() - state.calibration_start_time > 5:
                state.is_calibrating = False
                state.baseline_spine_angle = neck_deviation
                state.baseline_eye_y = shoulder_tilt
                status = "Calibrated!"
        elif state.baseline_spine_angle is not None:
            neck_diff = abs(neck_deviation - state.baseline_spine_angle)
            tilt_diff = abs(shoulder_tilt - state.baseline_eye_y)

            bad_posture = False
            if neck_diff > 12: # Strict threshold for forward hunching/text neck
                status = "Warning: Hunching / Forward Lean"
                bad_posture = True
            elif tilt_diff > 8: # Strict threshold for bad lateral leaning (Left/Right Sitting)
                status = "Warning: Sideways Slouch"
                bad_posture = True
            elif shoulder_width > 0.6: # Shoulders take up >60% of the camera width! Too close!
                status = "Warning: Too Close to Screen"
                bad_posture = True

            if bad_posture:
                color = (0, 0, 255)
                state.fatigue_score = min(MAX_FATIGUE, state.fatigue_score + FATIGUE_INCREMENT)
            else:
                # Recover slightly when posture is fixed
                state.fatigue_score = max(0, state.fatigue_score - 0.05)

        state.current_posture_status = status

        h, w, _ = frame.shape
        pt1 = (int(mid_shoulder[0] * w), int(mid_shoulder[1] * h))
        pt2 = (int(mid_hip[0] * w), int(mid_hip[1] * h))
        pt3 = (int(mid_ear[0] * w), int(mid_ear[1] * h))
        
        # Draw skeleton directly on frame
        overlay = frame.copy()
        cv2.line(overlay, pt1, pt2, color, 4)
        cv2.line(overlay, pt3, pt1, color, 4)
        cv2.circle(overlay, pt1, 6, color, -1)
        cv2.circle(overlay, pt2, 6, color, -1)
        cv2.circle(overlay, pt3, 6, color, -1)
        
        # Add visual glow blending
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Blinking & EAR Tracking
    if face_results and face_results.face_landmarks:
        face = face_results.face_landmarks[0]
        # Common eye landmark indices natively
        left_indices = [362, 385, 387, 263, 373, 380]
        right_indices = [33, 160, 158, 133, 153, 144]
        
        # Also draw glowing eyes for wow-factor
        h, w, _ = frame.shape
        left_eye = [(face[i].x, face[i].y) for i in left_indices]
        right_eye = [(face[i].x, face[i].y) for i in right_indices]
        
        for e_pt in left_eye + right_eye:
            px, py = int(e_pt[0] * w), int(e_pt[1] * h)
            cv2.circle(frame, (px, py), 2, (255, 0, 0), -1)

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        state.ear = round(ear, 2)
        
        # Blink detection
        if ear < 0.2:
            state.frames_closed += 1
        else:
            if state.frames_closed >= 1:
                state.blinks += 1
                state.fatigue_score = min(MAX_FATIGUE, state.fatigue_score + 1.0) # Penalty for rapid blinks
            state.frames_closed = 0
            
    # Remove old status drawing
    # cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    # cv2.putText(frame, f"EAR: {state.ear}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()

def generate_frames():
    opts_pose = vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path='pose_landmarker.task'))
    opts_face = vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path='face_landmarker.task'))
    pose_detector = vision.PoseLandmarker.create_from_options(opts_pose)
    face_detector = vision.FaceLandmarker.create_from_options(opts_face)
    
    cap = None
    try:
        while True:
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Prevent frame queuing (real-time only)
                time.sleep(1) # Let camera warm up
            
            success, frame = cap.read()
            if success:
                frame_bytes = process_frame(frame, pose_detector, face_detector)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        if cap is not None:
            cap.release()

# ================= API & WEBSOCKET ================= #

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # --- KEYBOARD FATIGUE DIRECT INFLUENCE ---
            if bg_tracker.avg_lat > 200 or bg_tracker.backspaces > 8:
                state.fatigue_score = min(100.0, state.fatigue_score + 0.8)
            else:
                state.fatigue_score = max(0.0, state.fatigue_score - 0.1)

            # Update History (sliding window of 30)
            state.fatigue_history.append(float(round(state.fatigue_score, 1)))
            state.fatigue_history.pop(0)
            state.latency_history.append(int(bg_tracker.avg_lat))
            state.latency_history.pop(0)

            # --- TEAMS INTEGRATION LOGIC ---
            if state.fatigue_score > FATIGUE_DANGER:
                state.teams_status = "Do Not Disturb"
            else:
                state.teams_status = "Available"

            data = {
                "score": round(state.fatigue_score, 1),
                "history": {
                    "fatigue": state.fatigue_history,
                    "latency": state.latency_history
                },
                "keyboard": {
                    "latency": int(bg_tracker.avg_lat),
                    "errors": bg_tracker.backspaces
                },
                "camera": {
                    "posture_status": state.current_posture_status,
                    "ear": state.ear,
                    "blinks": state.blinks
                },
                "integration": {
                    "teams_active": state.teams_integration_active,
                    "teams_status": state.teams_status
                }
            }
            await websocket.send_text(json.dumps(data))
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        pass

@app.post("/api/calibrate")
async def start_calibration():
    if not state.is_calibrating:
        state.is_calibrating = True
        state.calibration_start_time = time.time()
        return {"message": "Calibration started"}
    return {"message": "Already calibrating"}

@app.post("/api/reset_fatigue")
async def reset_fatigue():
    # Phase 4 Reset: Restore Teams to Available, reset scores
    state.fatigue_score = 0.0
    bg_tracker.key_latencies.clear()
    bg_tracker.avg_lat = 0
    bg_tracker.backspaces = 0
    state.teams_status = "Available"
    return {"message": "Fatigue reset"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
