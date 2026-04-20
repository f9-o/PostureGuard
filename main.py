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
from fastapi import FastAPI, Request, Form, Response
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
try:
    from sklearn.ensemble import IsolationForest
except ImportError:
    pass
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

# In-memory users database
users_db = {
    "sara": {"password": "1234", "role": "employee"},
    "admin": {"password": "admin", "role": "manager"},
    "raghad": {"password": "1234", "role": "employee"}
}

# ================= BACKGROUND TRACKER ================= #
class BackgroundTracker:
    def __init__(self):
        self.key_latencies = deque(maxlen=20)
        self.backspaces = 0
        self.mouse_moves = 0
        self.mouse_events = deque(maxlen=120)
        self.mouse_intervals = deque(maxlen=20)
        self.last_mouse_move = time.time()
        self.last_key_time = time.time()
        self.last_mouse_time = time.time()
        self.avg_lat = 0
        self.performance_drop = False
        
        # Machine Learning: IsolationForest for Unsupervised Anomaly Detection
        try:
            self.ml_model = IsolationForest(contamination=0.1, random_state=42)
            self.ml_ready = True
        except NameError:
            self.ml_model = None
            self.ml_ready = False
            
        self.ml_trained = False
        self.training_data = []  # Features: [latency, backspaces, mouse_rate]
        self.is_training_active = False
        
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
        self.last_mouse_time = now  # Keyboard activity counts as activity
        state.last_activity_time = now  # Update global activity time

        if key == keyboard.Key.backspace:
            self.backspaces += 1

        if len(self.key_latencies) > 0:
            self.avg_lat = sum(self.key_latencies) / len(self.key_latencies)
            if self.avg_lat > 300 or self.backspaces > 15:
                self.performance_drop = True

    def on_mouse_move(self, x, y):
        now = time.time()
        interval = now - self.last_mouse_move
        if interval > 0 and interval < 10:
            self.mouse_intervals.append(interval)
        self.last_mouse_move = now
        self.mouse_events.append(now)
        self.last_mouse_time = now
        state.last_activity_time = now  # Update global activity time
        self.mouse_moves += 1

    def recent_mouse_rate(self, window=30):
        now = time.time()
        return sum(1 for t in self.mouse_events if now - t <= window) / max(1, window)

    def is_idle(self):
        """Check if user has been idle for more than 30 seconds"""
        now = time.time()
        return (now - self.last_key_time > 30) and (now - self.last_mouse_time > 30)

    def is_idle_5min(self):
        """Check if user has been idle for more than 5 minutes"""
        now = time.time()
        return (now - state.last_activity_time > 300)  # 5 minutes

    def start_ml_training(self):
        self.is_training_active = True
        self.training_data = []
        # Simulate baseline if they didn't type much
        for _ in range(10):
            self.training_data.append([np.random.normal(150, 20), 0, np.random.normal(5, 1)])

    def stop_ml_training_and_fit(self):
        self.is_training_active = False
        if self.ml_ready and len(self.training_data) > 0:
            import numpy as np
            X = np.array(self.training_data)
            self.ml_model.fit(X)
            self.ml_trained = True
            print("🚀 AI ML Model Trained via IsolationForest on User Baseline!")

    def detect_anomaly(self):
        if not self.ml_trained or not self.ml_ready:
            return False
        import numpy as np
        # Feature vector: [Avg Latency, Backspaces, Mouse Moves]
        X_test = np.array([[self.avg_lat, self.backspaces, self.recent_mouse_rate()]])
        # Prediction: 1 for normal, -1 for anomaly
        prediction = self.ml_model.predict(X_test)[0]
        return prediction == -1

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
                
            # Collect ML Data or Predict
            if self.is_training_active and self.avg_lat > 0:
                self.training_data.append([self.avg_lat, self.backspaces, self.recent_mouse_rate()])
                
            if self.ml_trained and self.avg_lat > 0:
                is_anomaly = self.detect_anomaly()
                if is_anomaly:
                    self.performance_drop = True

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
        
        # CALIBRATION: Personal baseline metrics
        self.calibration_complete = False
        self.baseline_typing_speed = None  # avg latency in ms
        self.baseline_backspace_rate = None  # backspaces per minute
        self.baseline_mouse_activity = None  # mouse moves per second
        
        # History arrays for dynamic charts (last 30 data points)
        self.fatigue_history = [0] * 30
        self.latency_history = [0] * 30

        # Camera state
        self.camera_active = False
        self.camera_persistent = False  # NEW: Hackathon continuous mode
        self.camera_start_time = 0
        self.camera_last_closed = 0.0
        
        # Blink tracking
        self.ear = 0.0
        self.blinks = 0
        self.last_blink_time = time.time()
        self.frames_closed = 0
        self.ear_history = deque(maxlen=150) # Approx 5 seconds of frames at 30fps
        self.perclos = 0.0

        # Streak System
        self.streak_days = 1
        self.breaks_taken_today = 0
        self.breaks_skipped_today = 0
        self.daily_fatigue_average = 0.0
        self.last_session_date = time.strftime("%Y-%m-%d")
        
        # Break state
        self.break_active = False
        self.break_duration = 0
        self.break_start_time = 0
        self.last_break_popup_time = 0  # prevent repeated popups
        self.break_popup_shown = False  # only show once per fatigue cycle
        
        # Suspicion tracking for 4-step system
        self.suspicious_activity_count = 0  # Track consecutive suspicious updates
        self.last_typing_baseline = None  # Previous typing speed for comparison
        self.last_mouse_baseline = None  # Previous mouse movement for comparison
        self.camera_check_completed = False  # Whether last camera check was done
        self.needs_camera_verification = False  # Step 2: Camera needed for verification
        
        # Desk assessment state for tracking improvements
        self.previous_desk_assessment = {}  # Track previous warnings to show improvements
        
        # Presence and idle detection
        self.user_present = True  # Presence detection via camera
        self.manually_paused = False  # Manual pause button
        self.idle_popup_shown = False  # 5-minute idle popup shown
        self.last_activity_time = time.time()  # Last keyboard/mouse activity
        self.last_presence_check = time.time()  # Last camera presence check
        self.presence_check_interval = 300  # 5 minutes normally, 120 when away
        self.fatigue_frozen = False  # Whether fatigue calculation is frozen
        self.presence_check_mode = False  # Camera currently checking presence only
        self.presence_faces_seen = 0  # Face detections during check
        self.presence_message = ''  # Message to send when presence state changes
        self.break_popup_allowed = True  # Show a break popup only once per cycle
        self.break_popup_shown = False  # Keep track of shown break popup
        self.break_active = False

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
def process_frame(frame, pose_detector, face_detector, presence_only=False, can_increase_fatigue=True):
    global state
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

    if presence_only:
        face_results = face_detector.detect(mp_image)
        if face_results and face_results.face_landmarks:
            state.presence_faces_seen += 1
        ret, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes()

    # Check if user has recent keyboard/mouse activity
    has_recent_activity = not bg_tracker.is_idle_5min()
    # Only allow fatigue increase from posture/blinks if user is actively typing/moving mouse
    can_fatigue_increase = can_increase_fatigue and has_recent_activity

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
        
        # Geometric posture engine inspired by SitPose research
        # We calculate vectors to determine spinal curvature and shoulder imbalance
        L_S, R_S = 11, 12
        L_H, R_H = 23, 24
        L_E, R_E = 7, 8
        
        def get_midpoint(l, r):
            return np.array([(landmarks[l].x + landmarks[r].x)/2, (landmarks[l].y + landmarks[r].y)/2])
            
        mid_shoulder = get_midpoint(L_S, R_S)
        mid_hip = get_midpoint(L_H, R_H)
        mid_ear = get_midpoint(L_E, R_E)

        # 1. Craniovertebral Angle (CVA) - Medical standard for Forward Head Posture
        # Angle between a horizontal line through the shoulder and a line to the ear.
        # Vector from shoulder to ear:
        dx = mid_ear[0] - mid_shoulder[0]
        dy = mid_ear[1] - mid_shoulder[1] # Note: y grows downwards in images
        # CVA is the angle relative to horizontal. dy is negative if ear is above shoulder.
        cva_angle = np.degrees(np.arctan2(abs(dy), abs(dx))) 

        # 2. Lateral Posture / Leaning (Shoulder Tilt)
        shoulder_dx = landmarks[R_S].x - landmarks[L_S].x
        shoulder_dy = landmarks[R_S].y - landmarks[L_S].y
        shoulder_tilt = np.degrees(np.arctan2(abs(shoulder_dy), abs(shoulder_dx)))
        
        # 3. Screen Distance (Qurb al-Shasha) using shoulder width proportion (Phase 3)
        # If the shoulders take up a lot of the frame width, user is too close.
        shoulder_width = math.dist([landmarks[L_S].x, landmarks[L_S].y], [landmarks[R_S].x, landmarks[R_S].y])

        if state.is_calibrating:
            status = "جاري المعايرة..."
            color = (0, 255, 255)
            if time.time() - state.calibration_start_time > 5:
                state.is_calibrating = False
                state.baseline_spine_angle = cva_angle
                state.baseline_eye_y = shoulder_tilt
                status = "✅ تمت المعايرة!"
        elif state.baseline_spine_angle is not None:
            # For CVA, a lower angle (< 48 deg) means severe forward head posture
            # If the current CVA drops significantly below baseline (e.g. by 10+ degrees), it's bad.
            cva_diff = state.baseline_spine_angle - cva_angle
            tilt_diff = abs(shoulder_tilt - state.baseline_eye_y)

            bad_posture = False
            # Medical CVA threshold for FHP is deeply lower than baseline
            if cva_diff > 10 or tilt_diff > 5 or shoulder_width > 0.5:  
                status = "عدّل رقبتك (CVA Danger) ⚠️"
                bad_posture = True
            else:
                status = "جلستك ممتازة ✅"

            if bad_posture:
                color = (0, 0, 255)
                # Only increase fatigue if user has recent keyboard/mouse activity
                if can_fatigue_increase:
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
        state.ear_history.append(ear)
        
        # PERCLOS Calculation (Percentage of Eye Closure > 80% (ear < 0.2))
        closed_frames_count = sum(1 for e in state.ear_history if e < 0.2)
        state.perclos = closed_frames_count / max(1, len(state.ear_history))
        
        # If PERCLOS > 15% in the recent window, fatigue sharply increases (Medical Threshold)
        if state.perclos > 0.15 and len(state.ear_history) > 30 and can_fatigue_increase:
            state.fatigue_score = min(MAX_FATIGUE, state.fatigue_score + 1.0)
            status = "العين مرهقة (نعاس) ⚠️"
        
        # Blink detection (Rapid blinking)
        if ear < 0.2:
            state.frames_closed += 1
        else:
            if state.frames_closed >= 1:
                state.blinks += 1
                if can_fatigue_increase:
                    state.fatigue_score = min(MAX_FATIGUE, state.fatigue_score + 0.2) 
            state.frames_closed = 0
            
    # Remove old status drawing
    # cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    # cv2.putText(frame, f"EAR: {state.ear}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()

def generate_frames():
    opts_pose = vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path="pose_landmarker.task"))
    opts_face = vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path="face_landmarker.task"))
    pose_detector = vision.PoseLandmarker.create_from_options(opts_pose)
    face_detector = vision.FaceLandmarker.create_from_options(opts_face)

    cap = None
    blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        while True:
            if not state.camera_active:
                if cap is not None:
                    cap.release()
                    cap = None
                _, buffer = cv2.imencode('.jpg', blank_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.5)
                continue

            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                time.sleep(1)
                print("Camera opened")
                start_time = time.time()

            while state.camera_active and (time.time() - start_time < 5 or state.camera_persistent):
                success, frame = cap.read()
                if not success:
                    break
                # Don't allow fatigue increase during presence checks (unless persistent hackathon mode)
                can_increase = (not state.presence_check_mode) or state.camera_persistent
                frame_bytes = process_frame(frame, pose_detector, face_detector, presence_only=state.presence_check_mode and not state.camera_persistent, can_increase_fatigue=can_increase)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            if not state.camera_persistent:
                if cap is not None:
                    cap.release()
                    cap = None

            if state.presence_check_mode:
                if state.presence_faces_seen > 0:
                    if not state.user_present:
                        state.presence_message = '▶ مرحباً بعودتك — استُؤنف القياس'
                    state.user_present = True
                    state.presence_check_interval = 300
                else:
                    if state.user_present:
                        state.presence_message = '⏸ لم يتم اكتشاف وجهك — القياس متوقف'
                    state.user_present = False
                    state.presence_check_interval = 300

                state.presence_faces_seen = 0
                state.presence_check_mode = False

            if not state.camera_persistent:
                state.camera_active = False
            time.sleep(0.5)
    finally:
        if cap is not None:
            cap.release()
# ================= API & WEBSOCKET ================= #

@app.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    return templates.TemplateResponse(request=request, name="landing.html")

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    role = request.query_params.get("role", "employee")
    return templates.TemplateResponse(request=request, name="login.html", context={"role": role})

@app.post("/login", response_class=HTMLResponse)
async def login_post(request: Request, username: str = Form(...), password: str = Form(...)):
    if username in users_db and users_db[username]["password"] == password:
        user_role = users_db[username]["role"]
        response_url = "/manager" if user_role == 'manager' else "/calibration"
        response = RedirectResponse(url=response_url, status_code=303)
        response.set_cookie("role", user_role, httponly=True)
        response.set_cookie("username", username, httponly=True)
        return response
    
    return templates.TemplateResponse(
        request=request,
        name="login.html",
        context={"request": request, "error": "بيانات خاطئة، حاول مجدداً", "role": "employee"}
    )

@app.post("/signup", response_class=HTMLResponse)
async def signup_post(request: Request, fullname: str = Form(...), username: str = Form(...), 
                     password: str = Form(...), password_confirm: str = Form(...), 
                     role: str = Form(...)):
    if password != password_confirm:
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={"request": request, "signup_error": "كلمتا المرور غير متطابقتين", "role": role}
        )
    
    if username in users_db:
        return templates.TemplateResponse(
            request=request,
            name="login.html",
            context={"request": request, "signup_error": "اسم المستخدم مستخدم مسبقاً", "role": role}
        )
    
    # Add new user
    users_db[username] = {"password": password, "role": role}
    
    # Redirect managers directly to manager dashboard; employees calibrate first
    response_url = "/manager" if role == 'manager' else "/calibration"
    response = RedirectResponse(url=response_url, status_code=303)
    response.set_cookie("role", role, httponly=True)
    response.set_cookie("username", username, httponly=True)
    return response

@app.get("/calibration", response_class=HTMLResponse)
async def calibration_page(request: Request):
    role = request.cookies.get("role")
    username = request.cookies.get("username")
    if not role or not username:
        return RedirectResponse(url="/login")
    if role == "manager":
        return RedirectResponse(url="/manager")
    
    bg_tracker.start_ml_training()
    return templates.TemplateResponse(request=request, name="calibration.html")

@app.post("/api/calibration-complete")
async def calibration_complete(request: Request):
    data = await request.json()
    # Store calibration baselines 
    state.calibration_complete = True
    state.baseline_typing_speed = data.get('typing_count', 0) / 2  
    state.baseline_backspace_rate = data.get('backspace_count', 0) / 2  
    state.baseline_mouse_activity = data.get('mouse_count', 0) / 120  
    
    bg_tracker.stop_ml_training_and_fit()
    return {"status": "calibrated"}

@app.post("/api/desk-assessment")
async def desk_assessment(request: Request):
    data = await request.json()
    height = data.get('height')  # Height in cm
    
    # Mock desk assessment results with improvement tracking
    import random
    
    # Generate current assessment
    current_results = {
        "screen_height": "✅ مناسب" if random.random() > 0.4 else "⚠️ يحتاج تعديل: الشاشة منخفضة، ارفعها بمقدار 10 سم",
        "screen_distance": "✅ مناسب" if random.random() > 0.4 else "⚠️ يحتاج تعديل: أقرب من الشاشة، ابتعد مسافة ذراع واحدة",
        "lighting": "✅ مناسب" if random.random() > 0.4 else "⚠️ يحتاج تعديل: الإضاءة خافتة، أضف مصدر إضاءة إضافي"
    }
    
    # Chair height assessment based on user height
    if height:
        if height < 160:
            current_results["chair_height"] = "⚠️ يحتاج تعديل: الكرسي مرتفع جداً بالنسبة لطولك، اخفضه بمقدار 5-10 سم"
        elif 160 <= height <= 175:
            current_results["chair_height"] = "✅ مناسب" if random.random() > 0.3 else "⚠️ يحتاج تعديل: الكرسي مرتفع قليلاً، اخفضه بمقدار 2-3 سم"
        else:  # height > 175
            current_results["chair_height"] = "⚠️ يحتاج تعديل: الكرسي منخفض جداً بالنسبة لطولك، ارفعه بمقدار 5-10 سم"
    else:
        current_results["chair_height"] = "✅ مناسب" if random.random() > 0.4 else "⚠️ يحتاج تعديل: الكرسي مرتفع جداً، اخفضه قليلاً"
    
    # Check for improvements from previous assessment
    improved_results = {}
    for item, current_status in current_results.items():
        previous_status = state.previous_desk_assessment.get(item)
        if previous_status and "⚠️" in previous_status and "✅" in current_status:
            # Previously warned, now fixed
            improved_results[item] = "✅ مناسب - أحسنت! تم التصحيح"
        else:
            # Keep current status (either still good or still needs fixing)
            improved_results[item] = current_status
    
    # Update previous assessment for next time
    state.previous_desk_assessment = current_results.copy()
    
    return improved_results

@app.post("/api/pause")
async def pause_tracking():
    state.manually_paused = True
    return {"status": "paused"}

@app.post("/api/resume")
async def resume_tracking():
    state.manually_paused = False
    state.idle_popup_shown = False  # Reset idle popup when manually resuming
    state.last_activity_time = time.time()  # Reset activity time
    return {"status": "resumed"}

@app.post("/api/idle-confirm-present")
async def idle_confirm_present():
    state.idle_popup_shown = False
    state.last_activity_time = time.time()  # Reset activity time
    return {"status": "confirmed"}

@app.post("/api/idle-confirm-away")
async def idle_confirm_away():
    state.idle_popup_shown = False
    state.manually_paused = True  # Freeze fatigue when user confirms they're away
    return {"status": "away"}

@app.post("/api/skip_break")
async def skip_break():
    # Reset break popup flag so it can show again in next cycle
    state.break_popup_shown = False
    state.break_active = False
    return {"status": "break_skipped"}

@app.post("/api/demo-posture-check")
async def demo_posture_check():
    """Demo endpoint - manually trigger camera for presentation"""
    if not state.camera_active:
        state.camera_active = True
        state.presence_check_mode = False  # This is a posture check, not presence
        state.camera_start_time = time.time()
        state.needs_camera_verification = True
        print("DEMO: Manual posture check triggered")
    return {"status": "camera_activated"}

@app.post("/api/toggle-persistent-camera")
async def toggle_persistent_camera():
    """Toggle Hackathon Legendary Mode (Always ON)"""
    state.camera_persistent = not state.camera_persistent
    if state.camera_persistent:
        state.camera_active = True
        state.presence_check_mode = False
        state.camera_start_time = time.time()
        print("HACKATHON MODE: Camera is now PERSISTENT ON")
    else:
        state.camera_active = False
        print("HACKATHON MODE: Camera is now NORMAL/OFF")
    return {"status": "persistent_toggled", "persistent": state.camera_persistent}

@app.get("/logout")
async def logout():
    # Reset calibration state for next login
    state.calibration_complete = False
    state.baseline_typing_speed = None
    state.baseline_backspace_rate = None
    state.baseline_mouse_activity = None
    state.fatigue_score = 0.0
    state.break_popup_shown = False
    
    response = RedirectResponse(url="/", status_code=303)
    response.delete_cookie("role")
    response.delete_cookie("username")
    return response

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    role = request.cookies.get("role")
    if role != "employee":
        return RedirectResponse(url="/login?role=employee")
    if not state.calibration_complete:
        return RedirectResponse(url="/calibration")
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/manager", response_class=HTMLResponse)
async def manager_page(request: Request):
    role = request.cookies.get("role")
    if role != "manager":
        return RedirectResponse(url="/login?role=manager")
    # Managers do NOT need calibration
    return templates.TemplateResponse(request=request, name="manager.html")

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            now = time.time()

            # ── Idle / presence state ──────────────────────────────────────────
            has_recent_activity = not bg_tracker.is_idle_5min()
            idle_long = bg_tracker.is_idle_5min()
            if idle_long and not state.idle_popup_shown and not state.manually_paused:
                state.idle_popup_shown = True

            state.fatigue_frozen = state.manually_paused or not state.user_present or not has_recent_activity

            # ── CAMERA SITUATION A: Presence check ────────────────────────────
            # Trigger when NO keyboard/mouse activity for 30 seconds
            # (bg_tracker.is_idle() checks 30-second inactivity window)
            idle_30s = bg_tracker.is_idle()  # True after 30s no keyboard+mouse
            if (idle_30s
                    and not state.camera_active
                    and not state.manually_paused
                    and not state.presence_check_mode
                    and (now - state.last_presence_check) > 30):          # cooldown 30s
                state.camera_active       = True
                state.presence_check_mode = True
                state.camera_start_time   = now
                state.last_presence_check = now
                state.presence_faces_seen = 0
                print("CAMERA A: presence check (30s idle)")

            # ── CAMERA SITUATION B: Behavior check ────────────────────────────
            # Trigger when typing is slower than baseline OR too many backspaces
            # Camera must NOT already be open and cooldown must be respected
            
            # Using our trained Isolation Forest ML model for anomaly detection instead of hardcoded numbers!
            is_anomaly       = bg_tracker.performance_drop
            typing_slow      = is_anomaly 
            backspace_surge  = is_anomaly

            if not state.is_calibrating:
                if is_anomaly:
                    state.suspicious_activity_count += 1
                else:
                    state.suspicious_activity_count = max(0, state.suspicious_activity_count - 1)

            behavior_trigger = (
                state.suspicious_activity_count >= 3
                and not state.camera_active
                and not state.presence_check_mode
                and not state.fatigue_frozen
                and (now - state.camera_last_closed) > 10   # 10s cooldown between checks
            )
            if behavior_trigger:
                state.camera_active             = True
                state.presence_check_mode       = False   # NOT a presence check
                state.camera_start_time         = now
                state.needs_camera_verification = True
                state.suspicious_activity_count = 0       # reset counter
                print(f"CAMERA B: ML Anomaly behavior check triggered!")

            # ── Fatigue from keyboard signals (passive, no camera needed) ──────
            if not state.fatigue_frozen:
                fatigue_increase = 0.0
                if is_anomaly:
                    fatigue_increase += 2.0
                
                if fatigue_increase > 0:
                    state.fatigue_score = min(100.0, state.fatigue_score + fatigue_increase)
                else:
                    state.fatigue_score = max(0.0, state.fatigue_score - 0.15)
                    if state.fatigue_score < 30:
                        state.break_popup_shown = False

            # ── Camera auto-close after 5 seconds ─────────────────────────────
            if state.camera_active and (now - state.camera_start_time) > 5:
                state.camera_active      = False
                state.camera_last_closed = now

                if state.presence_check_mode:
                    # --- Situation A result ---
                    if state.presence_faces_seen > 0:
                        if not state.user_present:
                            state.presence_message = '▶ مرحباً بعودتك — استُؤنف القياس'
                        state.user_present = True
                    else:
                        if state.user_present:
                            state.presence_message = '⏸ لم يتم اكتشاف وجهك — القياس متوقف'
                        state.user_present = False
                    state.presence_faces_seen = 0
                    state.presence_check_mode = False
                    print(f"CAMERA A result: user_present={state.user_present}")

                elif state.needs_camera_verification:
                    # --- Situation B result ---
                    state.needs_camera_verification = False
                    bad_posture = "⚠️" in state.current_posture_status

                    if bad_posture:
                        # Neck/posture warning already set in process_frame
                        state.fatigue_score = min(100.0, state.fatigue_score + 5)

                    # Suggest break based on fatigue level (never based on score alone)
                    if not state.break_popup_shown:
                        if state.fatigue_score >= 30:   # only suggest if meaningful fatigue
                            state.break_popup_shown = True
                            if state.fatigue_score < 60:
                                state.break_duration = 120    # 🟡 2 min
                            elif state.fatigue_score < 80:
                                state.break_duration = 300    # 🟠 5 min
                            else:
                                state.break_duration = 480    # 🔴 8-10 min
                            state.break_active     = True
                            state.break_start_time = now
                    print(f"CAMERA B result: posture={'bad' if bad_posture else 'ok'}, fatigue={state.fatigue_score:.1f}")

            # ── History (single append per tick) ──────────────────────────────
            state.fatigue_history.append(float(round(state.fatigue_score, 1)))
            state.fatigue_history.pop(0)
            state.latency_history.append(int(bg_tracker.avg_lat))
            state.latency_history.pop(0)

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
                    "blinks": state.blinks,
                    "active": state.camera_active
                },
                "breaks": {
                    "taken": state.breaks_taken_today,
                    "skipped": state.breaks_skipped_today,
                    "active": state.break_active,
                    "duration": state.break_duration,
                    "popup_shown": state.break_popup_shown
                },
                "daily_stats": {
                    "fatigue_average": round(state.daily_fatigue_average, 1)
                },
                "calibration_complete": state.calibration_complete,
                "idle": idle_30s,
                "idle_check": idle_long,
                "presence": {
                    "user_present": state.user_present,
                    "manually_paused": state.manually_paused,
                    "fatigue_frozen": state.fatigue_frozen,
                    "idle_popup_shown": state.idle_popup_shown,
                    "presence_check_mode": state.presence_check_mode,
                    "message": state.presence_message if state.presence_message else ""
                }
            }

            if state.presence_message:
                state.presence_message = ""

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
    state.fatigue_score = 0.0
    bg_tracker.key_latencies.clear()
    bg_tracker.avg_lat = 0
    bg_tracker.backspaces = 0
    state.break_popup_shown = False  # Reset for next cycle
    state.break_active = False
    state.breaks_taken_today += 1
    return {"message": "Fatigue reset"}

@app.post("/api/skip_break")
async def skip_break():
    state.breaks_skipped_today += 1
    return {"message": "Break skipped"}

@app.get("/api/end_of_day_report")
async def end_of_day_report():
    total_breaks = state.breaks_taken_today + state.breaks_skipped_today
    completion_rate = (state.breaks_taken_today / total_breaks * 100) if total_breaks > 0 else 0
    
    # Mock comparison (in real app, this would be stored historically)
    improvement = "اليوم كنت أفضل من أمس بنسبة 30%" if state.daily_fatigue_average < 50 else "اليوم كان أكثر إرهاقاً من أمس بنسبة 15%"
    
    return {
        "breaks_taken": state.breaks_taken_today,
        "breaks_skipped": state.breaks_skipped_today,
        "fatigue_average": round(state.daily_fatigue_average, 1),
        "completion_rate": round(completion_rate, 1),
        "comparison": improvement
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)