import warnings
warnings.filterwarnings("ignore")

# --- SILENCE NOISE ---
import sys, traceback, logging
log = logging.getLogger('werkzeug'); log.setLevel(logging.ERROR)
def silent_ssl(etype, value, tb, limit=None, file=None, chain=True):
    if "SSLV3" in str(value) or "HTTP_REQUEST" in str(value): return
    traceback.print_exception(etype, value, tb, limit, file, chain)
traceback.print_exception = silent_ssl
# --- END SILENCE ---

import eventlet
eventlet.monkey_patch()

import cv2
import mediapipe as mp
import numpy as np
import base64
import socket
import os
import math
import time
from flask import Flask, render_template_string
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet')

# --- SMART IMAGE OVERLAY ---
def overlay_image_alpha(img, img_overlay, x, y, size=None):
    try:
        if size is not None:
            old_h, old_w = img_overlay.shape[:2]
            aspect_ratio = old_w / old_h
            new_w = size
            new_h = int(new_w / aspect_ratio)
            img_overlay = cv2.resize(img_overlay, (new_w, new_h))

        h_ov, w_ov, c_ov = img_overlay.shape
        h, w, c = img.shape

        x1 = int(x - w_ov // 2)
        y1 = int(y - h_ov // 2)
        x2 = x1 + w_ov
        y2 = y1 + h_ov

        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        if x2 > w: x2 = w
        if y2 > h: y2 = h
        
        w_ov = x2 - x1
        h_ov = y2 - y1
        if w_ov <= 0 or h_ov <= 0: return img

        ov_x_start = 0
        ov_y_start = 0
        if x - (img_overlay.shape[1] // 2) < 0: ov_x_start = abs(int(x - (img_overlay.shape[1] // 2)))
        if y - (img_overlay.shape[0] // 2) < 0: ov_y_start = abs(int(y - (img_overlay.shape[0] // 2)))

        small_overlay = img_overlay[ov_y_start:ov_y_start+h_ov, ov_x_start:ov_x_start+w_ov]
        bg_roi = img[y1:y2, x1:x2]

        if small_overlay.shape[2] == 4:
            alpha = small_overlay[:, :, 3] / 255.0
            alpha_inv = 1.0 - alpha
            for c in range(0, 3):
                bg_roi[:, :, c] = (alpha * small_overlay[:, :, c] +
                                   alpha_inv * bg_roi[:, :, c])
            img[y1:y2, x1:x2] = bg_roi
        else:
            img[y1:y2, x1:x2] = small_overlay[:, :, :3]

        return img
    except Exception as e:
        return img

# --- SPRITE MANAGER ---
class SpritePuppet:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.state = "NEUTRAL"
        self.sprites = {}
        self.dance_frames = []
        self.frame_index = 0
        self.last_anim_update = time.time()
        
        # ZOOM: 3.0 means 100px image -> 300px on screen
        self.zoom = 4.0
        
        self.load_assets()

    def create_dummy_texture(self, color, text):
        img = np.zeros((150, 150, 4), dtype=np.uint8)
        cv2.circle(img, (75, 75), 70, color, -1) 
        cv2.putText(img, text, (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)
        return img

    def load_assets(self):
        folder = "assets"
        if not os.path.exists(folder): os.makedirs(folder)
        
        def load(name, color, fallback_text):
            path = os.path.join(folder, name)
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is not None: return img
            print(f"⚠️ Missing: {name}")
            return self.create_dummy_texture(color, fallback_text)

        # Static States
        self.sprites['NEUTRAL'] = load("handpuppet_idle.png", (100, 100, 100, 255), ":|")
        self.sprites['HAPPY']   = load("handpuppet_happy.png",   (0, 200, 200, 255),   ":D")
        self.sprites['ANGRY']   = load("handpuppet_angry.png",   (0, 0, 200, 255),     ">:(")
        self.sprites['SHY']     = load("handpuppet_shy.png",     (200, 100, 200, 255), "O_o")
        
        # --- LOAD FRAME SEQUENCE ---
        print("💃 Loading dance frames...")
        for i in range(20): 
            fname = f"pixel_dance_{i}.png"
            path = os.path.join(folder, fname)
            
            if os.path.exists(path):
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is not None:
                    self.dance_frames.append(img)
            else:
                if i > 0: break 
        
        if len(self.dance_frames) > 0:
            print(f"✅ Loaded {len(self.dance_frames)} dance frames!")
        else:
            print("⚠️ No 'pixel_dance_X.png' frames found. Using Happy sprite for dance.")
            self.dance_frames.append(self.sprites['HAPPY'])

    def update(self, target_x, target_y, gesture_state):
        # Physics
        self.x += (target_x - self.x) * 0.2
        self.y += (target_y - self.y) * 0.2
        self.state = gesture_state

        if self.state == "ANGRY":
            self.x += np.random.randint(-5, 5)
            self.y += np.random.randint(-5, 5)

    def draw(self, frame):
        current_img = self.sprites.get(self.state, self.sprites['NEUTRAL'])

        if self.state == "DANCING" and self.dance_frames:
            if time.time() - self.last_anim_update > 0.1:
                self.frame_index = (self.frame_index + 1) % len(self.dance_frames)
                self.last_anim_update = time.time()
            current_img = self.dance_frames[self.frame_index]
            draw_y = self.y + math.sin(time.time() * 15) * 20
        else:
            draw_y = self.y

        h, w = current_img.shape[:2]
        new_width = int(w * self.zoom)

        frame = overlay_image_alpha(frame, current_img, int(self.x), int(draw_y), size=new_width)
        return frame

# --- GESTURE LOGIC ---
def get_finger_status(lm_list):
    fingers = []
    tips = [8, 12, 16, 20]; pips = [6, 10, 14, 18]
    for i in range(4):
        fingers.append(lm_list[tips[i]][2] < lm_list[pips[i]][2])
    return fingers 

def detect_gesture(lm_list):
    x1, y1 = lm_list[4][1], lm_list[4][2] # Thumb
    x2, y2 = lm_list[8][1], lm_list[8][2] # Index
    dist_pinch = math.hypot(x2 - x1, y2 - y1)
    fingers_up = get_finger_status(lm_list)
    count_up = fingers_up.count(True)

    if count_up == 0: return "ANGRY"     # Fist
    if dist_pinch < 60: return "SHY"     # Pinch
    if fingers_up[0] and not fingers_up[1] and not fingers_up[2] and not fingers_up[3]: return "DANCING" # Point
    if count_up >= 3: return "HAPPY"     # Open Hand
    return "NEUTRAL"

# --- MAIN LOOP ---
puppet = SpritePuppet()

def background_thread():
    print("👾 Frame Sequence Puppet Active")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280); cap.set(4, 720)

    while True:
        success, frame = cap.read()
        if not success: eventlet.sleep(0.1); continue

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        gesture = "NEUTRAL"
        target_x, target_y = puppet.x, puppet.y 

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                wrist = hand_lms.landmark[0]
                target_x = int(wrist.x * w)
                target_y = int(wrist.y * h) - 120

                lm_list = []
                for id, lm in enumerate(hand_lms.landmark):
                    lm_list.append([id, int(lm.x * w), int(lm.y * h)])
                
                gesture = detect_gesture(lm_list)

        # Update and Draw Puppet
        puppet.update(target_x, target_y, gesture)
        frame = puppet.draw(frame)

        # --- DRAW ON-SCREEN MENU (TOP LEFT) ---
        # Dark Background Rectangle for text
        cv2.rectangle(frame, (10, 10), (250, 170), (0, 0, 0), -1)
        # Text
        ui_color = (255, 255, 255)
        cv2.putText(frame, "GESTURES:",       (20, 40),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "Open  : Happy",   (20, 70),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, ui_color, 2)
        cv2.putText(frame, "Fist  : Angry",   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ui_color, 2)
        cv2.putText(frame, "Pinch : Shy",     (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ui_color, 2)
        cv2.putText(frame, "Point : Dance!",  (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ui_color, 2)

        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        b64 = base64.b64encode(buffer).decode('utf-8')
        socketio.emit('new_frame', {'image': b64})
        eventlet.sleep(0.015)

def get_ip():
    try: s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("8.8.8.8", 80)); return s.getsockname()[0]
    except: return "127.0.0.1"

@app.route('/')
def index(): return render_template_string(HTML)

@socketio.on('connect')
def connect(): socketio.start_background_task(background_thread)

HTML = """
<body style="margin:0;background:#222;display:flex;justify-content:center;height:100vh;overflow:hidden;font-family:sans-serif">
<div style="position:relative;width:100%;height:100%">
    <img id="vid" style="width:100%;height:100%;object-fit:contain">
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
<script>
const s = io(), i = document.getElementById('vid');
s.on('new_frame', d => i.src = 'data:image/jpeg;base64,' + d.image);
</script>
</body>
"""

if __name__ == "__main__":
    print(f"Server: http://{get_ip()}:5000")
    try: socketio.run(app, host='0.0.0.0', port=5000)
    except: os._exit(0)