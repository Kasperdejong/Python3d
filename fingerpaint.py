import warnings
warnings.filterwarnings("ignore")

import sys, traceback, logging
log = logging.getLogger('werkzeug'); log.setLevel(logging.ERROR)
def silent_ssl(etype, value, tb, limit=None, file=None, chain=True):
    if "SSLV3" in str(value) or "HTTP_REQUEST" in str(value): return
    traceback.print_exception(etype, value, tb, limit, file, chain)
traceback.print_exception = silent_ssl

import eventlet
eventlet.monkey_patch()

import cv2
import mediapipe as mp
import numpy as np
import base64
import socket
import os
import random
import math  # Added for distance calculation
from flask import Flask, render_template_string
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet')

# --- CONFIGURATION ---
DETECT_CONF = 0.5 
TRACK_CONF = 0.5
MAX_HANDS = 4  
DRAW_THRESHOLD = 100 # If hand moves >100px in 1 frame, don't draw (prevents laser beams)

# --- UI ELEMENTS ---
class ColorHeader:
    def __init__(self, w):
        self.colors = [
            ((0, 0, 255), "Red"),
            ((0, 255, 0), "Green"),
            ((255, 0, 0), "Blue"),
            ((0, 255, 255), "Yellow"),
            ((255, 0, 255), "Purple"),
            ((255, 100, 0), "Orange"),
            ((0, 0, 0), "Eraser")
        ]
        self.w = w
        self.h = 80
        self.btn_w = w // len(self.colors)

    def draw(self, frame):
        cv2.rectangle(frame, (0, 0), (self.w, self.h), (40, 40, 40), -1)
        for i, (color, name) in enumerate(self.colors):
            x = i * self.btn_w
            if name == "Eraser":
                cv2.rectangle(frame, (x, 0), (x + self.btn_w, self.h), (255, 255, 255), -1)
                cv2.putText(frame, "ERASE", (x + 10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            else:
                cv2.rectangle(frame, (x, 0), (x + self.btn_w, self.h), color, -1)
                cv2.rectangle(frame, (x, 0), (x + self.btn_w, self.h), (200,200,200), 1)
        return frame

    def get_color_at(self, x, y):
        if y < self.h:
            idx = int(x // self.btn_w)
            if 0 <= idx < len(self.colors):
                return self.colors[idx][0]
        return None

# --- STATE MANAGER ---
class HandStateManager:
    def __init__(self):
        self.states = {}

    def get_state(self, hand_index):
        if hand_index not in self.states:
            # Assign colors based on index for variety
            start_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
            c = start_colors[hand_index % len(start_colors)]
            self.states[hand_index] = {
                'prev_x': 0, 
                'prev_y': 0, 
                'color': c
            }
        return self.states[hand_index]

    def reset_pos(self, hand_index):
        if hand_index in self.states:
            self.states[hand_index]['prev_x'] = 0
            self.states[hand_index]['prev_y'] = 0

# --- MAIN LOOP ---
def background_thread():
    print(f"🎨 Multi-User Paint Active ({MAX_HANDS} Hands)")
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        min_detection_confidence=DETECT_CONF, 
        min_tracking_confidence=TRACK_CONF, 
        max_num_hands=MAX_HANDS
    )

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280); cap.set(4, 720)
    w, h = 1280, 720

    img_canvas = np.zeros((h, w, 3), np.uint8)
    header = ColorHeader(w)
    state_manager = HandStateManager()
    
    while True:
        success, frame = cap.read()
        if not success: eventlet.sleep(0.1); continue

        frame = cv2.flip(frame, 1)
        # Brighten slightly
        frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        seen_indices = set()

        if results.multi_hand_landmarks:
            for h_idx, hand_lms in enumerate(results.multi_hand_landmarks):
                seen_indices.add(h_idx)
                hand_data = state_manager.get_state(h_idx)

                lm_list = []
                for id, lm in enumerate(hand_lms.landmark):
                    lm_list.append((int(lm.x * w), int(lm.y * h)))

                if len(lm_list) != 0:
                    x1, y1 = lm_list[8]  # Index Tip
                    x2, y2 = lm_list[12] # Middle Tip
                    
                    index_up = lm_list[8][1] < lm_list[6][1]
                    middle_up = lm_list[12][1] < lm_list[10][1]

                    # 1. SELECTION MODE (Two fingers)
                    if index_up and middle_up:
                        state_manager.reset_pos(h_idx)
                        cv2.rectangle(frame, (x1-20, y1-20), (x2+20, y1+20), hand_data['color'], 2)
                        cv2.putText(frame, "SELECT", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand_data['color'], 2)
                        
                        new_col = header.get_color_at(x1, y1)
                        if new_col is not None:
                            hand_data['color'] = new_col

                    # 2. DRAW MODE (Index only)
                    elif index_up and not middle_up:
                        cv2.circle(frame, (x1, y1), 10, hand_data['color'], -1)
                        
                        px, py = hand_data['prev_x'], hand_data['prev_y']
                        
                        # Initialize if new
                        if px == 0 and py == 0:
                            px, py = x1, y1
                        
                        # --- FIX: DISTANCE CHECK ---
                        # Calculate distance moved in this 1 frame
                        dist = math.hypot(x1 - px, y1 - py)
                        
                        # Only draw if distance is reasonable (prevents cross-screen laser beams)
                        if dist < DRAW_THRESHOLD:
                            color = hand_data['color']
                            thickness = 40 if color == (0,0,0) else 15
                            cv2.line(img_canvas, (px, py), (x1, y1), color, thickness)
                        
                        # Always update position
                        hand_data['prev_x'], hand_data['prev_y'] = x1, y1

                    else:
                        state_manager.reset_pos(h_idx)

        # Reset unseen hands
        for existing_idx in state_manager.states:
            if existing_idx not in seen_indices:
                state_manager.reset_pos(existing_idx)

        # Merge
        img_gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, img_inv)
        frame = cv2.bitwise_or(frame, img_canvas)

        header.draw(frame)
        
        # Clear Button
        cv2.rectangle(frame, (w-100, h-50), (w, h), (0,0,200), -1)
        cv2.putText(frame, "CLEAR", (w-90, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                if hand_lms.landmark[8].x * w > w-100 and hand_lms.landmark[8].y * h > h-50:
                     img_canvas = np.zeros((h, w, 3), np.uint8)

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
<body style="margin:0;background:#111;display:flex;justify-content:center;height:100vh;overflow:hidden;font-family:sans-serif">
<div style="position:relative;width:100%;height:100%">
    <img id="vid" style="width:100%;height:100%;object-fit:contain">
    <div style="position:absolute; bottom:20px; left:20px; color:white; background:rgba(0,0,0,0.6); padding:15px; border-radius:10px;">
        <h2 style="margin:0 0 10px 0;">🎨 Multi-User Paint</h2>
        👉 <b>Index Finger:</b> Draw<br>
        ✌️ <b>Two Fingers:</b> Select Color<br>
        🖐 <b>Supports up to 4 people!</b>
    </div>
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