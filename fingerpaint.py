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
import math
from flask import Flask, render_template_string
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet')

# config
MAX_HANDS = 4  
DRAW_THRESHOLD = 80 

# UI
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

# stickyhand logic
class StickyHand:
    def __init__(self, id, start_color):
        self.id = id
        self.x = 0
        self.y = 0
        self.prev_x = 0
        self.prev_y = 0
        self.color = start_color
        self.active = False 
        self.landmarks = None

    def update(self, new_x, new_y, landmarks):
        self.x = new_x
        self.y = new_y
        self.landmarks = landmarks
        self.active = True

    def reset_draw_pos(self):
        # Call this when hand reappears so we don't draw a line from 0,0
        self.prev_x = self.x
        self.prev_y = self.y

    def lost(self):
        self.active = False
        self.prev_x = 0
        self.prev_y = 0

def solve_hand_assignment(slots, detections):
    """
    Greedy Distance Matcher (Fixed for Drawing)
    """
    # NOTE: We DO NOT reset 'active' here anymore. 
    # We only mark them lost if they are not assigned at the end.

    possible_matches = []
    
    # 1. Calculate Distances
    for det_idx, det in enumerate(detections):
        det_x, det_y = det['x'], det['y']
        
        for slot_idx, slot in enumerate(slots):
            # If slot is inactive (0,0), give it lower priority but still allow match
            if slot.x == 0 and slot.y == 0:
                dist = 9999
            else:
                dist = math.hypot(det_x - slot.x, det_y - slot.y)
            
            possible_matches.append((dist, slot_idx, det_idx))

    # 2. Sort by distance (closest first)
    possible_matches.sort(key=lambda x: x[0])

    assigned_slots = set()
    assigned_detections = set()

    # 3. Assign
    for dist, slot_idx, det_idx in possible_matches:
        if slot_idx in assigned_slots or det_idx in assigned_detections:
            continue
        
        det = detections[det_idx]
        slot = slots[slot_idx]
        
        # If this slot was NOT active previously, reset the drawing line
        # so we don't draw a laser beam from 0,0
        if not slot.active:
            slot.x = det['x']
            slot.y = det['y']
            slot.reset_draw_pos()

        slot.update(det['x'], det['y'], det['lms'])
        
        assigned_slots.add(slot_idx)
        assigned_detections.add(det_idx)

    # 4. Handle Lost Hands
    # Only mark as inactive if they were NOT assigned this frame
    for i, slot in enumerate(slots):
        if i not in assigned_slots:
            slot.lost()

# --- MAIN LOOP ---
def background_thread():
    print(f"🎨 Multi-User Paint Active ({MAX_HANDS} Hands)")
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=MAX_HANDS)

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280); cap.set(4, 720)
    w, h = 1280, 720

    img_canvas = np.zeros((h, w, 3), np.uint8)
    header = ColorHeader(w)
    
    default_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
    hand_slots = [StickyHand(i, default_colors[i]) for i in range(4)]
    
    while True:
        success, frame = cap.read()
        if not success: eventlet.sleep(0.1); continue

        frame = cv2.flip(frame, 1)
        # Brighten
        frame = cv2.convertScaleAbs(frame, alpha=1.1, beta=10)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)
        
        detections = []
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                wrist_x = int(hand_lms.landmark[0].x * w)
                wrist_y = int(hand_lms.landmark[0].y * h)
                lm_list = []
                for lm in hand_lms.landmark:
                    lm_list.append((int(lm.x * w), int(lm.y * h)))
                detections.append({'x': wrist_x, 'y': wrist_y, 'lms': lm_list})

        # Run Sticky Logic
        solve_hand_assignment(hand_slots, detections)

        # Process Active Hands
        for hand in hand_slots:
            if not hand.active:
                continue 
            
            lm_list = hand.landmarks
            x1, y1 = lm_list[8]  # Index Tip
            x2, y2 = lm_list[12] # Middle Tip
            
            index_up = lm_list[8][1] < lm_list[6][1]
            middle_up = lm_list[12][1] < lm_list[10][1]

            # A. SELECTION MODE (Two fingers up)
            if index_up and middle_up:
                hand.reset_draw_pos() 
                cv2.rectangle(frame, (x1-25, y1-25), (x2+25, y1+25), hand.color, 2)
                cv2.putText(frame, "SELECT", (x1, y1-35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hand.color, 2)
                
                new_col = header.get_color_at(x1, y1)
                if new_col is not None:
                    hand.color = new_col

            # B. DRAW MODE (Index only)
            elif index_up and not middle_up:
                cv2.circle(frame, (x1, y1), 10, hand.color, -1)
                
                # Logic: If prev is 0 (just appeared), set to current
                if hand.prev_x == 0 and hand.prev_y == 0:
                    hand.prev_x, hand.prev_y = x1, y1
                
                dist = math.hypot(x1 - hand.prev_x, y1 - hand.prev_y)
                
                # Only draw if distance is valid (not a teleport)
                if dist < DRAW_THRESHOLD:
                    color = hand.color
                    thickness = 40 if color == (0,0,0) else 15
                    # DRAWING HAPPENS HERE
                    cv2.line(img_canvas, (hand.prev_x, hand.prev_y), (x1, y1), color, thickness)
                else:
                    hand.reset_draw_pos()

                # Update Previous for next frame
                hand.prev_x, hand.prev_y = x1, y1

            # C. IDLE
            else:
                hand.reset_draw_pos()

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
        
        for hand in hand_slots:
            if hand.active and hand.landmarks:
                ix, iy = hand.landmarks[8] 
                if ix > w-100 and iy > h-50:
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
        <h2 style="margin:0 0 10px 0;">🎨 4-Player Paint</h2>
        👉 <b>Index Finger:</b> Draw<br>
        ✌️ <b>Two Fingers:</b> Select Color<br>
        🧠 <b>Memory:</b> Hands remember their colors!
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