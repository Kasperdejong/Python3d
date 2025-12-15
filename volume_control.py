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
import subprocess
import threading
import time
from flask import Flask, render_template_string
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet')

# --- SYSTEM CONTROL MANAGER ---
class SystemControlManager:
    def __init__(self):
        self.target_vol = 50
        self.target_bright = 0.75 
        self.running = True
        
        # Start background workers
        threading.Thread(target=self._volume_worker, daemon=True).start()
        threading.Thread(target=self._brightness_worker, daemon=True).start()

    def set_volume(self, val):
        self.target_vol = int(max(0, min(100, round(val))))

    def set_brightness(self, val):
        self.target_bright = max(0.0, min(1.0, val / 100.0))

    def _volume_worker(self):
        last_vol = -1
        while self.running:
            # HYSTERESIS: Only update if changed by at least 2 steps (prevents 49-50-49 spam)
            # OR if it's 0 or 100 (exact endpoints)
            diff = abs(self.target_vol - last_vol)
            if diff >= 2 or (diff > 0 and (self.target_vol in [0, 100])):
                try:
                    subprocess.run(f"osascript -e 'set volume output volume {self.target_vol}'", 
                                   shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    last_vol = self.target_vol
                except: pass
            time.sleep(0.05)

    def _brightness_worker(self):
        last_bright = -1.0
        while self.running:
            # Only update if change is significant (> 1%)
            if abs(self.target_bright - last_bright) > 0.015:
                try:
                    subprocess.run(f"brightness {self.target_bright}", 
                                   shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    last_bright = self.target_bright
                except: pass
            time.sleep(0.1) 

sys_manager = SystemControlManager()

# --- PROCESSING ---
def background_thread():
    print("🚀 Stable Gesture System Active")
    
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    # High confidence decreases noise
    hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8, max_num_hands=2)

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280); cap.set(4, 720)

    bar_top, bar_bottom = 150, 600

    start_bright_h = np.interp(75, [0, 100], [bar_bottom, bar_top])
    start_vol_h = np.interp(50, [0, 100], [bar_bottom, bar_top])

    # State: 'display_val' is the smoothed number we show on screen
    state = [
        {'val': 75.0, 'display_val': 75, 'bar': start_bright_h, 'pinched': False, 'offset': 0, 'color': (0, 255, 255)}, 
        {'val': 50.0, 'display_val': 50, 'bar': start_vol_h, 'pinched': False, 'offset': 0, 'color': (0, 255, 0)}    
    ]
    
    while True:
        success, frame = cap.read()
        if not success: eventlet.sleep(0.1); continue

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        cv2.line(frame, (w//2, 0), (w//2, h), (50, 50, 50), 2)
        cv2.putText(frame, "BRIGHTNESS", (w//4 - 80, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
        cv2.putText(frame, "VOLUME", (3*w//4 - 60, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
                
                lm_x = [int(lm.x * w) for lm in hand_lms.landmark]
                lm_y = [int(lm.y * h) for lm in hand_lms.landmark]
                
                if len(lm_x) < 9: continue

                x1, y1 = lm_x[4], lm_y[4]
                x2, y2 = lm_x[8], lm_y[8]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                side = 0 if cx < w // 2 else 1
                s = state[side]

                length = math.hypot(x2 - x1, y2 - y1)
                is_pinching = length < 50

                col = s['color']
                cv2.circle(frame, (cx, cy), 10, (255, 0, 255), -1)
                
                if is_pinching:
                    cv2.circle(frame, (cx, cy), 15, col, -1)
                    
                    if not s['pinched']:
                        s['offset'] = s['bar'] - cy
                        s['pinched'] = True
                    
                    target_y = cy + s['offset']
                    target_y = max(bar_top, min(bar_bottom, target_y))
                    
                    # FIX 1: Heavy Smoothing (0.1 instead of 0.3)
                    # This makes it ignore quick jitter
                    s['bar'] = (0.9 * s['bar']) + (0.1 * target_y)
                    
                    # Calculate exact percentage float
                    pct = np.interp(s['bar'], [bar_top, bar_bottom], [100, 0])
                    s['val'] = pct

                    # FIX 2: Stable Display Value (Deadzone)
                    # Only update the stored integer if it changed by > 0.5 to stop 49.9 <-> 50.0 flip
                    if abs(pct - s['display_val']) > 0.8:
                        s['display_val'] = int(round(pct))

                    if side == 0: sys_manager.set_brightness(s['val'])
                    else: sys_manager.set_volume(s['val'])
                else:
                    s['pinched'] = False
                    cv2.circle(frame, (cx, cy), 15, (50, 50, 255), -1)

        for i in range(2):
            s = state[i]
            x_base = (w // 4) if i == 0 else (3 * w // 4)
            cv2.rectangle(frame, (x_base - 20, bar_top), (x_base + 20, bar_bottom), (40, 40, 40), 3)
            fill_h = int(s['bar'])
            cv2.rectangle(frame, (x_base - 20, fill_h), (x_base + 20, bar_bottom), s['color'], -1)
            # Use the stabilized integer for text
            cv2.putText(frame, f"{s['display_val']}%", (x_base - 30, bar_bottom + 40), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
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
<body style="margin:0;background:#111;display:flex;justify-content:center;height:100vh;overflow:hidden">
<img id="vid" style="height:100%;width:auto;object-fit:contain">
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