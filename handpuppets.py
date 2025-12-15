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
import time
# Added send_from_directory to serve images to the browser
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet')

# --- LOGIC ONLY (No Drawing) ---
class PuppetLogic:
    def __init__(self):
        self.x = 0.5
        self.y = 0.5
        self.state = "NEUTRAL"

    def update(self, target_x_norm, target_y_norm, gesture_state):
        # Physics Smoothing
        self.x += (target_x_norm - self.x) * 0.2
        self.y += (target_y_norm - self.y) * 0.2
        self.state = gesture_state

        if self.state == "ANGRY":
            self.x += np.random.uniform(-0.01, 0.01)
            self.y += np.random.uniform(-0.01, 0.01)

# --- GESTURE MATH ---
def get_finger_status(lm_list):
    fingers = []
    tips = [8, 12, 16, 20]; pips = [6, 10, 14, 18]
    for i in range(4): fingers.append(lm_list[tips[i]][2] < lm_list[pips[i]][2])
    return fingers 

def detect_gesture(lm_list):
    x1, y1 = lm_list[4][1], lm_list[4][2]
    x2, y2 = lm_list[8][1], lm_list[8][2]
    dist_pinch = math.hypot(x2 - x1, y2 - y1)
    fingers = get_finger_status(lm_list)
    count = fingers.count(True)

    if count == 0: return "ANGRY"
    if dist_pinch < 60: return "SHY"
    if fingers[0] and not any(fingers[1:]): return "DANCING"
    if count >= 3: return "HAPPY"
    return "NEUTRAL"

# --- MAIN LOOP ---
puppet_logic = PuppetLogic()

def background_thread():
    print("Hybrid Stream Active: Sending Video + Data")
    hands = mp.solutions.hands.Hands(min_detection_confidence=0.7, max_num_hands=1)
    cap = cv2.VideoCapture(0)
    cap.set(3, 640) # Lower res for speed, browser scales it up
    cap.set(4, 480)

    while True:
        success, frame = cap.read()
        if not success: eventlet.sleep(0.1); continue

        # 1. Process Logic
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        gesture = "NEUTRAL"
        tx, ty = puppet_logic.x, puppet_logic.y

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                wrist = hand_lms.landmark[0]
                # Normalize coordinates (0.0 to 1.0) so they work on any screen size
                tx = wrist.x
                ty = wrist.y - 0.2 # Offset up
                
                lm_list = [[id, int(lm.x * w), int(lm.y * h)] for id, lm in enumerate(hand_lms.landmark)]
                gesture = detect_gesture(lm_list)

        puppet_logic.update(tx, ty, gesture)

        # 2. Prepare Data Packet
        data_packet = {
            'x': puppet_logic.x,
            'y': puppet_logic.y,
            'state': puppet_logic.state
        }
        
        # 3. Prepare Video Packet (Raw Background)
        # We compress this more because it's just a background now
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        b64_video = base64.b64encode(buffer).decode('utf-8')

        # 4. Emit Both
        socketio.emit('puppet_data', data_packet)
        socketio.emit('video_frame', {'image': b64_video})
        
        eventlet.sleep(0.015)

# --- FLASK SETUP ---
def get_ip():
    try: s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("8.8.8.8", 80)); return s.getsockname()[0]
    except: return "127.0.0.1"

@app.route('/')
def index(): return render_template('puppetindex.html')

# NEW: This allows the HTML to load images from your assets folder
@app.route('/assets/<path:path>')
def send_assets(path):
    return send_from_directory('assets', path)

@socketio.on('connect')
def connect(): socketio.start_background_task(background_thread)

if __name__ == "__main__":
    ip = get_ip()
    print(f"Server: https://{ip}:5000") 
    
    # Add ssl_context here
    try: socketio.run(app, host='0.0.0.0', port=5000, certfile='cert.pem', keyfile='key.pem')
    except: os._exit(0)