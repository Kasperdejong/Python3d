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
import socket
import os
import math
import time
from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet')

# --- LOGIC CLASS ---
class PuppetLogic:
    def __init__(self, start_x):
        self.x = start_x
        self.y = 0.5
        self.state = "NEUTRAL"
        self.last_seen = time.time() # Keeps track of when we last saw a hand

    def update(self, target_x_norm, target_y_norm, gesture_state):
        # Physics Smoothing
        self.x += (target_x_norm - self.x) * 0.2
        self.y += (target_y_norm - self.y) * 0.2
        self.state = gesture_state
        self.last_seen = time.time() # Reset the timer

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
    
    w_x, w_y = lm_list[0][1], lm_list[0][2]
    m_x, m_y = lm_list[9][1], lm_list[9][2]
    hand_size = math.hypot(m_x - w_x, m_y - w_y)

    fingers = get_finger_status(lm_list)
    count = fingers.count(True)

    if count == 0: return "ANGRY"
    if dist_pinch < (hand_size * 0.20): return "SHY"
    if fingers[0] and not any(fingers[1:]): return "DANCING"
    if count >= 3: return "HAPPY"
    return "NEUTRAL"

# --- SMART TRACKING ALGORITHM ---
def assign_hands_to_puppets(puppets, new_hands):
    if len(new_hands) == 0: return 

    if len(new_hands) == 1:
        hand = new_hands[0]
        d0 = math.hypot(hand['x'] - puppets[0].x, hand['y'] - puppets[0].y)
        d1 = math.hypot(hand['x'] - puppets[1].x, hand['y'] - puppets[1].y)
        target = puppets[0] if d0 < d1 else puppets[1]
        target.update(hand['x'], hand['y'], hand['gesture'])

    elif len(new_hands) >= 2:
        h1 = new_hands[0]
        h2 = new_hands[1]
        
        # Calculate straight match vs crossed match
        dist_straight = math.hypot(h1['x'] - puppets[0].x, h1['y'] - puppets[0].y) + \
                        math.hypot(h2['x'] - puppets[1].x, h2['y'] - puppets[1].y)
                 
        dist_cross = math.hypot(h1['x'] - puppets[1].x, h1['y'] - puppets[1].y) + \
                     math.hypot(h2['x'] - puppets[0].x, h2['y'] - puppets[0].y)

        if dist_straight < dist_cross:
            puppets[0].update(h1['x'], h1['y'], h1['gesture'])
            puppets[1].update(h2['x'], h2['y'], h2['gesture'])
        else:
            puppets[1].update(h1['x'], h1['y'], h1['gesture'])
            puppets[0].update(h2['x'], h2['y'], h2['gesture'])

# --- MAIN LOOP ---
puppets = [PuppetLogic(0.25), PuppetLogic(0.75)]

def background_thread():
    print("🚀 2-Player Smart-Tracking Stream Active")
    hands = mp.solutions.hands.Hands(min_detection_confidence=0.7, max_num_hands=2)
    cap = cv2.VideoCapture(0)
    cap.set(3, 640) 
    cap.set(4, 480)

    while True:
        success, frame = cap.read()
        if not success: eventlet.sleep(0.1); continue

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        detected_hands = []

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                wrist = hand_lms.landmark[0]
                tx = wrist.x
                ty = wrist.y - 0.2 
                lm_list = [[id, int(lm.x * w), int(lm.y * h)] for id, lm in enumerate(hand_lms.landmark)]
                
                gesture = detect_gesture(lm_list)
                detected_hands.append({'x': tx, 'y': ty, 'gesture': gesture})

        # 1. Update positions based on sticky logic
        assign_hands_to_puppets(puppets, detected_hands)

        # 2. CHECK TIMEOUTS (The Idle Fix)
        # If a puppet hasn't been updated in 0.5 seconds, reset state
        current_time = time.time()
        for p in puppets:
            if current_time - p.last_seen > 0.5:
                p.state = "NEUTRAL"

        # 3. Prepare Data
        players_data = []
        for i, p in enumerate(puppets):
            players_data.append({
                'id': i,
                'x': p.x,
                'y': p.y,
                'state': p.state
            })

        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        byte_data = buffer.tobytes()

        socketio.emit('puppet_data', players_data)
        socketio.emit('video_frame', byte_data)
        
        eventlet.sleep(0.015)

# --- FLASK SETUP ---
def get_ip():
    try: s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("8.8.8.8", 80)); return s.getsockname()[0]
    except: return "127.0.0.1"

@app.route('/')
def index(): return render_template('puppetindex.html')

@app.route('/assets/<path:path>')
def send_assets(path):
    return send_from_directory('assets', path)

@socketio.on('connect')
def connect(): socketio.start_background_task(background_thread)

if __name__ == "__main__":
    ip = get_ip()
    print(f"Server: https://{ip}:5000") 
    try: socketio.run(app, host='0.0.0.0', port=5000, certfile='cert.pem', keyfile='key.pem')
    except: os._exit(0)