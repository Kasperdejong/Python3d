import warnings
warnings.filterwarnings("ignore")

# --- SILENCE THE NOISE ---
import sys
import traceback

_original_print_exception = traceback.print_exception

def silent_ssl_print_exception(etype, value, tb, limit=None, file=None, chain=True):
    err_str = str(value)
    if "SSLV3_ALERT_CERTIFICATE_UNKNOWN" in err_str: return
    if "HTTP_REQUEST" in err_str: return
    if "Removing descriptor" in err_str: return
    _original_print_exception(etype, value, tb, limit=limit, file=file, chain=chain)

traceback.print_exception = silent_ssl_print_exception

class CleanStderr:
    def __init__(self, stream): self.stream = stream
    def write(self, data):
        if "Removing descriptor" in data: return
        self.stream.write(data)
    def flush(self): self.stream.flush()

sys.stderr = CleanStderr(sys.stderr)
# --- END SILENCER ---

import eventlet
eventlet.monkey_patch()

import cv2
import mediapipe as mp
import numpy as np
import base64
import socket
import signal
import os
import math
import random
import json
import requests
from flask import Flask, render_template_string
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet', ping_timeout=60)

# --- IMAGE OVERLAY FUNCTION ---
def overlay_image_alpha(img, img_overlay, x, y, width, height):
    try:
        if width <= 0 or height <= 0: return img
        img_overlay = cv2.resize(img_overlay, (width, height))
        
        y1 = y - height
        y2 = y
        x1 = x - width // 2
        x2 = x + width // 2

        h, w, c = img.shape
        if y1 < 0: y1 = 0
        if y2 > h: y2 = h
        if x1 < 0: x1 = 0
        if x2 > w: x2 = w
        
        overlay_h = y2 - y1
        overlay_w = x2 - x1
        if overlay_h <= 0 or overlay_w <= 0: return img

        small_overlay = img_overlay[0:overlay_h, 0:overlay_w]
        
        if small_overlay.shape[2] == 4:
            alpha_mask = small_overlay[:, :, 3] / 255.0
            alpha_inv = 1.0 - alpha_mask
            for c in range(0, 3):
                img[y1:y2, x1:x2, c] = (alpha_mask * small_overlay[:, :, c] + 
                                        alpha_inv * img[y1:y2, x1:x2, c])
        else:
            img[y1:y2, x1:x2] = small_overlay

        return img
    except Exception:
        return img

# --- GESTURE LOGIC ---
def is_hand_open(landmarks):
    wrist = landmarks[0]
    fingers = [(8, 6), (12, 10), (16, 14), (20, 18)]
    open_count = 0
    for tip_idx, pip_idx in fingers:
        tip = landmarks[tip_idx]
        pip = landmarks[pip_idx]
        dist_tip = (tip.x - wrist.x)**2 + (tip.y - wrist.y)**2
        dist_pip = (pip.x - wrist.x)**2 + (pip.y - wrist.y)**2
        if dist_tip > dist_pip:
            open_count += 1
    return open_count >= 3

# --- PLANT SYSTEM ---
class PlantSystem:
    def __init__(self, screen_width, screen_height):
        self.w = screen_width
        self.h = screen_height
        self.slot_size = 60
        self.num_slots = screen_width // self.slot_size
        self.plant_heights = np.zeros(self.num_slots, dtype=np.float32)
        self.plant_types = [-1] * self.num_slots
        self.plant_char = np.zeros(self.num_slots, dtype=np.float32)
        self.max_heights = np.random.randint(200, 500, size=self.num_slots)
        self.loaded_images = []
        self.load_plant_data()

    def load_plant_data(self):
        json_path = os.path.join("JSON", "plants.json")
        print(f"🌐 Connecting to Plant Database via {json_path}...")
        try:
            if not os.path.exists(json_path): raise Exception("Missing JSON")
            with open(json_path, 'r', encoding='utf-8') as f: data = json.load(f)
            count = 0; max_load = 50 
            for plant_id, plant_info in data.items():
                if count >= max_load: break
                url = plant_info.get('springimgpng_med')
                if not url: url = plant_info.get('summerimgpng_med')
                if not url: url = plant_info.get('springimgpng_low')
                if url and "http" in url:
                    try:
                        response = requests.get(url, timeout=3)
                        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                        img = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
                        if img is not None:
                            self.loaded_images.append(img)
                            count += 1
                            if count % 10 == 0: print(f"   Downloaded {count} plants...")
                    except Exception: continue
            print(f"✅ Ready! Loaded {len(self.loaded_images)} plant species.")
        except Exception as e:
            print(f"⚠️ Error initializing plants: {e}")
            dummy = np.zeros((100, 100, 4), dtype=np.uint8); dummy[:] = [20, 200, 20, 255]
            self.loaded_images.append(dummy)

    def interact(self, particle):
        slot_idx = int(particle.x // self.slot_size)
        if slot_idx < 0 or slot_idx >= self.num_slots: return False, False

        if particle.element_type == "Water":
            if particle.y >= self.h - 15:
                if self.plant_heights[slot_idx] == 0 and len(self.loaded_images) > 0:
                    self.plant_types[slot_idx] = random.randint(0, len(self.loaded_images)-1)
                    self.plant_char[slot_idx] = 0.0
                
                if self.plant_char[slot_idx] > 0:
                    self.plant_char[slot_idx] -= 0.1
                    if self.plant_char[slot_idx] < 0: self.plant_char[slot_idx] = 0
                else:
                    current_h = self.plant_heights[slot_idx]
                    target_h = self.max_heights[slot_idx]
                    if current_h < target_h: self.plant_heights[slot_idx] += 3.0
                return True, False

        elif particle.element_type == "Fire":
            plant_h = self.plant_heights[slot_idx]
            if plant_h > 0:
                plant_top = self.h - plant_h
                if particle.y > plant_top:
                    if self.plant_char[slot_idx] < 1.0:
                        self.plant_char[slot_idx] += 0.05
                    else:
                        self.plant_heights[slot_idx] -= 8.0 
                        if self.plant_heights[slot_idx] < 0: 
                            self.plant_heights[slot_idx] = 0; self.plant_types[slot_idx] = -1; self.plant_char[slot_idx] = 0
                    return True, True 
        return False, False

    def draw(self, frame):
        cv2.rectangle(frame, (0, self.h - 15), (self.w, self.h), (20, 50, 20), -1)
        cv2.line(frame, (0, self.h - 15), (self.w, self.h - 15), (50, 200, 50), 2)
        for i in range(self.num_slots):
            height = int(self.plant_heights[i])
            img_idx = self.plant_types[i]
            char_level = self.plant_char[i]
            if height > 0 and img_idx != -1:
                x_center = i * self.slot_size + (self.slot_size // 2)
                src_img = self.loaded_images[img_idx]
                orig_h, orig_w = src_img.shape[:2]; aspect = orig_w / orig_h
                draw_w = int(height * aspect); draw_h = height
                resized_plant = cv2.resize(src_img, (draw_w, draw_h))
                if char_level > 0:
                    plant_float = resized_plant.astype(np.float32)
                    darkness_factor = 1.0 - (char_level * 0.8) 
                    plant_float[:, :, :3] *= darkness_factor
                    resized_plant = plant_float.astype(np.uint8)
                frame = overlay_image_alpha(frame, resized_plant, x_center, self.h - 5, draw_w, draw_h)

# --- PARTICLE CLASS ---
class Particle:
    def __init__(self, x, y, element_type, velocity=None):
        self.x = x
        self.y = y
        self.element_type = element_type
        self.life = 1.0 
        
        if self.element_type == "Fire":
            if velocity:
                self.vx = velocity[0] + random.uniform(-2, 2); self.vy = velocity[1] + random.uniform(-2, 2)
                self.decay = 0.04; self.size = random.randint(6, 12)
            else:
                self.vx = random.uniform(-2, 2); self.vy = random.uniform(-4, -9) 
                self.decay = 0.06; self.size = random.randint(4, 10)
        elif self.element_type == "Water":
            self.vx = random.uniform(-0.5, 0.5); self.vy = random.uniform(5, 15)     
            self.size = random.randint(2, 5); self.decay = 0.04
        elif self.element_type == "Ash":
            self.vx = random.uniform(-1, 1); self.vy = random.uniform(-1, -3) 
            self.size = random.randint(2, 5); self.decay = 0.03; self.color = (50, 50, 50)

    def update(self):
        self.life -= self.decay
        
        # --- FIX: Apply Gravity to Water ---
        if self.element_type == "Water":
            self.vy += 0.5 
        
        # --- FIX: ACTUALLY MOVE THE PARTICLE ---
        self.x += self.vx
        self.y += self.vy

    def draw(self, frame):
        if self.life <= 0: return
        ix, iy = int(self.x), int(self.y)

        if self.element_type == "Fire":
            if self.life > 0.7: color = (255, 255, 255) 
            elif self.life > 0.4: color = (0, 165, 255) 
            else: color = (0, 0, 200) 
            cv2.circle(frame, (ix, iy), int(self.size * self.life), color, -1)
        elif self.element_type == "Water":
            tail_y = int(iy - self.vy)
            color = (255, 255, 255) if self.life > 0.6 else (255, 200, 0) 
            cv2.line(frame, (ix, iy), (ix, tail_y), color, self.size)
        elif self.element_type == "Ash":
            col = int(50 + (self.life * 100)) 
            color = (col, col, col)
            cv2.circle(frame, (ix, iy), self.size, color, -1)

# --- VIDEO PROCESSING ---
def background_thread():
    print("Avatar Stream Active (Visualized Hands)")

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, refine_face_landmarks=True)
    hand_connections = mp.solutions.hands.HAND_CONNECTIONS

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    width = 1280; height = 720
    
    garden = PlantSystem(width, height)
    particles = []

    while True:
        success, frame = cap.read()
        if not success: eventlet.sleep(0.1); continue

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        def get_coords(landmark): return int(landmark.x * w), int(landmark.y * h)

        # 1. RIGHT HAND (FIRE)
        if results.right_hand_landmarks:
            landmarks = results.right_hand_landmarks.landmark
            
            # DRAW SKELETON (Yellow/Orange)
            for connection in hand_connections:
                pt1 = get_coords(landmarks[connection[0]])
                pt2 = get_coords(landmarks[connection[1]])
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)

            if is_hand_open(landmarks):
                wrist = landmarks[0]; tip = landmarks[12]
                dx = (tip.x - wrist.x); dy = (tip.y - wrist.y)
                dist = math.sqrt(dx*dx + dy*dy)
                aim_vx = (dx / dist) * 30; aim_vy = (dy / dist) * 30
                
                # Spawn Fire
                for connection in hand_connections:
                    pt1 = get_coords(landmarks[connection[0]]); pt2 = get_coords(landmarks[connection[1]])
                    mid_x = int(pt1[0] + (pt2[0] - pt1[0]) * random.random())
                    mid_y = int(pt1[1] + (pt2[1] - pt1[1]) * random.random())
                    particles.append(Particle(mid_x, mid_y, "Fire", velocity=(aim_vx, aim_vy)))
                for lm in landmarks: particles.append(Particle(*get_coords(lm), "Fire", velocity=(aim_vx, aim_vy)))

        # 2. LEFT HAND (WATER)
        if results.left_hand_landmarks:
            landmarks = results.left_hand_landmarks.landmark
            
            # # DRAW SKELETON (Blue/Cyan)
            # for connection in hand_connections:
            #     pt1 = get_coords(landmarks[connection[0]])
            #     pt2 = get_coords(landmarks[connection[1]])
            #     cv2.line(frame, pt1, pt2, (255, 255, 0), 2)

            if is_hand_open(landmarks):
                for lm in landmarks: 
                    # Spawn Water
                    particles.append(Particle(*get_coords(lm), "Water"))

        # 3. DRAW & UPDATE
        garden.draw(frame)
        frame = cv2.convertScaleAbs(frame, alpha=0.8, beta=-10)
        
        alive_particles = []
        for p in particles:
            p.update()
            absorbed, spawn_ash = garden.interact(p)
            if spawn_ash:
                for _ in range(2): alive_particles.append(Particle(p.x, p.y, "Ash"))
            if not absorbed and p.life > 0:
                p.draw(frame)
                alive_particles.append(p)
        particles = alive_particles

        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        b64_string = base64.b64encode(buffer).decode('utf-8')
        socketio.emit('new_frame', {'image': b64_string})
        eventlet.sleep(0.01)

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close(); return ip
    except: return "127.0.0.1"

@app.route('/')
def index(): return render_template_string(HTML_PAGE)

@socketio.on('connect')
def handle_connect():
    global thread
    if 'thread' not in globals(): thread = socketio.start_background_task(background_thread)

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Online AR Garden</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        html, body { margin: 0; padding: 0; width: 100%; height: 100%; background-color: #000; overflow: hidden; }
        #video-container { position: absolute; top: 0; left: 0; width: 100%; height: 100%; }
        img { width: 100%; height: 100%; object-fit: cover; display: block; }
        .overlay-text { position: absolute; top: 20px; left: 20px; color: rgba(255, 255, 255, 0.7); font-family: sans-serif; font-size: 14px; pointer-events: none; z-index: 10; }
    </style>
</head>
<body>
    <div id="video-container">
        <div class="overlay-text"><b>Left Hand:</b> Water (Heal/Grow)<br><b>Right Hand:</b> Fire (Char/Burn)</div>
        <img id="videoStream" src="" alt="Loading Elements...">
    </div>
    <script>
        const socket = io();
        const img = document.getElementById('videoStream');
        socket.on('new_frame', function(data) { img.src = 'data:image/jpeg;base64,' + data.image; });
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    ip = get_local_ip()
    cert_exists = os.path.exists('cert.pem') and os.path.exists('key.pem')
    ssl_args = {'certfile': 'cert.pem', 'keyfile': 'key.pem'} if cert_exists else {}
    protocol = "https" if cert_exists else "http"
    print(f"🚀 SERVER STARTED at {protocol}://{ip}:5000")
    try: socketio.run(app, host='0.0.0.0', port=5000, **ssl_args)
    except KeyboardInterrupt: os._exit(0)