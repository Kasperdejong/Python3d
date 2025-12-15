import warnings
warnings.filterwarnings("ignore")

import sys
import traceback
import os  

_original_print_exception = traceback.print_exception
def silent_ssl_print_exception(etype, value, tb, limit=None, file=None, chain=True):
    err_str = str(value)
    if "SSLV3_ALERT_CERTIFICATE_UNKNOWN" in err_str: return
    if "HTTP_REQUEST" in err_str: return
    if "Removing descriptor" in err_str: return
    _original_print_exception(etype, value, tb, limit=limit, file=file, chain=chain)
traceback.print_exception = silent_ssl_print_exception

import eventlet
eventlet.monkey_patch()

import cv2
import mediapipe as mp
import numpy as np
import base64
import socket
import math
import random
import requests
from flask import Flask, render_template_string
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet', ping_timeout=60)

# --- CONFIGURATION ---
TREE_IMAGE_URL = "https://cms.pollinator.art/media/pages/plants/echium-pininana/5eec736c79-1730985047/73-echium-pininana-250cm-a-1-241107-1024x1024-q80.png"

# --- UTILS ---
def overlay_image_alpha(img, img_overlay, x, y, width, height):
    try:
        if width <= 0 or height <= 0: return img
        img_overlay = cv2.resize(img_overlay, (width, height))
        
        y1 = y - height
        y2 = y
        x1 = x - (width // 2)
        x2 = x + (width // 2)

        h, w, c = img.shape
        
        if y1 < 0: 
            img_overlay = img_overlay[abs(y1):, :, :]
            y1 = 0
        if y2 > h: y2 = h
        if x1 < 0: 
            img_overlay = img_overlay[:, abs(x1):, :]
            x1 = 0
        if x2 > w: 
            img_overlay = img_overlay[:, :-(x2-w), :]
            x2 = w

        overlay_h, overlay_w = img_overlay.shape[:2]
        if overlay_h == 0 or overlay_w == 0: return img

        small_overlay = img_overlay
        if small_overlay.shape[2] == 4:
            alpha_mask = small_overlay[:, :, 3] / 255.0
            alpha_inv = 1.0 - alpha_mask
            for c in range(0, 3):
                img[y1:y1+overlay_h, x1:x1+overlay_w, c] = (
                    alpha_mask * small_overlay[:, :, c] + 
                    alpha_inv * img[y1:y1+overlay_h, x1:x1+overlay_w, c]
                )
        else:
            img[y1:y1+overlay_h, x1:x1+overlay_w] = small_overlay[:, :, :3]
        return img
    except Exception:
        return img

# --- GAME LOGIC ---
class TreeGame:
    def __init__(self):
        self.tree_img = None
        self.score = 0
        self.growth_factor = 0.6 
        self.max_growth = 3.5
        self.tree_bbox = None 
        self.download_tree()

    def download_tree(self):
        print(f"🌲 Downloading Echium...")
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(TREE_IMAGE_URL, headers=headers, timeout=5)
            if response.status_code == 200:
                image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
                self.tree_img = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
                print("✅ Tree Ready!")
            else:
                self.create_dummy_tree()
        except:
            self.create_dummy_tree()

    def create_dummy_tree(self):
        self.tree_img = np.zeros((300, 100, 4), dtype=np.uint8)
        self.tree_img[:] = [50, 200, 50, 255] 

    def grow(self):
        self.score += 1
        if self.growth_factor < self.max_growth:
            self.growth_factor += 0.05 

    def draw_tree(self, frame, root_x, root_y):
        if self.tree_img is None: return

        orig_h, orig_w = self.tree_img.shape[:2]
        
        target_h = int(frame.shape[0] * 0.45 * self.growth_factor)
        aspect_ratio = orig_w / orig_h
        target_w = int(target_h * aspect_ratio)

        # Draw anchored at Navel
        overlay_image_alpha(frame, self.tree_img, root_x, root_y, target_w, target_h)
        
        # Calculate Hitbox
        hitbox_w = int(target_w * 0.6)
        x1 = root_x - (hitbox_w // 2)
        x2 = root_x + (hitbox_w // 2)
        y1 = root_y - target_h
        y2 = root_y
        
        self.tree_bbox = (x1, y1, x2, y2)
        
        # Draw HUD
        cv2.putText(frame, f"Biomass: {int(self.growth_factor * 100)} kg", (root_x + 80, root_y - 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)

class SunOrb:
    def __init__(self, screen_w):
        self.x = random.randint(20, screen_w - 20)
        self.y = -50
        self.size = random.randint(12, 22)
        self.speed = random.uniform(6, 14)
        self.active = True

    def update(self):
        self.y += self.speed

    def draw(self, frame):
        if not self.active: return
        cv2.circle(frame, (int(self.x), int(self.y)), self.size + 5, (0, 255, 255), -1) 
        cv2.circle(frame, (int(self.x), int(self.y)), self.size - 2, (255, 255, 255), -1) 

# --- MAIN LOOP ---
def background_thread():
    print("🚀 SERVER STARTED.")

    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    game = TreeGame()
    suns = []
    
    while True:
        success, frame = cap.read()
        if not success: 
            eventlet.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        # Darken bg
        frame = cv2.convertScaleAbs(frame, alpha=0.65, beta=-30)

        # 1. SPAWN SUNS
        if random.random() < 0.1: 
            suns.append(SunOrb(w))

        # 2. TRACK BODY
        collision_zones = [] 
        navel_coords = None

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            
            def get_pt(idx):
                if lm[idx].visibility > 0.5:
                    return (int(lm[idx].x * w), int(lm[idx].y * h))
                return None

            l_shldr = get_pt(11); r_shldr = get_pt(12)
            l_hip = get_pt(23); r_hip = get_pt(24)
            l_elbow = get_pt(13); r_elbow = get_pt(14)
            l_wrist = get_pt(15); r_wrist = get_pt(16)
            nose = get_pt(0)

            # --- CALCULATE NAVEL ---
            if l_shldr and r_shldr and l_hip and r_hip:
                mid_shldr_x = (l_shldr[0] + r_shldr[0]) // 2
                mid_shldr_y = (l_shldr[1] + r_shldr[1]) // 2
                mid_hip_x = (l_hip[0] + r_hip[0]) // 2
                mid_hip_y = (l_hip[1] + r_hip[1]) // 2
                
                navel_x = (mid_shldr_x + mid_hip_x) // 2
                navel_y = (mid_shldr_y + mid_hip_y) // 2
                navel_coords = (navel_x, navel_y)

                # Draw Tree
                game.draw_tree(frame, navel_x, navel_y)

                # --- BODY COLLISION LOGIC ---
                branch_color = (50, 200, 50)
                branch_thick = int(6 * game.growth_factor) 

                # 1. HEAD (Huge Hitbox)
                if nose:
                    collision_zones.append({'pos': nose, 'r': 80}) 
                    cv2.circle(frame, nose, 80, (255, 255, 255), 1) 

                # 2. LEFT ARM
                if l_shldr and l_elbow:
                    cv2.line(frame, navel_coords, l_shldr, branch_color, branch_thick)
                    cv2.line(frame, l_shldr, l_elbow, branch_color, branch_thick)
                    collision_zones.append({'pos': l_shldr, 'r': 70})
                    collision_zones.append({'pos': l_elbow, 'r': 70})
                    if l_wrist:
                        cv2.line(frame, l_elbow, l_wrist, branch_color, branch_thick)
                        cv2.circle(frame, l_wrist, 20, (0, 255, 0), -1)
                        collision_zones.append({'pos': l_wrist, 'r': 70})

                # 3. RIGHT ARM
                if r_shldr and r_elbow:
                    cv2.line(frame, navel_coords, r_shldr, branch_color, branch_thick)
                    cv2.line(frame, r_shldr, r_elbow, branch_color, branch_thick)
                    collision_zones.append({'pos': r_shldr, 'r': 70})
                    collision_zones.append({'pos': r_elbow, 'r': 70})
                    if r_wrist:
                        cv2.line(frame, r_elbow, r_wrist, branch_color, branch_thick)
                        cv2.circle(frame, r_wrist, 20, (0, 255, 0), -1)
                        collision_zones.append({'pos': r_wrist, 'r': 70})

        # 3. UPDATE SUNS & CHECK ALL HITBOXES
        active_suns = []
        for sun in suns:
            sun.update()
            
            caught = False
            
            # A. Check Tree Trunk Collision
            if game.tree_bbox:
                tx1, ty1, tx2, ty2 = game.tree_bbox
                if tx1 < sun.x < tx2 and ty1 < sun.y < ty2:
                    caught = True
                    cv2.circle(frame, (int(sun.x), int(sun.y)), 40, (0, 255, 0), 2)

            # B. Check Body Collision
            if not caught:
                for zone in collision_zones:
                    pt = zone['pos']
                    radius = zone['r']
                    dist = math.hypot(sun.x - pt[0], sun.y - pt[1])
                    
                    if dist < radius:
                        caught = True
                        color = (255, 255, 255)
                        cv2.circle(frame, pt, radius, color, 2) 
                        break

            if caught:
                game.grow()
            elif sun.y < h:
                sun.draw(frame)
                active_suns.append(sun)
        
        suns = active_suns

        # 4. ENCODE
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
    <title>Echium Full Body</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        html, body { margin: 0; padding: 0; width: 100%; height: 100%; background-color: #111; overflow: hidden; font-family: 'Arial', sans-serif; }
        #video-container { position: absolute; top: 0; left: 0; width: 100%; height: 100%; display: flex; justify-content: center; align-items: center; }
        img { width: 100%; height: 100%; object-fit: cover; display: block; }
        .overlay { 
            position: absolute; bottom: 30px; left: 50%; transform: translateX(-50%);
            color: #fff; text-align: center; text-shadow: 2px 2px 4px #000;
            background: rgba(0, 50, 0, 0.6); padding: 20px; border-radius: 15px; border: 2px solid #4CAF50;
        }
        h1 { margin: 0 0 10px 0; color: #ADFF2F; text-transform: uppercase; letter-spacing: 2px; }
        p { margin: 5px 0; font-size: 18px; }
    </style>
</head>
<body>
    <div id="video-container">
        <img id="videoStream" src="" alt="Waiting for camera...">
    </div>
    <div class="overlay">
        <h1>Full Body Photosynthesis</h1>
        <p>1. Step back until your hips are visible.</p>
        <p>2. Use your <b>Head (Big Hitbox!), Body, and Arms</b> to catch sun.</p>
        <p>3. Let the tree absorb the light!</p>
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
    print(f"🚀 GAME RUNNING AT: {protocol}://{ip}:5000")
    try: socketio.run(app, host='0.0.0.0', port=5000, **ssl_args)
    except KeyboardInterrupt: os._exit(0)