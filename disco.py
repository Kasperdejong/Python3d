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
import os
import time
import random
from flask import Flask, render_template_string
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet', ping_timeout=60)

# ==========================================
# 🎛️ SETTINGS - CHANGE THIS TO TOGGLE DARKNESS
# ==========================================
DARK_MODE = True  # Set to False for normal brightness
# ==========================================

# --- DISCO LOGIC CLASSES ---

class DiscoEcho:
    def __init__(self, contours, color):
        self.contours = contours
        self.color = color      
        self.life = 1.0         
        self.decay = 0.015      

    def update(self):
        self.life -= self.decay

    def is_alive(self):
        return self.life > 0

    def draw_on_layer(self, layer):
        intensity = max(0.0, self.life)
        faded_color = (
            int(self.color[0] * intensity),
            int(self.color[1] * intensity),
            int(self.color[2] * intensity)
        )
        cv2.drawContours(layer, self.contours, -1, faded_color, -1)

def get_random_neon_color():
    hue = random.randint(0, 179)
    hsv_color = np.uint8([[[hue, 200, 255]]])
    bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
    return (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2]))

# --- VIDEO PROCESSING ---
def background_thread():
    print(f"🕺 DISCO MODE STARTED. Dark Mode is: {DARK_MODE}")

    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    segmenter = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    echo_trails = [] 
    last_spawn_time = time.time()
    spawn_interval = 0.5 
    
    active_color = get_random_neon_color()

    while True:
        success, frame = cap.read()
        if not success: 
            eventlet.sleep(0.1)
            continue

        frame = cv2.flip(frame, 1)

        # --- KEY FIX: PROCESS AI ON ORIGINAL IMAGE ---
        # We process the frame BEFORE darkening it. 
        # This gives MediaPipe maximum detail for tracking.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = segmenter.process(rgb_frame)

        # --- APPLY DARKNESS EFFECT (OPTIONAL) ---
        # We only darken the frame NOW, after the AI has already looked at it.
        if DARK_MODE:
            frame = cv2.convertScaleAbs(frame, alpha=0.6, beta=-50)

        # 1. Analyze Frame
        binary_mask = None
        current_contours = []
        
        if results.segmentation_mask is not None:
            binary_mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
            
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                current_contours = contours
                
                current_time = time.time()
                if current_time - last_spawn_time > spawn_interval:
                    active_color = get_random_neon_color()
                    echo_trails.append(DiscoEcho(contours, active_color))
                    last_spawn_time = current_time

        # 2. Draw Echoes onto a separate layer
        glow_layer = np.zeros_like(frame)
        
        alive_echoes = []
        for echo in echo_trails:
            echo.update()
            if echo.is_alive():
                echo.draw_on_layer(glow_layer)
                alive_echoes.append(echo)
        echo_trails = alive_echoes

        # 3. MASKING (Behind User)
        if binary_mask is not None:
            mask_inv = cv2.bitwise_not(binary_mask)
            glow_layer_behind = cv2.bitwise_and(glow_layer, glow_layer, mask=mask_inv)
            frame = cv2.add(frame, glow_layer_behind)
        else:
            frame = cv2.add(frame, glow_layer)

        # 4. Draw Live Outline
        if current_contours:
            cv2.drawContours(frame, current_contours, -1, active_color, 3)

        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 65])
        b64_string = base64.b64encode(buffer).decode('utf-8')
        socketio.emit('new_frame', {'image': b64_string})
        
        eventlet.sleep(0.01)

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except: return "127.0.0.1"

@app.route('/')
def index(): return render_template_string(HTML_PAGE)

@socketio.on('connect')
def handle_connect():
    global thread
    if 'thread' not in globals(): 
        thread = socketio.start_background_task(background_thread)

# HTML Client
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>FULL SCREEN DISCO</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        html, body { 
            margin: 0; 
            padding: 0; 
            width: 100%; 
            height: 100%; 
            background-color: #000; 
            overflow: hidden; 
        }
        #video-container { 
            position: absolute;
            top: 0; left: 0;
            width: 100vw; height: 100vh;
            z-index: 1;
        }
        img { 
            width: 100%; height: 100%; 
            object-fit: cover; 
            display: block; 
        }
        .overlay-text { 
            position: absolute; 
            bottom: 30px; 
            left: 50%;
            transform: translateX(-50%);
            color: rgba(255, 255, 255, 0.8); 
            font-family: sans-serif; 
            font-size: 20px; 
            letter-spacing: 2px;
            pointer-events: none; 
            z-index: 10; 
            text-shadow: 0 0 15px #ff00ff, 0 0 30px #00ffff;
            text-transform: uppercase;
        }
    </style>
</head>
<body>
    <div id="video-container">
        <img id="videoStream" src="" alt="">
    </div>
    <div class="overlay-text">Move to create afterimages!</div>
    
    <script>
        const socket = io();
        const img = document.getElementById('videoStream');
        socket.on('new_frame', function(data) { 
            img.src = 'data:image/jpeg;base64,' + data.image; 
        });
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
    try: 
        socketio.run(app, host='0.0.0.0', port=5000, **ssl_args)
    except KeyboardInterrupt: 
        os._exit(0)