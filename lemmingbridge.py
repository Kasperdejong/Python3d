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
import math
import time
import random
from flask import Flask, render_template_string
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet')

# --- 1. STABILIZER ---
class SimpleStabilizer:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.prev_landmarks = {} 

    def update(self, hand_index, lm_id, new_x, new_y):
        if hand_index not in self.prev_landmarks: self.prev_landmarks[hand_index] = {}
        if lm_id in self.prev_landmarks[hand_index]:
            prev_x, prev_y = self.prev_landmarks[hand_index][lm_id]
            smooth_x = self.alpha * new_x + (1 - self.alpha) * prev_x
            smooth_y = self.alpha * new_y + (1 - self.alpha) * prev_y
        else:
            smooth_x, smooth_y = new_x, new_y
        self.prev_landmarks[hand_index][lm_id] = (smooth_x, smooth_y)
        return int(smooth_x), int(smooth_y)

# --- MATH ---
def dist(p1, p2): return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

# Get Y coordinate on line at X
def get_line_y(p1, p2, x):
    x1, y1 = p1; x2, y2 = p2
    if abs(x2 - x1) < 0.1: return y1 
    m = (y2 - y1) / (x2 - x1)
    return y1 + m * (x - x1)

# Get X coordinate on line at Y (For walls)
def get_line_x(p1, p2, y):
    x1, y1 = p1; x2, y2 = p2
    if abs(y2 - y1) < 0.1: return x1
    m_inv = (x2 - x1) / (y2 - y1)
    return x1 + m_inv * (y - y1)

# --- PIXEL ART ---
def draw_pixel_lemming(frame, x, y, facing_right, frame_idx, color):
    s = 4; ix, iy = int(x), int(y)
    sprite_stand = [[0,3,3,3,3,3,0,0],[0,3,3,3,3,3,0,0],[0,0,1,1,1,0,0,0],[0,1,1,2,1,1,0,0], 
                    [0,1,1,1,1,1,0,0],[0,0,1,1,1,0,0,0],[0,0,1,0,1,0,0,0],[0,1,1,0,1,1,0,0]]
    sprite_walk =  [[0,3,3,3,3,3,0,0],[0,3,3,3,3,3,0,0],[0,0,1,1,1,0,0,0],[0,1,1,2,1,1,0,0],
                    [0,1,1,1,1,1,0,0],[0,0,1,1,1,0,0,0],[0,0,0,1,0,0,0,0],[0,0,1,1,1,0,0,0]]
    pixels = sprite_walk if (frame_idx % 10 < 5) else sprite_stand
    offset_x, offset_y = -(4 * s), -(8 * s)
    r, g, b = color; hair_col = (0, 200, 0)
    for row in range(8):
        for col in range(8):
            sprite_col = col if facing_right else 7 - col
            val = pixels[row][sprite_col]
            if val == 0: continue
            draw_c = (r,g,b) if val == 1 else ((0,0,0) if val == 2 else hair_col)
            cv2.rectangle(frame, (ix + (col * s) + offset_x, iy + (row * s) + offset_y), 
                                 (ix + (col * s) + offset_x + s, iy + (row * s) + offset_y + s), draw_c, -1)

def draw_dotted_line(img, pt1, pt2, color, thickness=1, gap=15):
    dist_tot = dist(pt1, pt2)
    if dist_tot == 0: return
    x1, y1 = pt1; x2, y2 = pt2
    vx, vy = (x2 - x1) / dist_tot, (y2 - y1) / dist_tot
    curr = 0
    while curr < dist_tot:
        p1_ = (int(x1 + vx * curr), int(y1 + vy * curr))
        p2_ = (int(x1 + vx * min(curr+gap, dist_tot)), int(y1 + vy * min(curr+gap, dist_tot)))
        if int(curr // gap) % 2 == 0: cv2.line(img, p1_, p2_, color, thickness)
        curr += gap

# --- UI ---
class Button:
    def __init__(self, x, y, w, h, text, color):
        self.rect = (x, y, w, h); self.text = text; self.color = color
        self.cooldown = 0; self.hover_state = False
    
    def draw(self, frame):
        x, y, w, h = self.rect
        col = (min(self.color[0]+50,255), min(self.color[1]+50,255), min(self.color[2]+50,255)) if self.hover_state else self.color
        cv2.rectangle(frame, (x, y), (x+w, y+h), col, -1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255,255,255), 2)
        font_scale = 0.7
        (tw, th), _ = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
        cv2.putText(frame, self.text, (x + (w-tw)//2, y + (h+th)//2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), 2)

    def update(self, pointers):
        if self.cooldown > 0: self.cooldown -= 1
        self.hover_state = False; triggered = False
        x, y, w, h = self.rect
        for px, py in pointers:
            if x < px < x+w and y < py < y+h:
                self.hover_state = True
                if self.cooldown == 0: self.cooldown = 30; triggered = True
        return triggered

# --- WALKER (SOLID PHYSICS) ---
class Walker:
    def __init__(self, start_x, start_y):
        self.x, self.y = start_x, start_y
        self.vx = random.uniform(2.2, 3.2)
        self.vy = 0
        self.state = "ALIVE"
        self.frame_count = random.randint(0, 10)
        self.color = (random.randint(50, 255), random.randint(50, 255), 255)
        self.on_ground = False

    def update(self, platforms, w, h):
        if self.state != "ALIVE": return
        self.frame_count += 1

        # --- STEP 1: HORIZONTAL MOVEMENT & WALL CHECKS ---
        
        # INVISIBLE LEFT WALL
        if self.x < 10:
            self.x = 10
            self.vx = abs(self.vx) # Force Right
        
        next_x = self.x + self.vx
        hit_wall = False
        
        for p1, p2 in platforms:
            # Check Slope: Is it steep? (Height > Width)
            # 0.5 factor means if it's > 45 degrees, it's a wall
            dx = abs(p2[0] - p1[0])
            dy = abs(p2[1] - p1[1])
            
            is_steep = dy > dx * 0.8 # It is a wall
            
            if is_steep:
                # Are we vertically within the wall's height?
                min_y, max_y = min(p1[1], p2[1]), max(p1[1], p2[1])
                if self.y > min_y - 10 and self.y < max_y:
                    # Are we hitting it horizontally?
                    wall_x = get_line_x(p1, p2, self.y)
                    
                    # Distance to wall
                    dist_x = abs(next_x - wall_x)
                    
                    if dist_x < 15:
                        # Ensure we are hitting the face, not the back
                        if (self.vx > 0 and self.x < wall_x) or (self.vx < 0 and self.x > wall_x):
                            hit_wall = True
        
        if hit_wall:
            self.vx *= -1 # Bounce
        else:
            self.x = next_x # Apply movement

        # --- STEP 2: VERTICAL MOVEMENT & FLOOR CHECKS ---

        # If we are walking on ground
        if self.on_ground:
            # We treat ground as a state where we just snap Y to the floor
            # But we must check if the floor still exists or if we walked off
            
            found_floor_below = False
            best_y = 99999
            
            for p1, p2 in platforms:
                 # Is it Flat?
                dx = abs(p2[0] - p1[0])
                dy = abs(p2[1] - p1[1])
                is_flat = dx >= dy * 0.8
                
                if is_flat:
                    min_x, max_x = min(p1[0], p2[0]), max(p1[0], p2[0])
                    # Are we within X bounds?
                    if self.x >= min_x - 5 and self.x <= max_x + 5:
                        line_y = get_line_y(p1, p2, self.x)
                        # Are we close to it?
                        if abs(self.y - line_y) < 20:
                            if line_y < best_y:
                                best_y = line_y
                                found_floor_below = True

            if found_floor_below:
                self.y = best_y - 1
                self.vy = 0
            else:
                self.on_ground = False # Walked off edge
        
        # If Falling
        if not self.on_ground:
            self.vy += 0.8
            self.y += self.vy
            
            # Check for landing
            feet_y = self.y
            best_y_dist = 9999
            landed = False
            land_y = 0
            
            for p1, p2 in platforms:
                min_x, max_x = min(p1[0], p2[0]), max(p1[0], p2[0])
                
                if self.x < min_x - 5 or self.x > max_x + 5: continue
                
                # Slope Check
                dx = abs(p2[0] - p1[0])
                dy = abs(p2[1] - p1[1])
                is_flat = dx >= dy * 0.8
                
                if is_flat:
                    line_y = get_line_y(p1, p2, self.x)
                    dist_y = line_y - feet_y # Positive means floor is below feet
                    
                    # Logic: We fall INTO the floor.
                    # Previous frame we were above, now we are below OR we are just close
                    # We accept slightly negative dist_y (tunneling fix)
                    if abs(dist_y) < 20 and self.vy >= 0:
                        if abs(dist_y) < best_y_dist:
                            best_y_dist = abs(dist_y)
                            land_y = line_y
                            landed = True

            if landed:
                self.on_ground = True
                self.y = land_y - 1
                self.vy = 0

        # Bounds
        if self.y > h + 50: self.state = "DEAD"
        if self.x > w - 20: self.state = "SAVED"

    def draw(self, frame):
        if self.state == "ALIVE":
            draw_pixel_lemming(frame, self.x, self.y, self.vx > 0, self.frame_count, self.color)

# --- MAIN ---
def background_thread():
    print("🏆 Lemming Solid Physics Edition")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2)
    stabilizer = SimpleStabilizer(alpha=0.5) 

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280); cap.set(4, 720)
    w, h = 1280, 720
    
    active_walkers = []; built_bridges = []
    
    is_game_active = False; is_game_over = False
    last_spawn = 0; spawn_rate = 1.5
    count_spawn = 0; MAX_LEM = 10
    s_saved = 0; s_died = 0

    btn_go = Button(w//2 - 60, 20, 120, 50, "START", (0, 200, 0))
    btn_rst = Button(w//2 + 80, 20, 120, 50, "CLEAR", (0, 0, 200))
    btn_rpl = Button(w//2 - 100, h//2 + 60, 200, 60, "REPLAY", (0, 150, 0))

    static = [((0, h//2+50), (150, h//2+50)), ((w-150, h//2+50), (w, h//2+50))]
    was_pinching = False 

    while True:
        success, frame = cap.read()
        if not success: eventlet.sleep(0.1); continue
        
        frame = cv2.flip(frame, 1) 
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        ghosts = []
        pointers = []
        is_pinching = False

        if results.multi_hand_landmarks:
            for h_idx, h_lms in enumerate(results.multi_hand_landmarks):
                coords = {}
                for idx in [4, 5, 6, 7, 8]: 
                    rx, ry = h_lms.landmark[idx].x * w, h_lms.landmark[idx].y * h
                    coords[idx] = stabilizer.update(h_idx, idx, rx, ry)
                
                pointers.append(coords[8]) 
                
                if not is_game_over:
                    if dist(coords[4], coords[8]) < 30:
                        is_pinching = True
                        cv2.circle(frame, coords[8], 10, (0,0,255), -1)
                    else:
                        p_start, p_end = coords[5], coords[8]
                        if dist(p_start, p_end) > 15:
                            ghosts.append((p_start, p_end))
                        cv2.circle(frame, coords[8], 5, (0,255,255), -1)

        # Build
        if not is_game_over:
            if is_pinching and not was_pinching:
                if len(ghosts) > 0:
                    built_bridges.extend(ghosts)
                    cv2.rectangle(frame, (0,0), (w,h), (255,255,255), 5) 
            was_pinching = is_pinching
        
        # Game Loop
        if is_game_active and not is_game_over:
            if count_spawn < MAX_LEM:
                if time.time() - last_spawn > spawn_rate:
                    active_walkers.append(Walker(50, h//2))
                    last_spawn = time.time(); count_spawn += 1
            
            if count_spawn == MAX_LEM and len(active_walkers) == 0:
                is_game_active = False; is_game_over = True
        
        # Buttons
        if not is_game_over:
            if btn_go.update(pointers):
                if not is_game_active:
                    is_game_active = True; count_spawn = 1
                    active_walkers.append(Walker(50, h//2)); last_spawn = time.time()
            if btn_rst.update(pointers):
                built_bridges = []; active_walkers = []
                is_game_active = False; count_spawn = 0
                s_saved = 0; s_died = 0
        else:
            if btn_rpl.update(pointers):
                built_bridges = []; active_walkers = []
                is_game_active = False; is_game_over = False
                count_spawn = 0; s_saved = 0; s_died = 0

        # Draw
        for p1, p2 in static: cv2.rectangle(frame, (int(p1[0]),int(p1[1])), (int(p2[0]),int(p1[1])+20), (50,150,50), -1)
        for p1, p2 in built_bridges: cv2.line(frame, p1, p2, (180,180,180), 8)

        if not is_pinching and not is_game_over:
            for p1, p2 in ghosts: draw_dotted_line(frame, p1, p2, (0,255,255), 2)

        for walker in active_walkers[:]:
            walker.update(static + built_bridges, w, h)
            walker.draw(frame)
            if walker.state == "SAVED": s_saved += 1; active_walkers.remove(walker)
            elif walker.state == "DEAD": s_died += 1; active_walkers.remove(walker)

        if not is_game_over:
            btn_go.draw(frame); btn_rst.draw(frame)
            cv2.putText(frame, f"SAVED: {s_saved}  DIED: {s_died}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        else:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0,0,0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            cv2.rectangle(frame, (w//2-150, h//2-100), (w//2+150, h//2+100), (50,50,50), -1)
            cv2.rectangle(frame, (w//2-150, h//2-100), (w//2+150, h//2+100), (255,255,255), 3)
            cv2.putText(frame, f"SAVED: {s_saved}", (w//2-60, h//2-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, f"DIED: {s_died}", (w//2-60, h//2+20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            btn_rpl.draw(frame)
            for px, py in pointers: cv2.circle(frame, (px, py), 8, (255,255,0), -1)

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
    <img id="vid" style="width:100%;height:100%;object-fit:contain; image-rendering: pixelated;">
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
<script>
const s = io();
const i = document.getElementById('vid');
s.on('new_frame', d => i.src = 'data:image/jpeg;base64,' + d.image);
</script>
</body>
"""

if __name__ == "__main__":
    print(f"Server: http://{get_ip()}:5000")
    try: socketio.run(app, host='0.0.0.0', port=5000)
    except: os._exit(0)