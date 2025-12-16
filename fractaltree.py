import warnings
warnings.filterwarnings("ignore")

import sys, traceback, os, time, math, random
import eventlet
eventlet.monkey_patch()

import cv2
import mediapipe as mp
import numpy as np
import base64
import socket
from flask import Flask, render_template_string
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet')

# --- CONFIG ---
BARK_COLOR = (19, 69, 139)      # Dark Brown
HIGHLIGHT_COLOR = (40, 90, 160) # Light Brown
LEAF_COLOR = (50, 180, 50)      # Green
LEAF_SHADOW = (30, 120, 30)     # Darker Green
FLOWER_COLORS = [
    (200, 150, 255), # Pink
    (255, 200, 200), # Pale White
    (255, 100, 200)  # Magenta
]

# --- MATH UTILS ---
def get_interp_point(p1, p2, t):
    return (int(p1[0] + (p2[0]-p1[0])*t), int(p1[1] + (p2[1]-p1[1])*t))

def point_line_distance(px, py, p1, p2):
    x1, y1 = p1; x2, y2 = p2
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy == 0: return math.hypot(px - x1, py - y1)
    t = ((px - x1) * dx + (py - y1) * dy) / (dx*dx + dy*dy)
    t = max(0, min(1, t)) 
    cx = x1 + t * dx
    cy = y1 + t * dy
    return math.hypot(px - cx, py - cy)

# --- RENDERING ENGINE ---

def draw_foliage_cluster(frame, x, y, density, seed_val):
    """Draws leaves/flowers based on density (0.0 - 1.0)"""
    # Seed random with position so leaves stick to the arm as it moves
    random.seed(seed_val)
    
    # Draw Leaves
    count = int(3 + density * 5)
    for _ in range(count):
        ox = random.randint(-15, 15)
        oy = random.randint(-15, 15)
        scale = random.randint(3, 6)
        color = LEAF_COLOR if random.random() > 0.3 else LEAF_SHADOW
        cv2.circle(frame, (x+ox, y+oy), scale, color, -1)
    
    # Draw Flowers (Only if density is high)
    if density > 0.6:
        flower_count = 1 if density < 0.8 else random.randint(1, 3)
        for _ in range(flower_count):
            ox = random.randint(-10, 10)
            oy = random.randint(-10, 10)
            f_col = random.choice(FLOWER_COLORS)
            cv2.circle(frame, (x+ox, y+oy), 5, f_col, -1)
            cv2.circle(frame, (x+ox, y+oy), 2, (255, 255, 255), -1)

def draw_thick_trunk(frame, l_shldr, r_shldr, l_hip, r_hip, growth_pct):
    """Draws the main body with bark and growing moss/vines"""
    # 1. Base Trunk
    pts = np.array([l_shldr, r_shldr, r_hip, l_hip], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(frame, [pts], BARK_COLOR)
    
    # 2. Bark Texture
    for i in range(1, 5):
        t = i / 5
        top = get_interp_point(l_shldr, r_shldr, t)
        bot = get_interp_point(l_hip, r_hip, t)
        
        # Add slight curve
        mid_x = (top[0] + bot[0]) // 2 + random.randint(-5, 5)
        mid_y = (top[1] + bot[1]) // 2
        
        # Draw curved bark lines
        cv2.line(frame, top, (mid_x, mid_y), HIGHLIGHT_COLOR, 2)
        cv2.line(frame, (mid_x, mid_y), bot, HIGHLIGHT_COLOR, 2)

    # 3. Body Foliage (Moss on chest/torso)
    if growth_pct > 0.2:
        # Interpolate points along the spine/chest
        steps = 4
        for i in range(1, steps):
            t = i / steps
            # Left side
            lp = get_interp_point(l_shldr, l_hip, t)
            draw_foliage_cluster(frame, lp[0], lp[1], growth_pct, lp[0]*lp[1])
            # Right side
            rp = get_interp_point(r_shldr, r_hip, t)
            draw_foliage_cluster(frame, rp[0], rp[1], growth_pct, rp[0]*rp[1])

def draw_tapered_limb(frame, start, end, start_thick, end_thick):
    """Draws the wooden branch part"""
    vx, vy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(vx, vy)
    if length < 1: return
    
    nx, ny = -vy/length, vx/length
    
    p1 = (int(start[0] + nx * start_thick/2), int(start[1] + ny * start_thick/2))
    p2 = (int(start[0] - nx * start_thick/2), int(start[1] - ny * start_thick/2))
    p3 = (int(end[0] - nx * end_thick/2), int(end[1] - ny * end_thick/2))
    p4 = (int(end[0] + nx * end_thick/2), int(end[1] + ny * end_thick/2))
    
    pts = np.array([p1, p2, p3, p4], np.int32)
    cv2.fillPoly(frame, [pts], BARK_COLOR)

def draw_twig_decoration(frame, start, end, growth_pct):
    """Adds small sticks and DYNAMIC leaves/flowers"""
    vx, vy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(vx, vy)
    if length < 20: return

    # Number of twigs increases slightly with length
    count = 3
    
    for i in range(1, count + 1):
        t = i / (count + 1)
        base = get_interp_point(start, end, t)
        
        # Deterministic random based on position (prevents flickering)
        seed = base[0] * base[1]
        random.seed(seed)
        
        angle_offset = random.uniform(0.3, 0.8)
        direction = 1 if i % 2 == 0 else -1
        
        # Twig Vector
        tx = int(base[0] + (vy * angle_offset * direction))
        ty = int(base[1] - (vx * angle_offset * direction))
        
        # Draw Twig Wood
        cv2.line(frame, base, (tx, ty), BARK_COLOR, 3)
        
        # Draw Foliage at tip of twig
        if growth_pct > 0.3:
            draw_foliage_cluster(frame, tx, ty, growth_pct, seed)

# --- GAME LOGIC ---
class GameState:
    def __init__(self):
        self.xp = 0.0
        self.max_xp = 200.0 # Increased max XP so you can enjoy the blooming
        self.won = False
        self.win_timer = 0
    
    def absorb(self):
        if self.won: return
        self.xp += 2.0
        if self.xp >= self.max_xp:
            self.xp = self.max_xp
            self.won = True
            self.win_timer = time.time()
    
    def reset(self):
        self.xp = 0
        self.won = False

class SunOrb:
    def __init__(self, w):
        self.x = random.randint(50, w-50)
        self.y = -50
        self.r = random.randint(15, 20)
        self.speed = random.uniform(5, 9)
        self.active = True
    def update(self): self.y += self.speed
    def draw(self, frame):
        if not self.active: return
        # Glowing Orb
        cv2.circle(frame, (int(self.x), int(self.y)), self.r+5, (100,255,255), -1)
        cv2.circle(frame, (int(self.x), int(self.y)), self.r, (255,255,255), -1)

# --- MAIN SERVER ---
def background_thread():
    print("🌸 DRYAD BLOOM ENGINE STARTED")
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1)
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280); cap.set(4, 720)
    
    game = GameState()
    suns = []
    
    while True:
        success, frame = cap.read()
        if not success: eventlet.sleep(0.1); continue
        
        frame = cv2.flip(frame, 1)
        # Darken slightly to make flowers pop
        frame = cv2.convertScaleAbs(frame, alpha=0.85, beta=-15)
        h, w = frame.shape[:2]
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        
        active_hitboxes = [] 
        
        # Percentage 0.0 to 1.0
        growth_pct = game.xp / game.max_xp
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            def pt(idx): 
                if lm[idx].visibility > 0.5: return (int(lm[idx].x*w), int(lm[idx].y*h))
                return None
            
            nose = pt(0)
            l_shldr, r_shldr = pt(11), pt(12)
            l_elbow, r_elbow = pt(13), pt(14)
            l_wrist, r_wrist = pt(15), pt(16)
            l_hip, r_hip = pt(23), pt(24)
            
            if l_shldr and r_shldr and l_hip and r_hip:
                # 1. DRAW TRUNK & ROOTS
                root_l = (l_hip[0] - 15, h)
                root_r = (r_hip[0] + 15, h)
                draw_tapered_limb(frame, l_hip, root_l, 25, 10)
                draw_tapered_limb(frame, r_hip, root_r, 25, 10)
                
                # Main Body
                draw_thick_trunk(frame, l_shldr, r_shldr, l_hip, r_hip, growth_pct)
                
                # Head Connection & Crown
                mid_shldr = ((l_shldr[0]+r_shldr[0])//2, (l_shldr[1]+r_shldr[1])//2)
                if nose:
                    draw_tapered_limb(frame, mid_shldr, nose, 30, 15)
                    # Crown grows with XP
                    crown_size = 20 + int(30 * growth_pct)
                    draw_foliage_cluster(frame, nose[0], nose[1] - 10, growth_pct + 0.2, nose[0])
                    active_hitboxes.append((mid_shldr, nose, 40))

            # 2. DRAW ARMS
            # Growth stages logic
            limb_reach = 0.2 + (growth_pct * 0.8) # Min 20%, Max 100%
            upper_growth = min(1.0, limb_reach * 2)
            lower_growth = max(0.0, (limb_reach - 0.5) * 2)

            def process_limb(p_start, p_end, progress, thick_a, thick_b):
                if not p_start or not p_end: return
                tip = get_interp_point(p_start, p_end, progress)
                
                draw_tapered_limb(frame, p_start, tip, thick_a, thick_b)
                draw_twig_decoration(frame, p_start, tip, growth_pct)
                
                active_hitboxes.append((p_start, tip, thick_a))
                
                # Guide line for ungrown part
                if progress < 1.0:
                    cv2.line(frame, tip, p_end, (60, 60, 60), 2)
            
            if l_shldr and l_elbow:
                process_limb(l_shldr, l_elbow, upper_growth, 35, 25)
                if l_wrist and upper_growth >= 1.0:
                    process_limb(l_elbow, l_wrist, lower_growth, 25, 15)
            
            if r_shldr and r_elbow:
                process_limb(r_shldr, r_elbow, upper_growth, 35, 25)
                if r_wrist and upper_growth >= 1.0:
                    process_limb(r_elbow, r_wrist, lower_growth, 25, 15)

        # --- SUN LOGIC ---
        if random.random() < 0.12: suns.append(SunOrb(w))
        
        kept_suns = []
        for s in suns:
            s.update()
            caught = False
            for (p1, p2, thickness) in active_hitboxes:
                dist = point_line_distance(s.x, s.y, p1, p2)
                if dist < (s.r + thickness):
                    caught = True
                    break
            
            if caught:
                game.absorb()
                # Sparkle explosion
                cv2.circle(frame, (int(s.x), int(s.y)), 35, (255, 255, 255), -1)
            elif s.y < h:
                s.draw(frame)
                kept_suns.append(s)
        suns = kept_suns

        # --- HUD ---
        bar_x, bar_y = 50, h - 50
        bar_w = w - 100
        
        # Determine bar color based on growth stage
        bar_col = LEAF_COLOR
        if growth_pct > 0.6: bar_col = FLOWER_COLORS[2] # Pink when blooming
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20), (30, 30, 30), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * growth_pct), bar_y + 20), bar_col, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20), (255, 255, 255), 2)
        
        label = "GROWING..."
        if growth_pct > 0.3: label = "LEAVES SPROUTING"
        if growth_pct > 0.6: label = "FLOWERS BLOOMING"
        if growth_pct >= 1.0: label = "MATURE TREE"
        
        cv2.putText(frame, f"{label}: {int(growth_pct*100)}%", (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)

        if game.won:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (w,h), (0, 30, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # Draw giant flower in center
            cx, cy = w//2, h//2
            cv2.circle(frame, (cx, cy), 100, (255, 100, 200), -1)
            cv2.circle(frame, (cx, cy), 80, (255, 200, 200), -1)
            cv2.putText(frame, "FULL BLOOM", (cx - 160, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 50, 50), 4)
            
            if time.time() - game.win_timer > 6: game.reset()

        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        b64 = base64.b64encode(buffer).decode('utf-8')
        socketio.emit('new_frame', {'image': b64})
        eventlet.sleep(0.01)

@app.route('/')
def index(): return render_template_string(HTML)

@socketio.on('connect')
def connect(): socketio.start_background_task(background_thread)

HTML = """
<body style="margin:0;background:#050505;display:flex;justify-content:center;height:100vh;overflow:hidden;font-family:sans-serif">
<div style="position:relative;width:100%;height:100%">
    <img id="vid" style="width:100%;height:100%;object-fit:contain;">
</div>
<div style="position:absolute; top:20px; left:20px; color:#cfc; font-family:monospace; background:#00000088; padding:15px; border-radius:8px; border-left: 4px solid #f0f;">
    <h2 style="margin:0;">DRYAD BLOOM</h2>
    <p>> Phase 1: Grow Wood (Brown)</p>
    <p>> Phase 2: Grow Leaves (Green)</p>
    <p>> Phase 3: Bloom Flowers (Pink)</p>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
<script>
const s = io();
document.getElementById('vid').onload = () => URL.revokeObjectURL(this.src);
s.on('new_frame', d => document.getElementById('vid').src = 'data:image/jpeg;base64,'+d.image);
</script>
</body>
"""

if __name__ == '__main__':
    try: socketio.run(app, host='0.0.0.0', port=5000)
    except: os._exit(0)