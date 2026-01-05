import warnings
warnings.filterwarnings("ignore")

import sys, traceback, os, time, math, random
_original_print_exception = traceback.print_exception
def silent_ssl_print_exception(etype, value, tb, limit=None, file=None, chain=True):
    err_str = str(value)
    if "SSLV3_ALERT_CERTIFICATE_UNKNOWN" in err_str: return
    if "HTTP_REQUEST" in err_str: return
    _original_print_exception(etype, value, tb, limit=limit, file=file, chain=chain)
traceback.print_exception = silent_ssl_print_exception

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
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='eventlet', ping_timeout=60)

thread = None

# autumn theme
BARK_COLOR = (19, 69, 139)      # Dark Brown
HIGHLIGHT_COLOR = (40, 90, 160) # Light Brown

# autumn colors (BGR Format)
LEAF_COLOR = (40, 80, 60)       # Dark Olive Green / Brownish
LEAF_SHADOW = (20, 40, 30)      # Dark Shadow
FLOWER_COLORS = [
    (30, 100, 230),  # Bright Orange
    (30, 50, 200),   # Deep Red
    (20, 140, 255),  # Golden Yellow
    (10, 30, 160)    # Rust
]

# utils
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

def overlay_image_alpha(img, img_overlay, x, y, width, height):
    try:
        if width <= 0 or height <= 0: return img
        img_overlay = cv2.resize(img_overlay, (width, height))
        y1 = y - height; y2 = y; x1 = x - (width // 2); x2 = x + (width // 2)
        h, w = img.shape[:2]
        if y1 < 0: img_overlay = img_overlay[abs(y1):, :, :]; y1 = 0
        if y2 > h: y2 = h
        if x1 < 0: img_overlay = img_overlay[:, abs(x1):, :]; x1 = 0
        if x2 > w: img_overlay = img_overlay[:, :-(x2-w), :]; x2 = w
        overlay_h, overlay_w = img_overlay.shape[:2]
        if overlay_h == 0 or overlay_w == 0: return img
        
        small_overlay = img_overlay
        
        # Check if image has alpha channel
        if small_overlay.shape[2] == 4:
            alpha_mask = small_overlay[:, :, 3] / 255.0
            alpha_inv = 1.0 - alpha_mask
            for c in range(0, 3):
                img[y1:y1+overlay_h, x1:x1+overlay_w, c] = (
                    alpha_mask * small_overlay[:, :, c] + 
                    alpha_inv * img[y1:y1+overlay_h, x1:x1+overlay_w, c]
                )
        else:
            # If no alpha, just paste it (fallback)
            img[y1:y1+overlay_h, x1:x1+overlay_w] = small_overlay[:, :, :3]
            
        return img
    except: return img

def load_tree_asset():
    path = "assets/treebush.png"
    
    # Check if file exists
    if not os.path.exists(path):
        print(f"⚠️ WARNING: Could not find {path}. Generating fallback square.")
        # Fallback: Orange Square
        fallback = np.zeros((300, 300, 4), dtype=np.uint8)
        fallback[:] = (30, 100, 230, 255) 
        return fallback

    try:
        # Load with Alpha Channel (IMREAD_UNCHANGED)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        # If user provides JPG/PNG without alpha, add 255 alpha channel
        if img.shape[2] == 3:
            b, g, r = cv2.split(img)
            alpha = np.ones_like(b) * 255
            img = cv2.merge((b, g, r, alpha))
            
        return img
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return np.zeros((100, 100, 4), dtype=np.uint8)

# rendering engine

def draw_sturdy_trunk(frame, ls, rs, lh, rh):
    # this draws a solid tree
    shoulder_w = math.hypot(rs[0]-ls[0], rs[1]-ls[1])
    center_top = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
    center_base = ((lh[0] + rh[0]) // 2, (lh[1] + rh[1]) // 2)
    
    # shape config
    top_width_mult = 0.9   
    base_width_mult = 0.6 
    
    trunk_top_w = shoulder_w * top_width_mult
    trunk_base_w = shoulder_w * base_width_mult
    
    p1 = (int(center_top[0] - trunk_top_w//2), center_top[1])  
    p2 = (int(center_top[0] + trunk_top_w//2), center_top[1])  
    p3 = (int(center_base[0] + trunk_base_w//2), center_base[1]) 
    p4 = (int(center_base[0] - trunk_base_w//2), center_base[1]) 
    
    pts = np.array([p1, p2, p3, p4], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(frame, [pts], BARK_COLOR)
    
    steps = 6
    for i in range(1, steps):
        t = i / steps
        top = get_interp_point(p1, p2, t)
        bot = get_interp_point(p4, p3, t)
        mid_x = (top[0] + bot[0]) // 2 + random.randint(-3, 3)
        mid_y = (top[1] + bot[1]) // 2
        cv2.line(frame, top, (mid_x, mid_y), HIGHLIGHT_COLOR, 2)
        cv2.line(frame, (mid_x, mid_y), bot, HIGHLIGHT_COLOR, 2)

def draw_foliage_cluster(frame, x, y, density, seed_val):
    # this draws leaves on the arms.
    random.seed(seed_val)
    count = int(3 + density * 5)
    for _ in range(count):
        ox, oy = random.randint(-15, 15), random.randint(-15, 15)
        scale = random.randint(3, 6)
        color = LEAF_COLOR if random.random() > 0.3 else LEAF_SHADOW
        cv2.circle(frame, (x+ox, y+oy), scale, color, -1)
    if density > 0.6:
        # Draw Autumn Flowers/Berries
        for _ in range(random.randint(1, 3)):
            ox, oy = random.randint(-10, 10), random.randint(-10, 10)
            cv2.circle(frame, (x+ox, y+oy), 5, random.choice(FLOWER_COLORS), -1)

def draw_tapered_limb(frame, start, end, start_thick, end_thick):
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
    mid_x = (start[0] + end[0]) // 2 + random.randint(-2, 2)
    mid_y = (start[1] + end[1]) // 2
    cv2.line(frame, start, (mid_x, mid_y), HIGHLIGHT_COLOR, 1)
    cv2.line(frame, (mid_x, mid_y), end, HIGHLIGHT_COLOR, 1)

def draw_twig_decoration(frame, start, end, growth_pct):
    vx, vy = end[0] - start[0], end[1] - start[1]
    if math.hypot(vx, vy) < 20: return
    count = 3
    for i in range(1, count + 1):
        t = i / (count + 1)
        base = get_interp_point(start, end, t)
        random.seed(base[0] * base[1])
        angle_offset = random.uniform(0.3, 0.8)
        direction = 1 if i % 2 == 0 else -1
        tx = int(base[0] + (vy * angle_offset * direction))
        ty = int(base[1] - (vx * angle_offset * direction))
        cv2.line(frame, base, (tx, ty), BARK_COLOR, 2)
        if growth_pct > 0.3:
            draw_foliage_cluster(frame, tx, ty, growth_pct, base[0])

# Game logic
class GameState:
    def __init__(self):
        self.xp = 0.0
        self.max_xp = 200.0
        self.won = False
        self.win_timer = 0
        self.plant_asset = load_tree_asset()
    
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
        cv2.circle(frame, (int(self.x), int(self.y)), self.r+5, (100,255,255), -1)
        cv2.circle(frame, (int(self.x), int(self.y)), self.r, (255,255,255), -1)

# main server
def background_thread():
    print("Autumn bloom engine started")
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
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        
        active_hitboxes = [] 
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
                # Key Point: neck center. This connects the shoulders
                neck = ((l_shldr[0] + r_shldr[0]) // 2, (l_shldr[1] + r_shldr[1]) // 2)
                
                # 1 draw trunk
                draw_sturdy_trunk(frame, l_shldr, r_shldr, l_hip, r_hip)
                
                if nose:
                    bush_w = 150 + int(100 * growth_pct)
                    bush_h = 200 + int(100 * growth_pct)
                    
                    # Head position offset. Higher number is higher on your head
                    head_offset_y = 0 
                    
                    overlay_image_alpha(frame, game.plant_asset, neck[0], neck[1] - head_offset_y, bush_w, bush_h)
                    active_hitboxes.append((neck, nose, 60))

                # 3. Draw arms
                limb_reach = 0.2 + (growth_pct * 0.8)
                upper_growth = min(1.0, limb_reach * 2)
                lower_growth = max(0.0, (limb_reach - 0.5) * 2)

                def process_limb(p_start, p_end, progress, thick_a, thick_b):
                    if not p_start or not p_end: return
                    tip = get_interp_point(p_start, p_end, progress)
                    draw_tapered_limb(frame, p_start, tip, thick_a, thick_b)
                    draw_twig_decoration(frame, p_start, tip, growth_pct)
                    active_hitboxes.append((p_start, tip, thick_a))
                    if progress < 1.0: cv2.line(frame, tip, p_end, (60, 60, 60), 1)
                
                # Clavicles
                draw_tapered_limb(frame, neck, l_shldr, 45, 30)
                draw_tapered_limb(frame, neck, r_shldr, 45, 30)

                # Arms
                if l_elbow:
                    process_limb(l_shldr, l_elbow, upper_growth, 30, 20)
                    if l_wrist and upper_growth >= 1.0:
                        process_limb(l_elbow, l_wrist, lower_growth, 20, 10)

                if r_elbow:
                    process_limb(r_shldr, r_elbow, upper_growth, 30, 20)
                    if r_wrist and upper_growth >= 1.0:
                        process_limb(r_elbow, r_wrist, lower_growth, 20, 10)

        # Sun logic (random right now)
        if random.random() < 0.12: suns.append(SunOrb(w))
        kept_suns = []
        for s in suns:
            s.update()
            caught = False
            for (p1, p2, thickness) in active_hitboxes:
                dist = point_line_distance(s.x, s.y, p1, p2)
                if dist < (s.r + thickness):
                    caught = True; break
            
            if caught:
                game.absorb()
                cv2.circle(frame, (int(s.x), int(s.y)), 35, (255, 255, 255), -1)
            elif s.y < h:
                s.draw(frame)
                kept_suns.append(s)
        suns = kept_suns

        # HUD
        bar_x, bar_y, bar_w = 50, h - 50, w - 100
        bar_col = LEAF_COLOR
        if growth_pct > 0.6: bar_col = FLOWER_COLORS[0]
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20), (30, 30, 30), -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * growth_pct), bar_y + 20), bar_col, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20), (255, 255, 255), 2)
        
        label = "GROWING..."
        if growth_pct > 0.6: label = "AUTUMN BLOOM"
        cv2.putText(frame, f"{label}: {int(growth_pct*100)}%", (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)

        if game.won:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (w,h), (0, 30, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            cx, cy = w//2, h//2
            cv2.circle(frame, (cx, cy), 100, FLOWER_COLORS[0], -1)
            cv2.circle(frame, (cx, cy), 80, FLOWER_COLORS[1], -1)
            cv2.putText(frame, "NATURE RESTORED", (cx - 180, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            if time.time() - game.win_timer > 6: game.reset()

        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
        b64 = base64.b64encode(buffer).decode('utf-8')
        socketio.emit('new_frame', {'image': b64})
        eventlet.sleep(0.01)

def get_local_ip():
    try: s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]; s.close(); return ip
    except: return "127.0.0.1"

@app.route('/')
def index(): return render_template_string(HTML)

@socketio.on('connect')
def connect():
    global thread
    if thread is None: thread = socketio.start_background_task(background_thread)

HTML = """
<body style="margin:0;background:#050505;display:flex;justify-content:center;height:100vh;overflow:hidden;font-family:sans-serif">
<div style="position:relative;width:100%;height:100%">
    <img id="vid" style="width:100%;height:100%;object-fit:contain;">
</div>
<div style="position:absolute; top:20px; left:20px; color:#cfc; font-family:monospace; background:#00000088; padding:15px; border-radius:8px; border-left: 4px solid #E6641E;">
    <h2 style="margin:0;">AUTUMN BLOOM</h2>
    <p>> Phase 1: Grow Wood (Bark)</p>
    <p>> Phase 2: Grow Foliage (Olive)</p>
    <p>> Phase 3: Autumn Bloom (Orange)</p>
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
    print(f"🚀 SERVER RUNNING AT: http://{get_local_ip()}:5000")
    try: socketio.run(app, host='0.0.0.0', port=5000)
    except: os._exit(0)