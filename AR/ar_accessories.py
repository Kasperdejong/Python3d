import cv2
import mediapipe as mp
import numpy as np
import math

# --- HELPER FUNCTIONS ---
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    abs_cos = abs(rot_mat[0,0])
    abs_sin = abs(rot_mat[0,1])
    bound_w = int(image.shape[0] * abs_sin + image.shape[1] * abs_cos)
    bound_h = int(image.shape[0] * abs_cos + image.shape[1] * abs_sin)
    rot_mat[0, 2] += bound_w/2 - image_center[0]
    rot_mat[1, 2] += bound_h/2 - image_center[1]
    result = cv2.warpAffine(image, rot_mat, (bound_w, bound_h), flags=cv2.INTER_LINEAR)
    return result

def overlay_image_alpha(img, img_overlay, x, y, overlay_size=None):
    try:
        if overlay_size is not None:
            img_overlay = cv2.resize(img_overlay, overlay_size)
        h, w, _ = img_overlay.shape
        rows, cols, _ = img.shape
        y = y - h // 2
        x = x - w // 2
        if y < 0 or y + h > rows or x < 0 or x + w > cols:
            return img
        alpha_mask = img_overlay[:, :, 3] / 255.0
        alpha_inv = 1.0 - alpha_mask
        for c in range(0, 3):
            img[y:y+h, x:x+w, c] = (alpha_mask * img_overlay[:, :, c] + 
                                    alpha_inv * img[y:y+h, x:x+w, c])
        return img
    except Exception:
        return img

# --- MAIN APP ---
def main():
    mp_pose = mp.solutions.pose
    
    # MODIFICATION 1: Enable Segmentation here so we can find the body shape
    pose = mp_pose.Pose(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5, 
        enable_segmentation=True
    )

    cap = cv2.VideoCapture(0)

    # Load Images (Using your exact filenames)
    helmet_img = cv2.imread("motorcyclehelmet.webp", cv2.IMREAD_UNCHANGED)
    watch_img = cv2.imread("wristwatch.webp", cv2.IMREAD_UNCHANGED)

    if helmet_img is None or watch_img is None:
        print("Error: Could not find motorcyclehelmet.webp or wristwatch.webp!")
        return

    print("AR Biker Mode: Black Suit + Helmet + Watch.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # --- MODIFICATION 2: BIKER SUIT EFFECT ---
        if results.segmentation_mask is not None:
            # Create a mask where Body is White (True)
            binary_mask = results.segmentation_mask > 0.5
            
            # A. Color the body Black (Dark Grey for detail)
            # We assume the mask defines the body. We set those pixels to [25, 25, 25]
            frame[binary_mask] = [25, 25, 25]

            # B. Draw the Silver Outline
            mask_uint8 = (binary_mask.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Draw contours: Color (192, 192, 192) is Silver
            cv2.drawContours(frame, contours, -1, (192, 192, 192), 2)


        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # --- 1. HELMET LOGIC ---
            left_ear = landmarks[7]
            right_ear = landmarks[8]
            nose = landmarks[0]

            le_x, le_y = int(left_ear.x * w), int(left_ear.y * h)
            re_x, re_y = int(right_ear.x * w), int(right_ear.y * h)
            head_width = math.dist([le_x, le_y], [re_x, re_y])
            
            # Scale
            scale_factor = 3.0 
            new_width = int(head_width * scale_factor)
            orig_h, orig_w = helmet_img.shape[:2]
            aspect_ratio = orig_h / orig_w
            new_height = int(new_width * aspect_ratio)

            # Rotation
            dy = re_y - le_y
            dx = re_x - le_x
            angle = math.degrees(math.atan2(dy, dx))
            
            center_x = int(nose.x * w)
            center_y = int(nose.y * h)
            
            # Position: Using your exact offset (0.03)
            center_y += int(new_height * 0.03)

            if new_width > 0 and new_height > 0:
                resized_helmet = cv2.resize(helmet_img, (new_width, new_height))
                rotated_helmet = rotate_image(resized_helmet, -angle + 180)
                frame = overlay_image_alpha(frame, rotated_helmet, center_x, center_y)


            # --- 2. WATCH LOGIC ---
            wrist = landmarks[15]
            elbow = landmarks[13]
            
            w_x, w_y = int(wrist.x * w), int(wrist.y * h)
            e_x, e_y = int(elbow.x * w), int(elbow.y * h)

            arm_length = math.dist([w_x, w_y], [e_x, e_y])
            watch_scale = 0.25 # Using your exact scale
            w_width = int(arm_length * watch_scale)
            
            orig_wh, orig_ww = watch_img.shape[:2]
            w_aspect = orig_wh / orig_ww
            w_height = int(w_width * w_aspect)

            dy = e_y - w_y
            dx = e_x - w_x
            angle = math.degrees(math.atan2(dy, dx)) + 0

            if w_width > 0 and w_height > 0:
                resized_watch = cv2.resize(watch_img, (w_width, w_height))
                rotated_watch = rotate_image(resized_watch, -angle)
                frame = overlay_image_alpha(frame, rotated_watch, w_x, w_y)

        cv2.imshow('AR Biker Accessories', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()