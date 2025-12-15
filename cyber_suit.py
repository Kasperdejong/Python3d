import cv2
import mediapipe as mp
import numpy as np
import math

def main():
    # 1. Setup Holistic (Tracks Body AND does Segmentation)
    mp_holistic = mp.solutions.holistic
    
    # enable_segmentation=True tells it to cut out the body background
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        enable_segmentation=True,
        refine_face_landmarks=True
    )

    cap = cv2.VideoCapture(0)

    # Indices of joints we want to connect to the skin
    # Shoulders (11,12), Elbows (13,14), Wrists (15,16), Hips (23,24), Knees (25,26)
    connect_joints = [11, 12, 13, 14, 15, 16, 23, 24]

    print("CYBER SUIT ACTIVATED... Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        # Create a dark "Sci-Fi" background version of the frame
        # We darken the real video so the neon lines pop
        display_frame = cv2.convertScaleAbs(frame, alpha=0.6, beta=-30)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        # --- 1. GET THE OUTLINE (HULL) ---
        if results.segmentation_mask is not None:
            # Create binary mask
            mask = (results.segmentation_mask > 0.5).astype(np.uint8) * 255
            
            # Find the edge of the body
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # If we found a body...
            if contours:
                # Get the largest contour (your body)
                body_contour = max(contours, key=cv2.contourArea)
                
                # Draw the Outer Shell (Neon Blue)
                cv2.drawContours(display_frame, [body_contour], -1, (255, 255, 0), 2)

                # --- 2. GET THE SKELETON ---
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Convert contour to a simpler format for math
                    # Shape becomes (N, 2)
                    contour_points = body_contour.reshape(-1, 2)
                    
                    # Loop through specific joints to draw "Struts"
                    for idx in connect_joints:
                        lm = landmarks[idx]
                        
                        # Only draw if the AI is confident it sees the joint
                        if lm.visibility > 0.5:
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            
                            # Draw the Joint Node
                            cv2.circle(display_frame, (cx, cy), 4, (0, 255, 255), -1)

                            # --- 3. THE CONNECTION LOGIC ---
                            # We need to find the closest point on the outline to this joint
                            
                            # Calculate distance to ALL contour points
                            # (Math: Square Root of (x2-x1)^2 + (y2-y1)^2)
                            distances = np.linalg.norm(contour_points - np.array([cx, cy]), axis=1)
                            
                            # Find index of the minimum distance
                            min_idx = np.argmin(distances)
                            
                            # Get that coordinate
                            closest_x, closest_y = contour_points[min_idx]
                            
                            # Draw the "Cable" from bone to skin
                            # Thin Green Line
                            cv2.line(display_frame, (cx, cy), (closest_x, closest_y), (0, 255, 100), 1)

                    # --- 4. DRAW INTERNAL SKELETON LINES ---
                    # Draw lines between shoulders
                    p11 = landmarks[11]; p12 = landmarks[12]
                    if p11.visibility > 0.5 and p12.visibility > 0.5:
                         cv2.line(display_frame, (int(p11.x*w), int(p11.y*h)), (int(p12.x*w), int(p12.y*h)), (0, 255, 255), 1)

                    # Shoulders to Elbows to Wrists
                    arm_paths = [(11, 13), (13, 15), (12, 14), (14, 16)]
                    for start, end in arm_paths:
                        l1 = landmarks[start]; l2 = landmarks[end]
                        if l1.visibility > 0.5 and l2.visibility > 0.5:
                            pt1 = (int(l1.x*w), int(l1.y*h))
                            pt2 = (int(l2.x*w), int(l2.y*h))
                            cv2.line(display_frame, pt1, pt2, (0, 255, 255), 1)

        cv2.imshow('CYBER SUIT UI', display_frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()