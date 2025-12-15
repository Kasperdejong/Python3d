import cv2
import mediapipe as mp
import numpy as np

def main():
    # 1. Setup MediaPipe classes
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # 2. Configure the AI
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1 # 0=Fast, 1=Balanced, 2=Accurate
    )

    cap = cv2.VideoCapture(0)

    print("Starting Mocap Stickman... Press 'q' to quit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Flip for mirror view
        frame = cv2.flip(frame, 1)
        
        # Create a Black Canvas of the same size as the camera feed
        h, w, c = frame.shape
        black_canvas = np.zeros((h, w, c), dtype=np.uint8)

        # Process the image
        # (MediaPipe needs RGB, OpenCV uses BGR)
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        # --- DRAWING THE STICKMAN ---
        # We draw on 'black_canvas' instead of 'frame' to get the isolated look
        
        # 1. Draw Face Mesh (Tessellation)
        # We use a custom style to make it look like a wireframe
        mp_drawing.draw_landmarks(
            black_canvas,
            results.face_landmarks,
            mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )

        # 2. Draw Pose (Body Skeleton)
        mp_drawing.draw_landmarks(
            black_canvas,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )

        # 3. Draw Left Hand
        mp_drawing.draw_landmarks(
            black_canvas,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        # 4. Draw Right Hand
        mp_drawing.draw_landmarks(
            black_canvas,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        # Combine views: Camera on Left, Stickman on Right (Optional)
        # combined = np.hstack((frame, black_canvas))
        
        # Or just show the Stickman
        cv2.imshow('Digital Stickman', black_canvas)
        
        # Show the real camera in a small separate window just for reference
        cv2.imshow('Reference Camera', cv2.resize(frame, (320, 240)))

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()