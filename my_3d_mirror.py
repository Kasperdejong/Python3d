import cv2
import mediapipe as mp
import open3d as o3d
import numpy as np

def main():
    # 1. Initialize MediaPipe Holistic
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_face_landmarks=True
    )

    # 2. Setup Open3D Window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Skeleton Hands", width=800, height=600)

    # --- GEOMETRY SETUP ---

    # A. Face Point Cloud (Keep as dots)
    face_pcd = o3d.geometry.PointCloud()
    vis.add_geometry(face_pcd)

    # B. Hand LineSets (The lines connecting the joints)
    # We need to define the connections (bones) once
    hand_connections = list(mp_holistic.HAND_CONNECTIONS)
    # Convert to numpy array for Open3D [[0,1], [1,2], ...]
    lines_indices = np.array(hand_connections)

    # Create LineSet for Left Hand
    left_hand_lines = o3d.geometry.LineSet()
    left_hand_lines.lines = o3d.utility.Vector2iVector(lines_indices)
    # Initialize with dummy points (collapsed at 0,0,0)
    left_hand_lines.points = o3d.utility.Vector3dVector(np.zeros((21, 3)))
    left_hand_lines.paint_uniform_color([1, 0.5, 0]) # Orange
    vis.add_geometry(left_hand_lines)

    # Create LineSet for Right Hand
    right_hand_lines = o3d.geometry.LineSet()
    right_hand_lines.lines = o3d.utility.Vector2iVector(lines_indices)
    right_hand_lines.points = o3d.utility.Vector3dVector(np.zeros((21, 3)))
    right_hand_lines.paint_uniform_color([1, 0.5, 0]) # Orange
    vis.add_geometry(right_hand_lines)

    # Add coordinate frame for reference
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    vis.add_geometry(axis)

    # 3. Open Webcam
    cap = cv2.VideoCapture(0)

    print("Running Skeleton Mode... Press 'q' in the camera window to quit.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        # --- UPDATE FACE (Dots) ---
        if results.face_landmarks:
            face_points = []
            for lm in results.face_landmarks.landmark:
                # x=Left/Right, y=Up/Down (inverted), z=Depth (inverted)
                face_points.append([lm.x - 0.5, -(lm.y - 0.5), -lm.z])
            
            face_pcd.points = o3d.utility.Vector3dVector(np.array(face_points))
            face_pcd.paint_uniform_color([0, 1, 1]) # Teal
            vis.update_geometry(face_pcd)
        else:
            # Hide face if lost
            face_pcd.points = o3d.utility.Vector3dVector(np.zeros((1, 3)))
            vis.update_geometry(face_pcd)

        # --- UPDATE HANDS (Lines) ---
        
        def update_hand(hand_landmarks, line_set_geometry):
            if hand_landmarks:
                hand_points = []
                for lm in hand_landmarks.landmark:
                    hand_points.append([lm.x - 0.5, -(lm.y - 0.5), -lm.z])
                
                # Update the vertices of the lines
                line_set_geometry.points = o3d.utility.Vector3dVector(np.array(hand_points))
            else:
                # Collapse points to 0 if hand not found to "hide" it
                line_set_geometry.points = o3d.utility.Vector3dVector(np.zeros((21, 3)))
            
            vis.update_geometry(line_set_geometry)

        update_hand(results.left_hand_landmarks, left_hand_lines)
        update_hand(results.right_hand_landmarks, right_hand_lines)

        vis.poll_events()
        vis.update_renderer()

        cv2.imshow('Webcam Feed', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    vis.destroy_window()

if __name__ == "__main__":
    main()