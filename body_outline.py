import cv2
import mediapipe as mp
import numpy as np
import time

def main():
    # 1. Setup MediaPipe Selfie Segmentation
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    
    # model_selection=1 is for landscape mode (more accurate for computers)
    segmenter = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    cap = cv2.VideoCapture(0)

    print("Body Outline Mode... Press 'q' to quit.")
    
    # Variables for the color cycle (Rainbow effect)
    hue = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Flip frame for mirror view
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # --- PROCESS SEGMENTATION ---
        results = segmenter.process(rgb_frame)
        
        # The result is a 'mask' of probabilities (0.0 to 1.0)
        # We assume anything > 0.6 is "Human"
        binary_mask = (results.segmentation_mask > 0.6).astype(np.uint8) * 255

        # --- EDGE DETECTION ---
        # We use OpenCV to find the contours (the outline) of the white mask
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # --- COLOR LOGIC (Rainbow Cycle) ---
        hue = (hue + 1) % 180
        # Convert HSV color to RGB for OpenCV
        # H=hue, S=255 (Full Saturation), V=255 (Full Brightness)
        neon_color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
        neon_color = (int(neon_color[0]), int(neon_color[1]), int(neon_color[2]))

        # --- DRAWING ---
        
        # Option A: Draw thick outline on the original video
        # We draw it twice: once thick and dark (shadow), once thin and bright (glow)
        cv2.drawContours(frame, contours, -1, (0, 0, 0), 6)          # Black Border
        cv2.drawContours(frame, contours, -1, neon_color, 3)         # Neon Line

        # Option B: Add a semi-transparent "Forcefield" effect inside the body
        # (Uncomment lines below to enable)
        
        # colored_mask = np.zeros_like(frame)
        # colored_mask[:] = neon_color
        # # Extract just the body part of the color
        # body_overlay = cv2.bitwise_and(colored_mask, colored_mask, mask=binary_mask)
        # # Blend it with original frame
        # frame = cv2.addWeighted(frame, 1.0, body_overlay, 0.3, 0) # 0.3 is opacity


        cv2.imshow('Neon Body Scanner', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()