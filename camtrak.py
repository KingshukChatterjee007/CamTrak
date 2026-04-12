import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import numpy as np
import time
import os
from utils import HandSmoothing, calculate_distance
from video_stream import ThreadedCamera

# --- Configurations ---
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
CAM_WIDTH, CAM_HEIGHT = 640, 480
PROC_WIDTH, PROC_HEIGHT = 320, 240
FRAME_REDUCTION = 90 # Narrower for better reach (Increased DPI)

# Ratio-based thresholds (Pinch Distance / Palm Size)
# This makes recognition perfect at any distance (3D Scale-Invariant)
PINCH_RATIO_START = 0.35
PINCH_RATIO_RELEASE = 0.55
CLICK_COOLDOWN = 0.25
DOUBLE_PINCH_WINDOW = 0.6
MODEL_PATH = 'hand_landmarker.task'
pyautogui.FAILSAFE = True

# Colors
COLOR_GLOW = (255, 255, 0)
COLOR_TEXT = (0, 255, 255)

# High-Precision Engine Tuning
# min_cutoff: stationary jitter elimination
# beta: motion response speed
smoother = HandSmoothing(min_cutoff=0.01, beta=0.12)
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))

def count_fingers(landmarks):
    fingers = []
    # Thumb: check relative to palm
    dist_thumb = calculate_distance(landmarks[4], landmarks[2])
    if dist_thumb > 0.08: fingers.append(1)
    else: fingers.append(0)
    for id in [8, 12, 16, 20]:
        if landmarks[id].y < landmarks[id-2].y: fingers.append(1)
        else: fingers.append(0)
    return sum(fingers)

def draw_landmarks_manual(img, landmarks):
    h, w, _ = img.shape
    pts = [(int(l.x * w), int(l.y * h)) for l in landmarks]
    for tip, pip in [(4,2), (8,6), (12,10), (16,14), (20,18)]:
        cv2.line(img, pts[tip], pts[pip], (0, 255, 0), 1)
    for p in pts:
        cv2.circle(img, p, 2, (0, 255, 255), -1)

def run_camtrak():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        return

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    camera = ThreadedCamera(src=0, width=CAM_WIDTH, height=CAM_HEIGHT).start()
    cv2.namedWindow("CamTrak HUD", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("CamTrak HUD", cv2.WND_PROP_TOPMOST, 1)
    cv2.resizeWindow("CamTrak HUD", 350, 260)

    print("CamTrak 2.7 Ultra-Precision Engine Started.")
    
    last_left_click_time = 0
    last_ring_pinch_time = 0
    is_left_clicking = False
    start_time = time.time()

    while True:
        success, raw_img = camera.read()
        if not success or raw_img is None: 
            time.sleep(0.01)
            continue

        img = cv2.flip(raw_img, 1)
        lh, lw, _ = img.shape
        
        # 1. Contrast Pre-processing
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        img_p = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        
        img_small = cv2.resize(img_p, (PROC_WIDTH, PROC_HEIGHT))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB))
        timestamp_ms = int((time.time() - start_time) * 1000)
        
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        status_text = "Tracking..."
        mode_text = "IDLE"

        if result.hand_landmarks:
            lms = result.hand_landmarks[0]
            
            # Distance-Invariant Scale Factor
            palm_size = calculate_distance(lms[0], lms[5])
            if palm_size < 0.01: palm_size = 0.01

            # A. Cursor (Index Only)
            index_tip = (int(lms[8].x * lw), int(lms[8].y * lh))
            s_x, s_y = smoother.smooth(8, index_tip)
            
            # Mapping with overshoot logic
            x_mapped = np.interp(s_x, (FRAME_REDUCTION, lw - FRAME_REDUCTION), (0, SCREEN_WIDTH))
            y_mapped = np.interp(s_y, (FRAME_REDUCTION, lh - FRAME_REDUCTION), (0, SCREEN_HEIGHT))
            x_mapped = np.clip(x_mapped, 0, SCREEN_WIDTH - 1)
            y_mapped = np.clip(y_mapped, 0, SCREEN_HEIGHT - 1)
            
            pyautogui.moveTo(x_mapped, y_mapped, _pause=False)
            mode_text = "ACTIVE"

            current_time = time.time()

            # B. Left Click (Thumb + Index Ratio)
            dist_index = calculate_distance(lms[4], lms[8])
            ratio_index = dist_index / palm_size
            
            if not is_left_clicking:
                if ratio_index < PINCH_RATIO_START and (current_time - last_left_click_time) > CLICK_COOLDOWN:
                    pyautogui.mouseDown()
                    is_left_clicking = True
                    last_left_click_time = current_time
                    status_text = "LEFT PINCH"
            else:
                if ratio_index > PINCH_RATIO_RELEASE:
                    pyautogui.mouseUp()
                    is_left_clicking = False

            # C. Right Click (Thumb + Ring Double Pinch)
            dist_ring = calculate_distance(lms[4], lms[16])
            ratio_ring = dist_ring / palm_size
            
            if ratio_ring < PINCH_RATIO_START:
                if (current_time - last_ring_pinch_time) < DOUBLE_PINCH_WINDOW and (current_time - last_ring_pinch_time) > 0.05:
                    pyautogui.rightClick()
                    status_text = "RIGHT DOUBLE"
                    last_ring_pinch_time = 0
                else:
                    last_ring_pinch_time = current_time

            # Visualization
            draw_landmarks_manual(img, lms)
            cv2.circle(img, (int(s_x), int(s_y)), 6, COLOR_GLOW, -1)
                
        # HUD
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (lw, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
        cv2.putText(img, f"CAMTRAK 2.7 | {mode_text}", (15, 25), cv2.FONT_HERSHEY_PLAIN, 1, COLOR_TEXT, 1)
        cv2.putText(img, status_text, (15, 50), cv2.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255), 2)

        cv2.imshow("CamTrak HUD", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    camera.stop()
    cv2.destroyAllWindows()
    landmarker.close()

if __name__ == "__main__":
    run_camtrak()
