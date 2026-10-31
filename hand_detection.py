# -*- coding: utf-8 -*-
"""
Hand Detection & Finger Name Detector with Voice & Visual Labels
================================================================
- Detects Left and Right hands in real-time using OpenCV & MediaPipe Tasks.
- Accurately identifies and labels each individual finger:
    * Thumb
    * Index Finger
    * Middle Finger
    * Ring Finger
    * Pinky Finger
- Shows whether each finger is UP (extended) or DOWN (folded).
- Text-to-Speech (TTS): Announces "Right Hand", "Left Hand", and active finger names.
  Press 'V' to toggle voice on/off.
- Press 'S' to save screenshot.
- Press 'Q' or 'ESC' to exit.
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import time
import os
import sys
import threading
import queue
import urllib.request

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────
#  Model Download Helper
# ─────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

def ensure_model_file():
    """Ensure the MediaPipe hand landmarker model exists locally."""
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading hand landmarker model to: {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete!")

# ─────────────────────────────────────────────────────────────
#  Text-To-Speech (TTS) Worker Thread
# ─────────────────────────────────────────────────────────────
class VoiceSpeaker:
    def __init__(self):
        self.queue = queue.Queue()
        self.enabled = True
        self.last_speech_text = ""
        self.last_speech_time = 0
        self.debounce_sec = 2.5  # Don't repeat the exact same announcement too quickly
        
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 165)
            engine.setProperty('volume', 0.9)
        except Exception as e:
            print(f"[Voice Warning] TTS engine could not be initialized: {e}")
            return

        while True:
            text = self.queue.get()
            if text is None:
                break
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[Voice Error] {e}")
            finally:
                self.queue.task_done()

    def speak(self, text, force=False):
        if not self.enabled:
            return
        now = time.time()
        # Avoid flood if text didn't change or if spoken very recently
        if not force and text == self.last_speech_text and (now - self.last_speech_time < self.debounce_sec):
            return
        if not force and (now - self.last_speech_time < 1.8):
            return
        self.last_speech_text = text
        self.last_speech_time = now
        self.queue.put(text)

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled

# ─────────────────────────────────────────────────────────────
#  Camera Helper (DirectShow on Windows for reliable opening)
# ─────────────────────────────────────────────────────────────
def open_camera(target_index=0):
    """
    Tries to open camera using cv2.CAP_DSHOW (DirectShow) first on Windows,
    then falls back to MSMF and default ANY. Scans indices 0, 1, 2.
    """
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    indices_to_try = [target_index] + [i for i in [0, 1, 2] if i != target_index]

    for idx in indices_to_try:
        for backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                # Attempt to set HD resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    print(f"[OK] Camera initialized successfully on index {idx} ({actual_w}x{actual_h})")
                    return cap, idx
                cap.release()

    return None, None

# ─────────────────────────────────────────────────────────────
#  Landmarks & Finger Configurations
# ─────────────────────────────────────────────────────────────
FINGER_TIPS = {
    "Thumb":  4,
    "Index":  8,
    "Middle": 12,
    "Ring":   16,
    "Pinky":  20,
}

FINGER_PIPS = {
    "Index":  6,
    "Middle": 10,
    "Ring":   14,
    "Pinky":  18,
}

FINGER_MCPS = {
    "Thumb":  2,
    "Index":  5,
    "Middle": 9,
    "Ring":   13,
    "Pinky":  17,
}

# Standard skeleton connection pairs
SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]

# Color Scheme (BGR)
COLOR_RIGHT = (40, 200, 80)     # Neon Green for Right Hand
COLOR_LEFT  = (230, 110, 30)    # Blue-Cyan / Coral for Left Hand
COLOR_UP    = (0, 255, 255)     # Yellow for raised finger
COLOR_DOWN  = (130, 130, 130)   # Dim Gray for folded finger
COLOR_BOX   = (20, 20, 20)

def draw_pill(img, text, pos, font_scale=0.55, thickness=1,
              text_color=(255, 255, 255), bg_color=(30, 30, 30), padding=5):
    """Draws text with a rounded-style solid background box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    x1, y1 = x - padding, y - th - padding
    x2, y2 = x + tw + padding, y + baseline + padding
    cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, cv2.FILLED)
    cv2.rectangle(img, (x1, y1), (x2, y2), (200, 200, 200), 1)
    cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    return tw + padding * 2, th + baseline + padding * 2

def check_fingers_status(landmarks, handedness_label):
    """
    Determines whether each finger (Thumb, Index, Middle, Ring, Pinky) is UP or DOWN.
    Returns: dict { 'Thumb': True/False, ... }
    """
    status = {}
    
    # 4 Main fingers: tip higher than PIP in image coordinates (smaller Y means higher)
    for name in ["Index", "Middle", "Ring", "Pinky"]:
        tip_y = landmarks[FINGER_TIPS[name]].y
        pip_y = landmarks[FINGER_PIPS[name]].y
        status[name] = tip_y < pip_y

    # Thumb: Use geometric distance from Thumb Tip (4) to Pinky MCP (17)
    # When thumb is extended out, distance is greater than Thumb IP (3) to Pinky MCP (17)
    tip = landmarks[4]
    ip = landmarks[3]
    pinky_mcp = landmarks[17]
    dist_tip_pinky = math.hypot(tip.x - pinky_mcp.x, tip.y - pinky_mcp.y)
    dist_ip_pinky  = math.hypot(ip.x - pinky_mcp.x,  ip.y - pinky_mcp.y)
    
    status["Thumb"] = dist_tip_pinky > (dist_ip_pinky * 1.15)
    return status

# ─────────────────────────────────────────────────────────────
#  Main Application
# ─────────────────────────────────────────────────────────────
def main():
    ensure_model_file()
    
    # Initialize MediaPipe Hand Landmarker
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # Initialize Voice Speaker
    speaker = VoiceSpeaker()

    # Open Webcam using robust DirectShow scanner
    current_cam_idx = 0
    cap, current_cam_idx = open_camera(target_index=current_cam_idx)
    if cap is None:
        print("\n" + "!"*65)
        print("❌ Camera could not be opened.")
        print("  Troubleshooting steps:")
        print("  1. Make sure no other program (Teams, Zoom, browser, Discord)")
        print("     is currently using the camera.")
        print("  2. Check Windows Camera Privacy Settings:")
        print("     Settings > Privacy & Security > Camera -> Turn ON")
        print("     'Let desktop apps access your camera'.")
        print("  3. If using an external USB webcam, unplug and re-plug it.")
        print("!"*65 + "\n")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    screenshot_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")
    os.makedirs(screenshot_dir, exist_ok=True)

    prev_time = time.time()
    last_spoken_state = None

    print("\n" + "="*60)
    print(">> HAND & FINGER DETECTION SYSTEM RUNNING")
    print("="*60)
    print(" Controls:")
    print("  [V]       Toggle Voice Announcements (Currently: ON)")
    print("  [C]       Switch / Cycle Camera (Index 0, 1, 2...)")
    print("  [S]       Save Screenshot")
    print("  [Q / ESC] Quit application")
    print("="*60 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        h, w, _ = frame.shape

        # We pass the original camera frame to MediaPipe for true handedness
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection_result = detector.detect(mp_image)

        # Mirror frame horizontally for intuitive display (selfie view)
        display = cv2.flip(frame, 1)

        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-5)
        prev_time = curr_time

        # Top Bar: FPS & Voice status
        draw_pill(display, f"FPS: {int(fps)}", (15, 30), font_scale=0.55, bg_color=(20, 20, 20))
        voice_str = "Voice: ON (Press V to Mute)" if speaker.enabled else "Voice: MUTED (Press V to Unmute)"
        voice_bg  = (0, 100, 0) if speaker.enabled else (50, 50, 50)
        draw_pill(display, voice_str, (120, 30), font_scale=0.55, bg_color=voice_bg)

        # Bottom Instructions
        draw_pill(display, f"Cam #{current_cam_idx} | 'C' Switch Cam | 'V' Voice | 'S' Screenshot | 'Q' Quit",
                  (15, h - 15), font_scale=0.48, bg_color=(15, 15, 15))

        announcements = []

        if detection_result.hand_landmarks:
            for hand_idx, landmarks in enumerate(detection_result.hand_landmarks):
                # Retrieve handedness (Right / Left)
                handedness_cat = detection_result.handedness[hand_idx][0]
                hand_label = handedness_cat.category_name  # "Right" or "Left"
                confidence = handedness_cat.score

                hand_color = COLOR_RIGHT if hand_label.lower() == "right" else COLOR_LEFT

                # In mirrored display, x is inverted: screen_x = (1.0 - x) * w
                coords = []
                for lm in landmarks:
                    sx = int((1.0 - lm.x) * w)
                    sy = int(lm.y * h)
                    coords.append((sx, sy))

                # 1. Draw Skeleton Lines
                for p1, p2 in SKELETON_CONNECTIONS:
                    cv2.line(display, coords[p1], coords[p2], (210, 210, 210), 2, cv2.LINE_AA)

                # 2. Draw Landmarks Circles
                for i, (sx, sy) in enumerate(coords):
                    cv2.circle(display, (sx, sy), 4, hand_color, -1)

                # 3. Compute Hand Bounding Box
                xs = [pt[0] for pt in coords]
                ys = [pt[1] for pt in coords]
                bx1, bx2 = max(min(xs) - 25, 0), min(max(xs) + 25, w)
                by1, by2 = max(min(ys) - 25, 0), min(max(ys) + 25, h)
                cv2.rectangle(display, (bx1, by1), (bx2, by2), hand_color, 2)

                # Hand Name Banner (e.g. "RIGHT HAND - 98%")
                banner_text = f"{hand_label.upper()} HAND ({int(confidence * 100)}%)"
                draw_pill(display, banner_text, (bx1, max(by1 - 10, 25)),
                          font_scale=0.65, thickness=2, text_color=(255, 255, 255), bg_color=hand_color)

                # 4. Check each Finger (Thumb, Index, Middle, Ring, Pinky)
                finger_status = check_fingers_status(landmarks, hand_label)
                raised_fingers = [name for name, is_up in finger_status.items() if is_up]

                # Draw Finger Labels at each tip
                for name, tip_idx in FINGER_TIPS.items():
                    tx, ty = coords[tip_idx]
                    is_up = finger_status[name]

                    tip_color = COLOR_UP if is_up else COLOR_DOWN
                    # Draw glowing fingertip circle
                    cv2.circle(display, (tx, ty), 8, tip_color, cv2.FILLED)
                    cv2.circle(display, (tx, ty), 10, (255, 255, 255), 1)

                    # Label with finger name and state
                    status_symbol = "UP" if is_up else "DOWN"
                    f_label = f"{name} ({status_symbol})"
                    
                    label_color = (0, 255, 255) if is_up else (180, 180, 180)
                    draw_pill(display, f_label, (tx - 35, ty - 15),
                              font_scale=0.45, thickness=1,
                              text_color=label_color, bg_color=(25, 25, 25))

                # 5. Summary Info below hand box
                if raised_fingers:
                    summary_text = f"{len(raised_fingers)} Up: {', '.join(raised_fingers)}"
                else:
                    summary_text = "Fist (No fingers up)"

                draw_pill(display, summary_text, (bx1, min(by2 + 25, h - 25)),
                          font_scale=0.52, text_color=(255, 255, 255), bg_color=(40, 40, 40))

                # Build speech text for announcement
                if raised_fingers:
                    fingers_str = ", ".join(raised_fingers) + " finger" + ("s" if len(raised_fingers) > 1 else "")
                    announcements.append(f"{hand_label} hand. {fingers_str}")
                else:
                    announcements.append(f"{hand_label} hand fist")

        else:
            # Center banner when no hand is in frame
            center_msg = "Show your hand to the camera"
            (msg_w, _), _ = cv2.getTextSize(center_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
            draw_pill(display, center_msg, ((w - msg_w) // 2, h // 2),
                      font_scale=0.75, thickness=2,
                      text_color=(240, 240, 240), bg_color=(40, 40, 90))

        # ── Trigger Voice Announcement ──────────────────
        if announcements:
            combined_state = " and ".join(announcements)
            if combined_state != last_spoken_state:
                speaker.speak(combined_state)
                last_spoken_state = combined_state

        cv2.imshow("Hand & Finger Detector (OpenCV)", display)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):   # Q or ESC to quit
            break
        elif key == ord('c'):       # Switch to next camera
            next_idx = (current_cam_idx + 1) % 3
            print(f"[CAMERA] Switching to camera index {next_idx}...")
            cap.release()
            new_cap, new_idx = open_camera(target_index=next_idx)
            if new_cap is not None:
                cap = new_cap
                current_cam_idx = new_idx
            else:
                print(f"[!] Could not open camera {next_idx}, returning to {current_cam_idx}...")
                cap, current_cam_idx = open_camera(target_index=current_cam_idx)
        elif key == ord('v'):       # Toggle Voice
            is_active = speaker.toggle()
            print(f"[Voice] Switched to: {'ON' if is_active else 'MUTED'}")
        elif key == ord('s'):       # Save Screenshot
            t_str = time.strftime("%Y%m%d_%H%M%S")
            s_path = os.path.join(screenshot_dir, f"hand_detection_{t_str}.png")
            cv2.imwrite(s_path, display)
            print(f"[SNAPSHOT] Screenshot saved: {s_path}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("Program exited cleanly.")

if __name__ == "__main__":
    main()
