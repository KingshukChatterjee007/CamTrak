# CamTrak

CamTrak is a Python-based computer vision application that allows you to control your computer's mouse cursor and perform clicks using hand gestures via a webcam. Built with MediaPipe and OpenCV, it features a high-precision, lag-free engine for a smooth user experience.

## Features

* **Mouse Movement:** Control the cursor by moving your Index finger.
* **Left Click:** Perform a left click by pinching your Thumb and Index finger together.
* **Right Click:** Perform a right click by double-pinching your Thumb and Ring finger.
* **Distance-Invariant Tracking:** Uses ratio-based thresholds (pinch distance relative to palm size) so gestures work perfectly at any distance from the camera.
* **Ultra-Precision Smoothing:** Implements a 1 Euro Filter to eliminate stationary jitter and provide highly responsive motion.
* **Lag-Free Video:** Uses a threaded camera stream to process frames in the background, preventing input lag.
* **Heads-Up Display (HUD):** Real-time visual feedback of your tracking status and active gestures.

## Prerequisites

Ensure you have Python 3.x installed. The following libraries are required:

* `opencv-python`
* `mediapipe`
* `pyautogui`
* `numpy`

You can install the dependencies using pip:

```bash
pip install opencv-python mediapipe pyautogui numpy
