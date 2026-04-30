# 🎥 CamTrak 2.8 | Ultra-Precision Gestural Interface

CamTrak is a high-performance, scale-invariant virtual mouse system that transforms your webcam into a precision spatial controller. Leveraging MediaPipe's cutting-edge neural networks and custom signal filtering, CamTrak provides a seamless, touchless computing experience.

---

## ✨ Key Features

- **🎯 Ultra-Precision Engine**: Uses the **1 Euro Filter** algorithm to eliminate jitter while maintaining high-speed responsiveness.
- **📏 Scale-Invariant Logic**: Gesture recognition is based on hand-to-palm ratios, ensuring perfect performance whether you are close to or far from the camera.
- **🖐️ Intuitive Gesture Set**:
  - **Cursor Movement**: Track your index finger with smooth, adaptive acceleration.
  - **Left Click & Drag**: Pinch your thumb and index finger.
  - **Right Click**: Double pinch with your thumb and ring finger.
  - **Smart Scroll**: Open your palm and move vertically to scroll through documents or web pages.
- **🖥️ Heads-Up Display (HUD)**: Real-time visual feedback with a minimalist, semi-transparent interface.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- A working webcam

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/KingshukChatterjee007/CamTrak.git
   cd CamTrak
   ```

2. **Install dependencies**:
   ```bash
   pip install opencv-python mediapipe pyautogui numpy
   ```

3. **Download the Task Model**:
   Ensure `hand_landmarker.task` is in the root directory.

### Running CamTrak
```bash
python camtrak.py
```

---

## 🛠️ Technology Stack

- **MediaPipe**: Hand landmark detection and tracking.
- **PyAutoGUI**: System-level mouse and keyboard control.
- **OpenCV**: Image processing and HUD rendering.
- **One Euro Filter**: Advanced signal smoothing for surgical precision.

---

## 📜 License
This project is for educational and experimental purposes. See the license file for details.

---
*Developed by [Kingshuk Chatterjee](https://github.com/KingshukChatterjee007)*
