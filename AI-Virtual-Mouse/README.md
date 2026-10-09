# AI Virtual Mouse — Image & Video Processing Project

An AI-powered virtual mouse that lets users control the Windows mouse cursor using hand gestures captured through a webcam. The application also supports drawing in the air using the index finger.

**Technologies:** Python 3.11, OpenCV, MediaPipe 0.10.21, PyAutoGUI, NumPy

## Team Members

- Nancy Khurana
- Tanmay Paliwal

## 1. Project Overview

The AI Virtual Mouse uses real-time computer vision and hand landmark detection to interpret hand gestures and translate them into mouse actions or drawing strokes.

For every webcam frame, the application:
1. Captures and mirrors the camera frame.
2. Converts the frame from BGR to RGB.
3. Detects and tracks hand landmarks using MediaPipe Hands.
4. Recognizes gestures using geometric rules and state machines.
5. Performs mouse actions or draws strokes on a virtual canvas.
6. Displays the processed frame with landmarks, FPS, and gesture status.

## 2. Features

- **Cursor Control:** Move the mouse cursor using the index fingertip.
- **Left Click:** Pinch the thumb and index finger.
- **Right Click:** Pinch the thumb and middle finger.
- **Air Drawing:** Draw using the index finger while the middle finger is folded.
- **Pen Lift:** Move the index and middle fingers upward together to stop drawing temporarily.
- **Smooth Cursor Movement:** Adaptive exponential moving average smoothing.
- **Stable Gesture Recognition:** Frame confirmation, hysteresis, cooldowns, and release detection.
- **Drawing Canvas:** Persistent strokes blended with the live camera frame.
- **Keyboard Controls:** Switch modes, clear the canvas, change pen colours, view help, and quit.
- **Safety Handling:** Releases held mouse buttons on exit, errors, or camera failure.
- **Webcam Error Handling:** Reports an error if the camera cannot be opened.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python 3.11 | Core application logic |
| OpenCV | Camera capture, image processing, drawing, and display |
| MediaPipe 0.10.21 | Hand detection and 21-landmark tracking |
| NumPy | Landmark arrays, geometric calculations, and drawing canvas |
| PyAutoGUI | Mouse movement and left/right clicks |

The application uses MediaPipe's `mp.solutions.hands` API.

## 4. Project Structure

```text
AI-Virtual-Mouse/
├── assets/
├── tests/
├── .gitignore
├── config.py
├── drawing.py
├── gesture_detector.py
├── hand_data.py
├── hand_tracker.py
├── main.py
├── mouse_controller.py
├── README.md
├── requirements.txt
└── utils.py
```

- `main.py` — Main application loop, keyboard input, and on-screen interface.
- `hand_tracker.py` — Webcam handling and MediaPipe integration.
- `hand_data.py` — Landmark data structures and landmark indices.
- `gesture_detector.py` — Finger-state detection and gesture state machines.
- `mouse_controller.py` — Cursor mapping, smoothing, clicking, and safety handling.
- `drawing.py` — Drawing canvas, stroke processing, and image blending.
- `config.py` — Configuration values and gesture thresholds.
- `utils.py` — FPS calculation and smoothing utilities.
- `tests/` — Automated tests for gesture logic and other supported behaviours.
- `requirements.txt` — Python dependencies.
- `assets/` — Project resources, if applicable.

## 5. Requirements

- Windows operating system
- Python 3.11
- A working webcam
- A mouse and display
- Internet access for installing dependencies

## 6. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/AI-Virtual-Mouse-IVP-Project.git
cd AI-Virtual-Mouse-IVP-Project/AI-Virtual-Mouse
```

Replace `YOUR-USERNAME` with your GitHub username.

If you downloaded the project ZIP instead, extract it and open the `AI-Virtual-Mouse` folder in VS Code.

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

### Step 3: Activate the Environment

In Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

If you are using Command Prompt instead:

```bat
venv\Scripts\activate.bat
```

### Step 4: Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Compatibility note:** MediaPipe 0.10.21 requires a compatible NumPy version. This project expects NumPy below 2.0; follow the pinned versions in `requirements.txt`.

Use a fresh virtual environment to avoid conflicts with previously installed packages.

### Step 5: Run the Application

```bash
python main.py
```

Allow camera access if Windows requests permission. Keep the camera window visible while using gestures.

## 7. Gesture Controls

### Mouse Mode (Default)

| Action | Gesture | Description |
|---|---|---|
| Move cursor | Point with the index finger | Maps the active camera region to screen coordinates |
| Left click | Pinch thumb and index finger | Triggers a left click when the pinch is confirmed |
| Right click | Pinch thumb and middle finger | Triggers a right click when the pinch is confirmed |
| Double click | Perform two quick left pinches | Windows may interpret them as a double click |

A pinch triggers one click; holding the pinch does not continuously repeat the click. Open the fingers before performing the next click.

### Drawing Mode

Press `D` while the camera window is focused to enter drawing mode. Press `D` again to return to mouse mode.

| Action | Gesture |
|---|---|
| Draw | Index finger extended, middle finger folded |
| Lift pen | Index and middle fingers extended together |
| Move without drawing | Open hand, fist, or another supported pen-up gesture |

### Keyboard Shortcuts

The camera window must have keyboard focus.

| Key | Action |
|---|---|
| `D` | Toggle drawing mode |
| `C` | Clear the drawing canvas |
| `H` | Display help |
| `1–5` | Select pen colour |
| `Q` or `Esc` | Quit the application |

Drag-and-drop and scrolling are not supported in this version because they conflicted with reliable click recognition.

## 8. Image and Video Processing Concepts

This project demonstrates several concepts from Image and Video Processing.

### 8.1 Image Acquisition

`cv2.VideoCapture()` captures live frames from the webcam. The application processes these frames continuously.

### 8.2 Frame Processing

Each frame is mirrored using `cv2.flip()` to provide a natural, selfie-style interaction.

### 8.3 Colour Space Conversion

OpenCV captures images in BGR format, whereas MediaPipe expects RGB input.

```python
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
```

This converts the frame before hand landmark detection.

### 8.4 Hand and Landmark Detection

MediaPipe Hands detects and tracks 21 landmarks on each hand. The landmarks provide normalized coordinates and relative depth information.

Important landmarks include:

- Thumb tip: `4`
- Index fingertip: `8`
- Middle fingertip: `12`
- Wrist: `0`

These landmarks are used to identify finger positions and recognize gestures.

### 8.5 Coordinate Transformation

Landmark coordinates are normalized between 0 and 1. They are converted into camera pixel coordinates and mapped to screen coordinates so that hand movement controls the cursor.

### 8.6 Gesture Recognition

Gestures are recognized using finger extension, joint angles, and distances between fingertips.

The pinch distance is normalized using palm size, helping the system work at different distances from the camera.

### 8.7 Smoothing

Exponential moving average (EMA) smoothing reduces cursor jitter.

The general formula is:

\[
s_t = s_{t-1} + \alpha(x_t-s_{t-1})
\]

Where:
- \(s_t\) is the smoothed position.
- \(s_{t-1}\) is the previous smoothed position.
- \(x_t\) is the current raw position.
- \(\alpha\) controls the degree of smoothing.

### 8.8 Real-Time Video Processing

The application processes webcam frames continuously and displays the results with gesture information and FPS measurements.

### 8.9 Image Blending

The drawing canvas is combined with the live frame using OpenCV's `cv2.addWeighted()` function.

\[
I_{\text{output}}=(1-\alpha)I_{\text{frame}}+\alpha I_{\text{canvas}}
\]

This overlays the drawing on the live camera image.

### 8.10 Gesture Stability

Frame confirmation, hysteresis, cooldowns, and finite-state machines help reduce accidental clicks and unstable gesture transitions.

## 9. Testing

The project includes automated tests for supported logic and simulated scenarios.

Run the tests from the project directory:

```bash
python -m unittest discover -s tests -v
```

These tests do not require a webcam if they only exercise the headless logic included in the test suite.

## 10. Troubleshooting

| Problem | Possible Solution |
|---|---|
| Unable to open webcam | Close applications using the camera or try another camera index in the configuration |
| Keyboard shortcuts do not work | Bring the camera window into focus |
| Low FPS | Improve lighting, close resource-intensive applications, or use a lower model complexity |
| Cursor jitters | Adjust the smoothing parameters in `config.py` |
| Clicks trigger too easily or not at all | Adjust the pinch and release thresholds |
| Accidental right clicks | Tune the middle-finger pinch settings and confirmation frames |
| Drawing looks shaky | Adjust drawing smoothing and curve parameters |
| Pen lifts unexpectedly | Adjust the pen release settings and keep the hand facing the camera |
| Packages conflict | Recreate the virtual environment and reinstall the pinned requirements |
| Elevated application is not controlled | Check Windows permission levels and run the application with appropriate privileges if required |

**Emergency stop:** Move the physical mouse to the top-left corner to activate PyAutoGUI's fail-safe, or press `Ctrl+C` in the terminal.

## 11. Limitations

- Designed for Windows.
- Requires a working webcam and sufficient lighting.
- Gesture recognition can be affected by occlusion, camera quality, and hand orientation.
- Performance depends on hardware and the environment.
- Drag-and-drop and scrolling are not implemented in this version.

## 12. Future Improvements

- Multi-hand gesture support for zooming and additional controls.
- Eraser and shape-drawing tools.
- Export drawings as PNG images.
- Gesture calibration wizard.
- Advanced cursor smoothing using a One Euro filter or Kalman prediction.
- On-screen virtual keyboard.
- Additional accessibility features.

## 13. Conclusion

The AI Virtual Mouse demonstrates how computer vision, hand landmark detection, geometric gesture recognition, and real-time image processing can be combined to create a touch-free mouse interface.

The project also demonstrates image acquisition, colour space conversion, coordinate transformation, smoothing, image blending, and real-time video processing.

## Project Report

The detailed project report is included in this repository for academic evaluation.
