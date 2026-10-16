# AirHoops

A motion-controlled basketball game built with Python. Using your webcam and MediaPipe tracking, a virtual ball falls from the top of the screen and you move your hand or body to catch it and score points. No special hardware needed, just a regular laptop webcam.

## Requirements
- Windows, Python 3.10 or newer
- A webcam (built-in or USB)

## Install
1. Download or clone the project.
2. Open a terminal in the project folder and create a virtual environment:
```
   python -m venv venv
   venv\Scripts\activate
```
3. Install the libraries:
```
   pip install opencv-python mediapipe numpy
```
   (Add any other libraries your code imports.)

## Run
```
python main.py
```
(Replace `main.py` with your game's file name.)

## How to play
1. Stand or sit so your upper body or hand is clearly visible.
2. Move your hand or body to catch the falling ball.
3. Each catch scores a point.
4. Press `q` to quit.

## Tips
- Use good, even lighting.
- Keep a plain background if possible.
- If the webcam hangs, use `cv2.VideoCapture(0, cv2.CAP_DSHOW)` and close other apps using the camera.
