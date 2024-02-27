This project was developed during a group project under the supervision of a Professor, specializing in image processing. It explores the use of facial and hand landmark detection for potential applications in lie detection. It leverages libraries like mediapipe, opencv-python, and fer to accomplish various functionalities.

Note: This project is a work-in-progress (WIP) and does not aim to be a conclusive method for lie detection.

Dependencies (Using python 3.7.X)
fer==22.4.0
ffpyplayer==4.3.5
matplotlib==3.5.1
mediapipe==0.9.0
mss==6.1.0
numpy==1.22.2
opencv_contrib_python==4.5.5.64
scipy==1.8.0
tensorflow==2.10.0

To run it:
  python intercept.py --input 0 --landmarks LANDMARKS --bpm BPM --flip FLIP --ttl TTL --record RECORD

all options:
  --source: Specify video source (e.g., 0 for webcam, video.mp4 for file, screen for screen capture) (default: 0)
  --landmarks: Enable drawing of landmarks and metrics (default: True)
  --record: Enable recording the output video (default: False)
  --bpm: Enable heart rate detection using webcam (requires external library) (default: False)
  --flip: For mirroring view
  --ttl: For number of frames
