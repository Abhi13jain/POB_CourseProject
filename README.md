⚙️ Features

🧠 Pose Estimation: Uses MoveNet to detect 17 body keypoints in real time.

📏 Calibration: Press ‘c’ to calibrate your baseline (good) posture distance.

🧍 Posture Evaluation: Compares current head-to-shoulder distance and tilt angle with the calibrated baseline to detect slouching or leaning.

🎯 Angle Analysis: Calculates head tilt and shoulder tilt angles for more accurate detection.

💡 Visual Feedback: Displays keypoints, lines, posture status (“Good posture” / “Bad posture”), and FPS on screen.

⚡ Optimized for Speed: Uses TensorFlow Hub’s lightweight 192×192 MoveNet model for low latency inference.

🧰 Requirements
🐍 Python Dependencies

Install the following packages before running the script:

pip install tensorflow tensorflow-hub opencv-python numpy

🖥️ Hardware

A system with a webcam

(Optional) GPU acceleration for TensorFlow (for higher FPS)

▶️ Usage Instructions

Clone or Download the repository.

Run the Python script:

python posture_monitor.py


The webcam will open and begin pose detection.

Controls:

c → Calibrate baseline posture (sit straight)

r → Reset calibration

q → Quit the application

🧩 How It Works
1. Pose Detection

The MoveNet model outputs 17 keypoints (e.g., nose, shoulders, elbows, hips, etc.), each with coordinates (y, x, confidence).

The script extracts the nose, left shoulder, and right shoulder keypoints for posture analysis.

2. Posture Analysis

Computes the distance between the head and shoulder center to estimate how far the head leans forward.

Calculates head tilt angle and shoulder tilt (in degrees) using trigonometric functions.

Uses a baseline calibration (the “good posture” distance) to detect deviations:

If current distance < 80% of baseline and tilt exceeds threshold → Bad Posture

Otherwise → Good Posture

3. Stability and Smoothing

Uses Exponential Moving Average (EMA) to smooth keypoint positions over time and avoid jitter.

4. Visualization

Draws detected keypoints and connecting lines.

Displays posture status, FPS, and live data overlays on the frame using OpenCV.

📊 Example Output

Green text: Good posture

Red text: Bad posture

Yellow text: Calibration message

On-screen metrics:

Head–shoulder distance and ratio

Head tilt and shoulder tilt (in degrees)

Live FPS counter

🧪 Future Enhancements

🔔 Add audio alerts for prolonged bad posture

📈 Log posture data over time for analysis

🌐 Integrate with IoT or health tracking dashboards

📷 Support multiple person detection using MoveNet MultiPose model

🏁 Conclusion

This project demonstrates how TensorFlow MoveNet can be leveraged for real-time human pose estimation and posture tracking. It can be extended for ergonomic assessment, exercise monitoring, or workplace health applications.
