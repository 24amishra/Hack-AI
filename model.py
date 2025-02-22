import cv2 as cv
import numpy as np
import mediapipe as mp

# Load MediaPipe pose detection
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Path to model
model_path = 'pose_landmarker_lite (1).task'

# Create a PoseLandmarker instance in VIDEO mode
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO
)

# Open video file
cap = cv.VideoCapture('IMG_5901.mov')

# Get frame rate
frame_rate = cap.get(cv.CAP_PROP_FPS)
frame_count = 0  # Track frame index

# Process video using PoseLandmarker
with PoseLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()  # Read a frame
        if not ret:
            break  # Exit if video ends

        # Convert frame from OpenCV format (BGR) to RGB
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # Convert frame to MediaPipe image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Get timestamp for this frame
        timestamp_ms = frame_count * (1000 / frame_rate)

        # Run pose estimation (must use detect_for_video in VIDEO mode)
        pose_result = landmarker.detect_for_video(mp_image, timestamp_ms=int(timestamp_ms))

        # Draw pose landmarks on frame
        if pose_result and pose_result.pose_landmarks:
            for landmark in pose_result.pose_landmarks[0]:  # Assuming single person
                x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                cv.circle(frame, (x, y), 3, (0, 255, 0), -1)  # Draw green circles

        # Show the frame
        cv.imshow("Pose Detection", frame)

        # Wait for 'q' key to exit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1  # Increment frame count

cap.release()
cv.destroyAllWindows()
