import cv2 as cv
import numpy as np
import mediapipe as mp
import pandas as pd
import math
FILENAME = '/Users/agastyamishra/Downloads/HackAI/Hack-AI/TrainingData/IMG_5917 (2).MOV'

# Load MediaPipe pose detection
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def calculate_angle(point1, point2, point3):
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)
def calculate_distance(point1,point2):
    a = np.array(point1)
    b = np.array(point2)

    return np.linalg.norm(a - b)

# Path to model
model_path = 'pose_landmarker_lite (1).task'

# Create a PoseLandmarker instance in VIDEO mode
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO
)

# Open video file
cap = cv.VideoCapture(FILENAME)

# Get frame rate
frame_rate = cap.get(cv.CAP_PROP_FPS)
frame_count = 0  # Track frame index

## Filter in required/desired nodes
solutions = mp.solutions.pose.PoseLandmark
filtered_nodes = [solutions.LEFT_SHOULDER,
                  solutions.RIGHT_SHOULDER,
                  solutions.LEFT_HIP,
                  solutions.RIGHT_HIP,
                  solutions.LEFT_KNEE,
                  solutions.RIGHT_KNEE,
                  solutions.LEFT_HEEL,
                  solutions.LEFT_ANKLE,
                  solutions.LEFT_INDEX
                  ]

# Create a DataFrame to store angles
angle_df = pd.DataFrame(columns=["Angle"])
distance_df = pd.DataFrame(columns=["Distance"])

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

        # Initialize empty DataFrames for each landmark
        leftShoulder = pd.DataFrame(columns=['X', 'Y'])
        leftHip = pd.DataFrame(columns=['X', 'Y'])
        leftKnee = pd.DataFrame(columns=['X', 'Y'])
        leftHeel = pd.DataFrame(columns=['X', 'Y'])
        leftAnkle = pd.DataFrame(columns=['X', 'Y'])
        leftIndex = pd.DataFrame(columns=['X', 'Y'])

        # Draw pose landmarks on frame
        if pose_result and pose_result.pose_landmarks:
            for landmark in filtered_nodes:  # Assuming single person
                node = landmark
                landmark = pose_result.pose_landmarks[0][landmark]
                
                x, y ,z= int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]),int(landmark.z * frame.shape[2])

                # Create a new row with the x, y values
                new_row = pd.DataFrame([[x, y,z]], columns=['X', 'Y','Z'])

                # Append the new row to the corresponding DataFrame based on the node
                if node == solutions.LEFT_SHOULDER:
                    leftShoulder = pd.concat([leftShoulder, new_row], ignore_index=True)
                elif node == solutions.LEFT_HIP:
                    leftHip = pd.concat([leftHip, new_row], ignore_index=True)
                elif node == solutions.LEFT_KNEE:
                    leftKnee = pd.concat([leftKnee, new_row], ignore_index=True)
                elif node == solutions.LEFT_HEEL:
                    leftHeel = pd.concat([leftHeel, new_row], ignore_index=True)
                elif node == solutions.LEFT_ANKLE:
                    leftAnkle = pd.concat([leftAnkle, new_row], ignore_index=True)
                elif node == solutions.LEFT_INDEX:
                        leftIndex = pd.concat([leftIndex, new_row], ignore_index=True)






                cv.circle(frame, (x, y), 3, (0, 255, 0), -1)  # Draw green circles

        for x in range(0, leftShoulder.shape[0]):
            # Access 'X' and 'Y' values for each landmark in the corresponding DataFrames
            point1 = (leftShoulder.iloc[x]['X'], leftShoulder.iloc[x]['Y'])
            point2 = (leftHip.iloc[x]['X'], leftHip.iloc[x]['Y'])
            point3 = (leftKnee.iloc[x]['X'], leftKnee.iloc[x]['Y'])
            
            
            pointShoulder = (leftShoulder.iloc[x]['X'], leftShoulder.iloc[x]['Y'], leftShoulder.iloc[x]['Z'])
            pointIndex = (leftIndex.iloc[x]['X'], leftIndex.iloc[x]['Y'], leftIndex.iloc[x]['Z'])

            bendHip = (leftHip.iloc[x]['X'], leftHip.iloc[x]['Y'])  
            bendAnkle =(leftAnkle.iloc[x]['X'], leftAnkle.iloc[x]['Y'])   
            bendKnee =  (leftKnee.iloc[x]['X'], leftKnee.iloc[x]['Y'])  


            # Calculate angle
            angle = calculate_angle(point1, point2, point3)
            
            distance = calculate_distance(pointShoulder,pointIndex)

            # Add the calculated angle to the DataFrame
            angle_df = pd.concat([angle_df, pd.DataFrame({"Angle": [angle]})], ignore_index=True)
            distance_df = pd.concat([distance_df, pd.DataFrame({"Distance": [distance]})], ignore_index=True)
        # Show the frame
        cv.imshow("Pose Detection", frame)

        # Wait for 'q' key to exit
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1  # Increment frame count

# Write the angles DataFrame to a CSV file after processing all frames
angle_df.to_csv('ExampleAngle.csv', index=False)
distance_df.to_csv("exampleDistance.csv",index = False)

cap.release()
cv.destroyAllWindows()

print("Distances saved to distanceIndex.csv")
