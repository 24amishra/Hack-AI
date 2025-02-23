from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import cross_val_score
import numpy as np

# Load and prepare data function
def load_and_prepare_data():
    # Load all CSV files
    good_angles = pd.read_csv('DataOutputs/Goodform.csv')
    bad_angles = pd.read_csv('DataOutputs/RoundBack.csv')
    good_distances = pd.read_csv('DataOutputs/distanceGood.csv')
    wide_distances = pd.read_csv('DataOutputs/distanceWideGrip.csv')
    good_knees = pd.read_csv('DataOutputs/FilteredGoodKnee.csv')
    bad_knees = pd.read_csv('DataOutputs/FilteredBadKnee.csv')

    # Prepare data for back angle classification (0: good_form, 1: rounded_back)
    good_angles['label'] = 0
    bad_angles['label'] = 1
    angles_data = pd.concat([good_angles, bad_angles])

    # Prepare data for grip width classification (0: good_grip, 2: wide_grip)
    good_distances['label'] = 0
    wide_distances['label'] = 2
    distances_data = pd.concat([good_distances, wide_distances])

    # Prepare data for knee angle classification (0: good_knees, 3: bad_knees)
    good_knees['label'] = 0
    bad_knees['label'] = 3
    knees_data = pd.concat([good_knees, bad_knees])

    # Rename columns for consistency
    for data in [angles_data, distances_data, knees_data]:
        data.columns = ['features', 'label']

    return angles_data, distances_data, knees_data

# Train and evaluate a classifier for a binary classification task
def train_and_evaluate_classifier(X, y):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the Gradient Boosting Classifier
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Evaluate the classifier
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Use cross-validation to evaluate the model
    cross_val_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cross_val_scores}")
    print(f"Mean accuracy: {cross_val_scores.mean()}")

    return model, scaler  # Return the trained model and scaler

# Load and prepare data
angles_data, distances_data, knees_data = load_and_prepare_data()

# 1. Back Angle Classification (Good Form vs Rounded Back)
X_angles = angles_data['features'].apply(pd.Series)  # Convert to multiple columns if necessary
y_angles = angles_data['label']
print("Angle Classifications Training Result: ")
model_angles, scaler_angles = train_and_evaluate_classifier(X_angles, y_angles)

# 2. Grip Width Classification (Good Grip vs Wide Grip)
X_distances = distances_data['features'].apply(pd.Series)  # Convert to multiple columns if necessary
y_distances = distances_data['label']
print("Distance Classifications Training Result: ")

model_distances, scaler_distances = train_and_evaluate_classifier(X_distances, y_distances)

# 3. Knee Angle Classification (Good Knees vs Bad Knees)
X_knees = knees_data['features'].apply(pd.Series)  # Convert to multiple columns if necessary
y_knees = knees_data['label']
print("Knee Angle Classifications Training Result: ")

model_knees, scaler_knees = train_and_evaluate_classifier(X_knees, y_knees)

# Now to test the model on new files (example angle and distance files)

# Function to test a new file on the model
def test_new_file(file_path, model, scaler):
    new_data = pd.read_csv(file_path)

    # Preprocess new data (ensure the same format as the training data)
    X_new = new_data['features'].apply(pd.Series)  # Split 'features' into separate columns if needed

    # Scale the new data using the same scaler
    X_new_scaled = scaler.transform(X_new)

    # Make predictions
    probabilities = model.predict_proba(X_new_scaled)  # Get probability scores
    safety_score = probabilities[:, 0].mean()  # Average probability of "good form" (label 0)
    print("Safety Score:", safety_score)
    np.set_printoptions(threshold=np.inf)
    print(probabilities)
    

    return probabilities

# Example: Test new angle and distance files
# Test on angle file
angle_predictions = test_new_file('/Users/agastyamishra/Downloads/HackAI/Hack-AI/ExampleAngle.csv', model_angles, scaler_angles)
print(f"Predictions for the example angle file: {angle_predictions}")
import matplotlib.pyplot as plt

# Assuming df is your DataFrame with 'x' and 'y' columns
plt.figure(figsize=(8, 5))
plt.plot(angle_predictions[0], angle_predictions[1], marker='o', linestyle='-')  # Line plot

plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('X vs. Y Plot')
plt.grid(True)
plt.show()


