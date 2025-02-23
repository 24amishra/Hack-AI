from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
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

    # Reset and rename columns if necessary
    for data in [angles_data, distances_data, knees_data]:
        if data.shape[1] > 2:
            data = data.reset_index(drop=True)
        if data.shape[1] == 2:
            data.columns = ['Feature', 'label']

    return angles_data, distances_data, knees_data

# Train and evaluate a classifier for a binary classification task
def train_and_evaluate_classifier(X, y):
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(X_test)
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
    
    return model, X_train_scaled, X_test_scaled, y_train, y_test  # Return model and data for regression

# Regression using the predicted probabilities from classifier
# Call to load and prepare data
angles_data, distances_data, knees_data = load_and_prepare_data()

# Example: Use angles_data for classification
X_angles = angles_data.drop(columns=['label'])
y_angles = angles_data['label']

# Train and evaluate classifier for back angle classification (good form vs. rounded back)
model_angles, X_train_angles, X_test_angles, y_train_angles, y_test_angles = train_and_evaluate_classifier(X_angles, y_angles)

# Example: Use distances_data for classification
X_distances = distances_data.drop(columns=['label'])
y_distances = distances_data['label']

# Train and evaluate classifier for grip width classification (good grip vs. wide grip)
model_distances, X_train_distances, X_test_distances, y_train_distances, y_test_distances = train_and_evaluate_classifier(X_distances, y_distances)

# Example: Use knees_data for classification
X_knees = knees_data.drop(columns=['label'])
y_knees = knees_data['label']

# Train and evaluate classifier for knee angle classification (good knees vs. bad knees)
model_knees, X_train_knees, X_test_knees, y_train_knees, y_test_knees = train_and_evaluate_classifier(X_knees, y_knees)


# Example single input (a new data point for back angle classification)
new_input = np.array([0.25, -1.1])  # This should match the number of features in X (two features here as per your data)

# Reshape the input to be 2D (single sample, multiple features)
new_input_reshaped = new_input.reshape(1, -1)

# Scale the input using the same scaler used for training

# Make a prediction using the trained model
prediction = model_angles.predict()

# Print the prediction (0: good form, 1: rounded back)
print(f"Prediction: {prediction[0]}")