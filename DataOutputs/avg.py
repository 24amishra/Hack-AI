import pandas as pd

 

# Load the CSV file (update 'data.csv' with your actual file path)
df = pd.read_csv("/Users/agastyamishra/Downloads/HackAI/Hack-AI/DataOutputs/GoodKneeBendAngle.csv")

# Compute the mean of all numeric columns
averages = df.mean(numeric_only=True)

# Print the results
print(averages)
