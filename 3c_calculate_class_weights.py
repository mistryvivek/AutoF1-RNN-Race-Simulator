"""
Script used to calculate class weights using Skilearn.
"""

import os
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

def calculate_class_weights(training_folder, column_name, session_filter="Race"):
    data = []

    # Iterate through all files in the Training folder
    for file_name in os.listdir(training_folder):
        if file_name.endswith(".csv"):
            file_path = os.path.join(training_folder, file_name)
            try:
                # Read the CSV file
                df = pd.read_csv(file_path)
                # Check if required columns exist
                if "Session" in df.columns and column_name in df.columns:
                    # Filter rows where SessionType matches the filter
                    filtered_data = df[df["Session"] == session_filter]

                    # Handle specific transformations for the Compound column
                    if column_name == "Compound":
                        filtered_data = filtered_data[filtered_data["Compound"] != "UNKNOWN"]  # Drop unknown values
                        filtered_data["Compound"] = filtered_data["Compound"].replace(
                            {"HYPERSOFT": "SOFT", "ULTRASOFT": "SOFT", "SUPERSOFT": "SOFT"}
                        )

                    # Handle specific transformations for the PitInTime column
                    if column_name == "PitInTime":
                        # Convert to binary: 1 if not NaN, 0 if NaN
                        filtered_data["PitInTime"] = filtered_data["PitInTime"].notna().astype(int)

                    # Collect data for the specified column
                    data.extend(filtered_data[column_name].tolist())
            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

    # Remove NaN values from data
    data = [value for value in data if not pd.isna(value)]

    # Compute class weights if data is available
    if data:
        unique_classes = np.unique(data)
        class_weights = compute_class_weight(
            class_weight="balanced",
            classes=unique_classes,
            y=data
        )
        print(f"Class Weights for {column_name}:")
        for cls, weight in zip(unique_classes, class_weights):
            print(f"{column_name} {cls}: {weight:.4f}")
    else:
        print(f"No data found for column '{column_name}' in sessions labeled '{session_filter}'.")

if __name__ == "__main__":
    training_folder = os.path.abspath("../Data Gathering")
    calculate_class_weights(training_folder, "nGear")
    calculate_class_weights(training_folder, "Compound")
    calculate_class_weights(training_folder, "PitInTime")