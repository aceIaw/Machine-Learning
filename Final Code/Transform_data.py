import pandas as pd

# Load the CSV file
file_path = r"Master Inge/Machine learning/Projet/raw_data.csv"
df = pd.read_csv(file_path, header=None)  # Use header=None if the file has no header

# Rename columns from 1 to 7
df.columns = ['user', 'gesture', 'sequence', 'x', 'y', 'z', 'time']

# Select only the sensor data columns for processing
sensor_columns = ['x', 'y', 'z']

# Configuration, if you only want the raw data with columns names set both to False
standardize = False
normalize = False

# Clean + preprocess
for col in sensor_columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mean())

    if standardize:
        std = df[col].std()
        if std == 0:
            print(f"Skipping standardization for '{col}' due to zero std.")
        else:
            df[col] = (df[col] - df[col].mean()) / std

    elif normalize:
        min_val = df[col].min()
        max_val = df[col].max()
        if max_val - min_val == 0:
            print(f"Skipping normalization for '{col}' due to zero range.")
        else:
            df[col] = (df[col] - min_val) / (max_val - min_val)

# Save the final result
output_suffix = "standardize" if standardize else "normalize" if normalize else "raw_data_with_columns"
output_path = fr"C:\Users\RS10\Documents\Python\Master Inge\Machine learning\Projet\{output_suffix}.csv"
df.to_csv(output_path, index=False)

# Print statistics
print("\nProcessed Data Statistics:")
print("Means:\n", df[sensor_columns].mean())
print("Standard Deviations:\n", df[sensor_columns].std())
print("Min values:\n", df[sensor_columns].min())
print("Max values:\n", df[sensor_columns].max())

