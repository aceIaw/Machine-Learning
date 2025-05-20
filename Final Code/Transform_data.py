import pandas as pd


file_path = r"Master Inge/Machine learning/Projet/raw_data.csv" # Path to the raw data file
df = pd.read_csv(file_path, header=None) # Read the CSV file without headers

df.columns = ['user', 'gesture', 'sequence', 'x', 'y', 'z', 'time'] # Assign column names

sensor_columns = ['x', 'y', 'z'] # Columns to be processed

# If you only want your raw data, keep false for both
standardize = False 
normalize = False

for col in sensor_columns: # Iterate over sensor columns

    if standardize: # Standardization
        std = df[col].std() # Calculate standard deviation
        df[col] = (df[col] - df[col].mean()) / std # Standardize the column

    elif normalize: # Normalization
        min_val = df[col].min() # Calculate minimum value
        max_val = df[col].max() # Calculate maximum value
        df[col] = (df[col] - min_val) / (max_val - min_val) # Normalize the column


output_suffix = "standardize" if standardize else "normalize" if normalize else "raw_data_with_columns" # Suffix for the output file
output_path = fr"C:\Users\RS10\Documents\Python\Master Inge\Machine learning\Projet\{output_suffix}.csv" # Path to save the processed data
df.to_csv(output_path, index=False)

# Print statistics
print("\nProcessed Data Statistics:")
print("Means:\n", df[sensor_columns].mean())
print("Standard Deviations:\n", df[sensor_columns].std())
print("Min values:\n", df[sensor_columns].min())
print("Max values:\n", df[sensor_columns].max())

