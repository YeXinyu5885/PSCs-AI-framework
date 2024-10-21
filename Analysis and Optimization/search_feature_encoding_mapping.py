
import pandas as pd
from category_encoders import TargetEncoder
import json

# Load the CSV file (adjust to a descriptive name)
data = pd.read_csv("customer_psv_data.csv", encoding='latin1')

# Fill missing values with column means for numeric columns
data.fillna(data.mean(numeric_only=True), inplace=True)

# Define the target column for encoding
target = 'JV_default_PCE'

# Initialize the target encoder
encoder = TargetEncoder()

# Identify non-numeric columns (for target encoding)
non_numeric_columns = data.select_dtypes(include=['object', 'category']).columns

# Dictionary to store the encoding map for each category
encoding_maps = {}

# Apply target encoding to each non-numeric column
for column in non_numeric_columns:
    if column != target:  # Ensure the target column is not included in the encoding
        # Perform target encoding
        data[column + '_encoded'] = encoder.fit_transform(data[column], data[target])
        
        # Store the encoding map for future reference
        encoding_maps[column] = dict(zip(data[column], data[column + '_encoded']))

# Remove the original non-numeric columns after encoding
data.drop(non_numeric_columns, axis=1, inplace=True)

# Save the processed data with encoded columns to a new CSV file
data.to_csv("encoded_customer_psv_data.csv", index=False)

# Optionally save the encoding maps as a JSON file for reference
with open('encoding_maps.json', 'w') as file:
    json.dump(encoding_maps, file)

# To load and inspect encoding maps
with open('encoding_maps.json', 'r') as file:
    encoding_maps = json.load(file)

# Query the encoding map for a specific column (change column_name to inspect)
column_name = 'Cell_architecture'  # Replace with the actual column name you want to query
if column_name in encoding_maps:
    # Print the original and encoded values
    for original_value, encoded_value in encoding_maps[column_name].items():
        print(f"Original: {original_value}, Encoded: {encoded_value}")
else:
    print(f"No encoding map found for column: {column_name}")
