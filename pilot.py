import pandas as pd
import numpy as np
import os

# Load txt file
file_path = './LeSaS1/Data/'
dataframes = []

# Iterate over each file in the directory
for file in os.listdir(file_path):
    if file.endswith('.txt'):
        full_path = os.path.join(file_path, file)
        # Read the txt file into a DataFrame
        df = pd.read_csv(full_path, sep='\t', skiprows=1)
        dataframes.append(df)

# Concatenate all the DataFrames into one
concatenated_df = pd.concat(dataframes, ignore_index=True)

# Display the concatenated DataFrame
print(concatenated_df['OutcomeValue'].mean())