import pandas as pd
import numpy as np
import os
from utils.VisualSearchModels import dict_generator_VS, VisualSearchModels
from utils.ComputationalModeling import dict_generator
from utils.DualProcess import DualProcessModel

# Load txt file
file_path = './LeSaS1/Data/'
dataframes = []

# Iterate over each file in the directory
for file in os.listdir(file_path):
    if file.endswith('.txt') and file.startswith('Data_'):
        full_path = os.path.join(file_path, file)
        # Extract subject number from filename; We need this because:
        # SubID 40 occurs in both 40 and 42;
        # SubID 41 occurs in both 41 and 43;
        # SubID 45 occurs in both 45 and 53
        subject_number = int(file.split('_')[2].split('.')[0])
        # Read the txt file into a DataFrame
        df = pd.read_csv(full_path, sep='\t', skiprows=1)
        # Update SubNo column with the correct subject number
        df['SubNo'] = subject_number
        dataframes.append(df)

# Concatenate all the DataFrames into one
concatenated_df = pd.concat(dataframes, ignore_index=True)

# Remove practice trials (Block 0)
concatenated_df = concatenated_df[concatenated_df['Block'] != 0]

# Remove non-responses
concatenated_df = concatenated_df[~concatenated_df['Optimal_Choice'].isna()]

# Change the dtype of optimal choice to int
concatenated_df['Optimal_Choice'] = concatenated_df['Optimal_Choice'].astype(int)

# Change the unit of the RT column to seconds
concatenated_df['RT'] = concatenated_df['RT'] / 1000

# Display the concatenated DataFrame
data_dict = dict_generator(concatenated_df)

# get the first subject as testing data
test_data = concatenated_df[concatenated_df['SubNo'] == 1]
test_dict = dict_generator(test_data)

if __name__ == "__main__":
    delta = VisualSearchModels('delta')
    decay = VisualSearchModels('decay')
    WSLS = VisualSearchModels('WSLS')
    WSLS_delta = VisualSearchModels('WSLS_delta')
    WSLS_delta_weight = VisualSearchModels('WSLS_delta_weight')
    WSLS_decay_weight = VisualSearchModels('WSLS_decay_weight')
    dual_process = DualProcessModel(task='IGT_SGT', num_options=2)

    # Fit the model to the data
    delta_results = delta.fit(data_dict, num_iterations=100)
    decay_results = decay.fit(data_dict, num_iterations=100)
    WSLS_results = WSLS.fit(data_dict, num_iterations=100)
    WSLS_delta_results = WSLS_delta.fit(data_dict, num_iterations=100)
    WSLS_delta_weight_results = WSLS_delta_weight.fit(data_dict, num_iterations=100)
    WSLS_decay_weight_results = WSLS_decay_weight.fit(data_dict, num_iterations=100)

    delta_results.to_csv('./LeSaS1/Model/delta_results.csv', index=False)
    decay_results.to_csv('./LeSaS1/Model/decay_results.csv', index=False)
    WSLS_results.to_csv('./LeSaS1/Model/WSLS_results.csv', index=False)
    WSLS_delta_results.to_csv('./LeSaS1/Model/WSLS_delta_results.csv', index=False)
    WSLS_delta_weight_results.to_csv('./LeSaS1/Model/WSLS_delta_weight_results.csv', index=False)
    WSLS_decay_weight_results.to_csv('./LeSaS1/Model/WSLS_decay_weight_results.csv', index=False)

    # Print the results
    print(f'Delta AIC: {delta_results["AIC"].mean()}; Delta BIC: {delta_results["BIC"].mean()}')
    print(f'Decay AIC: {decay_results["AIC"].mean()}; Decay BIC: {decay_results["BIC"].mean()}')
    print(f'WSLS AIC: {WSLS_results["AIC"].mean()}; WSLS BIC: {WSLS_results["BIC"].mean()}')
    print(f'WSLS Delta AIC: {WSLS_delta_results["AIC"].mean()}; WSLS Delta BIC: {WSLS_delta_results["BIC"].mean()}')
    print(f'WSLS Delta Weight AIC: {WSLS_delta_weight_results["AIC"].mean()}; WSLS Delta Weight BIC: {WSLS_delta_weight_results["BIC"].mean()}')
    print(f'WSLS Decay Weight AIC: {WSLS_decay_weight_results["AIC"].mean()}; WSLS Decay Weight BIC: {WSLS_decay_weight_results["BIC"].mean()}')



