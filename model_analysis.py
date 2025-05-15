import os
import pandas as pd
import numpy as np

# load all model files
model_path = './LeSaS1/Model/'
# Dictionary to store model results
model_results = {}

# Iterate through files in the model directory
for file in os.listdir(model_path):
    if file.endswith('.csv'):
        # Get model name from filename
        model_name = file.split('.')[0].replace('_results', '')
        model_results[model_name] = pd.read_csv(model_path + file)
        # Load CSV file and store in a dictionary
        file_path = os.path.join(model_path, file)
        model_results[model_name] = pd.read_csv(file_path)
        print(f'[{model_name}]')
        print(f'BIC: {model_results[model_name]["BIC"].mean():.2f}; AIC: {model_results[model_name]["AIC"].mean():.2f}')

print(f'Best Model: {min(model_results, key=lambda x: model_results[x]["BIC"].mean())}; '
      f'BIC: {min([model_results[x]["BIC"].mean() for x in model_results]):.2f}; '
      f'AIC: {min([model_results[x]["AIC"].mean() for x in model_results]):.2f}')

# Read in the cleaned data
data_path = './LeSaS1/Data/cleaned_data.csv'
data = pd.read_csv(data_path)

# Get the subset data with only subno and condition
print(data.columns)
data_subset = data[['SubNo', 'Group(1=OptHighReward;2=OptLowReward)']].drop_duplicates()
data_subset.columns = ['participant_id', 'Group']

# Print the average AIC and BIC for each model for each group
# Initialize dictionaries to store best models and their BIC scores for each group
best_models = {1: {'model': None, 'BIC': float('inf')},
               2: {'model': None, 'BIC': float('inf')}}

for model_name, results in model_results.items():
    # Merge the model results with the data subset
    merged_data = pd.merge(data_subset, results, on='participant_id', how='left')

    # Calculate the average AIC and BIC for each group
    avg_aic = merged_data.groupby('Group')['AIC'].mean().to_dict()
    avg_bic = merged_data.groupby('Group')['BIC'].mean().to_dict()

    # Update best model for each group if current model has lower BIC
    for group in [1, 2]:
        if avg_bic[group] < best_models[group]['BIC']:
            best_models[group]['model'] = model_name
            best_models[group]['BIC'] = avg_bic[group]

    print(f'[{model_name}]')
    print(f'Average AIC by Group:\n{avg_aic}')
    print(f'Average BIC by Group:\n{avg_bic}')

print(f'Best Model for Group 1: {best_models[1]["model"]}; BIC: {best_models[1]["BIC"]:.2f}')
print(f'Best Model for Group 2: {best_models[2]["model"]}; BIC: {best_models[2]["BIC"]:.2f}')
# Print the average AIC and BIC for each model for each group
