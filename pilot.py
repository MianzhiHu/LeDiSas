import pandas as pd
import numpy as np
import os
from utils.VisualSearchModels import VisualSearchModels
from utils.ComputationalModeling import dict_generator
from utils.DualProcess import DualProcessModel
from utils.VisualSearchModels import VisualSearchModels

# Load txt file
file_path = './LeSaS1/Data/'
dataframes = []
exclude_list = [6, 23, 74, 90, 96, 97]

# Iterate over each file in the directory
i = 0
for file in os.listdir(file_path):
    if file.endswith('.txt') and file.startswith('Data_'):
        full_path = os.path.join(file_path, file)
        subject_number = int(file.split('_')[2].split('.')[0])
        # Skip excluded subjects
        if subject_number in exclude_list:
            continue
        i += 1
        # Read the txt file into a DataFrame
        df = pd.read_csv(full_path, sep='\t', skiprows=1)
        # Reindex subject number
        df['SubNo'] = i
        dataframes.append(df)

# Concatenate all the DataFrames into one
concatenated_df = pd.concat(dataframes, ignore_index=True)
print(concatenated_df['SubNo'].nunique())

# Remove practice trials (Block 0)
concatenated_df = concatenated_df[concatenated_df['Block'] != 0]

# Remove non-responses
concatenated_df = concatenated_df[~concatenated_df['Optimal_Choice'].isna()]

# Change the dtype of optimal choice to int
concatenated_df['Optimal_Choice'] = concatenated_df['Optimal_Choice'].astype(int)

# Change the unit of the RT column to seconds
concatenated_df['RT'] = concatenated_df['RT'] / 1000
print(concatenated_df['RT'].describe())

# Calculate mean and std of RT for each subject
rt_stats = concatenated_df.groupby('SubNo')['RT'].agg(['mean', 'std']).reset_index()
rt_stats.columns = ['SubNo', 'rt_mean', 'rt_std']

# Merge stats back to main dataframe
concatenated_df = concatenated_df.merge(rt_stats, on='SubNo')

# Remove trials with RT > 3 SD from subject's mean
concatenated_df = concatenated_df[concatenated_df['RT'] <= (concatenated_df['rt_mean'] + 3 * concatenated_df['rt_std'])]

# Drop the helper columns
concatenated_df = concatenated_df.drop(['rt_mean', 'rt_std'], axis=1)
print(concatenated_df['RT'].describe())
print(f'min reward: {concatenated_df["OutcomeValue"].min()}')
# concatenated_df.to_csv('./LeSaS1/Data/cleaned_data.csv', index=False)

# Display the concatenated DataFrame
data_dict = dict_generator(concatenated_df, task='VS')

# get the first subject as testing data
test_data = concatenated_df[concatenated_df['SubNo'] == 1]
test_dict = dict_generator(test_data, task='VS')

if __name__ == "__main__":
    n_iterations = 100

    # delta = VisualSearchModels('delta')
    delta_PVL = VisualSearchModels('delta_PVL_relative')
    # decay = VisualSearchModels('decay')
    decay_PVL = VisualSearchModels('decay_PVL_relative')
    # WSLS = VisualSearchModels('WSLS')
    # WSLS_delta = VisualSearchModels('WSLS_delta')
    # WSLS_delta_weight = VisualSearchModels('WSLS_delta_weight')
    # WSLS_decay_weight = VisualSearchModels('WSLS_decay_weight')
    # dual_process = DualProcessModel(task='IGT_SGT', num_options=2)
    RT_exp_basic = VisualSearchModels('RT_exp_basic')
    RT_delta = VisualSearchModels('RT_delta')
    RT_delta_PVL = VisualSearchModels('RT_delta_PVL')
    RT_decay = VisualSearchModels('RT_decay')
    RT_decay_PVL = VisualSearchModels('RT_decay_PVL')
    RT_exp_delta = VisualSearchModels('RT_exp_delta')
    RT_exp_decay = VisualSearchModels('RT_exp_decay')
    hybrid_delta_delta = VisualSearchModels('hybrid_delta_delta')

    # # Fit the model to the data
    # dual_process_results = dual_process.fit(data_dict, num_iterations=n_iterations, weight_Gau='softmax', weight_Dir='softmax',
    #                                         arbi_option='Entropy', Dir_fun='Linear_Recency', Gau_fun='Naive_Recency', num_t=1)
    # delta_results = delta.fit(data_dict, num_iterations=n_iterations)
    delta_PVL_results = delta_PVL.fit(data_dict, num_iterations=n_iterations)
    delta_PVL_results.to_csv('./LeSaS1/Model/delta_PVL_results.csv', index=False)
    # decay_results = decay.fit(data_dict, num_iterations=n_iterations)
    decay_PVL_results = hybrid_delta_delta.fit(data_dict, num_iterations=n_iterations)
    decay_PVL_results.to_csv('./LeSaS1/Model/decay_PVL_results.csv', index=False)
    # WSLS_results = WSLS.fit(data_dict, num_iterations=n_iterations)
    # WSLS_delta_results = WSLS_delta.fit(data_dict, num_iterations=n_iterations)
    # WSLS_delta_weight_results = WSLS_delta_weight.fit(data_dict, num_iterations=n_iterations)
    # WSLS_decay_weight_results = WSLS_decay_weight.fit(data_dict, num_iterations=n_iterations)
    RT_exp_basic_results = RT_exp_basic.fit(data_dict, num_iterations=n_iterations)
    RT_exp_basic_results.to_csv('./LeSaS1/Model/RT_exp_basic_results.csv', index=False)
    RT_delta_results = RT_delta.fit(data_dict, num_iterations=n_iterations)
    RT_delta_results.to_csv('./LeSaS1/Model/RT_delta_results.csv', index=False)
    RT_decay_results = RT_decay.fit(data_dict, num_iterations=n_iterations)
    RT_decay_results.to_csv('./LeSaS1/Model/RT_decay_results.csv', index=False)
    RT_exp_delta_results = RT_exp_delta.fit(data_dict, num_iterations=n_iterations)
    RT_exp_delta_results.to_csv('./LeSaS1/Model/RT_exp_delta_results.csv', index=False)
    RT_exp_decay_results = RT_exp_decay.fit(data_dict, num_iterations=n_iterations)
    RT_exp_decay_results.to_csv('./LeSaS1/Model/RT_exp_decay_results.csv', index=False)
    RT_delta_PVL_results = RT_delta_PVL.fit(data_dict, num_iterations=n_iterations)
    RT_delta_PVL_results.to_csv('./LeSaS1/Model/RT_delta_PVL_results.csv', index=False)
    RT_decay_PVL_results = RT_decay_PVL.fit(data_dict, num_iterations=n_iterations)
    RT_decay_PVL_results.to_csv('./LeSaS1/Model/RT_decay_PVL_results.csv', index=False)
    hybrid_delta_delta_results = hybrid_delta_delta.fit(data_dict, num_iterations=n_iterations)
    hybrid_delta_delta_results.to_csv('./LeSaS1/Model/hybrid_delta_delta_results.csv', index=False)


    # dual_process_results.to_csv('./LeSaS1/Model/dual_process_results.csv', index=False)
    # delta_results.to_csv('./LeSaS1/Model/delta_results.csv', index=False)
    # decay_results.to_csv('./LeSaS1/Model/decay_results.csv', index=False)
    # delta_PVL_results.to_csv('./LeSaS1/Model/delta_PVL_results.csv', index=False)
    # WSLS_results.to_csv('./LeSaS1/Model/WSLS_results.csv', index=False)
    # WSLS_delta_results.to_csv('./LeSaS1/Model/WSLS_delta_results.csv', index=False)
    # WSLS_delta_weight_results.to_csv('./LeSaS1/Model/WSLS_delta_weight_results.csv', index=False)
    # WSLS_decay_weight_results.to_csv('./LeSaS1/Model/WSLS_decay_weight_results.csv', index=False)\
    # RT_exp_basic_results.to_csv('./LeSaS1/Model/RT_exp_basic_results.csv', index=False)
    # RT_delta_results.to_csv('./LeSaS1/Model/RT_delta_results.csv', index=False)
    # RT_decay_results.to_csv('./LeSaS1/Model/RT_decay_results.csv', index=False)
    # RT_exp_delta_results.to_csv('./LeSaS1/Model/RT_exp_delta_results.csv', index=False)
    # RT_exp_decay_results.to_csv('./LeSaS1/Model/RT_exp_decay_results.csv', index=False)



