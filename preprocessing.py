import pandas as pd
import numpy as np
import os


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

# print the number of trials per subject
n = (concatenated_df['SubNo'].value_counts())
print(n.describe())

concatenated_df.to_csv('./LeSaS1/Data/cleaned_data.csv', index=False)



