import pandas as pd
import numpy as np
import os


def calculate_rt_stats(data):
    # Change the unit of the RT column to seconds
    data['RT'] = (data['RT'] / 1000).astype(float)
    # Calculate mean and std of RT for each subject
    rt_stats = data.groupby('SubNo')['RT'].agg(['mean', 'std']).reset_index()
    rt_stats.columns = ['SubNo', 'rt_mean', 'rt_std']
    data_merged = data.merge(rt_stats, on='SubNo')
    return data_merged


def preprocess_data(file_path, exclude_list, out_dir):
    """
    Preprocess the data from the given file path.
    This function reads all txt files, skips excluded subjects, and processes the data.
    """

    dataframes = []

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
    print(f'Number of participants: {len(concatenated_df["SubNo"].unique())}')

    # Remove practice trials (Block 0)
    concatenated_df = concatenated_df[concatenated_df['Block'] != 0]

    # Check how many participants reached Block 5 and remove Block 5s
    block_counts = concatenated_df.groupby('SubNo')['Block'].nunique().value_counts().to_dict()
    print(f'Block counts: {block_counts}')
    if 5 in block_counts:
        print(f'Number of participants who reached Block 5: {block_counts[5]}')
        concatenated_df = concatenated_df[concatenated_df['Block'] != 5]
    else:
        print('No participants reached Block 5.')

    # Remove non-responses
    processed_df = concatenated_df[~concatenated_df['Optimal_Choice'].isna()].copy()

    # Change the dtype of optimal choice to int if it is not NaN
    processed_df['Optimal_Choice'] = processed_df['Optimal_Choice'].astype(int)

    # Process the RT column
    processed_df = calculate_rt_stats(processed_df)

    # Remove trials with RT > 3 SD from subject's mean
    processed_df = processed_df[processed_df['RT'] <= (processed_df['rt_mean'] + 3 * processed_df['rt_std'])]

    # Remove trials with RT < 0.3 seconds
    processed_df = processed_df[processed_df['RT'] >= 0.3]

    # Drop the helper columns
    processed_df = processed_df.drop(['rt_mean', 'rt_std'], axis=1)
    print(processed_df['RT'].describe())
    print(f'min reward: {processed_df["OutcomeValue"].min()}')

    # Save the dataFrames
    concatenated_df.to_csv(os.path.join(out_dir, 'raw_data.csv'), index=False)
    processed_df.to_csv(os.path.join(out_dir, 'cleaned_data.csv'), index=False)

    return concatenated_df, processed_df

if __name__ == "__main__":
    # Load txt file
    lesas1_file_path = './LeSaS1/Data/'
    lesas1_exclude_list = [6, 23, 74, 90, 96, 97]
    lesas1_out_dir = './LeSaS1/Data/'

    ledis1_file_path = './LeDiS1/Data/'
    ledis1_out_dir = './LeDiS1/Data/'

    # Preprocess LeSaS1 data
    lesas1_raw, lesas1_cleaned = preprocess_data(lesas1_file_path, lesas1_exclude_list, lesas1_out_dir)

    # Preprocess LeDiS1 data
    ledis1_raw, ledis1_cleaned = preprocess_data(ledis1_file_path, [], ledis1_out_dir)

    # read a txt
    summary = pd.read_csv('./LeDiS1/Data/summaryStats/N59Data_LEDIS1_03-Jun-2025_Summary.txt', sep='\t').dropna()

    # extract the subject number array
    subject_numbers = summary['Sub'].unique()




