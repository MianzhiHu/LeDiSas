import os
from collections import defaultdict
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from plotting_functions import *
from utils.ComputationalModeling import parameter_extractor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pingouin as pg
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load the cleaned data
data_path = './LeSaS1/Data/cleaned_data.csv'
data = pd.read_csv(data_path)
data.rename(columns={'Group(1=OptHighReward;2=OptLowReward)': 'Group'}, inplace=True)
group_assignment = pd.read_csv('./LeSaS1/Data/group_assignment.csv')

# Calculate % of optimal choices
optimal_choices = data.groupby('SubNo')['Optimal_Choice'].mean()
optimal_df = optimal_choices.reset_index()
optimal_df.columns = ['participant_id', 'Optimal_Choice']
optimal_df = pd.merge(optimal_df, group_assignment, on='participant_id')

# Behavioral windows
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
concatenated_df = concatenated_df[concatenated_df['Block'].isin([1, 2, 3, 4])]

# Calculate optimal choice percentage in moving windows
window_size = 10

# ----------------------------------------------------------------------------------------------------------------------
# All data version
# ----------------------------------------------------------------------------------------------------------------------
optimal_choices = []

for _, participant_data in concatenated_df.groupby('SubNo'):
    for i in range(len(participant_data) - window_size + 1):
        window = participant_data.iloc[i:i + window_size]
        optimal_percent = np.mean(window['Optimal_Choice'])
        optimal_choices.append({
            'participant_id': participant_data['SubNo'].iloc[0],
            'window_id': i + 1,
            'optimal_percentage': optimal_percent,
            'Group': participant_data['Group(1=OptHighReward;2=OptLowReward)'].iloc[0]
        })

optimal_window_df = pd.DataFrame(optimal_choices)
print(max(optimal_window_df['window_id']))

model = smf.mixedlm("optimal_percentage ~ Group * window_id", optimal_window_df, groups=optimal_window_df["participant_id"]).fit()
print(model.summary())

# ----------------------------------------------------------------------------------------------------------------------
# All data block-wise version
# ----------------------------------------------------------------------------------------------------------------------
optimal_choices = []
for _, participant_data in concatenated_df.groupby('SubNo'):
    window_counter = 0
    # Within each participant, split off each block
    for block_num, block_data in participant_data.groupby('Block'):
        print(f'Block {block_num} for participant {participant_data["SubNo"].iloc[0]}')
        # Now run a length‑10 sliding window *only* on this block
        for i in range(len(block_data) - window_size + 1):
            window = block_data.iloc[i : i + window_size]
            window_counter += 1
            optimal_percent = window['Optimal_Choice'].mean()

            optimal_choices.append({
                'participant_id': participant_data['SubNo'].iloc[0],
                'Block': block_num,
                'window_id': window_counter,
                'optimal_percentage': optimal_percent,
                'Group': participant_data['Group(1=OptHighReward;2=OptLowReward)'].iloc[0]
            })

optimal_window_df = pd.DataFrame(optimal_choices)

# ----------------------------------------------------------------------------------------------------------------------
# Cleaned data block-wise version
# ----------------------------------------------------------------------------------------------------------------------
optimal_choices = []
for _, participant_data in data.groupby('SubNo'):
    # Within each participant, split off each block
    for block_num, block_data in participant_data.groupby('Block'):
        if block_num not in [1, 2, 3, 4]:
            continue
        window_counter = 0
        print(f'Block {block_num} for participant {participant_data["SubNo"].iloc[0]}')
        # Now run a length‑10 sliding window *only* on this block
        for i in range(len(block_data) - window_size + 1):
            window = block_data.iloc[i : i + window_size]
            window_counter += 1
            window_id = window_counter + (block_num - 1) * 75  # Adjust window_id to be continuous across blocks
            optimal_percent = window['Optimal_Choice'].mean()
            optimal_choices.append({
                'participant_id': participant_data['SubNo'].iloc[0],
                'Block': block_num,
                'window_id': window_id,
                'optimal_percentage': optimal_percent,
                'Group': participant_data['Group'].iloc[0]
            })

optimal_window_df = pd.DataFrame(optimal_choices)


# Create plot for optimal choice percentages
break_positions = set()

# Loop over each participant’s windows in ascending order
for pid, df in optimal_window_df.groupby("participant_id"):
    df = df.sort_values("window_id", ignore_index=True)

    # Identify where the block changes
    block_shift = df["Block"].shift(1)
    change_mask = (df["Block"] != block_shift) & (block_shift.notna())
    new_block_windows = df.loc[change_mask, "window_id"].values
    break_positions.update(new_block_windows)

break_positions = sorted(break_positions)

plt.figure(figsize=(10, 6))
sns.lineplot(data=optimal_window_df, x='window_id', y='optimal_percentage', hue='Group')
plt.title('Optimal Choice Percentage by Window Step Grouped by Participants')
plt.xlabel('Window Step')
plt.ylabel('Optimal Choice Percentage')
# Add vertical lines to indicate block changes
for pos in break_positions:
    plt.axvline(x=pos, color='gray', linestyle='--', linewidth=0.5)
plt.savefig('./figures/optimal_choice_moving_window_blockwise.png', dpi=600, bbox_inches='tight')


# ======================================================================================================================
# Model Analysis
# ======================================================================================================================
# Load model results
model_path = './LeSaS1/Model/BlockWise/'
model_results = {}
grouped = defaultdict(list)

# Iterate through files in the model directory
for file in os.listdir(model_path):
    if file.endswith('.csv'):
        # Get model name from filename
        model_name = file.split('.')[0].replace('_results', '')
        block = int(model_name.split('_')[-1])
        base_name = '_'.join(model_name.split('_')[:-2])
        df = pd.read_csv(model_path + file)
        df['Block'] = block
        if block in [1, 2, 3, 4]:
            # Merge with group assignment
            df_grouped = df.merge(group_assignment, on='participant_id', how='left')
            grouped[base_name].append(df_grouped)

            print(f'Loaded {model_name} (block={block}) → shape = {df_grouped.shape}')

# Now concatenate all blocks for each base model
for base, dfs in grouped.items():
    # combine along rows, reset index if you like
    combined = pd.concat(dfs, ignore_index=True)
    model_results[base] = combined
    print(f'[{base}] combined shape = {combined.shape}')
    print(f'  • BIC mean:')
    print(f'    Group 1 = {combined[combined["Group"] == 1]["BIC"].mean():.2f}')
    print(f'    Group 2 = {combined[combined["Group"] == 2]["BIC"].mean():.2f}')

print("Available models:")
print(model_results.keys())

# Print block-wise model results
for group in [1, 2]:
    print(f'\nGroup {group} Model Results:')
    for block in [1, 2, 3, 4]:
        print(f'\nBlock {block}:')
        for model_name, df in model_results.items():
            if model_name in ['delta', 'RT_delta', 'hybrid_delta_delta', 'hybrid_delta_delta_3']:
                mean_bic = df[df['Block'] == block]['BIC'].mean()
                mean_aic = df[df['Block'] == block]['AIC'].mean()
                print(f'{model_name}: BIC = {mean_bic:.2f}, AIC = {mean_aic:.2f}')

# ======================================================================================================================
# Moving Window Analysis
# ======================================================================================================================
# define parameter map
parameter_map = {
    'delta': ['t', 'alpha'],
    'delta_RPUT': ['t', 'alpha'],
    'RT_delta': ['t', 'alpha', 'RT0Sub', 'RT0Opt'],
    'RT_exp_basic': ['t', 'alpha', 'RT0Sub', 'RT0Opt'],
    'RT_exp_delta': ['t', 'alpha', 'RT0Sub', 'RT0Opt', 'k'],
    'hybrid_delta_delta': ['t', 'alpha', 'RT0Sub', 'RT0Opt', 'Weight'],
    'hybrid_delta_delta_3': ['t', 'alpha', 'beta', 'RT0Sub', 'RT0Opt', 'Weight']
}

model_mv_path = './LeSaS1/Model/BlockWise_Moving_Window/'
blockwise_mv_results = {}
grouped_mv = defaultdict(list)

# Iterate through files in the model directory
for file in os.listdir(model_mv_path):
    if file.endswith('.csv'):
        # Get model name from filename
        file_name = file.split('.')[0].replace('_results', '')
        block = int(file_name.split('_')[-2])
        model_name = file_name.split('_block')[0]
        base_name = '_'.join(file_name.split('_')[:-3])
        df = pd.read_csv(model_mv_path + file)
        df['Block'] = block
        if block in [1, 2, 3, 4]:
            # Merge with group assignment
            df_grouped = df.merge(group_assignment, on='participant_id', how='left')
            df_grouped = parameter_extractor(df_grouped, parameter_map[model_name])
            grouped_mv[base_name].append(df_grouped)

            print(f'Loaded {model_name} (block={block}) → shape = {df_grouped.shape}')

# Now concatenate all blocks for each base model
for base, dfs in grouped_mv.items():
    # combine along rows, reset index if you like
    combined_mv = pd.concat(dfs, ignore_index=True)
    blockwise_mv_results[base] = combined_mv
    blockwise_mv_results[base]['continuous_window'] = (blockwise_mv_results[base]['window_id'] +
                                                       (blockwise_mv_results[base]['Block'] - 1) * 75)

# get a complete df
model_mv_results_df = pd.DataFrame()
for model_name in blockwise_mv_results.keys():
    model_df = blockwise_mv_results[model_name]
    model_df['Model'] = model_name
    model_mv_results_df = pd.concat([model_mv_results_df, model_df], ignore_index=True)

# Create plot for optimal choice percentages
model_of_interest = blockwise_mv_results['hybrid_delta_delta']

# Mixed effects model
model = smf.mixedlm("Weight ~ Group * continuous_window", model_of_interest, groups=model_of_interest["participant_id"]).fit()
print(model.summary())

# quadratic model
model_quadratic = smf.mixedlm("Weight ~ Group * continuous_window + I(continuous_window ** 2)", model_of_interest,
                                groups=model_of_interest["participant_id"]).fit()
print(model_quadratic.summary())

# plot the quadratic model
g = sns.lmplot(
    data=model_of_interest,
    x="continuous_window",
    y="Weight",
    hue="Group",         # color‐code each Group
    col="Group",         # if you want separate panels per Group (side‐by‐side)
    palette="tab10",     # or choose any palette you like
    scatter=False,
    line_kws={"linewidth": 2},            # thicker fit line
    order=2,             # fit a 2nd‐degree polynomial
    ci=95,             # omit the confidence‐interval shading
    height=4,            # height (inches) of each facet
    aspect=1.2           # width = aspect * height
)

# 2) Adjust axis labels and title
g.set_axis_labels("Window (continuous index)", "Weight")
g.set_titles(col_template="{col_name}")
g.fig.suptitle("Quadratic Fit of Weight vs. Window by Group", y=1.02)

plt.tight_layout()
plt.savefig('./figures/weight_moving_window_blockwise.png', dpi=600, bbox_inches='tight')
plt.show()

# Identify break positions for block changes

break_positions = set()

# Loop over each participant’s windows in ascending order
for pid, df in model_of_interest.groupby("participant_id"):
    df = df.sort_values("continuous_window", ignore_index=True)

    # Identify where the block changes
    block_shift = df["Block"].shift(1)
    change_mask = (df["Block"] != block_shift) & (block_shift.notna())
    new_block_windows = df.loc[change_mask, "continuous_window"].values
    break_positions.update(new_block_windows)

break_positions = sorted(break_positions)

# Create the plot
side_by_side_plot(model_of_interest, 'continuous_window', 'Weight', 'Window Step', 'Weight Value',
                  'weight_moving_window_blockwise', block_div=True)

# Plot BIC values for each model in moving window analysis
models_chosen = ['delta', 'RT_delta']
model_mv_chosen = model_mv_results_df[model_mv_results_df['Model'].isin(models_chosen)]

for group in [1, 2]:
    plt.figure(figsize=(10, 6))
    group_data = model_mv_chosen[model_mv_chosen['Group'] == group]
    sns.lineplot(data=group_data, x='continuous_window', y='BIC', hue='Model')
    plt.title(f'BIC Values by Window Step for Group {group}')
    plt.xlabel('Window Step')
    plt.ylabel('BIC Value')
    plt.legend(title='Model')
    plt.savefig(f'./figures/BIC_moving_window_group_{group}_blockwise.png', dpi=600, bbox_inches='tight')
    plt.close()

# Perform regression analysis for each group
optimal_window_df.rename(columns={'window_id': 'continuous_window'}, inplace=True)
model_of_interest = pd.merge(model_of_interest, optimal_window_df, on=['participant_id', 'Group', 'Block',
                                                                       'continuous_window'], how='left')
regression_results = []
for window_id in range(1, max(model_of_interest['continuous_window']) + 1):
    window_data = model_of_interest[model_of_interest['continuous_window'] == window_id]
    print(f'Processing window {window_id} with {len(window_data)} participants.')

    if len(window_data) < 10:
        print(f'Window {window_id} has less than 10 participants. Skipping.')
        continue  # Skip windows with less than 10 participants. There should not be any, but just in case.

    for group in [1, 2]:
        group_data = window_data[window_data['Group'] == group]
        if len(group_data) < 10:
            print(f'Group {group} in window {window_id} has less than 10 participants. Skipping.')
            continue  # Skip groups with less than 10 participants. There should not be any, but just in case.

        # Perform linear regression
        lr_results = pg.linear_regression(group_data[['t', 'alpha', 'RT0Sub', 'RT0Opt', 'Weight']],
                                          group_data['optimal_percentage'], add_intercept=True)
        slope = lr_results['coef'][5]
        intercept = lr_results['coef'][0]
        r_squared = lr_results['r2'][1]
        p_value = lr_results['pval'][1]
        upper_ci = lr_results['CI[2.5%]'][1]
        lower_ci = lr_results['CI[97.5%]'][1]
        regression_results.append({
            'window_id': window_id,
            'Group': group,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'p_value': p_value,
            'upper_ci': upper_ci,
            'lower_ci': lower_ci
        })
regression_results_df = pd.DataFrame(regression_results)
regression_results_df.to_csv('./LeSaS1/Data/regression_results_blockwise.csv', index=False)

# Plot regression results
for group in [1, 2]:
    group_data = regression_results_df[regression_results_df['Group'] == group]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=group_data, x='window_id', y='slope', label='Slope')
    sns.lineplot(data=group_data, x='window_id', y='p_value', label='p value')
    plt.axhline(y=0.05, color='r', linestyle='--')
    plt.fill_between(group_data['window_id'], group_data['lower_ci'], group_data['upper_ci'], alpha=0.2)
    plt.title(f'Regression Results for Group {group}')
    plt.xlabel('Window Step')
    plt.ylabel('Regression Coefficient')
    plt.legend()
    plt.savefig(f'./figures/regression_results_group_{group}_blockwise.png', dpi=600, bbox_inches='tight')