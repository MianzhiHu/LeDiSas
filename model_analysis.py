import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.ComputationalModeling import parameter_extractor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pingouin as pg
from scipy import stats
from plotting_functions import *

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
data_subset.to_csv('./LeSaS1/Data/group_assignment.csv', index=False)

# Calculate % of optimal choices
optimal_choices = data.groupby('SubNo')['Optimal_Choice'].mean()
optimal_df = optimal_choices.reset_index()
optimal_df.columns = ['participant_id', 'Optimal_Choice']
optimal_df = pd.merge(optimal_df, data_subset, on='participant_id')

# Print the average AIC and BIC for each model for each group
# Initialize dictionaries to store best models and their BIC scores for each group
best_models = {1: {'model': None, 'BIC': float('inf')},
               2: {'model': None, 'BIC': float('inf')}}

for model_name in model_results:
    # Merge the model results with the data subset
    model_results[model_name] = pd.merge(optimal_df, model_results[model_name], on='participant_id', how='left')
    merged_data = model_results[model_name]

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

# ======================================================================================================================
# Data Analysis
# ======================================================================================================================
# Correlation between optimal choice and parameters
hybrid_delta_delta = model_results['hybrid_delta_delta']
hybrid_delta_delta = parameter_extractor(hybrid_delta_delta, ['t', 'alpha', 'RT0Sub', 'RT0Opt', 'Weight'])
hybrid_delta_delta['RT0Diff'] = hybrid_delta_delta['RT0Opt'] - hybrid_delta_delta['RT0Sub']
hybrid_delta_delta.to_csv('./LeSaS1/Data/hybrid_delta_delta.csv', index=False)
corr = pg.pairwise_corr(hybrid_delta_delta, columns=['Optimal_Choice', 't', 'alpha', 'RT0Sub', 'RT0Opt', 'Weight', 'RT0Diff'], method='pearson')

hybrid_delta_delta_1 = hybrid_delta_delta[hybrid_delta_delta['Group'] == 1]
hybrid_delta_delta_2 = hybrid_delta_delta[hybrid_delta_delta['Group'] == 2]
corr_1 = pg.pairwise_corr(hybrid_delta_delta_1, columns=['Optimal_Choice', 't', 'alpha', 'RT0Sub', 'RT0Opt', 'Weight', 'RT0Diff'], method='pearson')
corr_2 = pg.pairwise_corr(hybrid_delta_delta_2, columns=['Optimal_Choice', 't', 'alpha', 'RT0Sub', 'RT0Opt', 'Weight', 'RT0Diff'], method='pearson')

# t-test
t_test = pg.ttest(hybrid_delta_delta_1['Optimal_Choice'], hybrid_delta_delta_2['Optimal_Choice'])
t_test1 = pg.ttest(hybrid_delta_delta_1['Optimal_Choice'], 0.5)
t_test2 = pg.ttest(hybrid_delta_delta_2['Optimal_Choice'], 0.5)

# ======================================================================================================================
# Moving Window Analysis
# ======================================================================================================================
# define parameter map
parameter_map = {
    'delta_mv': ['t', 'alpha'],
    'RT_delta_mv': ['t', 'alpha', 'RT0Sub', 'RT0Opt'],
    'hybrid_delta_delta_mv': ['t', 'alpha', 'RT0Sub', 'RT0Opt', 'Weight'],
    'hybrid_delta_delta_3_mv': ['t', 'alpha', 'beta', 'RT0Sub', 'RT0Opt', 'Weight']
}

model_mv_path = './LeSaS1/Model/Moving_Window/'
model_mv_results = {}
for file in os.listdir(model_mv_path):
    if file.endswith('.csv'):
        # Get model name from filename
        model_name = file.split('.')[0].replace('_results', '')
        model_mv_results[model_name] = pd.read_csv(model_mv_path + file)
        # Load CSV file and store in a dictionary
        file_path = os.path.join(model_mv_path, file)
        model_mv_results[model_name] = pd.read_csv(file_path)
        # merge with the group data
        model_mv_results[model_name] = pd.merge(model_mv_results[model_name], optimal_df, on='participant_id', how='left')
        # use the parameter map to extract the parameters
        model_mv_results[model_name] = parameter_extractor(model_mv_results[model_name], parameter_map[model_name])
        # print average AIC and BIC for each group
        print(f'[{model_name}]')
        print(f'Average AIC by Group:\n{model_mv_results[model_name].groupby("Group")["AIC"].mean()}')
        print(f'Average BIC by Group:\n{model_mv_results[model_name].groupby("Group")["BIC"].mean()}')


# get a complete df
model_mv_results_df = pd.DataFrame()
for model_name in model_mv_results:
    model_mv_results[model_name]['Model'] = model_name.replace('_mv', '')
    model_mv_results_df = pd.concat([model_mv_results_df, model_mv_results[model_name]], axis=0)
print(model_mv_results_df['Model'].value_counts())

# find the window where there are at least 10 participants
window_counts = model_mv_results['delta_mv'].groupby(['Group', 'window_id'])['participant_id'].nunique()
window_counts = window_counts.reset_index()
window_counts = window_counts[window_counts['participant_id'] >= 10]
window_ids = window_counts.groupby('Group')['window_id'].max() # 317 for Group 1 and 318 for Group 2
print(f'We are using the first {min(window_ids)} windows as there are at least 10 participants in each group.')

# Visualize hybrid delta delta
hybrid = model_mv_results['hybrid_delta_delta_mv']
hybrid = hybrid[hybrid['window_id'] <= 317]  # Limit to first 334 trials

# Create the plot
side_by_side_plot(hybrid, 'window_id', 'Weight', 'Window Step', 'Weight Value',
                  'Weight Values by Window Step Grouped by Participants', block_div=False)

# Visualize model fit by time
# Filter and create plots for BIC values across window steps for each group
model_mv_results_df = model_mv_results_df[model_mv_results_df['window_id'] <= 317]

for group in [1, 2]:
    plt.figure(figsize=(10, 6))
    group_data = model_mv_results_df[model_mv_results_df['Group'] == group]
    sns.lineplot(data=group_data, x='window_id', y='BIC', hue='Model')
    plt.title(f'BIC Values by Window Step for Group {group}')
    plt.xlabel('Window Step')
    plt.ylabel('BIC Value')
    plt.legend(title='Model')
    plt.savefig(f'./figures/BIC_moving_window_group_{group}.png', dpi=600, bbox_inches='tight')
    plt.close()

# ----------------------------------------------------------------------------------------------------------------------
# Behavioral windows
# ----------------------------------------------------------------------------------------------------------------------
# Calculate optimal choice percentage in moving windows
window_size = 10
optimal_choices = []

for _, participant_data in data.groupby('SubNo'):
    participant_data = participant_data.iloc[:326]  # Limit to first 317 windows
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

# merge the data
hybrid = pd.merge(hybrid, optimal_window_df, on=['participant_id', 'Group', 'window_id'], how='left')
hybrid.to_csv('./LeSaS1/Data/hybrid_delta_delta_mv.csv', index=False)

# Create plot for optimal choice percentages
plt.figure(figsize=(10, 6))
sns.lineplot(data=optimal_window_df, x='window_id', y='optimal_percentage', hue='Group')
plt.title('Optimal Choice Percentage by Window Step Grouped by Participants')
plt.xlabel('Window Step')
plt.ylabel('Optimal Choice Percentage')
plt.savefig('./figures/optimal_choice_moving_window.png', dpi=600, bbox_inches='tight')

# Perform regression analysis for each group
regression_results = []
for window_id in range(1, 318):
    window_data = hybrid[hybrid['window_id'] == window_id]
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
        lr_results = pg.linear_regression(group_data['Weight'], group_data['optimal_percentage'])
        slope = lr_results['coef'][1]
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
regression_results_df.to_csv('./LeSaS1/Data/regression_results.csv', index=False)

# Plot regression results
for group in [1, 2]:
    group_data = regression_results_df[regression_results_df['Group'] == group]
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=group_data, x='window_id', y='slope', label='Slope')
    sns.lineplot(data=group_data, x='window_id', y='p_value', label='Intercept')
    plt.axhline(y=0.05, color='r', linestyle='--')
    plt.fill_between(group_data['window_id'], group_data['lower_ci'], group_data['upper_ci'], alpha=0.2)
    plt.title(f'Regression Results for Group {group}')
    plt.xlabel('Window Step')
    plt.ylabel('Regression Coefficient')
    plt.legend()
    plt.savefig(f'./figures/regression_results_group_{group}.png', dpi=600, bbox_inches='tight')