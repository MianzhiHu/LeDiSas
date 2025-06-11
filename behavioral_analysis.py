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
from model_fitting import exclusionary_criteria
from preprocessing import calculate_rt_stats

# load the data
lesas1_data_raw = pd.read_csv('./LeSaS1/Data/raw_data.csv')
lesas1_data_clean = pd.read_csv('./LeSaS1/Data/cleaned_data.csv')
ledis1_data_raw = pd.read_csv('./LeDiS1/Data/raw_data.csv')
ledis1_data_clean = pd.read_csv('./LeDiS1/Data/cleaned_data.csv')
lesas1_group_assignment = pd.read_csv('./LeSaS1/Data/group_assignment.csv')
ledis1_group_assignment = pd.read_csv('./LeDiS1/Data/group_assignment.csv')

lesas1_data_raw = calculate_rt_stats(lesas1_data_raw)
ledis1_data_raw = calculate_rt_stats(ledis1_data_raw)

for df in [lesas1_data_raw, lesas1_data_clean, ledis1_data_raw, ledis1_data_clean, lesas1_group_assignment, ledis1_group_assignment]:
    df = df.rename(columns={'Group(1=OptHighReward;2=OptLowReward)': 'Group'}, inplace=True)

# print the number of participants reaching each block per group
def print_block_counts(data, group_col='Group'):
    block_counts = data.groupby(['SubNo', group_col])['Block'].nunique().reset_index()
    block_counts = block_counts.groupby(group_col)['Block'].value_counts().unstack(fill_value=0)
    print(block_counts)

print(f'LeSaS1 Data:')
print_block_counts(lesas1_data_clean)
print(f'LeDiS1 Data:')
print_block_counts(ledis1_data_clean)

# ======================================================================================================================
# LeSaS1 Behavioral Analysis
# ======================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------
# Behavioral Data Analysis
# ----------------------------------------------------------------------------------------------------------------------
# Define the number of blocks
lesas1_data_3blocks = lesas1_data_clean[lesas1_data_clean['Block'].isin([1, 2, 3])]
lesas1_data_4blocks = lesas1_data_clean[lesas1_data_clean['Block'].isin([1, 2, 3, 4])]

# one-sample t-test for optimal choice percentage
for data_block in [lesas1_data_3blocks, lesas1_data_4blocks]:
    print(f'\nAnalyzing data for {len(data_block["SubNo"].unique())} participants in {data_block["Block"].nunique()} blocks:')
    # Calculate % of optimal choices
    optimal_choices = data_block.groupby('SubNo')['Optimal_Choice'].mean()
    optimal_df = optimal_choices.reset_index()
    optimal_df.columns = ['participant_id', 'Optimal_Choice']
    optimal_df = pd.merge(optimal_df, lesas1_group_assignment, on='participant_id')
    t_between, p_between = stats.ttest_ind(
        optimal_df[optimal_df['Group'] == 1]['Optimal_Choice'],
        optimal_df[optimal_df['Group'] == 2]['Optimal_Choice']
    )
    print(f'Between Groups - T-test: t-statistic = {t_between:.3f}, p-value = {p_between:.3f}')
    for group in [1, 2]:
        group_data = optimal_df[optimal_df['Group'] == group]['Optimal_Choice']
        t_stat, p_value = stats.ttest_1samp(group_data, 0.5)
        print(f'Group {group} - T-test: t-statistic = {t_stat:.3f}, p-value = {p_value:.3f}')

# Perform mixed ANOVA for 3 blocks
lesas1_mix_anova_results_3b = pg.mixed_anova(data=lesas1_data_3blocks, dv='Optimal_Choice', between='Group', within='Block', subject='SubNo')
lesas1_mix_anova_results_4b = pg.mixed_anova(data=lesas1_data_4blocks, dv='Optimal_Choice', between='Group', within='Block', subject='SubNo')

# ----------------------------------------------------------------------------------------------------------------------
# Behavioral Windows
# ----------------------------------------------------------------------------------------------------------------------
# Define the window size
window_size = 10

lesas1_optimal_choices = []

for _, participant_data in lesas1_data_raw.groupby('SubNo'):
    for i in range(len(participant_data) - window_size + 1):
        window = participant_data.iloc[i:i + window_size]
        window = exclusionary_criteria(window)
        optimal_percent = np.mean(window['Optimal_Choice'])
        lesas1_optimal_choices.append({
            'participant_id': participant_data['SubNo'].iloc[0],
            'window_id': i + 1,
            'optimal_percentage': optimal_percent,
            'Group': participant_data['Group'].iloc[0]
        })

lesas1_optimal_window_df = pd.DataFrame(lesas1_optimal_choices)
lesas1_optimal_window_df['Group'] = lesas1_optimal_window_df['Group'].map({1: 'High-Reward-Optimal', 2: 'Low-Reward-Optimal'})

# Create a plot for optimal choice percentages
plt.figure(figsize=(10, 6))
sns.lineplot(data=lesas1_optimal_window_df, x='window_id', y='optimal_percentage', hue='Group')
plt.axvline(x=75, color='gray', linestyle='--', alpha=0.5, label='Block 1-2 Transition')
plt.axvline(x=159, color='gray', linestyle='--', alpha=0.5, label='Block 2-3 Transition')
plt.axvline(x=243, color='gray', linestyle='--', alpha=0.5, label='Block 3-4 Transition')
plt.title('Optimal Choice Percentage by Window Steps')
plt.xlabel('Window Step')
plt.ylabel('Optimal Choice Percentage')
plt.savefig('./figures/optimal_choice_moving_window.png', dpi=600, bbox_inches='tight')

# Fit quadratic regression model to optimal choice percentages
model = smf.ols('optimal_percentage ~ window_id + I(window_id**2) + C(Group)', data=lesas1_optimal_window_df).fit()
print(model.summary())

# Plot the fitted quadratic regression model
plt.clf()
g = sns.lmplot(data=lesas1_optimal_window_df, x='window_id', y='optimal_percentage', hue='Group',
               order=2, ci=95, height=6, aspect=1.5, scatter=False, palette='tab10', facet_kws={"legend_out": False})
g.set_axis_labels('Window Step', 'Optimal Choice Percentage')
plt.axvline(x=75, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=159, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=243, color='gray', linestyle='--', alpha=0.5)
plt.title('Optimal Choice Percentage by Window Steps with Quadratic Fit')
plt.tight_layout()
plt.savefig('./figures/optimal_choice_moving_window_quadratic_fit.png', dpi=600, bbox_inches='tight')

# ======================================================================================================================
# LeDiS1 Behavioral Analysis
# ======================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------
# Behavioral Data Analysis
# ----------------------------------------------------------------------------------------------------------------------
# Define the number of blocks
ledis1_data_3blocks = ledis1_data_clean[ledis1_data_clean['Block'].isin([1, 2, 3])]
ledis1_data_4blocks = ledis1_data_clean[ledis1_data_clean['Block'].isin([1, 2, 3, 4])]

# one-sample t-test for optimal choice percentage
for data_block in [ledis1_data_3blocks, ledis1_data_4blocks]:
    print(f'\nAnalyzing data for {len(data_block["SubNo"].unique())} participants in {data_block["Block"].nunique()} blocks:')
    # Calculate % of optimal choices
    subj_means = data_block.groupby(['SubNo', 'Group'])['Optimal_Choice'].mean().reset_index()
    t_between, p_between = stats.ttest_ind(
        subj_means[subj_means['Group'] == 1]['Optimal_Choice'],
        subj_means[subj_means['Group'] == 2]['Optimal_Choice']
    )
    print(f'Average for Group 1: {subj_means[subj_means["Group"] == 1]["Optimal_Choice"].mean():.3f}')
    print(f'Average for Group 2: {subj_means[subj_means["Group"] == 2]["Optimal_Choice"].mean():.3f}')
    print(f'Between Groups - T-test: t-statistic = {t_between:.3f}, p-value = {p_between:.3f}')
    for group in [1, 2]:
        group_data = subj_means[subj_means['Group'] == group]['Optimal_Choice']
        t_stat, p_value = stats.ttest_1samp(group_data, 0.5)
        print(f'Group {group} - T-test: t-statistic = {t_stat:.3f}, p-value = {p_value:.3f}')

# Perform mixed ANOVA for 3 blocks
ledis1_mix_anova_results_3b = pg.mixed_anova(data=ledis1_data_3blocks, dv='Optimal_Choice', between='Group', within='Block', subject='SubNo')
ledis1_mix_anova_results_4b = pg.mixed_anova(data=ledis1_data_4blocks, dv='Optimal_Choice', between='Group', within='Block', subject='SubNo')

# ----------------------------------------------------------------------------------------------------------------------
# Behavioral Windows
# ----------------------------------------------------------------------------------------------------------------------
# Define the window size
window_size = 10
ledis1_optimal_choices = []

for _, participant_data in ledis1_data_raw.groupby('SubNo'):
    for i in range(len(participant_data) - window_size + 1):
        window = participant_data.iloc[i:i + window_size]
        window = exclusionary_criteria(window)
        optimal_percent = np.mean(window['Optimal_Choice'])
        ledis1_optimal_choices.append({
            'participant_id': participant_data['SubNo'].iloc[0],
            'window_id': i + 1,
            'optimal_percentage': optimal_percent,
            'Group': participant_data['Group'].iloc[0]
        })
ledis1_optimal_window_df = pd.DataFrame(ledis1_optimal_choices)
ledis1_optimal_window_df['Group'] = ledis1_optimal_window_df['Group'].map({1: 'High-Variance-Optimal', 2: 'Low-Variance-Optimal'})

# Create a plot for optimal choice percentages
plt.figure(figsize=(10, 6))
sns.lineplot(data=ledis1_optimal_window_df, x='window_id', y='optimal_percentage', hue='Group')
plt.axvline(x=75, color='gray', linestyle='--', alpha=0.5, label='Block 1-2 Transition')
plt.axvline(x=159, color='gray', linestyle='--', alpha=0.5, label='Block 2-3 Transition')
plt.axvline(x=243, color='gray', linestyle='--', alpha=0.5, label='Block 3-4 Transition')
plt.title('Optimal Choice Percentage by Window Steps')
plt.xlabel('Window Step')
plt.ylabel('Optimal Choice Percentage')
plt.savefig('./figures/ledis1_optimal_choice_moving_window.png', dpi=600, bbox_inches='tight')
# ======================================================================================================================
