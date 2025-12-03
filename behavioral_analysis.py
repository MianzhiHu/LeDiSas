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
from utils.VisualSearchModels import behavioral_moving_window
import functools

# load the data
lesas1_data_raw = pd.read_csv('./LeSaS1/Data/raw_data.csv')
lesas1_data_clean = pd.read_csv('./LeSaS1/Data/cleaned_data.csv')
ledisas_data_raw = pd.read_csv('./LeDiSaS/Data/raw_data.csv')
ledisas_data_clean = pd.read_csv('./LeDiSaS/Data/cleaned_data.csv')
lesas1_group_assignment = pd.read_csv('./LeSaS1/Data/group_assignment.csv')
ledisas_group_assignment = pd.read_csv('./LeDiSaS/Data/group_assignment.csv')

# load participant indices
value_based_index = pd.read_csv('./LeSaS1/Data/value_based_participants.csv')
RT_based_index = pd.read_csv('./LeSaS1/Data/RT_based_participants.csv')
RPUT_based_index = pd.read_csv('./LeSaS1/Data/RPUT_based_participants.csv')

lesas1_data_raw = calculate_rt_stats(lesas1_data_raw)
ledisas_data_raw = calculate_rt_stats(ledisas_data_raw)

for df in [lesas1_data_raw, lesas1_data_clean, lesas1_group_assignment]:
    df = df.rename(columns={'Group(1=OptHighReward;2=OptLowReward)': 'Group'}, inplace=True)

for df in [ledisas_data_raw, ledisas_data_clean]:
    df = df.rename(columns={'Group(1=OHRHV,SLRLV; 2=OHRLV,SLRHV; 3=OLRHV,SHRLV, 4 = OLRLV,SHRHV)': 'Group'}, inplace=True)

# Get participants
value_based_participants = lesas1_data_raw[lesas1_data_raw['SubNo'].isin(value_based_index['participant_id'])]
RT_based_participants = lesas1_data_raw[lesas1_data_raw['SubNo'].isin(RT_based_index['participant_id'])]
RPUT_based_participants = lesas1_data_raw[lesas1_data_raw['SubNo'].isin(RPUT_based_index['participant_id'])]
# print the number of participants grouped by group
print("\nValue-based participants by group:")
print(value_based_participants.groupby('Group')['SubNo'].nunique())
print("\nRT-based participants by group:")
print(RT_based_participants.groupby('Group')['SubNo'].nunique())
print("\nRPUT-based participants by group:")
print(RPUT_based_participants.groupby('Group')['SubNo'].nunique())

# Incorporate the best-fitted model into data
model_assignments = pd.DataFrame({
    'SubNo': list(value_based_index['participant_id']) +
             list(RT_based_index['participant_id']) +
             list(RPUT_based_index['participant_id']),
    'model_type': ['value_based'] * len(value_based_index) +
                  ['RT_based'] * len(RT_based_index) +
                  ['RPUT_based'] * len(RPUT_based_index)
})

# Calculate total points per participant
points_by_participant = lesas1_data_clean.groupby(['SubNo', 'Original_SubNo'])['OutcomeValue'].mean().reset_index()
cum_points_by_participant = lesas1_data_clean.groupby('SubNo')['OutcomeValue'].sum().reset_index()
cum_points_by_participant = cum_points_by_participant.rename(columns={'OutcomeValue': 'Cumulative_OutcomeValue'})
optimal_choice_by_participant = lesas1_data_clean.groupby('SubNo')['Optimal_Choice'].mean().reset_index()
rt_by_participant = lesas1_data_clean.groupby('SubNo')['RT'].mean().reset_index()
# Merge all metrics into a summary DataFrame
summary = functools.reduce(lambda left, right: pd.merge(left, right, on='SubNo'),
                           [points_by_participant, cum_points_by_participant, optimal_choice_by_participant,
                            rt_by_participant, model_assignments])
summary = pd.merge(summary, lesas1_group_assignment[['participant_id', 'Group']], left_on='SubNo',
                   right_on='participant_id')
# Drop redundant participant_id column
summary = summary.drop(columns=['participant_id'])
summary.to_csv('./LeSaS1/Data/summary.csv', index=False)
lesas1_data_clean = pd.merge(lesas1_data_clean, model_assignments, on='SubNo', how='left')
lesas1_data_clean.to_csv('./LeSaS1/Data/data_clean_model.csv', index=False)

# print the number of participants reaching each block per group
def print_block_counts(data, group_col='Group'):
    block_counts = data.groupby(['SubNo', group_col])['Block'].nunique().reset_index()
    block_counts = block_counts.groupby(group_col)['Block'].value_counts().unstack(fill_value=0)
    print(block_counts)

print(f'LeSaS1 Data:')
print_block_counts(lesas1_data_clean)
print_block_counts(ledisas_data_clean)

print(ledisas_data_clean.groupby(['Group', 'OutcomeCond(0=inaccurate;1=lowReward;2=highReward)'])['OutcomeValue'].mean())
print(ledisas_data_clean.groupby(['Group', 'OutcomeCond(0=inaccurate;1=lowReward;2=highReward)'])['OutcomeValue'].std())
print(ledisas_data_clean.groupby(['Group', 'OutcomeCond(0=inaccurate;1=lowReward;2=highReward)'])['OutcomeValue'].count())
print(ledisas_data_clean.groupby(['Group', 'OutcomeCond(0=inaccurate;1=lowReward;2=highReward)'])['Optimal_Choice'].count())
print(lesas1_data_clean.groupby(['Group', 'OutcomeCond(0=inaccurate;1=lowReward;2=highReward)'])['Optimal_Choice'].count())
print(ledisas_data_clean.groupby(['Group'])['Optimal_Choice'].mean())
print(lesas1_data_clean.groupby(['Group'])['Optimal_Choice'].mean())

# T test for outcome value between groups
grouped = ledisas_data_clean.groupby(['SubNo', 'Group', 'OutcomeCond(0=inaccurate;1=lowReward;2=highReward)'])['OutcomeValue'].mean().reset_index()
an = pg.mixed_anova(grouped, dv='OutcomeValue', between='Group', within='OutcomeCond(0=inaccurate;1=lowReward;2=highReward)', subject='SubNo')
print(an)

# T test for RT between larger and smaller subsets
an_rt = pg.mixed_anova(lesas1_data_clean, dv='RT', within='Optimal_Choice', between='Group', subject='SubNo')
print(lesas1_data_clean.groupby(['Optimal_Choice'])['RT'].mean())

# # plot outcome value by group and outcome condition
plt.figure(figsize=(10, 6))
sns.boxplot(data=grouped, x='OutcomeCond(0=inaccurate;1=lowReward;2=highReward)', y='OutcomeValue', hue='Group')
plt.title('Outcome Value by Group and Outcome Condition')
plt.xlabel('Outcome Condition (1=low variance;2=high variance)')
plt.ylabel('Outcome Value')
plt.savefig('./figures/outcome_value_by_group_and_condition.png', dpi=600, bbox_inches='tight')
# ======================================================================================================================
# Behavioral Windows
# =====================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------
# LeSaS1 Behavioral Windows
# ----------------------------------------------------------------------------------------------------------------------
# Define the window size
window_size = 10
lesas1_optimal_window_df = behavioral_moving_window(lesas1_data_raw, variable='Optimal_Choice', window_size=window_size, exclusionary_criteria=exclusionary_criteria)
lesas1_rt_window_df = behavioral_moving_window(lesas1_data_raw, variable='RT', window_size=window_size, exclusionary_criteria=exclusionary_criteria)

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

# # Create a plot for optimal choice percentages
# plt.figure(figsize=(10, 6))
# sns.lineplot(data=lesas1_rt_window_df, x='window_id', y='optimal_percentage', hue='Group')
# plt.axvline(x=75, color='gray', linestyle='--', alpha=0.5, label='Block 1-2 Transition')
# plt.axvline(x=159, color='gray', linestyle='--', alpha=0.5, label='Block 2-3 Transition')
# plt.axvline(x=243, color='gray', linestyle='--', alpha=0.5, label='Block 3-4 Transition')
# plt.title('React Time by Window Steps')
# plt.xlabel('Window Step')
# plt.ylabel('React Time')
# plt.savefig('./figures/react_time_moving_window.png', dpi=600, bbox_inches='tight')

# Fit quadratic regression model to optimal choice percentages
model = smf.ols('optimal_percentage ~ window_id + I(window_id**2) + C(Group)', data=lesas1_optimal_window_df).fit()
print(model.summary())

# Plot the fitted quadratic regression model
plt.clf()
g = sns.lmplot(data=lesas1_optimal_window_df, x='window_id', y='optimal_percentage', hue='Group',
               order=1, ci=95, height=6, aspect=1.5, scatter=False, palette='tab10', facet_kws={"legend_out": False})
g.set_axis_labels('Window Step', 'Optimal Choice Percentage')
plt.axvline(x=75, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=159, color='gray', linestyle='--', alpha=0.5)
plt.axvline(x=243, color='gray', linestyle='--', alpha=0.5)
plt.title('Optimal Choice Percentage by Window Steps with Quadratic Fit')
plt.tight_layout()
plt.savefig('./figures/optimal_choice_moving_window_quadratic_fit.png', dpi=600, bbox_inches='tight')

if __name__ == "__main__":
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
    # Now we group participants by their choice patterns and reanalyze the data
    # By Blocks
    # ----------------------------------------------------------------------------------------------------------------------
    # Group participants based on the final block they reached
    # Get participants who reached block 3 as their final block
    lesas1_max_block = lesas1_data_clean.groupby('SubNo')['Block'].max()
    lesas1_b3 = lesas1_data_raw[lesas1_data_raw['SubNo'].isin(lesas1_max_block[lesas1_max_block == 3].index)]
    lesas1_b4 = lesas1_data_raw[lesas1_data_raw['SubNo'].isin(lesas1_max_block[lesas1_max_block == 4].index)]
    print(f'Number of participants who reached Block 3: {len(lesas1_b3["SubNo"].unique())}')
    print(f'Number of participants who reached Block 4: {len(lesas1_b4["SubNo"].unique())}')

    lesas1_b3_optimal_window = behavioral_moving_window(lesas1_b3, window_size, exclusionary_criteria)
    lesas1_b4_optimal_window = behavioral_moving_window(lesas1_b4, window_size, exclusionary_criteria)

    # Create a plot for optimal choice percentages for Block 3
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=lesas1_b4_optimal_window, x='window_id', y='optimal_percentage', hue='Group')
    plt.axvline(x=75, color='gray', linestyle='--', alpha=0.5, label='Block 1-2 Transition')
    plt.axvline(x=159, color='gray', linestyle='--', alpha=0.5, label='Block 2-3 Transition')
    plt.axvline(x=243, color='gray', linestyle='--', alpha=0.5, label='Block 3-4 Transition')
    plt.title('Optimal Choice Percentage by Window Steps')
    plt.xlabel('Window Step')
    plt.ylabel('Optimal Choice Percentage')
    plt.savefig('./figures/optimal_choice_moving_window_b4.png', dpi=600, bbox_inches='tight')

    # ----------------------------------------------------------------------------------------------------------------------
    # By model fit
    # ----------------------------------------------------------------------------------------------------------------------
    # Create moving windows for each group
    value_based_optimal_window = behavioral_moving_window(value_based_participants, variable='Optimal_Choice',
                                                          window_size=window_size, exclusionary_criteria=exclusionary_criteria)
    value_based_rt_window = behavioral_moving_window(value_based_participants, variable='RT',
                                                          window_size=window_size, exclusionary_criteria=exclusionary_criteria)
    RT_based_optimal_window = behavioral_moving_window(RT_based_participants, variable='Optimal_Choice',
                                                       window_size=window_size, exclusionary_criteria=exclusionary_criteria)
    RT_based_rt_window = behavioral_moving_window(RT_based_participants, variable='RT',
                                                       window_size=window_size, exclusionary_criteria=exclusionary_criteria)
    RPUT_based_optimal_window = behavioral_moving_window(RPUT_based_participants, variable='Optimal_Choice',
                                                         window_size=window_size, exclusionary_criteria=exclusionary_criteria)
    RPUT_based_rt_window = behavioral_moving_window(RPUT_based_participants, variable='RT',
                                                         window_size=window_size, exclusionary_criteria=exclusionary_criteria)

    # # average optimal choice rate by model type
    # RPUT_averaged = RPUT_based_participants.groupby('SubNo')['Optimal_Choice'].mean().reset_index()
    #
    # # pick out the RT_based participants with the lowest average optimal choice rate
    # RPUT_worst_idx = RPUT_averaged.nsmallest(1, 'Optimal_Choice')
    # RPUT_worst = RPUT_based_optimal_window[(RPUT_based_optimal_window['participant_id'].isin(RPUT_worst_idx['SubNo'])) & (RPUT_based_optimal_window['window_id'] <= 243)]
    # RPUT_worst_rt = RPUT_based_participants[RPUT_based_participants['SubNo'].isin(RPUT_worst_idx['SubNo'])]
    # RPUT_worst_rt = RPUT_worst_rt.iloc[:60]
    #
    # RPUT_best_idx = RPUT_averaged.nlargest(1, 'Optimal_Choice')
    # RPUT_best = RPUT_based_optimal_window[(RPUT_based_optimal_window['participant_id'].isin(RPUT_best_idx['SubNo'])) & (RPUT_based_optimal_window['window_id'] <= 243)]
    # RPUT_best_rt = RPUT_based_participants[RPUT_based_participants['SubNo'].isin(RPUT_best_idx['SubNo'])]
    # RPUT_best_rt = RPUT_best_rt.iloc[:60]

    # Create a plot for optimal choice percentages by model fit
    group = RT_based_optimal_window
    group = group[group['window_id'] <= 243]
    palette = ['#0E2841', '#AD849F']

    # rename groups
    group['Group'] = group['Group'].map({'High-Reward-Optimal': 'Reward-Optimal Group', 'Low-Reward-Optimal': 'Reward-Suboptimal Group'})
    group['Group'] = pd.Categorical(group['Group'])

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=group, x='window_id', y='optimal_percentage', hue='Group', palette=palette)
    plt.axvline(x=75, color='gray', linestyle='--', alpha=0.5, label='Block Transition')
    plt.axvline(x=159, color='gray', linestyle='--', alpha=0.5)
    plt.ylim(0, 1)
    plt.title('RT-Based Model', fontsize=22)
    plt.xlabel('Window Step', fontsize=18)
    plt.ylabel('% Optimal Choice Percentage', fontsize=18)
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, labels=labels, title='Group / Transition', loc='lower left')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    sns.despine()
    plt.savefig('./figures/optimal_choice_moving_window_RT.png', dpi=600, bbox_inches='tight')

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=RT_based_rt_window, x='window_id', y='optimal_percentage', hue='Group')
    plt.axvline(x=75, color='gray', linestyle='--', alpha=0.5, label='Block 1-2 Transition')
    plt.axvline(x=159, color='gray', linestyle='--', alpha=0.5, label='Block 2-3 Transition')
    plt.axvline(x=243, color='gray', linestyle='--', alpha=0.5, label='Block 3-4 Transition')
    plt.ylim(0, 6)
    plt.title('RT by Window Steps')
    plt.xlabel('Window Step')
    plt.ylabel('RT')
    plt.savefig('./figures/rt_moving_window_RT.png', dpi=600, bbox_inches='tight')

    # Now analyze the total points earned by each group
    # rename group 1 and 2
    summary['Group'] = summary['Group'].map({1: 'Optimal-High-Reward', 2: 'Optimal-Low-Reward'})

    # t-test
    t_stat, p_value = stats.ttest_ind(
        summary[(summary['Group'] == 2) & (summary['model_type'] == 'RT_based')]['Optimal_Choice'], 0.5
    )
    print(f'\nT-test between groups for total points: t-statistic = {t_stat:.3f}, p-value = {p_value:.3f}')

    # ANOVA
    anova_results = pg.anova(data=summary, dv='Optimal_Choice', between=['model_type', 'Group'])
    print(summary.groupby(['model_type', 'Group'])['Optimal_Choice'].mean())
    # pairwise t-test
    pairwise_results = pg.pairwise_tests(data=summary, dv='Optimal_Choice', between=['model_type', 'Group'], padjust='fdr_bh')
    print(pairwise_results)

    anova_results_outcome = pg.anova(data=summary, dv='RT', between=['model_type', 'Group'])
    print(summary.groupby(['model_type', 'Group'])['OutcomeValue'].mean())
    print(summary.groupby(['model_type', 'Group'])['Cumulative_OutcomeValue'].mean())
    print(summary.groupby(['model_type', 'Group'])['RT'].mean())
    print(lesas1_data_clean.groupby(['model_type', 'Group', 'Optimal_Choice'])['RT'].mean())
    # pairwise t-test
    pairwise_results_outcome = pg.pairwise_tests(data=summary, dv='RT', between=['model_type', 'Group'], padjust='fdr_bh')


    # Create boxplot of points by model type and group
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=summary, x='model_type', y='Optimal_Choice', hue='Group')
    plt.title('Optimal Choice by Model Type and Group')
    plt.xlabel('Model Type')
    plt.ylabel('Optimal Choice')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('./figures/Optimal_Choice_by_model_type.png', dpi=600, bbox_inches='tight')

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=summary, x='model_type', y='Cumulative_OutcomeValue', hue='Group')
    plt.title('Total Reward Points by Model Type and Group')
    plt.xlabel('Model Type')
    plt.ylabel('Reward Points')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('./figures/points_by_model_type.png', dpi=600, bbox_inches='tight')

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=summary, x='model_type', y='RT', hue='Group')
    plt.title('RT by Model Type and Group')
    plt.xlabel('Model Type')
    plt.ylabel('RT')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('./figures/rt_by_model_type.png', dpi=600, bbox_inches='tight')

    # check b4 participants by model type
    b4_participants = lesas1_data_clean[lesas1_data_clean['SubNo'].isin(lesas1_max_block[lesas1_max_block == 4].index)]
    # print the number of participants in each model type
    print("\nB4 Participants by Model Type:")
    print(b4_participants.groupby(['Group', 'model_type'])['SubNo'].nunique())

    # ======================================================================================================================
    # LeDiSaS Behavioral Analysis
    # ======================================================================================================================
    # ----------------------------------------------------------------------------------------------------------------------
    # Behavioral Data Analysis
    # ----------------------------------------------------------------------------------------------------------------------
    # Define the number of blocks
    ledisas_data_3blocks = ledisas_data_clean[ledisas_data_clean['Block'].isin([1, 2, 3])]
    ledisas_data_4blocks = ledisas_data_clean[ledisas_data_clean['Block'].isin([1, 2, 3, 4])]

    # one-sample t-test for optimal choice percentage
    for data_block in [ledisas_data_3blocks, ledisas_data_4blocks]:
        print(f'\nAnalyzing data for {len(data_block["SubNo"].unique())} participants in {data_block["Block"].nunique()} blocks:')
        # Calculate % of optimal choices
        subj_means = data_block.groupby(['SubNo', 'Group'])['Optimal_Choice'].mean().reset_index()
        t_between, p_between = stats.ttest_ind(
            subj_means[subj_means['Group'] == 3]['Optimal_Choice'],
            subj_means[subj_means['Group'] == 4]['Optimal_Choice']
        )
        print(f'Average for Group 1: {subj_means[subj_means["Group"] == 3]["Optimal_Choice"].mean():.3f}')
        print(f'Average for Group 2: {subj_means[subj_means["Group"] == 4]["Optimal_Choice"].mean():.3f}')
        print(f'Between Groups - T-test: t-statistic = {t_between:.3f}, p-value = {p_between:.3f}')
        for group in [1, 2]:
            group_data = subj_means[subj_means['Group'] == group]['Optimal_Choice']
            t_stat, p_value = stats.ttest_1samp(group_data, 0.5)
            print(f'Group {group} - T-test: t-statistic = {t_stat:.3f}, p-value = {p_value:.3f}')

    # Perform mixed ANOVA for 3 blocks
    ledisas_mix_anova_results_3b = pg.mixed_anova(data=ledisas_data_3blocks, dv='Optimal_Choice', between='Group', within='Block', subject='SubNo')
    ledisas_mix_anova_results_4b = pg.mixed_anova(data=ledisas_data_4blocks, dv='Optimal_Choice', between='Group', within='Block', subject='SubNo')
    ledisas_pairwise_results_3b = pg.pairwise_tests(data=ledisas_data_3blocks, dv='Optimal_Choice', between='Group', within='Block', subject='SubNo', padjust='fdr_bh')
    ledisas_pairwise_results_4b = pg.pairwise_tests(data=ledisas_data_4blocks, dv='Optimal_Choice', between='Group', within='Block', subject='SubNo', padjust='fdr_bh')

    # ----------------------------------------------------------------------------------------------------------------------
    # Behavioral Windows
    # ----------------------------------------------------------------------------------------------------------------------
    # Define the window size
    window_size = 10
    ledisas_optimal_choices = []

    for _, participant_data in ledisas_data_raw.groupby('SubNo'):
        for i in range(len(participant_data) - window_size + 1):
            window = participant_data.iloc[i:i + window_size]
            window = exclusionary_criteria(window)
            optimal_percent = np.mean(window['Optimal_Choice'])
            ledisas_optimal_choices.append({
                'participant_id': participant_data['SubNo'].iloc[0],
                'window_id': i + 1,
                'optimal_percentage': optimal_percent,
                'Group': participant_data['Group'].iloc[0]
            })
    ledisas_optimal_window_df = pd.DataFrame(ledisas_optimal_choices)
    ledisas_optimal_window_df['Group'] = ledisas_optimal_window_df['Group'].map({1: 'OHRHV,SLRLV', 2: 'OHRLV,SLRHV', 3: 'OLRHV,SHRLV', 4: 'OLRLV,SHRHV'})

    # Create a plot for optimal choice percentages
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=ledisas_optimal_window_df, x='window_id', y='optimal_percentage', hue='Group')
    plt.axvline(x=75, color='gray', linestyle='--', alpha=0.5, label='Block 1-2 Transition')
    plt.axvline(x=159, color='gray', linestyle='--', alpha=0.5, label='Block 2-3 Transition')
    plt.axvline(x=243, color='gray', linestyle='--', alpha=0.5, label='Block 3-4 Transition')
    plt.title('Optimal Choice Percentage by Window Steps')
    plt.xlabel('Window Step')
    plt.ylabel('Optimal Choice Percentage')
    plt.savefig('./figures/ledisas_optimal_choice_moving_window.png', dpi=600, bbox_inches='tight')

