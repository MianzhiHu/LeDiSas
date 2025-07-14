import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from behavioral_analysis import lesas1_optimal_window_df
from model_fitting import lesas1_3block_dict
from utils.ComputationalModeling import parameter_extractor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pingouin as pg
from scipy import stats
from plotting_functions import *
from utils.VisualSearchModels import create_model_summary_table, create_model_summary_df
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Read in the cleaned data
lesas1_data_clean = pd.read_csv('./LeSaS1/Data/cleaned_data.csv')
ledis1_data_clean = pd.read_csv('./LeDiS1/Data/cleaned_data.csv')

# Get the subset data with only subno and condition
print(lesas1_data_clean.columns)
lesas1_data_subset = lesas1_data_clean[['SubNo', 'Group(1=OptHighReward;2=OptLowReward)']].drop_duplicates()
lesas1_data_subset.columns = ['participant_id', 'Group']
lesas1_data_subset.to_csv('./LeSaS1/Data/group_assignment.csv', index=False)

ledis1_data_subset = ledis1_data_clean[['SubNo', 'Group(1=OptHighReward;2=OptLowReward)']].drop_duplicates()
ledis1_data_subset.columns = ['participant_id', 'Group']
ledis1_data_subset.to_csv('./LeDiS1/Data/group_assignment.csv', index=False)

# load all model files
lesas1_model_path = './LeSaS1/Model/'
lesas1_model_3blocks_path = './LeSaS1/Model/3block/'
lesas1_model_4blocks_path = './LeSaS1/Model/4thblock/'
ledis1_model_path = './LeDiS1/Model/'
ledis1_model_3blocks_path = './LeDiS1/Model/3block/'
ledis1_model_4blocks_path = './LeDiS1/Model/4thblock/'


# Function load model results
def load_model_results(model_path, group_assignment):
    # Dictionary to store model results
    model_results = {}

    # Iterate through files in the model directory
    for file in os.listdir(model_path):
        if file.endswith('.csv'):
            # Get model name from filename
            model_name = file.split('.')[0].replace('_results', '')
            model_results[model_name] = pd.read_csv(model_path + file)
            model_results[model_name] = pd.merge(group_assignment, model_results[model_name], on='participant_id', how='left')
            print(f'[{model_name}]')
            print(f'BIC: {model_results[model_name]["BIC"].mean():.2f}; AIC: {model_results[model_name]["AIC"].mean():.2f}')

    print(f'Best Model: {min(model_results, key=lambda x: model_results[x]["BIC"].mean())}; '
          f'BIC: {min([model_results[x]["BIC"].mean() for x in model_results]):.2f}; '
          f'AIC: {min([model_results[x]["AIC"].mean() for x in model_results]):.2f}')

    return model_results


# Load model results for LeSaS1
lesas1_model_results = load_model_results(lesas1_model_path, lesas1_data_subset)
lesas1_3blocks_model_results = load_model_results(lesas1_model_3blocks_path, lesas1_data_subset)
lesas1_4blocks_model_results = load_model_results(lesas1_model_4blocks_path, lesas1_data_subset)
ledis1_model_results = load_model_results(ledis1_model_path, ledis1_data_subset)
ledis1_3blocks_model_results = load_model_results(ledis1_model_3blocks_path, ledis1_data_subset)
ledis1_4blocks_model_results = load_model_results(ledis1_model_4blocks_path, ledis1_data_subset)

# Calculate % of optimal choices
lesas1_subj_means = lesas1_data_clean.groupby(['SubNo', 'Group(1=OptHighReward;2=OptLowReward)'])['Optimal_Choice'].mean().reset_index()
lesas1_subj_means.columns = ['participant_id', 'Group', 'Optimal_Choice']

# ----------------------------------------------------------------------------------------------------------------------
# Produce Model Fitting Summary
# ----------------------------------------------------------------------------------------------------------------------
# Create model summary tables for all models
create_model_summary_table(lesas1_model_results, './LeSaS1/full_model_summary.docx')
create_model_summary_table(lesas1_3blocks_model_results, './LeSaS1/3blocks_model_summary.docx')
create_model_summary_table(lesas1_4blocks_model_results, './LeSaS1/4blocks_model_summary.docx')

# Create model summary tables for RT models
rt_models = ['RT_decay_PVL', 'RT_decay', 'RT_delta_PVL', 'RT_delta', 'RT_exp_basic', 'RT_exp_decay', 'RT_exp_delta']
create_model_summary_table({model: lesas1_model_results[model] for model in rt_models},
                            './LeSaS1/RT_model_summary.docx')

# Create model summary tables for reward-based models
reward_models = ['decay_PVL', 'decay', 'decay_RPUT', 'delta_perseveration', 'delta_PVL', 'delta', 'delta_RPUT',
                 'dual_process', 'WSLS_decay_weight', 'WSLS_delta', 'WSLS_delta_weight', 'WSLS']
create_model_summary_table({model: lesas1_model_results[model] for model in reward_models},
                            './LeSaS1/reward_model_summary.docx')

# Create model summary tables for hybrid models
hybrid_models = ['hybrid_decay_delta_3', 'hybrid_decay_delta', 'hybrid_delta_delta_3', 'hybrid_delta_delta',
                 'hybrid_WSLS_delta']
create_model_summary_table({model: lesas1_model_results[model] for model in hybrid_models},
                            './LeSaS1/hybrid_model_summary.docx')

# Create model summary tables for selected models
selected_models = ['delta', 'delta_RPUT', 'decay', 'decay_RPUT', 'RT_delta', 'RT_decay']
create_model_summary_table({model: lesas1_model_results[model] for model in selected_models},
                            './LeSaS1/selected_model_summary.docx')
create_model_summary_table({model: lesas1_3blocks_model_results[model] for model in selected_models},
                            './LeSaS1/3blocks_selected_model_summary.docx')
create_model_summary_table({model: lesas1_4blocks_model_results[model] for model in selected_models},
                            './LeSaS1/4blocks_selected_model_summary.docx')
# create_model_summary_table({model: ledis1_model_results[model] for model in selected_models},
#                            './LeDiS1/selected_model_summary.docx')
# create_model_summary_table({model: ledis1_3blocks_model_results[model] for model in selected_models},
#                             './LeDiS1/3blocks_selected_model_summary.docx')

_, best = create_model_summary_df({model: lesas1_model_results[model] for model in selected_models}, return_best=True)

# save the value-based participants
lesas1_value_based_participants = best[best['Model'].isin(['delta', 'decay'])]['participant_id']
lesas1_RT_based_participants = best[best['Model'].isin(['RT_delta', 'RT_decay'])]['participant_id']
lesas1_RPUT_based_participants = best[best['Model'].isin(['delta_RPUT', 'decay_RPUT'])]['participant_id']
lesas1_value_based_participants.to_csv('./LeSaS1/Data/value_based_participants.csv', index=False)
lesas1_RT_based_participants.to_csv('./LeSaS1/Data/RT_based_participants.csv', index=False)
lesas1_RPUT_based_participants.to_csv('./LeSaS1/Data/RPUT_based_participants.csv', index=False)

print('=' * 50)
print('Model summary tables created successfully.')

# ======================================================================================================================
# Data Analysis
# ======================================================================================================================
# Correlation between optimal choice and parameters
hybrid_delta_delta = lesas1_model_results['hybrid_delta_delta']
hybrid_delta_delta = pd.merge(hybrid_delta_delta, lesas1_subj_means, on=['participant_id', 'Group'], how='left')
hybrid_delta_delta = parameter_extractor(hybrid_delta_delta, ['t', 'alpha', 'RT0Sub', 'RT0Opt', 'Weight'])
hybrid_delta_delta['RT0Diff'] = hybrid_delta_delta['RT0Opt'] - hybrid_delta_delta['RT0Sub']
print(f'Weight mean for group 1: {hybrid_delta_delta[hybrid_delta_delta["Group"] == 1]["Weight"].mean():.2f}')
print(f'Weight mean for group 2: {hybrid_delta_delta[hybrid_delta_delta["Group"] == 2]["Weight"].mean():.2f}')
t_weight = stats.ttest_ind(
    hybrid_delta_delta[hybrid_delta_delta['Group'] == 1]['Weight'],
    hybrid_delta_delta[hybrid_delta_delta['Group'] == 2]['Weight']
)
print(f'Between Groups - T-test for Weight: t-statistic = {t_weight.statistic:.3f}, p-value = {t_weight.pvalue:.3f}')

hybrid_delta_delta.to_csv('./LeSaS1/Data/hybrid_delta_delta.csv', index=False)
corr = pg.pairwise_corr(hybrid_delta_delta, columns=['Optimal_Choice', 't', 'alpha', 'RT0Sub', 'RT0Opt', 'Weight', 'RT0Diff'], method='pearson')

hybrid_delta_delta_1 = hybrid_delta_delta[hybrid_delta_delta['Group'] == 1]
hybrid_delta_delta_2 = hybrid_delta_delta[hybrid_delta_delta['Group'] == 2]
corr_1 = pg.pairwise_corr(hybrid_delta_delta_1, columns=['Optimal_Choice', 't', 'alpha', 'RT0Sub', 'RT0Opt', 'Weight', 'RT0Diff'], method='pearson')
corr_2 = pg.pairwise_corr(hybrid_delta_delta_2, columns=['Optimal_Choice', 't', 'alpha', 'RT0Sub', 'RT0Opt', 'Weight', 'RT0Diff'], method='pearson')

# ======================================================================================================================
# Moving Window Analysis
# ======================================================================================================================
# define parameter map
parameter_map = {
    'delta': ['t', 'alpha'],
    'decay': ['t', 'alpha'],
    'WSLS': ['t', 'alpha'],
    'RT_delta': ['t', 'alpha', 'RT0Sub', 'RT0Opt'],
    'hybrid_delta_delta': ['t', 'alpha', 'RT0Sub', 'RT0Opt', 'Weight'],
    'hybrid_decay_delta': ['t', 'alpha', 'RT0Sub', 'RT0Opt', 'Weight'],
    'hybrid_WSLS_delta': ['t', 'alpha', 'p_ws', 'P_ls', 'RT0Sub', 'RT0Opt', 'Weight'],
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
        model_mv_results[model_name] = pd.merge(model_mv_results[model_name], lesas1_optimal_window_df,
                                                on=['participant_id', 'window_id'], how='left')
        # use the parameter map to extract the parameters
        model_mv_results[model_name] = parameter_extractor(model_mv_results[model_name], parameter_map[model_name])
        # print average AIC and BIC for each group
        print(f'[{model_name}]')
        print(f'Average AIC by Group:\n{model_mv_results[model_name].groupby("Group")["AIC"].mean()}')
        print(f'Average BIC by Group:\n{model_mv_results[model_name].groupby("Group")["BIC"].mean()}')


# get a complete df
model_mv_results_df = pd.DataFrame()
for model_name in model_mv_results:
    model_mv_results[model_name]['Model'] = model_name
    model_mv_results_df = pd.concat([model_mv_results_df, model_mv_results[model_name]], axis=0)
print(model_mv_results_df['Model'].value_counts())

# Visualize hybrid delta delta
hybrid = model_mv_results['hybrid_delta_delta']
hybrid.to_csv('./LeSaS1/Data/hybrid_decay_delta.csv', index=False)
hybrid['Group'] = hybrid['Group'].replace({1: 'High-Reward-Optimal', 2: 'Low-Reward-Optimal'})
hybrid['RT_Weight'] = 1 - hybrid['Weight']

# Create the plot
side_by_side_plot(hybrid, 'window_id', 'Weight', 'Window Step', 'Weight Value',
                  'Weight Values by Window Step Grouped by Participants', block_div=False)

# Visualize model fit by time
model_mv_results_selected = model_mv_results_df[model_mv_results_df['Model'].isin(['delta', 'decay', 'RT_delta'])]
model_mv_results_selected['Group'] = model_mv_results_selected['Group'].replace({1: 'High-Reward-Optimal', 2: 'Low-Reward-Optimal'}).copy()

for group in ['High-Reward-Optimal', 'Low-Reward-Optimal']:
    plt.figure(figsize=(10, 6))
    group_data = model_mv_results_selected[model_mv_results_selected['Group'] == group]
    sns.lineplot(data=group_data, x='window_id', y='BIC', hue='Model')
    plt.axvline(x=75, color='gray', linestyle='--', alpha=0.5, label='Block 1-2 Transition')
    plt.axvline(x=159, color='gray', linestyle='--', alpha=0.5, label='Block 2-3 Transition')
    plt.axvline(x=243, color='gray', linestyle='--', alpha=0.5, label='Block 3-4 Transition')
    plt.title(f'BIC Values by Window Step for Group {group}')
    plt.xlabel('Window Step')
    plt.ylabel('BIC Value')
    plt.legend(title='Model')
    plt.savefig(f'./figures/BIC_moving_window_group_{group}.png', dpi=600, bbox_inches='tight')
    plt.close()

# Mixed effects model
model = smf.mixedlm("Weight ~ Group * window_id", hybrid, groups=hybrid["participant_id"]).fit()
print(model.summary())

# quadratic model
model_quadratic = smf.mixedlm("Weight ~ Group * I(window_id ** 2)", hybrid,
                                groups=hybrid["participant_id"]).fit()
print(model_quadratic.summary())

# plot the quadratic model
g = sns.lmplot(
    data=hybrid,
    x="window_id",
    y="RT_Weight",
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

g.set_axis_labels("Window Step", "Weight of RT")
g.set_titles(col_template="{col_name}")
g.fig.suptitle("Quadratic Fit of Weight vs. Window by Group", y=1.02)

plt.tight_layout()
plt.savefig('./figures/weight_moving_window_blockwise.png', dpi=600, bbox_inches='tight')
plt.show()

# ======================================================================================================================
# # Perform regression analysis for each group
# regression_results = []
# for window_id in range(1, 318):
#     window_data = hybrid[hybrid['window_id'] == window_id]
#     print(f'Processing window {window_id} with {len(window_data)} participants.')
#
#     if len(window_data) < 10:
#         print(f'Window {window_id} has less than 10 participants. Skipping.')
#         continue  # Skip windows with less than 10 participants. There should not be any, but just in case.
#
#     for group in [1, 2]:
#         group_data = window_data[window_data['Group'] == group]
#         if len(group_data) < 10:
#             print(f'Group {group} in window {window_id} has less than 10 participants. Skipping.')
#             continue  # Skip groups with less than 10 participants. There should not be any, but just in case.
#
#         # Perform linear regression
#         lr_results = pg.linear_regression(group_data['Weight'], group_data['optimal_percentage'])
#         slope = lr_results['coef'][1]
#         intercept = lr_results['coef'][0]
#         r_squared = lr_results['r2'][1]
#         p_value = lr_results['pval'][1]
#         upper_ci = lr_results['CI[2.5%]'][1]
#         lower_ci = lr_results['CI[97.5%]'][1]
#         regression_results.append({
#             'window_id': window_id,
#             'Group': group,
#             'slope': slope,
#             'intercept': intercept,
#             'r_squared': r_squared,
#             'p_value': p_value,
#             'upper_ci': upper_ci,
#             'lower_ci': lower_ci
#         })
# regression_results_df = pd.DataFrame(regression_results)
# regression_results_df.to_csv('./LeSaS1/Data/regression_results.csv', index=False)
#
# # Plot regression results
# for group in [1, 2]:
#     group_data = regression_results_df[regression_results_df['Group'] == group]
#     plt.figure(figsize=(10, 6))
#     sns.lineplot(data=group_data, x='window_id', y='slope', label='Slope')
#     sns.lineplot(data=group_data, x='window_id', y='p_value', label='Intercept')
#     plt.axhline(y=0.05, color='r', linestyle='--')
#     plt.fill_between(group_data['window_id'], group_data['lower_ci'], group_data['upper_ci'], alpha=0.2)
#     plt.title(f'Regression Results for Group {group}')
#     plt.xlabel('Window Step')
#     plt.ylabel('Regression Coefficient')
#     plt.legend()
#     plt.savefig(f'./figures/regression_results_group_{group}.png', dpi=600, bbox_inches='tight')