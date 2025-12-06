import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# from behavioral_analysis import lesas1_optimal_window_df
from model_fitting import lesas1_3block_dict
from utils.ComputationalModeling import parameter_extractor, bayes_factor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pingouin as pg
from scipy import stats
from plotting_functions import *
from utils.VisualSearchModels import create_model_summary_table, create_model_summary_df, VisualSearchModels
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Read in the cleaned data
lesas1_data_clean = pd.read_csv('./LeSaS1/Data/cleaned_data.csv')
ledisas_data_clean = pd.read_csv('./LeDiSaS/Data/cleaned_data.csv')

# Get the subset data with only subno and condition
print(lesas1_data_clean.columns)
lesas1_data_subset = lesas1_data_clean[['SubNo', 'Group(1=OptHighReward;2=OptLowReward)']].drop_duplicates()
lesas1_data_subset.columns = ['participant_id', 'Group']
lesas1_data_subset.to_csv('./LeSaS1/Data/group_assignment.csv', index=False)

ledisas_data_subset = ledisas_data_clean[['SubNo', 'Group(1=OHRHV,SLRLV; 2=OHRLV,SLRHV; 3=OLRHV,SHRLV, 4 = OLRLV,SHRHV)']].drop_duplicates()
ledisas_data_subset.columns = ['participant_id', 'Group']
ledisas_data_subset.to_csv('./ledisas/Data/group_assignment.csv', index=False)

# load all model files
lesas1_model_path = './LeSaS1/Model/'
lesas1_model_3blocks_path = './LeSaS1/Model/3block/'
ledisas_model_path = './LeDiSaS/Model/'
ledisas_model_3blocks_path = './LeDiSaS/Model/3block/'

dummy_model = VisualSearchModels('delta')
param_map = dummy_model._PARAM_MAP
param_names_by_model = {model: list(params.keys()) for model, params in param_map.items()}

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
            if model_name in param_names_by_model:
                model_results[model_name] = parameter_extractor(model_results[model_name], param_names_by_model[model_name])
            print(f'[{model_name}]')
            print(f'BIC: {model_results[model_name]["BIC"].mean():.2f}; AIC: {model_results[model_name]["AIC"].mean():.2f}')

    print(f'Best Model: {min(model_results, key=lambda x: model_results[x]["BIC"].mean())}; '
          f'BIC: {min([model_results[x]["BIC"].mean() for x in model_results]):.2f}; '
          f'AIC: {min([model_results[x]["AIC"].mean() for x in model_results]):.2f}')

    return model_results


# Load model results for LeSaS1
lesas1_model_results = load_model_results(lesas1_model_path, lesas1_data_subset)
# lesas1_3blocks_model_results = load_model_results(lesas1_model_3blocks_path, lesas1_data_subset)
ledisas_model_results = load_model_results(ledisas_model_path, ledisas_data_subset)
# ledisas_3blocks_model_results = load_model_results(ledisas_model_3blocks_path, ledisas_data_subset)

# Calculate % of optimal choices
lesas1_subj_means = lesas1_data_clean.groupby(['SubNo', 'Group(1=OptHighReward;2=OptLowReward)'])['Optimal_Choice'].mean().reset_index()
lesas1_subj_means.columns = ['participant_id', 'Group', 'Optimal_Choice']

# ----------------------------------------------------------------------------------------------------------------------
# Produce Model Fitting Summary
# ----------------------------------------------------------------------------------------------------------------------
# Create model summary tables for all models
create_model_summary_table(lesas1_model_results, './LeSaS1/full_model_summary.docx')
create_model_summary_table(lesas1_3blocks_model_results, './LeSaS1/3blocks_model_summary.docx')

# Create model summary tables for RT models
rt_models = ['RT_decay_PVL', 'RT_decay', 'RT_delta_PVL', 'RT_delta', 'RT_exp_basic', 'RT_exp_decay', 'RT_exp_delta']
create_model_summary_table({model: lesas1_model_results[model] for model in rt_models},
                            './LeSaS1/RT_model_summary.docx')

# Create model summary tables for reward-based models
reward_models = ['delta', 'decay', 'mean_var', 'mean_var_delta', 'mean_var_unc', 'kalman_filter']
create_model_summary_table({model: lesas1_model_results[model] for model in reward_models},
                            './LeSaS1/reward_model_summary.docx')
create_model_summary_table({model: ledisas_model_results[model] for model in reward_models},
                           './LeDiSaS/reward_model_summary.docx',
                           group_names=['OHRHV,SLRLV', 'OHRLV,SLRHV', 'OLRHV,SHRLV', 'OLRLV,SHRHV'])

# Create model summary tables for reward-based models
RPUT_models = ['delta_RPUT', 'decay_RPUT', 'delta_RPUT_unc', 'decay_RPUT_unc']
create_model_summary_table({model: lesas1_model_results[model] for model in RPUT_models},
                            './LeSaS1/RPUT_model_summary.docx')
create_model_summary_table({model: ledisas_model_results[model] for model in RPUT_models},
                           './LeDiSaS/RPUT_model_summary.docx',
                           group_names=['OHRHV,SLRLV', 'OHRLV,SLRHV', 'OLRHV,SHRLV', 'OLRLV,SHRHV'])

# Create model summary tables for hybrid models
hybrid_models = ['hybrid_decay_delta_3', 'hybrid_decay_delta', 'hybrid_delta_delta_3', 'hybrid_delta_delta',
                 'hybrid_WSLS_delta']
create_model_summary_table({model: lesas1_model_results[model] for model in hybrid_models},
                            './LeSaS1/hybrid_model_summary.docx')

# Create model summary tables for selected models
selected_models = ['delta', 'decay', 'delta_RPUT', 'decay_RPUT', 'RT_delta', 'RT_decay', 'delta_RPUT_unc', 'decay_RPUT_unc',
                   'mean_var_delta', 'mean_var_unc',
                   'perseveration', 'random']
selected_models = ['delta', 'decay', 'delta_RPUT', 'decay_RPUT', 'RT_delta', 'RT_decay',
                   'perseveration', 'random']
create_model_summary_table({model: lesas1_model_results[model] for model in selected_models},
                            './LeSaS1/selected_model_summary.docx')
# create_model_summary_table({model: lesas1_3blocks_model_results[model] for model in selected_models},
#                             './LeSaS1/3blocks_selected_model_summary.docx')
create_model_summary_table({model: ledisas_model_results[model] for model in selected_models},
                           './ledisas/selected_model_summary.docx',
                           group_names=['OHRHV,SLRLV', 'OHRLV,SLRHV', 'OLRHV,SHRLV', 'OLRLV,SHRHV'])
create_model_summary_table({model: ledisas_3blocks_model_results[model] for model in selected_models},
                            './ledisas/3blocks_selected_model_summary.docx',
                           group_names=['OHRHV,SLRLV', 'OHRLV,SLRHV', 'OLRHV,SHRLV', 'OLRLV,SHRHV'])

_, best = create_model_summary_df({model: lesas1_model_results[model] for model in selected_models}, return_best=True)

all_bics = pd.concat((df[['participant_id', 'Group', 'BIC']].assign(Model=name)
                      for name, df in {model: ledisas_model_results[model] for model in selected_models}.items()), ignore_index=True)
summary_bic = all_bics.pivot_table(index=['participant_id', 'Group'], columns='Model', values='BIC').reset_index()
# compare the best and the second best model
summary_bic['Best_Model'] = summary_bic[selected_models].idxmin(axis=1)
summary_bic['Best_BIC'] = summary_bic[selected_models].min(axis=1)
summary_bic['Second_Best_Model'] = summary_bic[selected_models].apply(lambda row: row[row != row.min()].idxmin(), axis=1)
summary_bic['Second_Best_BIC'] = summary_bic[selected_models].apply(lambda row: row[row != row.min()].min(), axis=1)
# calculate the difference between the best and the second best model
summary_bic['BIC_Diff'] = summary_bic['Second_Best_BIC'] - summary_bic['Best_BIC']
print(summary_bic['BIC_Diff'].describe())
print(f'Number of participants with BIC difference > 3: {(summary_bic["BIC_Diff"] > 1).sum()}')
summary_bic['Bayes_Factor'] = np.exp((summary_bic['Second_Best_BIC'] - summary_bic['Best_BIC']) / 2)
print(summary_bic['Bayes_Factor'].describe())
print(f'Number of participants with Bayes Factor > 3: {(summary_bic["Bayes_Factor"] > 3).sum()}')
# how many participants per group with Bayes Factor difference > 3
grouped = summary_bic[summary_bic['Bayes_Factor'] > 3].groupby(['Best_Model', 'Group']).size()
print('Number of participants with Bayes Factor > 3 per group:')
print(grouped)

def BIC_weights(bic_values):
    """Calculate BIC weights from BIC values."""
    min_bic = np.min(bic_values)
    delta_bic = bic_values - min_bic
    weights = np.exp(-0.5 * delta_bic)
    return weights / np.sum(weights)

bic_weights = summary_bic[selected_models].apply(BIC_weights, axis=1)
bic_weights['participant_id'] = summary_bic['participant_id']
bic_weights = bic_weights[['participant_id'] + selected_models]
bic_weights['max_weight_model'] = bic_weights[selected_models].idxmax(axis=1)
bic_weights['max_weight'] = bic_weights[selected_models].max(axis=1)

# check if the max weight model is the same as the best model
bic_weights = pd.merge(bic_weights, summary_bic[['participant_id', 'Best_Model']], on='participant_id', how='left')
bic_weights['model_match'] = bic_weights['max_weight_model'] == bic_weights['Best_Model']
print(f'Number of participants with matching best model and max weight model: {bic_weights["model_match"].sum()} out of {len(bic_weights)}')
bic_weights.to_csv('./LeSaS1/bic_weights.csv', index=False)

# save the value-based participants
lesas1_value_based_participants = best[best['Model'].isin(['delta', 'decay'])]['participant_id']
lesas1_RT_based_participants = best[best['Model'].isin(['RT_delta', 'RT_decay'])]['participant_id']
lesas1_RPUT_based_participants = best[best['Model'].isin(['delta_RPUT', 'decay_RPUT'])]['participant_id']
lesas1_perseveration_based_participants = best[best['Model'].isin(['perseveration'])]['participant_id']
lesas1_random_based_participants = best[best['Model'].isin(['random'])]['participant_id']
lesas1_value_based_participants.to_csv('./LeSaS1/Data/value_based_participants.csv', index=False)
lesas1_RT_based_participants.to_csv('./LeSaS1/Data/RT_based_participants.csv', index=False)
lesas1_RPUT_based_participants.to_csv('./LeSaS1/Data/RPUT_based_participants.csv', index=False)
lesas1_perseveration_based_participants.to_csv('./LeSaS1/Data/perseveration_based_participants.csv', index=False)
lesas1_random_based_participants.to_csv('./LeSaS1/Data/random_based_participants.csv', index=False)

# calculate the mean perseveration parameter for the perseveration based participants
perseveration = lesas1_model_results['perseveration']
perseveration_params = perseveration[perseveration['participant_id'].isin(lesas1_perseveration_based_participants)]
mean_perseveration = perseveration_params['w'].mean()
print(f'Mean perseveration parameter for perseveration based participants: {mean_perseveration:.2f}')

print('=' * 50)
print('Model summary tables created successfully.')

# Calculate bayes factor
refer_model = 'decay'
optimalhighreward_refer = lesas1_model_results[refer_model][lesas1_model_results[refer_model]['Group'] == 1]
optimallowreward_refer = lesas1_model_results[refer_model][lesas1_model_results[refer_model]['Group'] == 2]

for key, model in lesas1_model_results.items():
    if key in selected_models:
        optimalhighreward = model[model['Group'] == 1]
        optimallowreward = model[model['Group'] == 2]
        print(f'[{key}]')
        bf = bayes_factor(optimalhighreward, optimalhighreward_refer)
        print(f'Bayes Factor between decay and {key} for High-Reward-Optimal group: {bf:.2f}')
        bf = bayes_factor(optimallowreward, optimallowreward_refer)
        print(f'Bayes Factor between decay and {key} for Low-Reward-Optimal group: {bf:.2f}')


optimalhighreward_refer = lesas1_3blocks_model_results[refer_model][lesas1_3blocks_model_results[refer_model]['Group'] == 1]
optimallowreward_refer = lesas1_3blocks_model_results[refer_model][lesas1_3blocks_model_results[refer_model]['Group'] == 2]
for key, model in lesas1_3blocks_model_results.items():
    if key in selected_models:
        optimalhighreward = model[model['Group'] == 1]
        optimallowreward = model[model['Group'] == 2]
        print(f'[{key}]')
        bf = bayes_factor(optimalhighreward, optimalhighreward_refer)
        print(f'Bayes Factor between decay and {key} for High-Reward-Optimal group: {bf:.2f}')
        bf = bayes_factor(optimallowreward, optimallowreward_refer)
        print(f'Bayes Factor between decay and {key} for Low-Reward-Optimal group: {bf:.2f}')

# ======================================================================================================================
# Data Analysis
# ======================================================================================================================
# Correlation between optimal choice and parameters
hybrid_delta_delta = lesas1_model_results['hybrid_decay_decay']
hybrid_delta_delta = pd.merge(hybrid_delta_delta, lesas1_subj_means, on=['participant_id', 'Group'], how='left')
hybrid_delta_delta = parameter_extractor(hybrid_delta_delta, ['t', 'alpha', 'RTinitial', 'Weight'])
print(f'Weight mean for group 1: {hybrid_delta_delta[hybrid_delta_delta["Group"] == 1]["Weight"].mean():.2f}')
print(f'Weight mean for group 2: {hybrid_delta_delta[hybrid_delta_delta["Group"] == 2]["Weight"].mean():.2f}')
t_weight = stats.ttest_ind(
    hybrid_delta_delta[hybrid_delta_delta['Group'] == 1]['Weight'],
    hybrid_delta_delta[hybrid_delta_delta['Group'] == 2]['Weight']
)
print(f'Between Groups - T-test for Weight: t-statistic = {t_weight.statistic:.3f}, p-value = {t_weight.pvalue:.3f}')

hybrid_delta_delta.to_csv('./LeSaS1/Data/hybrid_delta_delta.csv', index=False)
corr = pg.pairwise_corr(hybrid_delta_delta, columns=['Optimal_Choice', 't', 'alpha', 'RTinitial', 'Weight', 'RT0Diff'], method='pearson')

hybrid_delta_delta_1 = hybrid_delta_delta[hybrid_delta_delta['Group'] == 1]
hybrid_delta_delta_2 = hybrid_delta_delta[hybrid_delta_delta['Group'] == 2]
corr_1 = pg.pairwise_corr(hybrid_delta_delta_1, columns=['Optimal_Choice', 't', 'alpha', 'RT', 'Weight', 'RT0Diff'], method='pearson')
corr_2 = pg.pairwise_corr(hybrid_delta_delta_2, columns=['Optimal_Choice', 't', 'alpha', 'RTinitial', 'Weight', 'RT0Diff'], method='pearson')

# ======================================================================================================================
# Parameter Analysis
# ======================================================================================================================
print(ledisas_model_results['mean_var_decay']['lamda'].groupby(ledisas_model_results['mean_var_decay']['Group']).mean())
print(ledisas_model_results['mean_var_delta']['lamda'].groupby(ledisas_model_results['mean_var_delta']['Group']).mean())
print(ledisas_model_results['mean_var_unc']['lamda'].groupby(ledisas_model_results['delta_RPUT_unc']['Group']).mean())
print(ledisas_model_results['delta_RPUT_unc']['lamda'].groupby(ledisas_model_results['delta_RPUT_unc']['Group']).mean())
print(ledisas_model_results['decay_RPUT_unc']['lamda'].groupby(ledisas_model_results['decay_RPUT_unc']['Group']).mean())

# visualize lambda distribution
for model_name in ['mean_var', 'mean_var_delta', 'mean_var_unc']:
    model_df = ledisas_model_results[model_name]
    # one panel per group
    g = sns.FacetGrid(model_df, col='Group', height=4, aspect=1)
    g.map(sns.histplot, 'lamda', bins=20, kde=True)
    g.set_axis_labels('Lambda', 'Count')
    g.set_titles(col_template='Group {col_name}')
    plt.suptitle(f'Lambda Distribution for {model_name} Model', y=1.05)
    plt.savefig(f'./figures/{model_name}_lambda_distribution.png', dpi=600, bbox_inches='tight')
    plt.close()

# ======================================================================================================================
# Moving Window Analysis
# ======================================================================================================================
# define parameter map
parameter_map = {
    'delta': ['t', 'alpha'],
    'decay': ['t', 'alpha'],
    'RT_delta': ['t', 'alpha', 'RT_initial'],
    'RT_decay': ['t', 'alpha', 'RT_initial'],
    'delta_RPUT': ['t', 'alpha'],
    'decay_RPUT': ['t', 'alpha'],
    'hybrid_delta_delta': ['t', 'alpha', 'RT_initial', 'weight'],
    'hybrid_decay_decay': ['t', 'alpha', 'RT_initial', 'weight'],
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
hybrid = model_mv_results['hybrid_decay_decay']
hybrid.to_csv('./LeSaS1/Data/hybrid_decay_decay.csv', index=False)
hybrid['Group'] = hybrid['Group'].replace({1: 'High-Reward-Optimal', 2: 'Low-Reward-Optimal'})
hybrid['RT_Weight'] = 1 - hybrid['weight']

# Create the plot
# separate by group
hybrid_value = hybrid[hybrid['participant_id'].isin(lesas1_value_based_participants)]
hybrid_RT = hybrid[hybrid['participant_id'].isin(lesas1_RT_based_participants)]
hybrid_RPUT = hybrid[hybrid['participant_id'].isin(lesas1_RPUT_based_participants)]
side_by_side_plot(hybrid_RT, 'window_id', 'weight', 'Window Step', 'Weight Value',
                  'Weight Values by Window Step Grouped by Participants', block_div=False)

# Visualize model fit by time
model_mv_results_selected = model_mv_results_df[model_mv_results_df['Model'].isin(['delta', 'decay', 'RT_delta', 'RT_decay',
                                                                                    'delta_RPUT', 'decay_RPUT'])]
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

# # Mixed effects model
# model = smf.mixedlm("Weight ~ Group * window_id", hybrid, groups=hybrid["participant_id"]).fit()
# print(model.summary())

# # quadratic model
# model_quadratic = smf.mixedlm("Weight ~ Group * I(window_id ** 2)", hybrid,
#                                 groups=hybrid["participant_id"]).fit()
# print(model_quadratic.summary())

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