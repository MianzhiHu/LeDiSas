import pandas as pd
import numpy as np
from utils.VisualSearchModels import VisualSearchModels
from utils.ComputationalModeling import dict_generator, moving_window_model_fitting
from utils.DualProcess import DualProcessModel
from utils.VisualSearchModels import VisualSearchModels
from preprocessing import calculate_rt_stats

# Read in the cleaned data
lesas1_data_raw = pd.read_csv('./LeSaS1/Data/raw_data.csv')
lesas1_data_clean = pd.read_csv('./LeSaS1/Data/cleaned_data.csv')
ledis1_data_raw = pd.read_csv('./LeDiS1/Data/raw_data.csv')
ledis1_data_clean = pd.read_csv('./LeDiS1/Data/cleaned_data.csv')

lesas1_trial_per_participant = lesas1_data_raw.groupby('SubNo').size().reset_index(name='TrialCount')
lesas1_freq = lesas1_trial_per_participant['TrialCount'].value_counts().reset_index()
print(f'LeSaS1 Trial Count Frequency:\n{lesas1_freq}')

ledis1_trial_per_participant = ledis1_data_raw.groupby('SubNo').size().reset_index(name='TrialCount')
ledis1_freq = ledis1_trial_per_participant['TrialCount'].value_counts().reset_index()
print(f'LeDiS1 Trial Count Frequency:\n{ledis1_freq}')

lesas1_full_dict = dict_generator(lesas1_data_clean, task='VS')
lesas1_3block_dict = dict_generator(lesas1_data_clean[lesas1_data_clean['Block'] <= 3], task='VS')
lesas1_4th_block_dict = dict_generator(lesas1_data_clean[lesas1_data_clean['Block'] == 4], task='VS')

ledis1_full_dict = dict_generator(ledis1_data_clean, task='VS')
ledis1_3block_dict = dict_generator(ledis1_data_clean[ledis1_data_clean['Block'] <= 3], task='VS')
ledis1_4th_block_dict = dict_generator(ledis1_data_clean[ledis1_data_clean['Block'] == 4], task='VS')

# Calculate RT stats for exclusionary criteria
lesas1_data_raw = calculate_rt_stats(lesas1_data_raw)
ledis1_data_raw = calculate_rt_stats(ledis1_data_raw)

if __name__ == "__main__":
    # Define the directories for model fitting results
    lesas1_full_folder = './LeSaS1/Model/'
    lesas1_3block_folder = './LeDiS1/Model/3block/'
    lesas1_4thblock_folder = './LeDiS1/Model/4thblock/'

    ledis1_full_folder = './LeDiS1/Model/'
    ledis1_3block_folder = './LeDiS1/Model/3block/'
    ledis1_4thblock_folder = './LeDiS1/Model/4thblock/'

    # Define the models
    delta = VisualSearchModels('delta')
    delta_perseveration = VisualSearchModels('delta_perseveration')
    delta_PVL = VisualSearchModels('delta_PVL_relative')
    delta_RPUT = VisualSearchModels('delta_RPUT', condition='Both')
    decay = VisualSearchModels('decay')
    decay_PVL = VisualSearchModels('decay_PVL_relative')
    decay_RPUT = VisualSearchModels('decay_RPUT', condition='Both')
    WSLS = VisualSearchModels('WSLS')
    WSLS_delta = VisualSearchModels('WSLS_delta')
    WSLS_delta_weight = VisualSearchModels('WSLS_delta_weight')
    WSLS_decay_weight = VisualSearchModels('WSLS_decay_weight')
    dual_process = DualProcessModel(task='IGT_SGT', num_options=2)
    RT_exp_basic = VisualSearchModels('RT_exp_basic')
    RT_delta = VisualSearchModels('RT_delta')
    RT_delta_PVL = VisualSearchModels('RT_delta_PVL')
    RT_decay = VisualSearchModels('RT_decay')
    RT_decay_PVL = VisualSearchModels('RT_decay_PVL')
    RT_exp_delta = VisualSearchModels('RT_exp_delta')
    RT_exp_decay = VisualSearchModels('RT_exp_decay')
    hybrid_delta_delta = VisualSearchModels('hybrid_delta_delta')
    hybrid_delta_delta_3 = VisualSearchModels('hybrid_delta_delta_3')
    hybrid_decay_delta = VisualSearchModels('hybrid_decay_delta')
    hybrid_decay_delta_3 = VisualSearchModels('hybrid_decay_delta_3')

    model_names = ['delta', 'delta_PVL', 'delta_RPUT', 'decay', 'decay_PVL', 'decay_RPUT', 'WSLS', 'WSLS_delta',
                   'WSLS_delta_weight', 'WSLS_decay_weight', 'RT_exp_basic', 'RT_delta', 'RT_delta_PVL', 'RT_decay',
                   'RT_decay_PVL', 'RT_exp_delta', 'RT_exp_decay', 'hybrid_delta_delta', 'hybrid_delta_delta_3',
                   'hybrid_decay_delta', 'hybrid_decay_delta_3', 'delta_perseveration']
    model_list = [delta, delta_PVL, delta_RPUT, decay, decay_PVL, decay_RPUT, WSLS, WSLS_delta,
                  WSLS_delta_weight, WSLS_decay_weight, RT_exp_basic, RT_delta, RT_delta_PVL, RT_decay,
                  RT_decay_PVL, RT_exp_delta, RT_exp_decay, hybrid_delta_delta, hybrid_delta_delta_3,
                  hybrid_decay_delta, hybrid_decay_delta_3, delta_perseveration]

    lesas1_folders = [lesas1_full_folder, lesas1_3block_folder, lesas1_4thblock_folder]
    ledis1_folders = [ledis1_full_folder, ledis1_3block_folder, ledis1_4thblock_folder]

    # ==================================================================================================================
    # LeSaS1 Model Fitting (4 blocks; 3 blocks; 4th block only)
    # ==================================================================================================================
    # Whole-task model fitting
    n_iterations = 100

    for i, lesas1_data in enumerate([lesas1_full_dict, lesas1_3block_dict, lesas1_4th_block_dict]):
        for j, model in enumerate(model_list):
                save_dir = f'{lesas1_folders[i]}{model_names[j]}_results.csv'
                # Check if the file already exists
                try:
                 existing_results = pd.read_csv(save_dir)
                 if not existing_results.empty:
                      print(f"File {save_dir} already exists. Skipping model fitting.")
                      continue
                except FileNotFoundError:
                 pass

                # Fit the model to the data
                model_results = model.fit(lesas1_data, num_iterations=n_iterations)
                model_results.to_csv(save_dir, index=False)

    # ------------------------------------------------------------------------------------------------------------------
    # Fit the models with sliding window
    # ------------------------------------------------------------------------------------------------------------------
    # Define exclusionary criteria for sliding window model fitting (This is the same as the one used in preprocessing)
    def exclusionary_criteria(data):
        # Exclude non-responses
        data = data[~data['Optimal_Choice'].isna()].copy()
        # Change the dtype of optimal choice to int if it is not NaN
        data['Optimal_Choice'] = data['Optimal_Choice'].astype(int)
        # Remove trials with RT > 3 SD from subject's mean
        data = data[data['RT'] <= (data['rt_mean'] + 3 * data['rt_std'])]
        # Remove trials with RT < 0.3 seconds
        data = data[data['RT'] >= 0.3]
        # Drop the helper columns
        data = data.drop(['rt_mean', 'rt_std'], axis=1)
        return data

    # n_iterations = 1
    # window_size = 10
    # hybrid_mv = moving_window_model_fitting(lesas1_data_raw, hybrid_delta_delta, task='VS', window_size=window_size,
    #                                         id_col='SubNo', filter_fn=exclusionary_criteria, num_iterations=n_iterations)
    # hybrid_mv.to_csv('./LeSaS1/Model/hybrid_delta_delta_mv_results.csv', index=False)
    # delta_mv = moving_window_model_fitting(data, delta, task='VS', window_size=window_size, id_col='SubNo',
    #                                        num_iterations=n_iterations)
    # delta_mv.to_csv('./LeSaS1/Model/Moving_Window/delta_mv_results.csv', index=False)
    # RT_delta_mv = moving_window_model_fitting(data, RT_delta, task='VS', window_size=window_size, id_col='SubNo',
    #                                     num_iterations=n_iterations)
    # RT_delta_mv.to_csv('./LeSaS1/Model/Moving_Window/RT_delta_mv_results.csv', index=False)
    # hybrid_delta_delta_3_mv = moving_window_model_fitting(data, hybrid_delta_delta_3, task='VS',
    #                                                       window_size=window_size, id_col='SubNo',
    #                                                       num_iterations=n_iterations)
    # hybrid_delta_delta_3_mv.to_csv('./LeSaS1/Model/Moving_Window/hybrid_delta_delta_3_mv_results.csv', index=False)

    # ==================================================================================================================
    # LeDiS1 Model Fitting (4 blocks; 3 blocks; 4th block only)
    # ==================================================================================================================
    # Whole-task model fitting
    for i, ledis1_data in enumerate([ledis1_full_dict, ledis1_3block_dict, ledis1_4th_block_dict]):
        for j, model in enumerate(model_list):
                save_dir = f'{ledis1_folders[i]}{model_names[j]}_results.csv'
                # Check if the file already exists
                try:
                 existing_results = pd.read_csv(save_dir)
                 if not existing_results.empty:
                      print(f"File {save_dir} already exists. Skipping model fitting.")
                      continue
                except FileNotFoundError:
                 pass

                # Fit the model to the data
                model_results = model.fit(ledis1_data, num_iterations=n_iterations)
                model_results.to_csv(save_dir, index=False)

    # ==================================================================================================================
    # Block-wise model fitting (Unused)
    # ==================================================================================================================
    # Fit block-wise model
    # model_names = ['delta', 'delta_RPUT', 'RT_delta', 'RT_exp_basic', 'RT_exp_delta', 'hybrid_delta_delta', 'hybrid_delta_delta_3']
    # for i in range(1, max(data['Block']) + 1):
    #     block_data = data[data['Block'] == i]
    #     block_dict = dict_generator(block_data, task='VS')
    #     for j, model in enumerate([delta, RT_delta, RT_exp_basic, RT_exp_delta, hybrid_delta_delta, hybrid_delta_delta_3]):
    #         save_dir = f'./LeSaS1/Model/Blockwise/{model_names[j]}_block_{i}_results.csv'
    #         # Check if the file already exists
    #         try:
    #             existing_results = pd.read_csv(save_dir)
    #             if not existing_results.empty:
    #                 print(f"File {save_dir} already exists. Skipping model fitting.")
    #                 continue
    #         except FileNotFoundError:
    #             pass
    #         model_results = model.fit(block_dict, num_iterations=n_iterations)
    #         model_results.to_csv(save_dir, index=False)

    # # Fit block-wise sliding window model
    # window_size = 10
    # model_names = ['delta', 'delta_RPUT', 'RT_delta', 'RT_exp_basic', 'RT_exp_delta', 'hybrid_delta_delta', 'hybrid_delta_delta_3']
    # for i in range(1, max(data['Block']) + 1):
    #     block_data = data[data['Block'] == i]
    #     block_dict = dict_generator(block_data, task='VS')
    #     for j, model in enumerate([delta, RT_delta, RT_exp_basic, RT_exp_delta, hybrid_delta_delta, hybrid_delta_delta_3]):
    #         save_dir = f'./LeSaS1/Model/BlockWise_Moving_Window/{model_names[j]}_block_{i}_mv_results.csv'
    #         # Check if the file already exists
    #         try:
    #             existing_results = pd.read_csv(save_dir)
    #             if not existing_results.empty:
    #                 print(f"File {save_dir} already exists. Skipping model fitting.")
    #                 continue
    #         except FileNotFoundError:
    #             pass
    #         model_results = moving_window_model_fitting(block_data, model, task='VS', window_size=window_size,
    #                                                     id_col='SubNo', num_iterations=n_iterations)
    #         model_results.to_csv(save_dir, index=False)
