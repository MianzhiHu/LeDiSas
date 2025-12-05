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
ledisas_data_raw = pd.read_csv('./Ledisas/Data/raw_data.csv')
ledisas_data_clean = pd.read_csv('./Ledisas/Data/cleaned_data.csv')

lesas1_full_dict = dict_generator(lesas1_data_clean, task='VS')
lesas1_3block_dict = dict_generator(lesas1_data_clean[lesas1_data_clean['Block'] <= 3], task='VS')

ledisas_full_dict = dict_generator(ledisas_data_clean, task='VS')
ledisas_3block_dict = dict_generator(ledisas_data_clean[ledisas_data_clean['Block'] <= 3], task='VS')

# Calculate RT stats for exclusionary criteria
lesas1_data_raw = calculate_rt_stats(lesas1_data_raw)
ledisas_data_raw = calculate_rt_stats(ledisas_data_raw)


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

if __name__ == "__main__":
    # Define the directories for model fitting results
    lesas1_full_folder = './LeSaS1/Model/'
    lesas1_3block_folder = './LeSaS1/Model/3block/'

    ledisas_full_folder = './Ledisas/Model/'
    ledisas_3block_folder = './Ledisas/Model/3block/'

    # Define the models
    delta = VisualSearchModels('delta')
    delta_perseveration = VisualSearchModels('delta_perseveration')
    delta_PVL = VisualSearchModels('delta_PVL_relative')
    delta_RPUT = VisualSearchModels('delta_RPUT')
    delta_RPUT_uncertainty = VisualSearchModels('delta_RPUT_unc')
    delta_uncertainty = VisualSearchModels('delta_uncertainty')
    decay = VisualSearchModels('decay')
    decay_PVL = VisualSearchModels('decay_PVL_relative')
    decay_RPUT = VisualSearchModels('decay_RPUT')
    decay_RPUT_uncertainty = VisualSearchModels('decay_RPUT_unc')
    decay_uncertainty = VisualSearchModels('decay_uncertainty')
    WSLS = VisualSearchModels('WSLS')
    WSLS_delta = VisualSearchModels('WSLS_delta')
    WSLS_delta_weight = VisualSearchModels('WSLS_delta_weight')
    WSLS_decay_weight = VisualSearchModels('WSLS_decay_weight')
    dual_process = DualProcessModel(task='IGT_SGT')
    dual_process_rt = DualProcessModel(task='IGT_SGT')
    mean_var_delta = VisualSearchModels('mean_var_delta')
    mean_var_decay = VisualSearchModels('mean_var_decay')
    kalman = VisualSearchModels('kalman_filter')
    RT_exp_basic = VisualSearchModels('RT_exp_basic')
    RT_delta = VisualSearchModels('RT_delta')
    RT_delta_PVL = VisualSearchModels('RT_delta_PVL')
    RT_decay = VisualSearchModels('RT_decay')
    RT_decay_PVL = VisualSearchModels('RT_decay_PVL')
    RT_exp_delta = VisualSearchModels('RT_exp_delta')
    hybrid_delta_delta = VisualSearchModels('hybrid_delta_delta')
    hybrid_delta_delta_3 = VisualSearchModels('hybrid_delta_delta_3')
    hybrid_decay_delta = VisualSearchModels('hybrid_decay_delta')
    hybrid_decay_delta_3 = VisualSearchModels('hybrid_decay_delta_3')
    hybrid_decay_decay = VisualSearchModels('hybrid_decay_decay')
    hybrid_decay_decay_3 = VisualSearchModels('hybrid_decay_decay_3')

    # model_names = ['delta', 'delta_PVL', 'delta_RPUT', 'decay', 'decay_PVL', 'decay_RPUT', 'WSLS', 'WSLS_delta',
    #                'WSLS_delta_weight', 'WSLS_decay_weight', 'RT_exp_basic', 'RT_delta', 'RT_delta_PVL', 'RT_decay',
    #                'RT_decay_PVL', 'RT_exp_delta', 'RT_exp_decay', 'hybrid_delta_delta', 'hybrid_delta_delta_3',
    #                'hybrid_decay_delta', 'hybrid_decay_delta_3', 'delta_perseveration', 'hybrid_WSLS_delta', 'dual_process']
    # model_list = [delta, delta_PVL, delta_RPUT, decay, decay_PVL, decay_RPUT, WSLS, WSLS_delta,
    #               WSLS_delta_weight, WSLS_decay_weight, RT_exp_basic, RT_delta, RT_delta_PVL, RT_decay,
    #               RT_decay_PVL, RT_exp_delta, RT_exp_decay, hybrid_delta_delta, hybrid_delta_delta_3,
    #               hybrid_decay_delta, hybrid_decay_delta_3, delta_perseveration, hybrid_WSLS_delta, dual_process]

    model_names = ['delta', 'decay', 'RT_delta', 'RT_decay', 'delta_RPUT', 'decay_RPUT', 'hybrid_delta_delta',
                   'hybrid_decay_delta', 'hybrid_decay_decay', 'mean_var_delta', 'mean_var_decay', 'kalman_filter',
                   'delta_RPUT_unc', 'decay_RPUT_unc']

    model_list = [delta, decay, RT_delta, RT_decay, delta_RPUT, decay_RPUT, hybrid_delta_delta, hybrid_decay_delta,
                  hybrid_decay_decay, mean_var_delta, mean_var_decay, kalman,
                  delta_RPUT_uncertainty, decay_RPUT_uncertainty]

    moving_window_model_names = ['delta', 'decay', 'RT_delta', 'RT_decay', 'delta_RPUT', 'decay_RPUT',
                                 'hybrid_delta_delta', 'hybrid_decay_delta', 'hybrid_decay_decay']
    moving_window_model_list = [delta, decay, RT_delta, RT_decay, delta_RPUT, decay_RPUT, hybrid_delta_delta,
                                hybrid_decay_delta, hybrid_decay_decay]

    lesas1_folders = [lesas1_full_folder, lesas1_3block_folder]
    ledisas_folders = [ledisas_full_folder, ledisas_3block_folder]

    n_iterations = 100
    window_size = 10

    # # ==================================================================================================================
    # # LeSaS1 Model Fitting (4 blocks; 3 blocks)
    # # ==================================================================================================================
    # # Whole-task model fitting
    # for i, lesas1_dict in enumerate([lesas1_full_dict, lesas1_3block_dict]):
    #     for j, model in enumerate(model_list):
    #             save_dir = f'{lesas1_folders[i]}{model_names[j]}_results.csv'
    #             # Check if the file already exists
    #             try:
    #              existing_results = pd.read_csv(save_dir)
    #              if not existing_results.empty:
    #                   print(f"File {save_dir} already exists. Skipping model fitting.")
    #                   continue
    #             except FileNotFoundError:
    #              pass
    #
    #             # If the model is dual-process, fit it with specific parameters
    #             if model_names[j] == 'dual_process':
    #                 model_results = dual_process.fit(lesas1_dict, 'Dual_Process_t2', Gau_fun='Naive_Recency',
    #                                  Dir_fun='Linear_Recency', weight_Dir='softmax', weight_Gau='softmax',
    #                                  num_training_trials=999, num_exp_restart=9999, initial_EV=[0.5, 0.5],
    #                                  initial_mode='fixed', num_iterations=n_iterations)
    #
    #             else:
    #                 # Fit the model to the data
    #                 model_results = model.fit(lesas1_dict, num_iterations=n_iterations, initial_mode='fixed',
    #                                           initial_EV=[0.5, 0.5])
    #
    #             model_results.to_csv(save_dir, index=False)
    #
    # # ------------------------------------------------------------------------------------------------------------------
    # # Fit the models with sliding window
    # # ------------------------------------------------------------------------------------------------------------------
    #
    # for i, model in enumerate(moving_window_model_list):
    #     save_dir = f'./LeSaS1/Model/Moving_Window/{moving_window_model_names[i]}_results.csv'
    #     # Check if the file already exists
    #     try:
    #         existing_results = pd.read_csv(save_dir)
    #         if not existing_results.empty:
    #             print(f"File {save_dir} already exists. Skipping model fitting.")
    #             continue
    #     except FileNotFoundError:
    #         pass
    #
    #     # Fit the model to the data with a sliding window
    #     model_results = moving_window_model_fitting(lesas1_data_raw, model, task='VS', id_col='SubNo',
    #                                                 num_iterations=n_iterations, window_size=window_size,
    #                                                 filter_fn=exclusionary_criteria, restart_EV=True,
    #                                                 initial_EV=[0.5, 0.5], initial_mode='fixed')
    #     model_results.to_csv(save_dir, index=False)

    # ==================================================================================================================
    # Ledisas Model Fitting (4 blocks; 3 blocks)
    # ==================================================================================================================
    # Whole-task model fitting
    for i, ledisas_data in enumerate([ledisas_full_dict, ledisas_3block_dict]):
        for j, model in enumerate(model_list):
                save_dir = f'{ledisas_folders[i]}{model_names[j]}_results.csv'
                # Check if the file already exists
                try:
                 existing_results = pd.read_csv(save_dir)
                 if not existing_results.empty:
                      print(f"File {save_dir} already exists. Skipping model fitting.")
                      continue
                except FileNotFoundError:
                 pass

                # If the model is dual-process, fit it with specific parameters
                if model_names[j] == 'dual_process':
                    model_results = dual_process.fit(ledisas_data, 'Dual_Process_t2', Gau_fun='Naive_Recency',
                                                     Dir_fun='Linear_Recency', weight_Dir='softmax', weight_Gau='softmax',
                                                     num_training_trials=999, num_exp_restart=9999,
                                                     initial_mode='first_trial_no_alpha',num_iterations=n_iterations)
                elif model_names[j] == 'dual_process_rt':
                    rt_data = ledisas_data.copy()
                    rt_data[1].pop("reward", None)
                    rt_data[1]["reward"] = rt_data[1].pop("react_time")
                    model_results = dual_process_rt.fit(rt_data, 'Dual_Process_Visual', Gau_fun='Naive_Recency',
                                                        Dir_fun='Linear_Recency_VS', weight_Dir='softmax', weight_Gau='softmax',
                                                        num_training_trials=999, num_exp_restart=9999,
                                                        initial_mode='first_trial_no_alpha', num_iterations=n_iterations)

                else:
                    # Fit the model to the data
                    model_results = model.fit(ledisas_data, num_iterations=n_iterations, initial_mode='first_trial_no_alpha')

                model_results.to_csv(save_dir, index=False)

    # # ------------------------------------------------------------------------------------------------------------------
    # # Fit the models with sliding window
    # # ------------------------------------------------------------------------------------------------------------------
    # for i, model in enumerate(moving_window_model_list):
    #     save_dir = f'./ledisas/Model/Moving_Window/{moving_window_model_names[i]}_results.csv'
    #     # Check if the file already exists
    #     try:
    #         existing_results = pd.read_csv(save_dir)
    #         if not existing_results.empty:
    #             print(f"File {save_dir} already exists. Skipping model fitting.")
    #             continue
    #     except FileNotFoundError:
    #         pass
    #
    #     # Fit the model to the data with a sliding window
    #     model_results = moving_window_model_fitting(ledisas_data_raw, model, task='VS', id_col='SubNo',
    #                                                 num_iterations=n_iterations, window_size=window_size,
    #                                                 filter_fn=exclusionary_criteria, restart_EV=True,
    #                                                 initial_mode='fixed', initial_EV=[0.5, 0.5])
    #     model_results.to_csv(save_dir, index=False)

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
