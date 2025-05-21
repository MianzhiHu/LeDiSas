import pandas as pd
import numpy as np
from utils.VisualSearchModels import VisualSearchModels
from utils.ComputationalModeling import dict_generator, moving_window_model_fitting
from utils.DualProcess import DualProcessModel
from utils.VisualSearchModels import VisualSearchModels

# Read in the cleaned data
data_path = './LeSaS1/Data/cleaned_data.csv'
data = pd.read_csv(data_path)
data_dict = dict_generator(data, task='VS')

# get the first subject as testing data
test_data = data[data['SubNo'] == 1]
test_dict = dict_generator(test_data, task='VS')

if __name__ == "__main__":
    n_iterations = 100

    delta = VisualSearchModels('delta')
    delta_PVL = VisualSearchModels('delta_PVL_relative')
    decay = VisualSearchModels('decay')
    decay_PVL = VisualSearchModels('decay_PVL_relative')
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

    # # Fit the model to the data
    # dual_process_results = dual_process.fit(data_dict, num_iterations=n_iterations, weight_Gau='softmax', weight_Dir='softmax',
    #                                         arbi_option='Entropy', Dir_fun='Linear_Recency', Gau_fun='Naive_Recency', num_t=1)
    # delta_results = delta.fit(data_dict, num_iterations=n_iterations)
    # delta_PVL_results = delta_PVL.fit(data_dict, num_iterations=n_iterations)
    # decay_results = decay.fit(data_dict, num_iterations=n_iterations)
    # decay_PVL_results = hybrid_delta_delta.fit(data_dict, num_iterations=n_iterations)
    # WSLS_results = WSLS.fit(data_dict, num_iterations=n_iterations)
    # WSLS_delta_results = WSLS_delta.fit(data_dict, num_iterations=n_iterations)
    # WSLS_delta_weight_results = WSLS_delta_weight.fit(data_dict, num_iterations=n_iterations)
    # WSLS_decay_weight_results = WSLS_decay_weight.fit(data_dict, num_iterations=n_iterations)
    # RT_exp_basic_results = RT_exp_basic.fit(data_dict, num_iterations=n_iterations)
    # RT_delta_results = RT_delta.fit(data_dict, num_iterations=n_iterations)
    # RT_decay_results = RT_decay.fit(data_dict, num_iterations=n_iterations)
    # RT_exp_delta_results = RT_exp_delta.fit(data_dict, num_iterations=n_iterations)
    # RT_exp_decay_results = RT_exp_decay.fit(data_dict, num_iterations=n_iterations)
    # RT_delta_PVL_results = RT_delta_PVL.fit(data_dict, num_iterations=n_iterations)
    # RT_decay_PVL_results = RT_decay_PVL.fit(data_dict, num_iterations=n_iterations)
    # hybrid_delta_delta_results = hybrid_delta_delta.fit(data_dict, num_iterations=n_iterations)
    # hybrid_delta_delta_3_results = hybrid_delta_delta_3.fit(data_dict, num_iterations=n_iterations)


    # dual_process_results.to_csv('./LeSaS1/Model/dual_process_results.csv', index=False)
    # delta_results.to_csv('./LeSaS1/Model/delta_results.csv', index=False)
    # delta_PVL_results.to_csv('./LeSaS1/Model/delta_PVL_results.csv', index=False)
    # decay_results.to_csv('./LeSaS1/Model/decay_results.csv', index=False)
    # decay_PVL_results.to_csv('./LeSaS1/Model/decay_PVL_results.csv', index=False)
    # WSLS_results.to_csv('./LeSaS1/Model/WSLS_results.csv', index=False)
    # WSLS_delta_results.to_csv('./LeSaS1/Model/WSLS_delta_results.csv', index=False)
    # WSLS_delta_weight_results.to_csv('./LeSaS1/Model/WSLS_delta_weight_results.csv', index=False)
    # WSLS_decay_weight_results.to_csv('./LeSaS1/Model/WSLS_decay_weight_results.csv', index=False)\
    # RT_exp_basic_results.to_csv('./LeSaS1/Model/RT_exp_basic_results.csv', index=False)
    # RT_delta_results.to_csv('./LeSaS1/Model/RT_delta_results.csv', index=False)
    # RT_decay_results.to_csv('./LeSaS1/Model/RT_decay_results.csv', index=False)
    # RT_exp_delta_results.to_csv('./LeSaS1/Model/RT_exp_delta_results.csv', index=False)
    # RT_exp_decay_results.to_csv('./LeSaS1/Model/RT_exp_decay_results.csv', index=False)
    # RT_delta_PVL_results.to_csv('./LeSaS1/Model/RT_delta_PVL_results.csv', index=False)
    # RT_decay_PVL_results.to_csv('./LeSaS1/Model/RT_decay_PVL_results.csv', index=False)
    # hybrid_delta_delta_results.to_csv('./LeSaS1/Model/hybrid_delta_delta_results.csv', index=False)
    # hybrid_delta_delta_3_results.to_csv('./LeSaS1/Model/hybrid_delta_delta_3_results.csv', index=False)

    # # Fit block-wise model
    # model_names = ['delta', 'RT_delta', 'RT_exp_basic', 'RT_exp_delta', 'hybrid_delta_delta', 'hybrid_delta_delta_3']
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

    # # fit sliding window model
    # window_size = 10
    # hybrid_mv = moving_window_model_fitting(data, hybrid_delta_delta, task='VS', window_size=window_size,
    #                                         id_col='SubNo', num_iterations=n_iterations)
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

    # Fit block-wise sliding window model
    window_size = 10
    model_names = ['delta', 'RT_delta', 'RT_exp_basic', 'RT_exp_delta', 'hybrid_delta_delta', 'hybrid_delta_delta_3']
    for i in range(1, max(data['Block']) + 1):
        block_data = data[data['Block'] == i]
        block_dict = dict_generator(block_data, task='VS')
        for j, model in enumerate([delta, RT_delta, RT_exp_basic, RT_exp_delta, hybrid_delta_delta, hybrid_delta_delta_3]):
            save_dir = f'./LeSaS1/Model/BlockWise_Moving_Window/{model_names[j]}_block_{i}_mv_results.csv'
            # Check if the file already exists
            try:
                existing_results = pd.read_csv(save_dir)
                if not existing_results.empty:
                    print(f"File {save_dir} already exists. Skipping model fitting.")
                    continue
            except FileNotFoundError:
                pass
            model_results = moving_window_model_fitting(block_data, model, task='VS', window_size=window_size,
                                                        id_col='SubNo', num_iterations=n_iterations)
            model_results.to_csv(save_dir, index=False)
