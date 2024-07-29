import math
import os
import pickle
import json
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import concurrent.futures
from collections import Counter

from tqdm import tqdm


# since we cannot add data and code of existing estimators, you need implement this script.

class NDVEstimator:
    # ['EB', 'GEE', 'Chao', 'Shlosser', 'ChaoLee', 'Goodman', 'Jackknife', 'Sichel', 'Method of Movement', 'Bootstrap', 'Horvitz Thompson', 'Method of Movement v2', 'Method of Movement v3', 'Smoothed Jackknife']
    def __init__(self, ):
        pass

    def estimator(self, n, N, profile, method):
        # implement it
        pass



ESTIMATOR_NUM = 14
current_path = os.path.dirname(os.path.abspath(__file__))
parquet_data_path = os.path.join(current_path, '../tablib-sample2') # data path
file_list = os.listdir(parquet_data_path) # all parquet files
# We ignore three files ('2d7d54b8', '8e1450ee', 'dc0e820c') for the memory issue
file_list = sorted(file_list) # fix the orders
file_list = list(set(file_list) - set(['2d7d54b8.parquet', '8e1450ee.parquet', 'dc0e820c.parquet']))
train_size = int(len(file_list) * 0.6)
test_size = int(len(file_list) * 0.2)
val_size = len(file_list) - train_size - test_size
train_files = file_list[:train_size]
test_files = file_list[train_size:train_size+test_size] 
val_files = file_list[-val_size:]
# the splits used in our paper are
# train_files = ['010c9cd9.parquet', '02dd2fd6.parquet', '08978e98.parquet', '0ad462e9.parquet', '0bb72755.parquet', '0c0841de.parquet', '0d3abe7a.parquet', '0fc73142.parquet', '104905f9.parquet', '1270ac8d.parquet', '13eba86c.parquet', '14980933.parquet', '1ba3f108.parquet', '1cddabe3.parquet', '1cff8001.parquet', '1e0552fb.parquet', '1e23438e.parquet', '24d9ed5d.parquet', '26919382.parquet', '2950d786.parquet', '2958de30.parquet', '2a417caa.parquet', '331b23f0.parquet', '3fc98861.parquet', '460f7423.parquet', '540a9e03.parquet', '573c792c.parquet', '59b8aef7.parquet', '5a16a1d1.parquet', '5c31885f.parquet', '60251ea2.parquet', '66132eca.parquet', '6843f9cf.parquet', '6b7a4a6a.parquet', '6b7a9919.parquet', '6d6e40af.parquet', '7876bbf3.parquet', '80da369f.parquet', '812ce757.parquet', '82b50980.parquet', '8908afcd.parquet', '8a8cb109.parquet', '8d4ea5d2.parquet', '8d69756e.parquet']
# test_files = ['8dd3234a.parquet', '906bbbf9.parquet', '94e4322f.parquet', '95173c5e.parquet', '973d7781.parquet', '99adcdcf.parquet', '9ccce5d2.parquet', 'a50f776c.parquet', 'a86091d3.parquet', 'a90e38f6.parquet', 'b1f3e176.parquet', 'b61ad443.parquet', 'b84e97c1.parquet', 'b87d128a.parquet']
# val_files = ['bbc47e69.parquet', 'bf2681fa.parquet', 'c2ae9ccf.parquet', 'cb2c7cf9.parquet', 'cb79572b.parquet', 'cc7c50e1.parquet', 'cf25ecd3.parquet', 'd29a35e5.parquet', 'd2e53a9b.parquet', 'd40966f7.parquet', 'e3cd0092.parquet', 'e4edf5b1.parquet', 'ea5c6e93.parquet', 'ebfc960d.parquet', 'f1776069.parquet', 'f9a7562e.parquet']

def compute_error(estimated, ground_truth):
    # set a large error for the inf estimate result
    if math.isinf(estimated) or estimated == 0 or math.isnan(estimated):
        err = 1e10
        return err
    assert estimated > 0 and ground_truth > 0, f"estimated and ground_truth NDV must be positive. {estimated}, {ground_truth}"
    err =  max(estimated, ground_truth) / min(estimated, ground_truth)
    if math.isinf(err):
        err = 1e10
    return err

def build_column_profile(data):
    value_counts = Counter(data)
    data_len = len(data)
    freq = [0] * (data_len)
    for value, count in value_counts.items():
        freq[count-1] += 1
    return freq

def load_parquet_file(file_path):
    assert file_path.endswith('.parquet')
    df = pd.read_parquet(file_path)
    tables = [pa.RecordBatchStreamReader(b).read_all() for b in df['arrow_bytes']]
    return tables

def sampling_data(tables, sample_rate_percent=1):
    save_data = []
    data_point = 0
    for j, pyarrow_table in tqdm(enumerate(tables)):
        table_data = []
        col_num = len(pyarrow_table.column_names)
        for i in range(col_num):
            col_data = pyarrow_table.column(i)
            try:
                col = pa.array(col_data).drop_null().to_pylist() 
                N = len(col)
                D = len(set(col))
                if N < 10000:
                    continue
                sample_num = round(N * sample_rate_percent / 100)
                sample_data = np.random.choice(col, sample_num)
                data_profile = build_column_profile(sample_data)
                col_data_point = (data_profile, N, D)
                table_data.append(col_data_point)
                data_point += 1
            except Exception as e:
                print('ERROR', e)
        save_data.append(table_data)
    return save_data



def read_and_construct(file_list, exist_datapoint):
    data_profile = []
    N_list = []
    D_list = []
    for file in file_list:
        file_path = os.path.join(parquet_data_path, file)
        tables = load_parquet_file(file_path)
        sample_content = sampling_data(tables)
        for tab_content in sample_content:
            if len(tab_content) == 0:
                continue
            for col_content in tab_content:
                unique_identifier = tuple([tuple(col_content[0]), col_content[1], col_content[2]])
                # ! ensure no data leak
                if unique_identifier in exist_datapoint:
                    continue
                data_profile.append(col_content[0])  
                N_list.append(col_content[1])
                D_list.append(col_content[2])
                exist_datapoint.add(unique_identifier)
    return data_profile, N_list, D_list, exist_datapoint

def estimate_ndv(data_profile, N_list, D_list):
    # ! the order of estimators should be fixed
    assert len(data_profile) == len(N_list)
    assert len(data_profile) == len(D_list)
    estimator_list = ['EB', 'GEE', 'Chao', 'Shlosser', 'ChaoLee', 'Goodman', 'Jackknife', 'Sichel', 'Method of Movement', 'Bootstrap', 'Horvitz Thompson', 'Method of Movement v2', 'Method of Movement v3', 'Smoothed Jackknife']

    ndv_estimator = NDVEstimator() # ! implemnt this class
    q_error_list = []
    estimate_ndv_list = []
    for i in tqdm(range(len(data_profile))):
        q_errors = []
        estimate_ndvs = []
        for method in estimator_list:
            d = float(ndv_estimator.estimator(len(data_profile[i]), N_list[i], data_profile[i], method))
            if d <= 0 or math.isinf(d):
                print(f'{method} has invalid value')
                d = 1
            q_error = compute_error(d, D_list[i])
            q_errors.append(q_error)
            estimate_ndvs.append(d)
        q_error_list.append(q_errors)
        estimate_ndv_list.append(estimate_ndvs)

    return q_error_list, estimate_ndv_list


def construct_ranking_label(esimate_ndv_log, threshold_D_log, pos):
    groundtruth = [0] * ESTIMATOR_NUM
    groundtruth = np.array(groundtruth)
    esimate_ndv_log_copy = np.array(esimate_ndv_log)
    if pos:
        indexes = np.where(esimate_ndv_log < threshold_D_log)[0]
        esimate_ndv_log_copy[indexes] += np.max(esimate_ndv_log_copy)
        idx = np.argsort(esimate_ndv_log_copy)
        for j,k in enumerate(idx):
            groundtruth[k]=ESTIMATOR_NUM - j - 1
    else:
        indexes = np.where(esimate_ndv_log > threshold_D_log)[0]
        esimate_ndv_log_copy[indexes] -= np.max(esimate_ndv_log_copy)
        idx = np.argsort(esimate_ndv_log_copy * -1)
        for j,k in enumerate(idx):
            groundtruth[k]=ESTIMATOR_NUM - j - 1
    groundtruth[indexes] = 0
    
    return groundtruth.tolist()


def save_process_data(file_list, name, exist_datapoint):
    data_profile, N_list, D_list, datapoint_set = read_and_construct(file_list, exist_datapoint)

    q_error_list, estimate_ndv_list = estimate_ndv(data_profile, N_list, D_list)

    rank_label = [
        [
            construct_ranking_label(estimate_ndv_list[i], D_list[i], True),
            construct_ranking_label(estimate_ndv_list[i], D_list[i], False)
        ] 
        for i in range(len(data_profile))
    ]

    # cut-off data_profile and add logn, logd, and logN
    # ! Implement it.

    save_list = [data_profile, rank_label, estimate_ndv_list, D_list]

    with open(f'data/{name}.pkl', 'wb') as f:
        pickle.dump(save_list, f)
    
    # ensure no data leak
    return datapoint_set 
    

if __name__ == '__main__':
    train_set = save_process_data(train_files[0], 'train', set())
    val_set = save_process_data(val_files, 'val', train_set)
    save_process_data(test_files, 'test', train_set.union(val_set))