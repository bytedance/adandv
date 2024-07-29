import math
import pickle
import numpy as np
import os

current_path = os.path.dirname(os.path.abspath(__file__))

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

def show_error(predicted_q_error, method):
    print(method)
    print(f'mean: {np.mean(predicted_q_error):.2f}, 50%: {np.percentile(predicted_q_error, 50):.2f} 75%: {np.percentile(predicted_q_error, 75):.2f} 90%: {np.percentile(predicted_q_error, 90):.2f} 95%: {np.percentile(predicted_q_error, 95):.2f} 99%: {np.percentile(predicted_q_error, 99):.2f}')

def show_error_distribution(esimate_ndv, D_list):
    estimator_list = ['EB', 'GEE', 'Chao', 'Shlosser', 'ChaoLee', 'Goodman', 'Jackknife', 'Sichel', 'Method of Movement', 'Bootstrap', 'Horvitz Thompson', 'Method of Movement v2', 'Method of Movement v3', 'Smoothed Jackknife']
    esimate_ndv = np.array(esimate_ndv)
    print(f'The q-error distribution of the traditional estimators:')
    for i, estimator in enumerate(estimator_list):
        estimate_result = esimate_ndv[:,i]
        estimate_result = np.exp(estimate_result)
        q_error = []
        for i in range(len(D_list)):
            q_error.append(compute_error(estimate_result[i], D_list[i]))
        show_error(q_error, estimator)

def main():
    with open(os.path.join(current_path, 'data/test.pkl'), 'rb') as f:
        data = pickle.load(f)
    _, _, esimate_ndv, D_list = data
    show_error_distribution(esimate_ndv, D_list)


if __name__ == '__main__':
    main()