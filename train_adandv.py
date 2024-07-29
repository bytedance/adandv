import math
import pickle
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os

MODEL_INPUT = 100
BATCH_SIZE = 2048
LR = 0.001
K = 2
ESTIMATOR_NUM = 14
EPOCHS =100
current_path = os.path.dirname(os.path.abspath(__file__))

class Ranker(nn.Module):
    def __init__(self, input_len, output_len):
        super(Ranker, self).__init__()
        self.hidden_size = 128
        self.layers = nn.Sequential(
            nn.Linear(input_len, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(), 
            nn.Linear(64, output_len)
        )
    def forward(self, x):
        x = self.layers(x)
        return x

class Weighter(nn.Module):
    def __init__(self, input_len, output_len):
        super(Weighter, self).__init__()
        self.hidden_size = 64
        self.layers = nn.Sequential(
            nn.Linear(input_len, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 64),
            nn.ReLU(), 
            nn.Linear(64, output_len),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x

class AdaNDV(nn.Module):
    """
    See Figure 3
    """
    def __init__(self, input_len, output_len, k):
        super(AdaNDV, self).__init__()
        self.ranker_over = Ranker(input_len, output_len)
        self.ranker_under = Ranker(input_len, output_len)
        self.weighter = Weighter(int(input_len + 2 * k), int(2 * k))
        self.k = k
        
    def forward(self, x, estimated_logd):
        score_over, score_under, logd = self.run(x, estimated_logd)
        return score_over, score_under, logd

    def run(self, x, estimated_logd):
        score_over = self.ranker_over(x)
        _, over_idxs = torch.topk(score_over, self.k, dim=1, largest=True, sorted=True)
        
        score_under = self.ranker_under(x)
        _, under_idxs = torch.topk(score_under, self.k, dim=1, largest=True, sorted=True)
        
        over_estimate = estimated_logd[torch.arange(estimated_logd.shape[0]).unsqueeze(1), over_idxs]
        under_estimate = estimated_logd[torch.arange(estimated_logd.shape[0]).unsqueeze(1), under_idxs]
        estimate = torch.concat([over_estimate, under_estimate], dim=-1)
        
        x_prime = torch.concat([x, estimate], dim=-1)
        weights = self.weighter(x_prime)
        
        logd = torch.sum(estimate * weights, dim=-1).squeeze(-1)
        # logd = torch.prod(torch.pow(estimate, weights), dim=-1) # ablation of log
        return score_over, score_under, logd
    
    def inference(self, x, estimated_logd):
        score_over, score_under, logd = self.run(x, estimated_logd)
        return logd


class NDVDataset(Dataset):
    def __init__(self, data_profile, rank_label, esimate_ndv, D_list):
        self.data_profile = data_profile
        self.rank_label = rank_label
        self.esimate_ndv = esimate_ndv
        self.D_list = D_list
    def __getitem__(self, index):
        return torch.tensor(self.data_profile[index], dtype = torch.float32), \
            torch.tensor(self.rank_label[index], dtype = torch.float32), \
                torch.tensor(self.esimate_ndv[index], dtype = torch.float32), \
                    torch.tensor(self.D_list[index], dtype = torch.float32),
    def __len__(self):
        return len(self.data_profile)


def ranking_loss(y_pred, y_true):
    eps = 1e-5
    alpha = 1
    device = y_pred.device
    y_pred_sorted, indices_pred = y_pred.sort(descending=True, dim=-1)
    y_true_sorted, _ = y_true.sort(descending=True, dim=-1)
    true_sorted_by_preds = torch.gather(y_true, dim=1, index=indices_pred)
    true_diffs = true_sorted_by_preds[:, :, None] - true_sorted_by_preds[:, None, :]
    padded_pairs_mask = torch.isfinite(true_diffs)
    padded_pairs_mask.diagonal(dim1=-2, dim2=-1).zero_()
    true_sorted_by_preds.clamp_(min=0.)
    y_true_sorted.clamp_(min=0.)
    pos_idxs = torch.arange(1, y_pred.shape[1] + 1).to(device)
    D = torch.log2(1. + pos_idxs.float())[None, :]
    maxDCGs = torch.sum((torch.pow(2, y_true_sorted) - 1) / D, dim=-1).clamp(min=eps)
    G = (torch.pow(2, true_sorted_by_preds) - 1) / maxDCGs[:, None]
    scores_diffs = (y_pred_sorted[:, :, None] - y_pred_sorted[:, None, :])
    scores_diffs[~padded_pairs_mask] = 0.
    approx_pos = 1. + torch.sum(padded_pairs_mask.float() * (torch.sigmoid(-alpha * scores_diffs).clamp(min=eps)), dim=-1)
    approx_D = torch.log2(1. + approx_pos)
    approx_DCG = torch.sum((G / approx_D), dim=-1)
    return -torch.mean(approx_DCG)


def loss_function(y_pred_pos, y_true_pos, y_pred_neg, y_true_neg, log_d_hat, log_d):
    """
    See Section 3.1 Equation (5)
    """
    loss_pos = ranking_loss(y_pred_pos, y_true_pos)
    loss_neg = ranking_loss(y_pred_neg, y_true_neg)
    loss_reg = F.mse_loss(log_d_hat, log_d, reduction='mean')
    beta = 0.5
    loss = loss_pos + loss_neg + beta * loss_reg
    return loss

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

def get_data_loader():
    print(f'Loading data...')
    with open(os.path.join(current_path, 'data/train.pkl'), 'rb') as f:
        data = pickle.load(f)
    data_profile, rank_label, esimate_ndv, D_list = data
    train_dataset = NDVDataset(data_profile, rank_label, esimate_ndv, D_list)

    with open(os.path.join(current_path, 'data/test.pkl'), 'rb') as f:
        data = pickle.load(f)
    data_profile, rank_label, esimate_ndv, D_list = data
    test_dataset = NDVDataset(data_profile, rank_label, esimate_ndv, D_list)

    with open(os.path.join(current_path, 'data/val.pkl'), 'rb') as f:
        data = pickle.load(f)
    data_profile, rank_label, esimate_ndv, D_list = data
    validation_dataset = NDVDataset(data_profile, rank_label, esimate_ndv, D_list)

    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE)
    validation_loader = DataLoader(validation_dataset, batch_size = BATCH_SIZE)
    print(f'Data loaded')
    return train_loader, test_loader, validation_loader

def evaluate(model, loader, device, dataset_name):
    model.eval()
    predicted_q_error = []
    for data_profile, _, esimate_ndv, D_list in loader:
        data_profile = data_profile.to(device)
        esimate_ndv = esimate_ndv.to(device)
        model = model.to(device)
        logd = model.inference(data_profile, esimate_ndv)
        logd = logd.exp().cpu().detach().numpy().tolist()
        D_list = D_list.cpu().detach().numpy().tolist()
        estimated_q_error = [compute_error(logd[i], D_list[i]) for i in range(len(D_list))]
        predicted_q_error.extend(estimated_q_error)
    print(f'q-error distribution of {dataset_name} dataset')
    print(f'mean: {np.mean(predicted_q_error):.2f}, 50%: {np.percentile(predicted_q_error, 50):.2f} 75%: {np.percentile(predicted_q_error, 75):.2f} 90%: {np.percentile(predicted_q_error, 90):.2f} 95%: {np.percentile(predicted_q_error, 95):.2f} 99%: {np.percentile(predicted_q_error, 99):.2f}')
    return predicted_q_error


def train_model():
    model = AdaNDV(MODEL_INPUT, ESTIMATOR_NUM, K)
    model_path = os.path.join(current_path, 'adandv.pth')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_loader, test_loader, validation_loader = get_data_loader()
    for epoch in range(EPOCHS):
        for data_profile, label, esimate_ndv, D_list in train_loader:
            model = model.to(device)
            data_profile = data_profile.to(device)
            label = label.to(device)
            esimate_ndv = esimate_ndv.to(device)
            D_list = D_list.to(device)
            optimizer.zero_grad()
            
            score_over, score_under, logd = model(data_profile, esimate_ndv)
            loss = loss_function(score_over, label[:,0,:].squeeze(1),
                                score_under, label[:,1,:].squeeze(1),
                                logd, torch.log(D_list))
            loss.backward()
            optimizer.step()
        print(f'[{epoch+1}/{EPOCHS}]', '-'*50)
        predicted_q_error_train = evaluate(model, train_loader, device, 'train')
        predicted_q_error_val = evaluate(model, validation_loader, device, 'validation')
        predicted_q_error_test = evaluate(model, test_loader, device, 'test')

        if epoch == 0:
            best_metric = np.mean(predicted_q_error_val)
        save_metric = np.mean(predicted_q_error_val)
        if save_metric <= best_metric:
            best_metric = save_metric
            torch.save(model.cpu().state_dict(), model_path)
            print(f'model saved')
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print('='*50)
    print('the performance of the saved model')
    print('='*50)
    evaluate(model, train_loader, device, 'train')
    evaluate(model, validation_loader, device, 'val')
    evaluate(model, test_loader, device, 'test')

if __name__ == '__main__':
    train_model()
    
    