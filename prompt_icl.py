from tqdm import tqdm
import numpy as np
import datetime
import pickle
import torch
import os


import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch
import math
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_filename = os.path.basename(__file__)[:-3]
print(f"Running experiment: {script_filename}")

ITN = 100_000
ITN_EVAL = 5000
ITN_PRINT = 1000
lr = 0.001
bsize = 3072
d = 10

prompt_lengths = [1]
icl_sample_counts = [1, 10, 20, 30, 40, 50]
num_runs = 50
tasks = ["prompt and heads"] #, "none", "prompt", "heads"]


def data_generator(bsize, d, n, num_clusters, means, covariances, noise_levels, mixture_weights=None):
    if mixture_weights is None:
        mixture_weights = torch.tensor([1.0 / num_clusters for _ in range(num_clusters)], dtype=torch.float, device=device)
    else:
        mixture_weights = torch.tensor(mixture_weights, dtype=torch.float, device=device)

    mixture_weights = mixture_weights / torch.sum(mixture_weights)

    X = torch.randn(bsize, n, d, device=device)
    Xt = torch.randn(bsize, 1, d, device=device)

    cluster_assignments = torch.multinomial(mixture_weights, bsize, replacement=True)

    covariances = torch.stack(covariances)
    means = torch.stack(means)

    selected_covariances = covariances[cluster_assignments]
    selected_means = means[cluster_assignments]

    cholesky_decompositions = torch.linalg.cholesky(selected_covariances)

    random_vectors = torch.randn(bsize, d, 1, device=device)
    beta = torch.bmm(cholesky_decompositions, random_vectors) + selected_means.unsqueeze(-1)

    noise_levels = torch.tensor(noise_levels, device=device)
    selected_noise_levels = noise_levels[cluster_assignments].unsqueeze(1).unsqueeze(2)
    noise = torch.randn(bsize, n, 1, device=device) * selected_noise_levels
    noise_t = torch.randn(bsize, 1, 1, device=device) * selected_noise_levels

    Y = torch.bmm(X, beta) + noise
    Yt = torch.bmm(Xt, beta) + noise_t

    return torch.cat([X, Y], dim=-1), torch.cat([Xt, torch.zeros_like(Yt, device=device)], dim=-1), Yt[:,:,0]


class LinearAttn(nn.Module):
    def __init__(self, input_size, hidden_size=None, prompt_size=10, num_tasks=1):
        super(LinearAttn, self).__init__()
        self.dim = input_size
        self.num_tasks = num_tasks
        self.prompt_size = prompt_size

        if hidden_size is None:
            hidden_size = input_size

        self.prompts = nn.Parameter(torch.randn(num_tasks, prompt_size, input_size, device=device))

        self.query = nn.Linear(input_size, hidden_size, bias=False)
        self.key = nn.Linear(input_size, hidden_size, bias=False)
        self.value = nn.Linear(input_size, input_size, bias=False)
        self.v = torch.zeros(1, input_size, device=device)
        self.v[0,-1] = 1

        self.v_task = nn.ModuleList([
            nn.Linear(input_size, 1, bias=False)
            for _ in range(num_tasks)
        ])

    def forward(self, input_seq, task_id=-1, cross_input=None, attn_idx=-1, cross_attn=False, task_type="prompt"):
        n, T, d = input_seq.shape

        if task_type in ["prompt", "prompt and heads"]:
            prompt = self.prompts[task_id].expand(n, -1, -1)
            input_seq = torch.cat([prompt, input_seq], dim=1)

        if not cross_attn:
            cross_input = input_seq[:, attn_idx]

        Q = self.query(cross_input)
        K = self.key(input_seq)
        out = Q @ K.transpose(-2, -1) / math.sqrt(self.dim)
        out = out @ self.value(input_seq)

        if task_type in ["heads", "prompt and heads"]:
            return (self.v_task[task_id](out))[:, :, 0]
        else:
            return F.linear(out, self.v, None)[:, :, 0]


class OneStepGD(nn.Module):
    def __init__(self, input_size, hidden_size=None, prompt_size=1, num_tasks=1, device='cpu'):
        super(OneStepGD, self).__init__()
        self.dim = input_size
        self.num_tasks = num_tasks
        self.prompt_size = prompt_size
        self.device = device

        if hidden_size is None:
            hidden_size = input_size - 1  # Adjusted to match the dimension of X

        self.prompts = nn.Parameter(torch.randn(num_tasks, prompt_size, input_size, device=device) * 0.1)
        self.W = nn.Parameter(torch.randn(input_size - 1, input_size - 1, device=device) * 0.1)

    def forward(self, input_seq, task_id=0, cross_input=None, attn_idx=-1, cross_attn=False, task_type="prompt"):
        n = input_seq.size(0)  # Batch size

        if task_type in ["prompt"]:
            prompt = self.prompts[task_id].expand(n, -1, -1)
            input_seq = torch.cat([prompt, input_seq], dim=1)

        x = cross_input[:,0,:-1]           # Shape: (batch_size, input_size - 1)
        X = input_seq[:, :, :-1]      # Shape: (batch_size, seq_len, input_size - 1)
        y = input_seq[:, :, -1:]       # Shape: (batch_size, seq_len, 1)

        X_T = X.transpose(1, 2)         # Shape: (batch_size, input_size - 1, seq_len - 1)
        XTy = torch.bmm(X_T, y)    # Shape: (batch_size, input_size - 1, 1)

        x_T = x.unsqueeze(1)            # Shape: (batch_size, 1, input_size - 1)
        W_expanded = self.W.unsqueeze(0).expand(n, -1, -1)  # Shape: (batch_size, input_size - 1, input_size - 1)

        xTW = torch.bmm(x_T, W_expanded)  # Shape: (batch_size, 1, input_size - 1)

        result = torch.bmm(xTW, XTy)      # Shape: (batch_size, 1, 1)
        result = result.squeeze(-1) # Shape: (batch_size, 1)

        return result


class LinearSelfAttentionLayer(nn.Module):
    def __init__(self, input_size):
        super(LinearSelfAttentionLayer, self).__init__()
        self.query = nn.Linear(input_size, input_size, bias=False)
        self.key = nn.Linear(input_size, input_size, bias=False)
        self.value = nn.Linear(input_size, input_size, bias=False)

    def forward(self, input_seq, seq_size):
        Q = self.query(input_seq)  # (batch_size, seq_len, input_size)
        K = self.key(input_seq)    # (batch_size, seq_len, input_size)
        V = self.value(input_seq)  # (batch_size, seq_len, input_size)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1)) # (batch_size, seq_len, seq_len)
        out = torch.matmul(scores, V)  # (batch_size, seq_len, input_size)
        return out + input_seq # * (1/seq_size)


class MultiLayerLinearAttn(nn.Module):
    def __init__(self, input_size, prompt_size=10, num_tasks=1, L=2, device='cpu'):
        super(MultiLayerLinearAttn, self).__init__()
        self.dim = input_size
        self.num_tasks = num_tasks
        self.prompt_size = prompt_size
        self.device = device
        self.L = L

        self.prompts = nn.Parameter(torch.randn(num_tasks, prompt_size, self.dim, device=device))

        self.layers = nn.ModuleList([
            LinearSelfAttentionLayer(self.dim)
            for _ in range(L)
        ])

        self.v_task = nn.ModuleList([
            nn.Linear(self.dim, 1, bias=False)
            for _ in range(num_tasks)
        ])

        with torch.no_grad():
            for layer in self.layers:
                layer.query.weight.mul_((1 / self.dim))
                layer.key.weight.mul_((1 / self.dim))
                layer.value.weight.mul_((1 / self.dim))


    def forward(self, input_seq, task_id=-1, cross_input=None, attn_idx=-1, cross_attn=False, task_type="prompt"):
        n, T, d = input_seq.shape

        if task_type in ["prompt", "prompt and heads"]:
            prompt = self.prompts[task_id].expand(n, -1, -1)  # (batch_size, prompt_size, self.dim)
            input_seq = torch.cat([prompt, input_seq], dim=1)  # (batch_size, seq_len + prompt_size, self.dim)

        input_seq = torch.cat([input_seq, cross_input], dim=1)  # (batch_size, total_seq_len + 1, self.dim)

        for layer in self.layers:
            input_seq = input_seq + layer(input_seq, T)  # (batch_size, total_seq_len + 1, self.dim)

        last_vector = input_seq[:, -1, :]  # (batch_size, self.dim)

        if task_type in ["heads", "prompt and heads"]:
            output = self.v_task[task_id](last_vector)  # (batch_size, 1)
        else:
            output = last_vector[:, -1:]  # (batch_size, 1)

        return output


def evaluation(model, bsize, d, n, means, covariances, noise_levels, mixture_weights, task_type, ITN_EVAL=10):
    model.eval()
    test_loss = 0
    num_clusters = len(means)

    for _ in range(ITN_EVAL):
        if model.prompt_size == 0:
            data, Xt, Yt = data_generator(bsize, d, n, num_clusters, means, covariances, noise_levels, mixture_weights)
            with torch.no_grad():
                out = model(data, task_id=-1, cross_input=Xt, cross_attn=True, task_type=task_type)
                loss = torch.square(out[:, -1] - Yt[:, -1]).mean(dim=0).detach().cpu().numpy()
                test_loss += loss / ITN_EVAL
        else:
            weighted_loss = 0
            for task_id in range(num_clusters):
                data, Xt, Yt = data_generator(bsize, d, n, 1, [means[task_id]], [covariances[task_id]], [noise_levels[task_id]], mixture_weights=[1.0])
                with torch.no_grad():
                    out = model(data, task_id=task_id, cross_input=Xt, cross_attn=True, task_type=task_type)
                    loss = torch.square(out[:, -1] - Yt[:, -1]).mean(dim=0)
                    weighted_loss += mixture_weights[task_id] * loss
            test_loss += weighted_loss.item() / ITN_EVAL

    return test_loss


def joint_training(model, bsize, d, n, means, covariances, noise_levels, mixture_weights, task_type):
    model.to(device, non_blocking=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    losses = []
    progress_bar = tqdm(range(ITN), desc="Joint Training", leave=False)

    for it in progress_bar:
        total_loss = 0
        weighted_loss = 0
        for task_id in range(len(means)):
            data, Xt, Yt = data_generator(bsize, d, n, 1, [means[task_id]], [covariances[task_id]], [noise_levels[task_id]], mixture_weights=[1.0])
            optimizer.zero_grad()
            out = model(data, task_id=task_id, cross_input=Xt, cross_attn=True, task_type=task_type)
            loss = torch.square(out[:, -1] - Yt[:, -1]).mean()
            weighted_loss += mixture_weights[task_id] * loss

        weighted_loss.backward()
        optimizer.step()
        total_loss = weighted_loss.item()
        losses.append(total_loss)

        if it % ITN_PRINT == 0 or it == ITN - 1:
            avg_loss = np.mean(losses)
            min_loss = np.min(losses)
            progress_bar.set_postfix(Loss=total_loss, Avg_Loss=avg_loss, Min_Loss=min_loss)


def pretrain_fine_tuning(model, bsize, d, n, means, covariances, noise_levels, mixture_weights, task_type):
    model.to(device)

    # Pretraining Phase
    optimizer = torch.optim.Adam(
        [param for name, param in model.named_parameters() if "prompts" not in name and "v_task" not in name],
        lr=lr, weight_decay=1e-3
    )
    pretrain_losses = []
    pretrain_bar = tqdm(range(ITN // 2), desc="Pretraining", leave=False)

    for it in pretrain_bar:
        data, Xt, Yt = data_generator(bsize, d, n, len(means), means, covariances, noise_levels, mixture_weights)
        optimizer.zero_grad()
        out = model(data, task_id=-1, cross_input=Xt, cross_attn=True, task_type="none")
        loss = torch.square(out[:, -1] - Yt[:, -1]).mean()
        loss.backward()
        optimizer.step()
        pretrain_losses.append(loss.item())

        # Update progress bar with loss metrics
        if it % ITN_PRINT == 0 or it == ITN // 2 - 1:
            avg_loss = np.mean(pretrain_losses)
            min_loss = np.min(pretrain_losses)
            pretrain_bar.set_postfix(Loss=loss.item(), Avg_Loss=avg_loss, Min_Loss=min_loss)

    # Fine-tuning Phase
    finetune_losses = []
    for param in model.parameters():
        param.requires_grad = False
    if task_type in ["heads", "prompt and heads"]:
        for param in model.v_task.parameters():
            param.requires_grad = True
    if task_type in ["prompt", "prompt and heads"]:
        model.prompts.requires_grad = True

    optimizer = torch.optim.Adam(
        [param for param in model.parameters() if param.requires_grad],
        lr=lr, weight_decay=1e-3
    )
    finetune_bar = tqdm(range(ITN // 2), desc="Fine-tuning", leave=False)

    for it in finetune_bar:
        total_loss = 0
        for task_id in range(len(means)):
            data, Xt, Yt = data_generator(bsize, d, n, 1, [means[task_id]], [covariances[task_id]], [noise_levels[task_id]], mixture_weights=[1.0])
            optimizer.zero_grad()
            out = model(data, task_id=task_id, cross_input=Xt, cross_attn=True, task_type=task_type)
            loss = torch.square(out[:, -1] - Yt[:, -1]).mean()
            loss.backward()
            optimizer.step()
            finetune_losses.append(loss.item())
            total_loss += mixture_weights[task_id] * loss

        # Update progress bar with loss metrics
        if it % ITN_PRINT == 0 or it == ITN // 2 - 1:
            avg_loss = np.mean(finetune_losses)
            min_loss = np.min(finetune_losses)
            finetune_bar.set_postfix(Loss=total_loss.item(), Avg_Loss=avg_loss, Min_Loss=min_loss)

# Define the experiments with all their parameters (including previously common parameters)
experiments = [
    # {
    #     'name': 'zero_cov_11',
    #     'description': 'Mean (1.4)(0.3) + (-0.6)(0.7) = 0 and Cov 1-1',
    #     'means': [1.4, -0.6],
    #     'cov_matrices': [1.0, 1.0],
    #     'noise_levels': [0, 0],
    #     'mixture_weights': [0.3, 0.7],
    # },
    # {
    #     'name': 'zero_cov_11_different_means',
    #     'description': 'Mean (2.1)(0.3) + (-0.9)(0.7) = 0 and Cov 1-1 -- Diff Means',
    #     'means': [2.1, -0.9],
    #     'cov_matrices': [1.0, 1.0],
    #     'noise_levels': [0, 0],
    #     'mixture_weights': [0.3, 0.7],
    # },
    # {
    #     'name': 'zero_cov_22',
    #     'description': 'Mean (1.4)(0.3) + (-0.6)(0.7) = 0 and Cov 2-2',
    #     'means': [1.4, -0.6],
    #     'cov_matrices': [2.0, 2.0],
    #     'noise_levels': [0, 0],
    #     'mixture_weights': [0.3, 0.7],
    # },
    # {
    #     'name': 'zero_cov_12',
    #     'description': 'Mean (1.4)(0.3) + (-0.6)(0.7) = 0 and Cov 1-2',
    #     'means': [1.4, -0.6],
    #     'cov_matrices': [1.0, 2.0],
    #     'noise_levels': [0, 0],
    #     'mixture_weights': [0.3, 0.7],
    # },
    # {
    #     'name': 'nonzero_cov_11',
    #     'description': 'Mean (1)(0.3) + (-1)(0.7) = -0.4 and Cov 1-1',
    #     'means': [1.0, -1.0],
    #     'cov_matrices': [1.0, 1.0],
    #     'noise_levels': [0, 0],
    #     'mixture_weights': [0.3, 0.7],
    # },
    # {
    #     'name': 'nonzero_cov_11_different_means',
    #     'description': 'Mean (1.7)(0.3) + (-1.3)(0.7) = -0.4 and Cov 1-1 -- Diff Means',
    #     'means': [1.7, -1.3],
    #     'cov_matrices': [1.0, 1.0],
    #     'noise_levels': [0, 0],
    #     'mixture_weights': [0.3, 0.7],
    # },
    # {
    #     'name': 'nonzero_cov_22',
    #     'description': 'Mean (1)(0.3) + (-1)(0.7) = -0.4 and Cov 2-2',
    #     'means': [1.0, -1.0],
    #     'cov_matrices': [2.0, 2.0],
    #     'noise_levels': [0, 0],
    #     'mixture_weights': [0.3, 0.7],
    # },
    # {
    #     'name': 'nonzero_cov_12',
    #     'description': 'Mean (1)(0.3) + (-1)(0.7) = -0.4 and Cov 1-2',
    #     'means': [1.0, -1.0],
    #     'cov_matrices': [1.0, 2.0],
    #     'noise_levels': [0, 0],
    #     'mixture_weights': [0.3, 0.7],
    # },
# ]

# experiments = [
# {'name': 'single_task_mu_ratio_0.1_layers_1',
#   'description': 'Single-task experiment with mu_ratio 0.1, 1 layers, dimension 10.',
#   'means': [1],
#   'cov_matrices': [9.0],
#   'noise_levels': [0],
#   'mixture_weights': [1.0],
#   'layer_num': 1,
#   'mu_ratio': 0.1,
#   'dimension': 10},
#  {'name': 'single_task_mu_ratio_0.1_layers_2',
#   'description': 'Single-task experiment with mu_ratio 0.1, 2 layers, dimension 10.',
#   'means': [1],
#   'cov_matrices': [9.0],
#   'noise_levels': [0],
#   'mixture_weights': [1.0],
#   'layer_num': 2,
#   'mu_ratio': 0.1,
#   'dimension': 10},
#  {'name': 'single_task_mu_ratio_0.1_layers_3',
#   'description': 'Single-task experiment with mu_ratio 0.1, 3 layers, dimension 10.',
#   'means': [1],
#   'cov_matrices': [9.0],
#   'noise_levels': [0],
#   'mixture_weights': [1.0],
#   'layer_num': 3,
#   'mu_ratio': 0.1,
#   'dimension': 10},
#  {'name': 'single_task_mu_ratio_0.5_layers_1',
#   'description': 'Single-task experiment with mu_ratio 0.5, 1 layers, dimension 10.',
#   'means': [1],
#   'cov_matrices': [1.0],
#   'noise_levels': [0],
#   'mixture_weights': [1.0],
#   'layer_num': 1,
#   'mu_ratio': 0.5,
#   'dimension': 10},
#  {'name': 'single_task_mu_ratio_0.5_layers_2',
#   'description': 'Single-task experiment with mu_ratio 0.5, 2 layers, dimension 10.',
#   'means': [1],
#   'cov_matrices': [1.0],
#   'noise_levels': [0],
#   'mixture_weights': [1.0],
#   'layer_num': 2,
#   'mu_ratio': 0.5,
#   'dimension': 10},
#  {'name': 'single_task_mu_ratio_0.5_layers_3',
#   'description': 'Single-task experiment with mu_ratio 0.5, 3 layers, dimension 10.',
#   'means': [1],
#   'cov_matrices': [1.0],
#   'noise_levels': [0],
#   'mixture_weights': [1.0],
#   'layer_num': 3,
#   'mu_ratio': 0.5,
#   'dimension': 10},
#  {'name': 'single_task_mu_ratio_0.9_layers_1',
#   'description': 'Single-task experiment with mu_ratio 0.9, 1 layers, dimension 10.',
#   'means': [1],
#   'cov_matrices': [0.11111111111111108],
#   'noise_levels': [0],
#   'mixture_weights': [1.0],
#   'layer_num': 1,
#   'mu_ratio': 0.9,
#   'dimension': 10},
#  {'name': 'single_task_mu_ratio_0.9_layers_2',
#   'description': 'Single-task experiment with mu_ratio 0.9, 2 layers, dimension 10.',
#   'means': [1],
#   'cov_matrices': [0.11111111111111108],
#   'noise_levels': [0],
#   'mixture_weights': [1.0],
#   'layer_num': 2,
#   'mu_ratio': 0.9,
#   'dimension': 10},
#  {'name': 'single_task_mu_ratio_0.9_layers_3',
#   'description': 'Single-task experiment with mu_ratio 0.9, 3 layers, dimension 10.',
#   'means': [1],
#   'cov_matrices': [0.11111111111111108],
#   'noise_levels': [0],
#   'mixture_weights': [1.0],
#   'layer_num': 3,
#   'mu_ratio': 0.9,
#   'dimension': 10},
#  {'name': 'multi_task_mu_ratio_0.1_layers_1',
#   'description': 'Multi-task experiment with mu_ratio 0.1, 1 layers, dimension 10.',
#   'means': [-1, 1],
#   'cov_matrices': [22.799999999999997, 22.799999999999997],
#   'noise_levels': [0, 0],
#   'mixture_weights': [0.5, 0.5],
#   'layer_num': 1,
#   'mu_ratio': 0.1,
#   'dimension': 10},
#  {'name': 'multi_task_mu_ratio_0.1_layers_2',
#   'description': 'Multi-task experiment with mu_ratio 0.1, 2 layers, dimension 10.',
#   'means': [-1, 1],
#   'cov_matrices': [22.799999999999997, 22.799999999999997],
#   'noise_levels': [0, 0],
#   'mixture_weights': [0.5, 0.5],
#   'layer_num': 2,
#   'mu_ratio': 0.1,
#   'dimension': 10},
#  {'name': 'multi_task_mu_ratio_0.1_layers_3',
#   'description': 'Multi-task experiment with mu_ratio 0.1, 3 layers, dimension 10.',
#   'means': [-1, 1],
#   'cov_matrices': [22.799999999999997, 22.799999999999997],
#   'noise_levels': [0, 0],
#   'mixture_weights': [0.5, 0.5],
#   'layer_num': 3,
#   'mu_ratio': 0.1,
#   'dimension': 10},
#  {'name': 'multi_task_mu_ratio_0.5_layers_1',
#   'description': 'Multi-task experiment with mu_ratio 0.5, 1 layers, dimension 10.',
#   'means': [-1, 1],
#   'cov_matrices': [2.3200000000000003, 2.3200000000000003],
#   'noise_levels': [0, 0],
#   'mixture_weights': [0.5, 0.5],
#   'layer_num': 1,
#   'mu_ratio': 0.5,
#   'dimension': 10},
#  {'name': 'multi_task_mu_ratio_0.5_layers_2',
#   'description': 'Multi-task experiment with mu_ratio 0.5, 2 layers, dimension 10.',
#   'means': [-1, 1],
#   'cov_matrices': [2.3200000000000003, 2.3200000000000003],
#   'noise_levels': [0, 0],
#   'mixture_weights': [0.5, 0.5],
#   'layer_num': 2,
#   'mu_ratio': 0.5,
#   'dimension': 10},
#  {'name': 'multi_task_mu_ratio_0.5_layers_3',
#   'description': 'Multi-task experiment with mu_ratio 0.5, 3 layers, dimension 10.',
#   'means': [-1, 1],
#   'cov_matrices': [2.3200000000000003, 2.3200000000000003],
#   'noise_levels': [0, 0],
#   'mixture_weights': [0.5, 0.5],
#   'layer_num': 3,
#   'mu_ratio': 0.5,
#   'dimension': 10},
#  {'name': 'multi_task_mu_ratio_0.9_layers_1',
#   'description': 'Multi-task experiment with mu_ratio 0.9, 1 layers, dimension 10.',
#   'means': [-1, 1],
#   'cov_matrices': [0.04444444444444473, 0.04444444444444473],
#   'noise_levels': [0, 0],
#   'mixture_weights': [0.5, 0.5],
#   'layer_num': 1,
#   'mu_ratio': 0.9,
#   'dimension': 10},
#  {'name': 'multi_task_mu_ratio_0.9_layers_2',
#   'description': 'Multi-task experiment with mu_ratio 0.9, 2 layers, dimension 10.',
#   'means': [-1, 1],
#   'cov_matrices': [0.04444444444444473, 0.04444444444444473],
#   'noise_levels': [0, 0],
#   'mixture_weights': [0.5, 0.5],
#   'layer_num': 2,
#   'mu_ratio': 0.9,
#   'dimension': 10},
#  {'name': 'multi_task_mu_ratio_0.9_layers_3',
#   'description': 'Multi-task experiment with mu_ratio 0.9, 3 layers, dimension 10.',
#   'means': [-1, 1],
#   'cov_matrices': [0.04444444444444473, 0.04444444444444473],
#   'noise_levels': [0, 0],
#   'mixture_weights': [0.5, 0.5],
#   'layer_num': 3,
#   'mu_ratio': 0.9,
#   'dimension': 10}
    # {
    #     'name': 'isotropic_covariance_zero_mean',
    #     'description': 'isotropic_covariance_zero_mean',
    #     'means': [1.4, -0.6],
    #     'cov_matrices': [1.0, 1.0],
    #     'noise_levels': [0, 0],
    #     'mixture_weights': [0.3, 0.7],
    #     'dimension': 10,
    #     'isotropic': True
    # },
    {
        'name': 'noisy_zero_cov_11',
        'description': 'Mean (1.4)(0.3) + (-0.6)(0.7) = 0 and Cov 1-1',
        'means': [1.4, -0.6],
        'cov_matrices': [1.0, 1.0],
        'noise_levels': [0.3, 0.3],
        'mixture_weights': [0.3, 0.7],
        'dimension': 10,
        'isotropic': False
    },
    {
        'name': 'noisy_nonzero_cov_11',
        'description': 'Mean (1)(0.3) + (-1)(0.7) = -0.4 and Cov 1-1',
        'means': [1.0, -1.0],
        'cov_matrices': [1.0, 1.0],
        'noise_levels': [0.3, 0.3],
        'mixture_weights': [0.3, 0.7],
        'dimension': 10,
        'isotropic': False
    },
]

# Ensure the results directory exists
os.makedirs('results', exist_ok=True)

for r in tqdm(range(num_runs), desc='Runs'):
    for exp_idx, exp in enumerate(experiments):
        exp_name = exp['name']
        exp_description = exp['description']
        print(f"\nStarting experiment '{exp_name}' (Run {r + 1}/{num_runs}): {exp_description}")

        # Prepare means and covariance matrices
        means = [mean_value * torch.ones(d, device=device) for mean_value in exp['means']]
        if not exp.get('isotropic', False):
            cov_matrix = [cov_value * torch.eye(d, device=device) for cov_value in exp['cov_matrices']]
        else:
            diagonal_elements = torch.Tensor([1 / (2 ** i) for i in range(exp['dimension'])]).to(device=device, non_blocking=True)
            cov_matrix = [cov_value * torch.diag(diagonal_elements) for cov_value in exp['cov_matrices']]
        noise_levels = exp['noise_levels']
        mixture_weights = exp['mixture_weights']

        results_joint = np.full((len(prompt_lengths), len(icl_sample_counts), len(tasks)), np.inf)
        results_finetuning = np.full((len(prompt_lengths), len(icl_sample_counts), len(tasks)), np.inf)

        for i, prompt_length in enumerate(prompt_lengths):
            icl_samples_bar = tqdm(icl_sample_counts, desc='ICL Samples', leave=False)
            for j, icl_samples in enumerate(icl_samples_bar):
                icl_samples_bar.set_description(f"ICL Samples: {icl_samples}")
                task_bar = tqdm(tasks, leave=False)
                for k, task in enumerate(task_bar):
                    task_bar.set_description(f"Task: {task}")

                    # Joint Training
                    model = LinearAttn(input_size=d + 1, prompt_size=prompt_length, num_tasks=len(means)).to(device)
                    joint_training(
                        model,
                        bsize,
                        d,
                        icl_samples,
                        means,
                        cov_matrix,
                        noise_levels,
                        mixture_weights,
                        task_type=task,
                    )
                    joint_test_loss = evaluation(
                        model,
                        bsize,
                        d,
                        icl_samples,
                        means,
                        cov_matrix,
                        noise_levels,
                        mixture_weights,
                        task_type=task,
                    )
                    results_joint[i, j, k] = joint_test_loss

                    # Pretraining followed by Fine-tuning
                    if task not in ["none", "multihead"]:
                        model = LinearAttn(input_size=d + 1, prompt_size=prompt_length, num_tasks=len(means)).to(device)
                        pretrain_fine_tuning(
                            model,
                            bsize,
                            d,
                            icl_samples,
                            means,
                            cov_matrix,
                            noise_levels,
                            mixture_weights,
                            task_type=task,
                        )
                        finetuning_test_loss = evaluation(
                            model,
                            bsize,
                            d,
                            icl_samples,
                            means,
                            cov_matrix,
                            noise_levels,
                            mixture_weights,
                            task_type=task,
                        )
                        results_finetuning[i, j, k] = finetuning_test_loss

        # Save results after each experiment's run
        current_time = datetime.datetime.now().strftime("%H-%M %d-%m-%Y")
        filename = f"results/{exp_name}-run{r + 1}-{current_time}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(
                {
                    'results_joint': results_joint,
                    'results_finetuning': results_finetuning,
                    'experiment_description': exp_description,
                    'experiment_name': exp_name,
                    'run_number': r + 1,
                },
                f,
            )

        # Print results after each experiment's run
        print(f"Results for experiment '{exp_name}' after run {r + 1}:")
        print("Joint Pretraining Min Results:")
        print(results_joint)
        print("Pretrain → Fine-tuning Min Results:")
        print(results_finetuning)
