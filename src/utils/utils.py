import json
import os
import csv
import torch
import random
import numpy as np
from typing import List, Dict, Set, Iterable, Callable, Optional, Union, Tuple


def fix_seed(seed: int):
    # fix python random seed
    random.seed(seed)
    # fix numpy seed
    np.random.seed(seed)
    # fix torch seed
    torch.manual_seed(seed)
    # fix CUDA seed
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # fix CuDNN seed
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def to_csv(out_dict: Dict[str, Union[int, float, str]], path: str):
    is_exists = True
    header = list(out_dict.keys())
    if not os.path.isfile(path):
        with open(path, 'w') as f:
            is_exists = False
    with open(path, 'a') as f:
        writer = csv.writer(f)

        if not is_exists:
            writer.writerow(header)

        writer.writerow(list(out_dict.values()))


# Optimization routine
def optimize_dynamic_portfolio(returns, lambda_, max_iters=100, lr=0.01, covs=None):
    T, N = returns.shape
    weights = torch.randn(T, N, device=returns.device, requires_grad=True)  # One weight per timestep

    from torch.optim import Adam
    optimizer = Adam([weights], lr=lr)

    for _ in range(max_iters):
        optimizer.zero_grad()
        portfolio_returns = torch.einsum('tn,tn->t', weights, returns)
        total_return = portfolio_returns.sum()
        mean_portfolio_return = portfolio_returns.mean()
        volatility = torch.sqrt(((portfolio_returns - mean_portfolio_return) ** 2).mean())  # Scalar
        sparsity_penalty = torch.mean(torch.sum(weights ** 2, dim=1))
        objective = (1 - lambda_) * volatility - mean_portfolio_return + sparsity_penalty # rm lambda right
        objective.backward()
        optimizer.step()

        with torch.no_grad():
            weights.clamp_(-1, 1)  # Keep weights in [-1, 1]
            abs_sum = torch.sum(torch.abs(weights), dim=1, keepdim=True)
            weights.div_(abs_sum)  # Normalize to absolute sum = 1

    return weights.detach().cpu().numpy()
