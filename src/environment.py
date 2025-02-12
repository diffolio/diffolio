"""***********************************************************************
Diffusion Models for Risk-Aware Portfolio Optimization

-------------------------------------------------------------------------
File: environment.py

Version: 1.0
***********************************************************************"""


import os
import pandas as pd
import torch
import numpy as np
from mysql import fetch_dataset, StockData
from typing import List, Optional, Union, Tuple
from torch.utils.data import Dataset, DataLoader
from utils.timefeatures import time_features
from scipy.stats import percentileofscore
from utils.utils import optimize_dynamic_portfolio


def multi_opt_ls_port(**kwargs):
    ports = []
    for risk_weight_ in np.linspace(0, 0.9, kwargs["risk_levels"]):  # 0, 0.5
        ports.append(optimize_dynamic_portfolio(returns=torch.tensor(kwargs["returns"], device=kwargs["device"], dtype=torch.float32),
                                                lambda_=risk_weight_, covs=kwargs["covs"]))
    return np.array(ports).transpose(1, 0, 2)  # (Stack, B, N) -> (B, Stack, N)


def multi_hot_risk_adj_ls_port(returns, covs, num_asset: int, k: int = 3):
    assert k <= num_asset, f"Top-k: {k} should be less than the number of assets: {num_asset}"
    # Variance of the assets [T, N]
    opt_port = np.zeros_like(returns)
    top_k_indices = np.argsort(np.abs(returns), axis=1)[:, ::-1][:, :k]
    # Create mask (=1) at top-k return asset indices
    np.put_along_axis(opt_port, top_k_indices, 1.0, axis=1)
    # Fill values
    opt_port *= returns
    return opt_port / np.abs(opt_port).sum(axis=1, keepdims=True)


def pseudo_optimal_port(**kwargs):
    ports = []
    num_asset = kwargs["returns"].shape[1]
    for k_ in (np.arange(1, kwargs["risk_levels"]+1) * (num_asset // kwargs["risk_levels"]))[::-1]:
        ports.append(multi_hot_risk_adj_ls_port(returns=kwargs["returns"], covs=kwargs["covs"], num_asset=num_asset, k=int(k_)))
    return np.array(ports).transpose(1, 0, 2)  # (Stack, B, N) -> (B, Stack, N)


class DL_Dataset(Dataset):
    def __init__(self, x, y, glob_x, glob_y, glob_y_trn, args, port_type="returns", label_func=pseudo_optimal_port):
        # Split indices
        self.B, self.F, self.N, self.T_p = x.shape
        self.args = args
        self.label_len = self.args.label_len
        self.pred_len = self.args.pred_len
        self.seq_len = self.T_p - self.pred_len

        # [0, 256]
        self.s_begin = 0
        self.s_end = self.s_begin + self.seq_len

        self.x = x[:, :, :, self.s_begin:self.s_end].copy()              # (B, F, N, T)
        self.y = y                                                       # (B, 2, N,  )
        self.n_asset = self.y.shape[2]
        self.glob_x = glob_x[:, :, :, self.s_begin:self.s_end].copy()    # (B, F, 1, T)
        self.glob_y = glob_y                                             # (B, 2, 1,  )
        self.glob_y_trn = glob_y_trn
        self.x_corr, self.x_beta = self.comp_x_corr(is_var=True)
        self.y_risk_level, self.y_risk_percentile = self.comp_y_risk_level()  # [B,], [B, N]
        self.pre_temps = torch.linspace(1.2, 0.1, self.args.risk_levels) if self.args.risk_levels != 0 else None

        # Construct the optimal portfolio
        if port_type == "one_hot" or port_type is None:  # CE Loss
            self.y_return = self.one_hot_port()
        elif port_type == "multi_hot":  # CE Loss
            self.y_return = self.multi_hot_port()
        elif port_type == "soft_label":  # CE Loss
            self.y_return = self.soft_label_port()
        elif port_type == "binary":  # BCE Loss
            self.y_return = self.binary_port()
            self.y = self.binaries()
        elif port_type == "returns":  # MSE Loss
            from diff_utils.exp.exp_main import construct_portfolio_
            self.y = self.returns()
            self.y_return = torch.tensor(label_func(returns=self.y,
                                                    risk_levels=self.args.risk_levels,
                                                    device=self.args.device,
                                                    covs=self.x_corr)) if self.args.risk_levels else (
                construct_portfolio_(torch.tensor(self.y * 100)))
        elif port_type == "norm_return":
            self.y_return = self.norm_return_port()
            self.y = self.norm_return_port()
        else:
            raise ValueError(f"Unknown port type: {port_type}")

        self.x_opt_port = torch.zeros_like(self.y_return)

    def comp_x_corr(self, is_var: bool = False):
        price_diffs = self.x[:, 0]  # Price percentage diffs Open to Open (B, N, T)
        if is_var:  # Retrieve covariance matrix instead of correlation matrix
            corrs = np.stack([np.cov(b_) for b_ in price_diffs])  # (B, N, N)
        else:
            corrs = np.stack([np.corrcoef(b_) for b_ in price_diffs])  # (B, N, N)

        market_diffs = self.glob_x[:, 0]  # Market index's price percentage diffs Open to Open (B, 1, T)
        market_var = np.var(market_diffs, axis=2)
        price_diffs = np.concatenate([market_diffs, price_diffs], axis=1)  # (B, N+1, T)
        betas = np.stack([np.cov(b_) for b_ in price_diffs])[:, 0, 1:] / market_var # (B, N+1, N+1) -> (B, N)
        return corrs, betas

    def comp_y_risk_level(self):
        B, F, _, T = self.glob_x.shape
        prev_price_diffs = self.glob_x[:, 0, :, 1:] + 1.0  # (B, 1, T-1)
        five_days_prev_price_diffs = prev_price_diffs.reshape(B, 1, T // 5, 5).prod(axis=3).reshape(B, -1)
        five_days_prev_price_diffs -= 1.0
        y_price_diffs = self.glob_y[:, 0].reshape(-1) / 100  #
        percentiles = []
        for idx_, price_diffs_ in enumerate(five_days_prev_price_diffs):
            percentiles.append(percentileofscore(np.abs(price_diffs_), np.abs(y_price_diffs[idx_])))

        y_risk_percentile = np.array(percentiles).reshape(B)  # v4, v5
        y_risk_level = np.clip(y_risk_percentile / (100 // self.args.risk_levels), 0, self.args.risk_levels - 1)
        return y_risk_level.astype(np.int64), y_risk_percentile

    def returns(self):
        return self.y[:, 0] / 100

    # A dense portfolio with the weights derived by normalizing the returns
    def norm_return_port(self):
        rets = self.y[:, 0]
        return rets / np.sum(np.abs(rets), axis=1, keepdims=True)

    def binaries(self):
        ret_sign = np.sign(self.y[:, 0])
        ret_sign[ret_sign == -1] = 0
        return ret_sign

    def binary_port(self):
        ret_sign = np.sign(self.y[:, 0])
        ret_sign[ret_sign == 0] = -1
        equal_weight = 1 / self.n_asset
        return ret_sign * equal_weight

    # A sparse portfolio with max return asset having all the weight
    def one_hot_port(self):
        cr_o = self.y[:, 0]
        opt_port = np.zeros_like(cr_o)
        top_1_index = np.argmax(cr_o, axis=1, keepdims=True)
        # Fill values at top-1 return asset indices
        np.put_along_axis(opt_port, top_1_index, 1.0, axis=1)
        return opt_port

    # Generalization of one_hot portfolio (i.e., k=1 -> one_hot)
    def multi_hot_port(self, k: int = 3):
        assert k <= self.n_asset, f"Top-k: {k} should be less than the number of assets: {self.n_asset}"
        cr_o = self.y[:, 0]
        opt_port = np.zeros_like(cr_o)
        top_k_indices = np.argsort(cr_o, axis=1)[:, ::-1][:, :k]
        # Fill values at top-k return asset indices
        np.put_along_axis(opt_port, top_k_indices, 1/k, axis=1)
        return opt_port

    # A dense portfolio with the weights derived by dividing the actual return with the standard deviation
    def soft_label_port(self, mov_avg_len: int = 60, use_temp: bool = True):
        cr_o = self.y[:, 0]
        latest_rets = self.x[:, 3, :, -mov_avg_len:] * 100  # (B, F, N, T) --> (B, N, T)
        asset_vars = np.std(latest_rets, axis=2)  # (B, N)
        norm_weights = cr_o / asset_vars
        np.nan_to_num(norm_weights, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        temp = np.sqrt(self.n_asset) if use_temp else 1
        opt_port = torch.softmax(torch.tensor(norm_weights/temp), dim=1).numpy()
        return opt_port

    def multi_soft_abs_ls_port(self, mov_avg_len: int = 60):
        cr_o = self.y[:, 0]
        latest_rets = self.x[:, 0, :, -mov_avg_len:] * 100  # (B, F, N, T) --> (B, N, T)
        asset_vars = np.std(latest_rets, axis=2)

        ports = []
        def _comp_indiv_port(var_weight):  # Compute weighted port
            norm_weights = cr_o / (asset_vars * var_weight)
            np.nan_to_num(norm_weights, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
            norm_weights = (norm_weights ** 2) * np.sign(norm_weights)
            return norm_weights / np.sum(np.abs(norm_weights), axis=1, keepdims=True)

        for var_weight_ in np.linspace(1, 1 / asset_vars, self.args.risk_levels):
            ports.append(_comp_indiv_port(var_weight=var_weight_))

        return np.array(ports).transpose(1, 0, 2)  # (Stack, B, N) -> (B, Stack, N)

    def __getitem__(self, index):
        return (self.x[index],
                self.y[index],
                torch.tensor(0),
                torch.tensor(0),
                self.y_return[index],
                self.glob_x[index],
                self.x_corr[index],
                self.x_beta[index],
                self.y_risk_level[index],
                self.x_opt_port[index])

    def __len__(self):
        return len(self.x)


class MarketEnv:
    def __init__(self, date_from: str, date_to: str, type_: str, window_size: int, reb_freq: int,
                 split_ratio: List[int], use_short: bool = False, add_cash: bool = True,
                 normalize: Optional[str] = None, use_partial: Optional[int] = None,
                 f_l: float = 0.0, f_s: float = 0.0, batch_size: int = 32, test_batch_size: int = 8, args=None):
        self.args = args
        self.type_ = type_
        self.data: np.array = None
        self.dates: np.array = None
        self.assets: np.array = None
        self.glob_data: np.array = None

        if self.type_ == 'stocks_index_cn':
            self.data_cls: StockData = StockData(date_from=date_from, date_to=date_to)
            self.data, self.dates, self.assets = \
                self.data_cls.fetch_data(country='cn', type_='stocks_cn', to_numpy=True)
            self.glob_data, _, _ = self.data_cls.fetch_data(country='cn', type_='index', to_numpy=True)
        else:
            raise ValueError(f"{self.type_} unsupported data type.")
        assert np.sum(split_ratio) == 10, 'split ratios should be sum to 10.'

        self.f_l = f_l
        self.f_s = f_s
        self.reb_freq = reb_freq
        self.add_cash = add_cash
        self.use_short = use_short
        self.normalize = normalize
        self.use_partial = use_partial
        self.split_ratio = split_ratio
        self.window_size = window_size
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.use_glob = False if self.glob_data is None else True

        if self.use_partial is not None:
            self.data = self.data[:, :self.use_partial, :]

        type_config_ = (f"{self.type_}_{date_from}_{date_to}_rf_{reb_freq}_seq_{args.seq_len}_label_{args.label_len}_"
                        f"pred_{args.pred_len}_{str(split_ratio).replace(' ', '')}_uma_{1 if args.uma else 0}.save")
        type_save_path_ = os.path.join('../data', type_config_)
        if os.path.isfile(type_save_path_):
            print(f"Loading dataset from {type_save_path_} ...")
            dataset_pack, glob_dataset_pack = torch.load(type_save_path_)
        else:
            print(f"Creating dataset to {type_save_path_} ...")
            dataset_pack, glob_dataset_pack = fetch_dataset(data=self.data,
                                                            glob_data=None if self.glob_data is None else self.glob_data,
                                                            window_size=self.window_size, reb_freq=self.reb_freq,
                                                            normalize=self.normalize, type_=type_, dates=self.dates,
                                                            args=args)
            torch.save((dataset_pack, glob_dataset_pack), type_save_path_, pickle_protocol=4)

        self._trn_idx = int(len(dataset_pack[0]) * self.split_ratio[0] / 10)
        self._val_idx = self._trn_idx + int(len(dataset_pack[0]) * self.split_ratio[1] / 10)
        freq = 'd'

        print(f"Unpacking dataset ...")
        def _unpack(target_dataset: tuple):
            X, Y, dates, buying_date_indices, X_date_indices = target_dataset
            return (X[:self._trn_idx], Y[:self._trn_idx],
                    X[self._trn_idx:self._val_idx], Y[self._trn_idx:self._val_idx],
                    X[self._val_idx:], Y[self._val_idx:],
                    dates[buying_date_indices[:self._trn_idx]],
                    dates[buying_date_indices[self._trn_idx:self._val_idx]],
                    dates[buying_date_indices[self._val_idx:]],
                    np.array([time_features(pd.to_datetime(dates[date_indices]), freq=freq) for date_indices in X_date_indices[:self._trn_idx]]),
                    np.array([time_features(pd.to_datetime(dates[date_indices]), freq=freq) for date_indices in X_date_indices[self._trn_idx:self._val_idx]]),
                    np.array([time_features(pd.to_datetime(dates[date_indices]), freq=freq) for date_indices in X_date_indices[self._val_idx:]]))

        self.x_trn, self.y_trn, self.x_val, self.y_val, self.x_test, self.y_test, \
            self.d_trn, self.d_val, self.d_test, self.x_d_trn, self.x_d_val, self.x_d_test = _unpack(dataset_pack)
        if self.use_glob:
            self.glob_x_trn, self.glob_y_trn, self.glob_x_val, self.glob_y_val, self.glob_x_test, self.glob_y_test, \
                _, _, _, _, _, _ = _unpack(glob_dataset_pack)
            self.glob_x, self.glob_y = self.glob_x_trn, self.glob_y_trn

        self.x, self.y = self.x_trn, self.y_trn

        self.state_space: Tuple[int] = (*self.x[0].shape[:2], self.x[0].shape[2] - self.args.pred_len)
        self.action_space: Tuple[int] = self.y[0][0].shape
        self.n_asset: int = self.action_space[0]

        self.n_step: int = 0
        self.mode: str = 'train'
        self.total_step: int = self.x.shape[0]
        self.state: np.array = self.x[self.n_step]
        self.w = np.array([[1.0] + [0.0] * self.action_space[0]]) if self.add_cash \
            else np.array([[0.0] * self.action_space[0]])
        self.glob_state: np.array = self.glob_x[self.n_step] if self.use_glob else None

        print(f"Preparing dataloaders ...")
        self.train_loader = DataLoader(
            DL_Dataset(x=self.x_trn, y=self.y_trn, glob_x=self.glob_x_trn, glob_y=self.glob_y_trn, glob_y_trn=self.glob_y_trn, args=self.args),
            batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.vali_loader = DataLoader(
            DL_Dataset(x=self.x_val, y=self.y_val, glob_x=self.glob_x_val, glob_y=self.glob_y_val, glob_y_trn=self.glob_y_trn, args=self.args),
            batch_size=self.test_batch_size, shuffle=False, drop_last=True)
        self.test_loader = DataLoader(
            DL_Dataset(x=self.x_test, y=self.y_test, glob_x=self.glob_x_test, glob_y=self.glob_y_test, glob_y_trn=self.glob_y_trn, args=self.args),
            batch_size=self.test_batch_size, shuffle=False, drop_last=True)

        if len(self.train_loader.dataset.y_return.shape) != 3:
            self.trn_p_std = torch.std(self.train_loader.dataset.y_return, dim=1).mean()
        else:
            self.trn_p_std = torch.std(self.train_loader.dataset.y_return.reshape(-1, self.n_asset), dim=1).mean()

    def step(self, action: np.array) -> \
            Tuple[np.array, Union[np.array, np.float64], bool, Optional[np.array], float, float, np.array, np.array]:
        nxt_idx_gap = self.reb_freq

        if not self.add_cash:
            t_action = action  # No truncation required
        else:
            t_action = action[:, 1:]  # Truncated without cash = v_{t,\0} (1, N)

        tot_fee = self.calc_tot_fee(W=self.w[0], V=action[0])  # W=(1, N+1), V=(1, N+1), where 1 is size of the batch

        # Calc w(=_w), w'(=w), and profit & loss (pl)
        w_cash = np.expand_dims(action[:, 0], 1)  # Cash weights
        cr_o = self.y[self.n_step][0] / 100  # (N, )
        cr_h = self.y[self.n_step][1] / 100  # (N, )
        d = (1 + np.sign(t_action)) / 2 * t_action * cr_o + \
            (1 - np.sign(t_action)) / 2 * ((cr_h >= 1.0) * np.abs(t_action) + (cr_h < 1.0) * np.abs(t_action) * cr_o)

        pl = np.sum(np.sign(t_action) * d, axis=1)[0]
        if not self.add_cash:
            _w = t_action + d
        else:
            _w = np.concatenate([w_cash, t_action + d], axis=1)  # Changed weights
        self.w = _w / np.expand_dims(np.sum(np.abs(_w), axis=1), axis=1)

        r = pl - tot_fee
        self.n_step += nxt_idx_gap

        s_, done = self.x[self.n_step], False
        g_ = self.glob_x[self.n_step] if self.use_glob else None

        if self.n_step + nxt_idx_gap >= self.x.shape[0]:
            done = True

        return s_, r, done, g_, pl, tot_fee, d, cr_o

    def reset(self) -> np.array:
        self.n_step = 0
        self.state = self.x[self.n_step]
        self.glob_state: np.array = self.glob_x[self.n_step] if self.use_glob else None
        self.w = np.array([[1.0] + [0.0] * self.action_space[0]]) if self.add_cash \
            else np.array([[0.0] * self.action_space[0]])
        return self.state, self.glob_state

    def set_mode(self, mode: str):
        self.mode = mode

        if mode == 'train':
            self.x, self.y = self.x_trn, self.y_trn
        elif mode == 'validation':
            self.x, self.y = self.x_val, self.y_val
        elif mode == 'test':
            self.x, self.y = self.x_test, self.y_test

        if self.use_glob:
            if mode == 'train':
                self.glob_x, self.glob_y = self.glob_x_trn, self.glob_y_trn
            elif mode == 'validation':
                self.glob_x, self.glob_y = self.glob_x_val, self.glob_y_val
            elif mode == 'test':
                self.glob_x, self.glob_y = self.glob_x_test, self.glob_y_test

        self.reset()

    def calc_tot_fee(self, W: np.array, V: np.array) -> float:
        N = W.shape[0]
        C = np.zeros(N)
        O = np.zeros(N)
        eps = 1e-10
        sign_x = lambda x: np.sign(np.sign(x) + 0.5)

        while True:
            C_prev = np.copy(C)
            O_prev = np.copy(O)

            A = np.sum((np.abs(C_prev) * ((1+sign_x(C_prev)) * self.f_l / 2 + (1 - sign_x(C_prev)) * self.f_s / 2))[1:])
            C = O_prev + W + V * (A - 1)

            C[W * C < 0] = 0
            C[np.abs(C) > np.abs(W)] = W[np.abs(C) > np.abs(W)]

            A = np.sum((np.abs(C) * ((1 + sign_x(C)) * self.f_l / 2 + (1 - sign_x(C)) * self.f_s / 2))[1:])
            O = (C - W + V * (1 - A))
            O[(W - C) * O < 0] = 0

            if np.linalg.norm(C_prev - C) <= eps:
                break

        return float(A)
