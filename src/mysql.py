"""***********************************************************************
Diffusion Models for Risk-Aware Portfolio Optimization

-------------------------------------------------------------------------
File: mysql.py
- The utility functions for the data preparation.

Version: 1.0
***********************************************************************"""

import numpy as np
import pandas as pd
import datetime as dt
from normalizer import normalizer
from typing import List, Optional, Union, Tuple

pd.set_option('display.max_columns', 200, 'display.max_rows', 2000, 'display.expand_frame_repr', False)

STOCKS_CN_PATH = '../data/stocks_cn.csv'
INDEX_CN_PATH = '../data/index_cn.csv'


class StockData:
    def __init__(self, date_to: Optional[str] = None, date_from: Optional[str] = None):
        self.date_from: dt.datetime = dt.datetime.strptime(date_from, '%Y-%m-%d')
        self.date_to: dt.datetime = dt.datetime.strptime(date_to + ' 23:59:59', '%Y-%m-%d %H:%M:%S')
        assert (self.date_to - self.date_from).days >= 0, 'Invalid start or end date'

    def fetch_data(self, country: str = 'cn', type_: str = None, to_numpy: bool = False) -> \
            Tuple[np.array, np.array, np.array]:
        data: Union[pd.DataFrame, np.array] = pd.DataFrame()
        sorted_s_lst: List[str] = []  # Sorted Selection List of stocks
        dates = None
        names = None
        if type_ == 'index':
            path, indices, drop_cols = None, [], []
            if country == 'cn':
                indices.append('sse50')
                path = INDEX_CN_PATH
                drop_cols = ['date']

            data = pd.read_csv(path, header=0, parse_dates=['date']). \
                sort_values(by=['date'], ascending=True)
            data = data[(self.date_from <= data['date']) & (data['date'] <= self.date_to)].reset_index(drop=True)
            dates = data['date'].unique()
            names = np.array(indices)
            data.drop(drop_cols, axis='columns', inplace=True)
            sorted_s_lst = indices

        elif type_ == 'stocks_cn':
            data = pd.read_csv(STOCKS_CN_PATH, header=0, parse_dates=['date']). \
                sort_values(by=['date', 'stock_id'], ascending=[True, True])
            sorted_s_lst = data[data['date'] == data.iloc[0]['date']]['stock_id'].unique()
            data = data[(self.date_from <= data['date']) & (data['date'] <= self.date_to)].reset_index(drop=True)
            dates = data['date'].unique()
            names = data['stock_id'].unique()
            data.drop(['date', 'stock_id'], axis='columns', inplace=True)

        if to_numpy:
            data = data.to_numpy()
            data = data.reshape((-1, len(sorted_s_lst), data.shape[1]))

        return data, dates, names  # T, N, F


def fetch_dataset(data: np.array, type_: str, window_size: int, normalize: Optional[str] = None,
                  is_pre_label: Optional[bool] = False, transpose: Optional[Tuple[int, int, int, int]] = None,
                  reb_freq: int = 1, glob_data: Optional[np.array] = None, dates: np.array = None, args=None) -> \
        Union[tuple, List[tuple]]:
    prev_data = np.copy(data)  # preserved data before norm
    prev_glob_data = np.copy(glob_data) if glob_data is not None else None  # preserved data before norm
    pred_len = args.pred_len if args is not None else 0
    ma_len = 0  # Length for computing the Moving Average
    if args is not None:
        ma_len = 30 if args.uma else 0

    def prep(x: np.array, prev_x: np.array, dates: np.array, args):
        def moving_averages_(p_x, ma_len, periods):
            def moving_average_(arr, period):
                T_plus_ma_len, N, F = arr.shape
                T = T_plus_ma_len - period
                result = np.zeros((T, N))
                for t in range(T):
                    # (N, )... (mean[t-period, t] -[t]) / [t] \in (-1, +\infty)
                    ma_ = np.mean(arr[t:t+period, :, 0], axis=0)
                    result[t] = (arr[t+period, :, 0] - ma_) / ma_  # (N, )
                return result.reshape(T, N, 1)

            ma_lst = []
            for period in periods:
                ma_lst.append(moving_average_(arr=p_x[ma_len - period:], period=period))

            return np.concatenate(ma_lst, axis=2)

        if normalize:
            x, prev_x, dates = normalizer(features=x, prev_features=prev_x, norm_type_=normalize, dates=dates)
        x = x.transpose((2, 1, 0))  # (T, N, F) -> (F, N, T)

        if args.uma:
            # (T-ma_len, N, len(periods)) -> (len(periods), N, T-ma_len)
            mas = moving_averages_(p_x=prev_x, ma_len=ma_len, periods=np.arange(1, 7) * 5).transpose(2, 1, 0)
            mas = np.concatenate([np.zeros_like(mas)[:, :, :ma_len], mas], axis=2)  # Append zeros to the temporal dim
            x = np.concatenate([x, mas], axis=0)  # Concat moving average features to x

        F, N, T = x.shape
        R = T - (window_size + reb_freq + pred_len + ma_len + 1)
        assert R > 0, (f"At least one data point should be exist. "
                       f"current {T} <= {window_size + reb_freq + pred_len + ma_len + 1}")

        X: np.array = None
        Y: np.array = None
        buying_date_indices: list = []
        X_date_indices: list = []  # 2d-array containing date list for each instance of X

        _t = ma_len  # 0
        while _t + window_size + reb_freq + pred_len <= T - 1:
            buying_date_indices.append(_t + window_size)
            X_date_indices.append(np.arange(_t, _t + window_size + pred_len).tolist())

            _x = x[:, :, _t:_t + window_size + pred_len]
            X = _x if X is None else np.vstack((X, _x))

            _b_p = prev_x[_t + window_size, :, 0]  # Buying price (Open)
            _s_p = prev_x[_t + window_size + reb_freq, :, 0]  # Selling price (Open)
            _y_o = (_s_p - _b_p) / _b_p * 100  # cr_o

            # Replace the invalid open change rate due to zero opening price
            _y_o[_y_o == -100.0] = 0.0

            # Highest of the highest price
            _h_p = np.max(prev_x[_t + window_size: _t + window_size + reb_freq, :, 1], axis=0)
            _y_h = (_h_p - _b_p) / _b_p * 100  # cr_h

            Y = np.stack([_y_o, _y_h]) if Y is None else np.vstack((Y, np.stack([_y_o, _y_h])))
            _t += 1

        X = X.reshape((-1, F, N, window_size + pred_len))
        Y = Y.reshape((-1, 2, N))

        # # Handle anomalies
        X[(X < -0.6) | (X > 0.6)], Y[(Y < -60) | (Y > 60)] = 0.0, 0.0
        np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

        if transpose:
            X = X.transpose(transpose)

        return X, Y, dates, buying_date_indices, X_date_indices

    dataset_pack = prep(data, prev_data, dates, args)
    glob_dataset_pack = prep(glob_data, prev_glob_data, dates, args) if glob_data is not None else None

    return dataset_pack, glob_dataset_pack
