"""***********************************************************************
Diffusion Models for Risk-Aware Portfolio Optimization

-------------------------------------------------------------------------
File: main.py
- The main python file works with multiple command-line arguments in CLI.

Version: 1.0
***********************************************************************"""


import socket
import argparse
import numpy as np
import datetime as dt

import utils.utils as utils
from environment import MarketEnv

import torch
from diff_utils.exp.exp_main import Exp_Main

from datetime import datetime
from utils import tools


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def add_arguments():
    parser_ = argparse.ArgumentParser(description='Diffusion Models for Risk-Aware Portfolio Optimization')
    # Basic config
    parser_.add_argument('--comment', type=str, default=None)
    parser_.add_argument('--id_', type=str, default=None, help='id of the model')
    parser_.add_argument('--mode', type=str, default='end-to-end', help="options: ['end-to-end']")
    parser_.add_argument('--from_yaml', type=str, default='configs/cn_train.yml', help='override the config from the indicated yaml.')
    parser_.add_argument('--is_training', type=bool, default=True, help='status')
    parser_.add_argument('--test_type', type=str, default='denoising')  # deprecated
    parser_.add_argument('--date_from', type=str, default='2009-01-05')
    parser_.add_argument('--date_to', type=str, default='2020-12-31')
    parser_.add_argument('--type_', type=str, default='stocks_index_cn')

    # method settings
    parser_.add_argument('--dim1', type=int, default=64)
    parser_.add_argument('--dim2', type=int, default=64)
    parser_.add_argument('--window_size', type=int, default=256)
    parser_.add_argument('--reb_freq', type=int, default=5)
    parser_.add_argument('--normalize', type=str, default='irvpop')
    parser_.add_argument('--lambda_', type=float, default=1)
    parser_.add_argument('--pred_type', type=str, default='y_0', help="prediction type, options: ['eps', 'y_0']")
    parser_.add_argument('--risk_levels', type=int, default=5, help='number of risk levels')

    # GPU
    parser_.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser_.add_argument('--gpu', type=int, default=3)
    parser_.add_argument('--seed', type=int, default=0, help='random seed')  # 0, 3283, 6901, 9879, 7849, 6676, ...
    parser_.add_argument('--checkpoints', type=str, default='out/checkpoints/', help='location of model checkpoints')

    # model define
    parser_.add_argument('--mul_port_', type=int, default=0, help='Use multiple portfolio')
    parser_.add_argument('--timesteps', type=int, default=100, help='number of diffusion steps (T)')
    parser_.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser_.add_argument('--num_samples', type=int, default=100, help='number of samples per forecast')

    # optimization
    parser_.add_argument('--train_epochs', type=int, default=400, help='train epochs')
    parser_.add_argument('--batch_size', type=int, default=8, help='batch size of train input data')
    parser_.add_argument('--test_batch_size', type=int, default=8, help='batch size of test input data')
    parser_.add_argument('--patience', type=int, default=100, help='early stopping patience')
    parser_.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')

    # deprecated
    parser_.add_argument('--max_portfolio', type=int, default=0)
    parser_.add_argument('--pw_dim', type=int, default=1)
    parser_.add_argument('--num_ef', type=int, default=1)
    parser_.add_argument('--dim3', type=int, default=1)
    parser_.add_argument('--var1', type=int, default=1)
    parser_.add_argument('--uma', type=str2bool, nargs='?', const=True, default=False, help='use moving-average features')

    # forecasting task (deprecated)
    parser_.add_argument('--seq_len', type=int, default=256, help='input sequence length == window_size')
    parser_.add_argument('--label_len', type=int, default=36, help='start token length (deprecated)')
    parser_.add_argument('--pred_len', type=int, default=0, help='prediction sequence length (deprecated)')
    return parser_


if __name__ == '__main__':
    parser_ = add_arguments()
    args = parser_.parse_args()
    if args.from_yaml is not None and args.from_yaml != "None":
        yaml_config: dict = tools.conf_from_yaml(args.from_yaml)
        for k, v in yaml_config.items():
            setattr(args, k, v[0])
    args.window_size = args.seq_len
    args.device = f"cuda:{args.gpu}"
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f'Args:\n{args}')

    if args.id_ is None:
        args.id_ = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.cuda.set_device(args.gpu)
    utils.fix_seed(args.seed)
    np.seterr(all='ignore')
    np.set_printoptions(precision=3, linewidth=2000, edgeitems=50, suppress=False)
    torch.set_printoptions(linewidth=2000, edgeitems=50)

    new_model_name = str(dt.datetime.now())
    log_path = f"out/log/{new_model_name}_{socket.gethostname()[-1]}_{args.gpu}.log"
    with open(log_path, "a") as f:
        f.write(str(args.__dict__) + "\n")

    env = MarketEnv(date_from=args.date_from, date_to=args.date_to, type_=args.type_,
                    window_size=args.window_size,
                    reb_freq=args.reb_freq, use_short=True, use_partial=None, split_ratio=[7, 1, 2],
                    normalize=args.normalize, add_cash=False,
                    batch_size=args.batch_size, test_batch_size=args.test_batch_size, args=args)

    exp = Exp_Main(args, env)
    exp.train()
    res, rank_score, post_res, _ = exp.run_test()
    pv, arr, asr, mdd, avol, cr, sor = res
    post_res_mean = post_res.mean(axis=(0, 2))
    pv_, arr_, asr_, mdd_, avol_, cr_, sor_ = post_res_mean
    evals = {"pv": pv,
             "arr": arr,
             "asr": asr,
             "mdd": mdd,
             "avol": avol,
             "cr": cr,
             "sor": sor,
             "pv_": pv_,
             "arr_": arr_,
             "asr_": asr_,
             "mdd_": mdd_,
             "avol_": avol_,
             "cr_": cr_,
             "sor_": sor_,
             "rank_score": rank_score}

    utils.to_csv(path="result.csv", out_dict={**evals, **args.__dict__})  # save the result to the current path
