import os
import time
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from utils.tools import EarlyStopping
from diff_utils.exp.exp_basic import Exp_Basic
from network import CondDiffModel, asset_selection
import experiment
from diff_utils.diffusion_models.diffusion_utils import *

# warnings.filterwarnings('ignore')

TEMPERATURE = 1.0  # global variable for the portfolio construction temperature adjustment
MAX_PORTFOLIO = 0


def set_max_port(max_portfolio: int):
    global MAX_PORTFOLIO
    MAX_PORTFOLIO = max_portfolio


def construct_portfolio_(values: torch.Tensor,
                         type: str = "abs_ls",
                         risk: int = 0,
                         temperature=None,
                         multi_temperature: list = None,
                         tt_as: bool = False,  # Test Time Asset Selection
                         max_portfolio: int = 0,
                         ):
    assert len(values.shape) == 2, f"values dimension must be 2"

    if MAX_PORTFOLIO and tt_as:  # deprecated option
        values = asset_selection(values, max_portfolio=max_portfolio if max_portfolio else MAX_PORTFOLIO)

    temp = None
    if temperature is not None:  # Use set temperature if it is not None
        temp = temperature.reshape(-1, 1)
    else:
        temp = TEMPERATURE  # k_z

    if type == "long":
        return torch.softmax(values, dim=1)
    elif type == "temp_long":
        temp_ = 4  # [if > 1 -->  ===> x <=== smoothing ]       [else [0,1] -->  <=== x ===>  hard labeling]
        return torch.softmax(values / temp_, dim=1)
    elif type == "abs_ls":
        return values / torch.sum(torch.abs(values), dim=1, keepdim=True)
    elif type == "sigmoid_threshold":
        threshold = 0.5  # This can be modified to the user-defined level (e.g., 0.7 = 70% confident assets are long)
        centered_values = values - threshold
        return centered_values / torch.sum(torch.abs(centered_values), dim=1, keepdim=True)
    elif type == "sigmoid_threshold_equal":
        threshold = 0.5
        centered_values = values - threshold
        equal_weight = 1 / values.shape[1]
        values_sign = torch.sign(centered_values)
        values_sign[values_sign == 0] = -1
        return values_sign * equal_weight
    elif type == "temp_max_ls":
        temp_abs_max = torch.nn.functional.softmax(torch.abs(values) / temp, dim=1)
        sign_ = torch.sign(values)
        sign_[sign_ == 0] = 1
        return sign_ * temp_abs_max
    elif type == "temp_max_ls_partial":  # Mask out zeros for softmax computation (used for asset-selection option)
        sign_ = torch.sign(values)
        sign_[sign_ == 0] = 1
        non_zero_mask = values != 0

        def _get_temperatured_output(sign_, non_zero_mask, values, temp):
            tensor_2d_modified = torch.where(non_zero_mask, torch.abs(values) / temp,
                                             torch.tensor(float('-inf')).to(values.device))
            softmaxed_tensor = torch.nn.functional.softmax(tensor_2d_modified, dim=1)
            return softmaxed_tensor * non_zero_mask.float() * sign_

        if multi_temperature is not None:
            output_tensor = []
            for temp_ in multi_temperature:
                output_tensor.append(_get_temperatured_output(sign_, non_zero_mask, values, temp_))
            output_tensor = torch.stack(output_tensor, dim=1)  # (B, Stack, N)
        else:
            output_tensor = _get_temperatured_output(sign_, non_zero_mask, values, temp)

        return output_tensor
    elif type == "risk_auto_temp_max_ls_partial":
        var_ratio_multiplier = torch.tensor(5, device=values.device)  # For high and low temperatures
        std_ = torch.std(values, dim=1, keepdim=True)
        T_high = torch.sqrt(var_ratio_multiplier) * std_  # var(x / T) = 1/10 -> T = sqrt(10 * Var(x))
        T_low = std_ / torch.log(
            var_ratio_multiplier)  # exp((mu + 2 * std(x)) / T) / exp((mu + std(x)) / T) = 10 -> T = std(x) / log(10)
        T_linear_interpolate = T_high + (risk) * T_low
        return T_linear_interpolate


def compute_entropy(values: torch.Tensor):
    return torch.sum(-torch.abs(values) * (torch.abs(values) + 1e-10).log(), dim=1)


def load_model(model_name: str, model_save_path: str, map_location: str = None):
    checkpoint = torch.load(model_save_path + f"{model_name}.pth")
    return checkpoint['model']


class Exp_Main(Exp_Basic):
    def __init__(self, args, env):
        super(Exp_Main, self).__init__(args, env)
        self.num_ld_iter = 0  # Langevin Dynamics Sampling Iteration Number
        self.num_mps_iter = 0  # Multiple Proxy Sampling Iteration Number
        self.dl_type = 'DL'
        self.aux = False
        self.mul_port_ = self.args.mul_port_
        self.early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        if self.dl_type == 'DL':
            self.criterion = self._select_criterion(type='reward', aux=self.aux,
                                                    mul_port_=self.args.mul_port_)
        elif self.dl_type == 'DL_2':
            self.criterion = self._select_criterion(type='MSE')

        self.target_entropy = torch.linspace(np.log(self.env.n_asset), 0, self.args.risk_levels)
        self.cached_z_low = None
        self.cached_z_high = None

    def _build_model(self):
        settings = (self.env.state_space,
                    self.env.action_space,
                    False,
                    self.env.use_glob,
                    0,
                    self.args,
                    self.env.trn_p_std)
        net_settings = (None, None, self.args.dim1, self.args.dim2, self.args.dim3, self.args.var1)
        model = CondDiffModel(settings=settings, net_settings=net_settings,
                              std_=False,
                              mul_port_=self.args.mul_port_,
                              risk_levels=self.args.risk_levels)

        if not self.args.is_training and not self.args.test_type == 'label':
            print('Loading model for testing')
            test_model = \
                load_model(model_save_path=self.args.checkpoints, model_name=self.args.id_, map_location=self.args.device)
            model.load_state_dict(test_model)

        return model, None, None

    def _select_optimizer(self):
        model_optim = optim.Adam(
            [{'params': self.model.parameters()}],
            lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, type, aux=False, mul_port_=False):
        criterion = None
        if type == "MSE":
            criterion = nn.MSELoss()
        elif type == "CE":
            criterion = nn.CrossEntropyLoss()
        elif type == "BCE":
            criterion = nn.BCELoss()
        elif type == "reward":
            criterion = lambda portfolio, returns: -torch.sum(portfolio * returns, dim=1)
        aux_criterion = lambda portfolio, risks: torch.mean(torch.sum(torch.abs(portfolio) * risks, dim=1))
        return criterion, aux_criterion if aux or mul_port_ else None

    def unpack_data(self, data):
        (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_y_return, batch_glob_x, batch_x_corr, batch_x_beta,
         batch_y_risk_level, batch_x_opt_port) = data
        if self.dl_type == "DL":
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = None
            batch_y_mark = None
            batch_y_return = batch_y_return.float().to(self.device)
            batch_glob_x = batch_glob_x.float().to(self.device)
            batch_x_corr = batch_x_corr.float().to(self.device)
            batch_x_beta = batch_x_beta.float().to(self.device)
            batch_y_risk_level = batch_y_risk_level.int().to(self.device)
            batch_x_opt_port = batch_x_opt_port.float().to(self.device)
            x_dec = None
            n = batch_x.size(0)

        elif self.dl_type == "DL_2":
            batch_x = batch_x.flatten(0, 1).float().to(self.device)  # (32, N, 256, 4) -> (32 * N, 256, 4)
            batch_y = batch_y.flatten(0, 1).float().to(self.device)  # (32, N, 56,  3) -> (32 * N, 56,  3)
            batch_x_mark = batch_x_mark.float().to(self.device)  # (32,    256, 3)
            batch_y_mark = batch_y_mark.float().to(self.device)  # (32,    56,  3)
            batch_glob_x = None
            batch_y_return = batch_y_return.float().to(self.device)

            # (32, 256, 3) -> (N, 32, 256, 3) -> (32, N, 256, 3) -> (32 * N, 256, 3)
            batch_x_mark = batch_x_mark.repeat(self.env.n_asset, 1, 1, 1).transpose(1, 0).flatten(0, 1)
            # (32, 56, 3) -> (N, 32, 56, 3) -> (32, N, 56, 3) -> (32 * N, 56, 3)
            batch_y_mark = batch_y_mark.repeat(self.env.n_asset, 1, 1, 1).transpose(1, 0).flatten(0, 1)

            # decoder input
            x_dec = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()  # (B, pred_len, :)
            x_dec = torch.cat([batch_y[:, :self.args.label_len, :], x_dec], dim=1).float().to(self.device)

            n = batch_x.size(0) // self.env.n_asset  # batch size

        return (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_y_return, batch_glob_x, batch_x_corr, x_dec, n,
                batch_x_beta, batch_y_risk_level, batch_x_opt_port)

    def run_one_epoch(self, data_loader, criterion, model_optim=None, run_type='train', epoch=0):
        if run_type == 'train':
            self.model.train()
        elif run_type == 'vali':
            self.model.eval()
        total_losses = []
        pred_mean_losses = []
        pred_mean_sec_losses = []
        aux_mean_losses = []
        z_bayes_losses = []
        eps_bayes_losses = []
        eps_losses = []
        ret_losses = []
        risk_losses = []

        for i, data in enumerate(data_loader):
            (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_y_return, batch_glob_x, batch_x_corr, x_dec, n,
             batch_x_beta, batch_y_risk_level, batch_x_opt_port) = self.unpack_data(data)

            # Create conditional representation \hat{y} for y^T
            (predictive_std_mse_loss, y_0_hat_std_batch, cond_model_input_, eps_loss, neg_log_loss, eps_bayes_loss,
             bayes_loss, z_bayes_loss) = (
                0, None, None, torch.tensor(0), torch.tensor(0), torch.tensor(0.0, device=batch_x.device),
                torch.tensor(0), torch.tensor(0))

            # Sample t and risk uniformly (Version2)
            batch_risk = torch.randint(low=0, high=self.args.risk_levels, size=(n,)).to(self.device) \
                if self.args.risk_levels else torch.tensor(0)
            t = torch.randint(low=0, high=self.args.timesteps, size=(n,)).to(self.device)

            batch_y_return = batch_y_return[torch.arange(batch_x.shape[0]), batch_risk] \
                if self.args.risk_levels and len(batch_y_return.shape) == 3 \
                else batch_y_return
            e = self.env.trn_p_std * torch.randn_like(batch_y_return).to(self.device)  # epsilons

            y_0_hat_batch, risk_loss, aux_ret_loss, aux_mean_loss, pred_mean_sec_loss, pred_mean_loss = \
                [None, *[torch.tensor([0], device=self.device)] * 5]

            # sample y_t ~ q(y_t| y_0, \hat{y}, y_t)
            y_t_batch = q_sample(
                batch_y_return, y_0_hat_batch,
                self.model.alphas_bar_sqrt, self.model.one_minus_alphas_bar_sqrt, t,
                noise=e
            )

            eps_pred, aux_out, _ = self.model(x=batch_x, y_t=y_t_batch, g=batch_glob_x, risk=batch_risk, t=t)
            aux_ret_loss = torch.mean(criterion[0](aux_out, batch_y))

            eps_bayes_loss_lst = []
            eps_bayes_loss /= len(eps_bayes_loss_lst)
            bayes_loss = z_bayes_loss + eps_bayes_loss

            if self.args.pred_type == 'eps':
                eps_loss = (e - eps_pred).square().mean()
            elif self.args.pred_type == 'y_0':
                eps_loss = (batch_y_return - eps_pred).square().mean()

            loss = eps_loss + self.args.lambda_ * aux_ret_loss

            pred_mean_losses.append(pred_mean_loss.item())
            pred_mean_sec_losses.append(pred_mean_sec_loss.item())
            aux_mean_losses.append(aux_mean_loss.item())
            total_losses.append(loss.item())
            eps_bayes_losses.append(eps_bayes_loss.item())
            z_bayes_losses.append(z_bayes_loss.item())
            eps_losses.append(eps_loss.item())
            ret_losses.append(aux_ret_loss.item())
            risk_losses.append(risk_loss.item())

            if (i + 1) % 100 == 0 and run_type == 'train':
                print(f"\titers: {i + 1}, "
                      f"epoch: {epoch + 1} | "
                      f"loss: {loss.item():.7f} | "
                      f"eps_loss: {eps_loss.item():.5f} | "
                      f"bayes_loss: {bayes_loss.item():.4f} | "
                      f"z_bayes_loss: {z_bayes_loss.item():.4f} | "
                      f"eps_bayes_loss: {eps_bayes_loss.item():.4f}")

            if run_type == 'train':
                model_optim.zero_grad()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                loss.backward()
                model_optim.step()

        return (np.average(total_losses),
                np.average(pred_mean_losses),
                np.average(pred_mean_sec_losses),
                np.average(aux_mean_losses),
                np.average(z_bayes_losses),
                np.average(eps_bayes_losses),
                np.average(eps_losses),
                np.average(ret_losses),
                np.average(risk_losses),)

    def train(self):
        path = os.path.join(self.args.checkpoints, self.args.id_ + ".pth")
        if not os.path.exists(self.args.checkpoints):
            os.makedirs(self.args.checkpoints)
        model_optim = self._select_optimizer()

        for epoch in range(self.args.train_epochs):
            epoch_time = time.time()
            (train_total_losses, train_pred_losses, train_pred_mean_sec_losses, train_aux_mean_losses,
             train_z_bayes_losses, train_eps_bayes_losses, train_eps_losses, train_ret_losses, train_risk_losses) = (
                self.run_one_epoch(run_type='train',
                                   epoch=epoch,
                                   data_loader=self.env.train_loader,
                                   model_optim=model_optim,
                                   criterion=self.criterion))

            with torch.no_grad():
                (vali_loss, vali_pred_losses, val_pred_mean_sec_losses, vali_aux_mean_losses,
                 vali_z_bayes_losses, vali_eps_bayes_losses, vali_eps_losses, vali_ret_losses, vali_risk_losses) = (
                    self.run_one_epoch(data_loader=self.env.vali_loader, criterion=self.criterion, run_type='vali'))
                (test_loss, test_pred_losses, test_pred_mean_sec_losses, test_aux_mean_losses,
                 test_z_bayes_losses, test_eps_bayes_losses, test_eps_losses, test_ret_losses, test_risk_losses) = (
                    self.run_one_epoch(data_loader=self.env.test_loader, criterion=self.criterion, run_type='vali'))

            real_ = lambda x: np.sqrt(x) * self.env.n_asset
            to_ = lambda x: x * 1e2
            PM = f"[PM: {-to_(train_pred_losses):.3f}|{-to_(vali_pred_losses):.3f}|{-to_(test_pred_losses):.3f}] -- "
            Rets = f"[Ret: {-to_(train_ret_losses):.3f}|{-to_(vali_ret_losses):.3f}|{-to_(test_ret_losses):.3f}] -- " \
                if self.args.risk_levels else PM
            print(f"Epoch: {(epoch + 1):<70}"
                  f"[Total losses: {train_total_losses:.2f}|{vali_loss:.2f}|{test_loss:.2f}] -- "
                  f"[Eps: {real_(train_eps_losses):.3f}|{real_(vali_eps_losses):.3f}|{real_(test_eps_losses):.3f}] -- "
                  + Rets +
                  f"cost time: {(time.time() - epoch_time):.2f} -- ")

            tracking_loss = vali_eps_losses
            self.early_stopping(tracking_loss,
                                {'model': self.model.state_dict()},
                                path)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break

        # Substitute the models with the best validation model before the test
        test_model = load_model(model_save_path=self.args.checkpoints, model_name=self.args.id_)
        self.model.load_state_dict(test_model)
        return self.model

    @torch.no_grad()
    def run_test(self,
                 run_type: str = 'default',
                 risk_guidance: bool = True,
                 opt_y_risk: bool = False,
                 all_timestep: bool = False):
        risk_levels = self.args.risk_levels
        rank_score = 0.0

        if risk_levels:
            act_mode_ = self.args.mode
            for mode_ in ['denoising']:  # 'label', 'denoising'
                if mode_ == 'label':
                    self.args.mode = 'label'
                elif mode_ == 'denoising':
                    self.args.mode = act_mode_

                print(
                    f"============================ Start evaluate in {self.args.mode} mode ============================")
                res, entropy_stats, denoising_preds_all_risks, denoising_preds_all_ts_risks = [], [], [], []
                if self.args.mode == 'label':
                    for risk_level in range(risk_levels):
                        print(f"Label for risk level: {risk_level}")
                        res_ = self.test(risk_level=risk_level)
                        entropy_stats.append(res_[-3])
                        res.append(res_[:-3])

                elif run_type == 'default':
                    for risk_level in list(range(risk_levels)):  # [-999] +
                        print(f"Risk level: {risk_level}")
                        if opt_y_risk:
                            print(f"Create hypothetically optimal risk portfolio")
                        res_ = self.test(risk_level=risk_level, risk_guidance=risk_guidance, opt_y_risk=opt_y_risk)
                        denoising_preds_all_ts_risks.append(res_[-1])
                        denoising_preds_all_risks.append(res_[-2])
                        entropy_stats.append(res_[-3])
                        res.append(res_[:-3])

                res = np.array(res)
                entropy_stats = np.array(entropy_stats)
                str_fmt = lambda x: " | ".join([f"{x_:.3f}" for x_ in x])
                print(f"--------------------------------------------------")
                print(f"MDDs = {str_fmt(res[:, 3])}\n"
                      f"AVols = {str_fmt(res[:, 4])}\n"
                      f"y_hat_ents = {str_fmt(entropy_stats[:, 0])}\n"
                      f"pred_ents = {str_fmt(entropy_stats[:, 1])}")
                print(f"--------------------------------------------------")

                denoising_preds = None
                if len(denoising_preds_all_risks) > 0 and len(denoising_preds_all_risks[0]) > 0:

                    if all_timestep:
                        # Post Construction Results
                        post_res = np.zeros((risk_levels, self.args.timesteps + 1, 7, self.args.num_samples))
                        # [risk_levels, num_test_steps, timesteps + 1, num_samples, N]
                        denoising_preds_all_ts_risks = torch.stack(denoising_preds_all_ts_risks)
                        denoising_preds = denoising_preds_all_ts_risks

                        # (timesteps * risk_levels * num_samples) loops
                        for _t in range(self.args.timesteps + 1):
                            for _r in range(risk_levels):
                                for _s in range(self.args.num_samples):
                                    pv, arr, asr, mdd, avol, cr, sor = compute_metrics(env=self.env,
                                                                                       actions=denoising_preds_all_ts_risks[_r, :, _t, _s, :].numpy())
                                    post_res[_r, _t, :, _s] = [pv, arr, asr, mdd, avol, cr, sor]
                        print(f"post_res.shape = {post_res.shape}")

                    else:
                        from collections import defaultdict
                        # Post Construction Mean Results
                        post_res = np.zeros((risk_levels, 7, self.args.num_samples))
                        # [risk_levels, num_test_steps, num_samples, N]
                        denoising_preds_all_risks = torch.stack(denoising_preds_all_risks)
                        denoising_preds = denoising_preds_all_risks
                        # For each path, there are "risk_levels" AVols
                        path_risk_avol_dict = defaultdict(list)

                        for _r in range(risk_levels):
                            for _s in range(self.args.num_samples):
                                pv, arr, asr, mdd, avol, cr, sor = compute_metrics(env=self.env,
                                                                                   actions=denoising_preds_all_risks[_r, :, _s, :].numpy())
                                path_risk_avol_dict[_s].append(avol)
                                post_res[_r, :, _s] = [pv, arr, asr, mdd, avol, cr, sor]

                        rank_score_method = 'spearman'
                        rank_score = compute_rank_score(path_risk_avol_dict=path_risk_avol_dict, risk_levels=risk_levels,
                                                        method=rank_score_method)
                        pv_, arr_, asr_, mdd_, avol_, cr_, sor_ = post_res.mean(axis=(0, 2))
                        print(f"Mean Rank Score ({rank_score_method}): {rank_score:.3f}")
                        print(f"Post Construction Mean Res: {pv_, arr_, asr_, mdd_, avol_, cr_, sor_}")

            return res.mean(axis=0), rank_score, post_res, denoising_preds

        else:
            return self.test()[:-2], rank_score

    @torch.no_grad()
    def test(self, risk_level=-1, risk_guidance=False, opt_y_risk=False):
        denoising_preds_all = []
        denoising_preds_all_ts = []
        denoising_preds = []
        denoising_trues = []
        entropy_stats = [0, 0, 0]

        minibatch_sample_start = time.time()
        self.model.eval()

        print(f"Generating portfolios ... [mode: {self.args.mode}]")
        if self.args.mode == 'label':
            for i, data in enumerate(self.env.test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_y_return, batch_glob_x, batch_x_corr, x_dec, n, \
                    batch_x_beta, batch_y_risk_level, batch_x_opt_port = self.unpack_data(data)
                batch_risk = torch.tensor([risk_level] * n).to(self.device)
                batch_y_return = batch_y_return[torch.arange(batch_x.shape[0]), batch_risk] \
                    if self.args.risk_levels and len(batch_y_return.shape) == 3 \
                    else batch_y_return
                denoising_preds.append(batch_y_return)

        else:
            tot_time_min = 0
            (mean_eps_stds, pred_mean_losses, ret_losses, p_var_lst, beta_mean_lst, mean_y_0_hat_entropy_lst,
             mean_pred_entropy_lst) = [], [], [], [], [], [], []
            for i, data in enumerate(self.env.test_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark, batch_y_return, batch_glob_x, batch_x_corr, x_dec, n, \
                    batch_x_beta, batch_y_risk_level, batch_x_opt_port = self.unpack_data(data)

                B, F, N, T = batch_x.shape
                y_0_hat_batch, risk_loss, ret_loss, aux_mean_loss, pred_mean_sec_loss, pred_mean_loss = \
                    [None, *[torch.tensor([0], device=self.device)] * 5]

                # Reverse risk level for the optimal construction
                batch_y_risk_level = torch.abs(batch_y_risk_level - 4)

                if opt_y_risk:
                    batch_risk = torch.tensor(batch_y_risk_level.cpu(), dtype=torch.int64, device=self.device)
                else:
                    batch_risk = torch.tensor([risk_level] * n, device=self.device)

                mean_y_0_hat_entropy_lst.append(0)
                pred_mean_losses.append(pred_mean_loss.item())
                ret_losses.append(ret_loss.item())
                num_samples = self.args.num_samples
                y_0_hat_tile = None
                y_0_hat_std_tile = None

                timesteps = self.args.timesteps
                risk_weight, temperature = None, None
                if risk_level != -1 and risk_guidance:
                    risk_levels = self.args.risk_levels
                    risk_weight = np.linspace(10, 0, risk_levels)
                    if risk_level == -999 or opt_y_risk:  # Use the foresight risk levels (in manager's perspective)
                        risk_weight = risk_weight[batch_y_risk_level.cpu()].tolist()
                    else:
                        risk_weight = risk_weight[risk_level].tolist()

                cur_y = self.env.trn_p_std * torch.randn(B * num_samples, N).to(self.device)  # z ~ N(0, trn_p_std ** 2 I)
                y_tile_seq, mean_eps_std, p_var_sum, beta_t_mean = p_sample_loop(self.model, y_0_hat_tile,
                                                                                 timesteps,
                                                                                 self.model.alphas,
                                                                                 self.model.one_minus_alphas_bar_sqrt,
                                                                                 y_0_hat_std=y_0_hat_std_tile,
                                                                                 x_corr=batch_x_corr,
                                                                                 x_beta=batch_x_beta,
                                                                                 pred_type=self.args.pred_type,
                                                                                 risk_weight=risk_weight,
                                                                                 trn_p_std=self.env.trn_p_std,
                                                                                 temperature=temperature,
                                                                                 batch_risk=batch_risk,
                                                                                 cur_y=cur_y,
                                                                                 batch_x=batch_x,
                                                                                 batch_glob_x=batch_glob_x,
                                                                                 args=self.args)

                if p_var_sum is not None:
                    p_var_lst.append(p_var_sum)
                if beta_t_mean is not None:
                    beta_mean_lst.append(beta_t_mean)

                pred = y_tile_seq[timesteps]. \
                    reshape(self.args.test_batch_size, num_samples, -1)  # (B, num_samples, N)

                # append [B, num_samples, N] but no need construction since, y_0_hat is portfolio
                denoising_preds_all.append(pred.detach().cpu())
                # [num_steps + 1, B, num_samples, N] -> [B, num_steps + 1, num_samples, N]
                denoising_preds_all_ts.append(construct_portfolio_(torch.stack(y_tile_seq, dim=0).detach().cpu().
                                                                   reshape(-1, N)).reshape(timesteps + 1, -1, num_samples, N).transpose(0, 1))
                pred = pred.mean(dim=1)
                pred = construct_portfolio_(pred.detach().cpu(),
                                            temperature=temperature.detach().cpu() if temperature is not None else torch.tensor(1.0),
                                            tt_as=True,
                                            max_portfolio=self.args.max_portfolio)

                mean_pred_entropy_lst.append(compute_entropy(pred).mean().detach().cpu().numpy())
                true = batch_y.detach().cpu()

                denoising_preds.append(pred)
                denoising_trues.append(true)

                consumed_time_min = (time.time() - minibatch_sample_start) / 60
                tot_time_min += consumed_time_min
                if i % 5 == 0 and i != 0:
                    print('Testing: %d/%d cost time: %f min' % (
                        i, len(self.env.test_loader), consumed_time_min))
                    minibatch_sample_start = time.time()
            print(f"Total cost time: {tot_time_min:.3f} min")
            entropy_stats = [np.mean(mean_y_0_hat_entropy_lst),
                             np.mean(mean_pred_entropy_lst),
                             self.target_entropy[risk_level]]
            print(f"Mean p_var_sum: {np.mean(p_var_lst):.5f}, "
                  f"Mean beta: {np.mean(beta_mean_lst):.5f}, "
                  f"Mean y_hat|pred|target entropy: {entropy_stats[0]:.3f}|{entropy_stats[1]:.3f}|{entropy_stats[2]:.3f}")

        if len(denoising_preds_all) > 0:  # [B, num_samples, N] * num_test_batch
            denoising_preds_all = torch.vstack(denoising_preds_all)  # [B * num_test_batch, num_samples, N]
            denoising_preds_all_ts = torch.vstack(denoising_preds_all_ts)

        preds = torch.stack(denoising_preds)  # (num_test_batch, B, N)

        print(f"Start evaluating ...")
        print(f"period: [{self.args.date_from} - {self.args.date_to}]\n"
              f"number of executions: {int(len(self.env.test_loader) * self.args.test_batch_size / self.env.reb_freq)}")

        # (num_test_batch * B, N) = (num trading time steps, N)
        actions = preds.reshape(-1, self.env.n_asset).cpu().numpy()
        pv, arr, asr, mdd, avol, cr, sor = compute_metrics(env=self.env, actions=actions)

        print(f"PV: {(pv * 100):.2f}%\t"
              f"ARR {arr:.2f}\t"
              f"ASR: {asr:.2f}\t"
              f"MDD: {mdd:.2f}\t"
              f"AVol: {avol:.5f}\t"
              f"CR: {cr:.2f}\t"
              f"SOR: {sor:.2f}")

        return pv, arr, asr, mdd, avol, cr, sor, entropy_stats, denoising_preds_all, denoising_preds_all_ts


def compute_metrics(env, actions: np.ndarray, freq_adjusted: bool = False):
    if np.isnan(actions).any():
        raise ValueError(f"NaN value detected in actions")

    env.set_mode(mode='test')
    env.reset()
    done, score, a_hist, r_hist = False, 0.0, [], []
    reb_freq = 1 if freq_adjusted else env.reb_freq
    for a in actions[::reb_freq, :]:
        _, r, done, *_ = env.step(np.expand_dims(a, axis=0))
        score += r
        a_hist.append(a)
        r_hist.append(r)

        if done:
            break

    a_hist = np.array(a_hist)
    res = {'type_': env.type_, 'rewards': np.array(r_hist), 'reb_freq': env.reb_freq}
    pv = experiment.calc_pv(rewards=np.array(r_hist))
    arr, asr, mdd, avol, cr, sor = experiment.calc_all_metrics(res)
    return pv, arr, asr, mdd, avol, cr, sor

def compute_rank_score(path_risk_avol_dict: dict, risk_levels: int, method: str = 'spearman'):
    from scipy.stats import rankdata
    scores = []
    for _s, _risks in path_risk_avol_dict.items():  # For each sample path
        _risks = np.array(_risks)

        # sample path rank score
        if method == 'inversion':
            scores.append(((_risks[1:] - _risks[:-1]) > 0).sum() / (risk_levels - 1))
        elif method == 'spearman':  # \sum_i d_i ** 2 = \sum_i (i - (n-i+1)) ** 2, max sum = n(n^2-1)/3
            scores.append(1 - (3 * ((rankdata(_risks) - np.arange(1, risk_levels+1)) ** 2).sum()) / (risk_levels ** 3 - risk_levels))
    return np.mean(scores)
