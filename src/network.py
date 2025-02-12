"""***********************************************************************
Diffusion Models for Risk-Aware Portfolio Optimization

-------------------------------------------------------------------------
File: network.py
- The main neural network modules for conditional diffusion models.

Version: 1.0
***********************************************************************"""


import torch
import torch.nn as nn
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)


def asset_selection(portfolio: torch.Tensor, max_portfolio: int = 0, tot_short_weight: torch.Tensor = None,
                    mask_equal_weight: bool = False):
    num_batch, num_asset = portfolio.size()
    if max_portfolio == 0:
        max_portfolio = num_asset
    num_min_max = max_portfolio // 2

    _, indices = torch.sort(portfolio, dim=1)
    mask = torch.zeros_like(portfolio, dtype=torch.bool)
    top_indices = indices[:, -num_min_max:]
    bottom_indices = indices[:, :num_min_max]
    mask.scatter_(1, top_indices, True)
    mask.scatter_(1, bottom_indices, True)
    masked_portfolio = portfolio * mask.float()

    if tot_short_weight is not None:
        scaling = torch.ones_like(portfolio)
        for b in range(num_batch):
            bottom_indices = indices[b, :num_min_max]
            top_indices = indices[b, -num_min_max:]
            scaling[b, bottom_indices] = tot_short_weight[b]
            scaling[b, top_indices] = 1 - tot_short_weight[b]
        selected_portfolio = masked_portfolio * scaling
    elif mask_equal_weight:
        selected_portfolio = torch.zeros_like(portfolio)
        for b in range(num_batch):
            bottom_indices = indices[b, :num_min_max]
            top_indices = indices[b, -num_min_max:]
            selected_portfolio[b, bottom_indices] = -1 / max_portfolio
            selected_portfolio[b, top_indices] = 1 / max_portfolio
    else:
        selected_portfolio = masked_portfolio

    return selected_portfolio


class ScoringUnit(nn.Module):
    def __init__(self, settings, net_settings, risk_levels):
        super(ScoringUnit, self).__init__()
        self.state_space, self.action_space, self.use_short, self.use_glob, self.max_portfolio, self.args, self.trn_p_std = settings
        self.risk_levels = risk_levels
        self.F, self.T = self.state_space[0], self.state_space[2]
        self.N = self.action_space[0]
        _, _, dim1, dim2, dim3, var1, *_ = net_settings

        self.k_conv = nn.Conv2d(in_channels=self.F, out_channels=self.F, kernel_size=(1, 3), dilation=2, padding='same', bias=False)
        self.q_conv = nn.Conv2d(in_channels=self.F, out_channels=self.F, kernel_size=(1, 3), dilation=2, padding='same', bias=False)
        self.v_conv = nn.Conv2d(in_channels=self.F, out_channels=self.F, kernel_size=(1, 3), dilation=2, padding='same', bias=False)
        self.scale = (self.F * self.T) ** -0.5

        self.drop_out = nn.Dropout(p=0.1)
        self.act = nn.ReLU()

        self.norm1 = nn.LayerNorm(self.F * self.T)
        self.ff_conv1 = nn.Conv2d(in_channels=self.F, out_channels=self.F, kernel_size=(1, 3), dilation=2, padding='same')
        self.ff_conv2 = nn.Conv2d(in_channels=self.F, out_channels=self.F, kernel_size=(1, 3), dilation=2, padding='same')
        self.norm2 = nn.LayerNorm(self.F * self.T)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ff_conv2(self.drop_out(self.act(self.ff_conv1(x))))
        return self.drop_out(x)

    def forward(self, x, risk):
        N_FT = (self.N, self.F * self.T)
        N_F_T = (self.N, self.F, self.T)

        # [B, F, N, T] -> [B, F, N, T]
        _q = self.q_conv(x)
        _k = self.k_conv(x)
        _v = self.v_conv(x)

        # [B, N, F, T] x [B, F, T, N] -> [B, N, N]
        _s = (torch.einsum('abcd,acde->abe', _q.transpose(1, 2), _k.transpose(2, 3))) * self.scale

        # [B, N, N]
        s = F.softmax(_s, dim=2)

        # [B, N, N] x [B, N, F*T] -> [B, N, F*T]
        _h = torch.bmm(s, _v.transpose(1, 2).reshape(-1, *N_FT))

        # [B, N, F*T] -> [B, N, F, T] -> [B, F, N, T]
        _x = self.norm1(x.transpose(1, 2).reshape(-1, *N_FT) + self.drop_out(_h)).reshape(-1, *N_F_T).transpose(1, 2)

        # [B, F, N, T]
        return self.norm2((_x + self._ff_block(_x)).transpose(1, 2).reshape(-1, *N_FT)).reshape(-1, *N_F_T).transpose(1, 2)

class CrossAttention(nn.Module):
    def __init__(self, dim1):
        super(CrossAttention, self).__init__()
        self.k_linear = nn.Linear(in_features=dim1, out_features=dim1, bias=False)
        self.v_linear = nn.Linear(in_features=dim1, out_features=dim1, bias=False)
        self.scale = dim1 ** -0.5
        self.act = nn.LeakyReLU()
        self.dropout= nn.Dropout(0.1)

        self.ff_linear1 = nn.Linear(in_features=dim1, out_features=dim1)
        self.ff_linear2 = nn.Linear(in_features=dim1, out_features=dim1)
        self.norm1 = nn.LayerNorm(dim1)
        self.norm2 = nn.LayerNorm(dim1)

    # feed forward block
    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ff_linear2(self.dropout(self.act(self.ff_linear1(x))))
        return self.dropout(x)

    def forward(self, q, k_v):
        # Perform Cross-Attention
        # [B, N, F'] x [B, F', 1] -> [B, N, 1] -> [B, N]
        s_ = torch.softmax(torch.bmm(self.k_linear(k_v), q.unsqueeze(2)).squeeze(2) * self.scale, dim=1)
        # [B, 1, N] x [B, N, F'] -> [B, 1, F'] -> [B, F']
        out = self.norm1(self.dropout(torch.bmm(s_.unsqueeze(1), self.v_linear(k_v))).squeeze(1) + q)

        # norm[X' + FFN(X')], out = X' = [B, F']
        out = self.norm2(out + self._ff_block(x=out))

        return out

class CondDiffModel(nn.Module):
    def __init__(self, settings, net_settings, classifier_=True, std_=False, mul_port_=False, risk_levels=0):
        super(CondDiffModel, self).__init__()
        self.state_space, self.action_space, self.use_short, self.use_glob, self.max_portfolio, self.args, self.trn_p_std = settings
        self.F, self.T = self.state_space[0], self.state_space[2]
        self.N = self.action_space[0]
        _, _, dim1, dim2, dim3, var1, *_ = net_settings
        self.dim1 = dim1
        self.classifier_ = classifier_
        self.risk_levels = risk_levels
        self.mul_port_ = mul_port_
        self.std_ = std_
        self.num_units = 2

        self.scoring_units = nn.ModuleList(
            [
                ScoringUnit(settings=settings, net_settings=net_settings, risk_levels=self.risk_levels)
                for _ in range(self.num_units)
            ])
        self.spat_out_linear_0 = nn.Linear(in_features=self.F * self.T, out_features=dim1)
        self.glob_loc_linear_out = nn.Linear(in_features=dim1, out_features=dim1)

        # Market Scoring Unit
        self.market_hidden_dim = dim1
        self.lstm = nn.LSTM(input_size=self.F, hidden_size=self.market_hidden_dim, num_layers=2, batch_first=True)
        self.hidden_linear_1 = nn.Linear(in_features=self.market_hidden_dim * 2, out_features=self.market_hidden_dim,
                                         bias=False)
        self.g_linear = nn.Linear(in_features=self.F, out_features=self.market_hidden_dim, bias=False)
        self.hidden_linear_2 = nn.Linear(in_features=self.market_hidden_dim, out_features=1, bias=False)

        self.g_out = nn.Linear(in_features=self.market_hidden_dim, out_features=self.market_hidden_dim)

        if self.mul_port_:
            self.sec_out = nn.Sequential(
                nn.Linear(in_features=self.F * self.T, out_features=self.F * 4, bias=True),
                nn.ReLU(),
                nn.Linear(in_features=self.F * 4, out_features=1, bias=True)
            )

        # Learnable parameters for loss weight (deprecated)
        self.sigs = nn.Parameter(torch.ones(10), requires_grad=True)
        self.act = nn.ReLU()
        self.dropout= nn.Dropout(0.2)

        self.num_timesteps = self.args.timesteps
        self.device = self.args.device

        from diff_utils.diffusion_models.diffusion_utils import make_beta_schedule
        betas = make_beta_schedule(num_timesteps=self.num_timesteps,
                                   start=0.0001,
                                   end=0.02)
        betas = self.betas = betas.float().to(self.device)
        self.betas_sqrt = torch.sqrt(betas)
        alphas = 1.0 - betas
        self.alphas = alphas
        self.one_minus_betas_sqrt = torch.sqrt(alphas)
        alphas_cumprod = alphas.to('cpu').cumprod(dim=0).to(self.device)
        self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=self.device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.posterior_mean_coeff_1 = (
                betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_mean_coeff_2 = (
                torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
        )
        posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.posterior_variance = posterior_variance

        from diff_utils.diffusion_models.model import ConditionalLinear
        self.cond_linear_1 = ConditionalLinear(num_in=self.N, num_out=dim1, n_steps=self.args.timesteps, risk_levels=self.risk_levels)
        self.cond_linear_2 = ConditionalLinear(num_in=dim1, num_out=dim1, n_steps=self.args.timesteps, risk_levels=self.risk_levels)
        self.cond_linear_3 = ConditionalLinear(num_in=dim1, num_out=dim1, n_steps=self.args.timesteps, risk_levels=self.risk_levels)
        self.cond_linear_4 = ConditionalLinear(num_in=dim1, num_out=dim1, n_steps=self.args.timesteps, risk_levels=self.risk_levels)

        self.res_mlp1 = nn.Sequential(
            nn.Linear(in_features=dim1 * 2, out_features=dim1 * 2),
            self.act,
            self.dropout,
            nn.Linear(in_features=dim1 * 2, out_features=dim1),
            self.dropout
        )
        self.res_mlp2 = nn.Sequential(
            nn.Linear(in_features=dim1, out_features=dim1),
            self.act,
            self.dropout
        )
        self.res_out = nn.Linear(in_features=dim1, out_features=self.N)
        self.score_mapping1 = nn.Sequential(
            nn.Linear(in_features=self.dim1, out_features=self.dim1),
            nn.Tanh(),
            nn.Linear(in_features=self.dim1, out_features=1)
        )
        self.aux_linear_out = nn.Linear(in_features=self.dim1, out_features=self.N)

    def agg_N_F_to_F(self, in_: torch.Tensor):
        """
        Aggregate N x F' matrix into single vector form with Attention
        in_ = [B, N, F']
        out_ = [B, N]
        """
        # [B, N, F'] -> [B, N, 1] -> [B, 1, N]
        scores = self.score_mapping1(in_).transpose(1, 2)
        # [B, 1, N]
        alphas = torch.softmax(scores, dim=2)
        # [B, 1, N] * [B, N, F'] -> [B, 1, F'] -> [B, F']
        return torch.bmm(alphas, in_).squeeze(dim=1)

    def forward(self, x, y_t, g, risk, t=None, null_index=None, det_outs=None):
        x = x.contiguous()                                                  # (B, F, N, T)
        g = g.contiguous()                                                  # (B, F, 1, T)

        if y_t is not None:
            y_t = y_t.contiguous()                                          # [B, N]

        if det_outs is not None:  # Deterministic outputs from previous path (for the fast inference)
            out, aux_out = det_outs
        else:
            for scoring_unit in self.scoring_units:
                x = scoring_unit(x, risk)           # (B, F, N, T)

            x = x.transpose(1, 2)                                               # [B, N, F, T]
            x_prime = self.spat_out_linear_0(x.reshape(-1, self.N, self.F * self.T))

            g = g.permute(0, 2, 3, 1).squeeze(1)                                # (B, 1, T, F) -> (B, T, F)
            h_k, _ = self.lstm(g)                                               # (B, T, F') F'= market_hidden_dim
            c = torch.cat([h_k, h_k[:, [-1], :].repeat(1, self.T, 1)], dim=2)   # (B, T, 2*F')
            c = torch.tanh(self.hidden_linear_1(c) + self.g_linear(g))          # (B, T, F')
            alpha_k = F.softmax(self.hidden_linear_2(c), dim=1).transpose(1, 2) # (B, T, 1) -> (B, 1, T)
            hat_h_K = torch.bmm(alpha_k, h_k).squeeze(1)                        # (B, 1, T) * (B, T, F') -> (B, 1, F') -> (B, F')
            g_out = self.g_out(hat_h_K).unsqueeze(1)                            # (B, F') -> (B, F') -> (B, 1, F')

            out_lg = x_prime + g_out                                            # (B, N, F')
            out = self.dropout(self.act(self.glob_loc_linear_out(out_lg)))      # [B, N, F'] = History condition \phi(h)
            out = self.agg_N_F_to_F(out)                                        # [B, N, F'] -> [B, F']
            aux_out = self.aux_linear_out(out)                                  # [B, F'] -> [B, N]

            det_outs = out, aux_out  # Save it for next path

        # Condition risk and timestep
        if t.shape[0] == 1:
            t = torch.ones_like(risk) * t

        y_t_out_0 = self.act(self.cond_linear_1(x=y_t, t=t, risk_level=risk))  # [B, F']
        y_t_out_1 = self.act(self.cond_linear_2(x=y_t_out_0, t=t, risk_level=risk))
        y_t_out_1 = self.act(self.cond_linear_3(x=y_t_out_1, t=t, risk_level=risk)) + y_t_out_0
        y_t_out_1 = self.cond_linear_4(x=y_t_out_1, t=t, risk_level=risk)  # [B, F']

        num_samples = y_t.shape[0] // out.shape[0]
        if num_samples != 1:
            out = out.repeat(num_samples, 1, 1).transpose(0, 1).reshape(-1, self.dim1)

        out_0 = self.res_mlp1(torch.concatenate([out, y_t_out_1], dim=1))
        out = self.res_mlp2(out_0) + out_0
        out = self.res_out(out)

        if not self.classifier_:
            return out

        from diff_utils.exp.exp_main import construct_portfolio_
        return out, construct_portfolio_(aux_out), det_outs
