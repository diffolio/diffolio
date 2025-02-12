import math
import torch


def make_beta_schedule(schedule="linear", num_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == "linear":
        betas = torch.linspace(start, end, num_timesteps)
    elif schedule == "const":
        betas = end * torch.ones(num_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, num_timesteps) ** 2
    elif schedule == "jsd":
        betas = 1.0 / torch.linspace(num_timesteps, 1, num_timesteps)
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, num_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    elif schedule == "cosine" or schedule == "cosine_reverse":
        max_beta = 0.999
        cosine_s = 0.008
        betas = torch.tensor(
            [min(1 - (math.cos(((i + 1) / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2) / (
                    math.cos((i / num_timesteps + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2), max_beta) for i in
             range(num_timesteps)])
        if schedule == "cosine_reverse":
            betas = betas.flip(0)  # starts at max_beta then decreases fast
    elif schedule == "cosine_anneal":
        betas = torch.tensor(
            [start + 0.5 * (end - start) * (1 - math.cos(t / (num_timesteps - 1) * math.pi)) for t in
             range(num_timesteps)])
    return betas


def extract(input, t, x):  # (alphas, t, y_t)
    shape = x.shape  # (32, N), N is the number of assets
    out = torch.gather(input, 0, t.to(input.device))  # (32)
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)  # (32, 1)
    return out.reshape(*reshape)  # (32, 1)


# Forward functions
def q_sample(y, y_0_hat, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, t, noise=None, predictive_std=None):
    if noise is None:
        noise = torch.randn_like(y).to(y.device)
    sqrt_alpha_bar_t = extract(alphas_bar_sqrt, t, y)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y)
    # q(y_t | y_0, x)
    if y_0_hat is None: # Sample without predicitve model (cdiff setting)
        y_t = sqrt_alpha_bar_t * y + sqrt_one_minus_alpha_bar_t * noise

    elif predictive_std is None:  # (pcdiff setting)
        y_t = sqrt_alpha_bar_t * y + (1 - sqrt_alpha_bar_t) * y_0_hat \
              + sqrt_one_minus_alpha_bar_t * noise

    else:
        y_t = sqrt_alpha_bar_t * y + (1 - sqrt_alpha_bar_t) * y_0_hat \
              + sqrt_one_minus_alpha_bar_t * noise * predictive_std
    return y_t


def p_sample(model, y_t, y_0_hat, t, alphas, one_minus_alphas_bar_sqrt, pred_type,
             y_0_hat_std=None, guidance=False, x_corr=None, x_beta=None, risk_weight=None, trn_p_std=1.0,
             temperature=None, batch_risk=None, batch_x=None, batch_glob_x=None, args=None, det_outs=None):
    def get_alpha_gamma_(step_t: torch.Tensor):
        alpha_t = extract(alphas, step_t, y_t)
        sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, step_t, y_t)
        sqrt_one_minus_alpha_bar_t_m_1 = extract(one_minus_alphas_bar_sqrt, step_t - 1, y_t)
        sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
        sqrt_alpha_bar_t_m_1 = (1 - sqrt_one_minus_alpha_bar_t_m_1.square()).sqrt()
        gamma_0 = (1 - alpha_t) * sqrt_alpha_bar_t_m_1 / (sqrt_one_minus_alpha_bar_t.square())
        gamma_1 = (sqrt_one_minus_alpha_bar_t_m_1.square()) * (alpha_t.sqrt()) / (sqrt_one_minus_alpha_bar_t.square())
        gamma_2 = 1 + (sqrt_alpha_bar_t - 1) * (alpha_t.sqrt() + sqrt_alpha_bar_t_m_1) / (
            sqrt_one_minus_alpha_bar_t.square())
        return (alpha_t, sqrt_one_minus_alpha_bar_t, sqrt_one_minus_alpha_bar_t_m_1, sqrt_alpha_bar_t,
                sqrt_alpha_bar_t_m_1, gamma_0, gamma_1, gamma_2)

    from diff_utils.exp.exp_main import construct_portfolio_
    device = next(model.parameters()).device
    z = trn_p_std * torch.randn_like(y_t)  # if t > 1 else torch.zeros_like(y)
    t = torch.tensor([t]).to(device)

    (alpha_t, sqrt_one_minus_alpha_bar_t, sqrt_one_minus_alpha_bar_t_m_1, sqrt_alpha_bar_t, sqrt_alpha_bar_t_m_1,
     gamma_0, gamma_1, gamma_2) = get_alpha_gamma_(step_t=t)

    p_var_sum_ = None
    eps_std = None
    beta_t_mean = 0

    if guidance and risk_weight is not None:
        with ((torch.enable_grad())):
            model.zero_grad()
            B, N = y_t.shape
            y_t.requires_grad_(True)
            res = model(y_t, y_0_hat, t, risk_level=batch_risk) if y_0_hat is not None else model(x=batch_x,
                                                                                                  y_t=y_t,
                                                                                                  g=batch_glob_x,
                                                                                                  risk=batch_risk,
                                                                                                  t=t,
                                                                                                  det_outs=det_outs)
            res = list(res)
            num_samples = int(B / x_corr.shape[0])
            res[0] = construct_portfolio_(res[0])
            p_t = res[0]  # [B, N]
            bc_x_corr = x_corr.repeat(num_samples, 1, 1, 1).transpose(0, 1).reshape(B, N, N).detach()  # [B, N, N]
            # Portfolio Variance w^T * \Sigma * w = [B, 1, N] * [B, N, N] * [B, N, 1] -> [B, 1, 1]
            risk_t = torch.bmm(torch.bmm(p_t.unsqueeze(1), bc_x_corr), p_t.unsqueeze(2))
            bc_x_beta = x_beta.repeat(num_samples, 1, 1).transpose(0, 1).reshape(B, N).detach()
            beta_t = bc_x_beta * p_t
            beta_t_mean = beta_t.sum(dim=1).mean().detach().cpu()

            sum_ = risk_t.sum()
            p_var_sum_ = sum_.detach().cpu()

            if isinstance(risk_weight, list):
                risk_weight = torch.tensor(risk_weight, device=sum_.device)
                risk_weight = risk_weight.repeat(num_samples, 1).transpose(0, 1).reshape(-1, 1)
            g_grad = risk_weight * torch.autograd.grad(sum_, y_t)[0]
    else:
        res = model(batch_x, y_t, batch_glob_x, batch_risk, t=t, det_outs=det_outs)
        res = list(res)
        res[0] = construct_portfolio_(res[0])

    eps_theta = res[0].to(device).detach() if isinstance(res, tuple) or isinstance(res, list) else res.to(device).detach()
    eps_theta.clamp_(min=-1., max=1.)  # NOTE:: Clamping diff model output (https://github.com/lucidrains/denoising-diffusion-pytorch/blob/4019202f6829fcf3772e770ccd1869c672b4c85c/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L668)

    if pred_type == 'eps':  # y_0 reparameterization
        y_0_reparam = 1 / sqrt_alpha_bar_t * (
                y_t - (1 - sqrt_alpha_bar_t) * y_0_hat - eps_theta * sqrt_one_minus_alpha_bar_t)
    elif pred_type == 'y_0':  # We are predicting initial (y_0) rather than the noise (\epsilon)
        y_0_reparam = eps_theta

    beta_t_hat = (sqrt_one_minus_alpha_bar_t_m_1.square()) / (sqrt_one_minus_alpha_bar_t.square()) * (1 - alpha_t)
    beta_t_hat = beta_t_hat if y_0_hat_std is None else beta_t_hat * y_0_hat_std.square()
    y_t_m_1_hat = ((sqrt_alpha_bar_t_m_1 * (1 - alpha_t) * y_0_reparam +
                alpha_t.sqrt() * sqrt_one_minus_alpha_bar_t_m_1.square() * y_t) /
               sqrt_one_minus_alpha_bar_t.square())

    # posterior variance
    y_t_m_1 = y_t_m_1_hat.to(device) - g_grad + beta_t_hat.sqrt().to(device) * z.to(device) \
        if guidance and risk_weight is not None else (
            y_t_m_1_hat.to(device) + beta_t_hat.sqrt().to(device) * z.to(device))

    return y_t_m_1, eps_std, p_var_sum_, beta_t_mean, res[2]


# Reverse function -- sample y_0 given y_1
def p_sample_t_1to0(model, y_t, y_0_hat, one_minus_alphas_bar_sqrt, pred_type, batch_risk, batch_x, batch_glob_x, args, det_outs):
    device = next(model.parameters()).device
    t = torch.tensor([0]).to(device)  # corresponding to timestep 1 (i.e., t=1 in diffusion models)
    sqrt_one_minus_alpha_bar_t = extract(one_minus_alphas_bar_sqrt, t, y_t)
    sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
    res = model(batch_x, y_t, batch_glob_x, batch_risk, t=t, det_outs=det_outs)
    res = list(res)
    from diff_utils.exp.exp_main import construct_portfolio_
    res[0] = construct_portfolio_(res[0])

    eps_theta = res[0].to(device).detach() if isinstance(res, tuple) or isinstance(res, list) else res.to(device).detach()
    eps_theta.clamp_(min=-1., max=1.)

    y_0_reparam = None
    if pred_type == 'eps':  # y_0 reparameterization
        y_0_reparam = 1 / sqrt_alpha_bar_t * (
                y_t - (1 - sqrt_alpha_bar_t) * y_0_hat - eps_theta * sqrt_one_minus_alpha_bar_t)
    elif pred_type == 'y_0':  # We are predicting initial (y_0) rather than the noise (\epsilon)
        y_0_reparam = eps_theta
    y_t_m_1 = y_0_reparam.to(device)

    return y_t_m_1


def p_sample_loop(model, y_0_hat, n_steps, alphas, one_minus_alphas_bar_sqrt, y_0_hat_std=None, x_corr=None,
                  x_beta=None,
                  pred_type='eps', risk_weight=None, trn_p_std=1.0, temperature=None, batch_risk=None,
                  cur_y=None, batch_x=None, batch_glob_x=None, args=None):
    device = next(model.parameters()).device
    y_p_seq = [cur_y]
    eps_stds = []
    p_var_sum_lst_ = []
    beta_t_mean_lst_ = []
    det_outs = None
    for t in reversed(range(1, n_steps)):  # t from T to 2
        y_t = cur_y
        cur_y, eps_std, p_var_sum_, beta_t_mean, det_outs = p_sample(model, y_t, y_0_hat, t, alphas,
                                                                     one_minus_alphas_bar_sqrt,
                                                                     y_0_hat_std=y_0_hat_std,
                                                                     x_corr=x_corr,
                                                                     x_beta=x_beta,
                                                                     pred_type=pred_type,
                                                                     guidance=True,
                                                                     risk_weight=risk_weight,
                                                                     trn_p_std=trn_p_std,
                                                                     temperature=temperature,
                                                                     batch_risk=batch_risk,
                                                                     batch_x=batch_x,
                                                                     batch_glob_x=batch_glob_x,
                                                                     args=args,
                                                                     det_outs=det_outs
                                                                     )  # y_{t-1}
        y_p_seq.append(cur_y)
        eps_stds.append(eps_std)
        p_var_sum_lst_.append(p_var_sum_)
        beta_t_mean_lst_.append(beta_t_mean)
    y_0 = p_sample_t_1to0(model, y_p_seq[-1], y_0_hat, one_minus_alphas_bar_sqrt, pred_type=pred_type,
                          batch_risk=batch_risk, batch_x=batch_x, batch_glob_x=batch_glob_x, args=args,
                          det_outs=det_outs)
    y_p_seq.append(y_0)
    return y_p_seq, None, p_var_sum_lst_[-1], beta_t_mean_lst_[-1]
