import torch
import torch.nn as nn


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        import math
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps, risk_levels=None):
        super(ConditionalLinear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.act = nn.ReLU()
        self.embed = SinusoidalPositionEmbeddings(dim=num_out)
        self.embed_lin = nn.Linear(in_features=num_out, out_features=num_out)
        self.risk_levels = risk_levels

        if self.risk_levels:
            self.risk_embed = nn.Parameter(torch.ones(risk_levels, num_out))


    def forward(self, x, t, risk_level=None):
        out = self.lin(x)  # (B, num_out)
        gamma = self.act(self.embed_lin(self.embed(t)))  # (B, num_out)
        beta = self.risk_embed[[risk_level]]

        num_samples = x.shape[0] // t.shape[0]
        if num_samples != 1:
            gamma = gamma.repeat(num_samples, 1, 1).transpose(0, 1).reshape(-1, self.num_out)
            beta = beta.repeat(num_samples, 1, 1).transpose(0, 1).reshape(-1, self.num_out)

        if self.risk_levels and risk_level is not None:
            gamma = gamma + beta

        return gamma * out


# early stopping scheme for hyperparameter tuning
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=10, delta=0):
        """
        Args:
            patience (int): Number of steps to wait after average improvement is below certain threshold.
                            Default: 10
            delta (float): Minimum change in the monitored quantity to qualify as an improvement;
                           shall be a small positive value.
                           Default: 0
            best_score: value of the best metric on the validation set.
        """
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_cost, epoch, verbose=False):

        score = val_cost

        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score - self.delta:
            self.counter += 1
            if verbose:
                print("EarlyStopping counter: {} out of {}...".format(
                    self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
