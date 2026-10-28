"""Prediction layer: a linear model, the NN1-NN5 feed-forward networks of Appendix A.2.1, and a
random-Fourier-feature model (a kernel approximation, used with the epsilon-insensitive SVR loss)."""
import math

import torch
import torch.nn as nn

HIDDEN = [32, 16, 8, 4, 2]  # geometric pyramid rule; NNk uses the first k layers
MODELS = ["Linear", "NN1", "NN2", "NN3", "NN4", "NN5"]


class RFF(nn.Module):
    """phi(x) = sqrt(2/D) cos(x W + b), W ~ N(0, 1/sigma^2): approximates an RBF kernel; linear head on phi.

    sigma follows the median heuristic for features on [-1, 1] (variance 1/3 each): sigma^2 = n_in / 3.
    """

    def __init__(self, n_in, n_features=1024, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("W", torch.randn(n_in, n_features, generator=g) / math.sqrt(n_in / 3))
        self.register_buffer("b", torch.rand(n_features, generator=g) * 2 * math.pi)
        self.head = nn.Linear(n_features, 1)
        self.scale = math.sqrt(2.0 / n_features)

    def forward(self, x):
        return self.head(self.scale * torch.cos(x @ self.W + self.b))


def make_model(name, n_in, dropout=0.05):
    if name == "Linear":
        return nn.Linear(n_in, 1)
    if name == "RFF":
        return RFF(n_in)
    layers, d = [nn.LayerNorm(n_in)], n_in
    for h in HIDDEN[:int(name[2:])]:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(dropout)]
        d = h
    layers.append(nn.Linear(d, 1))
    return nn.Sequential(*layers)
