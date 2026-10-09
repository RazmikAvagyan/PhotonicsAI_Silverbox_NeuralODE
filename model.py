import torch
import torch.nn as nn


class AccelerationNet(nn.Module):
    """3-layer MLP acting as a neural ODE right-hand side to predict acceleration: y'' = f(y, v, u)."""

    def __init__(self, hidden_dim=32, activation=nn.Tanh):
        super().__init__()
        self.layer1 = nn.Linear(3, hidden_dim)  # Map 3 physical states (y, v, u) to hidden feature space
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)   # Process interactions between state features
        self.layer3 = nn.Linear(hidden_dim, 1) # Project features down to scalar acceleration output
        self.act = activation()                           # Smooth activation function for continuous dynamics

    def forward(self, ys, vs, us):
        # Promote 0D scalars to 1D vectors to avoid torch.stack dimension errors
        if ys.dim() == 0:
            ys = ys.unsqueeze(0)
            vs = vs.unsqueeze(0)
            us = us.unsqueeze(0)

        x = torch.stack([ys, vs, us], dim=-1)     # Combine inputs into state vector: [..., 3]
        x = self.act(self.layer1(x))                      # First hidden layer with activation
        x = self.act(self.layer2(x))                      # Second hidden layer with activation
        return self.layer3(x).squeeze(-1)                 # Output scalar acceleration y'': [...]
