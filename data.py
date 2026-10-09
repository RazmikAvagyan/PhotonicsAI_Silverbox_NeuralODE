import scipy.io
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

FS = 610.35  # sampling frequency [Hz]
DT = 1.0 / FS  # time between samples [s]
START = 42650  # first index of the excerpt
N = 4096  # length of the excerpt
NTRAIN = 3072  # first 3072 samples = training, last 1024 = test

# Load the MATLAB file
mat_data = scipy.io.loadmat('SNLS80mV.mat')

# Extract raw V1 (input voltage u) and V2 (output voltage y)
u_raw = mat_data['V1'].squeeze()
y_raw = mat_data['V2'].squeeze()

# Extract the excerpt: N samples starting at index START
u_excerpt = u_raw[START:START + N]
y_excerpt = y_raw[START:START + N]

# Center the signals by subtracting their respective means
u_centered = u_excerpt - np.mean(u_excerpt)
y_centered = y_excerpt - np.mean(y_excerpt)

# Split into training and testing sets
# First 3072 samples = training (~5.03 seconds)
# Last 1024 samples = test (~1.68 seconds)
u_train = u_centered[:NTRAIN]
y_train = y_centered[:NTRAIN]

u_test = u_centered[NTRAIN:]
y_test = y_centered[NTRAIN:]

# Velocity of y (a rollout needs a starting velocity, and we only measured y)
v_centered = np.gradient(y_centered, DT)

# In real units the numbers differ a lot in size (y ~ 0.1, velocity ~ 50, acceleration ~ 10000),
# and a neural network learns badly from that. We divide each quantity by its typical size, so that everything is of order 1.
YS = y_train.std()  # typical size of y [V] (training part only)
US = u_train.std()  # typical size of u [V] (training part only)
W0 = v_centered[:NTRAIN].std() / YS  # typical oscillation speed [rad/s] = velocity size / position size
H = W0 * DT  # the sampling step measured in the scaled time unit (1/W0 seconds), RK4 uses H, not DT

# 7. Convert to PyTorch tensors
ys = torch.tensor(y_centered / YS, dtype=torch.float32)  # position
us = torch.tensor(u_centered / US, dtype=torch.float32)  # input
vs = torch.tensor(v_centered / (W0 * YS), dtype=torch.float32)  # velocity


def get_dataloaders(seq_len=20, batch_size=64):
    """
    Slices the training signals into overlapping short trajectory windows
    and wraps them in a PyTorch DataLoader for mini-batch training.
    """
    # Slice only the training portion (first 3072 samples)
    y_tr, v_tr, u_tr = ys[:NTRAIN], vs[:NTRAIN], us[:NTRAIN]

    # Extract overlapping sliding sequence windows of length `seq_len`
    num_windows = len(y_tr) - seq_len + 1
    y_wins = torch.stack([y_tr[i: i + seq_len] for i in range(num_windows)])
    v_wins = torch.stack([v_tr[i: i + seq_len] for i in range(num_windows)])
    u_wins = torch.stack([u_tr[i: i + seq_len] for i in range(num_windows)])

    # Construct PyTorch DataLoader for batch shuffling and optimization
    train_loader = DataLoader(
        TensorDataset(y_wins, v_wins, u_wins),
        batch_size=batch_size,
        shuffle=True
    )
    return train_loader