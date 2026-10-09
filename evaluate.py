import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Scaled signals, scaling constant YS, train split length, and scaled step H
from data import ys, vs, us, NTRAIN, YS, H
from model import AccelerationNet
from integrator import rk4_step
from train import train


def free_run(model):
    """
    One free run over ALL 4096 samples, starting at sample 0 with the measured position and
    finite-difference velocity. The rollout reads only u, never y. The state is carried
    straight from the training samples into the test samples.
    Returns (train RMSE [V], test RMSE [V], predicted trajectory [V]).
    """
    model.eval()
    y, v = ys[:1], vs[:1]
    preds = [y]
    with torch.no_grad():
        for t in range(len(ys) - 1):
            y, v = rk4_step(model, y, v, us[t:t + 1], H)
            preds.append(y)
    pred = torch.cat(preds)

    train_rmse = torch.sqrt(torch.mean((ys[:NTRAIN] - pred[:NTRAIN]) ** 2)).item() * YS
    test_rmse = torch.sqrt(torch.mean((ys[NTRAIN:] - pred[NTRAIN:]) ** 2)).item() * YS
    return train_rmse, test_rmse, (pred * YS).numpy()


def train_and_test(act, dim, **settings):
    """Train one model with the given settings, reload its best checkpoint, and free-run it."""
    train(hidden_dim=dim, activation=act, save_path="tmp_model.pt", verbose=False, **settings)
    model = AccelerationNet(hidden_dim=dim, activation=act)
    model.load_state_dict(torch.load("tmp_model.pt"))
    return free_run(model)


def evaluate():
    # Zero-prediction baseline: RMS of the test output
    baseline_rmse = (torch.mean(ys[NTRAIN:] ** 2).sqrt() * YS).item()
    y_true = (ys * YS).numpy()

    print("=" * 70)
    print("   SILVERBOX NEURAL-ODE EVALUATION (free-run RMSE in Volts)")
    print("=" * 70)
    print(f"Zero-prediction test baseline RMSE: {baseline_rmse:.4f} V\n")

    # ---- Part 1: architectures (all trained with the same default settings) ----
    print("PART 1: ARCHITECTURES")
    print("-" * 70)
    best_label, best_test, best_pred, best_train = None, float("inf"), None, None   # best over ALL runs (Parts 1 and 2)
    arch_preds = {}                                               # free runs of the six architectures
    for act in (nn.Tanh, nn.SiLU, nn.GELU):
        for dim in (32, 64):
            label = f"{act.__name__}_dim{dim}"
            train_rmse, test_rmse, pred = train_and_test(act, dim)
            print(f"{label:<12} | Train RMSE: {train_rmse:.4f} V | Test RMSE: {test_rmse:.4f} V")
            arch_preds[label] = pred
            if test_rmse < best_test:
                best_label, best_test, best_pred, best_train = label, test_rmse, pred, train_rmse
    print(f"\nBest architecture (default training): {best_label} with Test RMSE {best_test:.4f} V")

    # ---- Part 2: training hyperparameters, one change at a time, for EVERY architecture ----
    print("\nPART 2: HYPERPARAMETERS (every architecture; one change from the base run)")
    runs = [
        ("base (Adam, lr 0.002, 20 epochs, rollout 20)", {}),
        ("rollout length = 10", {"seq_len": 10}),
        ("rollout length = 40", {"seq_len": 40}),
        ("learning rate = 0.001", {"lr": 0.001}),
        ("learning rate = 0.005", {"lr": 0.005}),
        ("epochs = 10", {"epochs": 10}),
        ("epochs = 60", {"epochs": 60}),
        ("optimizer = AdamW", {"optimizer_name": "adamw"}),
        ("optimizer = RMSprop", {"optimizer_name": "rmsprop"}),
    ]
    for act in (nn.Tanh, nn.SiLU, nn.GELU):
        for dim in (32, 64):
            label = f"{act.__name__}_dim{dim}"
            print("-" * 70)
            print(label)
            for name, change in runs:
                train_rmse, test_rmse, pred = train_and_test(act, dim, **change)
                print(f"{name:<46} | Train: {train_rmse:.4f} V | Test: {test_rmse:.4f} V")
                if test_rmse < best_test:
                    best_label, best_test, best_pred, best_train = f"{label}, {name}", test_rmse, pred, train_rmse
    print("=" * 70)
    print(f"KEPT MODEL (best test RMSE over all runs): {best_label} | Train RMSE {best_train:.4f} V | Test RMSE {best_test:.4f} V")
    print("=" * 70)

    # ---- Figure 2: all six architectures, first 250 test samples, and their absolute errors ----
    n = 250
    t2 = range(n)
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(t2, y_true[NTRAIN:NTRAIN + n], "k-", linewidth=2, label="Measured")
    for label, pred in arch_preds.items():
        ax[0].plot(t2, pred[NTRAIN:NTRAIN + n], "--", alpha=0.8, label=label)
        ax[1].plot(t2, abs(y_true[NTRAIN:NTRAIN + n] - pred[NTRAIN:NTRAIN + n]), alpha=0.8, label=label)
    ax[0].set_ylabel("Output voltage (V)"); ax[0].set_title("Free run of all six architectures (first 250 test samples)")
    ax[1].set_ylabel("|measured - predicted| (V)"); ax[1].set_xlabel("Test sample"); ax[1].set_title("Absolute error")
    ax[0].legend(fontsize=7); ax[0].grid(True); ax[1].grid(True)
    plt.tight_layout(); plt.savefig("architecture_comparison.png", dpi=300); plt.close()

    # ---- Figure 1: kept model (best over all runs), free run vs measured on the test segment ----, free run vs measured on the test segment ----
    t = range(NTRAIN, len(ys))
    plt.figure(figsize=(10, 4))
    plt.plot(t, y_true[NTRAIN:], "k-", label="Measured (Ground Truth)", linewidth=1.8)
    plt.plot(t, best_pred[NTRAIN:], "r--", label=f"Free-Run Prediction ({best_label})", alpha=0.85)
    plt.title(f"Silverbox Free-Run Prediction on Test Segment ({best_label})")
    plt.xlabel("Sample Index (t)")
    plt.ylabel("Output Voltage (V)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("test_prediction.png", dpi=300)
    print("Plots saved: test_prediction.png (kept model), architecture_comparison.png.")


if __name__ == "__main__":
    evaluate()