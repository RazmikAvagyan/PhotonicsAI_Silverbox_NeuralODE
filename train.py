import torch
import torch.nn as nn
import torch.optim as optim

# Import windowed DataLoader generator and scaled integration time step H directly from data.py
from data import get_dataloaders, H
from model import AccelerationNet
from integrator import rk4_step


def train(
    hidden_dim=32,
    activation=nn.Tanh,
    batch_size=128,
    seq_len=20,
    epochs=20,
    lr=0.002,
    save_path="model.pt",
    optimizer_name="adam",  # "adam", "adamw" or "rmsprop"
    seed=0,  # fixed seed so a run can be repeated
    verbose=True,  # Set to True to display progress logs during training
):
    torch.manual_seed(seed)

    # Select hardware accelerator (GPU/CUDA) if available, otherwise fallback to CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fetch training DataLoader from data.py
    train_loader = get_dataloaders(seq_len=seq_len, batch_size=batch_size)

    # Instantiate AccelerationNet with chosen activation class, MSE loss, and Adam optimizer
    model = AccelerationNet(hidden_dim=hidden_dim, activation=activation).to(device)
    criterion = nn.MSELoss()
    opt_cls = {"adam": optim.Adam, "adamw": optim.AdamW, "rmsprop": optim.RMSprop}[optimizer_name]
    optimizer = opt_cls(model.parameters(), lr=lr)

    best_loss = float("inf")

    # Main Optimization Loop across training epochs
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for y_b, v_b, u_b in train_loader:
            y_b, v_b, u_b = y_b.to(device), v_b.to(device), u_b.to(device)

            # Set initial trajectory conditions at t = 0 (start of window)
            y_pred, v_pred = y_b[:, 0], v_b[:, 0]
            preds = [y_pred]

            # Integrate forward across sequence length using RK4 solver with step H
            for t in range(seq_len - 1):
                y_pred, v_pred = rk4_step(model, y_pred, v_pred, u_b[:, t], H)
                preds.append(y_pred)

            # Stack predictions into sequence tensor: [batch_size, seq_len]
            y_pred_seq = torch.stack(preds, dim=1)

            # Calculate MSE loss and update model weights via backpropagation
            loss = criterion(y_pred_seq, y_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Save model checkpoint to save_path whenever training loss reaches a new best record
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)

        # Print training loss every 10 epochs and on the final epoch
        if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == epochs):
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.6f}")

    return model


if __name__ == "__main__":
    train()