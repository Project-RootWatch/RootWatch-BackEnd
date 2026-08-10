"""
Trains MoistureLSTM on the windowed dataset, with early stopping on
validation loss so we keep whichever epoch actually generalized best
instead of whatever the final epoch happened to produce.
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

from dataset import build_datasets
from model import MoistureLSTM

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
DATA_DIR = Path(__file__).parent / "data"

BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 10        # stop if val loss hasn't improved in this many epochs
LEARNING_RATE = 1e-3


def make_loader(X, y, batch_size=BATCH_SIZE, shuffle=False):
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader, loss_fn, optimizer=None):
    """One pass over `loader`. Pass optimizer=None for eval (no weight updates)."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss = 0.0
    n_examples = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for X_batch, y_batch in loader:
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(X_batch)
            n_examples += len(X_batch)

    return total_loss / n_examples


def train():
    data = build_datasets()
    X_train, y_train = data["train"]
    X_val, y_val = data["val"]

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader = make_loader(X_val, y_val, shuffle=False)

    model = MoistureLSTM()
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = CHECKPOINT_DIR / "best_model.pt"

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, loss_fn, optimizer)
        val_loss = run_epoch(model, val_loader, loss_fn, optimizer=None)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), best_path)
        else:
            epochs_without_improvement += 1

        flag = " <- best so far" if improved else ""
        print(f"epoch {epoch:3d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}{flag}")

        if epochs_without_improvement >= PATIENCE:
            print(f"\nNo val improvement for {PATIENCE} epochs — stopping early at epoch {epoch}.")
            break

    print(f"\nBest val_loss: {best_val_loss:.4f}  (saved to {best_path})")
    plot_curves(train_losses, val_losses)
    return train_losses, val_losses


def plot_curves(train_losses, val_losses):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(train_losses, label="train loss")
    ax.plot(val_losses, label="val loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss (scaled units)")
    ax.set_title("Training curve")
    ax.legend()
    fig.tight_layout()
    out_path = DATA_DIR / "training_curve.png"
    fig.savefig(out_path, dpi=130)
    print(f"Saved training curve to {out_path}")


if __name__ == "__main__":
    train()
