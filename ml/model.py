"""
The LSTM forecaster itself.

Shape contract:
    input  x: (batch, WINDOW, n_features)   -- 96 steps of [moisture, temp, light]
    output  : (batch, HORIZON)               -- next 24 steps of soil_moisture

We read the whole window with a (stacked) LSTM, take its final hidden
state as a summary of "everything that mattered in the last 24 hours",
and map that summary straight to all 24 future values with one linear
layer (direct multi-output, not autoregressive — see conversation notes
on why that's the right tradeoff at this horizon length).
"""

import torch
import torch.nn as nn


class MoistureLSTM(nn.Module):
    def __init__(self, n_features=3, hidden_size=64, num_layers=2, horizon=24, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            # PyTorch only applies inter-layer dropout when num_layers > 1;
            # it's a no-op (and PyTorch warns) if you pass it with a single layer.
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers, batch, hidden_size) — final hidden state of every
        # stacked layer. We want the LAST layer's, i.e. index -1.
        last_hidden = h_n[-1]  # (batch, hidden_size)
        last_hidden = self.dropout(last_hidden)
        return self.head(last_hidden)  # (batch, horizon)


if __name__ == "__main__":
    model = MoistureLSTM()
    n_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"\nTotal trainable parameters: {n_params:,}")

    # Smoke test: does a dummy batch actually flow through with the right shape?
    dummy_batch = torch.randn(8, 96, 3)  # 8 examples, 96 steps, 3 features
    output = model(dummy_batch)
    print(f"\nInput shape:  {tuple(dummy_batch.shape)}")
    print(f"Output shape: {tuple(output.shape)}  (expected: (8, 24))")
