import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
import pandas as pd

# Constants
N_HEADS = 4 #TODO to delete them to come from hyperparameters
N_LAYERS = 2
DROPOUT = 0.1
HIDDEN_CHANNELS = 64
LOOKBACK_WINDOW = 52
EMBED_DIM    = 256                            # your embedding dimension
N_GEOS = 47 #TODO to delete them to come from input data
TIME_STEPS = 104  # Number of weeks (2 years)
MEDIA_CHANNELS = ["TV","Radio","Search","Display"]  # your channel names
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NNNModel(nn.Module):
    def __init__(
        self,
        n_channels,
        embed_dim,
        n_heads=N_HEADS,
        lookback_window=LOOKBACK_WINDOW,
        hidden_channels=HIDDEN_CHANNELS,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
        n_geos=N_GEOS,
        time_steps=TIME_STEPS,
        use_geo=True,
        channel_mixing=True,
        attention_by_channel=True,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.embed_dim = embed_dim
        self.use_geo = use_geo
        self.channel_mixing = channel_mixing
        self.attention_by_channel = attention_by_channel

        # Embedding layers per channel
        self.input_proj = nn.Linear(n_channels, n_channels * embed_dim)
        
        # Positional encoding for time
        self.time_fc1 = nn.Linear(time_steps, time_steps)
        self.time_fc2 = nn.Linear(time_steps, time_steps)

        # Multi-head attention over time or channel
        if attention_by_channel:
            self.time_attn_layers = nn.ModuleList(
                [nn.MultiheadAttention(embed_dim, n_heads, batch_first=True) for _ in range(n_layers)]
            )
        else:
            self.time_attn_layers = nn.ModuleList(
                [nn.MultiheadAttention(n_channels * embed_dim, n_heads, batch_first=True) for _ in range(n_layers)]
            )

        self.dropout = nn.Dropout(dropout)

        # Feedforward layers per channel
        self.channel_fc1_layers = nn.ModuleList([nn.Linear(n_channels, hidden_channels) for _ in range(n_layers)])
        self.channel_fc2_layers = nn.ModuleList([nn.Linear(hidden_channels, hidden_channels) for _ in range(n_layers)])
        self.channel_fc3_layers = nn.ModuleList([nn.Linear(hidden_channels, n_channels) for _ in range(n_layers)])

        # Output heads
        self.head_sales = nn.Sequential(
            nn.Linear(n_channels * embed_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
        )
        self.head_search = nn.Sequential(
            nn.Linear(n_channels * embed_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, 1),
            nn.Softplus(),
        )

        # Geo bias
        self.geo_bias = nn.Embedding(n_geos, 2)

    def forward(self, x, geo_idx):
        B, T, C, D = x.shape
        assert C == self.n_channels and D == self.embed_dim, "Input tensor shape mismatch."

        x_seq = x  # shape (B, T, C, D)
        attn_mask = None
        if self.attn_mask is not None:
            attn_mask = self.attn_mask[:T, :T]

        for i in range(self.n_layers):
            if self.attention_by_channel:
                x_flat = x_seq.permute(0, 2, 1, 3).reshape(B * C, T, D)  # (B*C, T, D)
            else:
                x_flat = x_seq.reshape(B, T, C * D)  # (B, T, C*D)

            out, _ = self.time_attn_layers[i](x_flat, x_flat, x_flat, attn_mask=attn_mask)
            x_flat = x_flat + self.dropout(out)

            if self.attention_by_channel:
                x_time = x_flat.view(B, C, T, D).permute(0, 2, 1, 3)  # (B, T, C, D)
            else:
                x_time = x_flat.view(B, T, C, D)

            if self.channel_mixing:
                x_mix = x_time.permute(0, 1, 3, 2).reshape(-1, C)  # (B*T*d, C)
                x_mix = F.relu(self.channel_fc1_layers[i](x_mix))
                x_mix = F.relu(self.channel_fc2_layers[i](x_mix))
                x_mix = self.channel_fc3_layers[i](x_mix)  # (B*T*d, C) output
                x_mixed = x_mix.view(B, T, D, C).permute(0, 1, 3, 2)  # back to (B, T, C, D)
                x_seq = x_time + self.dropout(x_mixed)  # back to (B, T, C, D)
            else:
                x_seq = x_time  # Skip channel mixing entirely

        x_flattened = x_seq.reshape(B, T, C * D)  # (B, T, C*D)

        # Predict log1p(sales) directly (no Sigmoid cap)
        norm = x_flattened.norm(dim=-1, keepdim=True)  # (B, T, 1)
        unit = x_flattened / (norm + 1e-6)  # (B, T,C*D)
        log_sales = self.head_sales(unit).squeeze(-1)  # (B, T)
        # (we no longer do p*norm or Sigmoid—head itself must learn scale)
        sales = log_sales
        search = self.head_search(x_flattened).squeeze(-1)  # (B, T)
        y = torch.stack([sales, search], dim=-1)  # (B, T, 2)
        if self.use_geo:
            # — §15.9: per‐geo multiplier in log‐space is additive (not multiplicative) —
            # geo_bias holds log(multiplier) for [sales,search]
            geo_bias = self.geo_bias(geo_idx).unsqueeze(1)  # (B, 1, 2)
            y = y + geo_bias
        return y

class GeoTimeSeriesDataset(Dataset):
    def __init__(self, X, Y, seq_length):
        self.X = X[:, :seq_length]
        self.Y = Y[:, :seq_length]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return idx, torch.tensor(self.X[idx]), torch.tensor(self.Y[idx])

# Real animation update function for multi-channel impulse response
import numpy as np

def update(t):
    # Determine plotting window
    start = max(0, t-1)
    end = min(TIME_STEPS-1, t+13)
    weeks = np.arange(start, end+1)

    # For each channel, simulate impulse and update its line
    for j, ln in enumerate(lines):
        # Clone base inputs
        X_scen = X_t.clone()
        # Compute raw spend + 1σ at week t for channel j
        s = media_spend[:, t, j]
        inc = std_rg[:, j]
        s_prime = s + inc
        # Rebuild embedding only for channel j at time t
        for g in range(X_scen.shape[0]):
            v = torch.log1p(torch.tensor(s_prime[g], device=device))
            V = v.repeat(EMBED_DIM)
            norm = torch.norm(V)
            E = (V / (norm + eps)) * torch.log1p(norm)
            X_scen[g, t, j, :] = E

        # Predict under this single-channel impulse
        with torch.no_grad():
            Yi = model(X_scen, full_geo_ids)
        sales_imp = torch.expm1(Yi[..., 0]).cpu().numpy().sum(axis=0)

        # Compute percentage change vs baseline
        delta_pct = (sales_imp - sales_base) / (sales_base + eps) * 100

        # Update line for channel j
        ln.set_data(weeks, delta_pct[start:end+1])

    # Keep x-limits consistent
    for ax in axes:
        ax.set_xlim(start, end)

    # Update subplot titles with frame info
    for idx, ax in enumerate(axes):
        ax.set_title(f"Impulse: {MEDIA_CHANNELS[idx]} at t={t}")

    return lines

def tokeniser(texts: list[str]) -> np.ndarray:
    """
    texts: list of strings
    returns: (len(texts), 256)-array of normalized embeddings
    """
    emb_raw = _embed_batch(texts)                     # (n,256)
    return (emb_raw - global_mean) / global_std       # normalize per-dimension





