import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Constants
N_GEOS = 47
TIME_STEPS = 104  # Number of weeks (2 years)
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.1
HIDDEN_CHANNELS = 64
LOOKBACK_WINDOW = 52

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
        x_flat = x.view(B * T, C, D)
        x_proj = self.input_proj(x_flat).view(B, T, C * D)

        # Apply time attention layers
        out = x_proj
        for attn in self.time_attn_layers:
            out, _ = attn(out, out, out)
            out = self.dropout(out) + out
        
        # Channel feedforward
        for fc1, fc2, fc3 in zip(self.channel_fc1_layers, self.channel_fc2_layers, self.channel_fc3_layers):
            h = torch.relu(fc1(out))
            h = torch.relu(fc2(h))
            out = fc3(h) + out

        # Heads
        sales = self.head_sales(out.view(B, T, C * D)).squeeze(-1)
        search = self.head_search(out.view(B, T, C * D)).squeeze(-1)
        y = torch.stack([sales, search], dim=-1)

        if self.use_geo:
            geo_b = self.geo_bias(geo_idx).unsqueeze(1)
            y = y + geo_b
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
