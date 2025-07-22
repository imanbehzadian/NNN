import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

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

# Function: impulse response analysis with flexible inputs
def impulse_response_analysis(model, X_t, media_spend, std_rg=None, raw_increase=None,
                              std_multiplier=1.0, channel_idx=None, geos=None, t=0,
                              D=16, eps=1e-8, device=None, media_channels=None, weeks=None):

    X_t = torch.tensor(X_t, device=device)
    N, T, C, _ = X_t.shape
    _, T, C_media = media_spend.shape
    if geos is None:
        geos = list(range(N))
    if media_channels is None:
        media_channels = [f"Channel {i}" for i in range(C)]
    if weeks is None:
        weeks = list(range(t, T))

    delta_pct_dict = {}
    plot_data = []

    with torch.no_grad():
        Y_base = model(X_t, torch.tensor(geos, device=device))
    sales_base = torch.expm1(Y_base[..., 0]).cpu().numpy()

    for j in range(C_media):
        X_scen = X_t.clone()

        for g in geos:
            base = media_spend[g, t, j]
            inc = raw_increase if raw_increase is not None else std_rg[g, j] * std_multiplier
            s_prime = base + inc

            D = X_t.shape[-1]  # ensures alignment
            v_val = torch.log1p(torch.tensor(s_prime, device=device)).unsqueeze(0)  # shape [1]
            V = v_val.expand(D)  # now [D], not [D * D]
            norm = torch.norm(V)
            E = (V / (norm + eps)) * torch.log1p(norm)
            X_scen[g, t, j, :] = E



        with torch.no_grad():
            Y_imp = model(X_scen, torch.tensor(geos, device=device))
        sales_imp = torch.expm1(Y_imp[..., 0]).cpu().numpy()

        total_base = sales_base.sum(axis=0)
        total_imp = sales_imp.sum(axis=0)
        delta = total_imp - total_base
        delta_pct = (delta / (total_base + eps)) * 100

        delta_pct_dict[j] = delta_pct

        plot_data.append({
            "channel": media_channels[j],
            "weeks": weeks,
            "delta_pct": delta_pct.tolist(),
            "label": (
                f"Added raw +{raw_increase:.2f}" if raw_increase is not None else f"Added {std_multiplier:.1f}σ"
            ),
            "summary_3w": delta_pct[t:t+3].sum(),
            "summary_13w": delta_pct[t:t+13].sum(),
        })

    return delta_pct_dict, plot_data




def scenario_planner_per_channel(
    model,
    X,
    channel_multipliers: list,
    geo_idx: int | list[int] = None,
    time_step: int | list[int] = None,
    device: torch.device = DEVICE,
    Num_channels: int = len(MEDIA_CHANNELS)
):
    """
    Simulate “what-if” scenarios by scaling each media channel separately,
    optionally for a single time step or for all time steps.

    Args:
      model               : trained NNNModel (in eval mode, on `device`)
      X                   : np.ndarray or torch.Tensor, shape (N_GEOS, T, C_in, D)
      channel_multipliers : list of length NUM_CHANNELS, one multiplier per channel
      geo_idx             : int, list[int] or None → simulate only that geo(s), or all if None
      time_step           : int, list[int] or None → if None apply to all weeks, else only to that week(s)
      device              : torch.device

    Returns:
      DataFrame with columns [
        "geo_idx",
        "time_step",
        "multipliers",
        "total_sales",
        "avg_search"
      ]
    """
    model.eval()

    if not isinstance(X, torch.Tensor):
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
    else:
        X_t = X.to(device)

    assert len(channel_multipliers) == Num_channels, (
        f"Need {Num_channels} multipliers, got {len(channel_multipliers)}"
    )

    # convert geo_idx and time_step to list
    if geo_idx is None:
        geo_list = list(range(X_t.shape[0]))
    elif isinstance(geo_idx, int):
        geo_list = [geo_idx]
    else:
        geo_list = geo_idx

    if time_step is None:
        time_list = None
    elif isinstance(time_step, int):
        time_list = [time_step]
    else:
        time_list = time_step

    results = []

    for g in geo_list:
        X_base = X_t[g : g + 1].clone()  # shape (1, T, C, D)

        # Apply multipliers
        if time_list is None:
            for j, m in enumerate(channel_multipliers):
                X_base[:, :, j, :] *= m
        else:
            for t in time_list:
                for j, m in enumerate(channel_multipliers):
                    X_base[:, t, j, :] *= m

        with torch.no_grad():
            Y_pred = model(X_base, torch.tensor([g], device=device))
            total_sales = torch.expm1(Y_pred[..., 0]).sum().item()
            avg_search  = torch.expm1(Y_pred[..., 1]).mean().item()

        results.append({
            "geo_idx": g,
            "time_step": time_step if isinstance(time_step, int) or time_step is None else time_list,
            "multipliers": channel_multipliers.copy(),
            "total_sales": total_sales,
            "avg_search": avg_search
        })

    return pd.DataFrame(results)