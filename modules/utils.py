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
def impulse_response_analysis(
    model,
    X: np.ndarray,
    media_spend: np.ndarray,
    std_multiplier: float = 1.0,
    raw_increase: float = None,
    channel_idx: int = None,
    time_step: int = None,
    geo_idx: int = None,
    device: torch.device = DEVICE,
    media_channels: list = MEDIA_CHANNELS
):
    """
    Flexible impulse-response: bump spend by std or raw, plot %Δ sales.

    Args:
      model         : trained NNNModel (eval mode)
      X             : array, shape (G, T, C_in, D)
      media_spend   : array, shape (G, T, C)
      std_multiplier: number of σ to add (unless raw_increase provided)
      raw_increase  : direct spend addition (mutually exclusive with std_multiplier)
      channel_idx   : int or None → which channel (None=all)
      time_step     : int or None → which week (None=0)
      geo_idx       : int or None → which geo (None=all)
      device        : torch.device

    Returns:
      dict mapping channel→delta_pct array of shape (T,)
    """
    # Check exclusivity
    if (raw_increase is not None) and (std_multiplier is not None and std_multiplier != 1.0):
        raise ValueError("Specify either raw_increase or std_multiplier, not both.")

    # Prepare tensors
    X_t = torch.tensor(X, dtype=torch.float32, device=device)
    G, T, C_in, D = X_t.shape
    full_geo_ids = torch.arange(G, device=device)
    model.eval()
    eps = 1e-9

    # Baseline sales
    with torch.no_grad():
        Yb = model(X_t, full_geo_ids)
    sales_base = torch.expm1(Yb[..., 0]).cpu().numpy()

    # Compute std devs of raw spend
    std_rg = np.std(media_spend, axis=1)

    # Determine indices
    geos = [geo_idx] if geo_idx is not None else list(range(G))
    channels = [channel_idx] if channel_idx is not None else list(range(media_spend.shape[2]))
    t = time_step if time_step is not None else 0

    # Plot setup
    n = len(channels)
    rows = int(np.ceil(n/2))
    cols = min(2, n)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5*rows), sharex=True)
    axes = np.array(axes).reshape(-1)
    # Define x-axis range from injection time t to t+12 (inclusive)
    start = t
    end = min(T, t + 13)
    weeks = np.arange(start, end)  # plot from week t to t+12

    delta_pct_dict = {}

    # Impulse-response per channel
    for idx, j in enumerate(channels):
        ax = axes[idx]
        X_scen = X_t.clone()

        # Apply bump per geo
        for g in geos:
            base = media_spend[g, t, j]
            if raw_increase is not None:
                inc = raw_increase
            else:
                inc = std_rg[g, j] * std_multiplier
            s_prime = base + inc
            # rebuild embedding vector
            v_val = torch.log1p(torch.tensor(s_prime, device=device))  # scalar tensor
            V = v_val.repeat(D)                                       # (D,)
            norm = torch.norm(V)
            E = (V / (norm + eps)) * torch.log1p(norm)               # (D,)
            X_scen[g, t, j, :] = E

        # Predict scenario
        with torch.no_grad():
            Yi = model(X_scen, full_geo_ids)
        sales_imp = torch.expm1(Yi[..., 0]).cpu().numpy()

        # Aggregate & compute percentage delta
        total_base = sales_base.sum(axis=0)
        total_imp = sales_imp.sum(axis=0)
        delta = total_imp - total_base
        delta_pct = (delta / (total_base + eps)) * 100
        delta_pct_dict[j] = delta_pct

        # Plot
        ax.plot(weeks, delta_pct[start:end], marker='o')  # plot Δ% sales from t to t+12
        title = media_channels[j] if channel_idx is None else f"Channel {channel_idx}"
        ax.set_title(f"Impulse response of the channel: {title}")
        ax.set_xlabel("Week")
        ax.set_ylabel("Δ Sales (%)")
        ax.grid(True)

        # Annotation position based on first value
        first_val = delta_pct[0]
        if first_val < 0:
            xpos, ypos, va, ha = 0.98, 0.02, 'bottom', 'right'
        else:
            xpos, ypos, va, ha = 0.98, 0.98, 'top', 'right'
        # Summary text
        label = f"Added raw +{raw_increase:.2f}" if raw_increase is not None else f"Added {std_multiplier:.1f}σ"
        # cumulative percentage change over first 3 weeks starting at t
        dS1 = delta_pct[start:start+3].sum()
        # cumulative percentage change over first 13 weeks starting at t
        dS2 = delta_pct[start:end].sum()
        # annotation text showing ranges relative to injection time t
        txt = (
            f"{label} at t={t}\n"
            f"Δ% Sales ({t}–{t+2}): {dS1:.1f}%\n"
            f"Δ% Sales ({t}–{end-1}): {dS2:.1f}%: {dS1:.1f}%\n"
            f"Δ% Sales (0–{len(weeks)-1}): {dS2:.1f}%"
        )
        ax.text(xpos, ypos, txt,
                transform=ax.transAxes, fontsize=9,
                va=va, ha=ha,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8)
        )


    plt.suptitle("Impulse Response Analysis", y=1.02)
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.show()

    return delta_pct_dict



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