import os
import glob
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from scipy.stats import pearsonr
from collections import defaultdict



from functions_fixed import (
    btl_matrix_from_scores,           # kept for compatibility, not used here
    generate_next_skill_params,       # used for AR rollout on test
    new_solve_for_phi_matrices,       # kept for compatibility
    optimize_alphas_until_converged,  # kept for compatibility
    play_games_erdos_renyi,           # kept for compatibility
    setup,                            # kept for compatibility
    solve_for_phi_matrices,
    SkillParametersOneFixed,
)

"""
Real-data BTL+AR experiment using ATP tennis matches with held-out future data.

Assumptions:
- You have yearly CSVs in ./matches/, e.g.:
    ./matches/atp_matches_2018.csv
    ./matches/atp_matches_2019.csv
  in Jeff Sackmann format (winner_id, loser_id, tourney_date, ...)

- You have atp_players.csv in the repo root with columns:
    player_id, name_first, name_list, ...

This script:
1) Loads all match CSVs from ./matches.
2) Discretizes time into months from the earliest tourney_date.
3) Restricts to first MAX_MONTHS months (if desired).
4) Splits months into TRAIN (early) and TEST (future).
5) Builds Z, W only on TRAIN months and trains BTL + AR.
6) Uses learned alphas + Phi to AR-rollout skills into TEST months.
7) Evaluates predictive log-likelihood on TRAIN and TEST matches.
8) Plots AR error, BTL likelihood, total likelihood on TRAIN.
9) Prints top-10 players by final-timestep skill.
"""

# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------
MATCHES_DIR = "./matches"
PLAYERS_PATH = "./atp_players.csv"

WIN_COL = "winner_id"
LOS_COL = "loser_id"
DATE_COL = "tourney_date"  # int/string YYYYMMDD

MAX_MONTHS = 59        # optional cap on number of months to use (set None for all)
num_players = 100      # number of players to include in the rating model
AR_order_p = 10
train_fraction = 0.7   # fraction of months used for training (rest is held out)

epochs = 100
N_grad_descent = 10
weight = 100
lr = 1e-2
STEP = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------------------------------------------------
# 1) Load players (for readable names later)
# ----------------------------------------------------------------------
if os.path.exists(PLAYERS_PATH):
    players_df = pd.read_csv(PLAYERS_PATH)
    id_to_name = {}
    for row in players_df.itertuples():
        pid = getattr(row, "player_id")
        first = getattr(row, "name_first")
        last = getattr(row, "name_list")
        id_to_name[pid] = f"{first} {last}"
else:
    print(f"Warning: {PLAYERS_PATH} not found; using IDs as names.")
    id_to_name = {}

# ----------------------------------------------------------------------
# 2) Load and concatenate all match CSVs from ./matches
# ----------------------------------------------------------------------
match_files = sorted(glob.glob(os.path.join(MATCHES_DIR, "atp_matches_*.csv")))
if not match_files:
    raise RuntimeError(f"No match files found in {MATCHES_DIR}/atp_matches_*.csv")

print("Found match files:")
for f in match_files:
    print("  ", f)

dfs = []
for f in match_files:
    df_year = pd.read_csv(f)
    if {WIN_COL, LOS_COL, DATE_COL}.issubset(df_year.columns):
        dfs.append(df_year[[WIN_COL, LOS_COL, DATE_COL]])
    else:
        missing = {WIN_COL, LOS_COL, DATE_COL} - set(df_year.columns)
        raise ValueError(f"File {f} is missing columns: {missing}")

matches = pd.concat(dfs, ignore_index=True)
print("Total raw matches loaded:", len(matches))

# ----------------------------------------------------------------------
# 3) Clean + time-discretize into month_index
# ----------------------------------------------------------------------
matches = matches.dropna(subset=[WIN_COL, LOS_COL, DATE_COL])

matches[DATE_COL] = pd.to_datetime(
    matches[DATE_COL].astype(str), format="%Y%m%d", errors="coerce"
)
matches = matches.dropna(subset=[DATE_COL])
matches = matches.sort_values(DATE_COL).reset_index(drop=True)

print("Date range:",
      matches[DATE_COL].min(),
      "->",
      matches[DATE_COL].max())

base_date = matches[DATE_COL].min()
matches["month_index"] = (
    (matches[DATE_COL].dt.year - base_date.year) * 12
    + (matches[DATE_COL].dt.month - base_date.month)
)

# Optional cap on months
if MAX_MONTHS is not None:
    matches = matches[
        (matches["month_index"] >= 0) & (matches["month_index"] < MAX_MONTHS)
    ].copy()
    print(
        f"After restricting to first {MAX_MONTHS} months: "
        f"{len(matches)} matches, {matches['month_index'].nunique()} months."
    )

if matches.empty:
    raise RuntimeError("No matches remain after time filtering.")

num_timesteps_total = int(matches["month_index"].max()) + 1
print("Total unique months (num_timesteps_total) =", num_timesteps_total)

# ----------------------------------------------------------------------
# 4) Choose top num_players by total match appearances
# ----------------------------------------------------------------------
all_players_series = pd.concat([matches[WIN_COL], matches[LOS_COL]])
player_counts = all_players_series.value_counts()
top_players = player_counts.head(num_players).index.tolist()
num_players = len(top_players)
print("Final num_players (actual) =", num_players)

matches = matches[
    matches[WIN_COL].isin(top_players) & matches[LOS_COL].isin(top_players)
].copy()
print("Matches after restricting to top players:", len(matches))

player_to_idx = {pid: i for i, pid in enumerate(top_players)}
idx_to_pid = {i: pid for pid, i in player_to_idx.items()}

# ----------------------------------------------------------------------
# 5) Train/test split in time (by month_index)
# ----------------------------------------------------------------------
# Ensure train covers at least AR_order_p+1 months
train_timesteps = max(AR_order_p + 1, int(train_fraction * num_timesteps_total))
train_timesteps = min(train_timesteps, num_timesteps_total - 1)  # leave at least 1 month for test
test_start = train_timesteps
print(f"Train timesteps: 0 .. {train_timesteps-1}, Test timesteps: {test_start} .. {num_timesteps_total-1}")

matches_train = matches[matches["month_index"] < train_timesteps].copy()
matches_test = matches[matches["month_index"] >= train_timesteps].copy()
print("Train matches:", len(matches_train), "Test matches:", len(matches_test))

if matches_test.empty:
    print("Warning: no test matches; consider decreasing train_fraction or MAX_MONTHS.")

# ----------------------------------------------------------------------
# 6) Build Z_train and W_train for TRAIN months only
# ----------------------------------------------------------------------
Z_train = torch.zeros((num_players, num_players, train_timesteps), dtype=torch.float32)
W_train = torch.zeros((num_players, num_players, train_timesteps), dtype=torch.float32)

for row in matches_train.itertuples():
    wi = getattr(row, WIN_COL)
    li = getattr(row, LOS_COL)
    t = int(row.month_index)
    if wi not in player_to_idx or li not in player_to_idx:
        continue
    i = player_to_idx[wi]
    j = player_to_idx[li]
    if i == j:
        continue

    Z_train[i, j, t] += 1.0  # i beat j
    W_train[j, i, t] += 1.0  # loser marked in W

print("Constructed Z_train, W_train with shapes:", Z_train.shape, W_train.shape)
print("Train comparisons (Z_train sum):", Z_train.sum().item())
print("Train wins (W_train sum):", W_train.sum().item())

Z_train = Z_train.to(device)
W_train = W_train.to(device)

# ----------------------------------------------------------------------
# 7) BTL + AR training on TRAIN data
# ----------------------------------------------------------------------
skill_params_est = SkillParametersOneFixed(num_players, train_timesteps, AR_order_p).to(
    device
)
optimizer = torch.optim.Adam(params=skill_params_est.parameters(), lr=lr)

Phi_matrices_estimate = torch.randn(
    (AR_order_p, num_players, num_players), device=device
)

ar_errors = []
btl_likelihoods = []
total_likelihoods = []

for epoch in tqdm(range(epochs)):
    for _ in range(N_grad_descent):
        optimizer.zero_grad()

        BTL_likelihood = skill_params_est.compute_log_BTL(Z_train, W_train, num_players)
        AR_error = skill_params_est.compute_AR_error(
            Phi_matrices_estimate, train_timesteps, AR_order_p
        )
        AR_error = AR_error / num_players

        total_likelihood = BTL_likelihood - weight * AR_error
        loss = -total_likelihood

        loss.backward()
        optimizer.step()

    with torch.no_grad():
        Phi_matrices_estimate = solve_for_phi_matrices(
            skill_params_est.alpha_estimates, num_players, AR_order_p, train_timesteps
        ).to(device)

    if epoch % STEP == 0:
        print("-" * 30)
        print("Epoch:", epoch)
        print("BTL_likelihood:", BTL_likelihood.item())
        print("AR_error:", AR_error.item())
        print("Total_likelihood:", total_likelihood.item())

        ar_errors.append(AR_error.item())
        btl_likelihoods.append(BTL_likelihood.item())
        total_likelihoods.append(total_likelihood.item())

epochs_logged = list(range(0, epochs, STEP))

# ----------------------------------------------------------------------
# 8) AR rollout to predict future skills on TEST months (corrected)
# ----------------------------------------------------------------------
with torch.no_grad():
    alpha_full = skill_params_est.alpha_estimates.detach().clone()  # (n, train_T + p)
    T_full = alpha_full.shape[1]

    # Remember: training used the *last* train_timesteps columns of alpha_full
    # for months 0..train_timesteps-1:
    #   month t  ↔ alpha_full[:, AR_order_p + t]
    month_skills_train = [
        alpha_full[:, AR_order_p + t : AR_order_p + t + 1]  # (n, 1)
        for t in range(train_timesteps)
    ]

    # Predicted alphas for *external* month indices 0..num_timesteps_total-1
    predicted_alphas = torch.zeros(
        (num_players, num_timesteps_total),
        dtype=torch.float32,
        device=device,
    )

    # Fill in training months with the same skills used in BTL training
    for t in range(train_timesteps):
        predicted_alphas[:, t] = month_skills_train[t][:, 0]

    # Build history for AR rollout in month-index space
    # history[t] = skill vector for month t (shape (n, 1))
    history = month_skills_train.copy()

    # Roll forward for t = train_timesteps .. num_timesteps_total-1
    for t in range(train_timesteps, num_timesteps_total):
        next_alpha = generate_next_skill_params(
            history, Phi_matrices_estimate, AR_order_p, std_dev=0.0
        )  # (n,1)

        if next_alpha.dim() == 2 and next_alpha.shape[1] == 1:
            predicted_alphas[:, t] = next_alpha[:, 0]
        else:
            predicted_alphas[:, t] = next_alpha

        if next_alpha.dim() == 1:
            next_alpha = next_alpha.unsqueeze(1)
        history.append(next_alpha)

# ----------------------------------------------------------------------
# 8.1) Static "snapshot AR" skills: keep final train-month skills fixed
# ----------------------------------------------------------------------
with torch.no_grad():
    # Skills at the last training month (train_timesteps - 1)
    alpha_snapshot = predicted_alphas[:, train_timesteps - 1].clone()
    # Ensure 1D tensor on correct device
    alpha_snapshot_torch = alpha_snapshot.to(device=device)

# ----------------------------------------------------------------------
# 8.2) Static BTL-only model (no AR term, MLE on TRAIN matches)
# ----------------------------------------------------------------------
class StaticBTLModel(torch.nn.Module):
    """
    Pure Bradley–Terry–Luce: one scalar skill per player, no time, no AR.
    Trained on TRAIN matches only.
    """
    def __init__(self, num_players, device):
        super().__init__()
        self.num_players = num_players
        self.alpha = torch.nn.Parameter(
            torch.zeros(num_players, device=device)  # init at 0
        )

    def compute_loglik_on_matches(self, matches_df, player_to_idx):
        total_loglik = torch.tensor(0.0, device=self.alpha.device)
        n_matches = 0
        for row in matches_df.itertuples():
            wi = getattr(row, WIN_COL)
            li = getattr(row, LOS_COL)
            if wi not in player_to_idx or li not in player_to_idx:
                continue
            i = player_to_idx[wi]
            j = player_to_idx[li]
            if i == j:
                continue

            diff = self.alpha[i] - self.alpha[j]
            p_win = torch.sigmoid(diff)
            p_win = torch.clamp(p_win, 1e-12, 1.0 - 1e-12)
            total_loglik += torch.log(p_win)
            n_matches += 1

        return total_loglik, n_matches


# Instantiate and train BTL-only model
btl_only_model = StaticBTLModel(num_players, device).to(device)
opt_btl_only = torch.optim.Adam(btl_only_model.parameters(), lr=lr)

btl_only_epochs = epochs  # you can change separately if desired

for epoch in range(btl_only_epochs):
    opt_btl_only.zero_grad()
    ll_btl, n_btl = btl_only_model.compute_loglik_on_matches(
        matches_train, player_to_idx
    )
    if n_btl == 0:
        break  # no matches? nothing to train
    loss_btl = -ll_btl / n_btl  # average loss
    loss_btl.backward()
    opt_btl_only.step()

with torch.no_grad():
    alpha_btl_only_torch = btl_only_model.alpha.detach().clone().to(device=device)

# ----------------------------------------------------------------------
# 8.3) Neural AR **joint** model (replaces linear Φ in the objective)
# ----------------------------------------------------------------------
class NeuralARJointForecaster(torch.nn.Module):
    """
    Neural AR model for BTL+AR joint training:
      input: concatenation of last p month skill vectors (n * p)
      output: next month skill vector (n)
    This will be trained *together* with skill parameters using
    the same BTL likelihood - weight * AR_error objective.
    """
    def __init__(self, num_players, p, hidden_dim=128):
        super().__init__()
        self.num_players = num_players
        self.p = p
        self.net = torch.nn.Sequential(
            torch.nn.Linear(num_players * p, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_players),
        )

    def forward(self, x):
        # x: (batch, num_players * p)
        return self.net(x)  # (batch, num_players)


def compute_neural_AR_error(neural_ar, alpha_estimates, num_timesteps, AR_order_p):
    """
    AR penalty term using a neural AR forecaster instead of linear Φ.
    alpha_estimates: (num_players, total_T)
    We use months t = p .. p + num_timesteps - 1, matching the BTL window.
    """
    est = alpha_estimates  # (n, T+p)
    n, T_total = est.shape
    p = AR_order_p

    total_se = est.new_tensor(0.0)
    count = 0

    # Same time window as linear AR: [p, p + num_timesteps)
    for t in range(p, p + num_timesteps):
        # history: last p months' skills  -> shape (n, p)
        hist = est[:, t - p : t]  # (n, p)
        x = hist.T.reshape(1, -1)  # (1, n*p)
        pred = neural_ar(x)[0]     # (n,)

        actual = est[:, t]         # (n,)
        se = (pred - actual).pow(2).sum()
        total_se = total_se + se
        count += n

    return total_se / (count + 1e-8)
# ----------------------------------------------------------------------
# 8.4) Joint BTL + Neural AR training (end-to-end)
# ----------------------------------------------------------------------
skill_params_neural = SkillParametersOneFixed(num_players, train_timesteps, AR_order_p).to(device)
neural_ar_joint = NeuralARJointForecaster(num_players, AR_order_p, hidden_dim=128).to(device)

optimizer_joint = torch.optim.Adam(
    list(skill_params_neural.parameters()) + list(neural_ar_joint.parameters()),
    lr=lr,
)

ar_errors_neural = []
btl_likelihoods_neural = []
total_likelihoods_neural = []

for epoch in tqdm(range(epochs), desc="Neural-AR joint training"):
    for _ in range(N_grad_descent):
        optimizer_joint.zero_grad()

        # BTL likelihood with neural skill parameters
        BTL_likelihood_nn = skill_params_neural.compute_log_BTL(
            Z_train, W_train, num_players
        )

        # Neural AR error, same structure as linear AR error
        AR_error_nn = compute_neural_AR_error(
            neural_ar_joint,
            skill_params_neural.alpha_estimates,
            train_timesteps,
            AR_order_p,
        )

        AR_error_nn = AR_error_nn / num_players

        total_likelihood_nn = BTL_likelihood_nn - weight * AR_error_nn
        loss_nn = -total_likelihood_nn

        loss_nn.backward()
        optimizer_joint.step()

    if epoch % STEP == 0:
        print("=" * 30)
        print("[Neural-AR] Epoch:", epoch)
        print("[Neural-AR] BTL_likelihood:", BTL_likelihood_nn.item())
        print("[Neural-AR] AR_error:", AR_error_nn.item())
        print("[Neural-AR] Total_likelihood:", total_likelihood_nn.item())

        ar_errors_neural.append(AR_error_nn.item())
        btl_likelihoods_neural.append(BTL_likelihood_nn.item())
        total_likelihoods_neural.append(total_likelihood_nn.item())
# ----------------------------------------------------------------------
# 8.5) Neural-AR joint rollout: predicted_alphas_neural_joint (n x num_timesteps_total)
# ----------------------------------------------------------------------
with torch.no_grad():
    alpha_full_neural = skill_params_neural.alpha_estimates.detach().clone()  # (n, train_T + p)

    # month t ↔ alpha_full_neural[:, AR_order_p + t]
    month_skills_train_neural = [
        alpha_full_neural[:, AR_order_p + t : AR_order_p + t + 1]  # (n, 1)
        for t in range(train_timesteps)
    ]

    predicted_alphas_neural_joint = torch.zeros(
        (num_players, num_timesteps_total),
        dtype=torch.float32,
        device=device,
    )

    # Fill training months
    for t in range(train_timesteps):
        predicted_alphas_neural_joint[:, t] = month_skills_train_neural[t][:, 0]

    # Roll forward for t = train_timesteps .. num_timesteps_total-1
    for t in range(train_timesteps, num_timesteps_total):
        if t < AR_order_p:
            continue

        # last p months from neural_joint predictions
        hist = predicted_alphas_neural_joint[:, t - AR_order_p : t]  # (n, p)
        x = hist.T.reshape(1, -1)                                   # (1, n*p)
        next_alpha_joint = neural_ar_joint(x)[0]                     # (n,)

        predicted_alphas_neural_joint[:, t] = next_alpha_joint

# ----------------------------------------------------------------------
# 8.6) Neural AR forecaster (nonlinear AR on skills)
# ----------------------------------------------------------------------
class NeuralARForecaster(torch.nn.Module):
    """
    Neural AR model:
      input: concatenation of last p month skill vectors (n * p)
      output: next month skill vector (n)
    Here we train it on the *learned* AR-BTL skills (month_skills_train).
    """
    def __init__(self, num_players, p, hidden_dim=128):
        super().__init__()
        self.num_players = num_players
        self.p = p
        self.net = torch.nn.Sequential(
            torch.nn.Linear(num_players * p, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_players),
        )

    def forward(self, x):
        # x: (batch, num_players * p)
        return self.net(x)  # (batch, num_players)


# Build training data for Neural AR from training skills
p = AR_order_p
# month_skills_train is a list of tensors (n,1) for months 0..train_timesteps-1
# We'll use months p..(train_timesteps-1) as targets, with last p months as inputs.
X_list = []
Y_list = []

with torch.no_grad():
    # Flatten (n,1) -> (n) for convenience
    month_skills_flat = [m[:, 0] for m in month_skills_train]  # len = train_timesteps

for t in range(p, train_timesteps):
    # history months: t-p ... t-1
    hist = torch.stack(month_skills_flat[t - p : t], dim=1)  # (n, p)
    x = hist.reshape(-1)  # (n * p,)
    y = month_skills_flat[t]  # (n,)

    X_list.append(x)
    Y_list.append(y)

if len(X_list) > 0:
    X_train_ar = torch.stack(X_list, dim=0).to(device)  # (num_samples, n*p)
    Y_train_ar = torch.stack(Y_list, dim=0).to(device)  # (num_samples, n)
else:
    X_train_ar = torch.empty((0, num_players * p), device=device)
    Y_train_ar = torch.empty((0, num_players), device=device)

neural_ar = NeuralARForecaster(num_players, p).to(device)
opt_neural_ar = torch.optim.Adam(neural_ar.parameters(), lr=1e-3)
neural_epochs = 200  # adjust if needed

for epoch in range(neural_epochs):
    if X_train_ar.size(0) == 0:
        break
    opt_neural_ar.zero_grad()
    pred = neural_ar(X_train_ar)       # (num_samples, n)
    loss = torch.mean((pred - Y_train_ar) ** 2)
    loss.backward()
    opt_neural_ar.step()
# ----------------------------------------------------------------------
# Neural AR rollout: predicted_alphas_neural (n x num_timesteps_total)
# ----------------------------------------------------------------------
with torch.no_grad():
    predicted_alphas_neural = torch.zeros(
        (num_players, num_timesteps_total),
        dtype=torch.float32,
        device=device,
    )

    # Fill training months with the same skills used in BTL training
    for t in range(train_timesteps):
        predicted_alphas_neural[:, t] = month_skills_flat[t]

    # For rollout, we maintain a sliding window of last p months of NEURAL AR predictions
    # Start from the training months (0..train_timesteps-1) already set
    for t in range(train_timesteps, num_timesteps_total):
        if t < p:
            # not enough history, but in practice train_timesteps >= p+1 by construction
            continue

        # last p months from the *neural* predictions
        hist_neural = []
        for k in range(t - p, t):
            hist_neural.append(predicted_alphas_neural[:, k])
        hist_neural = torch.stack(hist_neural, dim=1)  # (n, p)
        x = hist_neural.reshape(1, -1)                 # (1, n*p)

        next_alpha_neural = neural_ar(x)               # (1, n)
        predicted_alphas_neural[:, t] = next_alpha_neural[0]




# ----------------------------------------------------------------------
# 9) Predictive log-likelihood on TRAIN and TEST matches
# ----------------------------------------------------------------------
def compute_loglik_per_match(matches_subset, predicted_alphas, label=""):
    total_loglik = torch.tensor(0.0, device=device)
    n_matches = 0

    for row in matches_subset.itertuples():
        wi = getattr(row, WIN_COL)
        li = getattr(row, LOS_COL)
        t = int(row.month_index)
        if wi not in player_to_idx or li not in player_to_idx:
            continue
        if t < 0 or t >= num_timesteps_total:
            continue

        i = player_to_idx[wi]
        j = player_to_idx[li]
        if i == j:
            continue

        alpha_i = predicted_alphas[i, t]
        alpha_j = predicted_alphas[j, t]
        p_win = torch.sigmoid(alpha_i - alpha_j)
        # numerical safety
        p_win = torch.clamp(p_win, 1e-12, 1.0 - 1e-12)

        total_loglik += torch.log(p_win)
        n_matches += 1

    if n_matches == 0:
        print(f"[{label}] No matches; cannot compute log-likelihood.")
        return None, 0

    avg_loglik = (total_loglik / n_matches).item()
    return avg_loglik, n_matches


train_loglik, n_train = compute_loglik_per_match(
    matches_train, predicted_alphas, label="TRAIN"
)
test_loglik, n_test = compute_loglik_per_match(
    matches_test, predicted_alphas, label="TEST"
)

print("\n=== Predictive log-likelihood summary ===")
if train_loglik is not None:
    print(f"Train matches: {n_train}, avg log p(win) = {train_loglik:.4f}, "
          f"avg p(win) = {math.e**train_loglik:.4f}")
if test_loglik is not None:
    print(f"Test  matches: {n_test}, avg log p(win) = {test_loglik:.4f}, "
          f"avg p(win) = {math.e**test_loglik:.4f}")

# ----------------------------------------------------------------------
# 10) Robust plotting helpers (same as before)
# ----------------------------------------------------------------------
def robust_mask_mad(y, k: float = 3.5):
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(y)
    yv = y[finite]
    if yv.size == 0:
        return finite
    med = np.median(yv)
    mad = np.median(np.abs(yv - med))
    if mad == 0:
        return finite
    sigma = 1.4826 * mad
    keep = np.abs(y - med) <= k * sigma
    return finite & keep


def plot_series(ax, x, y, label=None, k=3.5, color=None, scatter_outliers=False, linewidth=2):
    x = np.asarray(x)
    y = np.asarray(y, dtype=float)
    mask = robust_mask_mad(y, k=k)
    ax.plot(x[mask], y[mask], linewidth=linewidth, label=label, color=color)
    if scatter_outliers and (~mask).any():
        ax.scatter(x[~mask], y[~mask], s=12, alpha=0.35, color=color)


# ----------------------------------------------------------------------
# 11) Plot AR Error, BTL Likelihood, Total Likelihood (TRAIN)
# ----------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)

plot_series(axes[0], epochs_logged, ar_errors, label="AR Error")
axes[0].set_title("AR Error (Train)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss/Likelihood")
axes[0].grid(True)

plot_series(axes[1], epochs_logged, btl_likelihoods, label="BTL Likelihood")
axes[1].set_title("BTL Likelihood (Train)")
axes[1].set_xlabel("Epoch")
axes[1].grid(True)

plot_series(axes[2], epochs_logged, total_likelihoods, label="Total Likelihood")
axes[2].set_title("Total Likelihood (Train)")
axes[2].set_xlabel("Epoch")
axes[2].grid(True)

fig.suptitle("AR Error, BTL Likelihood, and Total Likelihood (Train, ATP Tennis)")
fig.tight_layout()
fig.savefig("tennis_likelihoods_train.pdf")
print("Saved plot to tennis_likelihoods_train.pdf")
# ----------------------------------------------------------------------
# 12) Rank Centrality baseline on TEST matches
# ----------------------------------------------------------------------
def compute_rank_centrality(matches_subset, player_to_idx, num_players,
                            max_iter=1000, tol=1e-8, damping=0.15):
    """
    Rank Centrality (Negahban et al. 2012) on a subset of matches.

    We build a transition matrix P where
        C[i,j] = # times j beat i
        P[i,:]  = row-normalized C[i,:], then apply damping:
        P = (1 - damping) * P + damping * (1/n) * 1 1^T

    The stationary distribution pi (row vector) of P is the rank centrality score.
    """
    C = np.zeros((num_players, num_players), dtype=float)

    for row in matches_subset.itertuples():
        wi = getattr(row, WIN_COL)
        li = getattr(row, LOS_COL)
        if wi not in player_to_idx or li not in player_to_idx:
            continue
        # edge from loser -> winner
        i = player_to_idx[li]
        j = player_to_idx[wi]
        if i == j:
            continue
        C[i, j] += 1.0

    # Row-normalize to get transition probabilities
    P = np.zeros_like(C)
    row_sums = C.sum(axis=1, keepdims=True)
    nonzero = row_sums[:, 0] > 0
    P[nonzero] = C[nonzero] / row_sums[nonzero]

    n = num_players
    # Add teleportation / damping to ensure irreducible & aperiodic
    P = (1.0 - damping) * P + damping * (np.ones((n, n), dtype=float) / n)

    # Power iteration for stationary distribution pi (row vector)
    pi = np.ones(n, dtype=float) / n
    for _ in range(max_iter):
        pi_new = pi @ P
        if np.linalg.norm(pi_new - pi, 1) < tol:
            pi = pi_new
            break
        pi = pi_new

    pi = np.maximum(pi, 1e-12)  # avoid zeros
    pi = pi / pi.sum()
    return pi  # length num_players, sums to 1


# Rank Centrality on TEST matches only (future period)
rc_scores_test = compute_rank_centrality(matches_test, player_to_idx, num_players)
# Convert to a "skill" vector on log-scale so we can plug into logistic BTL
alpha_rc_test = np.log(rc_scores_test)
alpha_rc_test_torch = torch.from_numpy(alpha_rc_test).to(device=device, dtype=torch.float32)


# ----------------------------------------------------------------------
# 13) Per-timestep predictive log-likelihood on TEST:
#     AR-BTL forecast vs Rank Centrality
# ----------------------------------------------------------------------
def compute_loglik_ar_for_month(matches_subset, predicted_alphas, month_idx):
    """
    Average log-likelihood on matches in a specific month, using
    time-varying AR-BTL skills: predicted_alphas[:, month_idx].
    """
    if matches_subset.empty:
        return None, 0

    total_loglik = torch.tensor(0.0, device=device)
    n_matches = 0

    for row in matches_subset.itertuples():
        wi = getattr(row, WIN_COL)
        li = getattr(row, LOS_COL)
        if wi not in player_to_idx or li not in player_to_idx:
            continue
        i = player_to_idx[wi]
        j = player_to_idx[li]
        if i == j:
            continue

        alpha_i = predicted_alphas[i, month_idx]
        alpha_j = predicted_alphas[j, month_idx]
        p_win = torch.sigmoid(alpha_i - alpha_j)
        p_win = torch.clamp(p_win, 1e-12, 1.0 - 1e-12)
        total_loglik += torch.log(p_win)
        n_matches += 1

    if n_matches == 0:
        return None, 0

    avg_loglik = (total_loglik / n_matches).item()
    return avg_loglik, n_matches


def compute_loglik_static_for_month(matches_subset, alpha_vec, month_idx):
    """
    Average log-likelihood on matches in a specific month, using
    a *static* skill vector alpha_vec (e.g., Rank Centrality).
    """
    if matches_subset.empty:
        return None, 0

    total_loglik = torch.tensor(0.0, device=device)
    n_matches = 0

    for row in matches_subset.itertuples():
        wi = getattr(row, WIN_COL)
        li = getattr(row, LOS_COL)
        if wi not in player_to_idx or li not in player_to_idx:
            continue
        i = player_to_idx[wi]
        j = player_to_idx[li]
        if i == j:
            continue

        alpha_i = alpha_vec[i]
        alpha_j = alpha_vec[j]
        p_win = torch.sigmoid(alpha_i - alpha_j)
        p_win = torch.clamp(p_win, 1e-12, 1.0 - 1e-12)
        total_loglik += torch.log(p_win)
        n_matches += 1

    if n_matches == 0:
        return None, 0

    avg_loglik = (total_loglik / n_matches).item()
    return avg_loglik, n_matches

# ----------------------------------------------------------------------
# 13) Per-month predictive log-likelihood (TRAIN + TEST):
#     AR-BTL vs Rank Centrality vs AR-snapshot vs BTL-only vs Neural-AR
# ----------------------------------------------------------------------
all_months = np.arange(num_timesteps_total)  # 0 .. num_timesteps_total-1
ar_ll_all = []
rc_ll_all = []
snap_ll_all = []
btl_ll_all = []
neural_ll_all = []         # old two-stage neural AR
neural_joint_ll_all = []   # NEW joint neural AR-BTL model
n_matches_all = []

for t in all_months:
    if t < train_timesteps:
        matches_t = matches_train[matches_train["month_index"] == t].copy()
    else:
        matches_t = matches_test[matches_test["month_index"] == t].copy()

    ar_ll,        n_ar        = compute_loglik_ar_for_month(matches_t, predicted_alphas, t)
    rc_ll,        n_rc        = compute_loglik_static_for_month(matches_t, alpha_rc_test_torch, t)
    snap_ll,      n_snap      = compute_loglik_static_for_month(matches_t, alpha_snapshot_torch, t)
    btl_ll,       n_btl       = compute_loglik_static_for_month(matches_t, alpha_btl_only_torch, t)
    neural_ll,    n_neural    = compute_loglik_ar_for_month(matches_t, predicted_alphas_neural, t)
    neural_joint_ll, n_joint  = compute_loglik_ar_for_month(matches_t, predicted_alphas_neural_joint, t)

    # Require at least 1 match and valid values for all models
    if any(n == 0 for n in [n_ar, n_rc, n_snap, n_btl, n_neural, n_joint]):
        ar_ll_all.append(np.nan)
        rc_ll_all.append(np.nan)
        snap_ll_all.append(np.nan)
        btl_ll_all.append(np.nan)
        neural_ll_all.append(np.nan)
        neural_joint_ll_all.append(np.nan)
        n_matches_all.append(0)
    else:
        ar_ll_all.append(ar_ll)
        rc_ll_all.append(rc_ll)
        snap_ll_all.append(snap_ll)
        btl_ll_all.append(btl_ll)
        neural_ll_all.append(neural_ll)
        neural_joint_ll_all.append(neural_joint_ll)
        n_matches_all.append(n_ar)

ar_ll_all          = np.array(ar_ll_all,          dtype=float)
rc_ll_all          = np.array(rc_ll_all,          dtype=float)
snap_ll_all        = np.array(snap_ll_all,        dtype=float)
btl_ll_all         = np.array(btl_ll_all,         dtype=float)
neural_ll_all      = np.array(neural_ll_all,      dtype=float)
neural_joint_ll_all = np.array(neural_joint_ll_all, dtype=float)
n_matches_all      = np.array(n_matches_all,      dtype=int)


# Only plot months with at least 1 match
mask = n_matches_all > 0

# x-axis: month index relative to the very first month (0 is earliest month)
x_rel = all_months  # 0..num_timesteps_total-1
train_test_boundary_x = train_timesteps  # vertical line here


# ----------------------------------------------------------------------
# 14) Plot: all models, with train/test boundary
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    x_rel[mask],
    ar_ll_all[mask],
    marker="o",
    label="AR-BTL (linear Φ forecast)",
)
ax.plot(
    x_rel[mask],
    rc_ll_all[mask],
    marker="s",
    label="Rank Centrality (static)",
)
ax.plot(
    x_rel[mask],
    snap_ll_all[mask],
    marker="^",
    label="AR snapshot (last train month, static)",
)
ax.plot(
    x_rel[mask],
    btl_ll_all[mask],
    marker="x",
    label="BTL-only (static MLE)",
)
ax.plot(
    x_rel[mask],
    neural_ll_all[mask],
    marker="d",
    label="Neural AR (nonlinear forecast)",
)

ax.plot(
    x_rel[mask],
    neural_joint_ll_all[mask],
    marker="v",
    label="Neural AR (joint BTL+AR)",
)


# Vertical line separating train and test months
ax.axvline(
    train_test_boundary_x - 0.5,
    linestyle="--",
    color="k",
    alpha=0.8,
    label="Train/Test boundary",
)

ax.set_xlabel("Month index (0 = earliest month)")
ax.set_ylabel("Avg log p(win)")
ax.set_title(
    "Per-month predictive log-likelihood:\n"
    "AR-BTL vs RC vs AR-snapshot vs BTL-only vs Neural-AR"
)

ymin = np.nanmin(
    np.concatenate([
        ar_ll_all[mask],
        rc_ll_all[mask],
        snap_ll_all[mask],
        btl_ll_all[mask],
        neural_ll_all[mask],
        neural_joint_ll_all[mask],
    ])
)
ymax = np.nanmax(
    np.concatenate([
        ar_ll_all[mask],
        rc_ll_all[mask],
        snap_ll_all[mask],
        btl_ll_all[mask],
        neural_ll_all[mask],
        neural_joint_ll_all[mask],
    ])
)

ax.fill_between(
    x_rel,
    ymin,
    ymax,
    where=x_rel < train_timesteps,
    alpha=0.05,
    color="green",
    label="_train_region",
)
ax.fill_between(
    x_rel,
    ymin,
    ymax,
    where=x_rel >= train_timesteps,
    alpha=0.05,
    color="red",
    label="_test_region",
)

ax.grid(True)
ax.legend()

fig.tight_layout()
fig.savefig("loglik_by_month_all_models.pdf")
print("Saved plot to loglik_by_month_all_models.pdf")

print("\nPer-month log-likelihood (only months with matches):")

for t, ll_ar, ll_rc, ll_snap, ll_btl, ll_neural, ll_neural_joint, n_m in zip(
    x_rel[mask],
    ar_ll_all[mask],
    rc_ll_all[mask],
    snap_ll_all[mask],
    btl_ll_all[mask],
    neural_ll_all[mask],
    neural_joint_ll_all[mask],
    n_matches_all[mask],
):
    region = "TRAIN" if t < train_timesteps else "TEST "
    print(
        f"  month={t:2d} [{region}], matches={n_m:4d}, "
        f"AR ll={ll_ar: .4f}, RC ll={ll_rc: .4f}, "
        f"SNAP ll={ll_snap: .4f}, BTL ll={ll_btl: .4f}, "
        f"Neural(2-stage) ll={ll_neural: .4f}, "
        f"Neural(joint) ll={ll_neural_joint: .4f}"
    )
