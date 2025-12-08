import math as m

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from tqdm import tqdm

from functions_fixed import (
    btl_matrix_from_scores,
    generate_next_skill_params,
    optimize_alphas_until_converged,
    play_games_erdos_renyi,
    setup,
    solve_for_phi_matrices,
    SkillParametersOneFixed,
)

"""
Sweep experiment over different values of the AR weight parameter.
We:
  1) Generate a single synthetic dataset (true skills + games).
  2) For each weight value, re-initialize the model and learn.
  3) Log errors and likelihoods every STEP epochs.
  4) Plot curves comparing different weights.
"""

# -------------------------------------------------
# Parameters
# -------------------------------------------------
num_players = 100
AR_order_p = 10
erdos_renyi_p = 1
std_dev = 1e-1
num_timesteps = 30
epochs = 100
N_grad_descent = 100
lr = 1e-3
STEP = 10

# Sweep over these weight values
WEIGHTS_TO_TRY = [0.1, 1.0, 10.0, 100.0, 1000.0]

BASE_SEED = 0
torch.manual_seed(BASE_SEED)
np.random.seed(BASE_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------------------------
# 1) Generate synthetic data once (independent of weight)
# -------------------------------------------------
players = list(range(num_players))

Z = torch.zeros((num_players, num_players, num_timesteps))
W = torch.zeros((num_players, num_players, num_timesteps))

# initial params sum to 0, Phi matrices' columns sum to 1
initial_skill_params, Phi_matrices = setup(num_players, AR_order_p)

skill_params = initial_skill_params

for t in range(num_timesteps):
    next_skill_params = generate_next_skill_params(
        skill_params, Phi_matrices, AR_order_p, std_dev
    )
    skill_params.append(next_skill_params)
    play_games_erdos_renyi(next_skill_params, players, erdos_renyi_p, Z, W, t)

actual_skill_params = skill_params
actual_skill_params = torch.stack(actual_skill_params, dim=1)

# Move data to device once
Z = Z.to(device)
W = W.to(device)
actual_skill_params = actual_skill_params.to(device)
Phi_matrices = Phi_matrices.to(device)


# -------------------------------------------------
# Helper: normalize alpha along players (per timestep)
# -------------------------------------------------
def normalize_alpha(alpha: torch.Tensor) -> torch.Tensor:
    """
    Normalize alpha per time step:
      1) subtract mean over players
      2) divide by L2 norm over players
    Shape expected: [num_players, num_timesteps]
    """
    alpha_c = alpha.clone()
    alpha_c -= torch.mean(alpha_c, dim=0)
    alpha_c /= torch.norm(alpha_c, dim=0)
    return alpha_c


# -------------------------------------------------
# 2) Run experiment for each weight
# -------------------------------------------------
# Results dictionary:
# results[weight] = {
#   "true_alphas_error": [...],
#   "true_p_matrix_error": [...],
#   "ar_errors": [...],
#   "btl_likelihoods": [...],
#   "total_likelihoods": [...],
# }
results = {}

for idx, weight in enumerate(WEIGHTS_TO_TRY):
    print("\n========================================")
    print(f"Training with weight = {weight}")
    print("========================================")

    # Different seed per weight for the model initialization
    torch.manual_seed(BASE_SEED + idx + 1)

    # Fresh model and optimizer
    skill_params_est = SkillParametersOneFixed(
        num_players, num_timesteps, AR_order_p
    ).to(device)
    optimizer = torch.optim.Adam(params=skill_params_est.parameters(), lr=lr)

    # Initial Phi estimate
    Phi_matrices_estimate = torch.randn(
        (AR_order_p, num_players - 1, num_players - 1), device=device
    )

    # Per-weight logs
    true_p_matrix_error = []
    true_alphas_error = []
    ar_errors = []
    btl_likelihoods = []
    total_likelihoods = []

    for epoch in tqdm(range(epochs), desc=f"weight={weight}"):
        # First, optimize alphas via gradient descent steps
        for _ in range(N_grad_descent):
            optimizer.zero_grad()

            BTL_likelihood = skill_params_est.compute_log_BTL(
                Z, W, num_players
            )  # scalar

            AR_error = skill_params_est.compute_AR_error(
                Phi_matrices_estimate, num_timesteps, AR_order_p
            )  # scalar

            # normalizing per player
            AR_error = AR_error / num_players

            total_likelihood = BTL_likelihood - weight * AR_error
            loss = -total_likelihood

            loss.backward()
            optimizer.step()

        # Second, update Phi estimate in closed form
        with torch.no_grad():
            Phi_matrices_estimate = solve_for_phi_matrices(
                skill_params_est.alpha_estimates[1:, :],
                num_players - 1,
                AR_order_p,
                num_timesteps,
            )
            # ensure correct device (in case solve_for_phi_matrices returned CPU)
            Phi_matrices_estimate = Phi_matrices_estimate.to(device)

        # Logging
        if epoch % STEP == 0:
            print("------------------------------")
            print(f"[weight={weight}] Epoch: {epoch}")
            print("BTL_likelihood:", BTL_likelihood.item())
            print("AR_error:", AR_error.item())

            alpha_estimates_normalized = normalize_alpha(
                skill_params_est.alpha_estimates
            )
            actual_skill_params_normalized = normalize_alpha(
                torch.squeeze(actual_skill_params)
            )

            alpha_err = F.mse_loss(
                alpha_estimates_normalized, actual_skill_params_normalized
            ).item()
            p_err = F.mse_loss(Phi_matrices, Phi_matrices_estimate).item()

            print("skill_params error (MSE):", alpha_err)
            print("p_matrix error (MSE):", p_err)

            true_alphas_error.append(alpha_err)
            true_p_matrix_error.append(p_err)
            ar_errors.append(AR_error.item())
            btl_likelihoods.append(BTL_likelihood.item())
            total_likelihoods.append(total_likelihood.item())

    # store per-weight results
    results[weight] = {
        "true_alphas_error": true_alphas_error,
        "true_p_matrix_error": true_p_matrix_error,
        "ar_errors": ar_errors,
        "btl_likelihoods": btl_likelihoods,
        "total_likelihoods": total_likelihoods,
    }

# common x-axis for logged epochs
epochs_logged = list(range(0, epochs, STEP))


# -------------------------------------------------
# 3) Robust plotting helpers (MAD-based outlier masking)
# -------------------------------------------------
def robust_mask_mad(y, k: float = 3.5):
    """
    Return a boolean mask keeping points within k * sigma_MAD
    where sigma_MAD = 1.4826 * MAD. Works with NaNs/inf safely.
    """
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(y)
    yv = y[finite]
    if yv.size == 0:
        return finite  # nothing to filter
    med = np.median(yv)
    mad = np.median(np.abs(yv - med))
    if mad == 0:
        return finite  # no dispersion -> keep all finite points
    sigma = 1.4826 * mad
    keep = np.abs(y - med) <= k * sigma
    return finite & keep


def plot_series(
    ax, x, y, label=None, k=3.5, color=None, scatter_outliers=False, linewidth=2
):
    """
    Plot y vs x while ignoring outliers via MAD. Optionally scatter the removed outliers.
    """
    x = np.asarray(x)
    y = np.asarray(y, dtype=float)
    mask = robust_mask_mad(y, k=k)
    ax.plot(x[mask], y[mask], linewidth=linewidth, label=label, color=color)
    if scatter_outliers and (~mask).any():
        ax.scatter(x[~mask], y[~mask], s=12, alpha=0.35, color=color)


# -------------------------------------------------
# 4) Plot: Errors vs Epoch for each weight
# -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

for weight in WEIGHTS_TO_TRY:
    r = results[weight]
    plot_series(
        axes[0],
        epochs_logged,
        r["true_alphas_error"],
        label=f"w={weight}",
    )

axes[0].set_title("Alpha Estimates Error (MSE)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Error (MSE)")
axes[0].grid(True)
axes[0].legend()

for weight in WEIGHTS_TO_TRY:
    r = results[weight]
    plot_series(
        axes[1],
        epochs_logged,
        r["true_p_matrix_error"],
        label=f"w={weight}",
    )

axes[1].set_title("P Matrix Error (MSE)")
axes[1].set_xlabel("Epoch")
axes[1].grid(True)
axes[1].legend()

fig.suptitle("Model Estimation Errors Over Time (Weight Sweep)")
fig.tight_layout()
fig.savefig("errors_weight_sweep.pdf")

# -------------------------------------------------
# 5) Plot: AR Error, BTL Likelihood, Total Likelihood vs Epoch for each weight
# -------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)

for weight in WEIGHTS_TO_TRY:
    r = results[weight]
    plot_series(
        axes[0],
        epochs_logged,
        r["ar_errors"],
        label=f"w={weight}",
    )
axes[0].set_title("AR Error")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss/Likelihood")
axes[0].grid(True)
axes[0].legend()

for weight in WEIGHTS_TO_TRY:
    r = results[weight]
    plot_series(
        axes[1],
        epochs_logged,
        r["btl_likelihoods"],
        label=f"w={weight}",
    )
axes[1].set_title("BTL Likelihood")
axes[1].set_xlabel("Epoch")
axes[1].grid(True)
axes[1].legend()

for weight in WEIGHTS_TO_TRY:
    r = results[weight]
    plot_series(
        axes[2],
        epochs_logged,
        r["total_likelihoods"],
        label=f"w={weight}",
    )
axes[2].set_title("Total Likelihood")
axes[2].set_xlabel("Epoch")
axes[2].grid(True)
axes[2].legend()

fig.suptitle("AR Error, BTL Likelihood, and Total Likelihood (Weight Sweep)")
fig.tight_layout()
fig.savefig("likelihoods_weight_sweep.pdf")

print("Saved plots: errors_weight_sweep.pdf, likelihoods_weight_sweep.pdf")
