import math as m

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from tqdm import tqdm

from functions_fixed import (
    generate_next_skill_params,
    play_games_erdos_renyi,
    setup,
    solve_for_phi_matrices,
    SkillParametersOneFixed,
)

"""
NOW, I need to modify the parameterization, instead of a vector
for each person of skill parameters, I have a matrix s.t. the columns
sum to 1.

Furthermore, this property needs to maintain with the AR parameter update
step, meaning I will need to use Lagrange multipliers in the solution.
"""


# parameters

num_players = 30
AR_order_p = 1
erdos_renyi_p = 1
std_dev = 1e-1
num_timesteps = 100
epochs = 10_000
N_grad_descent = 10
weight = 100
lr = 1e-3
STEP = 10

# setup
players = list(range(0, num_players))

Z = torch.zeros((num_players, num_players, num_timesteps))
W = torch.zeros((num_players, num_players, num_timesteps))

# initial params sum to 0, Phi matrices' columns sum to 1
initial_skill_params, Phi_matrices = setup(num_players, AR_order_p)

# generating next skill params and then playing game
skill_params = initial_skill_params

for t in range(num_timesteps):
    next_skill_params = generate_next_skill_params(
        skill_params, Phi_matrices, AR_order_p, std_dev
    )

    skill_params.append(next_skill_params)
    play_games_erdos_renyi(
        next_skill_params,
        players,
        erdos_renyi_p,
        Z,
        W,
        t,
        games_per_pair=1,
    )




actual_skill_params = skill_params
actual_skill_params = torch.stack(actual_skill_params, dim=1)

# now Z and W have the necessary data, can run algorithm

skill_params_est = SkillParametersOneFixed(num_players, num_timesteps, AR_order_p)
optimizer = torch.optim.Adam(params=skill_params_est.parameters(), lr=lr)
Phi_matrices_estimate = torch.randn((AR_order_p, num_players - 1, num_players - 1))

skill_params_est = skill_params_est.cuda()
W = W.cuda()
Z = Z.cuda()
actual_skill_params = actual_skill_params.cuda()
Phi_matrices_estimate = Phi_matrices_estimate.cuda()
Phi_matrices = Phi_matrices.cuda()

true_p_matrix_error = []
true_alphas_error = []
ar_errors = []
btl_likelihoods = []
total_likelihoods = []
projected_mean_errors = []

for epoch in tqdm(range(epochs)):

    for _ in range(N_grad_descent):

        optimizer.zero_grad()
        BTL_likelihood = skill_params_est.compute_log_BTL(Z, W, num_players)

        AR_error = skill_params_est.compute_AR_error(
            Phi_matrices_estimate, num_timesteps, AR_order_p
        )

        # normalizing per player
        AR_error /= num_players - 1

        total_likelihood = BTL_likelihood - weight * AR_error

        loss = -total_likelihood

        loss.backward()
        optimizer.step()

    # second, update phi estimate
    with torch.no_grad():

        Phi_matrices_estimate = solve_for_phi_matrices(
            skill_params_est.alpha_estimates[1:, :],  # drop player 0
            num_players - 1,
            AR_order_p,
            num_timesteps,
        )

    if epoch % STEP == 0:

        print("------------------------------")
        print("Epoch: ", epoch)
        print("BTL_likelihood: ", BTL_likelihood.item())
        print("AR_error", AR_error.item())

        def normalize_alpha(alpha: torch.Tensor) -> torch.Tensor:
            alpha_c = alpha.clone()
            alpha_c -= torch.mean(alpha_c, dim=0)
            alpha_c /= torch.norm(alpha_c, dim=0)

            return alpha_c

        alpha_estimates_normalized = normalize_alpha(skill_params_est.alpha_estimates)
        actual_skill_params_normalized = normalize_alpha(
            torch.squeeze(actual_skill_params)
        )

        print(
            "skill_params error: ",
            F.mse_loss(
                alpha_estimates_normalized, actual_skill_params_normalized
            ).item(),
        )
        print(
            "p_matrix error: ", F.mse_loss(Phi_matrices, Phi_matrices_estimate).item()
        )

        true_alphas_error.append(
            F.mse_loss(
                alpha_estimates_normalized.data,
                actual_skill_params_normalized.data,
            ).item()
        )
        true_p_matrix_error.append(
            F.mse_loss(Phi_matrices, Phi_matrices_estimate).item()
        )
        ar_errors.append(AR_error.item())
        btl_likelihoods.append(BTL_likelihood.item())
        total_likelihoods.append(total_likelihood.item())



epochs_logged = list(range(0, epochs, STEP))

# ---- Robust outlier-aware plotting helpers ----
import numpy as np


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
    ax, x, y, label=None, k=1000, color=None,
    scatter_outliers=False, linewidth=2,
    remove_outliers=False,
):
    x = np.asarray(x)
    y = np.asarray(y, dtype=float)

    if remove_outliers:
        mask = robust_mask_mad(y, k=k)
    else:
        # Only drop non-finite values (or even keep everything if you want)
        mask = np.isfinite(y)

    ax.plot(x[mask], y[mask], linewidth=linewidth, label=label, color=color)

    if remove_outliers and scatter_outliers and (~mask).any():
        ax.scatter(x[~mask], y[~mask], s=12, alpha=0.35, color=color)




# --- Figure 1: Errors (individual subplots) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

plot_series(
    axes[0], epochs_logged, true_alphas_error, label="Alpha Estimates Error (MSE)"
)
axes[0].set_title("Alpha Estimates Error (MSE)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Error (MSE)")
axes[0].grid(True)

plot_series(axes[1], epochs_logged, true_p_matrix_error, label="P Matrix Error (MSE)")
axes[1].set_title("P Matrix Error (MSE)")
axes[1].set_xlabel("Epoch")
axes[1].grid(True)

fig.suptitle("Model Estimation Errors Over Time")
fig.tight_layout()
fig.savefig("errors.pdf")

# --- Figure 2: Likelihoods and AR Error (individual subplots) ---
fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharex=True)

plot_series(axes[0], epochs_logged, ar_errors, label="AR Error")
axes[0].set_title("AR Error")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss/Likelihood")
axes[0].grid(True)

plot_series(axes[1], epochs_logged, btl_likelihoods, label="BTL Likelihood")
axes[1].set_title("BTL Likelihood")
axes[1].set_xlabel("Epoch")
axes[1].grid(True)

plot_series(axes[2], epochs_logged, total_likelihoods, label="Total Likelihood")
axes[2].set_title("Total Likelihood")
axes[2].set_xlabel("Epoch")
axes[2].grid(True)

fig.suptitle("AR Error, BTL Likelihood, and Total Likelihood Over Time")
fig.tight_layout()
fig.savefig("likelihoods.pdf")
