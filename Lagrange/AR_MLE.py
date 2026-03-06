import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from functions import (
    SkillParameters,
    center_scores,
    column_sum_residual,
    generate_next_skill_params,
    new_solve_for_phi_matrices,
    play_games_erdos_renyi,
    setup,
)

# ----------------------- parameters -----------------------
num_players = 100
AR_order_p = 10
erdos_renyi_p = 1.0
std_dev = 1e-1
num_timesteps = 30
epochs = 2_000
N_grad_descent = 10
weight = 30.0
lr = 1e-3
STEP = 10
seed = 0

torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------- setup -----------------------
players = list(range(num_players))
Z = torch.zeros((num_players, num_players, num_timesteps), device=device)
W = torch.zeros((num_players, num_players, num_timesteps), device=device)

initial_skill_params, Phi_matrices = setup(num_players, AR_order_p, device=device)

# simulate latent skills + games
skill_params = initial_skill_params.copy()
for t in range(num_timesteps):
    next_skill_params = generate_next_skill_params(
        skill_params, Phi_matrices, AR_order_p, std_dev
    )
    skill_params.append(next_skill_params)
    play_games_erdos_renyi(next_skill_params, players, erdos_renyi_p, Z, W, t)

actual_skill_params = torch.cat(skill_params, dim=1)  # (n, T+p)
actual_skill_params = center_scores(actual_skill_params)

# ----------------------- estimation -----------------------
skill_params_est = SkillParameters(num_players, num_timesteps, AR_order_p).to(device)
skill_params_est.project_identifiability_()
optimizer = torch.optim.Adam(params=skill_params_est.parameters(), lr=lr)

# feasible random initialization for Phi
_, Phi_matrices_estimate = setup(num_players, AR_order_p, device=device)

true_p_matrix_error = []
true_alphas_error = []
ar_errors = []
btl_likelihoods = []
total_likelihoods = []
constraint_residuals = []



def normalize_alpha(alpha: torch.Tensor) -> torch.Tensor:
    alpha_c = center_scores(alpha.clone())
    norms = torch.norm(alpha_c, dim=0, keepdim=True).clamp_min(1e-8)
    return alpha_c / norms


for epoch in tqdm(range(epochs)):
    for _ in range(N_grad_descent):
        optimizer.zero_grad(set_to_none=True)

        BTL_likelihood = skill_params_est.compute_log_BTL_vectorized_new(Z, W, num_players)
        AR_error = skill_params_est.compute_AR_error_new(
            Phi_matrices_estimate, num_timesteps, AR_order_p
        )
        total_likelihood = BTL_likelihood - weight * AR_error
        loss = -total_likelihood
        loss.backward()
        optimizer.step()
        skill_params_est.project_identifiability_()

    with torch.no_grad():
        Phi_matrices_estimate = new_solve_for_phi_matrices(
            skill_params_est.alpha_estimates.detach(),
            num_players,
            AR_order_p,
            num_timesteps,
            ridge=1e-6,
        )

    if epoch % STEP == 0:
        with torch.no_grad():
            alpha_estimates_normalized = normalize_alpha(skill_params_est.alpha_estimates)
            actual_skill_params_normalized = normalize_alpha(actual_skill_params)

            alpha_mse = F.mse_loss(
                alpha_estimates_normalized, actual_skill_params_normalized
            ).item()
            phi_mse = F.mse_loss(Phi_matrices, Phi_matrices_estimate).item()
            constraint_resid = column_sum_residual(Phi_matrices_estimate).item()

            print("------------------------------")
            print("Epoch:", epoch)
            print("BTL_likelihood:", BTL_likelihood.item())
            print("AR_error:", AR_error.item())
            print("skill_params error:", alpha_mse)
            print("p_matrix error:", phi_mse)
            print("constraint residual:", constraint_resid)

            true_alphas_error.append(alpha_mse)
            true_p_matrix_error.append(phi_mse)
            ar_errors.append(AR_error.item())
            btl_likelihoods.append(BTL_likelihood.item())
            total_likelihoods.append(total_likelihood.item())
            constraint_residuals.append(constraint_resid)

epochs_logged = list(range(0, epochs, STEP))


def plot_series(ax, x, y, label=None, linewidth=2):
    x = np.asarray(x)
    y = np.asarray(y, dtype=float)
    ax.plot(x, y, linewidth=linewidth, label=label)


# --- Figure 1: Errors ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
plot_series(axes[0], epochs_logged, true_alphas_error, label="Alpha Estimates Error (MSE)")
axes[0].set_title("Alpha Estimates Error (MSE)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Error (MSE)")
axes[0].grid(True)

plot_series(axes[1], epochs_logged, true_p_matrix_error, label="Phi Matrix Error (MSE)")
axes[1].set_title("Phi Matrix Error (MSE)")
axes[1].set_xlabel("Epoch")
axes[1].grid(True)

fig.suptitle("Model Estimation Errors Over Time")
fig.tight_layout()
fig.savefig("errors.pdf")

# --- Figure 2: Likelihoods and AR Error ---
fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharex=True)
plot_series(axes[0], epochs_logged, ar_errors, label="AR Error")
axes[0].set_title("AR Error")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss / Likelihood")
axes[0].grid(True)

plot_series(axes[1], epochs_logged, btl_likelihoods, label="BTL Likelihood")
axes[1].set_title("BTL Likelihood")
axes[1].set_xlabel("Epoch")
axes[1].grid(True)

plot_series(axes[2], epochs_logged, total_likelihoods, label="Total Likelihood")
axes[2].set_title("Total Likelihood")
axes[2].set_xlabel("Epoch")
axes[2].grid(True)

fig.suptitle("AR Error, Likelihoods, and Constraint Satisfaction")
fig.tight_layout()
fig.savefig("likelihoods.pdf")
