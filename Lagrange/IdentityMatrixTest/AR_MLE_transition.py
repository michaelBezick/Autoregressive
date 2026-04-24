import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from Lagrange.functions import (
    SkillParameters,
    btl_matrix_from_scores,
    center_scores,
    column_sum_residual,
    generate_next_skill_params,
    new_solve_for_phi_matrices,
    play_games_erdos_renyi,
    setup,
)

# ----------------------- experiment mode -----------------------
USE_IDENTITY_AR1 = True

# ----------------------- parameters -----------------------
num_players = 100
AR_order_p = 1 if USE_IDENTITY_AR1 else 10
erdos_renyi_p = 1.0
std_dev = 0.0 if USE_IDENTITY_AR1 else 1e-1
num_timesteps = 30
epochs = 2_000
N_grad_descent = 10
weight = 30.0
lr = 1e-3
STEP = 10
seed = 0

if USE_IDENTITY_AR1 and AR_order_p != 1:
    raise ValueError("USE_IDENTITY_AR1=True requires AR_order_p=1.")

torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------- setup -----------------------
players = list(range(num_players))
Z = torch.zeros((num_players, num_players, num_timesteps), device=device)
W = torch.zeros((num_players, num_players, num_timesteps), device=device)

initial_skill_params, Phi_matrices = setup(
    num_players,
    AR_order_p,
    device=device,
    identity_ar1=USE_IDENTITY_AR1,
)

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
_, Phi_matrices_estimate = setup(num_players, AR_order_p, device=device, identity_ar1=False)

true_p_matrix_error = []
true_alphas_error = []
ar_errors = []
btl_likelihoods = []
total_likelihoods = []
constraint_residuals = []
transition_state_mse_history = []
transition_btl_mse_history = []
transition_map_mse_history = []


def normalize_alpha(alpha: torch.Tensor) -> torch.Tensor:
    alpha_c = center_scores(alpha.clone())
    norms = torch.norm(alpha_c, dim=0, keepdim=True).clamp_min(1e-8)
    return alpha_c / norms



def temporal_skill_drift(alpha: torch.Tensor) -> float:
    if alpha.shape[1] < 2:
        return 0.0
    return (alpha[:, 1:] - alpha[:, :-1]).pow(2).mean().item()



def predict_next_skills_from_histories(phi_stack: torch.Tensor, alpha_path: torch.Tensor, p: int) -> torch.Tensor:
    """
    One-step predictions using the provided latent histories.

    alpha_path: (n, T+p)
    returns:   (n, T), where column t corresponds to the prediction of alpha_{p+t}
               from [alpha_t, ..., alpha_{t+p-1}] under the same lag convention used
               elsewhere in the codebase.
    """
    n, total_steps = alpha_path.shape
    num_preds = total_steps - p
    preds = []
    for t in range(p, total_steps):
        history = alpha_path[:, t - p : t].T.unsqueeze(2)  # (p, n, 1)
        pred = torch.bmm(phi_stack, history).sum(dim=0).squeeze(1)  # (n,)
        preds.append(pred)
    return torch.stack(preds, dim=1) if preds else alpha_path.new_zeros((n, 0))



def transition_diagnostics(
    phi_stack_est: torch.Tensor,
    alpha_path_true: torch.Tensor,
    p: int,
    phi_stack_true: torch.Tensor | None = None,
):
    """
    Evaluate whether the learned Phi reproduces the one-step transition dynamics on the
    true latent histories, rather than whether it numerically matches the generating Phi.
    """
    actual_next = alpha_path_true[:, p:]
    pred_est = predict_next_skills_from_histories(phi_stack_est, alpha_path_true, p)

    state_mse = F.mse_loss(pred_est, actual_next).item()

    btl_mses = []
    for t in range(pred_est.shape[1]):
        pred_probs = btl_matrix_from_scores(pred_est[:, t])
        actual_probs = btl_matrix_from_scores(actual_next[:, t])
        btl_mses.append(F.mse_loss(pred_probs, actual_probs))
    btl_mse = torch.stack(btl_mses).mean().item() if btl_mses else 0.0

    out = {
        "transition_state_mse": state_mse,
        "transition_btl_mse": btl_mse,
    }

    if phi_stack_true is not None:
        pred_true = predict_next_skills_from_histories(phi_stack_true, alpha_path_true, p)
        out["transition_map_mse"] = F.mse_loss(pred_est, pred_true).item()

    return out


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
            constraint_resid = column_sum_residual(Phi_matrices_estimate).item()

            print("------------------------------")
            print("Epoch:", epoch)
            print("BTL_likelihood:", BTL_likelihood.item())
            print("AR_error:", AR_error.item())
            print("skill_params error:", alpha_mse)
            print("constraint residual:", constraint_resid)

            if USE_IDENTITY_AR1:
                dyn_stats = transition_diagnostics(
                    Phi_matrices_estimate,
                    actual_skill_params,
                    AR_order_p,
                    phi_stack_true=Phi_matrices,
                )
                print("transition state MSE on true latent histories:", dyn_stats["transition_state_mse"])
                print("transition BTL-probability MSE on true latent histories:", dyn_stats["transition_btl_mse"])
                print("transition-map MSE vs true Phi on true latent histories:", dyn_stats["transition_map_mse"])
                transition_state_mse_history.append(dyn_stats["transition_state_mse"])
                transition_btl_mse_history.append(dyn_stats["transition_btl_mse"])
                transition_map_mse_history.append(dyn_stats["transition_map_mse"])
            else:
                phi_mse = F.mse_loss(Phi_matrices, Phi_matrices_estimate).item()
                print("p_matrix error:", phi_mse)
                true_p_matrix_error.append(phi_mse)

            true_alphas_error.append(alpha_mse)
            ar_errors.append(AR_error.item())
            btl_likelihoods.append(BTL_likelihood.item())
            total_likelihoods.append(total_likelihood.item())
            constraint_residuals.append(constraint_resid)

epochs_logged = list(range(0, epochs, STEP))



def plot_series(ax, x, y, label=None, linewidth=2):
    x = np.asarray(x)
    y = np.asarray(y, dtype=float)
    ax.plot(x, y, linewidth=linewidth, label=label)


# --- Figure 1: Errors / Dynamics ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
plot_series(axes[0], epochs_logged, true_alphas_error, label="Alpha Estimates Error (MSE)")
axes[0].set_title("Alpha Estimates Error (MSE)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Error (MSE)")
axes[0].grid(True)

if USE_IDENTITY_AR1:
    plot_series(
        axes[1],
        epochs_logged,
        transition_state_mse_history,
        label="Transition state MSE",
    )
    axes[1].set_title("One-Step Transition State MSE")
else:
    plot_series(axes[1], epochs_logged, true_p_matrix_error, label="Phi Matrix Error (MSE)")
    axes[1].set_title("Phi Matrix Error (MSE)")
axes[1].set_xlabel("Epoch")
axes[1].grid(True)

fig.suptitle("Model Estimation Diagnostics Over Time")
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

if USE_IDENTITY_AR1:
    fig, ax = plt.subplots(figsize=(6, 4))

    plot_series(ax, epochs_logged, transition_state_mse_history, label="Transition state MSE")
    ax.set_title("Transition State MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Error (MSE)")
    ax.grid(True)

    fig.suptitle("Transition-Dynamics Diagnostics")
    fig.tight_layout()
    fig.savefig("transition_state_mse.pdf")

    with torch.no_grad():
        final_dyn_stats = transition_diagnostics(
            Phi_matrices_estimate,
            actual_skill_params,
            AR_order_p,
            phi_stack_true=Phi_matrices,
        )
        true_skill_drift = temporal_skill_drift(actual_skill_params)
        est_skill_drift = temporal_skill_drift(skill_params_est.alpha_estimates)

    print("\n========================================")
    print("Final identity AR(1) transition-dynamics diagnostics")
    print(
        "transition state MSE on true latent histories:",
        final_dyn_stats["transition_state_mse"],
    )
    print(
        "transition BTL-probability MSE on true latent histories:",
        final_dyn_stats["transition_btl_mse"],
    )
    print(
        "transition-map MSE vs true Phi on true latent histories:",
        final_dyn_stats["transition_map_mse"],
    )
    print("true latent skill drift across adjacent times (should be ~0):", true_skill_drift)
    print("estimated latent skill drift across adjacent times:", est_skill_drift)
    print("========================================")
