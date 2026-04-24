"""
Head-to-head comparison: Lagrange KKT model vs Neural Network predictor vs BTL-only,
all evaluated on synthetic data from a known *linear* AR process.

This script answers: when the true model is linear AR, does the KKT-constrained
closed-form Phi update outperform a learned neural predictor?

Run from repo root:
    python lagrange_vs_nn.py
"""
import math as m
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Lagrange.functions import (
    SkillParameters as LagrangeSkillParams,
    center_scores,
    generate_next_skill_params,
    new_solve_for_phi_matrices,
    play_games_erdos_renyi,
    setup,
)
from NeuralNetwork.nn_functions import (
    SkillParameters as NNSkillParams,
    PredictorMLP,
    optimize_alphas_until_converged_nn,
)

# ------------------------------------------------------------------
# Configuration (shared by all three models)
# ------------------------------------------------------------------
num_players        = 20
AR_order_p         = 3
num_timesteps      = 30
erdos_renyi_p      = 1.0
std_dev            = 0.1
weight             = 30.0
epochs             = 300
N_grad_descent     = 10
N_grad_descent_pred = 5
lr                 = 1e-3
lr_pred            = 1e-3
ridge              = 1e-6
seed               = 0
STEP               = 5

torch.manual_seed(seed)
np.random.seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ------------------------------------------------------------------
# Shared data (linear AR process)
# ------------------------------------------------------------------
players = list(range(num_players))
Z = torch.zeros((num_players, num_players, num_timesteps), device=device)
W = torch.zeros((num_players, num_players, num_timesteps), device=device)

initial, Phi_true = setup(num_players, AR_order_p, device=device)
skill_params = initial.copy()
for t in range(num_timesteps):
    next_s = generate_next_skill_params(skill_params, Phi_true, AR_order_p, std_dev)
    skill_params.append(next_s)
    play_games_erdos_renyi(next_s, players, erdos_renyi_p, Z, W, t)

actual = center_scores(torch.cat(skill_params, dim=1))  # (n, p+T)


def normalize_alpha(alpha: torch.Tensor) -> torch.Tensor:
    alpha_c = center_scores(alpha.clone())
    return alpha_c / torch.norm(alpha_c, dim=0, keepdim=True).clamp_min(1e-8)


# ==================================================================
# Model A: Lagrange KKT
# ==================================================================
print("\n--- Training Lagrange KKT ---")
lagrange_alpha_mse = []
lagrange_phi_mse   = []

lagrange = LagrangeSkillParams(num_players, num_timesteps, AR_order_p).to(device)
lagrange.project_identifiability_()
opt_lag = torch.optim.Adam(lagrange.parameters(), lr=lr)
_, Phi_est = setup(num_players, AR_order_p, device=device)

for epoch in tqdm(range(epochs)):
    for _ in range(N_grad_descent):
        opt_lag.zero_grad(set_to_none=True)
        BTL    = lagrange.compute_log_BTL_vectorized_new(Z, W, num_players)
        AR_err = lagrange.compute_AR_error_new(Phi_est, num_timesteps, AR_order_p)
        (-BTL + weight * AR_err).backward()
        opt_lag.step()
        lagrange.project_identifiability_()

    with torch.no_grad():
        Phi_est = new_solve_for_phi_matrices(
            lagrange.alpha_estimates.detach(),
            num_players, AR_order_p, num_timesteps, ridge=ridge,
        )

    if epoch % STEP == 0:
        with torch.no_grad():
            lagrange_alpha_mse.append(
                F.mse_loss(normalize_alpha(lagrange.alpha_estimates), normalize_alpha(actual)).item()
            )
            lagrange_phi_mse.append(F.mse_loss(Phi_true, Phi_est).item())


# ==================================================================
# Model B: Neural Network predictor
# ==================================================================
print("\n--- Training Neural Network predictor ---")
nn_alpha_mse = []

nn_model  = NNSkillParams(num_players, num_timesteps, AR_order_p).to(device)
nn_model.project_identifiability_()
opt_nn    = torch.optim.Adam(nn_model.parameters(), lr=lr)
predictor = PredictorMLP(
    input_dim=num_players * AR_order_p,
    output_dim=num_players,
    hidden_dim=256,
    num_layers=3,
).to(device)
opt_pred = torch.optim.Adam(predictor.parameters(), lr=lr_pred)

for epoch in tqdm(range(epochs)):
    # Alpha step: maximize BTL - weight * NN_MSE
    for _ in range(N_grad_descent):
        optimize_alphas_until_converged_nn(
            skill_params_est=nn_model,
            predictor=predictor,
            Z=Z, W=W,
            num_players=num_players,
            num_timesteps=num_timesteps,
            AR_order_p=AR_order_p,
            weight=weight,
            optimizer=opt_nn,
            max_steps=1,
            rel_tol=0.0, grad_tol=0.0, patience=1, grad_clip=5.0,
        )
    nn_model.project_identifiability_()

    # Predictor step: fit NN to current alpha estimates (mean-invariant)
    with torch.no_grad():
        est = nn_model.alpha_estimates
        X_list, Y_list = [], []
        for t in range(AR_order_p, AR_order_p + num_timesteps):
            pb = est[:, t - AR_order_p : t]
            pb = pb - pb.mean(dim=0, keepdim=True)
            X_list.append(pb.reshape(-1))
            Y_list.append(est[:, t] - est[:, t].mean())
        X = torch.stack(X_list, dim=0)
        Y = torch.stack(Y_list, dim=0)

    for _ in range(N_grad_descent_pred):
        opt_pred.zero_grad(set_to_none=True)
        Y_hat = predictor(X)
        Y_hat = Y_hat - Y_hat.mean(dim=1, keepdim=True)
        F.mse_loss(Y_hat, Y).backward()
        opt_pred.step()

    if epoch % STEP == 0:
        with torch.no_grad():
            nn_alpha_mse.append(
                F.mse_loss(normalize_alpha(nn_model.alpha_estimates), normalize_alpha(actual)).item()
            )


# ==================================================================
# Model C: BTL-only baseline (weight=0, no AR)
# ==================================================================
print("\n--- Training BTL-only baseline ---")
btl_alpha_mse = []

btl_model = LagrangeSkillParams(num_players, num_timesteps, AR_order_p).to(device)
btl_model.project_identifiability_()
opt_btl = torch.optim.Adam(btl_model.parameters(), lr=lr)

for epoch in tqdm(range(epochs)):
    for _ in range(N_grad_descent):
        opt_btl.zero_grad(set_to_none=True)
        BTL = btl_model.compute_log_BTL_vectorized_new(Z, W, num_players)
        (-BTL).backward()
        opt_btl.step()
        btl_model.project_identifiability_()

    if epoch % STEP == 0:
        with torch.no_grad():
            btl_alpha_mse.append(
                F.mse_loss(normalize_alpha(btl_model.alpha_estimates), normalize_alpha(actual)).item()
            )


# ==================================================================
# Plots
# ==================================================================
epochs_logged = list(range(0, epochs, STEP))

# --- Alpha MSE comparison ---
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epochs_logged, lagrange_alpha_mse, linewidth=2, label="Lagrange KKT (linear AR)")
ax.plot(epochs_logged, nn_alpha_mse,       linewidth=2, label="Neural Network predictor")
ax.plot(epochs_logged, btl_alpha_mse,      linewidth=2, linestyle="--", label="BTL-only (no AR)")
ax.set_xlabel("Epoch")
ax.set_ylabel("Alpha MSE (normalized)")
ax.set_title(
    f"Skill recovery: Lagrange KKT vs NN vs BTL-only\n"
    f"(n={num_players}, p={AR_order_p}, T={num_timesteps}, λ={weight}, σ={std_dev})"
)
ax.grid(True)
ax.legend()
fig.tight_layout()
fig.savefig("lagrange_vs_nn_alpha_mse.pdf")
print("\nSaved lagrange_vs_nn_alpha_mse.pdf")

# --- Phi MSE (Lagrange only, since NN has no explicit Phi) ---
fig2, ax2 = plt.subplots(figsize=(9, 5))
ax2.plot(epochs_logged, lagrange_phi_mse, linewidth=2, color="tab:blue",
         label="Lagrange KKT Phi MSE")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Phi MSE")
ax2.set_title("Transition matrix recovery (Lagrange KKT)")
ax2.grid(True)
ax2.legend()
fig2.tight_layout()
fig2.savefig("lagrange_vs_nn_phi_mse.pdf")
print("Saved lagrange_vs_nn_phi_mse.pdf")

# --- Summary ---
print("\n=== Final results ===")
print(f"  Lagrange KKT:     alpha_mse = {lagrange_alpha_mse[-1]:.4f},  phi_mse = {lagrange_phi_mse[-1]:.6f}")
print(f"  NN predictor:     alpha_mse = {nn_alpha_mse[-1]:.4f}")
print(f"  BTL-only:         alpha_mse = {btl_alpha_mse[-1]:.4f}")
