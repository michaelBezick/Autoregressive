import math as m

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from tqdm import tqdm

from nn_functions import (
    SkillParameters,
    PredictorMLP,
    btl_matrix_from_scores,
    generate_next_skill_params,
    optimize_alphas_until_converged_nn,
    play_games_erdos_renyi,
    setup,
    rank_centrality_mse_per_timestep,
)

# -------------------------------
# Configuration
# -------------------------------
num_players = 100
AR_order_p = 10
erdos_renyi_p = 1.0
std_dev = 1e-1
num_timesteps = 30
epochs = 2000
N_grad_descent_alpha = 10  # inner steps for alpha optimization
N_grad_descent_pred = 5  # inner steps for predictor update
weight = 1.0
lr_alpha = 1e-3
lr_pred = 1e-3
STEP = 10  # logging period

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# -------------------------------
# Synthetic data generation using a "true" NN
# -------------------------------
players = list(range(0, num_players))
Z = torch.zeros((num_players, num_players, num_timesteps))
W = torch.zeros((num_players, num_players, num_timesteps))

# p initial α columns
skill_params, Phi_matrices = setup(num_players, AR_order_p)

# ground-truth predictor used only to synthesize data
true_predictor = PredictorMLP(
    input_dim=num_players * AR_order_p,
    output_dim=num_players,
    hidden_dim=128,
    num_layers=2,
)
true_predictor.eval()  # fixed during data generation

for t in range(num_timesteps):
    next_skill_params = generate_next_skill_params(
        skill_params, Phi_matrices, AR_order_p, std_dev
    )
    skill_params.append(next_skill_params)
    play_games_erdos_renyi(next_skill_params, players, erdos_renyi_p, Z, W, t)

actual_skill_params = torch.stack(
    skill_params, dim=1
)  # (n, p+T, 1) -> after stack: (n, p+T, 1)
actual_skill_params = actual_skill_params.squeeze(2)  # (n, p+T)

# -------------------------------
# Estimation: alternating optimization (α, predictor)
# -------------------------------
skill_params_est = SkillParameters(num_players, num_timesteps, AR_order_p).to(device)
predictor_est = PredictorMLP(
    input_dim=num_players * AR_order_p,
    output_dim=num_players,
    hidden_dim=256,
    num_layers=3,
).to(device)

optimizer_alpha = torch.optim.Adam(params=skill_params_est.parameters(), lr=lr_alpha)
optimizer_pred = torch.optim.Adam(params=predictor_est.parameters(), lr=lr_pred)

Z = Z.to(device)
W = W.to(device)
actual_skill_params = actual_skill_params.to(device)

alpha_errors = []
nn_mses = []  # centered NN MSE (for plots)
btl_likelihoods = []
total_objectives = []
nn_pred_errors_true_inputs = []
nn_pred_errors_est_inputs = []


def normalize_alpha(alpha: torch.Tensor) -> torch.Tensor:
    alpha_c = alpha.clone()
    # keep columns zero-mean and unit-norm for comparison/scale
    alpha_c -= torch.mean(alpha_c, dim=0, keepdim=True)
    alpha_c = alpha_c / (torch.norm(alpha_c, dim=0, keepdim=True) + 1e-8)
    return alpha_c


for epoch in tqdm(range(epochs)):
    # --- Step A: optimize α with predictor fixed ---
    for _ in range(N_grad_descent_alpha):
        BTL, NN_mse_raw, total_obj, _, _ = optimize_alphas_until_converged_nn(
            skill_params_est=skill_params_est,
            predictor=predictor_est,
            Z=Z,
            W=W,
            num_players=num_players,
            num_timesteps=num_timesteps,
            AR_order_p=AR_order_p,
            weight=weight,
            optimizer=optimizer_alpha,
            max_steps=1,  # do a single step; outer loop repeats N_grad_descent_alpha times
            rel_tol=0.0,
            grad_tol=0.0,
            patience=1,
            grad_clip=5.0,
        )

    # --- Project α to zero-mean per timestep (offset invariance) ---
    with torch.no_grad():
        skill_params_est.project_identifiability_()

    # --- Step B: optimize predictor with α fixed (mean-invariant training) ---
    # Build dataset from current α estimates with column-centering and mean-free targets
    with torch.no_grad():
        est = skill_params_est.alpha_estimates  # (n, p+T)
        inputs = []
        targets = []
        for t in range(AR_order_p, AR_order_p + num_timesteps):
            past_block = est[:, t - AR_order_p : t]  # (n, p)
            past_block = past_block - past_block.mean(
                dim=0, keepdim=True
            )  # center each of the p columns
            x_t = past_block.reshape(-1)  # (n*p,)
            y_t = est[:, t] - est[:, t].mean()  # (n,)
            inputs.append(x_t)
            targets.append(y_t)
        X = torch.stack(inputs, dim=0).detach().to(device)  # (T, n*p)
        Y = torch.stack(targets, dim=0).detach().to(device)  # (T, n)

    for _ in range(N_grad_descent_pred):
        optimizer_pred.zero_grad(set_to_none=True)
        Y_hat = predictor_est(X)  # (T, n)
        Y_hat = Y_hat - Y_hat.mean(
            dim=1, keepdim=True
        )  # mean-free predictions per timestep
        pred_loss = F.mse_loss(Y_hat, Y)
        pred_loss.backward()
        optimizer_pred.step()

    # --- Logging / evaluation ---
    if epoch % STEP == 0:
        with torch.no_grad():
            # recompute parts for logging
            BTL_likelihood = skill_params_est.compute_log_BTL_vectorized_new(
                Z, W, num_players
            )
            # The objective inside alpha-step used NN_mse_raw; keep total_obj consistent with training.
            total_obj = BTL_likelihood / m.comb(num_players, 2) - weight * NN_mse_raw

            # normalized comparison to ground truth alphas
            alpha_estimates_normalized = normalize_alpha(
                skill_params_est.alpha_estimates
            )
            actual_skill_params_normalized = normalize_alpha(actual_skill_params)
            mse_alpha = F.mse_loss(
                alpha_estimates_normalized, actual_skill_params_normalized
            ).item()

            # --- Centered NN MSE for plotting (using estimated inputs) ---
            inputs_c = []
            targets_c = []
            for t in range(AR_order_p, AR_order_p + num_timesteps):
                pb = skill_params_est.alpha_estimates[:, t - AR_order_p : t]
                pb = pb - pb.mean(dim=0, keepdim=True)
                x_t = pb.reshape(-1)
                y_t = (
                    skill_params_est.alpha_estimates[:, t]
                    - skill_params_est.alpha_estimates[:, t].mean()
                )
                inputs_c.append(x_t)
                targets_c.append(y_t)
            Xc = torch.stack(inputs_c, dim=0).to(device)
            Yc = torch.stack(targets_c, dim=0).to(device)
            Yc_hat = predictor_est(Xc)
            Yc_hat = Yc_hat - Yc_hat.mean(dim=1, keepdim=True)
            NN_mse_centered = F.mse_loss(Yc_hat, Yc).item()

            print("------------------------------")
            print("Epoch: ", epoch)
            print("BTL_likelihood: ", BTL_likelihood.item())
            print("NN MSE (centered, est inputs):", NN_mse_centered)
            print("skill_params error (normalized MSE): ", mse_alpha)

            alpha_errors.append(mse_alpha)
            nn_mses.append(NN_mse_centered)
            btl_likelihoods.append(BTL_likelihood.item())
            total_objectives.append(total_obj.item())

            # --- NN prediction errors (logged every STEP epochs) ---
            # 1) Using TRUE past alphas as inputs (oracle inputs), mean-invariant
            X_true, Y_true = [], []
            for t_eval in range(AR_order_p, AR_order_p + num_timesteps):
                past_true = actual_skill_params[
                    :, t_eval - AR_order_p : t_eval
                ]  # (n, p)
                past_true = past_true - past_true.mean(
                    dim=0, keepdim=True
                )  # center columns
                X_true.append(past_true.reshape(-1))
                y_true_t = (
                    actual_skill_params[:, t_eval]
                    - actual_skill_params[:, t_eval].mean()
                )
                Y_true.append(y_true_t)
            X_true = torch.stack(X_true, dim=0).to(device)
            Y_true = torch.stack(Y_true, dim=0).to(device)
            Y_pred_true_in = predictor_est(X_true)
            Y_pred_true_in = Y_pred_true_in - Y_pred_true_in.mean(dim=1, keepdim=True)
            nn_pred_mse_true_in = F.mse_loss(Y_pred_true_in, Y_true).item()

            # 2) Using ESTIMATED past alphas as inputs (pipeline inputs), mean-invariant
            X_est = []
            for t_eval in range(AR_order_p, AR_order_p + num_timesteps):
                past_est = skill_params_est.alpha_estimates[
                    :, t_eval - AR_order_p : t_eval
                ]  # (n, p)
                past_est = past_est - past_est.mean(
                    dim=0, keepdim=True
                )  # center columns
                X_est.append(past_est.reshape(-1))
            X_est = torch.stack(X_est, dim=0).to(device)
            Y_pred_est_in = predictor_est(X_est)
            Y_pred_est_in = Y_pred_est_in - Y_pred_est_in.mean(dim=1, keepdim=True)
            # compare to TRUE next-step alphas (also mean-free)
            nn_pred_mse_est_in = F.mse_loss(Y_pred_est_in, Y_true).item()

            nn_pred_errors_true_inputs.append(nn_pred_mse_true_in)
            nn_pred_errors_est_inputs.append(nn_pred_mse_est_in)

# --- Plots ---
epochs_logged = list(range(0, epochs, STEP))

# Final per-timestep MSE of the trained model (centered)
with torch.no_grad():
    T = num_timesteps
    est_T = skill_params_est.alpha_estimates[:, -T:]  # (n, T)
    true_T = actual_skill_params[:, -T:]  # (n, T)

    final_alpha_mse_by_t = []
    for t in range(T):
        est_c = est_T[:, t] - est_T[:, t].mean()
        true_c = true_T[:, t] - true_T[:, t].mean()
        final_alpha_mse_by_t.append(F.mse_loss(est_c, true_c).item())

# Rank Centrality baseline (centered) per timestep
rc_mse_by_t = rank_centrality_mse_per_timestep(Z, W, actual_skill_params, AR_order_p)

cutoff = 15
# One figure, same axes, two lines
plt.figure(figsize=(10, 4))
plt.plot(
    range(num_timesteps)[:cutoff],
    final_alpha_mse_by_t[:cutoff],
    linewidth=2,
    label="Model α MSE (centered)",
)
plt.plot(
    range(num_timesteps)[:cutoff],
    rc_mse_by_t[:cutoff],
    linewidth=2,
    label="Rank Centrality MSE (centered)",
)
plt.title("Per-Timestep Final MSE (centered)")
plt.xlabel("Timestep")
plt.ylabel("MSE")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("nn_alpha_errors.pdf")

plt.figure(figsize=(15, 4))
plt.subplot(1, 3, 1)
plt.plot(epochs_logged, nn_mses, linewidth=2)
plt.title("NN MSE (centered, est inputs)")
plt.xlabel("Epoch")
plt.grid(True)
plt.subplot(1, 3, 2)
plt.plot(epochs_logged, btl_likelihoods, linewidth=2)
plt.title("BTL Likelihood")
plt.xlabel("Epoch")
plt.grid(True)
plt.subplot(1, 3, 3)
plt.plot(epochs_logged, total_objectives, linewidth=2)
plt.title("Total Objective")
plt.xlabel("Epoch")
plt.grid(True)
plt.tight_layout()
plt.savefig("nn_likelihoods.pdf")

# NN prediction errors (two views, mean-invariant)
plt.figure(figsize=(10, 4))
plt.plot(epochs_logged, nn_pred_errors_true_inputs, linewidth=2)
plt.title("NN Prediction Error vs Ground Truth (true past $\\alpha$ inputs, centered)")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(True)
plt.tight_layout()
plt.savefig("nn_pred_errors_true_inputs.pdf")

plt.figure(figsize=(10, 4))
plt.plot(epochs_logged, nn_pred_errors_est_inputs, linewidth=2)
plt.title(
    "NN Prediction Error vs Ground Truth (estimated past $\\alpha$ inputs, centered)"
)
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(True)
plt.tight_layout()
plt.savefig("nn_pred_errors_est_inputs.pdf")
# --- Combined subplot: NN pred error (true inputs) vs. α MSE over epochs ---
plt.figure(figsize=(12, 4))

# Left: NN prediction error using TRUE past alphas (centered)
plt.subplot(1, 2, 1)
plt.plot(epochs_logged, nn_pred_errors_true_inputs, linewidth=2)
plt.title("NN Pred Error (true past $\\alpha$ inputs)")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(True)

# Right: α estimate error (normalized/centered) vs. epochs
plt.subplot(1, 2, 2)
plt.plot(epochs_logged, alpha_errors, linewidth=2)
plt.title("α MSE vs. Ground Truth (centered)")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(True)

plt.tight_layout()
plt.savefig("nn_pred_true_and_alpha_mse_vs_epoch.pdf")

def btl_probs(alpha_vec: torch.Tensor) -> torch.Tensor:
    A = alpha_vec.unsqueeze(1) - alpha_vec.unsqueeze(0)  # (n,n)
    return torch.sigmoid(A)

def weighted_brier_and_nll(P_pred, Z_t, W_t):
    eps = 1e-12
    mask = W_t > 0
    # target = empirical probs from outcomes
    P_tgt = torch.where(mask, Z_t / (W_t + eps), torch.zeros_like(W_t, dtype=torch.float, device=W_t.device))

    # Weighted Brier
    brier = ((P_pred - P_tgt)**2)[mask]
    w = W_t[mask]
    brier_w = (brier * w).sum() / (w.sum() + eps)

    # Weighted NLL (binomial counts)
    Pp = torch.clamp(P_pred, eps, 1 - eps)
    nll_num = (Z_t[mask] * torch.log(Pp[mask]) + (W_t[mask] - Z_t[mask]) * torch.log(1 - Pp[mask])).sum()
    nll_w = - nll_num / (w.sum() + eps)
    return brier_w.item(), nll_w.item()

# --- 1) STATE-FIT at time t (uses estimated alpha_t directly) ---
brier_state_by_t, nll_state_by_t = [], []
with torch.no_grad():
    est_T  = skill_params_est.alpha_estimates[:, -num_timesteps:]   # (n,T)
    for t in range(num_timesteps):
        P_pred_t = btl_probs(est_T[:, t])
        brier_t, nll_t = weighted_brier_and_nll(P_pred_t, Z[:, :, t], W[:, :, t])
        brier_state_by_t.append(brier_t)
        nll_state_by_t.append(nll_t)

# --- 2) ONE-STEP-AHEAD FORECAST (passes through AR/predictor) ---
brier_forecast_by_t, nll_forecast_by_t = [], []
with torch.no_grad():
    for t in range(AR_order_p, AR_order_p + num_timesteps):
        # build predictor input from PAST estimated alphas (mean-center columns, same as training)
        past = skill_params_est.alpha_estimates[:, t-AR_order_p:t]
        past = past - past.mean(dim=0, keepdim=True)
        x = past.reshape(-1).unsqueeze(0).to(device)    # (1, n*p)
        y_hat = predictor_est(x).squeeze(0)             # (n,)
        y_hat = y_hat - y_hat.mean()                    # mean-free

        # probs and comparison for this time t
        P_pred_t = btl_probs(y_hat)
        idx = t - AR_order_p  # align to Z/W index [0..T-1]
        brier_t, nll_t = weighted_brier_and_nll(P_pred_t, Z[:, :, idx], W[:, :, idx])
        brier_forecast_by_t.append(brier_t)
        nll_forecast_by_t.append(nll_t)

# --- Example plotting: State-fit vs Forecast on Brier and NLL ---

import matplotlib.pyplot as plt

# x-axes (state-fit has T points; forecast has T points but starts after warm-up)
t_axis_state    = list(range(num_timesteps))                     # 0..T-1
t_axis_forecast = list(range(AR_order_p, AR_order_p + num_timesteps))  # aligns to absolute t in your model

# If you prefer both to share the same 0..T-1 axis, reindex forecast:
t_axis_forecast_rel = list(range(num_timesteps))  # relative (0..T-1) for forecast curves

# --- Figure 1: Brier ---
plt.figure(figsize=(10, 4))

# Option A: absolute-time x-axis (shows warm-up offset explicitly)
# plt.plot(t_axis_state,    brier_state_by_t,    linewidth=2, label="State-fit Brier")
# plt.plot(t_axis_forecast, brier_forecast_by_t, linewidth=2, label="Forecast Brier")

# Option B (recommended): relative 0..T-1 for both series for easy comparison
plt.plot(t_axis_state,         brier_state_by_t,    linewidth=2, label="State-fit Brier")
plt.plot(t_axis_forecast_rel,  brier_forecast_by_t, linewidth=2, label="Forecast Brier")

# Optional: shade the warm-up region where forecast is based on fewer effective steps
plt.axvspan(0, 0, 0)  # no-op; keep for easy toggling
# Example: if you want to visually mark that forecast uses past p steps, you can add:
# plt.axvline(x=0, color='k', linestyle='--', linewidth=1)

plt.title("BTL Probability MSE")
plt.xlabel("Timestep")
plt.ylabel("MSE")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("btl_MSE_state_vs_forecast.pdf")

# --- Figure 2: NLL ---
plt.figure(figsize=(10, 4))

# Same axis choice as above (relative)
plt.plot(t_axis_state,         nll_state_by_t,    linewidth=2, label="State-fit NLL")
plt.plot(t_axis_forecast_rel,  nll_forecast_by_t, linewidth=2, label="Forecast NLL")

plt.title("BTL Probability Error: Weighted NLL (lower = better)")
plt.xlabel("Timestep")
plt.ylabel("NLL")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("btl_nll_state_vs_forecast.pdf")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
ax1.plot(t_axis_state,        brier_state_by_t,    linewidth=2, label="State-fit")
ax1.plot(t_axis_forecast_rel, brier_forecast_by_t, linewidth=2, label="Forecast")
ax1.set_title("Weighted Brier"); ax1.set_xlabel("Timestep"); ax1.set_ylabel("Brier"); ax1.grid(True); ax1.legend()

ax2.plot(t_axis_state,        nll_state_by_t,    linewidth=2, label="State-fit")
ax2.plot(t_axis_forecast_rel, nll_forecast_by_t, linewidth=2, label="Forecast")
ax2.set_title("Weighted NLL"); ax2.set_xlabel("Timestep"); ax2.set_ylabel("NLL"); ax2.grid(True); ax2.legend()

fig.tight_layout()
fig.savefig("btl_brier_nll_state_vs_forecast.pdf")
