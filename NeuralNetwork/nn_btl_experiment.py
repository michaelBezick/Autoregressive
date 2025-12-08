import math as m
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F

from nn_functions import (
    SkillParameters,
    PredictorMLP,
    generate_next_skill_params_nn,
    optimize_alphas_until_converged_nn,
    play_games_erdos_renyi,
    setup_initial_alphas,
)

@torch.no_grad()
def _normalize_alpha_cols(alpha: torch.Tensor) -> torch.Tensor:
    # zero-mean each timestep column and unit-norm for comparison
    a = alpha - alpha.mean(dim=0, keepdim=True)
    return a / (a.norm(dim=0, keepdim=True) + 1e-8)

def _center_cols(mat: torch.Tensor) -> torch.Tensor:
    # center each column independently (mean-invariant for inputs)
    return mat - mat.mean(dim=0, keepdim=True)

def _center_vec(vec: torch.Tensor) -> torch.Tensor:
    return vec - vec.mean()

def run_nn_btl_experiment(
    *,
    num_players: int = 100,
    num_timesteps: int = 30,
    AR_order_p: int = 10,
    epochs: int = 500,
    weight: float = 30.0,
    erdos_renyi_p: float = 1.0,
    std_dev: float = 1e-1,
    N_grad_descent_alpha: int = 10,
    N_grad_descent_pred: int = 5,
    lr_alpha: float = 1e-3,
    lr_pred: float = 1e-3,
    STEP: int = 10,
    seed: Optional[int] = 0,
    device: Optional[torch.device] = None,
    save_plots_to: Optional[str] = None,   # if provided, save PDFs there
) -> Dict[str, Any]:
    """
    Mean-invariant alternating optimization (BTL + NN predictor).
    Returns logs & final metrics; optionally saves per-run plots.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        torch.manual_seed(seed)

    # --------- Synthetic data with a fixed "true" NN -----------
    players = list(range(num_players))
    Z = torch.zeros((num_players, num_players, num_timesteps), device=device)
    W = torch.zeros((num_players, num_players, num_timesteps), device=device)

    skill_params_cols = setup_initial_alphas(num_players, AR_order_p, device)

    # true NN only for data gen
    true_predictor = PredictorMLP(
        input_dim=num_players * AR_order_p,
        output_dim=num_players,
        hidden_dim=128,
        num_layers=2,
    ).to(device).eval()

    for t in range(num_timesteps):
        next_skill = generate_next_skill_params_nn(
            skill_params_cols, true_predictor, AR_order_p, std_dev
        )  # (n,1) on CPU by default in functions.py; move if needed
        next_skill = next_skill.to(device)
        skill_params_cols.append(next_skill)
        play_games_erdos_renyi(next_skill.squeeze(1), players, erdos_renyi_p, Z, W, t)

    actual_skill_params = torch.stack(skill_params_cols, dim=1).squeeze(2).to(device)  # (n, p+T)

    # --------- Estimation models ----------
    skill_params_est = SkillParameters(num_players, num_timesteps, AR_order_p).to(device)
    predictor_est = PredictorMLP(
        input_dim=num_players * AR_order_p,
        output_dim=num_players,
        hidden_dim=256,
        num_layers=3,
    ).to(device)

    opt_alpha = torch.optim.Adam(skill_params_est.parameters(), lr=lr_alpha)
    opt_pred  = torch.optim.Adam(predictor_est.parameters(), lr=lr_pred)

    # --------- Logs ----------
    epochs_logged = []
    alpha_errors = []
    nn_mses_centered = []
    btl_likelihoods = []
    total_objectives = []
    nn_pred_errors_true_inputs = []
    nn_pred_errors_est_inputs  = []

    for epoch in range(epochs):
        # ---- α-step (holds NN fixed) ----
        for _ in range(N_grad_descent_alpha):
            BTL_lik, NN_mse_raw, total_obj, _, _ = optimize_alphas_until_converged_nn(
                skill_params_est=skill_params_est,
                predictor=predictor_est,
                Z=Z,
                W=W,
                num_players=num_players,
                num_timesteps=num_timesteps,
                AR_order_p=AR_order_p,
                weight=weight,
                optimizer=opt_alpha,
                max_steps=1,
                rel_tol=0.0,
                grad_tol=0.0,
                patience=1,
                grad_clip=5.0,
            )

        # Project α to zero-mean per timestep (offset invariance)
        with torch.no_grad():
            skill_params_est.alpha_estimates -= skill_params_est.alpha_estimates.mean(dim=0, keepdim=True)

        # ---- NN-step (mean-invariant training) ----
        with torch.no_grad():
            est = skill_params_est.alpha_estimates  # (n, p+T)
            X_list, Y_list = [], []
            for t in range(AR_order_p, AR_order_p + num_timesteps):
                past = _center_cols(est[:, t-AR_order_p:t])           # (n,p)
                x_t = past.reshape(-1)
                y_t = _center_vec(est[:, t])
                X_list.append(x_t)
                Y_list.append(y_t)
            X = torch.stack(X_list, dim=0)  # (T, n*p)
            Y = torch.stack(Y_list, dim=0)  # (T, n)

        for _ in range(N_grad_descent_pred):
            opt_pred.zero_grad(set_to_none=True)
            Y_hat = predictor_est(X)
            Y_hat = Y_hat - Y_hat.mean(dim=1, keepdim=True)
            pred_loss = F.mse_loss(Y_hat, Y)
            pred_loss.backward()
            opt_pred.step()

        # ---- Logging ----
        if epoch % STEP == 0:
            with torch.no_grad():
                # objective pieces
                btl = skill_params_est.compute_log_BTL_vectorized_new(Z, W, num_players)
                total_obj = btl / m.comb(num_players, 2) - weight * NN_mse_raw

                # normalized α error vs ground truth
                mse_alpha = F.mse_loss(
                    _normalize_alpha_cols(skill_params_est.alpha_estimates),
                    _normalize_alpha_cols(actual_skill_params),
                ).item()

                # centered NN MSE (est inputs)
                Xc, Yc = [], []
                for t in range(AR_order_p, AR_order_p + num_timesteps):
                    pb = _center_cols(skill_params_est.alpha_estimates[:, t-AR_order_p:t])
                    Xc.append(pb.reshape(-1))
                    Yc.append(_center_vec(skill_params_est.alpha_estimates[:, t]))
                Xc = torch.stack(Xc, dim=0)
                Yc = torch.stack(Yc, dim=0)
                Yc_hat = predictor_est(Xc)
                Yc_hat = Yc_hat - Yc_hat.mean(dim=1, keepdim=True)
                nn_mse_centered = F.mse_loss(Yc_hat, Yc).item()

                # prediction MSE with TRUE inputs (oracle) and with EST inputs (pipeline)
                Xt, Yt, Xe = [], [], []
                for t in range(AR_order_p, AR_order_p + num_timesteps):
                    past_true = _center_cols(actual_skill_params[:, t-AR_order_p:t])
                    Xt.append(past_true.reshape(-1))
                    Yt.append(_center_vec(actual_skill_params[:, t]))
                    past_est  = _center_cols(skill_params_est.alpha_estimates[:, t-AR_order_p:t])
                    Xe.append(past_est.reshape(-1))
                Xt = torch.stack(Xt, dim=0); Yt = torch.stack(Yt, dim=0); Xe = torch.stack(Xe, dim=0)
                Y_hat_true = predictor_est(Xt); Y_hat_true = Y_hat_true - Y_hat_true.mean(dim=1, keepdim=True)
                Y_hat_est  = predictor_est(Xe); Y_hat_est  = Y_hat_est  - Y_hat_est.mean(dim=1, keepdim=True)
                mse_true_inputs = F.mse_loss(Y_hat_true, Yt).item()
                mse_est_inputs  = F.mse_loss(Y_hat_est , Yt).item()

                epochs_logged.append(epoch)
                alpha_errors.append(mse_alpha)
                nn_mses_centered.append(nn_mse_centered)
                btl_likelihoods.append(btl.item())
                total_objectives.append(total_obj.item())
                nn_pred_errors_true_inputs.append(mse_true_inputs)
                nn_pred_errors_est_inputs.append(mse_est_inputs)

    # Final metrics (last-logged)
    out = {
        "epochs_logged": epochs_logged,
        "alpha_errors": alpha_errors,
        "nn_mses_centered": nn_mses_centered,
        "btl_likelihoods": btl_likelihoods,
        "total_objectives": total_objectives,
        "nn_pred_errors_true_inputs": nn_pred_errors_true_inputs,
        "nn_pred_errors_est_inputs": nn_pred_errors_est_inputs,
        "final": {
            "alpha_mse": alpha_errors[-1] if alpha_errors else None,
            "nn_mse_centered": nn_mses_centered[-1] if nn_mses_centered else None,
            "pred_mse_true_inputs": nn_pred_errors_true_inputs[-1] if nn_pred_errors_true_inputs else None,
            "pred_mse_est_inputs": nn_pred_errors_est_inputs[-1] if nn_pred_errors_est_inputs else None,
            "btl_likelihood": btl_likelihoods[-1] if btl_likelihoods else None,
            "total_objective": total_objectives[-1] if total_objectives else None,
        },
        "config": {
            "num_players": num_players,
            "num_timesteps": num_timesteps,
            "AR_order_p": AR_order_p,
            "epochs": epochs,
            "weight": weight,
            "erdos_renyi_p": erdos_renyi_p,
            "std_dev": std_dev,
            "N_grad_descent_alpha": N_grad_descent_alpha,
            "N_grad_descent_pred": N_grad_descent_pred,
            "lr_alpha": lr_alpha,
            "lr_pred": lr_pred,
            "STEP": STEP,
            "seed": seed,
        },
    }

    # Optional: save per-run plots (kept minimal)
    if save_plots_to is not None:
        import os, matplotlib.pyplot as plt
        os.makedirs(save_plots_to, exist_ok=True)

        tag = f"n{num_players}_T{num_timesteps}_w{weight:g}_p{AR_order_p}"
        x = epochs_logged if epochs_logged else [0]

        plt.figure(figsize=(10,4)); plt.plot(x, alpha_errors); plt.grid(True)
        plt.title("Alpha Error (normalized)"); plt.xlabel("Epoch"); plt.ylabel("MSE")
        plt.tight_layout(); plt.savefig(os.path.join(save_plots_to, f"{tag}_alpha_err.pdf")); plt.close()

        plt.figure(figsize=(10,4)); plt.plot(x, nn_mses_centered); plt.grid(True)
        plt.title("NN MSE (centered, est inputs)"); plt.xlabel("Epoch"); plt.ylabel("MSE")
        plt.tight_layout(); plt.savefig(os.path.join(save_plots_to, f"{tag}_nn_mse_centered.pdf")); plt.close()

        plt.figure(figsize=(10,4)); plt.plot(x, nn_pred_errors_true_inputs, label="true inputs")
        plt.plot(x, nn_pred_errors_est_inputs, label="est inputs"); plt.legend(); plt.grid(True)
        plt.title("NN Prediction MSE (centered)"); plt.xlabel("Epoch"); plt.ylabel("MSE")
        plt.tight_layout(); plt.savefig(os.path.join(save_plots_to, f"{tag}_nn_pred_mse.pdf")); plt.close()

    return out
