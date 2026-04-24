import argparse
import csv
import json
import math as m
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from Lagrange.functions import (
    SkillParameters,
    center_scores,
    generate_next_skill_params,
    new_solve_for_phi_matrices,
    optimize_alphas_until_converged,
    play_games_erdos_renyi,
    setup,
)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def normalize_alpha(alpha: torch.Tensor) -> torch.Tensor:
    alpha_c = center_scores(alpha.clone())
    norms = torch.norm(alpha_c, dim=0, keepdim=True).clamp_min(1e-8)
    return alpha_c / norms



def average_log_likelihood(alpha: torch.Tensor, Z: torch.Tensor, W: torch.Tensor) -> float:
    num_players = alpha.shape[0]
    i_idx, j_idx = torch.triu_indices(num_players, num_players, offset=1, device=alpha.device)
    Z_pairs = Z[i_idx, j_idx, :]
    W_pairs = W[i_idx, j_idx, :]
    s_i = alpha[i_idx, :]
    s_j = alpha[j_idx, :]
    log_den = torch.logaddexp(s_i, s_j)
    raw_loglik = (Z_pairs * s_i + W_pairs * s_j - (Z_pairs + W_pairs) * log_den).sum()
    num_games = (Z_pairs + W_pairs).sum().clamp_min(1.0)
    return float((raw_loglik / num_games).detach().cpu())



def forecast_future(alpha_hist: torch.Tensor, Phi_matrices: torch.Tensor, horizon: int) -> torch.Tensor:
    """
    alpha_hist: (n, train_T + p)
    Phi_matrices: (p, n, n)
    returns forecasted future alphas of shape (n, horizon)
    """
    if horizon <= 0:
        return alpha_hist.new_zeros((alpha_hist.shape[0], 0))

    p = Phi_matrices.shape[0]
    history = [alpha_hist[:, t : t + 1].clone() for t in range(alpha_hist.shape[1])]
    preds = []

    for _ in range(horizon):
        last_p = torch.cat(history[-p:], dim=1).permute(1, 0).unsqueeze(2)  # (p, n, 1)
        next_alpha = torch.bmm(Phi_matrices, last_p).sum(dim=0)  # (n, 1)
        next_alpha = next_alpha - next_alpha.mean(dim=0, keepdim=True)
        preds.append(next_alpha)
        history.append(next_alpha)

    return torch.cat(preds, dim=1)



def simulate_dataset(
    num_players: int,
    true_p: int,
    num_timesteps: int,
    erdos_renyi_p: float,
    std_dev: float,
    device: torch.device,
):
    players = list(range(num_players))
    Z = torch.zeros((num_players, num_players, num_timesteps), device=device)
    W = torch.zeros((num_players, num_players, num_timesteps), device=device)

    initial_skill_params, Phi_true = setup(num_players, true_p, device=device)
    skill_params = initial_skill_params.copy()

    for t in range(num_timesteps):
        next_skill_params = generate_next_skill_params(skill_params, Phi_true, true_p, std_dev)
        skill_params.append(next_skill_params)
        play_games_erdos_renyi(next_skill_params, players, erdos_renyi_p, Z, W, t)

    actual_skill_params = center_scores(torch.cat(skill_params, dim=1))  # (n, true_p + T)
    return Z, W, actual_skill_params, Phi_true



def fit_candidate_model(
    Z_train: torch.Tensor,
    W_train: torch.Tensor,
    num_players: int,
    p: int,
    weight: float,
    lr: float,
    ridge: float,
    outer_epochs: int,
    inner_max_steps: int,
    rel_tol: float,
    grad_tol: float,
    patience: int,
    grad_clip: float,
    device: torch.device,
):
    train_T = Z_train.shape[-1]
    if train_T <= 0:
        raise ValueError("Training window must contain at least one timestep.")

    skill_params_est = SkillParameters(num_players, train_T, p).to(device)
    skill_params_est.project_identifiability_()
    optimizer = torch.optim.Adam(skill_params_est.parameters(), lr=lr)

    _, Phi_est = setup(num_players, p, device=device)

    history = []
    for outer_epoch in range(outer_epochs):
        BTL_likelihood, AR_error, total_likelihood, steps_taken, grad_norm_last = (
            optimize_alphas_until_converged(
                skill_params_est=skill_params_est,
                Phi_matrices_estimate=Phi_est,
                Z=Z_train,
                W=W_train,
                num_players=num_players,
                num_timesteps=train_T,
                AR_order_p=p,
                weight=weight,
                optimizer=optimizer,
                max_steps=inner_max_steps,
                rel_tol=rel_tol,
                grad_tol=grad_tol,
                patience=patience,
                grad_clip=grad_clip,
            )
        )

        with torch.no_grad():
            Phi_est = new_solve_for_phi_matrices(
                all_skill_parameters=skill_params_est.alpha_estimates.detach(),
                n=num_players,
                p=p,
                num_timesteps=train_T,
                ridge=ridge,
            )

        history.append(
            {
                "outer_epoch": outer_epoch,
                "btl_likelihood": float(BTL_likelihood.cpu()),
                "ar_error": float(AR_error.cpu()),
                "total_likelihood": float(total_likelihood.cpu()),
                "inner_steps": steps_taken,
                "grad_norm_last": grad_norm_last,
            }
        )

    return {
        "alpha_est": skill_params_est.alpha_estimates.detach().clone(),
        "phi_est": Phi_est.detach().clone(),
        "history": history,
    }



def split_time_series(num_timesteps: int, train_frac: float, val_frac: float):
    train_T = max(1, int(round(train_frac * num_timesteps)))
    val_T = max(1, int(round(val_frac * num_timesteps)))

    if train_T + val_T >= num_timesteps:
        val_T = max(1, num_timesteps - train_T - 1)
    test_T = num_timesteps - train_T - val_T

    if test_T <= 0:
        raise ValueError(
            "Need positive test horizon. Increase num_timesteps or reduce train/val fractions."
        )

    return train_T, val_T, test_T



def evaluate_one_replication(
    rep_seed: int,
    true_p: int,
    candidate_ps,
    args,
    device: torch.device,
):
    set_all_seeds(rep_seed)

    Z, W, actual_skill_params, Phi_true = simulate_dataset(
        num_players=args.num_players,
        true_p=true_p,
        num_timesteps=args.num_timesteps,
        erdos_renyi_p=args.erdos_renyi_p,
        std_dev=args.std_dev,
        device=device,
    )

    train_T, val_T, test_T = split_time_series(args.num_timesteps, args.train_frac, args.val_frac)

    Z_train = Z[:, :, :train_T]
    W_train = W[:, :, :train_T]
    Z_val = Z[:, :, train_T : train_T + val_T]
    W_val = W[:, :, train_T : train_T + val_T]
    Z_trainval = Z[:, :, : train_T + val_T]
    W_trainval = W[:, :, : train_T + val_T]
    Z_test = Z[:, :, train_T + val_T :]
    W_test = W[:, :, train_T + val_T :]

    candidate_rows = []
    valid_candidate_ps = [p for p in candidate_ps if p < train_T]
    if not valid_candidate_ps:
        raise ValueError("All candidate p values are too large for the training window.")

    for p in valid_candidate_ps:
        fit = fit_candidate_model(
            Z_train=Z_train,
            W_train=W_train,
            num_players=args.num_players,
            p=p,
            weight=args.weight,
            lr=args.lr,
            ridge=args.ridge,
            outer_epochs=args.outer_epochs,
            inner_max_steps=args.inner_max_steps,
            rel_tol=args.rel_tol,
            grad_tol=args.grad_tol,
            patience=args.patience,
            grad_clip=args.grad_clip,
            device=device,
        )

        val_forecast = forecast_future(fit["alpha_est"], fit["phi_est"], val_T)
        val_avg_loglik = average_log_likelihood(val_forecast, Z_val, W_val)

        val_true_alpha = actual_skill_params[:, true_p + train_T : true_p + train_T + val_T]
        val_alpha_mse = float(
            F.mse_loss(normalize_alpha(val_forecast), normalize_alpha(val_true_alpha)).cpu()
        )

        candidate_rows.append(
            {
                "rep_seed": rep_seed,
                "true_p": true_p,
                "candidate_p": p,
                "val_avg_loglik": val_avg_loglik,
                "val_alpha_mse": val_alpha_mse,
            }
        )

    best_row = max(candidate_rows, key=lambda row: row["val_avg_loglik"])
    selected_p = int(best_row["candidate_p"])

    refit = fit_candidate_model(
        Z_train=Z_trainval,
        W_train=W_trainval,
        num_players=args.num_players,
        p=selected_p,
        weight=args.weight,
        lr=args.lr,
        ridge=args.ridge,
        outer_epochs=args.outer_epochs,
        inner_max_steps=args.inner_max_steps,
        rel_tol=args.rel_tol,
        grad_tol=args.grad_tol,
        patience=args.patience,
        grad_clip=args.grad_clip,
        device=device,
    )

    test_forecast = forecast_future(refit["alpha_est"], refit["phi_est"], test_T)
    test_avg_loglik = average_log_likelihood(test_forecast, Z_test, W_test)
    test_true_alpha = actual_skill_params[:, true_p + train_T + val_T : true_p + args.num_timesteps]
    test_alpha_mse = float(
        F.mse_loss(normalize_alpha(test_forecast), normalize_alpha(test_true_alpha)).cpu()
    )

    selected_matches_true = int(selected_p == true_p)

    return {
        "rep_seed": rep_seed,
        "true_p": true_p,
        "selected_p": selected_p,
        "selected_matches_true": selected_matches_true,
        "best_val_avg_loglik": float(best_row["val_avg_loglik"]),
        "test_avg_loglik": test_avg_loglik,
        "test_alpha_mse": test_alpha_mse,
        "phi_true_fro_norm": float(torch.norm(Phi_true).cpu()),
        "candidate_rows": candidate_rows,
    }



def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)



def summarize_results(replication_rows):
    grouped = defaultdict(list)
    for row in replication_rows:
        grouped[row["true_p"]].append(row)

    summary_rows = []
    for true_p, rows in sorted(grouped.items()):
        selection_acc = sum(r["selected_matches_true"] for r in rows) / len(rows)
        mean_test_loglik = sum(r["test_avg_loglik"] for r in rows) / len(rows)
        mean_test_mse = sum(r["test_alpha_mse"] for r in rows) / len(rows)
        selection_hist = defaultdict(int)
        for r in rows:
            selection_hist[r["selected_p"]] += 1

        summary_rows.append(
            {
                "true_p": true_p,
                "num_replications": len(rows),
                "selection_accuracy": selection_acc,
                "mean_test_avg_loglik": mean_test_loglik,
                "mean_test_alpha_mse": mean_test_mse,
                "selected_p_histogram": json.dumps(dict(sorted(selection_hist.items()))),
            }
        )
    return summary_rows



def parse_args():
    parser = argparse.ArgumentParser(description="Synthetic model-selection experiment for AR-BTL order p.")
    parser.add_argument("--true-ps", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--candidate-ps", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 8, 10])
    parser.add_argument("--num-reps", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--num-players", type=int, default=20)
    parser.add_argument("--num-timesteps", type=int, default=60)
    parser.add_argument("--erdos-renyi-p", type=float, default=1.0)
    parser.add_argument("--std-dev", type=float, default=1e-1)
    parser.add_argument("--train-frac", type=float, default=0.6)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--weight", type=float, default=30.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--outer-epochs", type=int, default=25)
    parser.add_argument("--inner-max-steps", type=int, default=500)
    parser.add_argument("--rel-tol", type=float, default=1e-3)
    parser.add_argument("--grad-tol", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-dir", type=str, default="model_selection_results")
    return parser.parse_args()



def main():
    args = parse_args()
    device = torch.device(args.device)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows_all = []
    replication_rows = []

    total_runs = len(args.true_ps) * args.num_reps
    with tqdm(total=total_runs, desc="Replications") as pbar:
        for true_p in args.true_ps:
            for rep in range(args.num_reps):
                rep_seed = args.base_seed + 10_000 * true_p + rep
                result = evaluate_one_replication(rep_seed, true_p, args.candidate_ps, args, device)
                replication_rows.append(
                    {
                        "rep_seed": result["rep_seed"],
                        "true_p": result["true_p"],
                        "selected_p": result["selected_p"],
                        "selected_matches_true": result["selected_matches_true"],
                        "best_val_avg_loglik": result["best_val_avg_loglik"],
                        "test_avg_loglik": result["test_avg_loglik"],
                        "test_alpha_mse": result["test_alpha_mse"],
                        "phi_true_fro_norm": result["phi_true_fro_norm"],
                    }
                )
                candidate_rows_all.extend(result["candidate_rows"])
                pbar.update(1)

    summary_rows = summarize_results(replication_rows)

    write_csv(save_dir / "candidate_scores.csv", candidate_rows_all)
    write_csv(save_dir / "replication_results.csv", replication_rows)
    write_csv(save_dir / "summary.csv", summary_rows)

    config_path = save_dir / "config.json"
    with config_path.open("w") as f:
        json.dump(vars(args), f, indent=2)

    print("\n=== Summary ===")
    for row in summary_rows:
        print(
            f"true_p={row['true_p']}: "
            f"selection_accuracy={row['selection_accuracy']:.3f}, "
            f"mean_test_avg_loglik={row['mean_test_avg_loglik']:.4f}, "
            f"mean_test_alpha_mse={row['mean_test_alpha_mse']:.4f}, "
            f"hist={row['selected_p_histogram']}"
        )
    print(f"\nSaved results to: {save_dir.resolve()}")


if __name__ == "__main__":
    main()
