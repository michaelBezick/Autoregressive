"""
Hyperparameter sweep experiments for the Lagrange KKT model.

Covers three ablations, all sharing the same base configuration:
  weight   - ablates the AR regularization strength (includes weight=0 BTL baseline)
  sparsity - varies erdos_renyi_p to simulate sparser game observations
  noise    - varies std_dev to stress-test AR dynamics

Run from repo root:
    python Lagrange/sweep_experiments.py --sweep weight
    python Lagrange/sweep_experiments.py --sweep sparsity
    python Lagrange/sweep_experiments.py --sweep noise
    python Lagrange/sweep_experiments.py --sweep all
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from Lagrange.functions import (
    SkillParameters,
    center_scores,
    column_sum_residual,
    generate_next_skill_params,
    new_solve_for_phi_matrices,
    play_games_erdos_renyi,
    setup,
)

# ------------------------------------------------------------------
# Base configuration (fixed for all sweeps)
# ------------------------------------------------------------------
BASE = dict(
    num_players=20,
    AR_order_p=1,
    num_timesteps=30,
    erdos_renyi_p=1.0,
    std_dev=0.1,
    weight=30.0,
    epochs=500,
    N_grad_descent=10,
    lr=1e-3,
    ridge=1e-6,
)

# ------------------------------------------------------------------
# Sweep definitions
# ------------------------------------------------------------------
SWEEPS = {
    "weight":   {"key": "weight",        "values": [0.0, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]},
    "sparsity": {"key": "erdos_renyi_p", "values": [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]},
    "noise":    {"key": "std_dev",       "values": [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]},
}

XLABEL = {
    "weight":   "AR weight λ",
    "sparsity": "Erdős–Rényi edge probability",
    "noise":    "AR noise std dev",
}


def normalize_alpha(alpha: torch.Tensor) -> torch.Tensor:
    alpha_c = center_scores(alpha.clone())
    return alpha_c / torch.norm(alpha_c, dim=0, keepdim=True).clamp_min(1e-8)


def run_one(cfg: dict, seed: int, device: torch.device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    n   = cfg["num_players"]
    p   = cfg["AR_order_p"]
    T   = cfg["num_timesteps"]
    er  = cfg["erdos_renyi_p"]
    sig = cfg["std_dev"]
    w   = cfg["weight"]
    E   = cfg["epochs"]
    N   = cfg["N_grad_descent"]
    lr  = cfg["lr"]
    reg = cfg["ridge"]

    players = list(range(n))
    Z = torch.zeros((n, n, T), device=device)
    W = torch.zeros((n, n, T), device=device)

    initial, Phi_true = setup(n, p, device=device)
    skill_params = initial.copy()
    for t in range(T):
        next_s = generate_next_skill_params(skill_params, Phi_true, p, sig)
        skill_params.append(next_s)
        play_games_erdos_renyi(next_s, players, er, Z, W, t)

    actual = center_scores(torch.cat(skill_params, dim=1))

    skill_est = SkillParameters(n, T, p).to(device)
    skill_est.project_identifiability_()
    optimizer = torch.optim.Adam(skill_est.parameters(), lr=lr)
    _, Phi_est = setup(n, p, device=device)

    for _ in range(E):
        for _ in range(N):
            optimizer.zero_grad(set_to_none=True)
            BTL = skill_est.compute_log_BTL_vectorized_new(Z, W, n)
            AR_err = skill_est.compute_AR_error_new(Phi_est, T, p)
            (-BTL + w * AR_err).backward()
            optimizer.step()
            skill_est.project_identifiability_()

        with torch.no_grad():
            Phi_est = new_solve_for_phi_matrices(
                skill_est.alpha_estimates.detach(), n, p, T, ridge=reg
            )

    with torch.no_grad():
        alpha_mse = F.mse_loss(
            normalize_alpha(skill_est.alpha_estimates), normalize_alpha(actual)
        ).item()
        phi_mse = F.mse_loss(Phi_true, Phi_est).item()
        constraint_resid = column_sum_residual(Phi_est).item()

    return alpha_mse, phi_mse, constraint_resid


def run_sweep(name: str, n_seeds: int, device: torch.device) -> list:
    spec   = SWEEPS[name]
    key    = spec["key"]
    values = spec["values"]
    rows   = []

    with tqdm(total=len(values) * n_seeds, desc=name) as pbar:
        for val in values:
            cfg = {**BASE, key: val}
            for seed in range(n_seeds):
                alpha_mse, phi_mse, constraint_resid = run_one(cfg, seed, device)
                rows.append({
                    "sweep": name,
                    "param": key,
                    "value": val,
                    "seed": seed,
                    "alpha_mse": alpha_mse,
                    "phi_mse": phi_mse,
                    "constraint_residual": constraint_resid,
                })
                pbar.update(1)

    return rows


def save_csv(rows: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_sweep(rows: list, name: str, save_dir: Path):
    values = sorted(set(r["value"] for r in rows))
    save_dir.mkdir(parents=True, exist_ok=True)

    def gather(metric):
        means, stds = [], []
        for v in values:
            vals = [r[metric] for r in rows if r["value"] == v]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        return np.array(means), np.array(stds)

    def make_plot(means, stds, ylabel, filename, color):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(values, means, marker="o", linewidth=2, color=color)
        ax.fill_between(values, means - stds, means + stds, alpha=0.2, color=color)
        ax.set_xlabel(XLABEL[name])
        ax.set_ylabel(ylabel)
        ax.grid(True)
        if name == "weight":
            ax.set_xscale("symlog", linthresh=0.1)
        fig.tight_layout()
        fig.savefig(save_dir / filename)
        plt.close(fig)

    alpha_means, alpha_stds = gather("alpha_mse")
    phi_means, phi_stds     = gather("phi_mse")

    make_plot(alpha_means, alpha_stds, "Alpha MSE (normalized, mean ± std)",
              f"{name}_alpha_mse.pdf", "tab:blue")
    make_plot(phi_means, phi_stds, "Phi MSE (mean ± std)",
              f"{name}_phi_mse.pdf", "tab:orange")


def print_summary(rows: list, name: str):
    values = sorted(set(r["value"] for r in rows))
    key = SWEEPS[name]["key"]
    print(f"\n=== {name} sweep ({key}) ===")
    for v in values:
        mses = [r["alpha_mse"] for r in rows if r["value"] == v]
        print(f"  {key}={v:>6g}:  alpha_mse = {np.mean(mses):.4f} ± {np.std(mses):.4f}")


def parse_args():
    p = argparse.ArgumentParser(description="Hyperparameter sweeps for Lagrange KKT model.")
    p.add_argument("--sweep", choices=[*SWEEPS, "all"], required=True)
    p.add_argument("--n-seeds", type=int, default=5,
                   help="Number of random seeds per parameter value")
    p.add_argument("--save-dir", type=str, default="sweep_outputs",
                   help="Output directory (relative to this script)")
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device   = torch.device(args.device)
    save_dir = Path(os.path.dirname(os.path.abspath(__file__))) / args.save_dir

    names = list(SWEEPS) if args.sweep == "all" else [args.sweep]

    for name in names:
        rows = run_sweep(name, args.n_seeds, device)
        save_csv(rows, save_dir / f"{name}_results.csv")
        plot_sweep(rows, name, save_dir)
        print_summary(rows, name)

    print(f"\nAll outputs saved to {save_dir}")


if __name__ == "__main__":
    main()
