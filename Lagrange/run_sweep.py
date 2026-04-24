"""
Hyperparameter sweep for BTL + AR skill estimation model.

Sweeps: weight (λ), sparsity (Erdős–Rényi p), noise (AR std dev).
Each configuration is run across 5 random seeds.
Outputs CSVs to sweep_outputs/.
"""

import csv
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

# ── reproducible helpers ────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ── model primitives (CPU-only, simplified for sweep) ─────────────────────

def rescale_for_radius(Phi_list, target=0.99, iters=30):
    S = sum(Phi_list)
    v = torch.randn(S.size(0), 1)
    v = v / v.norm()
    for _ in range(iters):
        v = S @ v
        v = v / (v.norm() + 1e-8)
    lam = (v.T @ (S @ v) / (v.T @ v)).item()
    scale = target / max(abs(lam), 1e-8)
    return [scale * P for P in Phi_list]


def normalize_columns(M):
    return M / (M.sum(dim=0, keepdim=True) + 1e-8)


def setup(num_players, p):
    initial_skill_params = [torch.randn(num_players, 1) for _ in range(p)]
    initial_skill_params = [x - x.mean() for x in initial_skill_params]

    Phi_matrices = torch.randn(p, num_players, num_players)
    Phi_matrices = F.softmax(Phi_matrices, dim=1)
    Phi_list = [Phi_matrices[k] for k in range(p)]
    Phi_list = rescale_for_radius(Phi_list, target=0.995)
    Phi_list = [normalize_columns(P) for P in Phi_list]
    Phi_matrices = torch.stack(Phi_list, dim=0)
    return initial_skill_params, Phi_matrices


def generate_next_skill_params(prev, Phi, p, std_dev):
    last = torch.cat([prev[-p + i] for i in range(p)], dim=1)  # (n, p)
    last = last.T.unsqueeze(2)                                   # (p, n, 1)
    summed = torch.bmm(Phi, last).sum(dim=0)                    # (n, 1)
    return summed + torch.randn_like(summed) * std_dev


def play_games_erdos_renyi(skill_params, n, er_p, Z, W, t):
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < er_p:
                diff = skill_params[i] - skill_params[j]
                prob = torch.sigmoid(diff).item()
                outcome = np.random.binomial(1, min(max(prob, 1e-9), 1 - 1e-9))
                if outcome == 1:
                    Z[i, j, t] += 1
                else:
                    W[i, j, t] += 1


# ── SkillParameters module ─────────────────────────────────────────────────

import torch.nn as nn

class SkillParameters(nn.Module):
    def __init__(self, n, T, p):
        super().__init__()
        self.alpha = nn.Parameter(torch.randn(n, T + p))
        self.p = p

    def btl_loglik(self, Z, W, n):
        i_idx, j_idx = torch.triu_indices(n, n, offset=1)
        Z_pairs = Z[i_idx, j_idx, :]
        W_pairs = W[i_idx, j_idx, :]
        alpha = self.alpha[:, -Z.shape[-1]:]
        si, sj = alpha[i_idx], alpha[j_idx]
        log_den = torch.logaddexp(si, sj)
        return (Z_pairs * si + W_pairs * sj - (Z_pairs + W_pairs) * log_den).sum()

    def ar_error(self, Phi, T, p):
        est = self.alpha
        n = est.size(0)
        total, count = 0.0, 0
        for t in range(p, p + T):
            last = est[:, t - p:t].T.unsqueeze(2)        # (p, n, 1)
            pred = torch.bmm(Phi, last).sum(0).squeeze(1) # (n,)
            total = total + (pred - est[:, t]).pow(2).sum()
            count += n
        return total / (count + 1e-8)


def solve_phi(alpha, n, p, T):
    """Closed-form LS estimate of Phi (no constraint; fast for sweep)."""
    # alpha: (n, T+p)  ->  build Y = alpha[:, p:] and X lags
    # Use batched least squares per lag for speed
    alpha_t = alpha.permute(1, 0)  # (T+p, n)
    rows = []
    for t in range(p, p + T):
        rows.append(alpha_t[t - p:t].reshape(-1))  # (p*n,)
    X = torch.stack(rows, dim=0)   # (T, p*n)
    Y = alpha_t[p:p + T]           # (T, n)
    # LS: Phi_flat = (X^T X)^+ X^T Y, shape (p*n, n)
    Phi_flat = torch.linalg.lstsq(X, Y).solution  # (p*n, n)
    return Phi_flat.reshape(p, n, n)


def normalize_alpha(alpha):
    a = alpha.clone()
    a = a - a.mean(dim=0, keepdim=True)
    norms = a.norm(dim=0, keepdim=True).clamp(min=1e-8)
    return a / norms


# ── single run ─────────────────────────────────────────────────────────────

def run_one(seed, weight, er_p, std_dev,
            num_players=30, AR_order_p=3, num_timesteps=20,
            epochs=120, N_gd=10):
    set_seed(seed)

    skill_params, Phi_true = setup(num_players, AR_order_p)
    Z = torch.zeros(num_players, num_players, num_timesteps)
    W = torch.zeros(num_players, num_players, num_timesteps)

    for t in range(num_timesteps):
        nsp = generate_next_skill_params(skill_params, Phi_true, AR_order_p, std_dev)
        skill_params.append(nsp)
        play_games_erdos_renyi(nsp.squeeze(1), num_players, er_p, Z, W, t)

    actual = torch.stack(skill_params, dim=1).squeeze(2)  # (n, p+T)

    model = SkillParameters(num_players, num_timesteps, AR_order_p)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    Phi_est = torch.randn(AR_order_p, num_players, num_players)

    for epoch in range(epochs):
        for _ in range(N_gd):
            optimizer.zero_grad()
            ll = model.btl_loglik(Z, W, num_players)
            ar = model.ar_error(Phi_est, num_timesteps, AR_order_p) / num_players
            loss = -(ll - weight * ar)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            Phi_est = solve_phi(model.alpha.detach(), num_players, AR_order_p, num_timesteps)

    with torch.no_grad():
        alpha_est_norm = normalize_alpha(model.alpha[:, AR_order_p:])
        actual_norm    = normalize_alpha(actual[:, AR_order_p:])
        alpha_mse = F.mse_loss(alpha_est_norm, actual_norm).item()
        phi_mse   = F.mse_loss(Phi_est, Phi_true).item()
        ar_res    = model.ar_error(Phi_est, num_timesteps, AR_order_p).item()

    return alpha_mse, phi_mse, ar_res


# ── sweep definitions ──────────────────────────────────────────────────────

SEEDS   = [0, 1, 2, 3, 4]
WEIGHTS = [0, 0.1, 1, 3, 10, 30, 100, 300]
SPARSITIES = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
NOISES  = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]

DEFAULT_WEIGHT   = 10
DEFAULT_SPARSITY = 0.8
DEFAULT_NOISE    = 0.1

OUT_DIR = os.path.join(os.path.dirname(__file__), "sweep_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

FIELDNAMES = ["sweep", "param", "value", "seed", "alpha_mse", "phi_mse", "constraint_residual"]


def write_row(writer, sweep, param, value, seed, am, pm, cr):
    writer.writerow({
        "sweep": sweep, "param": param, "value": value,
        "seed": seed, "alpha_mse": f"{am:.6f}",
        "phi_mse": f"{pm:.6f}", "constraint_residual": f"{cr:.6f}",
    })


# ── weight sweep ───────────────────────────────────────────────────────────

print("=== Weight sweep ===")
with open(os.path.join(OUT_DIR, "weight_results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDNAMES)
    w.writeheader()
    for lam in WEIGHTS:
        for seed in SEEDS:
            print(f"  λ={lam}, seed={seed}", flush=True)
            am, pm, cr = run_one(seed, weight=lam, er_p=DEFAULT_SPARSITY, std_dev=DEFAULT_NOISE)
            write_row(w, "weight", "weight", lam, seed, am, pm, cr)

# ── sparsity sweep ─────────────────────────────────────────────────────────

print("=== Sparsity sweep ===")
with open(os.path.join(OUT_DIR, "sparsity_results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDNAMES)
    w.writeheader()
    for er_p in SPARSITIES:
        for seed in SEEDS:
            print(f"  p={er_p}, seed={seed}", flush=True)
            am, pm, cr = run_one(seed, weight=DEFAULT_WEIGHT, er_p=er_p, std_dev=DEFAULT_NOISE)
            write_row(w, "sparsity", "sparsity", er_p, seed, am, pm, cr)

# ── noise sweep ────────────────────────────────────────────────────────────

print("=== Noise sweep ===")
with open(os.path.join(OUT_DIR, "noise_results.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDNAMES)
    w.writeheader()
    for sigma in NOISES:
        for seed in SEEDS:
            print(f"  σ={sigma}, seed={seed}", flush=True)
            am, pm, cr = run_one(seed, weight=DEFAULT_WEIGHT, er_p=DEFAULT_SPARSITY, std_dev=sigma)
            write_row(w, "noise", "noise", sigma, seed, am, pm, cr)

print("Done. CSVs written to", OUT_DIR)
