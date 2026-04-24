import math as m
import random
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def center_scores(alpha: torch.Tensor) -> torch.Tensor:
    """Project each time slice to sum to zero for identifiability."""
    return alpha - alpha.mean(dim=0, keepdim=True)


class SkillParameters(nn.Module):
    def __init__(self, num_players: int, num_timesteps: int, AR_order_p: int) -> None:
        super().__init__()
        self.alpha_estimates = nn.Parameter(
            torch.randn((num_players, num_timesteps + AR_order_p)), requires_grad=True
        )
        self.AR_order_p = AR_order_p

    def project_identifiability_(self) -> None:
        with torch.no_grad():
            self.alpha_estimates.sub_(self.alpha_estimates.mean(dim=0, keepdim=True))

    def compute_log_BTL_vectorized_new(self, Z, W, num_players):
        i_idx, j_idx = torch.triu_indices(num_players, num_players, offset=1, device=Z.device)

        Z_pairs = Z[i_idx, j_idx, :]  # (pairs, T)
        W_pairs = W[i_idx, j_idx, :]  # (pairs, T)

        sliced_alpha = self.alpha_estimates[:, -Z.shape[-1] :]  # (n, T)
        s_i = sliced_alpha[i_idx, :]  # (pairs, T)
        s_j = sliced_alpha[j_idx, :]  # (pairs, T)

        log_den = torch.logaddexp(s_i, s_j)
        loglik = (Z_pairs * s_i + W_pairs * s_j - (Z_pairs + W_pairs) * log_den).sum()
        return loglik

    def compute_AR_error_new(self, Phi_matrices_estimate, num_timesteps, AR_order_p):
        """
        Mean squared one-step prediction error under the convention
            alpha_t \approx sum_{k=0}^{p-1} Phi_k alpha_{t-p+k}
        where Phi_0 multiplies the oldest lag and Phi_{p-1} the most recent lag.
        """
        est = self.alpha_estimates  # (n, T+p)
        n = est.size(0)
        total_se = est.new_tensor(0.0)
        count = 0

        for t in range(AR_order_p, AR_order_p + num_timesteps):
            last = est[:, t - AR_order_p : t].T.unsqueeze(2)  # (p, n, 1)
            pred = torch.bmm(Phi_matrices_estimate, last).sum(dim=0).squeeze(1)  # (n,)
            actual = est[:, t]
            total_se = total_se + (pred - actual).pow(2).sum()
            count += n

        return total_se / max(count, 1)


class MLP(nn.Module):
    def __init__(self, num_players, num_observations, p):
        super().__init__()
        h_dim = 100
        self.num_players = num_players
        self.p = p
        self.layers = nn.Sequential(
            nn.Linear(num_observations, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, num_players * p),
        )

    def forward(self, x):
        return self.layers(x).view(-1, self.num_players, self.p)



def optimize_alphas_until_converged(
    skill_params_est,
    Phi_matrices_estimate,
    Z,
    W,
    num_players,
    num_timesteps,
    AR_order_p,
    weight,
    optimizer,
    max_steps=5000,
    rel_tol=1e-3,
    grad_tol=5e-5,
    patience=20,
    grad_clip=5.0,
):
    """
    Runs the inner alpha optimization until convergence criteria are met.
    Returns (BTL_likelihood, AR_error, total_likelihood, steps_taken, grad_norm_last)
    """
    last_loss = None
    no_improve = 0
    steps_taken = 0
    gnorm_val = float("inf")

    for k in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)

        BTL_likelihood = skill_params_est.compute_log_BTL_vectorized_new(Z, W, num_players)
        AR_error = skill_params_est.compute_AR_error_new(
            Phi_matrices_estimate, num_timesteps, AR_order_p
        )

        BTL_likelihood = BTL_likelihood / m.comb(num_players, 2)
        total_likelihood = BTL_likelihood - weight * AR_error
        loss = -total_likelihood
        loss.backward()

        with torch.no_grad():
            sq_sum = 0.0
            for param in skill_params_est.parameters():
                if param.grad is not None:
                    sq_sum += (param.grad.detach() ** 2).sum().item()
            gnorm_val = sq_sum ** 0.5

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(skill_params_est.parameters(), grad_clip)

        optimizer.step()
        skill_params_est.project_identifiability_()

        cur = loss.detach().item()
        if last_loss is not None:
            rel_impr = (last_loss - cur) / max(1.0, abs(last_loss))
            if rel_impr < rel_tol:
                no_improve += 1
            else:
                no_improve = 0
        last_loss = cur
        steps_taken = k

        if gnorm_val < grad_tol or no_improve >= patience:
            break

    return (
        BTL_likelihood.detach(),
        AR_error.detach(),
        total_likelihood.detach(),
        steps_taken,
        gnorm_val,
    )



def btl_matrix_from_scores(s: torch.Tensor, diag: str = "half") -> torch.Tensor:
    s = s.flatten()
    diff = s[:, None] - s[None, :]
    P = torch.sigmoid(diff)

    if diag == "half":
        P.fill_diagonal_(0.5)
    elif diag == "nan":
        P.fill_diagonal_(float("nan"))
    elif diag == "zero":
        P.fill_diagonal_(0.0)
    else:
        raise ValueError("diag must be 'half', 'nan', or 'zero'")
    return P



def rescale_for_radius(Phi_list, target=0.99, iters=30):
    S = sum(Phi_list)
    device = S.device
    dtype = S.dtype
    v = torch.randn(S.size(0), 1, device=device, dtype=dtype)
    v = v / (v.norm() + 1e-8)
    for _ in range(iters):
        v = S @ v
        v = v / (v.norm() + 1e-8)
    lam = (v.T @ (S @ v) / (v.T @ v)).item()
    scale = target / max(abs(lam), 1e-8)
    return [scale * P for P in Phi_list]



def normalize_columns(M):
    return M / (M.sum(dim=0, keepdim=True) + 1e-8)



def identity_phi_matrices(num_players, p, device=None, dtype=torch.float32) -> torch.Tensor:
    if p != 1:
        raise ValueError("Identity Phi mode currently requires p=1.")
    if device is None:
        device = torch.device("cpu")
    eye = torch.eye(num_players, device=device, dtype=dtype)
    return eye.unsqueeze(0)



def setup(
    num_players,
    p,
    device=None,
    dtype=torch.float32,
    identity_ar1: bool = False,
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    if device is None:
        device = torch.device("cpu")

    if identity_ar1 and p != 1:
        raise ValueError("identity_ar1=True requires p=1.")

    initial_skill_params = [torch.randn((num_players, 1), device=device, dtype=dtype) for _ in range(p)]
    initial_skill_params = [alpha - alpha.mean() for alpha in initial_skill_params]

    if identity_ar1:
        Phi_matrices = identity_phi_matrices(num_players, p, device=device, dtype=dtype)
    else:
        Phi_matrices = torch.randn((p, num_players, num_players), device=device, dtype=dtype)
        Phi_matrices = F.softmax(Phi_matrices, dim=1)  # columns sum to 1

        Phi_list = [Phi_matrices[k] for k in range(p)]
        Phi_list = rescale_for_radius(Phi_list, target=0.995)
        Phi_list = [normalize_columns(P) for P in Phi_list]
        Phi_matrices = torch.stack(Phi_list, dim=0)

    return initial_skill_params, Phi_matrices



def play_game(skill_params, players, linearly_indexed_matrix):
    idx1, idx2 = random.sample(players, 2)
    diff = skill_params[idx1] - skill_params[idx2]
    prob_1_beats_2 = torch.sigmoid(diff)
    outcome = np.random.binomial(1, float(prob_1_beats_2.detach().cpu()))
    return linearly_indexed_matrix[idx1, idx2] if outcome == 1 else linearly_indexed_matrix[idx2, idx1]



def play_games_erdos_renyi(skill_params, players, p, Z, W, t):
    num_players = len(players)
    for i in range(num_players):
        for j in range(i + 1, num_players):
            if random.random() < p:
                diff = skill_params[i] - skill_params[j]
                prob_1_beats_2 = torch.sigmoid(diff)
                outcome = np.random.binomial(1, float(prob_1_beats_2.detach().cpu()))
                if outcome == 1:
                    Z[i, j, t] = 1
                else:
                    W[i, j, t] = 1



def generate_next_skill_params(
    previous_skill_params: List[torch.Tensor],
    Phi_matrices: torch.Tensor,
    p: int,
    std_dev: float,
):
    last_p_skill_params = [previous_skill_params[-p + i] for i in range(p)]
    last_p_skill_params = torch.cat(last_p_skill_params, dim=1)  # (n, p)
    last_p_skill_params = last_p_skill_params.permute(1, 0).unsqueeze(2)  # (p, n, 1)

    summed = torch.bmm(Phi_matrices, last_p_skill_params).sum(dim=0)  # (n, 1)
    summed_w_noise = summed + torch.randn_like(summed) * std_dev
    summed_w_noise = summed_w_noise - summed_w_noise.mean(dim=0, keepdim=True)
    return summed_w_noise



def _build_ar_design(all_skill_parameters: torch.Tensor, p: int, num_timesteps: int):
    """
    all_skill_parameters: (n, num_timesteps + p)

    Builds Y and X for the regression
        alpha_t \approx sum_{k=0}^{p-1} Phi_k alpha_{t-p+k}
    over t = p, ..., p + num_timesteps - 1.

    Returns
        Y: (n, num_timesteps)
        X: (p*n, num_timesteps)
    with X columns stacked as [alpha_{t-p}; alpha_{t-p+1}; ...; alpha_{t-1}].
    """
    if all_skill_parameters.ndim != 2:
        raise ValueError(
            f"Expected all_skill_parameters to have shape (n, T+p); got {tuple(all_skill_parameters.shape)}"
        )

    n, total_steps = all_skill_parameters.shape
    expected_steps = num_timesteps + p
    if total_steps < expected_steps:
        raise ValueError(
            f"Need at least {expected_steps} time columns for p={p} and num_timesteps={num_timesteps}; got {total_steps}."
        )

    windows = all_skill_parameters.unfold(dimension=1, size=p, step=1)  # (n, T+1, p)
    windows = windows[:, :num_timesteps, :]  # (n, T, p)
    X = windows.permute(2, 0, 1).contiguous().reshape(p * n, num_timesteps)
    Y = all_skill_parameters[:, p : p + num_timesteps]
    return Y, X



def solve_for_phi_matrices(
    all_skill_parameters: torch.Tensor,
    n: int,
    p: int,
    num_timesteps: int,
    ridge: float = 0.0,
) -> torch.Tensor:
    """
    Solve the constrained least-squares problem using the corrected horizontal-stack derivation.

    We solve for B = [Phi_0 ... Phi_{p-1}] in
        min_B ||Y - B X||_F^2
        s.t. B^T 1_n = 1_{pn}

    and then reshape B back into (p, n, n).
    """
    if all_skill_parameters.shape[0] != n:
        raise ValueError(
            f"Expected first dimension {n}, got {all_skill_parameters.shape[0]}"
        )

    device = all_skill_parameters.device
    dtype = all_skill_parameters.dtype

    Y, X = _build_ar_design(all_skill_parameters, p, num_timesteps)

    S = X @ X.T  # (pn, pn)
    if ridge > 0:
        S = S + ridge * torch.eye(p * n, device=device, dtype=dtype)
    R = Y @ X.T  # (n, pn)

    B_ols = R @ torch.linalg.pinv(S)

    ones_n = torch.ones(n, device=device, dtype=dtype)
    c = torch.ones(p * n, device=device, dtype=dtype)
    discrepancy = B_ols.T @ ones_n - c  # (pn,)

    B = B_ols - torch.outer(ones_n / n, discrepancy)
    Phi_matrices = B.reshape(n, p, n).permute(1, 0, 2).contiguous()
    return Phi_matrices



def new_solve_for_phi_matrices(
    all_skill_parameters: torch.Tensor,
    n: int,
    p: int,
    num_timesteps: int,
    ridge: float = 0.0,
) -> torch.Tensor:
    """Backward-compatible wrapper around the corrected solver."""
    return solve_for_phi_matrices(
        all_skill_parameters=all_skill_parameters,
        n=n,
        p=p,
        num_timesteps=num_timesteps,
        ridge=ridge,
    )



def column_sum_residual(Phi_matrices: torch.Tensor) -> torch.Tensor:
    """Max absolute violation of Phi_k^T 1 = 1 across all blocks."""
    n = Phi_matrices.size(1)
    ones_n = torch.ones(n, device=Phi_matrices.device, dtype=Phi_matrices.dtype)
    residual = Phi_matrices.transpose(1, 2) @ ones_n
    return (residual - 1.0).abs().max()
