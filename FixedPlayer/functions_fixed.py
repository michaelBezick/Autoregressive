import math as m
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    max_steps=5000,      # hard cap
    rel_tol=1e-3,        # relative loss improvement tolerance
    grad_tol=5e-5,       # gradient norm tolerance
    patience=20,         # stop if no meaningful improvement for this many steps
    grad_clip=5.0        # set None to disable
):
    """
    Runs the inner 'alpha' optimization until convergence criteria are met.
    Returns (BTL_likelihood, AR_error, total_likelihood, steps_taken, grad_norm_last)
    """
    last_loss = None
    no_improve = 0
    steps_taken = 0
    gnorm_val = float("inf")

    for k in range(1, max_steps + 1):

        optimizer.zero_grad(set_to_none=True)

        # compute objective parts
        BTL_likelihood = skill_params_est.compute_log_BTL_vectorized_new(Z, W, num_players)
        AR_error = skill_params_est.compute_AR_error_new(Phi_matrices_estimate, num_timesteps, AR_order_p)

        # normalize per game
        BTL_likelihood /= m.comb(num_players, 2)

        # normalize per player as before
        AR_error = AR_error / num_players

        total_likelihood = BTL_likelihood - weight * AR_error
        loss = -total_likelihood

        # backprop
        loss.backward()
        print(BTL_likelihood, AR_error, loss)

        # gradient norm for a stall criterion (computed BEFORE the step)
        with torch.no_grad():
            sq_sum = 0.0
            for p in skill_params_est.parameters():
                if p.grad is not None:
                    sq_sum += (p.grad.detach()**2).sum().item()
            gnorm_val = (sq_sum ** 0.5)

        # optional clipping
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(skill_params_est.parameters(), grad_clip)

        optimizer.step()

        # check relative improvement
        cur = loss.detach().item()
        if last_loss is not None:
            rel_impr = (last_loss - cur) / max(1.0, abs(last_loss))
            if rel_impr < rel_tol:
                no_improve += 1
            else:
                no_improve = 0
        last_loss = cur

        steps_taken = k

        # stop if either grads got tiny or improvement stalled for a while
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
    """
    s: shape (n,) or (n,1). Log-abilities.
    Returns P (n x n) with P[i,j] = Pr(i beats j) under BTL.
    diag: "half" -> P[ii]=0.5, "nan" -> NaN, "zero" -> 0.0
    """
    s = s.flatten()  # (n,)
    diff = s[:, None] - s[None, :]  # (n, n)
    P = torch.sigmoid(diff)  # (n, n); P_ij = 1 - P_ji

    if diag == "half":
        P.fill_diagonal_(0.5)
    elif diag == "nan":
        P.fill_diagonal_(float("nan"))
    elif diag == "zero":
        P.fill_diagonal_(0.0)
    else:
        raise ValueError("diag must be 'half', 'nan', or 'zero'")
    return P

import torch
import torch.nn as nn

class SkillParametersOneFixed(nn.Module):
    """
    Skill parameters where player 0 is fixed to skill 0 at all timesteps.
    We only learn skills for players 1..(num_players-1).
    """
    def __init__(self, num_players, num_timesteps, AR_order_p) -> None:
        super().__init__()

        assert num_players >= 1, "Need at least one player"

        self.num_players = num_players
        self.num_timesteps = num_timesteps
        self.AR_order_p = AR_order_p
        self.total_T = num_timesteps + AR_order_p

        # Learnable skills for players 1..(num_players-1)
        self.alpha_free = nn.Parameter(
            torch.randn((num_players - 1, self.total_T)),  # (n-1, T+p)
            requires_grad=True,
        )

    @property
    def alpha_estimates(self):
        """
        Full skill matrix (num_players, T+p), with player 0 fixed to 0.
        This is what you should use everywhere instead of a raw parameter.
        """
        # Row for the fixed player 0: identically zero, no gradient
        zero_row = self.alpha_free.new_zeros(1, self.total_T)  # (1, T+p)

        # Concatenate: player 0 on top, then learned players 1..(n-1)
        # Gradients only flow into self.alpha_free.
        return torch.cat([zero_row, self.alpha_free], dim=0)  # (n, T+p)

    def compute_log_BTL(self, Z, W, num_players):
        # Optionally sanity-check
        assert num_players == self.num_players

        i_idx, j_idx = torch.triu_indices(num_players, num_players, offset=1)

        Z_pairs = Z[i_idx, j_idx, :]  # (pairs, T)
        W_pairs = W[i_idx, j_idx, :]  # (pairs, T)

        # Use the *full* alpha with player 0 fixed to 0
        alpha = self.alpha_estimates  # (n, T+p)
        sliced_alpha = alpha[:, -Z.shape[-1]:]  # (n, T)
        s_i = sliced_alpha[i_idx, :]            # (pairs, T)
        s_j = sliced_alpha[j_idx, :]            # (pairs, T)

        log_den = torch.logaddexp(s_i, s_j)
        loglik = (Z_pairs * s_i + W_pairs * s_j - (Z_pairs + W_pairs) * log_den).sum()
        return loglik

    def compute_AR_error(self, Phi_matrices_estimate, num_timesteps, AR_order_p):
        """
        AR error only over players 1..(n-1).
        Player 0 is just the reference and is NOT part of the AR process.
        """
        est_full = self.alpha_estimates           # (n, T+p), row 0 fixed at 0
        est = est_full[1:, :]                     # (n_free, T+p)
        n_free = est.size(0)
        total_se = est.new_tensor(0.0)
        count = 0

        # Allow Phi to be either (p, n, n) or already (p, n_free, n_free)
        if Phi_matrices_estimate.size(1) == self.num_players:
            # Φ was built for all n players: drop row/col 0
            Phi_used = Phi_matrices_estimate[:, 1:, 1:]   # (p, n_free, n_free)
        else:
            # Assume it was built only for free players already
            Phi_used = Phi_matrices_estimate
            assert Phi_used.size(1) == n_free and Phi_used.size(2) == n_free

        # Use same time window as BTL target columns: [p, p+num_timesteps)
        for t in range(AR_order_p, AR_order_p + num_timesteps):
            # history of last p months, free players only
            # est[:, t-p:t]: (n_free, p) -> (p, n_free, 1)
            last = est[:, t - AR_order_p : t].T.unsqueeze(2)

            # Φ: (p, n_free, n_free); bmm -> (p, n_free, 1); sum over p -> (n_free, 1)
            pred = torch.bmm(Phi_used, last).sum(dim=0).squeeze(1)  # (n_free,)
            actual = est[:, t]                                      # (n_free,)

            se = (pred - actual).pow(2).sum()
            total_se = total_se + se
            count += n_free

        # MSE over the n-1 free players and all time points
        return total_se / (count + 1e-8)





class SkillParameters(nn.Module):
    def __init__(self, num_players, num_timesteps, AR_order_p) -> None:
        super().__init__()
        self.alpha_estimates = nn.Parameter(
            torch.randn((num_players, num_timesteps + AR_order_p)), requires_grad=True
        )

        self.AR_order_p = AR_order_p

    def compute_log_BTL_vectorized_new(self, Z, W, num_players):
        i_idx, j_idx = torch.triu_indices(num_players, num_players, offset=1)

        Z_pairs = Z[i_idx, j_idx, :]  # (pairs, T)
        W_pairs = W[i_idx, j_idx, :]  # (pairs, T)

        sliced_alpha = self.alpha_estimates[:, -Z.shape[-1]:]  # (n, T)
        s_i = sliced_alpha[i_idx, :]         # (pairs, T)
        s_j = sliced_alpha[j_idx, :]         # (pairs, T)

        # Stable: logaddexp
        log_den = torch.logaddexp(s_i, s_j)
        # Sum over all observed comparisons
        loglik = (Z_pairs * s_i + W_pairs * s_j - (Z_pairs + W_pairs) * log_den).sum()
        return loglik
    def compute_log_BTL_vectorized(self, Z, W, num_players):
        i_idx, j_idx = torch.triu_indices(num_players, num_players, offset=1)

        Z_pairs = Z[i_idx, j_idx, :]  # shape: (num_pairs, num_timesteps)
        W_pairs = W[i_idx, j_idx, :]  # shape: (num_pairs, num_timesteps)

        sliced_alpha = self.alpha_estimates[:, -Z.shape[-1] :]  # match time dimension
        s_i = sliced_alpha[i_idx, :]  # (num_pairs, num_timesteps)
        s_j = sliced_alpha[j_idx, :]  # (num_pairs, num_timesteps)

        log_BTL_sum = (
            Z_pairs * s_i
            + W_pairs * s_j
            - (Z_pairs + W_pairs) * torch.log(torch.exp(s_i) + torch.exp(s_j))
        ).sum()

        return log_BTL_sum

    def compute_log_BTL(self, Z, W, num_players, num_timesteps):

        skill_param_estimates = self.alpha_estimates

        # I believe num_players[a][b] means ath player and bth timestep

        log_BTL_sum = 0
        for t in range(num_timesteps):
            for i in range(num_players):
                for j in range(i + 1, num_players):
                    log_BTL_sum += (
                        Z[i, j, t] * skill_param_estimates[i][t + self.AR_order_p]
                        + W[i, j, t] * skill_param_estimates[j][t + self.AR_order_p]
                        - (Z[i, j, t] + W[i, j, t])
                        * torch.log(
                            m.e ** (skill_param_estimates[i][t + self.AR_order_p])
                            + m.e ** (skill_param_estimates[j][t + self.AR_order_p])
                        )
                    )

        return log_BTL_sum

    def compute_AR_error_new(self, Phi_matrices_estimate, num_timesteps, AR_order_p):
        est = self.alpha_estimates  # (n, T+p)
        n = est.size(0)
        total_se = 0.0
        count = 0

        # Use same time window as BTL target columns: [p, p+num_timesteps)
        for t in range(AR_order_p, AR_order_p + num_timesteps):
            # shape: (p, n, 1)
            last = est[:, t - AR_order_p : t].T.unsqueeze(2)
            # Φ: (p, n, n); bmm -> (p, n, 1); sum over p -> (n, 1)
            pred = torch.bmm(Phi_matrices_estimate, last).sum(dim=0).squeeze(1)  # (n,)
            actual = est[:, t]  # (n,)

            se = (pred - actual).pow(2).sum()
            total_se = total_se + se
            count += n

        # Return MSE to keep scale comparable
        return total_se / (count + 1e-8)
    def compute_AR_error(self, Phi_matrices_estimate, num_timesteps, AR_order_p):

        skill_param_estimates = self.alpha_estimates

        # players are independent, but for now we will sum up the MSE for all players

        total_error = 0

        for t in range(AR_order_p, AR_order_p + num_timesteps):
            # get block of previous p skill parameters, assuming first index is timestep (check later)
            last_p_skill_params = skill_param_estimates[:, t - AR_order_p : t]
            last_p_skill_params = torch.permute(last_p_skill_params, (1, 0)).unsqueeze(
                2
            )

            summed = torch.bmm(Phi_matrices_estimate, last_p_skill_params).sum(dim=0)

            actual = skill_param_estimates[:, t]

            squared_errors = (summed - actual) ** 2

            total_squared_errors = squared_errors.sum()

            total_error += total_squared_errors

        return total_error


class MLP(nn.Module):
    def __init__(self, num_players, num_observations, p):

        super().__init__()

        h_dim = 100
        self.num_players = num_players
        self.p = p

        self.layers = nn.Sequential(
            nn.Linear(num_observations, h_dim),
            nn.ReLU(),
            # nn.BatchNorm1d(h_dim),
            # nn.Linear(h_dim, h_dim),
            # nn.ReLU(),
            # nn.BatchNorm1d(h_dim),
            nn.Linear(h_dim, num_players * p),
        )

    def forward(self, x):

        return self.layers(x).view(-1, self.num_players, self.p)


def rescale_for_radius(Phi_list, target=0.99, iters=30):
    S = sum(Phi_list)                        # n x n
    v = torch.randn(S.size(0), 1)
    v = v / v.norm()
    for _ in range(iters):                   # power iteration
        v = S @ v
        v = v / (v.norm() + 1e-8)
    lam = (v.T @ (S @ v) / (v.T @ v)).item() # Rayleigh quotient
    scale = target / max(abs(lam), 1e-8)
    return [scale * P for P in Phi_list]

def normalize_columns(M):
    return M / (M.sum(dim=0, keepdim=True) + 1e-8)

def setup(num_players, p) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Initialize:
      - initial_skill_params: list of length p, each (num_players, 1),
        with player 0 fixed to 0 and players 1..(n-1) mean-centered.
      - Phi_matrices: (p, num_players-1, num_players-1), AR only on players 1..(n-1).
    """

    n_free = num_players - 1

    # --- Initial skill parameters (T = p "history" timesteps) ---
    initial_skill_params: List[torch.Tensor] = []
    for _ in range(p):
        # Random skills for free players
        free_skills = torch.randn((n_free, 1))
        # Center them so they have mean 0 (over players 1..n-1)
        free_skills = free_skills - free_skills.mean()

        # Player 0 is fixed at 0
        skill_vec = torch.cat(
            [torch.zeros(1, 1),  # player 0
             free_skills],       # players 1..n-1
            dim=0,
        )  # shape: (num_players, 1)

        initial_skill_params.append(skill_vec)

    # Phi_matrices = torch.randn((p, n_free, n_free))
    #
    # # Softmax over rows (dim=1) as a starting point
    # Phi_matrices = F.softmax(Phi_matrices, dim=1)
    #
    # # Spectral rescaling to keep the AR process stable
    # Phi_list = [Phi_matrices[k] for k in range(p)]
    # Phi_list = rescale_for_radius(Phi_list, target=0.995)
    #
    # # Renormalize columns to sum to 1
    # Phi_list = [normalize_columns(P) for P in Phi_list]
    # Phi_matrices = torch.stack(Phi_list, dim=0)  # (p, n_free, n_free)

    Phi_matrices = [torch.eye(n_free) for _ in range(p)]
    Phi_matrices = torch.stack(Phi_matrices, dim=0)

    return initial_skill_params, Phi_matrices


def play_game(skill_params, players, linearly_indexed_matrix):

    # choose random combination
    combination = random.sample(players, 2)  # check if uniform

    idx1, idx2 = combination

    player1_skill = skill_params[idx1]
    player2_skill = skill_params[idx2]

    diff = player1_skill - player2_skill
    prob_1_beats_2 = torch.sigmoid(diff)

    try:
        outcome = np.random.binomial(1, float(prob_1_beats_2))
    except:
        print(player1_skill)
        print(player2_skill)
        print(prob_1_beats_2)
        exit()

    if outcome == 1:
        # 1 beat 2
        return linearly_indexed_matrix[idx1, idx2]
    else:
        return linearly_indexed_matrix[idx2, idx1]
def play_games_erdos_renyi(
    skill_params,
    players,
    p,
    Z,
    W,
    t,
    games_per_pair: int = 1,
):
    """
    For each pair (i,j) at timestep t, play up to `games_per_pair` games.
    Z[i,j,t] and W[i,j,t] store *counts* of wins, not just 0/1.
    """
    num_players = len(players)

    for i in range(num_players):
        for j in range(i + 1, num_players):

            for _ in range(games_per_pair):
                if random.random() < p:
                    # play 1 game
                    player1_skill = skill_params[i]
                    player2_skill = skill_params[j]

                    diff = player1_skill - player2_skill
                    prob_1_beats_2 = torch.sigmoid(diff)

                    outcome = np.random.binomial(1, float(prob_1_beats_2))

                    if outcome == 1:
                        Z[i, j, t] += 1  # increment
                    else:
                        W[i, j, t] += 1  # increment


def generate_next_skill_params(
    previous_skill_params: List[torch.Tensor],
    Phi_matrices: torch.Tensor,
    p: int,
    std_dev: float,
):
    """
    One-step AR update of skill parameters, but ONLY for players 1..(n-1).
    Player 0 is treated as a fixed reference (kept at 0 here).

    previous_skill_params: list of length ≥ p, each (n, 1) or (n,)
    Phi_matrices: (p, n, n) for all players OR (p, n-1, n-1) for free players
    returns: (n, 1) new skill vector
    """

    # Take last p skill vectors
    last_p_skill_params = [previous_skill_params[-p + i] for i in range(p)]

    # Ensure each is (n, 1)
    last_p_skill_params = [
        s if s.dim() == 2 else s.unsqueeze(1) for s in last_p_skill_params
    ]  # each (n, 1)

    # Stack across time: (n, p)
    last_p_skill_params = torch.cat(last_p_skill_params, dim=1)

    n = last_p_skill_params.size(0)
    if n == 1:
        # Degenerate case: only the reference player
        summed = torch.zeros((1, 1),
                             device=last_p_skill_params.device,
                             dtype=last_p_skill_params.dtype)
        return summed + torch.randn_like(summed) * std_dev

    # (p, n, 1): time lags × players × 1
    last_p_full = last_p_skill_params.permute(1, 0).unsqueeze(2)

    # Restrict AR to players 1..(n-1)
    n_free = n - 1
    last_p_free = last_p_full[:, 1:, :]  # (p, n_free, 1)

    # Handle Φ shape: either (p, n, n) or already (p, n_free, n_free)
    if Phi_matrices.size(1) == n:
        Phi_used = Phi_matrices[:, 1:, 1:]          # (p, n_free, n_free)
    else:
        Phi_used = Phi_matrices
        assert Phi_used.size(1) == n_free and Phi_used.size(2) == n_free, \
            "Phi_matrices must be (p, n, n) or (p, n-1, n-1)"

    # AR prediction for free players: (p, n_free, 1) -> sum over p -> (n_free, 1)
    free_pred = torch.bmm(Phi_used, last_p_free).sum(dim=0)  # (n_free, 1)
    free_pred = free_pred + torch.randn_like(free_pred) * std_dev

    # Assemble full (n, 1) vector with player 0 fixed to 0
    next_skill = torch.zeros(
        (n, 1),
        device=free_pred.device,
        dtype=free_pred.dtype,
    )
    # player 0 stays at 0 by construction
    next_skill[1:, :] = free_pred

    return next_skill



def new_solve_for_phi_matrices(
    all_skill_parameters: torch.Tensor, n: int, p: int, num_timesteps: int
):
    all_skill_parameters = all_skill_parameters.permute(
        (1, 0)
    )  # (timesteps + AR_order) x (num_players)

    # first, create the pxp matrix of A matrices
    matrix_of_A_matrices = torch.zeros((p, p, n, n), device="cuda")  # 4D first

    for k in range(p):
        for i in range(p):
            A_ki = torch.zeros((n, n), device="cuda")
            for t in range(num_timesteps):
                A_ki += torch.outer(
                    all_skill_parameters[p + t - k], all_skill_parameters[p + t - i]
                )

            matrix_of_A_matrices[k, i] = A_ki

    # second, create a px1 vector of A matrices
    vector_of_A_matrices = torch.zeros((p, n, n), device="cuda")
    for i in range(p):
        A_neg_1_k = torch.zeros((n, n), device="cuda")

        for t in range(
            num_timesteps - 1
        ):  # -1 because formula would otherwise go past end

            A_neg_1_k += torch.outer(
                all_skill_parameters[p + t + 1], all_skill_parameters[p + t - i]
            )

        vector_of_A_matrices[i] = A_neg_1_k

    # third, solve for Lagrange multipliers
    matrix_of_A_matrices_2D = (
        matrix_of_A_matrices.permute(0, 2, 1, 3).contiguous().view(p * n, p * n)
    )

    vector_of_A_matrices_2D = vector_of_A_matrices.contiguous().view(p * n, n)

    S = torch.kron(
        torch.eye(p, device="cuda"), torch.ones(n, 1, device="cuda").t()
    ) @ torch.linalg.pinv(matrix_of_A_matrices_2D)
    T = vector_of_A_matrices_2D
    U = torch.ones((p, 1), device="cuda") @ torch.ones((n, 1), device="cuda").t()

    RHS = S @ T - U

    A = torch.kron(torch.eye(n, device="cuda"), S)

    # vectorize RHS by stacking columns
    c = RHS.T.reshape(-1)

    R = torch.kron(torch.eye(p * n, device="cuda"), torch.ones((n, 1), device="cuda"))

    A_hat = A @ R

    rank = torch.linalg.matrix_rank(A_hat)
    if rank != n * p:
        print(f"A_hat rank: {rank} is not equal to n * p: {n * p}")

    b_hat = torch.linalg.solve(A_hat, c)

    # reshape to get Lambda ^ top
    Lambda_t = b_hat.reshape(n, p).T.contiguous()

    assert Lambda_t.shape[0] == p and Lambda_t.shape[1] == n

    # lastly, plug back into equation (6) to find phi

    Phi_matrices = torch.linalg.pinv(matrix_of_A_matrices_2D) @ (
        vector_of_A_matrices_2D
        - torch.kron(Lambda_t, torch.ones((n, 1), device="cuda"))
    )

    Phi_matrices = Phi_matrices.reshape(p, n, n)

    return Phi_matrices


def solve_for_phi_matrices(
    all_skill_parameters: torch.Tensor, n: int, p: int, num_timesteps: int
):

    all_skill_parameters = all_skill_parameters.permute(
        (1, 0)
    )  # (timesteps + AR_order) x (num_players)

    # first, create the pxp matrix of A matrices
    matrix_of_A_matrices = torch.zeros((p, p, n, n), device="cuda")  # 4D first

    for k in range(p):
        for i in range(p):
            A_ki = torch.zeros((n, n), device="cuda")
            for t in range(num_timesteps):
                A_ki += torch.outer(
                    all_skill_parameters[p + t - k], all_skill_parameters[p + t - i]
                )

            matrix_of_A_matrices[k, i] = A_ki

    # second, create a px1 vector of A matrices
    vector_of_A_matrices = torch.zeros((p, n, n), device="cuda")
    for i in range(p):
        A_neg_1_k = torch.zeros((n, n), device="cuda")

        for t in range(
            num_timesteps - 1
        ):  # -1 because formula would otherwise go past end

            A_neg_1_k += torch.outer(
                all_skill_parameters[p + t + 1], all_skill_parameters[p + t - i]
            )

        vector_of_A_matrices[i] = A_neg_1_k

    matrix_of_A_matrices_2D = (
        matrix_of_A_matrices.permute(0, 2, 1, 3).contiguous().view(p * n, p * n)
    )

    vector_of_A_matrices_2D = vector_of_A_matrices.contiguous().view(p * n, n)
    # third, solve for Phi (unconstrained LS)

    A = matrix_of_A_matrices_2D      # (p*n, p*n)
    B = vector_of_A_matrices_2D      # (p*n, n)

    Phi_flat = torch.linalg.lstsq(A, B).solution   # (p*n, n)

    Phi_matrices = Phi_flat.view(p, n, n)
    return Phi_matrices

import torch

def solve_for_phi_matrices_multi_ridge(
    all_skill_parameters: torch.Tensor,  # (n, num_timesteps + p)
    n: int,
    p: int,
    num_timesteps: int,
    ridge_lambda: float = 1e-2,          # L2 shrinkage toward 0
    prior_lambda: float = 0.0,           # L2 shrinkage toward prior Phi0
    prior_mode: str = "identity",        # "identity" or "zero"
):
    """
    Ridge (and optional identity-prior) fit for vector AR(p):
        x_t ≈ sum_{k=1..p} Φ_k x_{t-k}

    Returns Φ of shape (p, n, n) consistent with your AR_error prediction:
        pred = sum_k Φ_k @ x_{t-k}
    """
    device = all_skill_parameters.device
    dtype = all_skill_parameters.dtype

    T_total = all_skill_parameters.shape[1]
    assert T_total >= num_timesteps + p, "Not enough time steps for AR order p"

    # Targets Y: (n, T)
    Y = all_skill_parameters[:, p : p + num_timesteps]  # (n, T)
    T = num_timesteps

    # Build design matrix X_row: (T, p*n) where each row is [x_{t-1}; x_{t-2}; ...; x_{t-p}]
    X_rows = []
    for tau in range(T):
        t_global = p + tau
        lags = []
        for k in range(1, p + 1):
            lags.append(all_skill_parameters[:, t_global - k])  # (n,)
        X_rows.append(torch.cat(lags, dim=0))                   # (p*n,)

    X_row = torch.stack(X_rows, dim=0).to(device=device, dtype=dtype)  # (T, p*n)

    # We'll solve in column-vector form:
    #   Y ≈ Φ_stack @ X_stack,  where X_stack = X_row^T is (p*n, T) and Φ_stack is (n, p*n)
    X_stack = X_row.T  # (p*n, T)

    # Normal equations with ridge/prior:
    #   Φ_stack = (Y X^T + prior_lambda Φ0) (X X^T + (ridge_lambda+prior_lambda) I)^(-1)
    XXt = X_stack @ X_stack.T                           # (p*n, p*n)
    YXt = Y @ X_stack.T                                 # (n, p*n)

    lam = float(ridge_lambda) + float(prior_lambda)
    A = XXt + lam * torch.eye(p * n, device=device, dtype=dtype)

    if prior_lambda > 0.0:
        if prior_mode == "identity":
            # Φ0: first lag = I, remaining lags = 0
            Phi0 = torch.zeros((p, n, n), device=device, dtype=dtype)
            Phi0[0] = torch.eye(n, device=device, dtype=dtype)
        elif prior_mode == "zero":
            Phi0 = torch.zeros((p, n, n), device=device, dtype=dtype)
        else:
            raise ValueError(f"Unknown prior_mode={prior_mode}")

        Phi0_stack = Phi0.permute(1, 0, 2).contiguous().view(n, p * n)  # (n, p*n)
        B = YXt + float(prior_lambda) * Phi0_stack                      # (n, p*n)
    else:
        B = YXt

    # Solve A * M = B^T  =>  M = A^{-1} B^T  =>  Φ_stack = M^T
    M = torch.linalg.solve(A, B.T)          # (p*n, n)
    Phi_stack = M.T                         # (n, p*n)

    # Unstack into (p, n, n): blocks along the input dimension
    Phi_matrices = Phi_stack.view(n, p, n).permute(1, 0, 2).contiguous()  # (p, n, n)
    return Phi_matrices

def solve_for_phi_matrices_multi(
    all_skill_parameters: torch.Tensor,  # shape: (n, num_timesteps + p)
    n: int,
    p: int,
    num_timesteps: int,
):
    """
    Solve for Φ_1, ..., Φ_p in the vector AR(p) model

        x_t ≈ Σ_{k=1}^p Φ_k x_{t-k},

    via ordinary least squares over all available timesteps.

    all_skill_parameters: (n, num_timesteps + p), where columns 0..p-1 are
                          the initial history and columns p..p+num_timesteps-1
                          are the timesteps used in the likelihood.
    Returns: Phi_matrices of shape (p, n, n).
    """
    device = all_skill_parameters.device
    dtype = all_skill_parameters.dtype

    T_total = all_skill_parameters.shape[1]
    assert T_total >= num_timesteps + p, "Not enough time steps for AR order p"

    # Targets: x_t for t = p, ..., p + num_timesteps - 1  → shape (n, T)
    Y = all_skill_parameters[:, p : p + num_timesteps]          # (n, T)
    T = num_timesteps

    # Design matrix: for each t, stack [x_{t-1}, x_{t-2}, ..., x_{t-p}] (each n-dim)
    # into a single column of length p*n. Then stack over t.
    X_cols = []
    for tau in range(T):
        t_global = p + tau  # this is t in x_t
        lags = []
        for k in range(1, p + 1):
            # x_{t - k} is at index t_global - k
            lags.append(all_skill_parameters[:, t_global - k])  # (n,)
        col = torch.cat(lags, dim=0)  # (p*n,)
        X_cols.append(col)

    # X: (T, p*n), Y: (T, n)
    X = torch.stack(X_cols, dim=0).to(device=device, dtype=dtype)  # (T, p*n)
    Y_reg = Y.T.to(device=device, dtype=dtype)                     # (T, n)

    # Solve min_β ||X β - Y||_F^2. β has shape (p*n, n).
    beta = torch.linalg.lstsq(X, Y_reg).solution  # (p*n, n)

    # Reshape into Φ_k of shape (n, n) stacked along k.
    Phi_matrices = beta.view(p, n, n)

    return Phi_matrices

