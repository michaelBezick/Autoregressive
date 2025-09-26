import math as m
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.matlib import matrix


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


class SkillParameters(nn.Module):
    def __init__(self, num_players, num_timesteps, AR_order_p) -> None:
        super().__init__()
        self.alpha_estimates = nn.Parameter(
            torch.randn((num_players, num_timesteps + AR_order_p)), requires_grad=True
        )

        self.AR_order_p = AR_order_p

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

    def compute_AR_error(self, Phi_matrices_estimate, num_timesteps, AR_order_p):

        skill_param_estimates = self.alpha_estimates

        # players are independent, but for now we will sum up the MSE for all players

        total_error = 0

        for t in range(AR_order_p, num_timesteps):
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


def setup(num_players, p) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Take the max skill param at each timestep, make the max 0 by linearly shifting down
    """

    initial_skill_params = [torch.randn((num_players, 1)) for _ in range(p)]

    # need to ensure skill params sum to 0
    initial_skill_params = [p - torch.mean(p) for p in initial_skill_params]

    # columns must sum to 1, so take softmax
    Phi_matrices = torch.randn((p, num_players, num_players))
    Phi_matrices = F.softmax(Phi_matrices, dim=1)

    # construct rank 1 approximation: compute eigenvector decomp.
    # take eigenvectors and divide by largest eigenvalue
    # compute all eigenvalues of p matrix
    # one will have maximum abs value
    # divide entire p_matrix by the maximum absolute value
    # also construct rank1 approx

    """Why do this? Don't compare how well learned p matrix, compare how well learned rank 1 approx"""
    """try both, rank 1 and normalized"""

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
        outcome = np.random.binomial(1, prob_1_beats_2)
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


def play_games_erdos_renyi(skill_params, players, p, Z, W, t):

    num_players = len(players)

    for i in range(num_players):
        for j in range(i + 1, num_players):
            if random.random() < p:
                # play game

                player1_skill = skill_params[i]
                player2_skill = skill_params[j]

                diff = player1_skill - player2_skill
                prob_1_beats_2 = torch.sigmoid(diff)

                outcome = np.random.binomial(1, prob_1_beats_2)

                if outcome == 1:
                    # 1 beat 2
                    Z[i][j][t] = 1
                else:
                    W[i][j][t] = 1


def generate_next_skill_params(
    previous_skill_params: List[torch.Tensor],
    Phi_matrices: torch.Tensor,
    p: int,
    std_dev: float,
):

    last_p_skill_params = [previous_skill_params[-p + i] for i in range(p)]
    last_p_skill_params = torch.cat(last_p_skill_params, dim=1)  # [nxp]

    last_p_skill_params = torch.permute(last_p_skill_params, (1, 0))  # [pxn]
    last_p_skill_params = last_p_skill_params.unsqueeze(
        2
    )  # [pxnx1], batch of size p column vectors

    # Phi matrices: [pxnxn], batch of p nxn matrices
    summed = torch.bmm(Phi_matrices, last_p_skill_params)  # [p x nx1]
    summed = summed.sum(dim=0)  # [nx1]

    summed_w_noise = summed + torch.randn_like(summed) * std_dev

    return summed_w_noise


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

    RHS = torch.linalg.pinv(S) @ (S @ T - U)

    # let's reshape so we have a p x n matrix of n-length kronecker products

    RHS = torch.reshape(RHS, (p, n, n))

    Lambda_t = torch.mean(RHS, dim=1)

    # lastly, plug back into equation (6) to find phi

    Phi_matrices = torch.linalg.pinv(matrix_of_A_matrices_2D) @ (
        vector_of_A_matrices_2D
        - torch.kron(Lambda_t, torch.ones((n, 1), device="cuda"))
    )

    Phi_matrices = Phi_matrices.reshape(p, n, n)

    return Phi_matrices
