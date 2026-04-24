import math as m
import random
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def rank_centrality_mse_per_timestep(
    Z: torch.Tensor,
    W: torch.Tensor,
    actual_skill_params: torch.Tensor,
    AR_order_p: int,
    gamma: float = 0.05,
    max_iter: int = 200,
    tol: float = 1e-10,
):
    """
    For each timestep t, build a Markov chain from match outcomes at time t,
    compute stationary distribution r_t with damping, set alpha_rc_t = log(r_t),
    center it, and compute MSE vs centered ground-truth alpha_t.
    """
    device = Z.device
    dtype = Z.dtype
    n, _, T = Z.shape
    i_idx, j_idx = torch.triu_indices(n, n, offset=1, device=device)

    rc_mses = []
    one_over_n = 1.0 / float(n)

    for t in range(T):
        # counts[i,j] = times j beat i at time t
        counts = torch.zeros((n, n), device=device, dtype=dtype)
        counts[i_idx, j_idx] = W[i_idx, j_idx, t]  # j beats i
        counts[j_idx, i_idx] = Z[i_idx, j_idx, t]  # i beats j

        # Row-normalize to transitions; put leftover mass on diag
        row_sums = counts.sum(dim=1, keepdim=True).clamp_min(1e-12)
        P = counts / row_sums
        P.fill_diagonal_(0.0)
        P[torch.arange(n, device=device), torch.arange(n, device=device)] = 1.0 - P.sum(dim=1)

        # Damping for irreducibility/aperiodicity
        P = (1.0 - gamma) * P + gamma * one_over_n

        # Power iteration for stationary distribution
        r = torch.full((n,), one_over_n, device=device, dtype=dtype)
        for _ in range(max_iter):
            r_next = P.t().matmul(r)
            r_next = r_next / r_next.sum()
            if torch.norm(r_next - r, p=1).item() < tol:
                r = r_next
                break
            r = r_next

        # Convert to centered log-skills
        alpha_rc_c = torch.log(r.clamp_min(1e-12))
        alpha_rc_c = alpha_rc_c - alpha_rc_c.mean()

        alpha_true = actual_skill_params[:, AR_order_p + t]
        alpha_true_c = alpha_true - alpha_true.mean()

        rc_mses.append(F.mse_loss(alpha_rc_c, alpha_true_c).item())

    return rc_mses
def rescale_for_radius(Phi_list, target=0.99, iters=30):
    S = sum(Phi_list)  # n x n
    v = torch.randn(S.size(0), 1)
    v = v / v.norm()
    for _ in range(iters):  # power iteration
        v = S @ v
        v = v / (v.norm() + 1e-8)
    lam = (v.T @ (S @ v) / (v.T @ v)).item()  # Rayleigh quotient
    scale = target / max(abs(lam), 1e-8)
    return [scale * P for P in Phi_list]


def normalize_columns(M):
    return M / (M.sum(dim=0, keepdim=True) + 1e-8)


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

    # spectral rescaling
    Phi_list = [Phi_matrices[k] for k in range(p)]
    Phi_list = rescale_for_radius(Phi_list, target=0.995)
    # renormalize columns
    Phi_list = [normalize_columns(P) for P in Phi_list]
    Phi_matrices = torch.stack(Phi_list, dim=0)

    # construct rank 1 approximation: compute eigenvector decomp.
    # take eigenvectors and divide by largest eigenvalue
    # compute all eigenvalues of p matrix
    # one will have maximum abs value
    # divide entire p_matrix by the maximum absolute value
    # also construct rank1 approx

    """Why do this? Don't compare how well learned p matrix, compare how well learned rank 1 approx"""
    """try both, rank 1 and normalized"""

    return initial_skill_params, Phi_matrices


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


# -------------------------------
# Utilities
# -------------------------------


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


# -------------------------------
# Model parameter objects
# -------------------------------


class SkillParameters(nn.Module):
    """
    Stores skill parameters α for all players across time.
    alpha_estimates: (n_players, AR_order_p + num_timesteps)
    """

    def __init__(self, num_players: int, num_timesteps: int, AR_order_p: int) -> None:
        super().__init__()
        self.alpha_estimates = nn.Parameter(
            torch.randn((num_players, num_timesteps + AR_order_p)), requires_grad=True
        )
        self.AR_order_p = AR_order_p

    def project_identifiability_(self) -> None:
        with torch.no_grad():
            self.alpha_estimates.sub_(self.alpha_estimates.mean(dim=0, keepdim=True))

    # --- BTL log-likelihood (vectorized & stable) ---
    def compute_log_BTL_vectorized_new(self, Z, W, num_players: int) -> torch.Tensor:
        """
        Z[i,j,t] = number of wins for i over j at time t
        W[i,j,t] = number of wins for j over i at time t
        """
        i_idx, j_idx = torch.triu_indices(num_players, num_players, offset=1)

        Z_pairs = Z[i_idx, j_idx, :]  # (pairs, T)
        W_pairs = W[i_idx, j_idx, :]  # (pairs, T)

        # Align α with the T observed timesteps: keep only the last T columns
        T = Z.shape[-1]
        sliced_alpha = self.alpha_estimates[:, -T:]  # (n, T)
        s_i = sliced_alpha[i_idx, :]  # (pairs, T)
        s_j = sliced_alpha[j_idx, :]  # (pairs, T)

        # Stable denominator using logaddexp
        log_den = torch.logaddexp(s_i, s_j)
        loglik = (Z_pairs * s_i + W_pairs * s_j - (Z_pairs + W_pairs) * log_den).sum()
        return loglik

    # --- NN regularizer: MSE between actual α_t and NN prediction given past p α's ---
    def compute_NN_mse(
        self, predictor: "PredictorMLP", num_timesteps: int, AR_order_p: int
    ) -> torch.Tensor:
        """
        For t in [p, p+T-1], flatten the previous p columns α_{t-p:t-1} (shape n x p)
        into a vector of length n*p and ask the predictor to output α_hat_t (length n).
        Return mean squared error over players and timesteps.
        """
        est = self.alpha_estimates  # (n, p+T)
        n = est.size(0)

        inputs = []
        targets = []
        for t in range(AR_order_p, AR_order_p + num_timesteps):
            past_block = est[:, t - AR_order_p : t]  # (n, p)
            x_t = past_block.reshape(-1)  # (n*p,)
            y_t = est[:, t]  # (n,)
            inputs.append(x_t)
            targets.append(y_t)

        X = torch.stack(inputs, dim=0)  # (T, n*p)
        Y = torch.stack(targets, dim=0)  # (T, n)

        Y_hat = predictor(X)  # (T, n)
        mse = F.mse_loss(Y_hat, Y)
        return mse


class PredictorMLP(nn.Module):
    """
    Simple MLP that maps flattened past p skill vectors (length n*p) -> next skill vector (length n).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, input_dim)
        return self.net(x)


# -------------------------------
# Synthetic data helpers
# -------------------------------


def setup_initial_alphas(num_players: int, p: int, device) -> List[torch.Tensor]:
    """
    Create p initial α vectors (each shape (n,1)) with zero-mean columns.
    """
    initial = [torch.randn((num_players, 1), device=device) for _ in range(p)]
    initial = [col - col.mean() for col in initial]
    return initial


def play_games_erdos_renyi(
    skill_params: torch.Tensor,
    players: List[int],
    p_edge: float,
    Z: torch.Tensor,
    W: torch.Tensor,
    t: int,
) -> None:
    """
    For timestep t, sample a directed comparison i vs j with probability p_edge.
    Record outcome in Z/W using BTL probability from skill differences.
    """
    num_players = len(players)
    for i in range(num_players):
        for j in range(i + 1, num_players):
            if random.random() < p_edge:
                diff = skill_params[i] - skill_params[j]
                prob_1_beats_2 = torch.sigmoid(diff).clamp(1e-6, 1 - 1e-6).item()
                outcome = np.random.binomial(1, prob_1_beats_2)
                if outcome == 1:
                    Z[i, j, t] += 1.0
                else:
                    W[i, j, t] += 1.0


@torch.no_grad()
def generate_next_skill_params_nn(
    previous_skill_params: List[torch.Tensor],
    predictor_true: "PredictorMLP",
    p: int,
    std_dev: float,
) -> torch.Tensor:
    """
    Use a fixed 'true' predictor to generate the next α_t from the last p α's.
    Returns tensor of shape (n,1).
    """
    last_p = [previous_skill_params[-p + i] for i in range(p)]  # list of (n,1)
    past_block = torch.cat(last_p, dim=1)  # (n, p)
    x = past_block.reshape(-1).unsqueeze(0)  # (1, n*p)
    y = predictor_true(x).squeeze(0)  # (n,)
    y = y.unsqueeze(1)  # (n,1)
    # zero-mean the column to keep logits well-behaved
    y = y - y.mean()
    # add noise
    y = y + torch.randn_like(y) * std_dev
    return y


# -------------------------------
# Alternating optimization helpers
# -------------------------------


def optimize_alphas_until_converged_nn(
    skill_params_est: SkillParameters,
    predictor: PredictorMLP,
    Z: torch.Tensor,
    W: torch.Tensor,
    num_players: int,
    num_timesteps: int,
    AR_order_p: int,
    weight: float,
    optimizer: torch.optim.Optimizer,
    max_steps: int = 5000,
    rel_tol: float = 1e-3,
    grad_tol: float = 5e-5,
    patience: int = 20,
    grad_clip: Optional[float] = 5.0,
):
    """
    Optimize α while holding predictor fixed.
    Objective:  (BTL / C(n,2)) - weight * MSE_NN
    We minimize negative of the above.
    """
    last_loss = None
    no_improve = 0
    steps_taken = 0
    gnorm_val = float("inf")

    for k in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)

        BTL_likelihood = skill_params_est.compute_log_BTL_vectorized_new(
            Z, W, num_players
        )
        NN_mse = skill_params_est.compute_NN_mse(predictor, num_timesteps, AR_order_p)

        # normalize
        BTL_norm = BTL_likelihood / m.comb(num_players, 2)
        loss = -(BTL_norm - weight * NN_mse)

        loss.backward()

        # gradient norm BEFORE step
        with torch.no_grad():
            sq_sum = 0.0
            for p_ in skill_params_est.parameters():
                if p_.grad is not None:
                    sq_sum += (p_.grad.detach() ** 2).sum().item()
            gnorm_val = sq_sum**0.5

        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(skill_params_est.parameters(), grad_clip)

        optimizer.step()

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

    # return useful stats
    total_obj = (
        BTL_likelihood.detach() / m.comb(num_players, 2) - weight * NN_mse.detach()
    )
    return (
        BTL_likelihood.detach(),
        NN_mse.detach(),
        total_obj,
        steps_taken,
        gnorm_val,
    )
