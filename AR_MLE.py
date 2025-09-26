import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr

from functions import (
    SkillParameters,
    generate_next_skill_params,
    play_games_erdos_renyi,
    setup,
    solve_for_phi_matrices,
    btl_matrix_from_scores
)

"""
NOW, I need to modify the parameterization, instead of a vector
for each person of skill parameters, I have a matrix s.t. the columns
sum to 1.

Furthermore, this property needs to maintain with the AR parameter update
step, meaning I will need to use Lagrange multipliers in the solution.
"""


# parameters

num_players = 3
AR_order_p = 1
erdos_renyi_p = 1
std_dev = 0
num_timesteps = 4
epochs = 1000
N_grad_descent = 100
weight = 1e-2

# setup
players = list(range(0, num_players))

Z = torch.zeros((num_players, num_players, num_timesteps))
W = torch.zeros((num_players, num_players, num_timesteps))

# initial params sum to 0, Phi matrices' columns sum to 1
initial_skill_params, Phi_matrices = setup(num_players, AR_order_p)

# generating next skill params and then playing game
skill_params = initial_skill_params

for t in range(num_timesteps):
    next_skill_params = generate_next_skill_params(
        skill_params, Phi_matrices, AR_order_p, std_dev
    )
    skill_params.append(next_skill_params)
    play_games_erdos_renyi(next_skill_params, players, erdos_renyi_p, Z, W, t)

actual_skill_params = skill_params
actual_skill_params = torch.stack(actual_skill_params, dim=1)

# now Z and W have the necessary data, can run algorithm

skill_params_est = SkillParameters(num_players, num_timesteps, AR_order_p)
optimizer = torch.optim.Adam(params=skill_params_est.parameters())
Phi_matrices_estimate = torch.randn((AR_order_p, num_players, num_players))

skill_params_est = skill_params_est.cuda()
W = W.cuda()
Z = Z.cuda()
actual_skill_params = actual_skill_params.cuda()
Phi_matrices_estimate = Phi_matrices_estimate.cuda()
Phi_matrices = Phi_matrices.cuda()

true_p_matrix_error = []
true_alphas_error = []
ar_errors = []
btl_likelihoods = []
total_likelihoods = []
# pearson_correlations = []

for epoch in range(epochs):
    # first, predict alphas
    # BTL_likelihood = skill_params.compute_log_BTL(Z, W, num_players, num_timesteps)

    for _ in range(N_grad_descent):

        optimizer.zero_grad()
        BTL_likelihood = skill_params_est.compute_log_BTL_vectorized(Z, W, num_players)
        AR_error = skill_params_est.compute_AR_error(
            Phi_matrices_estimate, num_timesteps, AR_order_p
        )

        total_likelihood = BTL_likelihood - weight * AR_error

        loss = -total_likelihood

        loss.backward()
        optimizer.step()

    # second, update phi estimate
    with torch.no_grad():

        Phi_matrices_estimate = solve_for_phi_matrices(
            skill_params_est.alpha_estimates, num_players, AR_order_p, num_timesteps
        )

    if epoch % 100 == 0:

        print("------------------------------")
        print("Epoch: ", epoch)
        print("BTL_likelihood: ", BTL_likelihood.item())
        print("AR_error", AR_error.item())

        def normalize_alpha(alpha: torch.Tensor) -> torch.Tensor:
            alpha_c = alpha.clone()
            alpha_c -= torch.mean(alpha_c, dim=0)
            alpha_c /= torch.norm(alpha_c, dim=0)

            return alpha_c

        alpha_estimates_normalized = normalize_alpha(skill_params_est.alpha_estimates)
        actual_skill_params_normalized = normalize_alpha(
            torch.squeeze(actual_skill_params)
        )

        print(
            "skill_params error: ",
            F.mse_loss(alpha_estimates_normalized, actual_skill_params_normalized).item(),
        )
        print("p_matrix error: ", F.mse_loss(Phi_matrices, Phi_matrices_estimate).item())
        print(
            "Alpha estimates:\n",
            alpha_estimates_normalized.data,
            "\nTrue alpha:\n",
            actual_skill_params_normalized.data,
        )

        true_alphas_error.append(
            F.mse_loss(
                alpha_estimates_normalized.data,
                actual_skill_params_normalized.data,
            ).item()
        )
        true_p_matrix_error.append(
            F.mse_loss(Phi_matrices, Phi_matrices_estimate).item()
        )
        ar_errors.append(AR_error.item())
        btl_likelihoods.append(BTL_likelihood.item())
        total_likelihoods.append(total_likelihood.item())


epochs_logged = list(range(0, epochs, 100))


print("BTL Matrices")
print("------------------------------")
for i in range(num_timesteps):
    print(f"Timestep {i}")

    print(f"Real:\n{btl_matrix_from_scores(torch.squeeze(actual_skill_params)[:, AR_order_p + i])}")
    print(f"Predicted:\n{btl_matrix_from_scores(skill_params_est.alpha_estimates[:, AR_order_p + i])}")
    print("------------------------------")

# --- Figure 1: Errors (same scale) ---
plt.figure(figsize=(10, 4))
plt.plot(
    epochs_logged, true_alphas_error, label="Alpha Estimates Error (MSE)", linewidth=2
)
plt.plot(epochs_logged, true_p_matrix_error, label="P Matrix Error (MSE)", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Error (MSE)")
plt.title("Model Estimation Errors Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("errors.pdf")

# --- Figure 2: Likelihoods and AR Error (same scale) ---
plt.figure(figsize=(10, 4))
plt.plot(epochs_logged, ar_errors, label="AR Error", linewidth=2)
plt.plot(epochs_logged, btl_likelihoods, label="BTL Likelihood", linewidth=2)
plt.plot(epochs_logged, total_likelihoods, label="Total Likelihood", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Loss/Likelihood")
plt.title("AR Error, BTL Likelihood, and Total Likelihood Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("likelihoods.pdf")

# # --- Figure 3: Pearson ---
# plt.figure(figsize=(10, 4))
# plt.plot(
#     epochs_logged, pearson_correlations, label="Pearson Correlation (α)", linewidth=2
# )
# plt.xlabel("Epoch")
# plt.ylabel("Correlation Coefficient")
# plt.title("Correlation Between True and Predicted Alphas")
# plt.grid(True)
# plt.legend()
# plt.tight_layout()
# plt.savefig("pearson_correlations.pdf")
