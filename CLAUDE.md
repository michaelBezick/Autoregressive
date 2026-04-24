# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running Experiments

This is a research codebase with no build system. Run scripts from the **repo root** (scripts use `sys.path` to locate shared modules):

```bash
python Lagrange/DefaultExperiment/AR_MLE.py
python Lagrange/CyclicShiftTest/AR_MLE_ring_mixing_ar1.py
python Lagrange/IdentityMatrixTest/AR_MLE_transition.py
python Lagrange/ModelSelection/experiment_model_selection.py
python NeuralNetwork/AR_MLE.py
```

Hyperparameter ablations (weight λ, sparsity, noise):
```bash
python Lagrange/sweep_experiments.py --sweep weight    # ablates AR regularization strength
python Lagrange/sweep_experiments.py --sweep sparsity  # varies erdos_renyi_p
python Lagrange/sweep_experiments.py --sweep noise     # varies std_dev
python Lagrange/sweep_experiments.py --sweep all       # runs all three
```

Lagrange KKT vs Neural Network direct comparison (synthetic data):
```bash
python lagrange_vs_nn.py
```

NN parameter sweep:
```bash
python NeuralNetwork/sweep_nn_btl.py
```

Real-world experiment (requires ATP CSV data in `NeuralNetwork/matches/`):
```bash
python NeuralNetwork/tennis.py
```

**Dependencies:** PyTorch, NumPy, Matplotlib, SciPy, tqdm. GPU is used automatically if available.

## Architecture

The codebase implements joint estimation of latent player skill parameters using a **Bradley-Terry-Luce (BTL) model** combined with **autoregressive (AR) dynamics**. The goal is to recover time-varying skill ratings from observed game outcomes.

**Mathematical framework:**
- BTL: `P(i beats j) = σ(s_i - s_j)` where `s_i` is player i's log-skill
- AR(p): `α_t = Φ_0 α_{t-p} + ... + Φ_{p-1} α_{t-1} + ε_t`
- Joint objective: maximize BTL log-likelihood − `weight` × AR prediction error
- Identifiability is handled differently per module (see below)

**Two main approaches, each self-contained in its own directory:**

The canonical shared functions live in `Lagrange/functions.py`. All sub-experiment scripts import from there using a `sys.path` insert pointing to the repo root.

All n skills are learned, with identifiability enforced by projecting skill vectors to the center-to-zero subspace after each update (`project_identifiability_()` subtracts column means). Four sub-experiments:
- `DefaultExperiment/` — standard BTL+AR with centering
- `CyclicShiftTest/` — tests a ring-mixing AR transition matrix
- `IdentityMatrixTest/` — tests specific transition matrix structure
- `ModelSelection/` — cross-validation sweep over AR order `p`, saves results to `sweep_outputs/`

### NeuralNetwork/
Replaces the closed-form Φ matrix update with a learned `PredictorMLP`. Alternates between optimizing skill parameters (BTL + NN prediction loss) and training the NN (MSE on skill predictions). Core classes in `nn_functions.py`. Also contains `tennis.py` — a real-world ATP data experiment that compares our Lagrange KKT model against static BTL, rank centrality, and neural AR baselines. ATP match data lives in `NeuralNetwork/matches/`.

## Experiment Flow

Each `AR_MLE.py` follows this pattern:
1. **Setup** — initialize skill parameters (mean-zero) and column-stochastic transition matrices
2. **Data generation** — simulate games via Erdős-Rényi graph; sample winners via BTL (`play_games_erdos_renyi()`)
3. **Estimation loop** — alternate between:
   - α-step: Adam gradient descent on BTL likelihood − weight × AR error
   - Φ-step: closed-form least-squares solve for transition matrices (`solve_for_phi_matrices()`)
   - Lagrange modules also call `project_identifiability_()` after α-step
4. **Output** — save convergence plots as PDFs (`errors_std{X}.pdf`, `likelihoods_std{X}.pdf`)

## Key Tunable Parameters

All parameters are hardcoded at the top of each script:

| Parameter | Meaning |
|---|---|
| `num_players` | Number of players in the simulated network |
| `AR_order_p` | Autoregressive lag order |
| `num_timesteps` | Length of time series |
| `weight` | Balance between BTL likelihood and AR constraint |
| `std_dev` | AR noise standard deviation |
| `epochs` | Training epochs |
| `N_grad_descent` | Inner gradient steps per epoch |

## Real-World Data (tennis.py)

Expects the following files relative to `NeuralNetwork/`:
- `./matches/atp_matches_*.csv` — yearly ATP match data
- `./matches/atp_players.csv` — player metadata with columns `player_id`, `name_first`, `name_last`

Time is discretized into monthly intervals. Train/test split is 70%/30% by time cutoff.
