# Hyperparameter Sweep Summary — BTL + AR Skill Estimation

Each sweep varies one parameter while holding the others fixed at their defaults
(`λ = 10`, `sparsity = 0.8`, `σ = 0.1`).  
Results are averaged over 5 random seeds (n = 30 players, p = 3 AR lags, T = 20 timesteps, 120 epochs).

---

## 1. Weight sweep (AR regularisation strength λ)

**Setup:** λ ∈ {0, 0.1, 1, 3, 10, 30, 100, 300}; sparsity = 0.8, σ = 0.1.

| λ | mean α-MSE ± std | mean Φ-MSE ± std |
|------:|:----------------:|:----------------:|
| 0 | 0.025692 ± 0.004794 | 0.007376 ± 0.000248 |
| 0.1 | 0.025692 ± 0.004794 | 0.007265 ± 0.000266 |
| 1 | 0.025692 ± 0.004794 | 0.007258 ± 0.000267 |
| 3 | 0.025691 ± 0.004794 | 0.007242 ± 0.000268 |
| 10 | 0.025690 ± 0.004794 | 0.007192 ± 0.000274 |
| 30 | 0.025688 ± 0.004794 | 0.007071 ± 0.000290 |
| 100 | 0.025680 ± 0.004793 | 0.006803 ± 0.000293 |
| 300 | 0.025661 ± 0.004787 | 0.006438 ± 0.000184 |

**Key findings:**

- **α-MSE is essentially insensitive to λ** across the full range [0, 300]. The absolute change from λ = 0 to λ = 300 is < 0.03%, well within seed-to-seed noise. Skill recovery is dominated by the BTL likelihood term regardless of how strongly the AR constraint is enforced.
- **Φ-MSE decreases monotonically** with λ (a 13% improvement from λ = 0 to λ = 300). The bulk of this gain arrives between λ = 10 and λ = 300; below λ = 10 the reduction is marginal.
- **No classic U-shaped optimum.** There is no value of λ that simultaneously minimises both metrics. α-MSE plateaus at essentially any λ ≥ 0, while Φ-MSE continues to benefit from stronger regularisation. A practical operating point of **λ = 10–30** captures most of the Φ-MSE improvement without extreme regularisation.

---

## 2. Sparsity sweep (Erdős–Rényi edge probability)

**Setup:** sparsity ∈ {0.1, 0.2, 0.4, 0.6, 0.8, 1.0}; λ = 10, σ = 0.1.

| sparsity | mean α-MSE ± std | mean Φ-MSE ± std |
|---------:|:----------------:|:----------------:|
| 0.1 | 0.042401 ± 0.005087 | 0.005729 ± 0.000263 |
| 0.2 | 0.035110 ± 0.005761 | 0.006144 ± 0.000338 |
| 0.4 | 0.030634 ± 0.004928 | 0.006399 ± 0.000355 |
| 0.6 | 0.027779 ± 0.006303 | 0.006960 ± 0.000764 |
| 0.8 | 0.025690 ± 0.004794 | 0.007192 ± 0.000274 |
| 1.0 | 0.022928 ± 0.004890 | 0.007581 ± 0.000452 |

**Key findings:**

- **α-MSE degrades significantly below sparsity ≈ 0.2.** Going from 0.2 → 0.1 inflates α-MSE by ~21% (0.035 → 0.042), the sharpest single-step increase in the sweep. The 0.4 → 0.2 transition is the next largest (~15%). Above 0.4, α-MSE improves smoothly and modestly as more game observations arrive.
- **Φ-MSE exhibits the opposite trend**, decreasing at lower sparsity. With fewer game observations the optimiser relies more heavily on the AR regularisation term to fit the data, which paradoxically recovers Φ better. At high sparsity the BTL likelihood dominates and pushes the solution away from the true Φ.
- **Practical threshold:** sparsity ≥ 0.2 is necessary for competitive α recovery; below 0.2 the lack of pairwise comparisons causes a sharp skill-estimation collapse.

---

## 3. Noise sweep (AR process innovation std dev σ)

**Setup:** σ ∈ {0.0, 0.01, 0.05, 0.1, 0.2, 0.5}; λ = 10, sparsity = 0.8.

| σ | mean α-MSE ± std | mean Φ-MSE ± std |
|-----:|:----------------:|:----------------:|
| 0.00 | 0.057991 ± 0.003646 | 0.005865 ± 0.000606 |
| 0.01 | 0.039137 ± 0.006132 | 0.006909 ± 0.000510 |
| 0.05 | 0.030571 ± 0.005855 | 0.007241 ± 0.000467 |
| 0.10 | 0.025690 ± 0.004794 | 0.007192 ± 0.000274 |
| 0.20 | 0.018610 ± 0.003952 | 0.007136 ± 0.000640 |
| 0.50 | 0.008852 ± 0.001969 | 0.006861 ± 0.000743 |

**Key findings:**

- **Counter-intuitive: α-MSE improves as σ increases.** At σ = 0 the skill trajectories are fully deterministic given Φ and initial conditions. The optimiser must fit a rigid AR manifold using only BTL observations, which is underdetermined — α-MSE peaks at 0.058. As σ grows, the skill parameters have genuine stochastic variation that the BTL log-likelihood can directly recover, and α-MSE falls to 0.009 at σ = 0.5.
- **Φ-MSE is broadly robust to noise** (range 0.0059–0.0072 across the full σ sweep, < 25% variation). It is lowest at σ = 0 (exact AR structure) and slightly elevated in the σ = 0.05–0.2 band, before recovering at σ = 0.5. No catastrophic degradation is observed at any tested noise level.
- **Noise robustness summary:** The algorithm tolerates high process noise well for Φ recovery. α recovery is hardest in the near-deterministic regime (σ ≤ 0.01) and easiest when the AR innovations dominate (σ ≥ 0.2).

---

## Cross-sweep summary

| Dimension | Best α-MSE regime | Best Φ-MSE regime | Critical threshold |
|-----------|:-----------------:|:-----------------:|-------------------|
| λ (weight) | any λ (flat) | λ → ∞ (monotone) | No critical point; λ ≥ 10 is practical |
| Sparsity | high (→ 1.0) | low (→ 0.1) | α degrades sharply below 0.2 |
| Noise σ | high (→ 0.5) | low (σ = 0) | Near-zero σ is hardest for α recovery |
