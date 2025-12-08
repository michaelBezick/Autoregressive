import itertools
import os
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

try:
    import pandas as pd
except ImportError:
    pd = None

from nn_btl_experiment import run_nn_btl_experiment

# ---------------------------
# Sweep configuration
# ---------------------------
NUM_PLAYERS_LIST   = [50, 100, 200]
NUM_TIMESTEPS_LIST = [20, 30, 50]
WEIGHT_LIST        = [5.0, 10.0, 30.0]

AR_ORDER_P = 10
EPOCHS     = 300
STEP       = 10
SEED       = 0

OUT_DIR    = "sweep_outputs"      # results + plots
RUN_PLOTS  = False                # per-run plots (lots of files); set True if desired

os.makedirs(OUT_DIR, exist_ok=True)

rows: List[Dict[str, Any]] = []

for n, T, w in itertools.product(NUM_PLAYERS_LIST, NUM_TIMESTEPS_LIST, WEIGHT_LIST):
    print(f"=== Running n={n}, T={T}, weight={w} ===")
    res = run_nn_btl_experiment(
        num_players=n,
        num_timesteps=T,
        AR_order_p=AR_ORDER_P,
        epochs=EPOCHS,
        weight=w,
        STEP=STEP,
        seed=SEED,
        save_plots_to=(OUT_DIR if RUN_PLOTS else None),
    )
    fin = res["final"]
    rows.append({
        "num_players": n,
        "num_timesteps": T,
        "weight": w,
        "alpha_mse": fin["alpha_mse"],
        "nn_mse_centered": fin["nn_mse_centered"],
        "pred_mse_true_inputs": fin["pred_mse_true_inputs"],
        "pred_mse_est_inputs": fin["pred_mse_est_inputs"],
        "btl_likelihood": fin["btl_likelihood"],
        "total_objective": fin["total_objective"],
    })

# ---------------------------
# Save tabular results
# ---------------------------
if pd is not None:
    df = pd.DataFrame(rows)
    df_path = os.path.join(OUT_DIR, "sweep_results.csv")
    df.to_csv(df_path, index=False)
    print(f"Saved: {df_path}")
else:
    # minimal CSV if pandas is unavailable
    import csv
    df_path = os.path.join(OUT_DIR, "sweep_results.csv")
    with open(df_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    print(f"Saved: {df_path} (no pandas)")

# ---------------------------
# Marginal plots (averaging the third variable)
# ---------------------------
def group_and_plot(x_vals, fixed_key, vary_key, metric_key, rows, out_name, xlabel, title_prefix):
    """
    Average metric over the remaining third variable.
    x_vals: list of x to plot (e.g., WEIGHT_LIST)
    fixed_key: which key to fix into separate curves (e.g., 'num_players')
    vary_key: x-axis key (e.g., 'weight')
    metric_key: which metric to plot
    """
    unique_fixed = sorted(set(r[fixed_key] for r in rows))
    plt.figure(figsize=(10,5))
    for fval in unique_fixed:
        ys = []
        for x in x_vals:
            # collect all rows with fixed_key=fval and vary_key=x (averaging over the third dim)
            vals = [r[metric_key] for r in rows if r[fixed_key]==fval and r[vary_key]==x]
            vals = [v for v in vals if v is not None]
            y = float(np.mean(vals)) if len(vals)>0 else np.nan
            ys.append(y)
        plt.plot(x_vals, ys, marker='o', label=f"{fixed_key}={fval}")
    plt.grid(True)
    plt.xlabel(xlabel)
    plt.ylabel(metric_key)
    plt.title(f"{title_prefix}: {metric_key}")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(OUT_DIR, out_name)
    plt.savefig(path); plt.close()
    print(f"Saved: {path}")

# 1) Vary weight; separate curves by num_players (averaged over num_timesteps)
for metric in ["alpha_mse", "pred_mse_true_inputs", "pred_mse_est_inputs", "nn_mse_centered"]:
    group_and_plot(
        x_vals=WEIGHT_LIST,
        fixed_key="num_players",
        vary_key="weight",
        metric_key=metric,
        rows=rows,
        out_name=f"marginal_weight_by_num_players_{metric}.pdf",
        xlabel="weight",
        title_prefix="Marginal over T",
    )

# 2) Vary weight; separate curves by num_timesteps (averaged over num_players)
for metric in ["alpha_mse", "pred_mse_true_inputs", "pred_mse_est_inputs", "nn_mse_centered"]:
    group_and_plot(
        x_vals=WEIGHT_LIST,
        fixed_key="num_timesteps",
        vary_key="weight",
        metric_key=metric,
        rows=rows,
        out_name=f"marginal_weight_by_timesteps_{metric}.pdf",
        xlabel="weight",
        title_prefix="Marginal over n",
    )

# 3) Optional heatmaps (averaged over the third variable)
def heatmap(metric_key, rows, row_key, col_key, row_vals, col_vals, avg_over_key, out_name, title):
    grid = np.full((len(row_vals), len(col_vals)), np.nan, dtype=float)
    for i, rv in enumerate(row_vals):
        for j, cv in enumerate(col_vals):
            vals = [r[metric_key] for r in rows if r[row_key]==rv and r[col_key]==cv]
            vals = [v for v in vals if v is not None]
            if len(vals)>0:
                grid[i,j] = float(np.mean(vals))
    plt.figure(figsize=(0.8*len(col_vals)+3, 0.8*len(row_vals)+3))
    im = plt.imshow(grid, aspect='auto', origin='upper')
    plt.colorbar(im, label=metric_key)
    plt.xticks(range(len(col_vals)), col_vals)
    plt.yticks(range(len(row_vals)), row_vals)
    plt.xlabel(col_key)
    plt.ylabel(row_key)
    plt.title(f"{title} (avg over {avg_over_key})")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, out_name)
    plt.savefig(path); plt.close()
    print(f"Saved: {path}")

for metric in ["alpha_mse", "pred_mse_true_inputs", "pred_mse_est_inputs", "nn_mse_centered"]:
    # Heatmap over (num_players x weight), averaged over num_timesteps
    heatmap(
        metric_key=metric, rows=rows,
        row_key="num_players", col_key="weight",
        row_vals=sorted(set(r["num_players"] for r in rows)),
        col_vals=sorted(set(r["weight"] for r in rows)),
        avg_over_key="num_timesteps",
        out_name=f"heatmap_n_vs_w_{metric}.pdf",
        title=metric,
    )
