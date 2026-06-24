"""
Aggregate results across seeds and produce:
  1. Summary CSV table
  2. Per-algorithm learning curves
  3. Comparative bar chart (all algorithms side by side)
  4. Hyperparameter sensitivity analysis

Run:
    python evaluate.py
"""
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    ALGORITHMS,
    HYPERPARAM_ALGO,
    HYPERPARAM_GRID,
    HYPERPARAM_REWARD,
    REWARD_CONFIGS,
    RESULTS_DIR,
    SEEDS,
    SUCCESS_THRESHOLD,
)

# Nice color palette for up to 5 algorithms 
REPORTS_DIR = "reports"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
TABLES_DIR = os.path.join(REPORTS_DIR, "tables")

ALGO_COLORS = {
    "ppo":  "#2196F3",  # Blue
    "a2c":  "#4CAF50",  # Green
    "sac":  "#FF9800",  # Orange
    "td3":  "#9C27B0",  # Purple
    "ddpg": "#F44336",  # Red
}

REWARD_COLORS = {
    "none":     "#78909C",  # Grey
    "distance": "#29B6F6",  # Light blue
    "angle":    "#66BB6A",  # Light green
    "combined": "#FFA726",  # Light orange
}


# DATA LOADING

def load_run(algo, reward, seed, lr=None, net_arch=None):
    """
    Load the evaluation data from a single training run.

    Returns (timesteps, mean_rewards_per_eval, raw_results_matrix).
    Raises FileNotFoundError if the run hasn't been completed yet.
    """
    # Build the folder name (must match train.py's naming convention)
    name = f"{algo}_{reward}_seed{seed}"
    if lr is not None:
        name += f"_lr{lr}"
    if net_arch is not None:
        arch_str = "-".join(str(n) for n in net_arch)
        name += f"_net{arch_str}"

    path = os.path.join(RESULTS_DIR, name, "evaluations.npz")
    data = np.load(path)
    timesteps = data["timesteps"]
    results = data["results"]               # shape: (n_evals, n_episodes)
    mean_rewards = results.mean(axis=1)      # average across eval episodes
    return timesteps, mean_rewards, results


def timesteps_to_threshold(timesteps, mean_rewards, threshold=SUCCESS_THRESHOLD):
    """How many timesteps until the agent first reaches the success threshold?"""
    above = np.where(mean_rewards >= threshold)[0]
    if len(above) == 0:
        return np.nan  # Never reached the threshold
    return float(timesteps[above[0]])


def area_under_curve(timesteps, mean_rewards):
    """Total area under the learning curve — higher means faster learning."""
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(mean_rewards, timesteps))


def nanmean_or_nan(values):
    """Mean that stays quiet when every value is NaN."""
    arr = np.asarray(values, dtype=float)
    finite = arr[~np.isnan(arr)]
    if len(finite) == 0:
        return np.nan
    return float(np.mean(finite))


def nanstd_or_nan(values):
    """Standard deviation that stays quiet when every value is NaN."""
    arr = np.asarray(values, dtype=float)
    finite = arr[~np.isnan(arr)]
    if len(finite) == 0:
        return np.nan
    return float(np.std(finite))


# SUMMARY TABLE

def build_summary_table(save_csv=True):
    """
    Build a DataFrame summarizing all completed runs.

    Columns: algorithm, reward_config, n_seeds, final_reward (mean/std),
    success_rate, timesteps_to_200, AUC.
    """
    rows = []
    for algo, reward in itertools.product(ALGORITHMS, REWARD_CONFIGS):
        ttt, aucs, successes, finals = [], [], [], []
        

        for seed in SEEDS:
            try:
                timesteps, mean_rewards, results = load_run(algo, reward, seed)
            except FileNotFoundError:
                continue

            ttt.append(timesteps_to_threshold(timesteps, mean_rewards))
            aucs.append(area_under_curve(timesteps, mean_rewards))
            successes.append(float((results[-1] >= SUCCESS_THRESHOLD).mean()))
            finals.append(float(mean_rewards[-1]))

        if not finals:
            continue

        rows.append({
            "algorithm": algo,
            "reward_config": reward,
            "n_seeds": len(finals),
            "n_seeds_completed": sum(1 for seed in SEEDS if _is_complete(f"{algo}_{reward}_seed{seed}")),
            "final_reward_mean": np.mean(finals),
            "final_reward_std": np.std(finals),
            "success_rate_mean": np.mean(successes),
            "timesteps_to_200_mean": nanmean_or_nan(ttt),
            "timesteps_to_200_std": nanstd_or_nan(ttt),
            "auc_mean": np.mean(aucs),
            "auc_std": np.std(aucs),
        })

    df = pd.DataFrame(rows)
    if save_csv and not df.empty:
        os.makedirs(TABLES_DIR, exist_ok=True)
        df.to_csv(os.path.join(TABLES_DIR, "results_summary.csv"), index=False)
    return df


# LEARNING CURVES (per algorithm)

def _collect_curves(algo, reward):
    """Gather learning curves from all seeds for one (algo, reward) pair."""
    curves, all_timesteps = [], None
    for seed in SEEDS:
        try:
            timesteps, mean_rewards, _ = load_run(algo, reward, seed)
        except FileNotFoundError:
            continue
        curves.append(mean_rewards)
        all_timesteps = timesteps
    return curves, all_timesteps


def make_learning_curve_figure(algo):
    """
    Build a matplotlib figure showing learning curves for one algorithm,
    with one line per reward configuration (mean ± std across seeds).
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    any_curve = False

    for reward in REWARD_CONFIGS:
        curves, all_timesteps = _collect_curves(algo, reward)
        if not curves:
            continue

        # Align curves to the shortest length (seeds may differ slightly)
        min_len = min(len(c) for c in curves)
        curves = np.array([c[:min_len] for c in curves])
        timesteps_trunc = all_timesteps[:min_len]
        mean_curve = curves.mean(axis=0)
        std_curve = curves.std(axis=0)

        color = REWARD_COLORS.get(reward, None)
        ax.plot(timesteps_trunc, mean_curve, label=reward, color=color)
        ax.fill_between(
            timesteps_trunc,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2,
            color=color,
        )
        any_curve = True

    if not any_curve:
        plt.close(fig)
        return None

    ax.axhline(
        SUCCESS_THRESHOLD,
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"success threshold ({int(SUCCESS_THRESHOLD)})",
    )
    ax.set_title(f"{algo.upper()} on LunarLanderContinuous — Learning Curves")
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Mean evaluation reward (unshaped)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_learning_curves():
    """Save learning curve plots for every algorithm that has data."""
    saved = []
    for algo in ALGORITHMS:
        fig = make_learning_curve_figure(algo)
        if fig is None:
            continue
        os.makedirs(FIGURES_DIR, exist_ok=True)
        out_path = os.path.join(FIGURES_DIR, f"{algo}_learning_curves.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved.append(out_path)
        print(f"Saved {out_path}")
    return saved


# COMPARATIVE BAR CHART (all algorithms × all reward configs)

def make_comparative_bar_chart():
    """
    Grouped bar chart: X-axis = algorithms, bars = reward configs.
    Y-axis = mean final reward. Error bars show ± std across seeds.

    This is the key figure for answering: "Which algorithm benefits
    most from reward shaping?"
    """
    df = build_summary_table(save_csv=False)
    if df.empty:
        return None

    algos_in_data = [a for a in ALGORITHMS if a in df["algorithm"].values]
    if not algos_in_data:
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    n_algos = len(algos_in_data)
    n_rewards = len(REWARD_CONFIGS)
    bar_width = 0.8 / n_rewards  # Space bars within each algorithm group
    x = np.arange(n_algos)

    for j, reward in enumerate(REWARD_CONFIGS):
        means, stds = [], []
        for algo in algos_in_data:
            row = df[(df["algorithm"] == algo) & (df["reward_config"] == reward)]
            if row.empty:
                means.append(0)
                stds.append(0)
            else:
                means.append(row["final_reward_mean"].values[0])
                stds.append(row["final_reward_std"].values[0])

        color = REWARD_COLORS.get(reward, None)
        offset = (j - n_rewards / 2 + 0.5) * bar_width
        ax.bar(x + offset, means, bar_width, yerr=stds,
               label=reward, color=color, capsize=3, alpha=0.85)

    ax.axhline(SUCCESS_THRESHOLD, color="gray", linestyle="--", linewidth=1,
               label=f"success ({int(SUCCESS_THRESHOLD)})")
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in algos_in_data])
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Mean Final Reward (unshaped)")
    ax.set_title("Algorithm Comparison — Final Reward by Reward Shaping Config")
    ax.legend(title="Reward Shaping")
    fig.tight_layout()
    return fig


def plot_comparative_bar_chart():
    """Save the comparative bar chart if data is available."""
    fig = make_comparative_bar_chart()
    if fig is None:
        return None
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, "comparative_bar_chart.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


# HYPERPARAMETER SENSITIVITY ANALYSIS

def make_hyperparam_sensitivity_figure():
    """
    Create a figure showing how different hyperparameter settings
    affect training performance under reward shaping.

    Layout: one subplot per learning rate, lines for each network size.
    """
    learning_rates = HYPERPARAM_GRID["learning_rate"]
    net_archs = HYPERPARAM_GRID["net_arch"]
    algo = HYPERPARAM_ALGO
    reward = HYPERPARAM_REWARD

    n_lr = len(learning_rates)
    fig, axes = plt.subplots(1, n_lr, figsize=(7 * n_lr, 5), sharey=True)

    # Handle the case where there's only one learning rate
    if n_lr == 1:
        axes = [axes]

    any_data = False
    line_colors = ["#1976D2", "#D32F2F", "#388E3C", "#F57C00"]

    for i, lr in enumerate(learning_rates):
        ax = axes[i]

        for j, net_arch in enumerate(net_archs):
            arch_str = "-".join(str(n) for n in net_arch)
            curves, all_timesteps = [], None

            for seed in SEEDS:
                try:
                    timesteps, mean_rewards, _ = load_run(
                        algo, reward, seed, lr=lr, net_arch=net_arch
                    )
                except FileNotFoundError:
                    continue
                curves.append(mean_rewards)
                all_timesteps = timesteps

            if not curves:
                continue

            min_len = min(len(c) for c in curves)
            curves = np.array([c[:min_len] for c in curves])
            timesteps_trunc = all_timesteps[:min_len]
            mean_curve = curves.mean(axis=0)
            std_curve = curves.std(axis=0)

            color = line_colors[j % len(line_colors)]
            ax.plot(timesteps_trunc, mean_curve,
                    label=f"net=[{arch_str}]", color=color)
            ax.fill_between(
                timesteps_trunc,
                mean_curve - std_curve,
                mean_curve + std_curve,
                alpha=0.15,
                color=color,
            )
            any_data = True

        ax.axhline(SUCCESS_THRESHOLD, color="gray", linestyle="--", linewidth=1)
        ax.set_title(f"lr = {lr}")
        ax.set_xlabel("Training timesteps")
        if i == 0:
            ax.set_ylabel("Mean evaluation reward (unshaped)")
        if ax.get_legend_handles_labels()[0]:
            ax.legend()

    if not any_data:
        plt.close(fig)
        return None

    fig.suptitle(
        f"Hyperparameter Sensitivity — {algo.upper()} with '{reward}' shaping",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    return fig


def plot_hyperparam_sensitivity():
    """Save the hyperparameter sensitivity figure if data is available."""
    fig = make_hyperparam_sensitivity_figure()
    if fig is None:
        return None
    os.makedirs(FIGURES_DIR, exist_ok=True)
    out_path = os.path.join(FIGURES_DIR, "hyperparam_sensitivity.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")
    return out_path


# UTILITY

def count_completed_runs():
    """Count how many runs have finished out of the full grid."""
    total = len(ALGORITHMS) * len(REWARD_CONFIGS) * len(SEEDS)
    done = 0
    for algo, reward, seed in itertools.product(ALGORITHMS, REWARD_CONFIGS, SEEDS):
        path = os.path.join(RESULTS_DIR, f"{algo}_{reward}_seed{seed}", "evaluations.npz")
        if os.path.exists(path):
            done += 1
    return done, total


# MAIN

def main():
    print("Building summary table...")
    df = build_summary_table()
    if not df.empty:
        print(df.to_string(index=False))
        print(f"\nSaved {os.path.join(TABLES_DIR, 'results_summary.csv')}")
    else:
        print("No results found. Have you run the training yet?")

    print("\nGenerating learning curves...")
    plot_learning_curves()

    print("\nGenerating comparative bar chart...")
    plot_comparative_bar_chart()

    print("\nGenerating hyperparameter sensitivity plot...")
    plot_hyperparam_sensitivity()

    print("\nDone! Check reports/figures and reports/tables.")


if __name__ == "__main__":
    main()
