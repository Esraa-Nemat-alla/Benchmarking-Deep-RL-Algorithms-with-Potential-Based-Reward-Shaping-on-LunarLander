"""
Aggregate results across seeds and produce summary tables and learning curves.
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
    REWARD_CONFIGS,
    RESULTS_DIR,
    SEEDS,
    SUCCESS_THRESHOLD,
)


def load_run(algo, reward, seed):
    path = os.path.join(RESULTS_DIR, f"{algo}_{reward}_seed{seed}", "evaluations.npz")
    data = np.load(path)
    timesteps = data["timesteps"]
    results = data["results"]
    mean_rewards = results.mean(axis=1)
    return timesteps, mean_rewards, results


def timesteps_to_threshold(timesteps, mean_rewards, threshold=SUCCESS_THRESHOLD):
    above = np.where(mean_rewards >= threshold)[0]
    if len(above) == 0:
        return np.nan
    return float(timesteps[above[0]])


def area_under_curve(timesteps, mean_rewards):
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(mean_rewards, timesteps))


def build_summary_table(save_csv=True):
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
            "final_reward_mean": np.mean(finals),
            "final_reward_std": np.std(finals),
            "success_rate_mean": np.mean(successes),
            "timesteps_to_200_mean": np.nanmean(ttt),
            "timesteps_to_200_std": np.nanstd(ttt),
            "auc_mean": np.mean(aucs),
            "auc_std": np.std(aucs),
        })

    df = pd.DataFrame(rows)
    if save_csv and not df.empty:
        df.to_csv("results_summary.csv", index=False)
    return df


def _collect_curves(algo, reward):
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
    """Build a matplotlib figure for one algorithm."""
    fig, ax = plt.subplots(figsize=(8, 5))
    any_curve = False

    for reward in REWARD_CONFIGS:
        curves, all_timesteps = _collect_curves(algo, reward)
        if not curves:
            continue

        min_len = min(len(c) for c in curves)
        curves = np.array([c[:min_len] for c in curves])
        timesteps_trunc = all_timesteps[:min_len]
        mean_curve = curves.mean(axis=0)
        std_curve = curves.std(axis=0)

        ax.plot(timesteps_trunc, mean_curve, label=reward)
        ax.fill_between(
            timesteps_trunc,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2,
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
    ax.set_title(f"{algo.upper()} on LunarLanderContinuous - learning curves")
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Mean evaluation reward (unshaped)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_learning_curves():
    saved = []
    for algo in ALGORITHMS:
        fig = make_learning_curve_figure(algo)
        if fig is None:
            continue
        out_path = f"{algo}_learning_curves.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved.append(out_path)
        print(f"Saved {out_path}")
    return saved


def count_completed_runs():
    total = len(ALGORITHMS) * len(REWARD_CONFIGS) * len(SEEDS)
    done = 0
    for algo, reward, seed in itertools.product(ALGORITHMS, REWARD_CONFIGS, SEEDS):
        path = os.path.join(RESULTS_DIR, f"{algo}_{reward}_seed{seed}", "evaluations.npz")
        if os.path.exists(path):
            done += 1
    return done, total


def main():
    df = build_summary_table()
    if not df.empty:
        print(df.to_string(index=False))
        print("\nSaved results_summary.csv")
    else:
        print("No results found. Have you run the training yet?")
    plot_learning_curves()


if __name__ == "__main__":
    main()
