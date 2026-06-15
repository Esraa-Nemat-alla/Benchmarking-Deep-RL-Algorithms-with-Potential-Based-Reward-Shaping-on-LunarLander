"""
Aggregate results across seeds and produce:
    - results_summary.csv : one row per (algorithm, reward config) with
        mean/std of final reward, success rate, sample efficiency, and AUC.
    - <algo>_learning_curves.png : mean +/- std evaluation reward over
        training, one curve per reward configuration.

Run this after train.py / run_all_experiments.py have produced files under
results/<algo>_<reward>_seed<seed>/evaluations.npz


"""

import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ALGORITHMS = ["dqn", "ppo", "a2c"]
REWARD_CONFIGS = ["none", "distance", "angle", "combined"]
SEEDS = [0, 1, 2]
RESULTS_DIR = "results"

SUCCESS_THRESHOLD = 200.0


def load_run(algo, reward, seed):
    """Load (timesteps, mean_reward_per_eval, raw_per_episode_rewards)."""
    path = os.path.join(RESULTS_DIR, f"{algo}_{reward}_seed{seed}", "evaluations.npz")
    data = np.load(path)
    timesteps = data["timesteps"]
    results = data["results"]  
    mean_rewards = results.mean(axis=1)
    return timesteps, mean_rewards, results


def timesteps_to_threshold(timesteps, mean_rewards, threshold=SUCCESS_THRESHOLD):
    """First timestep at which the mean evaluation reward reaches the threshold."""
    above = np.where(mean_rewards >= threshold)[0]
    if len(above) == 0:
        return np.nan
    return float(timesteps[above[0]])


def area_under_curve(timesteps, mean_rewards):
    """Area under the learning curve (trapezoidal rule)."""
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(mean_rewards, timesteps))


def build_summary_table():
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
    df.to_csv("results_summary.csv", index=False)
    return df


def plot_learning_curves():
    for algo in ALGORITHMS:
        plt.figure(figsize=(8, 5))
        any_curve = False

        for reward in REWARD_CONFIGS:
            curves, all_timesteps = [], None

            for seed in SEEDS:
                try:
                    timesteps, mean_rewards, _ = load_run(algo, reward, seed)
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

            plt.plot(timesteps_trunc, mean_curve, label=reward)
            plt.fill_between(timesteps_trunc, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)
            any_curve = True

        if not any_curve:
            plt.close()
            continue

        plt.axhline(SUCCESS_THRESHOLD, color="gray", linestyle="--", linewidth=1, label="success threshold (200)")
        plt.title(f"{algo.upper()} on LunarLander - learning curves")
        plt.xlabel("Training timesteps")
        plt.ylabel("Mean evaluation reward")
        plt.legend()
        plt.tight_layout()
        out_path = f"{algo}_learning_curves.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved {out_path}")


def main():
    df = build_summary_table()
    print(df.to_string(index=False))
    print("\nSaved results_summary.csv")
    plot_learning_curves()


if __name__ == "__main__":
    main()
