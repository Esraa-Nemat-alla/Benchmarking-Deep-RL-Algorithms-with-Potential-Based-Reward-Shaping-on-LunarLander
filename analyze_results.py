"""
Create higher-level analysis tables and a short interpretation report.

Run after evaluate.py:
    python analyze_results.py
"""
import os

import numpy as np
import pandas as pd


REPORTS_DIR = "reports"
TABLES_DIR = os.path.join(REPORTS_DIR, "tables")
SUMMARY_PATH = os.path.join(TABLES_DIR, "results_summary.csv")
BEST_CONFIGS_PATH = os.path.join(TABLES_DIR, "best_configs.csv")
SHAPING_EFFECTS_PATH = os.path.join(TABLES_DIR, "reward_shaping_effects.csv")
REPORT_PATH = os.path.join(REPORTS_DIR, "RESULTS_INTERPRETATION.md")


def _fmt(value, digits=1):
    if pd.isna(value):
        return "N/A"
    return f"{value:.{digits}f}"


def _load_summary():
    if not os.path.exists(SUMMARY_PATH):
        raise FileNotFoundError(
            f"Missing {SUMMARY_PATH}. Run `python evaluate.py` first."
        )
    return pd.read_csv(SUMMARY_PATH)


def build_best_configs(df):
    """Best reward configuration per algorithm by final mean reward."""
    idx = df.groupby("algorithm")["final_reward_mean"].idxmax()
    best = df.loc[idx].copy()
    return best.sort_values("final_reward_mean", ascending=False)


def build_shaping_effects(df):
    """Compare each shaping config to the no-shaping baseline per algorithm."""
    rows = []
    for algo, group in df.groupby("algorithm"):
        baseline = group[group["reward_config"] == "none"]
        if baseline.empty:
            continue

        base = baseline.iloc[0]
        for _, row in group[group["reward_config"] != "none"].iterrows():
            rows.append({
                "algorithm": algo,
                "reward_config": row["reward_config"],
                "baseline_final_reward": base["final_reward_mean"],
                "shaped_final_reward": row["final_reward_mean"],
                "delta_final_reward": (
                    row["final_reward_mean"] - base["final_reward_mean"]
                ),
                "baseline_success_rate": base["success_rate_mean"],
                "shaped_success_rate": row["success_rate_mean"],
                "delta_success_rate": (
                    row["success_rate_mean"] - base["success_rate_mean"]
                ),
                "baseline_auc": base["auc_mean"],
                "shaped_auc": row["auc_mean"],
                "delta_auc": row["auc_mean"] - base["auc_mean"],
                "n_seeds": min(row["n_seeds"], base["n_seeds"]),
            })

    effects = pd.DataFrame(rows)
    if effects.empty:
        return effects
    return effects.sort_values("delta_final_reward", ascending=False)


def write_interpretation(df, best_configs, effects):
    os.makedirs(REPORTS_DIR, exist_ok=True)

    best_overall = df.sort_values("final_reward_mean", ascending=False).iloc[0]
    best_shaping = effects.iloc[0] if not effects.empty else None
    strongest_success = df.sort_values("success_rate_mean", ascending=False).iloc[0]

    lines = [
        "# Results Interpretation",
        "",
        "## Current Experiment Coverage",
        "",
        f"- Completed summary rows: {len(df)}.",
        f"- Algorithms represented: {', '.join(sorted(df['algorithm'].unique()))}.",
        f"- Reward configs represented: {', '.join(sorted(df['reward_config'].unique()))}.",
        "- Current runs are mostly pilot-scale, so conclusions should be treated as preliminary until more seeds and longer training are added.",
        "",
        "## Best Current Configurations",
        "",
        "| Algorithm | Best reward config | Final reward | Success rate | Seeds |",
        "|---|---|---:|---:|---:|",
    ]

    for _, row in best_configs.sort_values("algorithm").iterrows():
        lines.append(
            "| {algorithm} | {reward_config} | {reward} | {success}% | {seeds} |".format(
                algorithm=row["algorithm"].upper(),
                reward_config=row["reward_config"],
                reward=_fmt(row["final_reward_mean"]),
                success=_fmt(100 * row["success_rate_mean"], 0),
                seeds=int(row["n_seeds"]),
            )
        )

    lines.extend([
        "",
        "## Main Observations",
        "",
        "- Best final reward so far: "
        f"{best_overall['algorithm'].upper()} with `{best_overall['reward_config']}` "
        f"({_fmt(best_overall['final_reward_mean'])}).",
        "- Highest success rate so far: "
        f"{strongest_success['algorithm'].upper()} with `{strongest_success['reward_config']}` "
        f"({_fmt(100 * strongest_success['success_rate_mean'], 0)}%).",
    ])

    if best_shaping is not None:
        lines.append(
            "- Largest shaping improvement over baseline so far: "
            f"{best_shaping['algorithm'].upper()} with `{best_shaping['reward_config']}` "
            f"(delta final reward = {_fmt(best_shaping['delta_final_reward'])})."
        )

    lines.extend([
        "",
        "## Reward Shaping Effect",
        "",
        "Positive delta means the shaped version outperformed the no-shaping baseline for that algorithm.",
        "",
        "| Algorithm | Shaping | Delta final reward | Delta success rate | Seeds |",
        "|---|---|---:|---:|---:|",
    ])

    if effects.empty:
        lines.append("| N/A | N/A | N/A | N/A | N/A |")
    else:
        for _, row in effects.iterrows():
            lines.append(
                "| {algorithm} | {reward_config} | {delta_reward} | {delta_success}% | {seeds} |".format(
                    algorithm=row["algorithm"].upper(),
                    reward_config=row["reward_config"],
                    delta_reward=_fmt(row["delta_final_reward"]),
                    delta_success=_fmt(100 * row["delta_success_rate"], 0),
                    seeds=int(row["n_seeds"]),
                )
            )

    lines.extend([
        "",
        "## Recommended Next Experiments",
        "",
        "1. Add seeds 1 and 2 for the strongest configurations, especially TD3 combined, SAC distance, and PPO none.",
        "2. Increase timesteps for the strongest candidates from 50k to 100k or 500k.",
        "3. Run the planned hyperparameter sensitivity study after the main comparison stabilizes.",
        "4. Keep evaluation on the unshaped reward so the shaped and unshaped agents remain comparable.",
        "",
        "## Limitations",
        "",
        "- Many rows currently have one seed, so variance estimates are not reliable yet.",
        "- Short training can make off-policy methods look unstable or undertrained.",
        "- GIF replay reward can differ from mean evaluation reward because it is one sampled episode.",
    ])

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    df = _load_summary()
    os.makedirs(TABLES_DIR, exist_ok=True)

    best_configs = build_best_configs(df)
    effects = build_shaping_effects(df)

    best_configs.to_csv(BEST_CONFIGS_PATH, index=False)
    effects.to_csv(SHAPING_EFFECTS_PATH, index=False)
    write_interpretation(df, best_configs, effects)

    print(f"Saved {BEST_CONFIGS_PATH}")
    print(f"Saved {SHAPING_EFFECTS_PATH}")
    print(f"Saved {REPORT_PATH}")


if __name__ == "__main__":
    main()
