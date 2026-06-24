"""
Run the next most useful experiments while skipping completed runs.

This is meant for improving the project after the pilot grid is working.

Examples:
    python run_priority_experiments.py --phase seeds --timesteps 50000
    python run_priority_experiments.py --phase stronger --timesteps 100000
    python run_priority_experiments.py --phase hyperparam --timesteps 50000
"""
import argparse
import os
import subprocess
import sys
import concurrent.futures
import numpy as np

from config import (
    ALGORITHMS,
    DEFAULT_EVAL_FREQ,
    HYPERPARAM_GRID,
    RESULTS_DIR,
    REWARD_CONFIGS,
    SEEDS,
)


PHASES = {
    # Add seed coverage for the strongest current configurations.
    "seeds": [
        {"algo": "td3", "reward": "combined", "seed": 1},
        {"algo": "td3", "reward": "combined", "seed": 2},
        {"algo": "sac", "reward": "distance", "seed": 1},
        {"algo": "sac", "reward": "distance", "seed": 2},
        {"algo": "ppo", "reward": "none", "seed": 2},
    ],
    # Longer runs for the most promising candidates.
    "stronger": [
        {"algo": "td3", "reward": "combined", "seed": 0},
        {"algo": "sac", "reward": "distance", "seed": 0},
        {"algo": "ppo", "reward": "none", "seed": 0},
    ],
    # Complete the planned 5 x 4 x 3 experiment grid.
    "coverage": [
        {"algo": algo, "reward": reward, "seed": seed}
        for algo in ALGORITHMS
        for reward in REWARD_CONFIGS
        for seed in SEEDS
    ],
}


def _run_name(algo, reward, seed, lr=None, net_arch=None):
    name = f"{algo}_{reward}_seed{seed}"
    if lr is not None:
        name += f"_lr{lr}"
    if net_arch is not None:
        name += "_net" + "-".join(str(n) for n in net_arch)
    return name


def _is_complete(run_name):
    file_path = os.path.join(RESULTS_DIR, run_name, "evaluations.npz")
    if not os.path.exists(file_path):
        return False
    
    try:
        # We don't just check for the file, we try to open it and read the data
        data = np.load(file_path)
        _ = data["timesteps"] # We make sure the timesteps array is actually there
        return True
    except Exception:
        # If the file is broken, we come here and consider the run completed
        print(f"[Warning] The evaluations.npz file is broken for the run {run_name}. The run will be retrained.")
        return False


def _train(job, timesteps, eval_freq, force=False):
    run_name = _run_name(
        job["algo"],
        job["reward"],
        job["seed"],
        lr=job.get("lr"),
        net_arch=job.get("net_arch"),
    )

    if _is_complete(run_name) and not force:
        print(f"[skip] {run_name} already has evaluations.npz")
        return

    cmd = [
        sys.executable,
        "train.py",
        "--algo",
        job["algo"],
        "--reward",
        job["reward"],
        "--seed",
        str(job["seed"]),
        "--timesteps",
        str(timesteps),
        "--eval-freq",
        str(eval_freq),
    ]
    if "lr" in job:
        cmd.extend(["--lr", str(job["lr"])])
    if "net_arch" in job:
        cmd.append("--net-arch")
        cmd.extend(str(n) for n in job["net_arch"])

    print(f"[run] {run_name}")
    subprocess.run(cmd, check=True)


def _hyperparam_jobs():
    jobs = []
    for lr in HYPERPARAM_GRID["learning_rate"]:
        for net_arch in HYPERPARAM_GRID["net_arch"]:
            for seed in [0, 1, 2]:
                jobs.append({
                    "algo": "ppo",
                    "reward": "combined",
                    "seed": seed,
                    "lr": lr,
                    "net_arch": net_arch,
                })
    return jobs


def main():
    max_workers = os.cpu_count()

    parser = argparse.ArgumentParser(
        description="Run prioritized experiments and skip completed runs."
    )
    parser.add_argument(
        "--phase",
        choices=["seeds", "stronger", "coverage", "hyperparam"],
        default="seeds",
        help="Which set of experiments to run.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50_000,
        help="Training timesteps per run.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=DEFAULT_EVAL_FREQ,
        help="Evaluation frequency.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run jobs even if evaluations.npz already exists.",
    )
    args = parser.parse_args()

    jobs = _hyperparam_jobs() if args.phase == "hyperparam" else PHASES[args.phase]
    print(f"Phase: {args.phase} | Jobs: {len(jobs)} | Timesteps: {args.timesteps}")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # We prepare all jobs and throw them to the library to distribute them to the processor
        futures = [executor.submit(_train, job, args.timesteps, args.eval_freq, args.force) for job in jobs]
        
        # Here we check the results and print if there was an unexpected error
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An unexpected error occurred during training: {e}")

    print("Done. Run `python evaluate.py` and `python analyze_results.py` next.")


if __name__ == "__main__":
    main()
