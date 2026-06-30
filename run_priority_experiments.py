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

from config import (
    ALGORITHMS,
    DEFAULT_EVAL_FREQ,
    HYPERPARAM_ALGO,
    HYPERPARAM_GRID,
    HYPERPARAM_REWARD,
    RESULTS_DIR,
    REWARD_CONFIGS,
    SEEDS,
    build_run_name,
    is_run_complete,
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
    # Complete the planned algorithm x reward x seed experiment grid.
    "coverage": [
        {"algo": algo, "reward": reward, "seed": seed}
        for algo in ALGORITHMS
        for reward in REWARD_CONFIGS
        for seed in SEEDS
    ],
}



def _train(job, timesteps, eval_freq, force=False, device="auto"):
    run_name = build_run_name(
        job["algo"],
        job["reward"],
        job["seed"],
        lr=job.get("lr"),
        net_arch=job.get("net_arch"),
    )

    if is_run_complete(run_name) and not force:
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
        "--device",
        device,
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
            for seed in SEEDS:
                jobs.append({
                    "algo": HYPERPARAM_ALGO,
                    "reward": HYPERPARAM_REWARD,
                    "seed": seed,
                    "lr": lr,
                    "net_arch": net_arch,
                })
    return jobs


def main():
    default_workers = min(os.cpu_count() or 1, 4)

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
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=f"Max parallel workers (default: {default_workers}).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Training device passed to train.py.",
    )
    args = parser.parse_args()

    jobs = _hyperparam_jobs() if args.phase == "hyperparam" else PHASES[args.phase]
    print(
        f"Phase: {args.phase} | Jobs: {len(jobs)} | "
        f"Timesteps: {args.timesteps} | Device: {args.device}"
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        # We prepare all jobs and throw them to the library to distribute them to the processor
        futures = [
            executor.submit(
                _train,
                job,
                args.timesteps,
                args.eval_freq,
                args.force,
                args.device,
            )
            for job in jobs
        ]
        
        # Here we check the results and print if there was an unexpected error
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An unexpected error occurred during training: {e}")

    print("Done. Run `python evaluate.py` and `python analyze_results.py` next.")


if __name__ == "__main__":
    main()
