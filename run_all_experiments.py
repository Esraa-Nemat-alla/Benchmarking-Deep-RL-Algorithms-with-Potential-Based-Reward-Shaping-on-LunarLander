"""
Run the full experiment grid (algorithm × reward × seed) plus a
hyperparameter sensitivity study.

Phase 1 — Default grid:
    5 algorithms × 4 reward configs × 3 seeds = 60 runs

Phase 2 — Hyperparameter sensitivity:
    Tests different learning rates and network sizes on one algorithm
    to see how hyperparameters interact with reward shaping.
"""
import itertools
import subprocess
import sys

from config import (
    ALGORITHMS,
    DEFAULT_EVAL_FREQ,
    DEFAULT_TIMESTEPS,
    HYPERPARAM_ALGO,
    HYPERPARAM_GRID,
    HYPERPARAM_REWARD,
    REWARD_CONFIGS,
    SEEDS,
)


def _run_train(algo, reward, seed, timesteps, eval_freq, lr=None, net_arch=None):
    """Launch train.py as a subprocess with the given arguments."""
    cmd = [
        sys.executable, "train.py",
        "--algo", algo,
        "--reward", reward,
        "--seed", str(seed),
        "--timesteps", str(timesteps),
        "--eval-freq", str(eval_freq),
    ]
    # Append optional hyperparameter overrides
    if lr is not None:
        cmd.extend(["--lr", str(lr)])
    if net_arch is not None:
        cmd.extend(["--net-arch"] + [str(n) for n in net_arch])

    subprocess.run(cmd, check=True)


def run_default_grid():
    """
    Phase 1: Run all (algorithm × reward × seed) combinations
    with default hyperparameters.
    """
    combos = list(itertools.product(ALGORITHMS, REWARD_CONFIGS, SEEDS))
    total = len(combos)
    print(f"\n{'='*60}")
    print(f"PHASE 1: Default grid — {total} experiments")
    print(f"{'='*60}\n")

    for i, (algo, reward, seed) in enumerate(combos, 1):
        print(f"\n[{i}/{total}] algo={algo}  reward={reward}  seed={seed}")
        _run_train(algo, reward, seed, DEFAULT_TIMESTEPS, DEFAULT_EVAL_FREQ)


def run_hyperparam_study():
    """
    Phase 2: Sweep over learning rates and network architectures
    for one algorithm + one reward config, across all seeds.

    This tells us whether the benefit of reward shaping holds up
    when we change the agent's learning dynamics.
    """
    learning_rates = HYPERPARAM_GRID["learning_rate"]
    net_archs = HYPERPARAM_GRID["net_arch"]
    combos = list(itertools.product(learning_rates, net_archs, SEEDS))
    total = len(combos)

    print(f"\n{'='*60}")
    print(f"PHASE 2: Hyperparameter sensitivity — {total} experiments")
    print(f"  Algorithm: {HYPERPARAM_ALGO}  |  Reward: {HYPERPARAM_REWARD}")
    print(f"  Learning rates: {learning_rates}")
    print(f"  Network architectures: {net_archs}")
    print(f"{'='*60}\n")

    for i, (lr, net_arch, seed) in enumerate(combos, 1):
        arch_str = "-".join(str(n) for n in net_arch)
        print(f"\n[{i}/{total}] lr={lr}  net_arch={arch_str}  seed={seed}")
        _run_train(
            HYPERPARAM_ALGO, HYPERPARAM_REWARD, seed,
            DEFAULT_TIMESTEPS, DEFAULT_EVAL_FREQ,
            lr=lr, net_arch=net_arch,
        )


def main():
    run_default_grid()
    run_hyperparam_study()
    print("\n All experiments completed!")


if __name__ == "__main__":
    main()
