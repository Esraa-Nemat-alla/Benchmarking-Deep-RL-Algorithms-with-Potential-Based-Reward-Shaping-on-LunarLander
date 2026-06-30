"""Run all missing experiments and regenerate reports from one command."""
import argparse
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
    build_run_name,
    is_run_complete,
)


def _run_train(
    algo,
    reward,
    seed,
    timesteps,
    eval_freq,
    device,
    force=False,
    lr=None,
    net_arch=None,
):
    """Launch train.py unless the run already has a valid evaluations.npz."""
    run_name = build_run_name(algo, reward, seed, lr=lr, net_arch=net_arch)
    if is_run_complete(run_name) and not force:
        print(f"[skip] {run_name} already has evaluations.npz")
        return

    cmd = [
        sys.executable,
        "train.py",
        "--algo",
        algo,
        "--reward",
        reward,
        "--seed",
        str(seed),
        "--timesteps",
        str(timesteps),
        "--eval-freq",
        str(eval_freq),
        "--device",
        device,
    ]
    if lr is not None:
        cmd.extend(["--lr", str(lr)])
    if net_arch is not None:
        cmd.extend(["--net-arch"] + [str(n) for n in net_arch])

    print(f"[run] {run_name}")
    subprocess.run(cmd, check=True)


def run_default_grid(args):
    """Run the default algorithm x reward x seed grid."""
    combos = list(itertools.product(ALGORITHMS, REWARD_CONFIGS, SEEDS))
    total = len(combos)
    print(f"\n{'=' * 60}", flush=True)
    print(f"PHASE 1: Default grid - {total} experiments", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    for index, (algo, reward, seed) in enumerate(combos, 1):
        print(f"\n[{index}/{total}] algo={algo} reward={reward} seed={seed}")
        _run_train(
            algo,
            reward,
            seed,
            args.timesteps,
            args.eval_freq,
            args.device,
            force=args.force,
        )


def run_hyperparam_study(args):
    """Run the hyperparameter sensitivity study."""
    combos = list(itertools.product(
        HYPERPARAM_GRID["learning_rate"],
        HYPERPARAM_GRID["net_arch"],
        SEEDS,
    ))
    total = len(combos)
    print(f"\n{'=' * 60}", flush=True)
    print(f"PHASE 2: Hyperparameter study - {total} experiments", flush=True)
    print(f"{'=' * 60}\n", flush=True)

    for index, (lr, net_arch, seed) in enumerate(combos, 1):
        arch_str = "-".join(str(n) for n in net_arch)
        print(
            f"\n[{index}/{total}] algo={HYPERPARAM_ALGO} "
            f"reward={HYPERPARAM_REWARD} lr={lr} net={arch_str} seed={seed}"
        )
        _run_train(
            HYPERPARAM_ALGO,
            HYPERPARAM_REWARD,
            seed,
            args.timesteps,
            args.eval_freq,
            args.device,
            force=args.force,
            lr=lr,
            net_arch=net_arch,
        )


def run_reports():
    """Regenerate evaluation tables, figures, and interpretation reports."""
    print(f"\n{'=' * 60}", flush=True)
    print("PHASE 3: Evaluation and analysis reports", flush=True)
    print(f"{'=' * 60}\n", flush=True)
    subprocess.run([sys.executable, "evaluate.py"], check=True)
    subprocess.run([sys.executable, "analyze_results.py"], check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run all missing experiments, then regenerate reports."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help="Training timesteps per missing run.",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=DEFAULT_EVAL_FREQ,
        help="Evaluation frequency during training.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Training device passed to Stable-Baselines3.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain runs even if evaluations.npz already exists.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Only regenerate evaluation and analysis reports.",
    )
    parser.add_argument(
        "--skip-hyperparam",
        action="store_true",
        help="Skip the hyperparameter sensitivity study.",
    )
    args = parser.parse_args()

    if not args.skip_training:
        run_default_grid(args)
        if not args.skip_hyperparam:
            run_hyperparam_study(args)

    run_reports()
    print("\nAll workflow steps completed!")


if __name__ == "__main__":
    main()
