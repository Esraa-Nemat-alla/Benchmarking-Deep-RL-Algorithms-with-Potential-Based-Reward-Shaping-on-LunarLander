"""Run the full experiment grid (algorithm x reward x seed)."""
import itertools
import subprocess
import sys

from config import ALGORITHMS, DEFAULT_EVAL_FREQ, DEFAULT_TIMESTEPS, REWARD_CONFIGS, SEEDS


def main():
    combos = list(itertools.product(ALGORITHMS, REWARD_CONFIGS, SEEDS))
    print(f"Running {len(combos)} experiments...")

    for i, (algo, reward, seed) in enumerate(combos, 1):
        print(f"\n[{i}/{len(combos)}] algo={algo} reward={reward} seed={seed}")
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
            str(DEFAULT_TIMESTEPS),
            "--eval-freq",
            str(DEFAULT_EVAL_FREQ),
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
