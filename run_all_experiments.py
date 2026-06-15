"""
Run the full experiment grid: 3 algorithms x 4 reward configurations x N seeds.

This just calls train.py once per combination as a separate process, so a
crash in one run doesn't take down the others . (e.g. TIMESTEPS = 100_000,
SEEDS = [0, 1]) then we will scale up in upcoming updates

"""

import itertools
import subprocess
import sys

ALGORITHMS = ["dqn", "ppo", "a2c"]
REWARD_CONFIGS = ["none", "distance", "angle", "combined"]
SEEDS = [0, 1, 2]
TIMESTEPS = 1_000_000
EVAL_FREQ = 10_000


def main():
    combos = list(itertools.product(ALGORITHMS, REWARD_CONFIGS, SEEDS))
    print(f"Running {len(combos)} experiments "
          f"({len(ALGORITHMS)} algos x {len(REWARD_CONFIGS)} reward configs x {len(SEEDS)} seeds)")

    for i, (algo, reward, seed) in enumerate(combos, 1):
        print(f"\n[{i}/{len(combos)}] algo={algo} reward={reward} seed={seed}")
        cmd = [
            sys.executable, "train.py",
            "--algo", algo,
            "--reward", reward,
            "--seed", str(seed),
            "--timesteps", str(TIMESTEPS),
            "--eval-freq", str(EVAL_FREQ),
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
