"""
Run the full experiment grid: algorithms x reward configurations x seeds.
"""

import itertools
import subprocess
import sys

from src.config import ALGORITHMS, REWARD_CONFIGS, DEFAULT_TIMESTEPS, DEFAULT_EVAL_FREQ

SEEDS = [0, 1, 2]

def main():
    combos = list(itertools.product(ALGORITHMS, REWARD_CONFIGS, SEEDS))
    print(f"Running {len(combos)} experiments "
          f"({len(ALGORITHMS)} algos x {len(REWARD_CONFIGS)} reward configs x {len(SEEDS)} seeds)")

    for i, (algo, reward, seed) in enumerate(combos, 1):
        print(f"\n[{i}/{len(combos)}] algo={algo} reward={reward} seed={seed}")
        cmd = [
            sys.executable, "-m", "src.train",
            "--algo", algo,
            "--reward", reward,
            "--seed", str(seed),
            "--timesteps", str(DEFAULT_TIMESTEPS),
            "--eval-freq", str(DEFAULT_EVAL_FREQ),
        ]
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
