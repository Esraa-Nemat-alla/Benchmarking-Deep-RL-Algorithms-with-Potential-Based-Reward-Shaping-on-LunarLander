"""
Script to automatically loop through and run all experiments.
"""
import itertools
import subprocess
import sys

# Define our search space
ALGORITHMS = ["ppo"]
REWARD_CONFIGS = ["none", "distance", "angle", "combined"]
SEEDS = [0, 1, 2]
TIMESTEPS = 1_000_000
EVAL_FREQ = 10_000

def main():
    combos = list(itertools.product(ALGORITHMS, REWARD_CONFIGS, SEEDS))
    print(f"Running {len(combos)} experiments...")

    for i, (algo, reward, seed) in enumerate(combos, 1):
        print(f"\n[{i}/{len(combos)}] algo={algo} reward={reward} seed={seed}")
        
        # Build the command to call train.py
        cmd = [
            sys.executable, "train.py",
            "--algo", algo,
            "--reward", reward,
            "--seed", str(seed),
            "--timesteps", str(TIMESTEPS),
            "--eval-freq", str(EVAL_FREQ),
        ]
        
        # Execute the command
        subprocess.run(cmd, check=True)

if __name__ == "__main__":
    main()
