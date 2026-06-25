"""
Shared experiment settings used across train, evaluate, and the GUI.

This is the single source of truth for all experiment parameters.
Change values here and every script will pick them up automatically.
"""
import os

import numpy as np

# Core RL setting 
GAMMA = 0.99  # Discount factor (must match the PBRS wrapper)

# Algorithms to benchmark 
# PPO & A2C  = on-policy  (good for comparison)
# SAC, TD3, DDPG = off-policy (designed for continuous control)
ALGORITHMS = ["ppo", "a2c", "sac", "td3", "ddpg"]

# Reward shaping configurations 
# "none" = vanilla environment reward (baseline)
# "distance" / "angle" / "combined" = PBRS with different potential functions
REWARD_CONFIGS = ["none", "distance", "angle", "combined", "velocity"]

# Experiment grid 
SEEDS = [0, 1, 2]                  # 3 seeds for statistical significance
DEFAULT_TIMESTEPS = 1_000_000      # Total training steps per run
DEFAULT_EVAL_FREQ = 10_000         # Evaluate the agent every N steps
SUCCESS_THRESHOLD = 200.0          # LunarLander "solved" threshold
RESULTS_DIR = "results"

# Hyperparameter sensitivity study 
# We pick one algorithm and test how different learning rates and
# network architectures affect the impact of reward shaping.
HYPERPARAM_ALGO = "ppo"            # Which algorithm to run the sweep on
HYPERPARAM_REWARD = "combined"     # Which reward config to pair with

HYPERPARAM_GRID = {
    # Two learning rates: the SB3 default and a smaller one
    "learning_rate": [3e-4, 1e-4],
    # Two network sizes: a small net and a bigger net
    "net_arch": [
        [64, 64],       # Small: two hidden layers of 64 neurons
        [256, 256],     # Large: two hidden layers of 256 neurons
    ],
}


# ---------------------------------------------------------------------------
# Shared utilities — used by train, evaluate, and experiment runners
# ---------------------------------------------------------------------------

def build_run_name(algo, reward, seed, lr=None, net_arch=None):
    """
    Create a unique folder name for an experiment run.

    If custom hyperparams are provided, they are encoded in the name
    so they don't overwrite the default runs.
    """
    name = f"{algo}_{reward}_seed{seed}"
    if lr is not None:
        name += f"_lr{lr}"
    if net_arch is not None:
        arch_str = "-".join(str(n) for n in net_arch)
        name += f"_net{arch_str}"
    return name


def is_run_complete(run_name):
    """
    Check whether a training run produced a valid evaluations file.

    Returns True only if evaluations.npz exists and can be loaded.
    """
    file_path = os.path.join(RESULTS_DIR, run_name, "evaluations.npz")
    if not os.path.exists(file_path):
        return False

    try:
        data = np.load(file_path)
        _ = data["timesteps"]  # Verify the file is not corrupted
        return True
    except Exception:
        print(
            f"[Warning] Corrupted evaluations.npz for {run_name}. "
            "The run will be retrained."
        )
        return False
