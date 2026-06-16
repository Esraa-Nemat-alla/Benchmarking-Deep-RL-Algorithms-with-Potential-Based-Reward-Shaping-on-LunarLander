"""Shared experiment settings used across train, evaluate, and the GUI."""

GAMMA = 0.99
ALGORITHMS = ["ppo"]
REWARD_CONFIGS = ["none", "distance", "angle", "combined"]
SEEDS = [0, 1, 2]
DEFAULT_TIMESTEPS = 1_000_000
DEFAULT_EVAL_FREQ = 10_000
SUCCESS_THRESHOLD = 200.0
RESULTS_DIR = "results"
