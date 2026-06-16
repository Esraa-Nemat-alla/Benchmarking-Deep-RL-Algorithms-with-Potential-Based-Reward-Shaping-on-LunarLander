import argparse

# Centralized configurations
ENV_NAME = "LunarLanderContinuous-v3"
ALGORITHMS = ["ppo"] # Start with PPO, add others (A2C, SAC, TD3, DDPG) later
REWARD_CONFIGS = ["none", "distance", "angle", "combined"]
DEFAULT_TIMESTEPS = 1_000_000
DEFAULT_EVAL_FREQ = 10_000
GAMMA = 0.99

def get_train_args():
    parser = argparse.ArgumentParser(description="Train an RL agent on LunarLanderContinuous-v3")
    parser.add_argument("--algo", choices=ALGORITHMS, required=True, help="Algorithm to train")
    parser.add_argument("--reward", choices=REWARD_CONFIGS, required=True, help="Reward shaping configuration")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS, help="Total training timesteps")
    parser.add_argument("--eval-freq", type=int, default=DEFAULT_EVAL_FREQ, help="Evaluation frequency (in timesteps)")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of episodes per evaluation")
    parser.add_argument("--logdir", default="results", help="Directory to save results")
    return parser.parse_args()
