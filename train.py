"""
Train one DRL algorithm on LunarLander-v3 under one reward configuration.

Example:
    python train.py --algo dqn --reward distance --seed 0 --timesteps 100000

Saves, under results/<algo>_<reward>_seed<seed>/:
    - monitor.csv          episode returns/lengths during training
    - evaluations.npz      periodic evaluation results (from EvalCallback)
    - best_model.zip       best model seen during training
    - final_model.zip      model at the end of training
"""

import argparse
import os

from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from reward_shaping import POTENTIAL_FUNCTIONS, make_lunarlander_env

ALGORITHMS = {"dqn": DQN, "ppo": PPO, "a2c": A2C}


GAMMA = 0.99


def build_env(reward_config, seed, log_path):
    env = make_lunarlander_env(reward_config, gamma=GAMMA)
    env = Monitor(env, filename=log_path)
    env.reset(seed=seed)
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algo", choices=ALGORITHMS.keys(), required=True)
    parser.add_argument("--reward", choices=POTENTIAL_FUNCTIONS.keys(), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                         help="Total training timesteps (proposal uses 1,000,000).")
    parser.add_argument("--eval-freq", type=int, default=10_000,
                         help="Run an evaluation every N training timesteps.")
    parser.add_argument("--eval-episodes", type=int, default=10,
                         help="Number of episodes per evaluation.")
    parser.add_argument("--logdir", default="results")
    args = parser.parse_args()

    run_name = f"{args.algo}_{args.reward}_seed{args.seed}"
    run_dir = os.path.join(args.logdir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    train_env = build_env(args.reward, args.seed, os.path.join(run_dir, "monitor"))

    eval_env = build_env(args.reward, args.seed + 10_000, os.path.join(run_dir, "eval_monitor"))

    model_cls = ALGORITHMS[args.algo]
    model = model_cls("MlpPolicy", train_env, gamma=GAMMA, seed=args.seed, verbose=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
    )

    model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    model.save(os.path.join(run_dir, "final_model"))

    print(f"\nFinished {run_name}. Results saved to {run_dir}/")


if __name__ == "__main__":
    main()
