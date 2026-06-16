"""
Train one DRL algorithm on LunarLanderContinuous-v3 under one reward configuration.

Example:
    python -m src.train --algo sac --reward distance --seed 0 --timesteps 100000
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.envs.reward_shaping import make_lunarlander_env
from src.config import get_train_args, GAMMA

ALGORITHMS_CLS = {
    "ppo": PPO,
}

def build_env(reward_config, seed, log_path):
    env = make_lunarlander_env(reward_config, gamma=GAMMA)
    env = Monitor(env, filename=log_path)
    env.reset(seed=seed)
    return env

def main():
    args = get_train_args()
    run_name = f"{args.algo}_{args.reward}_seed{args.seed}"
    run_dir = os.path.join(args.logdir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    train_env = build_env(args.reward, args.seed, os.path.join(run_dir, "monitor"))
    eval_env = build_env(args.reward, args.seed + 10_000, os.path.join(run_dir, "eval_monitor"))

    model_cls = ALGORITHMS_CLS[args.algo]
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
