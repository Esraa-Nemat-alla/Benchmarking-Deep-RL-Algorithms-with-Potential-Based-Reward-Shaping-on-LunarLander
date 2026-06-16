"""
Train a specific RL algorithm (e.g., PPO) on LunarLanderContinuous-v3.
"""
import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from config import GAMMA, REWARD_CONFIGS, RESULTS_DIR
from reward_shaping import make_lunarlander_env

ALGORITHMS_CLS = {
    "ppo": PPO,
}


def build_env(reward_config, seed, log_path, for_eval=False):
    """Create env; eval always uses unshaped reward for fair metrics."""
    config = "none" if for_eval else reward_config
    env = make_lunarlander_env(config, gamma=GAMMA)
    env = Monitor(env, filename=log_path)
    env.reset(seed=seed)
    return env


def train(algo, reward, seed, timesteps, eval_freq):
    """Run one training job. Returns the output directory path."""
    run_name = f"{algo}_{reward}_seed{seed}"
    run_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    train_env = build_env(reward, seed, os.path.join(run_dir, "monitor"))
    eval_env = build_env(reward, seed + 10_000, os.path.join(run_dir, "eval_monitor"), for_eval=True)

    model_cls = ALGORITHMS_CLS[algo]
    model = model_cls("MlpPolicy", train_env, gamma=GAMMA, seed=seed, verbose=1)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=eval_freq,
        n_eval_episodes=10,
        deterministic=True,
    )

    print(f"--- Starting training for {run_name} ---")
    model.learn(total_timesteps=timesteps, callback=eval_callback)
    model.save(os.path.join(run_dir, "final_model"))
    print(f"Finished {run_name}. Results saved to {run_dir}/")
    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=list(ALGORITHMS_CLS.keys()), default="ppo")
    parser.add_argument("--reward", choices=REWARD_CONFIGS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=1_000_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    args = parser.parse_args()

    train(args.algo, args.reward, args.seed, args.timesteps, args.eval_freq)


if __name__ == "__main__":
    main()
