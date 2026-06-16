"""
Train a specific RL algorithm (e.g., PPO) on LunarLanderContinuous-v3.
"""
import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

# Import our custom environment creator from the local file
from reward_shaping import make_lunarlander_env

# --- Configuration ---
# Currently we are only using PPO to keep things simple. 
# You can add SAC, TD3, etc., here later.
ALGORITHMS_CLS = {
    "ppo": PPO,
}
GAMMA = 0.99

def build_env(reward_config, seed, log_path):
    """Creates the environment and wraps it in a Monitor for logging."""
    env = make_lunarlander_env(reward_config, gamma=GAMMA)
    env = Monitor(env, filename=log_path)
    env.reset(seed=seed)
    return env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=list(ALGORITHMS_CLS.keys()), default="ppo", help="Algorithm to train")
    parser.add_argument("--reward", choices=["none", "distance", "angle", "combined"], required=True, help="Reward shaping config")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--timesteps", type=int, default=1_000_000, help="Total training steps")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="Evaluation frequency")
    args = parser.parse_args()

    # Create a unique folder for this run's results
    run_name = f"{args.algo}_{args.reward}_seed{args.seed}"
    run_dir = os.path.join("results", run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Build training and evaluation environments
    train_env = build_env(args.reward, args.seed, os.path.join(run_dir, "monitor"))
    eval_env = build_env(args.reward, args.seed + 10_000, os.path.join(run_dir, "eval_monitor"))

    # Initialize the model (e.g., PPO)
    model_cls = ALGORITHMS_CLS[args.algo]
    model = model_cls("MlpPolicy", train_env, gamma=GAMMA, seed=args.seed, verbose=1)

    # Callback to periodically evaluate the model and save the best one
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=10,
        deterministic=True,
    )

    print(f"--- Starting training for {run_name} ---")
    model.learn(total_timesteps=args.timesteps, callback=eval_callback)
    
    # Save the final model at the very end
    model.save(os.path.join(run_dir, "final_model"))
    print(f"Finished {run_name}. Results saved to {run_dir}/")

if __name__ == "__main__":
    main()
