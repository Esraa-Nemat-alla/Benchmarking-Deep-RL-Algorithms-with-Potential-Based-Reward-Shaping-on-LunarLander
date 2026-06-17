"""
Train a single RL algorithm on LunarLanderContinuous-v3 with optional PBRS.

Supports 5 algorithms from Stable-Baselines3:
  - PPO  (on-policy,  Proximal Policy Optimization)
  - A2C  (on-policy,  Advantage Actor-Critic)
  - SAC  (off-policy, Soft Actor-Critic)
  - TD3  (off-policy, Twin Delayed DDPG)
  - DDPG (off-policy, Deep Deterministic Policy Gradient)

Usage examples:
  python train.py --algo ppo  --reward distance --seed 0 --timesteps 50000
  python train.py --algo sac  --reward combined --seed 1
  python train.py --algo td3  --reward none     --seed 2 --lr 1e-4 --net-arch 256 256
"""
import os
import argparse

from stable_baselines3 import PPO, A2C, SAC, TD3, DDPG
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from config import GAMMA, REWARD_CONFIGS, RESULTS_DIR
from reward_shaping import make_lunarlander_env

# ── Map CLI names to SB3 classes ─────────────────────────────────────
ALGORITHMS_CLS = {
    "ppo":  PPO,
    "a2c":  A2C,
    "sac":  SAC,
    "td3":  TD3,
    "ddpg": DDPG,
}


def build_env(reward_config, seed, log_path, for_eval=False):
    """
    Create a LunarLanderContinuous environment.

    When for_eval=True, we always use the unshaped reward so that the
    evaluation metrics (success rate, AUC) are measured on the real
    environment reward — not the shaped one.
    """
    config = "none" if for_eval else reward_config
    env = make_lunarlander_env(config, gamma=GAMMA)
    env = Monitor(env, filename=log_path)
    env.reset(seed=seed)
    return env


def _build_run_name(algo, reward, seed, lr=None, net_arch=None):
    """
    Create a unique folder name for this experiment run.
    If custom hyperparams are provided, we encode them in the name
    so they don't overwrite the default runs.
    """
    name = f"{algo}_{reward}_seed{seed}"
    if lr is not None:
        name += f"_lr{lr}"
    if net_arch is not None:
        arch_str = "-".join(str(n) for n in net_arch)
        name += f"_net{arch_str}"
    return name


def train(algo, reward, seed, timesteps, eval_freq, lr=None, net_arch=None):
    """
    Run one training job. Returns the output directory path.

    Parameters
    ----------
    algo : str
        Algorithm name (ppo, a2c, sac, td3, ddpg).
    reward : str
        Reward shaping config (none, distance, angle, combined).
    seed : int
        Random seed for reproducibility.
    timesteps : int
        Total training timesteps.
    eval_freq : int
        How often (in timesteps) to evaluate the agent.
    lr : float or None
        Custom learning rate. Uses SB3 default if None.
    net_arch : list[int] or None
        Custom network architecture, e.g. [256, 256]. Uses SB3 default if None.
    """
    run_name = _build_run_name(algo, reward, seed, lr, net_arch)
    run_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Training env uses shaped reward; eval env always uses native reward
    train_env = build_env(reward, seed, os.path.join(run_dir, "monitor"))
    eval_env = build_env(reward, seed + 10_000,
                         os.path.join(run_dir, "eval_monitor"), for_eval=True)

    # ── Build model kwargs ───────────────────────────────────────────
    model_cls = ALGORITHMS_CLS[algo]
    model_kwargs = {
        "policy": "MlpPolicy",
        "env": train_env,
        "gamma": GAMMA,
        "seed": seed,
        "verbose": 1,
    }

    # Override learning rate if specified
    if lr is not None:
        model_kwargs["learning_rate"] = lr

    # Override network architecture if specified
    if net_arch is not None:
        model_kwargs["policy_kwargs"] = {"net_arch": list(net_arch)}

    model = model_cls(**model_kwargs)

    # ── Evaluation callback (tracks progress during training) ────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=run_dir,
        log_path=run_dir,
        eval_freq=eval_freq,
        n_eval_episodes=10,
        deterministic=True,
    )

    print(f"--- Starting training: {run_name} ---")
    model.learn(total_timesteps=timesteps, callback=eval_callback)
    model.save(os.path.join(run_dir, "final_model"))
    print(f"Finished {run_name}. Results saved to {run_dir}/")
    return run_dir


def main():
    parser = argparse.ArgumentParser(
        description="Train an RL agent on LunarLanderContinuous-v3 with optional PBRS."
    )
    parser.add_argument("--algo", choices=list(ALGORITHMS_CLS.keys()), default="ppo",
                        help="Which RL algorithm to use (default: ppo)")
    parser.add_argument("--reward", choices=REWARD_CONFIGS, required=True,
                        help="Reward shaping configuration")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Total training timesteps (default: 1M)")
    parser.add_argument("--eval-freq", type=int, default=10_000,
                        help="Evaluate every N steps (default: 10k)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Custom learning rate (optional, uses SB3 default)")
    parser.add_argument("--net-arch", type=int, nargs="+", default=None,
                        help="Custom network architecture, e.g. --net-arch 256 256")
    args = parser.parse_args()

    train(args.algo, args.reward, args.seed, args.timesteps, args.eval_freq,
          lr=args.lr, net_arch=args.net_arch)


if __name__ == "__main__":
    main()
