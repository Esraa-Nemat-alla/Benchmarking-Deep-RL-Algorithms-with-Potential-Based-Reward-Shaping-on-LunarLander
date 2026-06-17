"""
Record trained agents on the unshaped LunarLander environment.

Used by the Streamlit "Watch Agent" tab to replay a saved model as a GIF.
Supports all 5 algorithms by auto-detecting the class from the folder name.
"""
import os

import gymnasium as gym
from stable_baselines3 import PPO, A2C, SAC, TD3, DDPG

from config import RESULTS_DIR
from reward_shaping import ENV_NAME

# ── Map algo names to SB3 classes (same as train.py) ─────────────────
_ALGO_CLS = {
    "ppo":  PPO,
    "a2c":  A2C,
    "sac":  SAC,
    "td3":  TD3,
    "ddpg": DDPG,
}


def _detect_algo_class(run_name):
    """
    Figure out which SB3 class to use based on the run folder name.

    Folder names follow the pattern: <algo>_<reward>_seed<N>[_lr...][_net...]
    So the algorithm is always the first token before the first underscore.
    """
    algo_key = run_name.split("_")[0].lower()
    if algo_key in _ALGO_CLS:
        return _ALGO_CLS[algo_key]
    # Fallback to PPO if we can't detect (shouldn't happen)
    return PPO


def list_runs():
    """Return sorted run folder names that contain a saved model."""
    if not os.path.isdir(RESULTS_DIR):
        return []

    runs = []
    for name in os.listdir(RESULTS_DIR):
        run_dir = os.path.join(RESULTS_DIR, name)
        if not os.path.isdir(run_dir):
            continue
        if _find_model_path(run_dir):
            runs.append(name)
    return sorted(runs)


def _find_model_path(run_dir, prefer_best=True):
    """Look for a saved model file in the run directory."""
    candidates = []
    if prefer_best:
        candidates.append(os.path.join(run_dir, "best_model.zip"))
    candidates.extend([
        os.path.join(run_dir, "final_model.zip"),
        os.path.join(run_dir, "best_model"),
        os.path.join(run_dir, "final_model"),
    ])
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def record_episode(run_name, prefer_best=True, seed=0):
    """
    Run one deterministic episode and return RGB frames plus total reward.

    Uses the native (unshaped) environment so reward matches the 200 threshold.
    Auto-detects the correct algorithm class from the run folder name.
    """
    run_dir = os.path.join(RESULTS_DIR, run_name)
    model_path = _find_model_path(run_dir, prefer_best=prefer_best)
    if model_path is None:
        raise FileNotFoundError(f"No saved model found in {run_dir}")

    # Auto-detect which SB3 class to load
    algo_cls = _detect_algo_class(run_name)

    env = gym.make(ENV_NAME, render_mode="rgb_array")
    model = algo_cls.load(model_path)
    obs, _ = env.reset(seed=seed)
    frames = [env.render()]
    total_reward = 0.0
    terminated = truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        total_reward += float(reward)

    env.close()
    return frames, total_reward


def frame_to_gif_bytes(frames, duration_ms=50):
    """Convert RGB frames to GIF bytes for Streamlit."""
    from PIL import Image
    import io

    images = [Image.fromarray(frame) for frame in frames]
    buffer = io.BytesIO()
    images[0].save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    buffer.seek(0)
    return buffer.getvalue()
