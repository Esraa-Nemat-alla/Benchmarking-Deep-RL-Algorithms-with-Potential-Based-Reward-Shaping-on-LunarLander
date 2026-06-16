"""Record trained agents on the unshaped LunarLander environment."""
import os

import gymnasium as gym
from stable_baselines3 import PPO

from config import RESULTS_DIR
from reward_shaping import ENV_NAME


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
    """
    run_dir = os.path.join(RESULTS_DIR, run_name)
    model_path = _find_model_path(run_dir, prefer_best=prefer_best)
    if model_path is None:
        raise FileNotFoundError(f"No saved model found in {run_dir}")

    env = gym.make(ENV_NAME, render_mode="rgb_array")
    model = PPO.load(model_path)
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
