import numpy as np
import gymnasium as gym
from src.config import ENV_NAME

_REF_ENV = gym.make(ENV_NAME)
_X_HIGH = float(_REF_ENV.observation_space.high[0])
_Y_HIGH = float(_REF_ENV.observation_space.high[1])
_REF_ENV.close()

D_MAX = float(np.sqrt(_X_HIGH ** 2 + _Y_HIGH ** 2))

def phi_distance(obs):
    """Potential that increases (toward 0) as the lander nears the pad at the origin."""
    x, y = obs[0], obs[1]
    dist = np.sqrt(x ** 2 + y ** 2)
    return -min(1.0, dist / D_MAX)

def phi_angle(obs):
    """Potential that increases (toward 0) as the lander becomes upright."""
    theta = obs[4]
    return -abs(theta) / np.pi

def phi_combined(obs):
    """Weighted combination of distance and angle potentials (Eq. 6)."""
    return 0.7 * phi_distance(obs) + 0.3 * phi_angle(obs)

POTENTIAL_FUNCTIONS = {
    "none": None,
    "distance": phi_distance,
    "angle": phi_angle,
    "combined": phi_combined,
}

class PBRSRewardWrapper(gym.Wrapper):
    """Adds a potential-based shaping bonus to the environment's reward."""

    def __init__(self, env, potential_fn, gamma=0.99, lam=1.0):
        super().__init__(env)
        self.potential_fn = potential_fn
        self.gamma = gamma
        self.lam = lam
        self._prev_phi = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_phi = self.potential_fn(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        next_phi = 0.0 if terminated else self.potential_fn(obs)

        shaping = self.gamma * next_phi - self._prev_phi
        reward = reward + self.lam * shaping

        self._prev_phi = next_phi
        return obs, reward, terminated, truncated, info


def make_lunarlander_env(reward_config, gamma=0.99):
    """Create a LunarLanderContinuous-v3 environment, optionally wrapped with PBRS."""
    env = gym.make(ENV_NAME)
    potential_fn = POTENTIAL_FUNCTIONS[reward_config]
    if potential_fn is not None:
        env = PBRSRewardWrapper(env, potential_fn, gamma=gamma, lam=1.0)
    return env
