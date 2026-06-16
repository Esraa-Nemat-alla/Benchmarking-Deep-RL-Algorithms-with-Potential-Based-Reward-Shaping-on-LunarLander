import numpy as np
import gymnasium as gym

# We are using the continuous version of LunarLander for advanced algorithms (PPO, SAC, etc.)
ENV_NAME = "LunarLanderContinuous-v3"

# Get the observation space bounds to normalize our distance potential
_REF_ENV = gym.make(ENV_NAME)
_X_HIGH = float(_REF_ENV.observation_space.high[0])
_Y_HIGH = float(_REF_ENV.observation_space.high[1])
_REF_ENV.close()

# Maximum possible distance from the landing pad
D_MAX = float(np.sqrt(_X_HIGH ** 2 + _Y_HIGH ** 2))


def phi_distance(obs):
    """
    Potential function based on distance to the landing pad (0,0).
    It gives higher potential (closer to 0) as the agent gets closer to the target.
    """
    x, y = obs[0], obs[1]
    dist = np.sqrt(x ** 2 + y ** 2)
    return -min(1.0, dist / D_MAX)


def phi_angle(obs):
    """
    Potential function based on the angle of the lander.
    It gives higher potential (closer to 0) when the lander is perfectly upright (angle = 0).
    """
    theta = obs[4]
    return -abs(theta) / np.pi


def phi_combined(obs):
    """
    A weighted combination of the distance and angle potentials.
    """
    return 0.7 * phi_distance(obs) + 0.3 * phi_angle(obs)


# Map string names to the actual potential functions
POTENTIAL_FUNCTIONS = {
    "none": None,
    "distance": phi_distance,
    "angle": phi_angle,
    "combined": phi_combined,
}


class PBRSRewardWrapper(gym.Wrapper):
    """
    A Custom Gym Wrapper that applies Potential-Based Reward Shaping (PBRS).
    Formula: F(s, a, s') = gamma * Phi(s') - Phi(s)
    """

    def __init__(self, env, potential_fn, gamma=0.99, lam=1.0):
        super().__init__(env)
        self.potential_fn = potential_fn
        self.gamma = gamma
        self.lam = lam  # Scaling factor for the shaping reward
        self._prev_phi = 0.0

    def reset(self, **kwargs):
        # Reset the environment and calculate the initial potential
        obs, info = self.env.reset(**kwargs)
        self._prev_phi = self.potential_fn(obs)
        return obs, info

    def step(self, action):
        # Take a step in the environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # If the episode is over, the next potential must be 0
        next_phi = 0.0 if terminated else self.potential_fn(obs)

        # Calculate the shaping term F(s, a, s')
        shaping = self.gamma * next_phi - self._prev_phi
        
        # Add the shaping term to the original reward
        reward = reward + self.lam * shaping

        # Update previous potential for the next step
        self._prev_phi = next_phi
        return obs, reward, terminated, truncated, info


def make_lunarlander_env(reward_config, gamma=0.99):
    """
    Helper function to create the environment and wrap it with PBRS if needed.
    """
    env = gym.make(ENV_NAME)
    potential_fn = POTENTIAL_FUNCTIONS[reward_config]
    
    # Only wrap the environment if a reward shaping configuration is chosen
    if potential_fn is not None:
        env = PBRSRewardWrapper(env, potential_fn, gamma=gamma, lam=1.0)
    return env
