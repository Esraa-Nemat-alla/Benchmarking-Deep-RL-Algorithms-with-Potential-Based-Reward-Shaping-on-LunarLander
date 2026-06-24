# Project Scope Update After Proposal Feedback

## Feedback Summary

The feedback said the reward shaping idea is strong, but the original scope was too small because it used only three Stable-Baselines3 algorithms on a standard Gymnasium environment. The suggested improvements were:

- Add more algorithms and more experimental settings.
- Or use a more complex environment, such as Atari.

## Chosen Revision

We chose to expand the algorithm and experiment grid while keeping the LunarLander reward-shaping research question. This is the most direct way to preserve the original project idea and answer the feedback.

## Updated Experimental Scope

The revised project now uses:

- **5 algorithms:** PPO, A2C, SAC, TD3, DDPG.
- **4 reward configurations:** no shaping, distance shaping, angle shaping, combined shaping.
- **3 random seeds:** 0, 1, 2.
- **60 default benchmark runs:** 5 algorithms x 4 reward configurations x 3 seeds.
- **12 hyperparameter sensitivity runs:** 2 learning rates x 2 network architectures x 3 seeds.
- **72 total planned runs.**

## Why This Addresses the Feedback

The new design increases scope in two ways:

1. It broadens the algorithm comparison from three algorithms to five, covering both on-policy and off-policy methods.
2. It adds a hyperparameter sensitivity study, which creates extra experimental settings and tests whether potential-based reward shaping is robust.

The environment is also `LunarLanderContinuous-v3`, not the simpler discrete-action LunarLander. This keeps the project manageable while still making the task more challenging than the most common LunarLander examples.

## Implementation Status

Implemented in code:

- `config.py` defines the five-algorithm benchmark and hyperparameter grid.
- `train.py` supports PPO, A2C, SAC, TD3, and DDPG.
- `run_all_experiments.py` runs both the full benchmark grid and hyperparameter study.
- `evaluate.py` aggregates results across all algorithms and reward settings.
- `app.py` exposes the expanded scope through the Streamlit dashboard.
- `demo.py` can replay saved models from all five algorithms.

## Recommended Submission Response

We updated the project scope based on the feedback by expanding the benchmark from three algorithms to five algorithms and adding a hyperparameter sensitivity study. The revised project evaluates PPO, A2C, SAC, TD3, and DDPG on `LunarLanderContinuous-v3` under four reward-shaping configurations and three random seeds, for 60 default benchmark runs. We also added 12 additional hyperparameter runs to test robustness across learning rates and network sizes. This preserves the original potential-based reward shaping contribution while increasing the experimental depth expected for the course project.
