# Results Interpretation

## Current Experiment Coverage

- Completed summary rows: 25.
- Algorithms represented: a2c, ddpg, ppo, sac, td3.
- Reward configs represented: angle, combined, distance, none, velocity.
- Default-grid coverage: 75/75 runs complete.
- Hyperparameter-study coverage: 12/12 runs complete.
- Current runs are mostly 50k-timestep pilot runs, so conclusions should be treated as preliminary until longer training is added.

## Best Current Configurations

| Algorithm | Best reward config | Final reward | Success rate | Seeds |
|---|---|---:|---:|---:|
| A2C | combined | -214.1 | 0% | 3 |
| DDPG | distance | -127.8 | 0% | 3 |
| PPO | combined | 110.9 | 43% | 3 |
| SAC | combined | 31.2 | 23% | 3 |
| TD3 | combined | -36.7 | 3% | 3 |

## Main Observations

- Best final reward so far: PPO with `combined` (110.9).
- Highest success rate so far: PPO with `combined` (43%).
- Largest shaping improvement over baseline so far: PPO with `combined` (delta final reward = 209.6).

## Reward Shaping Effect

Positive delta means the shaped version outperformed the no-shaping baseline for that algorithm.

| Algorithm | Shaping | Delta final reward | Delta success rate | Seeds |
|---|---|---:|---:|---:|
| PPO | combined | 209.6 | 40% | 3 |
| A2C | combined | 140.1 | 0% | 3 |
| A2C | angle | 131.3 | 0% | 3 |
| A2C | distance | 126.4 | 0% | 3 |
| TD3 | combined | 43.1 | 3% | 3 |
| PPO | angle | 36.5 | 3% | 3 |
| A2C | velocity | 16.1 | 0% | 3 |
| SAC | combined | 13.2 | 7% | 3 |
| DDPG | distance | 10.9 | -3% | 3 |
| TD3 | distance | 2.8 | 0% | 3 |
| DDPG | combined | -1.7 | -3% | 3 |
| SAC | angle | -16.7 | 3% | 3 |
| SAC | velocity | -20.0 | 13% | 3 |
| TD3 | angle | -28.2 | 0% | 3 |
| PPO | velocity | -37.0 | 0% | 3 |
| SAC | distance | -45.6 | -3% | 3 |
| TD3 | velocity | -62.7 | 0% | 3 |
| PPO | distance | -65.1 | -3% | 3 |
| DDPG | angle | -79.6 | -3% | 3 |
| DDPG | velocity | -84.5 | -3% | 3 |

## Recommended Next Experiments

1. Increase timesteps for the strongest candidates from 50k to 100k, 500k, or 1M.
2. Re-run `evaluate.py` and `analyze_results.py` after adding new runs.
3. Keep evaluation on the unshaped reward so the shaped and unshaped agents remain comparable.
4. Add statistical significance tests once longer runs are available.

## Limitations

- The default grid is complete, but most runs are short-budget pilot experiments.
- `sac_none_seed0` is a 20k-timestep run, so SAC baseline comparisons should mention this difference.
- Short training can make off-policy methods look unstable or undertrained.
- GIF replay reward can differ from mean evaluation reward because it is one sampled episode.
