# Results Interpretation

## Current Experiment Coverage

- Completed summary rows: 8.
- Algorithms represented: a2c, ddpg, ppo, sac, td3.
- Reward configs represented: combined, distance, none, velocity.
- Default-grid coverage: 75/75 runs complete.
- Hyperparameter-study coverage: 12/12 runs complete.
- Current runs are mostly 50k-timestep pilot runs, so conclusions should be treated as preliminary until longer training is added.

## Best Current Configurations

| Algorithm | Best reward config | Final reward | Success rate | Seeds |
|---|---|---:|---:|---:|
| A2C | velocity | -338.1 | 0% | 3 |
| DDPG | combined | 123.5 | 55% | 2 |
| PPO | none | -16.9 | 5% | 2 |
| SAC | velocity | -2.0 | 30% | 3 |
| TD3 | velocity | -142.5 | 0% | 3 |

## Main Observations

- Best final reward so far: DDPG with `combined` (123.5).
- Highest success rate so far: DDPG with `combined` (55%).
- Largest shaping improvement over baseline so far: PPO with `velocity` (delta final reward = -118.7).

## Reward Shaping Effect

Positive delta means the shaped version outperformed the no-shaping baseline for that algorithm.

| Algorithm | Shaping | Delta final reward | Delta success rate | Seeds |
|---|---|---:|---:|---:|
| PPO | velocity | -118.7 | -2% | 2 |
| PPO | distance | -149.1 | -5% | 1 |

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
