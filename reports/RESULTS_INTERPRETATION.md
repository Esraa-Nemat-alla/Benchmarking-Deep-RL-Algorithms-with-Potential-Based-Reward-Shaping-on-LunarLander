# Results Interpretation

## Current Experiment Coverage

- Completed summary rows: 20.
- Algorithms represented: a2c, ddpg, ppo, sac, td3.
- Reward configs represented: angle, combined, distance, none.
- Current runs are mostly pilot-scale, so conclusions should be treated as preliminary until more seeds and longer training are added.

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
- Highest success rate so far: PPO with `none` (50%).
- Largest shaping improvement over baseline so far: A2C with `combined` (delta final reward = 140.1).

## Reward Shaping Effect

Positive delta means the shaped version outperformed the no-shaping baseline for that algorithm.

| Algorithm | Shaping | Delta final reward | Delta success rate | Seeds |
|---|---|---:|---:|---:|
| A2C | combined | 140.1 | 0% | 3 |
| A2C | angle | 131.3 | 0% | 3 |
| A2C | distance | 126.4 | 0% | 3 |
| PPO | combined | 64.7 | -7% | 3 |
| TD3 | combined | 43.1 | 3% | 3 |
| SAC | combined | 13.2 | 7% | 3 |
| DDPG | distance | 10.9 | -3% | 3 |
| TD3 | distance | 2.8 | 0% | 3 |
| DDPG | combined | -1.7 | -3% | 3 |
| SAC | angle | -16.7 | 3% | 3 |
| TD3 | angle | -28.2 | 0% | 3 |
| SAC | distance | -45.6 | -3% | 3 |
| DDPG | angle | -79.6 | -3% | 3 |
| PPO | angle | -108.4 | -43% | 3 |
| PPO | distance | -166.0 | -50% | 3 |

## Recommended Next Experiments

1. Add seeds 1 and 2 for the strongest configurations, especially TD3 combined, SAC distance, and PPO none.
2. Increase timesteps for the strongest candidates from 50k to 100k or 500k.
3. Run the planned hyperparameter sensitivity study after the main comparison stabilizes.
4. Keep evaluation on the unshaped reward so the shaped and unshaped agents remain comparable.

## Limitations

- Many rows currently have one seed, so variance estimates are not reliable yet.
- Short training can make off-policy methods look unstable or undertrained.
- GIF replay reward can differ from mean evaluation reward because it is one sampled episode.
