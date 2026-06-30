# Final Report: Benchmarking Deep RL Algorithms with Potential-Based Reward Shaping

## 1. Objective

This project evaluates whether Potential-Based Reward Shaping (PBRS) can improve deep reinforcement learning performance on `LunarLanderContinuous-v3`. The original proposal focused on comparing a smaller set of Stable-Baselines3 algorithms. After feedback that the scope was too limited, the project was expanded to include five algorithms, five reward configurations, three random seeds, and a hyperparameter sensitivity study.

## 2. Environment

The environment is `LunarLanderContinuous-v3` from Gymnasium. This version is more challenging than the discrete LunarLander environment because the agent must learn continuous control actions for the lander's engines.

Evaluation is always performed on the native, unshaped environment reward. This keeps shaped and unshaped agents comparable.

## 3. Algorithms

The benchmark includes five algorithms:

| Algorithm | Type |
|---|---|
| PPO | On-policy |
| A2C | On-policy |
| SAC | Off-policy |
| TD3 | Off-policy |
| DDPG | Off-policy |

This gives a broader comparison across policy-gradient, actor-critic, entropy-regularized, and deterministic control methods.

## 4. Reward Shaping Design

The project uses Potential-Based Reward Shaping:

```text
F(s, a, s') = gamma * Phi(s') - Phi(s)
```

The tested reward configurations are:

| Config | Description |
|---|---|
| `none` | Vanilla environment reward |
| `distance` | Potential based on distance from the landing pad |
| `angle` | Potential based on upright orientation |
| `combined` | Weighted distance and angle potential |
| `velocity` | Potential based on lander speed |

The `combined` potential uses:

```text
0.7 * phi_distance + 0.3 * phi_angle
```

## 5. Experiment Grid

Main benchmark:

```text
75 of 75 default-grid runs complete
PPO, A2C, SAC, TD3, and DDPG: 15/15 each
```

Hyperparameter sensitivity study:

```text
12 of 12 runs complete
```

Total completed runs:

```text
87 of 87 planned
```

The main benchmark used seeds `0`, `1`, and `2` for every algorithm and reward configuration. Most runs were trained for 50k timesteps; `sac_none_seed0` was trained for 20k timesteps.

## 6. Metrics

The analysis reports:

- Mean final evaluation reward.
- Standard deviation across seeds.
- Success rate using the LunarLander solved threshold of 200.
- Timesteps to reach 200 reward, when achieved.
- Area under the learning curve.

## 7. Key Results

Best configurations by mean final evaluation reward:

| Algorithm | Best reward config | Mean final reward | Success rate | Seeds |
|---|---|---:|---:|---:|
| PPO | `combined` | 110.9 | 43% | 3 |
| SAC | `combined` | 31.2 | 23% | 3 |
| TD3 | `combined` | -36.7 | 3% | 3 |
| DDPG | `distance` | -127.8 | 0% | 3 |
| A2C | `combined` | -214.1 | 0% | 3 |

Overall findings:

- Best mean final reward: **PPO with `combined` shaping**.
- Highest success rate: **PPO with `combined` shaping**, at 43%.
- Largest final-reward shaping gain: **PPO with `combined`**, improving over the PPO baseline by 209.6 final-reward points.
- `combined` shaping was the best reward configuration for PPO, A2C, SAC, and TD3. DDPG performed best with `distance` shaping.

## 8. Discussion

The results suggest that PBRS can improve reward performance, especially when distance and angle shaping are combined. PPO with `combined` shaping achieved the highest mean final reward among all configurations.

The strongest success rate also came from PPO with `combined` shaping. However, the results still show algorithm-specific behavior: some shaping variants improved final reward but not success rate, and several off-policy runs remained unstable under the short training budget.

A2C remained weak overall, but shaping variants improved its final reward compared with the unshaped baseline. SAC and TD3 also benefited most from `combined` shaping by final reward, while DDPG performed best with the simpler `distance` potential.

## 9. Hyperparameter Sensitivity

The project also includes a 12-run hyperparameter sensitivity study for PPO with `combined` shaping. It tests:

- Learning rates: `3e-4`, `1e-4`
- Network architectures: `[64, 64]`, `[256, 256]`
- Seeds: `0`, `1`, `2`

The generated sensitivity plot is available at:

```text
reports/figures/hyperparam_sensitivity.png
```

## 10. Limitations

- The completed runs are short-budget pilot experiments, mostly at 50k timesteps.
- `sac_none_seed0` was trained for 20k timesteps, so SAC baseline comparisons should mention this training-budget difference.
- The hyperparameter sensitivity study is complete, but it only covers PPO with combined shaping.
- GIF replay reward is from a single episode and can differ from the mean evaluation reward.
- Longer training budgets would give a stronger estimate of final algorithm performance.

## 11. Future Work

Recommended next steps:

1. Run the strongest configurations at 100k, 500k, or 1M timesteps with separate run names.
2. Add more potential functions, such as leg-contact shaping.
3. Evaluate on an additional continuous-control environment.
4. Add statistical significance tests once longer runs are available.
5. Compare PBRS against non-potential reward shaping to show why PBRS is theoretically safer.

## 12. Reproducibility

Create the environment:

```bash
conda env create -f environment-linux.yml
conda activate lunarlander-pbrs
```

On Windows:

```powershell
conda env create -f environment-windows.yml
conda activate lunarlander-pbrs
```

Regenerate results artifacts:

```bash
python evaluate.py
python analyze_results.py
```

Launch the dashboard:

```bash
streamlit run app.py
```

Important generated outputs:

```text
reports/tables/results_summary.csv
reports/tables/best_configs.csv
reports/tables/reward_shaping_effects.csv
reports/figures/comparative_bar_chart.png
reports/figures/hyperparam_sensitivity.png
reports/RESULTS_INTERPRETATION.md
```
