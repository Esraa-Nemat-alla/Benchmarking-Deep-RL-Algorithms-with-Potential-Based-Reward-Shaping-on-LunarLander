# Benchmarking DQN, PPO, and A2C with PBRS on LunarLander

Code for the proposal "Benchmarking Deep Reinforcement Learning Algorithms
with Potential-Based Reward Shaping on LunarLander".

It trains DQN, PPO, and A2C (from Stable-Baselines3) on the discrete
`LunarLander-v3` environment under 4 reward configurations - none,
distance potential, angle potential, and combined - and evaluates
sample efficiency and final performance.

## 1. Setup

```bash
python -m venv venv
source venv/bin/activate          
pip install -r requirements.txt
```

Quick sanity check:

```bash
python -c "import gymnasium as gym; gym.make('LunarLander-v3')"
```

If this fails with a Box2D error, your `pip install` of `gymnasium[box2d]`
didn't pick up Box2D - re-run `pip install "gymnasium[box2d]"` (you may
need `swig` installed on your system first: `apt install swig` /
`brew install swig`).

## 2. Files

| File | Purpose |
|---|---|
| `reward_shaping.py` | PBRS wrapper + the 3 potential functions (Eqs. 4-6) |
| `train.py` | Trains one (algorithm, reward config, seed) combination |
| `run_all_experiments.py` | Loops `train.py` over the full 3x4xN-seed grid |
| `evaluate.py` | Aggregates results into a summary table + learning-curve plots |

## 3. Running a single experiment

```bash
python train.py --algo dqn --reward distance --seed 0 --timesteps 1000000
```

- `--algo`: `dqn`, `ppo`, or `a2c`
- `--reward`: `none`, `distance`, `angle`, or `combined`
- `--seed`: random seed
- `--timesteps`: total training steps (proposal uses 1,000,000)
- `--eval-freq`: how often (in timesteps) to run an evaluation (default 10,000)
- `--eval-episodes`: episodes per evaluation (default 10)

This creates `results/dqn_distance_seed0/` containing:
- `monitor.monitor.csv` - training episode returns/lengths
- `evaluations.npz` - evaluation reward over training (used by `evaluate.py`)
- `best_model.zip` - best checkpoint seen during training
- `final_model.zip` - model at the end of training

**Start small first.** Try `--timesteps 50000` to confirm everything runs
on your machine in a few minutes before committing to the full
1,000,000-step runs.

## 4. Running the full grid

`run_all_experiments.py` loops over all combinations. Open it and adjust
the constants at the top:

```python
ALGORITHMS = ["dqn", "ppo", "a2c"]
REWARD_CONFIGS = ["none", "distance", "angle", "combined"]
SEEDS = [0, 1, 2]         
TIMESTEPS = 1_000_000
```

Then:

```bash
python run_all_experiments.py
```

**Compute warning:** the full grid is 3 algos x 4 configs x N seeds runs of
1,000,000 timesteps each (e.g. 36 runs for 3 seeds). On a laptop CPU, PPO/A2C
take roughly 15-30 minutes per million steps and DQN considerably longer
(its replay buffer + per-step gradient updates make it the slowest of the
three). Budget accordingly, or reduce `TIMESTEPS` / `SEEDS` for a first pass,
then scale up for the final report. Each run is an independent process, so
you can also split the grid across machines manually by running `train.py`
directly with different arguments.

## 5. Evaluating results

Once some runs have finished:

```bash
python evaluate.py
```

This produces:
- `results_summary.csv` - for each (algorithm, reward config): mean/std of
  final evaluation reward, success rate (fraction of final eval episodes
  with return >= 200), timesteps to first reach a mean reward of 200, and
  area under the learning curve (AUC).
- `<algo>_learning_curves.png` - one plot per algorithm, showing mean +/- std
  evaluation reward over training for each of the 4 reward configurations.

`evaluate.py` only includes combinations for which `results/.../evaluations.npz`
exists, so you can run it at any point to check progress - it doesn't need
the whole grid to be finished.

## 6. Design choices / how this maps to the proposal

- **Environment**: `LunarLander-v3` (Gymnasium), discrete action space
  `{0,1,2,3}`, default no-wind setting - matches the proposal's state and
  action space description.
- **Shaping formula** (`reward_shaping.py`): implements
  `r' = r + lambda * (gamma * Phi(s') - Phi(s))` with `lambda = 1.0` and
  `Phi(s_terminal) = 0`, as in Eq. 3. `gamma = 0.99` is used both for the
  shaping term and as the agent's discount factor (`GAMMA` in `train.py`),
  so they're consistent.
- **Potentials**:
  - `phi_distance`: `-min(1, sqrt(x^2+y^2) / D_MAX)`, where `D_MAX` is
    computed from the environment's observation-space bounds for x and y
    (`sqrt(x_high^2 + y_high^2)`) - a fixed, stationary scale as required.
  - `phi_angle`: `-|theta| / pi`.
  - `phi_combined`: `0.7 * phi_distance + 0.3 * phi_angle` (Eq. 6).
- **Algorithms**: SB3 `DQN`, `PPO`, `A2C` with default `MlpPolicy` and
  default hyperparameters (only `gamma` and `seed` are set explicitly), per
  "All algorithms use SB3 default hyperparameters; any tuning is reported
  explicitly."
- **Metrics**: `evaluate.py` reports the three things the proposal asks for
  per algorithm-reward combination, with mean and std across seeds:
  - sample efficiency (timesteps to first reach mean reward >= 200)
  - success rate (return >= 200 at the end of training)
  - area under the learning curve (overall learning speed/quality)

## 7. Suggested next steps for the report

1. Run a short pilot (e.g. 100k-200k timesteps, 2 seeds) for all 12
   combinations to check everything trains sensibly and estimate runtime.
2. Scale up to the full 1,000,000-timestep, 3-5 seed runs for the final
   results.
3. Use `results_summary.csv` to build the comparison table(s) for the report,
   and the `<algo>_learning_curves.png` plots to discuss whether PBRS speeds
   up learning (especially for DQN, per the hypothesis) without degrading
   final performance.
4. Optionally, after training, load `best_model.zip` for a given run and
   record a video / rollout with `model.predict()` to visually confirm the
   learned landing behavior - this mirrors the qualitative check in the
   example project's Figures 5-6.
