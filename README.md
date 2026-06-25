# Benchmarking Deep RL Algorithms with Potential-Based Reward Shaping

Code for the project **"Benchmarking Deep Reinforcement Learning Algorithms with Potential-Based Reward Shaping on LunarLander"**.

This version directly addresses the proposal feedback by increasing the experimental scope. Instead of benchmarking only three Stable-Baselines3 algorithms on a standard LunarLander setup, we benchmark **five deep RL algorithms** on the more challenging continuous-control variant, `LunarLanderContinuous-v3`, and add a dedicated hyperparameter sensitivity study.

## 1. Feedback-Driven Scope Update

The instructor feedback suggested either adding more algorithms and experimental settings, or switching to a more complex environment. We chose the first option because it preserves the reward-shaping research question while making the benchmark substantially broader.

Updated scope:

- **Algorithms:** PPO, A2C, SAC, TD3, and DDPG.
- **Environment:** `LunarLanderContinuous-v3`, which uses continuous actions and is harder than the discrete LunarLander task.
- **Reward settings:** `none`, `distance`, `angle`, and `combined`.
- **Seeds:** 3 independent seeds per default experiment.
- **Default grid:** 5 algorithms x 4 reward settings x 3 seeds = **60 runs**.
- **Hyperparameter study:** 2 learning rates x 2 network sizes x 3 seeds = **12 additional runs**.
- **Total planned experiments:** **72 runs**.

This keeps the project focused on potential-based reward shaping while expanding the comparison across on-policy and off-policy methods.

## 2. Key Results

Current completed experiment set:

- **Main benchmark:** 23/60 runs complete (PPO 12/12, A2C 11/12; SAC, TD3, DDPG pending).
- **Hyperparameter study:** 1/12 runs complete.
- **Total completed runs:** 24/72.
- **Algorithms with full coverage:** PPO.
- **Reward configurations:** `none`, `distance`, `angle`, `combined`.
- **Seeds:** 0, 1, 2 for every completed PPO configuration.

> **Note:** Run `python run_priority_experiments.py --phase coverage` to complete the remaining main grid experiments.

Best observed configurations from the currently available results (PPO and A2C):

| Finding | Result |
|---|---|
| Best mean final reward | PPO with `combined` shaping |
| Highest success rate | PPO with `none` |
| Largest final-reward shaping gain | A2C with `combined` over baseline |

Main takeaway: combined potential-based reward shaping improved mean final reward for PPO and A2C. SAC, TD3, and DDPG experiments are still pending and need to be run before cross-algorithm conclusions can be drawn.

Generated results are organized under:

```text
reports/
  figures/
  tables/
  demos/
  RESULTS_INTERPRETATION.md
```

## 3. Setup

### Option A: Conda/Mamba Environment

Linux:

```bash
conda env create -f environment-linux.yml
conda activate lunarlander-pbrs
```

Windows:

```powershell
conda env create -f environment-windows.yml
conda activate lunarlander-pbrs
```

If Box2D installation fails on Windows, install SWIG first and rerun the environment command:

```powershell
pip install swig
pip install "gymnasium[box2d]"
```

## 4. Experiment Design

### Algorithms

| Algorithm | Type | Key idea |
|---|---|---|
| PPO | On-policy | Clipped surrogate objective for stable updates |
| A2C | On-policy | Synchronous advantage actor-critic |
| SAC | Off-policy | Maximum entropy RL for exploration |
| TD3 | Off-policy | Twin critics and delayed policy updates |
| DDPG | Off-policy | Deterministic policy gradient with replay buffer |

### Reward Shaping Configurations

| Config | Potential function | Description |
|---|---|---|
| `none` | None | Vanilla environment reward baseline |
| `distance` | `phi(s) = -d / d_max` | Encourages proximity to the landing pad |
| `angle` | `phi(s) = -abs(theta) / pi` | Encourages upright orientation |
| `combined` | `0.7 * phi_distance + 0.3 * phi_angle` | Combines position and orientation shaping |

The shaping term follows potential-based reward shaping:

```text
F(s, a, s') = gamma * Phi(s') - Phi(s)
```

Evaluation is always performed on the native, unshaped environment reward so the comparison remains fair.

### Hyperparameter Sensitivity Study

The hyperparameter study tests whether the effect of reward shaping is robust under different training settings:

- Learning rates: `3e-4`, `1e-4`
- Network sizes: `[64, 64]`, `[256, 256]`
- Algorithm/reward pair: PPO with `combined` shaping
- Seeds: `0`, `1`, `2`

## 5. Web GUI

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

The GUI has five tabs:

| Tab | Purpose |
|---|---|
| Dashboard | Progress bar, results table, learning curves, comparative bar chart |
| Train | Run one experiment with custom settings and optional hyperparameter overrides |
| Full Grid | Launch all 60 default benchmark runs |
| Hyperparameter Study | Run and visualize the 12-run sensitivity analysis |
| Watch Agent | Replay a trained lander as a GIF |

## 6. Codebase Architecture

Project organization:

```text
Project_Proposal/       Original proposal PDF
reports/
  demos/                Agent replay GIFs
  figures/              Generated evaluation plots
  tables/               Generated CSV summaries
results/                Training outputs, saved models, monitor logs
```

### `config.py`

Shared settings for algorithms, seeds, timesteps, reward configs, hyperparameter grid, and paths.

### `reward_shaping.py`

Defines `LunarLanderContinuous-v3`, the potential functions, and the `PBRSRewardWrapper`.

### `train.py`

Trains one RL agent, evaluates on unshaped reward, and saves models/logs under `results/`. Supports PPO, A2C, SAC, TD3, and DDPG.

### `run_all_experiments.py`

Runs the full algorithm x reward x seed grid, then runs the hyperparameter sensitivity study.

### `run_priority_experiments.py`

Runs the next most useful experiments after the pilot grid, while skipping runs that already have `evaluations.npz`.

### `evaluate.py`

Loads evaluation files, computes summary metrics, and generates plots.

### `analyze_results.py`

Builds higher-level report artifacts from `results_summary.csv`, including best configurations, reward-shaping deltas, and a short interpretation report.

### `demo.py`

Loads a saved model, auto-detects the algorithm from the run folder name, and records an episode.

### `app.py`

Streamlit dashboard for training, evaluation, visualization, and replay.

## 7. Running Experiments

Quick pilot:

```bash
python train.py --algo ppo --reward distance --seed 0 --timesteps 50000
```

Test all algorithms quickly:

```bash
python train.py --algo ppo  --reward none --seed 0 --timesteps 50000
python train.py --algo a2c  --reward none --seed 0 --timesteps 50000
python train.py --algo sac  --reward none --seed 0 --timesteps 50000
python train.py --algo td3  --reward none --seed 0 --timesteps 50000
python train.py --algo ddpg --reward none --seed 0 --timesteps 50000
```

Full grid:

```bash
python run_all_experiments.py
```

Priority follow-up experiments:

```bash
python run_priority_experiments.py --phase seeds --timesteps 50000
python evaluate.py
python analyze_results.py
```

To complete the planned 5 algorithm x 4 reward x 3 seed grid while skipping
finished runs:

```bash
python run_priority_experiments.py --phase coverage --timesteps 50000
python evaluate.py
python analyze_results.py
```

Custom hyperparameters:

```bash
python train.py --algo sac --reward combined --seed 0 --timesteps 500000 --lr 1e-4 --net-arch 256 256
```

## 8. Evaluating Results

After some runs finish:

```bash
python evaluate.py
python analyze_results.py
```

Generated outputs:

- `reports/tables/results_summary.csv`
- `reports/tables/best_configs.csv`
- `reports/tables/reward_shaping_effects.csv`
- `reports/RESULTS_INTERPRETATION.md`
- `reports/figures/<algo>_learning_curves.png`
- `reports/figures/comparative_bar_chart.png`
- `reports/figures/hyperparam_sensitivity.png`

`evaluate.py` ignores missing runs, so it can be used while experiments are still in progress.

To generate a replay GIF from a trained run:

```bash
python demo.py --run td3_combined_seed0 --output reports/demos/td3_combined.gif
```

## 9. Main Design Choices

- `LunarLanderContinuous-v3` increases task difficulty by requiring continuous control.
- Five algorithms provide a broader comparison across on-policy and off-policy RL.
- Reward shaping is separated into interpretable potential functions.
- The PBRS discount factor matches the agents' `gamma = 0.99`.
- Evaluation uses unshaped reward to avoid inflating reported performance.
- The hyperparameter study tests whether observed PBRS benefits remain stable across learning rates and network capacity.
