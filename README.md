# Benchmarking Deep RL Algorithms with Potential-Based Reward Shaping

Code for the project **"Benchmarking Deep Reinforcement Learning Algorithms with Potential-Based Reward Shaping on LunarLander"**.

We benchmark **5 deep RL algorithms** (PPO, A2C, SAC, TD3, DDPG) on the challenging continuous control environment `LunarLanderContinuous-v3`, applying **Potential-Based Reward Shaping (PBRS)** under 4 reward configurations: `none`, `distance`, `angle`, and `combined`. We also include a hyperparameter sensitivity study to assess robustness.

---

## 1. Setup

**Create a virtual environment and install dependencies:**
```bash
python -m venv venv
source venv/bin/activate          
pip install -r requirements.txt
```

**Install Box2D for LunarLander:**
If you run into Box2D errors, install `swig` first, then install the gymnasium box2d package:
```bash
pip install swig
pip install "gymnasium[box2d]"
```

Quick sanity check:
```bash
python -c "import gymnasium as gym; gym.make('LunarLanderContinuous-v3')"
```

---

## 2. Experiment Design

### Algorithms (5 total)

| Algorithm | Type | Key Idea |
|---|---|---|
| **PPO** | On-policy | Clipped surrogate objective for stable updates |
| **A2C** | On-policy | Synchronous advantage actor-critic |
| **SAC** | Off-policy | Maximum entropy RL for exploration |
| **TD3** | Off-policy | Twin critics + delayed policy updates |
| **DDPG** | Off-policy | Deterministic policy gradient with replay buffer |

### Reward Shaping Configs (4 total)

| Config | Potential Function | Description |
|---|---|---|
| `none` | — | Vanilla environment reward (baseline) |
| `distance` | φ(s) = −d/d_max | Penalizes distance from the landing pad |
| `angle` | φ(s) = −|θ|/π | Penalizes non-upright orientation |
| `combined` | 0.7·φ_dist + 0.3·φ_angle | Weighted combination of both |

### Default Experiment Grid

- **5 algorithms × 4 reward configs × 3 seeds = 60 runs**
- Each run trains for 1,000,000 timesteps

### Hyperparameter Sensitivity Study

- Tests 2 learning rates (3e-4, 1e-4) × 2 network sizes ([64,64], [256,256])
- Run on PPO + `combined` shaping × 3 seeds = 12 additional runs

---

## 3. Web GUI (Recommended)

Launch the dashboard in your browser:

```bash
streamlit run app.py
```

The GUI has five tabs:

| Tab | Purpose |
|---|---|
| **📊 Dashboard** | Progress bar, results table, learning curves, comparative bar chart |
| **🏋️ Train** | Run one experiment with custom settings & optional hyperparameter overrides |
| **🔲 Full Grid** | Launch all 60 benchmark runs |
| **🔬 Hyperparameter Study** | Run and visualize the hyperparameter sensitivity analysis |
| **👀 Watch Agent** | Replay a trained lander as a GIF |

---

## 4. Codebase Architecture: What Each File Does

### `config.py` (Shared Settings)
* **What it does:** Single place for algorithms, seeds, timesteps, reward configs, hyperparameter grid, and paths. Used by every other script.
* **What to do here:** Add/remove algorithms, change `SEEDS`, `DEFAULT_TIMESTEPS`, or modify the `HYPERPARAM_GRID`.

### `reward_shaping.py` (The Environment & Math Hub)
* **What it does:** Contains the continuous environment setup and the PBRS logic. Defines `phi_distance`, `phi_angle`, and `phi_combined`. The `PBRSRewardWrapper` applies the shaping formula: F = γ·Φ(s') − Φ(s).
* **What to do here:** Add new potential functions (e.g., `phi_velocity`) and register them in the `POTENTIAL_FUNCTIONS` dictionary.

### `train.py` (The Single-Agent Worker)
* **What it does:** Trains one RL agent on the shaped environment, evaluates on the native (unshaped) reward, and saves models + logs to `results/`. Supports all 5 algorithms and optional `--lr` / `--net-arch` overrides.
* **What to do here:** Add new algorithms by importing them and adding to `ALGORITHMS_CLS`.

### `run_all_experiments.py` (The Automation Manager)
* **What it does:** Phase 1 runs the full algorithm × reward × seed grid. Phase 2 runs the hyperparameter sensitivity study. Both phases call `train.py` as subprocesses.
* **What to do here:** Edit `config.py` to change the grid dimensions.

### `evaluate.py` (The Analytics Script)
* **What it does:** Reads `evaluations.npz` files, computes metrics (success rate, AUC, sample efficiency), and generates: summary CSV, per-algorithm learning curves, comparative bar chart, and hyperparameter sensitivity plot.
* **What to do here:** Edit colors, titles, or add new metrics.

### `demo.py` (Agent Replay)
* **What it does:** Loads a saved model (auto-detects the algorithm from the folder name) and records an episode as RGB frames. Used by the Watch Agent tab.

### `app.py` (Web GUI)
* **What it does:** Streamlit dashboard with 5 tabs for training, evaluating, and visualizing results.

---

## 5. Running Experiments

### Quick Pilot (one algorithm, one config, ~2 minutes)

```bash
python train.py --algo ppo --reward distance --seed 0 --timesteps 50000
```

### Test All Algorithms Quickly

```bash
python train.py --algo ppo  --reward none --seed 0 --timesteps 50000
python train.py --algo a2c  --reward none --seed 0 --timesteps 50000
python train.py --algo sac  --reward none --seed 0 --timesteps 50000
python train.py --algo td3  --reward none --seed 0 --timesteps 50000
python train.py --algo ddpg --reward none --seed 0 --timesteps 50000
```

### Full Grid (60 runs at 1M steps each)

```bash
python run_all_experiments.py
```

*Note: Budget your compute time! On a laptop, each 1M-step run takes roughly 15–30 minutes. The full grid of 60 runs can take 15–30 hours.*

### Custom Hyperparameters

```bash
python train.py --algo sac --reward combined --seed 0 --timesteps 500000 --lr 1e-4 --net-arch 256 256
```

---

## 6. Evaluating Results

Once some runs have finished:

```bash
python evaluate.py
```

This produces:
* `results_summary.csv` — Data table with mean/std of final rewards, success rates, AUC
* `<algo>_learning_curves.png` — Per-algorithm learning curves (mean ± std across seeds)
* `comparative_bar_chart.png` — All algorithms compared side by side
* `hyperparam_sensitivity.png` — How learning rate and network size affect performance

`evaluate.py` gracefully ignores missing data, so you can run it mid-experiment to check progress.

---

## 7. Design Choices

- **Environment:** `LunarLanderContinuous-v3` — more challenging than discrete LunarLander, requires continuous control algorithms
- **Algorithms:** 5 algorithms from Stable-Baselines3 (2 on-policy + 3 off-policy) for a comprehensive comparison
- **Potentials:**
  - `phi_distance`: Normalized by the observation space bounds, rewards proximity to the pad
  - `phi_angle`: Punishes non-upright angles, normalized by π
  - `phi_combined`: 0.7/0.3 weighted scheme
- **Shaping Math:** γ = 0.99 synchronized between the PBRS wrapper and all agents' discount factors
- **Evaluation:** Metrics always computed on the native LunarLander reward (no shaping), even during training with PBRS
- **Hyperparameter Study:** Tests whether PBRS benefits are robust across different learning rates and network architectures
