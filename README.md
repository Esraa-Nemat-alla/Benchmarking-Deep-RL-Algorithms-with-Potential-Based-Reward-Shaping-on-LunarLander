# Benchmarking Deep RL Algorithms with Potential-Based Reward Shaping

Code for the proposal **"Benchmarking Deep Reinforcement Learning Algorithms with Potential-Based Reward Shaping on LunarLander"**.

In response to reviewer feedback to increase the scope for a 4-person project, this codebase has been upgraded to target the more challenging continuous control environment, `LunarLanderContinuous-v3`. Currently, it isolates **PPO** as the primary algorithm, applying Potential-Based Reward Shaping (PBRS) under 4 reward configurations: `none`, `distance`, `angle`, and `combined`. It evaluates sample efficiency, success rates, and final performance.

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

## 2. Web GUI (recommended)

Launch the dashboard in your browser:

```bash
streamlit run app.py
```

The GUI has four tabs:

| Tab | Purpose |
|---|---|
| **Dashboard** | Progress bar, results table, learning curves |
| **Train** | Run one experiment with custom settings |
| **Full Grid** | Launch all 12 benchmark runs |
| **Watch Agent** | Replay a trained lander as a GIF |

---

## 3. Codebase Architecture: What Each File Does

To make sure everyone on the team understands the codebase, here is a breakdown of the main Python scripts:

### `config.py` (Shared Settings)
* **What it does:** Single place for seeds, timesteps, reward configs, and paths. Used by train, evaluate, and the GUI.
* **What to do here:** Change `SEEDS`, `DEFAULT_TIMESTEPS`, or add new reward names when scaling the benchmark.

### `app.py` (Web GUI)
* **What it does:** Streamlit dashboard to train, evaluate, and watch agents without typing CLI commands.
* **What to do here:** Extend tabs or styling for your demo/presentation.

### `demo.py` (Agent Replay)
* **What it does:** Loads a saved model and records an episode as RGB frames (used by the Watch Agent tab).

### `reward_shaping.py` (The Environment & Math Hub)
* **What it does:** Contains the continuous environment setup and the custom logic for our reward shaping. It defines `phi_distance`, `phi_angle`, and `phi_combined` (Eqs. 4-6). It also contains the `PBRSRewardWrapper`, which applies the mathematical shaping formula: $F = \gamma \Phi(s') - \Phi(s)$.
* **What to do here:** If you want to invent a new shaping heuristic (e.g., punishing high speeds), you write your new `phi_velocity` function here and add it to the `POTENTIAL_FUNCTIONS` dictionary.

### `train.py` (The Single-Agent Worker)
* **What it does:** Trains on the shaped environment but **evaluates on the native (unshaped) reward** so success @ 200 is meaningful. Saves models and logs under `results/`.
* **What to do here:** Currently hardcoded to `PPO` to keep things simple. If you want to integrate `SAC`, `TD3`, or `DDPG` later, you import them at the top from `stable_baselines3` and add them to the `ALGORITHMS_CLS` dictionary.

### `run_all_experiments.py` (The Automation Manager)
* **What it does:** Loops through all combinations (Algorithms $\times$ Rewards $\times$ Seeds) and runs `train.py` automatically.
* **What to do here:** Edit `config.py` to change timesteps or seed count.

### `evaluate.py` (The Analytics Script)
* **What it does:** Once training runs are finished, it reads the `evaluations.npz` files inside the `results/` folder. It aggregates the math to calculate Success Rates, Area Under Curve (AUC), and Sample Efficiency. It generates a CSV table and learning curve graphs.
* **What to do here:** Edit this if you want to change the colors/titles of your matplotlib charts, or if you need to calculate a brand new metric for your final report.

---

## 4. Running a Single Experiment

To run a quick pilot test:
```bash
python train.py --algo ppo --reward distance --seed 0 --timesteps 50000
```
This creates `results/ppo_distance_seed0/` containing:
* `monitor.monitor.csv` - training episode returns/lengths
* `evaluations.npz` - evaluation reward over training (used by `evaluate.py`)
* `best_model.zip` - best checkpoint seen during training
* `final_model.zip` - model at the end of training

**Start small first:** Use `--timesteps 50000` to confirm everything runs in a few minutes before scaling up to 1,000,000 for your final paper.

---

## 5. Running the Full Grid

Once you are confident in your pilot runs, you can execute the full benchmark matrix:
```bash
python run_all_experiments.py
```
*Note: This will execute multiple 1,000,000 timestep runs sequentially. On a laptop, PPO takes roughly 15-30 minutes per million steps. Budget your compute time accordingly before the project deadline!*

---

## 6. Evaluating Results

Once some runs have finished, generate your plots and tables:
```bash
python evaluate.py
```
This produces:
* `results_summary.csv` - Data table containing mean/std of final evaluation rewards, success rates (returns $\ge$ 200), and AUC.
* `<algo>_learning_curves.png` - Mean $\pm$ std evaluation reward curves over training for each reward configuration.

`evaluate.py` will gracefully ignore missing data, meaning you can run it *during* your full grid execution to check on your progress without crashing!

---

## 7. Design Choices & Proposal Mapping

- **Environment:** Upgraded from the discrete `LunarLander-v3` to `LunarLanderContinuous-v3` to satisfy the reviewer's request for a more challenging scope.
- **Algorithms:** Currently restricted to **PPO** from Stable-Baselines3, providing a flawless baseline before integrating SAC/TD3/DDPG.
- **Potentials:** 
  - `phi_distance`: Fixed, stationary scale normalized by the environment's observation-space bounds.
  - `phi_angle`: Punishes non-upright angles.
  - `phi_combined`: Uses the $0.7/0.3$ weighting scheme from Eq. 6.
- **Shaping Math:** $\gamma = 0.99$ is synchronized perfectly between the wrapper and the agent's discount factor to preserve mathematical guarantees.
- **Evaluation:** Metrics are always computed on the **native** LunarLander reward (no shaping), even when training with PBRS.
