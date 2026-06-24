"""
Web GUI for training, evaluating, and watching LunarLander PBRS experiments.

Tabs:
  - Dashboard:           Progress, results table, learning curves, comparative chart
  - Train:               Run a single experiment with custom settings
  - Full Grid:           Launch all 60+ benchmark runs
  - Hyperparameter Study: Launch and view the hyperparameter sensitivity analysis
  - Watch Agent:         Replay a trained lander as a GIF

Launch:
    streamlit run app.py
"""
import os
import subprocess
import sys

import streamlit as st

from config import (
    ALGORITHMS,
    DEFAULT_TIMESTEPS,
    HYPERPARAM_ALGO,
    HYPERPARAM_GRID,
    HYPERPARAM_REWARD,
    REWARD_CONFIGS,
    SEEDS,
    SUCCESS_THRESHOLD,
)
from demo import frame_to_gif_bytes, list_runs, record_episode
from evaluate import (
    build_summary_table,
    count_completed_runs,
    make_comparative_bar_chart,
    make_hyperparam_sensitivity_figure,
    make_learning_curve_figure,
    plot_learning_curves,
)

st.set_page_config(
    page_title="LunarLander PBRS Benchmark",
    page_icon="rocket",
    layout="wide",
)

st.title("LunarLander PBRS Benchmark")
st.caption(
    "Benchmarking 5 Deep RL algorithms (PPO, A2C, SAC, TD3, DDPG) "
    "with Potential-Based Reward Shaping on LunarLanderContinuous-v3."
)


def run_command(cmd):
    """Run a shell command and return stdout/stderr."""
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode, output


# Tabs
tab_dashboard, tab_train, tab_grid, tab_hyperparam, tab_watch = st.tabs(
    ["Dashboard", "Train", "Full Grid", "Hyperparameter Study", "Watch Agent"]
)


# Dashboard
with tab_dashboard:
    done, total = count_completed_runs()
    st.progress(done / total if total else 0.0, text=f"Completed runs: {done}/{total}")

    col_refresh, col_eval = st.columns([1, 1])
    with col_refresh:
        if st.button("Refresh dashboard", use_container_width=True):
            st.rerun()
    with col_eval:
        if st.button("Regenerate reports", use_container_width=True):
            with st.spinner("Building summary and plots..."):
                df = build_summary_table()
                plot_learning_curves()
            if df.empty:
                st.warning("No evaluation files found yet.")
            else:
                st.success("Updated reports/tables and reports/figures.")

    # Results summary table
    df = build_summary_table(save_csv=False)
    if df.empty:
        st.info("No results yet. Train a model in the **Train** tab or run the full grid.")
    else:
        display_df = df.copy()
        display_df["success_rate_mean"] = display_df["success_rate_mean"].map(lambda x: f"{x:.0%}")
        for col in ["final_reward_mean", "final_reward_std", "auc_mean", "auc_std"]:
            display_df[col] = display_df[col].map(lambda x: f"{x:.1f}")
        for col in ["timesteps_to_200_mean", "timesteps_to_200_std"]:
            display_df[col] = display_df[col].map(
                lambda x: "-" if x != x else f"{x:,.0f}"
            )
        st.subheader("Results Summary")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Comparative bar chart (the key cross-algorithm comparison)
    fig_bar = make_comparative_bar_chart()
    if fig_bar is not None:
        st.subheader("Algorithm Comparison")
        st.pyplot(fig_bar, clear_figure=True)

    # Per-algorithm learning curves
    for algo in ALGORITHMS:
        fig = make_learning_curve_figure(algo)
        if fig is not None:
            st.subheader(f"{algo.upper()} Learning Curves")
            st.pyplot(fig, clear_figure=True)


# Train a single experiment
with tab_train:
    st.subheader("Single Experiment")

    c1, c2, c3 = st.columns(3)
    algo = c1.selectbox("Algorithm", ALGORITHMS, key="train_algo")
    reward = c2.selectbox("Reward shaping", REWARD_CONFIGS, key="train_reward")
    seed = c3.number_input("Seed", min_value=0, max_value=9999, value=0, step=1)

    c4, c5 = st.columns(2)
    timesteps = c4.selectbox(
        "Timesteps",
        [50_000, 100_000, 500_000, 1_000_000],
        index=0,
        help="Start with 50k for a quick test.",
    )
    eval_freq = c5.number_input(
        "Eval frequency", min_value=1_000, value=10_000, step=1_000
    )

    # Optional hyperparameter overrides
    with st.expander("Advanced: Custom Hyperparameters"):
        use_custom_lr = st.checkbox("Override learning rate")
        custom_lr = st.number_input(
            "Learning rate", min_value=1e-6, max_value=1e-1,
            value=3e-4, format="%.1e", disabled=not use_custom_lr,
        )
        use_custom_net = st.checkbox("Override network architecture")
        custom_net_str = st.text_input(
            "Network layers (comma-separated)", value="64,64",
            help="E.g. '256,256' for two layers of 256 neurons.",
            disabled=not use_custom_net,
        )

    st.caption(
        f"Training uses `{reward}` shaping. Evaluation always uses the native reward "
        f"(success threshold = {int(SUCCESS_THRESHOLD)})."
    )

    if st.button("Start training", type="primary", use_container_width=True):
        cmd = [
            sys.executable, "train.py",
            "--algo", algo,
            "--reward", reward,
            "--seed", str(seed),
            "--timesteps", str(timesteps),
            "--eval-freq", str(eval_freq),
        ]
        if use_custom_lr:
            cmd.extend(["--lr", str(custom_lr)])
        if use_custom_net:
            net_layers = [s.strip() for s in custom_net_str.split(",")]
            cmd.extend(["--net-arch"] + net_layers)

        with st.spinner(f"Training {algo}_{reward}_seed{seed}..."):
            code, output = run_command(cmd)
        st.code(output or "(no output)", language="text")
        if code == 0:
            st.success(f"Done. Results saved to results/{algo}_{reward}_seed{seed}/")
        else:
            st.error("Training failed. Check the log above.")


# Full Grid
with tab_grid:
    st.subheader("Full Benchmark Grid")
    grid_total = len(ALGORITHMS) * len(REWARD_CONFIGS) * len(SEEDS)
    st.write(
        f"Runs **{grid_total}** jobs sequentially: "
        f"{len(ALGORITHMS)} algorithm(s) x {len(REWARD_CONFIGS)} rewards x {len(SEEDS)} seeds "
        f"at {DEFAULT_TIMESTEPS:,} timesteps each."
    )
    st.warning(
        "This can take many hours on a laptop. "
        "Use the Train tab for quick pilots first."
    )

    if st.button("Run full grid", type="primary", use_container_width=True):
        cmd = [sys.executable, "run_all_experiments.py"]
        with st.spinner("Running full experiment grid..."):
            code, output = run_command(cmd)
        st.code(output or "(no output)", language="text")
        if code == 0:
            st.success("Full grid finished.")
        else:
            st.error("Grid run stopped with errors.")


# Hyperparameter Study
with tab_hyperparam:
    st.subheader("Hyperparameter Sensitivity Study")
    st.write(
        f"Tests how **learning rate** and **network architecture** affect the impact "
        f"of reward shaping on **{HYPERPARAM_ALGO.upper()}** with **'{HYPERPARAM_REWARD}'** shaping."
    )

    # Show the grid configuration
    col_lr, col_net = st.columns(2)
    col_lr.metric("Learning Rates", ", ".join(str(lr) for lr in HYPERPARAM_GRID["learning_rate"]))
    col_net.metric("Network Sizes", " | ".join(
        str(arch) for arch in HYPERPARAM_GRID["net_arch"]
    ))

    hp_total = (
        len(HYPERPARAM_GRID["learning_rate"])
        * len(HYPERPARAM_GRID["net_arch"])
        * len(SEEDS)
    )
    st.write(f"Total runs: **{hp_total}** ({len(SEEDS)} seeds each)")

    if st.button("Run hyperparameter study", type="primary", use_container_width=True):
        # We run just the hyperparameter phase of run_all_experiments
        # by calling train.py directly for each combo
        import itertools
        combos = list(itertools.product(
            HYPERPARAM_GRID["learning_rate"],
            HYPERPARAM_GRID["net_arch"],
            SEEDS,
        ))
        progress = st.progress(0.0)
        for idx, (lr, net_arch, seed) in enumerate(combos):
            arch_str = "-".join(str(n) for n in net_arch)
            progress.progress(
                (idx) / len(combos),
                text=f"Running lr={lr}, net=[{arch_str}], seed={seed}..."
            )
            cmd = [
                sys.executable, "train.py",
                "--algo", HYPERPARAM_ALGO,
                "--reward", HYPERPARAM_REWARD,
                "--seed", str(seed),
                "--timesteps", str(DEFAULT_TIMESTEPS),
                "--eval-freq", str(10_000),
                "--lr", str(lr),
                "--net-arch",
            ] + [str(n) for n in net_arch]
            code, output = run_command(cmd)
            if code != 0:
                st.error(f"Failed: lr={lr}, net=[{arch_str}], seed={seed}")
                st.code(output, language="text")
                break
        else:
            progress.progress(1.0, text="All hyperparameter runs completed!")
            st.success("Hyperparameter study finished.")

    # Display results if available
    fig_hp = make_hyperparam_sensitivity_figure()
    if fig_hp is not None:
        st.subheader("Sensitivity Results")
        st.pyplot(fig_hp, clear_figure=True)
    else:
        st.info("No hyperparameter study results yet. Run the study above first.")


# Watch Agent
with tab_watch:
    st.subheader("Watch a Trained Agent")
    runs = list_runs()
    if not runs:
        st.info("No saved models found. Train a run first.")
    else:
        run_name = st.selectbox("Run", runs)
        prefer_best = st.checkbox("Use best checkpoint (recommended)", value=True)
        demo_seed = st.number_input(
            "Episode seed", min_value=0, value=0, step=1, key="demo_seed"
        )

        if st.button("Play episode", type="primary", use_container_width=True):
            with st.spinner("Running episode..."):
                try:
                    frames, total_reward = record_episode(
                        run_name,
                        prefer_best=prefer_best,
                        seed=demo_seed,
                    )
                    gif_bytes = frame_to_gif_bytes(frames)
                except Exception as exc:
                    st.error(str(exc))
                else:
                    st.image(
                        gif_bytes,
                        caption=f"{run_name} - total reward: {total_reward:.1f}",
                    )
                    if total_reward >= SUCCESS_THRESHOLD:
                        st.success(f"Solved (reward >= {int(SUCCESS_THRESHOLD)})")
                    else:
                        st.info(f"Below solved threshold ({int(SUCCESS_THRESHOLD)})")
