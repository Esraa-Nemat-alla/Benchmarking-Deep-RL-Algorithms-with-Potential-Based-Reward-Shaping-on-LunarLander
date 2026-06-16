"""
Web GUI for training, evaluating, and watching LunarLander PBRS experiments.

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
    REWARD_CONFIGS,
    SEEDS,
    SUCCESS_THRESHOLD,
)
from demo import frame_to_gif_bytes, list_runs, record_episode
from evaluate import build_summary_table, count_completed_runs, make_learning_curve_figure, plot_learning_curves

st.set_page_config(
    page_title="LunarLander PBRS Benchmark",
    page_icon="🚀",
    layout="wide",
)

st.title("LunarLander PBRS Benchmark")
st.caption("Train PPO with potential-based reward shaping and compare results.")


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


tab_dashboard, tab_train, tab_grid, tab_watch = st.tabs(
    ["Dashboard", "Train", "Full Grid", "Watch Agent"]
)


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
                st.success("Updated results_summary.csv and learning curve plots.")

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
                lambda x: "—" if x != x else f"{x:,.0f}"
            )
        st.subheader("Results summary")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    for algo in ALGORITHMS:
        fig = make_learning_curve_figure(algo)
        if fig is not None:
            st.subheader(f"{algo.upper()} learning curves")
            st.pyplot(fig, clear_figure=True)


with tab_train:
    st.subheader("Single experiment")
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
    eval_freq = c5.number_input("Eval frequency", min_value=1_000, value=10_000, step=1_000)

    st.caption(
        f"Training uses `{reward}` shaping. Evaluation always uses the native reward "
        f"(success threshold = {int(SUCCESS_THRESHOLD)})."
    )

    if st.button("Start training", type="primary", use_container_width=True):
        cmd = [
            sys.executable,
            "train.py",
            "--algo",
            algo,
            "--reward",
            reward,
            "--seed",
            str(seed),
            "--timesteps",
            str(timesteps),
            "--eval-freq",
            str(eval_freq),
        ]
        with st.spinner(f"Training {algo}_{reward}_seed{seed}..."):
            code, output = run_command(cmd)
        st.code(output or "(no output)", language="text")
        if code == 0:
            st.success(f"Done. Results saved to results/{algo}_{reward}_seed{seed}/")
        else:
            st.error("Training failed. Check the log above.")


with tab_grid:
    st.subheader("Full benchmark grid")
    grid_total = len(ALGORITHMS) * len(REWARD_CONFIGS) * len(SEEDS)
    st.write(
        f"Runs **{grid_total}** jobs sequentially: "
        f"{len(ALGORITHMS)} algorithm(s) × {len(REWARD_CONFIGS)} rewards × {len(SEEDS)} seeds "
        f"at {DEFAULT_TIMESTEPS:,} timesteps each."
    )
    st.warning("This can take many hours on a laptop. Use the Train tab for quick pilots first.")

    if st.button("Run full grid", type="primary", use_container_width=True):
        cmd = [sys.executable, "run_all_experiments.py"]
        with st.spinner("Running full experiment grid..."):
            code, output = run_command(cmd)
        st.code(output or "(no output)", language="text")
        if code == 0:
            st.success("Full grid finished.")
        else:
            st.error("Grid run stopped with errors.")


with tab_watch:
    st.subheader("Watch a trained agent")
    runs = list_runs()
    if not runs:
        st.info("No saved models found. Train a run first.")
    else:
        run_name = st.selectbox("Run", runs)
        prefer_best = st.checkbox("Use best checkpoint (recommended)", value=True)
        demo_seed = st.number_input("Episode seed", min_value=0, value=0, step=1, key="demo_seed")

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
                    st.image(gif_bytes, caption=f"{run_name} — total reward: {total_reward:.1f}")
                    if total_reward >= SUCCESS_THRESHOLD:
                        st.success(f"Solved (reward ≥ {int(SUCCESS_THRESHOLD)})")
                    else:
                        st.info(f"Below solved threshold ({int(SUCCESS_THRESHOLD)})")
