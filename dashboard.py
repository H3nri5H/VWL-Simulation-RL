import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import time

st.set_page_config(
    page_title="VWL Simulation Training",
    page_icon="📊",
    layout="wide"
)

st.title("VWL Simulation Training Dashboard")


def load_training_history():
    checkpoint_dir = Path("./checkpoints")
    
    if not checkpoint_dir.exists():
        return []
    
    result_files = sorted(
        checkpoint_dir.glob("*/result.json"),
        key=lambda p: p.stat().st_mtime
    )
    
    history = []
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            env_runners = data.get('env_runners', {})
            
            history.append({
                'iteration': data.get('training_iteration', 0),
                'reward_mean': env_runners.get('episode_reward_mean', 0),
                'reward_min': env_runners.get('episode_reward_min', 0),
                'reward_max': env_runners.get('episode_reward_max', 0),
                'episode_len': env_runners.get('episode_len_mean', 0),
            })
        except:
            continue
    
    return history


placeholder = st.empty()

while True:
    history = load_training_history()
    
    with placeholder.container():
        if not history:
            st.warning("No training data found. Start training with: python train.py")
        else:
            df = pd.DataFrame(history)
            latest = history[-1]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Iteration", latest['iteration'])
            with col2:
                st.metric("Reward Mean", f"{latest['reward_mean']:.2f}")
            with col3:
                st.metric("Episode Length", f"{latest['episode_len']:.0f}")
            
            st.subheader("Episode Reward")
            fig_reward = go.Figure()
            fig_reward.add_trace(go.Scatter(
                x=df['iteration'],
                y=df['reward_mean'],
                mode='lines',
                name='Mean',
                line=dict(color='#82ca9d', width=3)
            ))
            fig_reward.add_trace(go.Scatter(
                x=df['iteration'],
                y=df['reward_max'],
                mode='lines',
                name='Max',
                line=dict(color='#8884d8', width=1)
            ))
            fig_reward.add_trace(go.Scatter(
                x=df['iteration'],
                y=df['reward_min'],
                mode='lines',
                name='Min',
                line=dict(color='#ffc658', width=1)
            ))
            fig_reward.update_layout(
                template='plotly_dark',
                height=400,
                xaxis_title='Iteration',
                yaxis_title='Reward'
            )
            st.plotly_chart(fig_reward, use_container_width=True)
            
            st.subheader("Episode Length")
            fig_len = go.Figure()
            fig_len.add_trace(go.Scatter(
                x=df['iteration'],
                y=df['episode_len'],
                mode='lines',
                name='Length',
                line=dict(color='#82ca9d', width=3)
            ))
            fig_len.update_layout(
                template='plotly_dark',
                height=400,
                xaxis_title='Iteration',
                yaxis_title='Steps'
            )
            st.plotly_chart(fig_len, use_container_width=True)
    
    time.sleep(2)
    st.rerun()
