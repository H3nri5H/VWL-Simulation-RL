import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import time

st.set_page_config(
    page_title="VWL Simulation Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("VWL Simulation Dashboard")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Training", 
    "🎮 Simulation Setup", 
    "🏢 Firms", 
    "👥 Households", 
    "📊 Summary"
])

# Tab 1: Training Metrics
with tab1:
    st.header("Training Progress")
    
    def load_training_history():
        metrics_dir = Path("./metrics")
        
        if not metrics_dir.exists():
            return []
        
        result_files = sorted(
            metrics_dir.glob("*/result.json"),
            key=lambda p: int(p.parent.name.split('_')[1])
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
    
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = True
    
    col_left, col_right = st.columns([3, 1])
    with col_right:
        if st.button("⏸️ Pause" if st.session_state.auto_refresh else "▶️ Resume"):
            st.session_state.auto_refresh = not st.session_state.auto_refresh
    
    while st.session_state.auto_refresh:
        history = load_training_history()
        
        with placeholder.container():
            if not history:
                st.warning("No training data found. Start training with: `python train.py`")
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

# Tab 2: Simulation Setup
with tab2:
    st.header("Simulation Setup")
    
    # Load available checkpoints
    checkpoint_dir = Path("./checkpoints")
    checkpoints = []
    
    if checkpoint_dir.exists():
        for cp_dir in checkpoint_dir.iterdir():
            if cp_dir.is_dir():
                metadata_file = cp_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                    checkpoints.append({
                        'path': str(cp_dir),
                        'name': cp_dir.name,
                        'iteration': metadata.get('iteration', 0),
                        'reward': metadata.get('reward_mean', 0),
                        'is_favorite': metadata.get('is_favorite', False)
                    })
    
    checkpoints.sort(key=lambda x: x['iteration'], reverse=True)
    
    if not checkpoints:
        st.warning("No checkpoints found. Train a model first with: `python train.py`")
    else:
        st.subheader("Select Checkpoint")
        
        # Find favorite checkpoint
        favorite_idx = next((i for i, cp in enumerate(checkpoints) if cp['is_favorite']), 0)
        
        checkpoint_options = [
            f"{'⭐ ' if cp['is_favorite'] else ''}Iteration {cp['iteration']} (Reward: {cp['reward']:.2f})"
            for cp in checkpoints
        ]
        
        selected_idx = st.selectbox(
            "Checkpoint",
            range(len(checkpoint_options)),
            index=favorite_idx,
            format_func=lambda i: checkpoint_options[i]
        )
        
        selected_checkpoint = checkpoints[selected_idx]
        
        st.success(f"Selected: {selected_checkpoint['name']}")
        
        st.subheader("Simulation Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            n_firms = st.number_input("Number of Firms", min_value=2, max_value=10, value=2)
            n_households = st.number_input("Number of Households", min_value=5, max_value=100, value=10)
        
        with col2:
            n_steps = st.number_input("Simulation Steps", min_value=10, max_value=1000, value=100)
            step_label = st.selectbox("Time Unit", ["Days", "Quarters", "Years"])
        
        st.subheader("Initial Firm Conditions")
        
        firm_configs = []
        for i in range(n_firms):
            with st.expander(f"Firm {i+1}"):
                col1, col2 = st.columns(2)
                with col1:
                    initial_price = st.number_input(
                        f"Initial Price", 
                        min_value=1.0, 
                        max_value=50.0, 
                        value=10.0,
                        key=f"price_{i}"
                    )
                with col2:
                    initial_wage = st.number_input(
                        f"Initial Wage", 
                        min_value=1.0, 
                        max_value=50.0, 
                        value=8.0,
                        key=f"wage_{i}"
                    )
                
                firm_configs.append({
                    'price': initial_price,
                    'wage': initial_wage
                })
        
        if st.button("🚀 Start Simulation", type="primary"):
            # Store simulation config in session state
            st.session_state.simulation_config = {
                'checkpoint_path': selected_checkpoint['path'],
                'n_firms': n_firms,
                'n_households': n_households,
                'n_steps': n_steps,
                'step_label': step_label,
                'firm_configs': firm_configs
            }
            st.session_state.simulation_running = True
            st.success("Simulation configured! Run simulation code to start.")
            st.info("Note: Full simulation runner will be implemented next.")

# Tab 3: Firms
with tab3:
    st.header("Firms Overview")
    
    if 'simulation_running' in st.session_state and st.session_state.simulation_running:
        st.info("Simulation results will appear here once implemented.")
    else:
        st.info("Configure and start a simulation in the 'Simulation Setup' tab.")

# Tab 4: Households
with tab4:
    st.header("Households Overview")
    
    if 'simulation_running' in st.session_state and st.session_state.simulation_running:
        st.info("Household results will appear here once implemented.")
    else:
        st.info("Configure and start a simulation in the 'Simulation Setup' tab.")

# Tab 5: Summary
with tab5:
    st.header("Summary Statistics")
    
    if 'simulation_running' in st.session_state and st.session_state.simulation_running:
        st.info("Summary statistics will appear here once implemented.")
    else:
        st.info("Configure and start a simulation in the 'Simulation Setup' tab.")
