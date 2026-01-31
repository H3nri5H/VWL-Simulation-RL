import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import time
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from simulate import run_simulation

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

# Helper function to load simulation results
def load_simulation_results():
    results_file = Path("./simulation_results/latest_simulation.json")
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return None

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
    
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False
    
    col_left, col_right = st.columns([3, 1])
    with col_right:
        if st.button("⏸️ Pause" if st.session_state.auto_refresh else "▶️ Auto-Refresh"):
            st.session_state.auto_refresh = not st.session_state.auto_refresh
            st.rerun()
    
    history = load_training_history()
    
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
            mode='lines+markers',
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
        st.plotly_chart(fig_reward, width='stretch')
        
        st.subheader("Episode Length")
        fig_len = go.Figure()
        fig_len.add_trace(go.Scatter(
            x=df['iteration'],
            y=df['episode_len'],
            mode='lines+markers',
            name='Length',
            line=dict(color='#82ca9d', width=3)
        ))
        fig_len.update_layout(
            template='plotly_dark',
            height=400,
            xaxis_title='Iteration',
            yaxis_title='Steps'
        )
        st.plotly_chart(fig_len, width='stretch')
    
    if st.session_state.auto_refresh:
        time.sleep(2)
        st.rerun()

# Tab 2: Simulation Setup  
with tab2:
    st.header("Simulation Setup")
    
    checkpoint_dir = Path("./checkpoints")
    checkpoints = []
    
    if checkpoint_dir.exists():
        for cp_dir in checkpoint_dir.iterdir():
            if cp_dir.is_dir():
                metadata_file = cp_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file) as f:
                            metadata = json.load(f)
                        checkpoints.append({
                            'path': str(cp_dir),
                            'name': cp_dir.name,
                            'iteration': metadata.get('iteration', 0),
                            'reward': metadata.get('reward_mean', 0),
                            'is_favorite': metadata.get('is_favorite', False)
                        })
                    except:
                        pass
    
    checkpoints.sort(key=lambda x: x['iteration'], reverse=True)
    
    if not checkpoints:
        st.warning("No checkpoints found. Train a model first with: `python train.py`")
    else:
        st.subheader("Select Checkpoint")
        
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
            with st.spinner("Running simulation..."):
                config = {
                    'checkpoint_path': selected_checkpoint['path'],
                    'n_firms': n_firms,
                    'n_households': n_households,
                    'n_steps': n_steps,
                    'step_label': step_label,
                    'firm_configs': firm_configs
                }
                
                try:
                    results = run_simulation(config)
                    st.session_state.simulation_results = results
                    st.session_state.simulation_complete = True
                    st.success("✅ Simulation complete! Check other tabs for results.")
                except Exception as e:
                    st.error(f"Error running simulation: {e}")

# Tab 3: Firms
with tab3:
    st.header("Firms Overview")
    
    results = load_simulation_results()
    
    if results is None:
        st.info("No simulation results. Run a simulation first.")
    else:
        firm_data = results['firms']
        n_firms = len(firm_data)
        
        # Firm selector
        firm_names = list(firm_data.keys())
        selected_firm = st.selectbox("Select Firm", firm_names)
        
        # Convert to DataFrame
        df = pd.DataFrame(firm_data[selected_firm])
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Final Price", f"{df['price'].iloc[-1]:.2f}€")
        with col2:
            st.metric("Final Wage", f"{df['wage'].iloc[-1]:.2f}€")
        with col3:
            st.metric("Avg Profit", f"{df['profit'].mean():.2f}€")
        with col4:
            st.metric("Avg Employees", f"{df['employees'].mean():.1f}")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Price & Wage")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['step'], y=df['price'], name='Price', line=dict(color='#8884d8')))
            fig.add_trace(go.Scatter(x=df['step'], y=df['wage'], name='Wage', line=dict(color='#82ca9d')))
            fig.update_layout(template='plotly_dark', height=300)
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.subheader("Profit")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['step'], y=df['profit'], fill='tozeroy', line=dict(color='#ffc658')))
            fig.update_layout(template='plotly_dark', height=300)
            st.plotly_chart(fig, width='stretch')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Revenue vs Costs")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['step'], y=df['revenue'], name='Revenue', line=dict(color='#82ca9d')))
            fig.add_trace(go.Scatter(x=df['step'], y=df['costs'], name='Costs', line=dict(color='#ff6b6b')))
            fig.update_layout(template='plotly_dark', height=300)
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.subheader("Employees")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['step'], y=df['employees'], fill='tozeroy', line=dict(color='#8884d8')))
            fig.update_layout(template='plotly_dark', height=300)
            st.plotly_chart(fig, width='stretch')

# Tab 4: Households
with tab4:
    st.header("Households Overview")
    
    results = load_simulation_results()
    
    if results is None:
        st.info("No simulation results. Run a simulation first.")
    else:
        df = pd.DataFrame(results['households'])
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Final Total Money", f"{df['total_money'].iloc[-1]:.2f}€")
        with col2:
            st.metric("Avg Employment Rate", f"{(df['employed'].mean() / (df['employed'] + df['unemployed']).iloc[0] * 100):.1f}%")
        with col3:
            st.metric("Avg Wage", f"{df['avg_wage'].mean():.2f}€")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Total Money")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['step'], y=df['total_money'], fill='tozeroy', line=dict(color='#82ca9d')))
            fig.update_layout(template='plotly_dark', height=350)
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.subheader("Employment")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['step'], y=df['employed'], name='Employed', line=dict(color='#82ca9d'), fill='tonexty'))
            fig.add_trace(go.Scatter(x=df['step'], y=df['unemployed'], name='Unemployed', line=dict(color='#ff6b6b'), fill='tozeroy'))
            fig.update_layout(template='plotly_dark', height=350)
            st.plotly_chart(fig, width='stretch')
        
        st.subheader("Average Wage")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['step'], y=df['avg_wage'], line=dict(color='#8884d8')))
        fig.update_layout(template='plotly_dark', height=300)
        st.plotly_chart(fig, width='stretch')

# Tab 5: Summary
with tab5:
    st.header("Summary Statistics")
    
    results = load_simulation_results()
    
    if results is None:
        st.info("No simulation results. Run a simulation first.")
    else:
        df = pd.DataFrame(results['summary'])
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Total Profit", f"{df['total_profit'].mean():.2f}€")
        with col2:
            st.metric("Avg Price", f"{df['avg_price'].mean():.2f}€")
        with col3:
            st.metric("Avg Employment", f"{df['employment_rate'].mean() * 100:.1f}%")
        with col4:
            st.metric("Total Wealth", f"{df['total_money'].iloc[-1]:.2f}€")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Total Profit Over Time")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['step'], y=df['total_profit'], fill='tozeroy', line=dict(color='#ffc658')))
            fig.update_layout(template='plotly_dark', height=350)
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.subheader("Average Price")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['step'], y=df['avg_price'], line=dict(color='#8884d8')))
            fig.update_layout(template='plotly_dark', height=350)
            st.plotly_chart(fig, width='stretch')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Total Money in Economy")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['step'], y=df['total_money'], fill='tozeroy', line=dict(color='#82ca9d')))
            fig.update_layout(template='plotly_dark', height=350)
            st.plotly_chart(fig, width='stretch')
        
        with col2:
            st.subheader("Employment Rate")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['step'], y=df['employment_rate'] * 100, line=dict(color='#ff6b6b')))
            fig.update_layout(template='plotly_dark', height=350, yaxis_title="%")
            st.plotly_chart(fig, width='stretch')
