import os
import json
import warnings
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from ray.rllib.algorithms.ppo import PPO
from env.economy_env import SimpleEconomyEnv

# Suppress deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
os.environ['PYTHONWARNINGS'] = 'ignore::DeprecationWarning'

st.set_page_config(page_title="VWL Simulation Dashboard", layout="wide")

def load_checkpoints():
    """Load all available checkpoints"""
    checkpoint_dir = Path("./checkpoints").absolute()
    
    if not checkpoint_dir.exists():
        return []
    
    # Ray saves directly to ./checkpoints/
    metadata_file = checkpoint_dir / "policies" / "metadata.json"
    rllib_config_file = checkpoint_dir / "rllib_checkpoint.json"
    
    if metadata_file.exists() and rllib_config_file.exists():
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        with open(rllib_config_file, 'r') as f:
            rllib_config = json.load(f)
        
        # Extract env_config from checkpoint
        env_config = rllib_config.get('env_config', {})
        
        return [{
            'path': str(checkpoint_dir),
            'iteration': metadata.get('iteration', 0),
            'reward': metadata.get('reward_mean', 0.0),
            'is_favorite': metadata.get('is_favorite', False),
            'n_firms': env_config.get('n_firms', 2),
            'n_households': env_config.get('n_households', 10),
            'max_steps': env_config.get('max_steps', 100),
        }]
    
    return []

def run_simulation(checkpoint_path, n_firms, n_households, n_steps, 
                   price_range, wage_range, money_range):
    """Run simulation using trained checkpoint"""
    
    checkpoint_path = os.path.abspath(checkpoint_path)
    
    env_config = {
        'n_firms': n_firms,
        'n_households': n_households,
        'max_steps': n_steps,
    }
    
    # Restore algorithm from checkpoint
    algo = PPO.from_checkpoint(checkpoint_path)
    
    # Create environment with custom initial ranges
    env = SimpleEconomyEnv(env_config)
    
    # Storage for results
    firm_data = {i: {'prices': [], 'wages': [], 'profits': [], 'employees': []} 
                 for i in range(n_firms)}
    household_data = {i: {'money': [], 'employed': []} 
                      for i in range(n_households)}
    
    obs, info = env.reset()
    
    # Override initial values with custom ranges
    import numpy as np
    for i in range(n_firms):
        firm_id = f"firm_{i}"
        env.firms[firm_id]['price'] = np.random.uniform(price_range[0], price_range[1])
        env.firms[firm_id]['wage'] = np.random.uniform(wage_range[0], wage_range[1])
    
    for hh in env.households:
        hh['money'] = np.random.uniform(money_range[0], money_range[1])
    
    done = False
    step = 0
    
    while not done and step < n_steps:
        # Get actions from trained model
        actions = {}
        for agent_id in obs.keys():
            action, _, _ = algo.get_policy("shared_policy").compute_single_action(obs[agent_id])
            actions[agent_id] = action
        
        # Step environment
        obs, rewards, dones, truncated, info = env.step(actions)
        
        # Record firm data
        for i in range(n_firms):
            firm_id = f"firm_{i}"
            firm = env.firms[firm_id]
            firm_data[i]['prices'].append(firm['price'])
            firm_data[i]['wages'].append(firm['wage'])
            firm_data[i]['profits'].append(firm['profit'])
            firm_data[i]['employees'].append(firm['employees'])
        
        # Record household data
        for i, hh in enumerate(env.households):
            household_data[i]['money'].append(hh['money'])
            household_data[i]['employed'].append(1 if hh['employer'] is not None else 0)
        
        done = dones.get('__all__', False)
        step += 1
    
    algo.stop()
    
    return firm_data, household_data

# Sidebar
st.sidebar.title("🎮 Simulation Control")

# Load available checkpoints
checkpoints = load_checkpoints()

if not checkpoints:
    st.sidebar.error("❌ No trained checkpoints found!")
    st.sidebar.info("Run training first:\n```bash\npython train.py --iterations 20\n```")
    st.stop()

# Checkpoint selection
st.sidebar.subheader("📦 Select Checkpoint")
checkpoint_options = [
    f"{'⭐ ' if cp['is_favorite'] else ''}Iteration {cp['iteration']} (Reward: {cp['reward']:.1f})"
    for cp in checkpoints
]
selected_idx = st.sidebar.selectbox("Trained Version", range(len(checkpoint_options)), 
                                     format_func=lambda i: checkpoint_options[i])

selected_checkpoint = checkpoints[selected_idx]

# Show fixed environment parameters from training
st.sidebar.info(
    f"📋 **Training Environment:**\n"
    f"- Firms: {selected_checkpoint['n_firms']}\n"
    f"- Households: {selected_checkpoint['n_households']}\n"
    f"- Max Steps: {selected_checkpoint['max_steps']}"
)

# Simulation parameters - only initial ranges!
st.sidebar.subheader("⚙️ Initial Parameter Ranges")

st.sidebar.markdown("**Initial Price Range (per Firm)**")
price_min = st.sidebar.number_input("Min Price", min_value=1.0, max_value=50.0, value=8.0, step=0.5)
price_max = st.sidebar.number_input("Max Price", min_value=1.0, max_value=50.0, value=15.0, step=0.5)

st.sidebar.markdown("**Initial Wage Range (per Firm)**")
wage_min = st.sidebar.number_input("Min Wage", min_value=1.0, max_value=20.0, value=5.0, step=0.5)
wage_max = st.sidebar.number_input("Max Wage", min_value=1.0, max_value=20.0, value=12.0, step=0.5)

st.sidebar.markdown("**Initial Money Range (per Household)**")
money_min = st.sidebar.number_input("Min Money", min_value=0.0, max_value=200.0, value=30.0, step=5.0)
money_max = st.sidebar.number_input("Max Money", min_value=0.0, max_value=200.0, value=70.0, step=5.0)

# Run button
if st.sidebar.button("🚀 Start Simulation", type="primary"):
    with st.spinner("Running simulation..."):
        firm_data, household_data = run_simulation(
            selected_checkpoint['path'],
            selected_checkpoint['n_firms'],
            selected_checkpoint['n_households'],
            selected_checkpoint['max_steps'],
            (price_min, price_max),
            (wage_min, wage_max),
            (money_min, money_max)
        )
        st.session_state['firm_data'] = firm_data
        st.session_state['household_data'] = household_data
        st.session_state['n_steps'] = selected_checkpoint['max_steps']
        st.session_state['n_firms'] = selected_checkpoint['n_firms']
        st.success("✅ Simulation complete!")

# Main content
st.title("📊 VWL Simulation Dashboard")

if 'firm_data' not in st.session_state:
    st.info("👈 Configure initial parameter ranges and click 'Start Simulation' to begin")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["🏢 Firms", "🏠 Households", "📈 Summary"])

with tab1:
    st.header("Firm Analysis")
    
    for firm_id, data in st.session_state['firm_data'].items():
        st.subheader(f"Firm {firm_id}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price chart
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(
                y=data['prices'],
                mode='lines',
                name='Price',
                line=dict(color='blue')
            ))
            fig_price.update_layout(
                title=f"Price Development",
                xaxis_title="Step",
                yaxis_title="Price",
                height=300
            )
            st.plotly_chart(fig_price, width='stretch')
            
            # Profit chart
            fig_profit = go.Figure()
            fig_profit.add_trace(go.Scatter(
                y=data['profits'],
                mode='lines',
                name='Profit',
                line=dict(color='green')
            ))
            fig_profit.update_layout(
                title=f"Profit Development",
                xaxis_title="Step",
                yaxis_title="Profit",
                height=300
            )
            st.plotly_chart(fig_profit, width='stretch')
        
        with col2:
            # Wage chart
            fig_wage = go.Figure()
            fig_wage.add_trace(go.Scatter(
                y=data['wages'],
                mode='lines',
                name='Wage',
                line=dict(color='orange')
            ))
            fig_wage.update_layout(
                title=f"Wage Development",
                xaxis_title="Step",
                yaxis_title="Wage",
                height=300
            )
            st.plotly_chart(fig_wage, width='stretch')
            
            # Employees chart
            fig_emp = go.Figure()
            fig_emp.add_trace(go.Scatter(
                y=data['employees'],
                mode='lines',
                name='Employees',
                line=dict(color='purple')
            ))
            fig_emp.update_layout(
                title=f"Employee Count",
                xaxis_title="Step",
                yaxis_title="Employees",
                height=300
            )
            st.plotly_chart(fig_emp, width='stretch')
        
        st.divider()

with tab2:
    st.header("Household Analysis")
    
    # Average money over time
    avg_money = [sum(st.session_state['household_data'][i]['money'][step] 
                     for i in range(len(st.session_state['household_data']))) 
                 / len(st.session_state['household_data'])
                 for step in range(st.session_state['n_steps'])]
    
    fig_money = go.Figure()
    fig_money.add_trace(go.Scatter(
        y=avg_money,
        mode='lines',
        name='Avg Money',
        line=dict(color='green')
    ))
    fig_money.update_layout(
        title="Average Household Money",
        xaxis_title="Step",
        yaxis_title="Money",
        height=400
    )
    st.plotly_chart(fig_money, width='stretch')
    
    # Employment rate
    employment_rate = [sum(st.session_state['household_data'][i]['employed'][step] 
                           for i in range(len(st.session_state['household_data']))) 
                       / len(st.session_state['household_data']) * 100
                       for step in range(st.session_state['n_steps'])]
    
    fig_emp = go.Figure()
    fig_emp.add_trace(go.Scatter(
        y=employment_rate,
        mode='lines',
        name='Employment Rate',
        line=dict(color='blue')
    ))
    fig_emp.update_layout(
        title="Employment Rate",
        xaxis_title="Step",
        yaxis_title="Employment %",
        height=400
    )
    st.plotly_chart(fig_emp, width='stretch')

with tab3:
    st.header("Economic Summary")
    
    col1, col2, col3 = st.columns(3)
    
    n_firms = st.session_state['n_firms']
    
    with col1:
        avg_price = sum(data['prices'][-1] for data in st.session_state['firm_data'].values()) / n_firms
        st.metric("Average Price", f"{avg_price:.2f}")
    
    with col2:
        avg_wage = sum(data['wages'][-1] for data in st.session_state['firm_data'].values()) / n_firms
        st.metric("Average Wage", f"{avg_wage:.2f}")
    
    with col3:
        total_profit = sum(data['profits'][-1] for data in st.session_state['firm_data'].values())
        st.metric("Total Profit", f"{total_profit:.2f}")
    
    st.divider()
    
    # All firms comparison
    fig_all = go.Figure()
    for firm_id, data in st.session_state['firm_data'].items():
        fig_all.add_trace(go.Scatter(
            y=data['prices'],
            mode='lines',
            name=f'Firm {firm_id} Price'
        ))
    
    fig_all.update_layout(
        title="Price Comparison Across Firms",
        xaxis_title="Step",
        yaxis_title="Price",
        height=500
    )
    st.plotly_chart(fig_all, width='stretch')
