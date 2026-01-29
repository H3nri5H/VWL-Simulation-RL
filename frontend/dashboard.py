"""Streamlit Dashboard for VWL-Simulation Visualization.

Interactive dashboard to:
- Load trained models
- Configure simulation parameters
- Run simulations
- Visualize economic dynamics
- Analyze firm and household behavior
"""

import streamlit as st
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
import os

# Configuration
st.set_page_config(
    page_title="VWL-Simulation Viewer",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend URL (can be configured via environment variable)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    </style>
""", unsafe_allow_html=True)


def fetch_models():
    """Fetch available models from backend."""
    try:
        response = requests.get(f"{BACKEND_URL}/api/models", timeout=5)
        if response.status_code == 200:
            return response.json()["models"]
        else:
            st.error(f"Failed to fetch models: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        st.error(f"Backend connection error: {e}")
        st.info("💡 Make sure backend is running: `cd backend && uvicorn app:app --reload`")
        return []


def run_simulation(model_name, n_firms, n_households, max_steps, start_prices, start_wages, seed):
    """Run simulation via backend API."""
    payload = {
        "model_name": model_name,
        "n_firms": n_firms,
        "n_households": n_households,
        "max_steps": max_steps,
        "start_prices": start_prices,
        "start_wages": start_wages,
        "seed": seed
    }
    
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/simulate",
            json=payload,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Simulation failed: {response.json().get('detail', 'Unknown error')}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Simulation request failed: {e}")
        return None


# ============================================================================
# MAIN APP
# ============================================================================

st.title("🏭 Volkswirtschafts-Simulation mit Multi-Agent RL")
st.markdown("---")

# ============================================================================
# SIDEBAR: Configuration
# ============================================================================

with st.sidebar:
    st.header("⚙️ Simulation Setup")
    
    # Model selection
    st.subheader("🧠 Modell-Auswahl")
    available_models = fetch_models()
    
    if available_models:
        selected_model = st.selectbox(
            "Trainiertes Modell",
            available_models,
            help="Wähle ein trainiertes RL-Modell aus"
        )
    else:
        st.warning("Keine trainierten Modelle gefunden!")
        st.info("💡 Trainiere erst ein Modell oder nutze 'random' Policy")
        selected_model = "random"
    
    st.markdown("---")
    
    # Simulation parameters
    st.subheader("🎯 Parameter")
    
    n_firms = st.slider(
        "Anzahl Firmen",
        min_value=2,
        max_value=10,
        value=2,
        help="Anzahl konkurrierender Unternehmen"
    )
    
    n_households = st.slider(
        "Anzahl Haushalte",
        min_value=10,
        max_value=100,
        value=10,
        step=10,
        help="Anzahl Konsumenten im Markt"
    )
    
    max_steps = st.slider(
        "Quartale",
        min_value=10,
        max_value=200,
        value=100,
        step=10,
        help="Simulationslänge in Quartalen"
    )
    
    seed = st.number_input(
        "Random Seed (optional)",
        min_value=0,
        max_value=9999,
        value=42,
        help="Für reproduzierbare Ergebnisse"
    )
    
    st.markdown("---")
    
    # Start parameters
    st.subheader("📊 Start-Parameter")
    
    with st.expander("⚡ Erweiterte Einstellungen"):
        st.caption("Initiale Preise und Löhne pro Firma")
        
        start_prices = []
        start_wages = []
        
        for i in range(n_firms):
            col1, col2 = st.columns(2)
            with col1:
                price = st.number_input(
                    f"Firma {i} Preis",
                    min_value=1.0,
                    max_value=50.0,
                    value=10.0,
                    step=0.5,
                    key=f"price_{i}"
                )
                start_prices.append(price)
            
            with col2:
                wage = st.number_input(
                    f"Firma {i} Lohn",
                    min_value=1.0,
                    max_value=30.0,
                    value=8.0,
                    step=0.5,
                    key=f"wage_{i}"
                )
                start_wages.append(wage)
    
    st.markdown("---")
    
    # Run button
    run_button = st.button(
        "🚀 Simulation starten",
        type="primary",
        use_container_width=True
    )


# ============================================================================
# MAIN AREA: Results
# ============================================================================

if run_button:
    with st.spinner("🔄 Simulation läuft..."):
        result = run_simulation(
            model_name=selected_model,
            n_firms=n_firms,
            n_households=n_households,
            max_steps=max_steps,
            start_prices=start_prices,
            start_wages=start_wages,
            seed=seed
        )
        
        if result and result["status"] == "success":
            st.session_state['history'] = result['data']
            st.session_state['metadata'] = result['metadata']
            st.success("✅ Simulation abgeschlossen!")
        else:
            st.error("❌ Simulation fehlgeschlagen")


# Display results if available
if 'history' in st.session_state:
    history = st.session_state['history']
    metadata = st.session_state.get('metadata', {})
    
    # Summary metrics at top
    st.header("📊 Simulation Zusammenfassung")
    
    col1, col2, col3, col4 = st.columns(4)
    
    final_market = history['market'][-1]
    initial_market = history['market'][0]
    
    with col1:
        st.metric(
            "Finales BIP",
            f"{final_market['gdp']:.2f}€",
            delta=f"{final_market['gdp'] - initial_market['gdp']:.2f}€"
        )
    
    with col2:
        st.metric(
            "Durchschnittspreis",
            f"{final_market['avg_price']:.2f}€",
            delta=f"{final_market['avg_price'] - initial_market['avg_price']:.2f}€"
        )
    
    with col3:
        st.metric(
            "Durchschnittslohn",
            f"{final_market['avg_wage']:.2f}€",
            delta=f"{final_market['avg_wage'] - initial_market['avg_wage']:.2f}€"
        )
    
    with col4:
        total_steps = len(history['market'])
        st.metric(
            "Simulierte Quartale",
            total_steps,
            delta=f"{total_steps/4:.1f} Jahre"
        )
    
    st.markdown("---")
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Überblick",
        "🏭 Firmen-Details",
        "🏠 Haushalte",
        "💾 Daten Export"
    ])
    
    # ========================================================================
    # TAB 1: Overview
    # ========================================================================
    
    with tab1:
        st.header("Wirtschafts-Entwicklung")
        
        # GDP Chart
        st.subheader("💰 Bruttoinlandsprodukt (BIP)")
        gdp_data = [step['gdp'] for step in history['market']]
        steps = list(range(len(gdp_data)))
        
        fig_gdp = go.Figure()
        fig_gdp.add_trace(go.Scatter(
            x=steps,
            y=gdp_data,
            mode='lines',
            name='BIP',
            line=dict(color='#1f77b4', width=3),
            fill='tozeroy',
            fillcolor='rgba(31, 119, 180, 0.2)'
        ))
        fig_gdp.update_layout(
            xaxis_title='Quartal',
            yaxis_title='BIP (€)',
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_gdp, use_container_width=True)
        
        # Prices and Wages
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏷️ Preisentwicklung")
            fig_prices = go.Figure()
            
            for firm_name, firm_data in history['firms'].items():
                prices = [step['price'] for step in firm_data]
                fig_prices.add_trace(go.Scatter(
                    x=steps,
                    y=prices,
                    mode='lines',
                    name=firm_name,
                    line=dict(width=2)
                ))
            
            fig_prices.update_layout(
                xaxis_title='Quartal',
                yaxis_title='Preis (€)',
                hovermode='x unified',
                height=350
            )
            st.plotly_chart(fig_prices, use_container_width=True)
        
        with col2:
            st.subheader("💵 Lohnentwicklung")
            fig_wages = go.Figure()
            
            for firm_name, firm_data in history['firms'].items():
                wages = [step['wage'] for step in firm_data]
                fig_wages.add_trace(go.Scatter(
                    x=steps,
                    y=wages,
                    mode='lines',
                    name=firm_name,
                    line=dict(width=2)
                ))
            
            fig_wages.update_layout(
                xaxis_title='Quartal',
                yaxis_title='Lohn (€)',
                hovermode='x unified',
                height=350
            )
            st.plotly_chart(fig_wages, use_container_width=True)
        
        # Profit comparison
        st.subheader("📈 Profit-Vergleich")
        fig_profit = go.Figure()
        
        for firm_name, firm_data in history['firms'].items():
            profits = [step['profit'] for step in firm_data]
            fig_profit.add_trace(go.Scatter(
                x=steps,
                y=profits,
                mode='lines',
                name=firm_name,
                line=dict(width=2)
            ))
        
        fig_profit.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_profit.update_layout(
            xaxis_title='Quartal',
            yaxis_title='Profit (€)',
            hovermode='x unified',
            height=400
        )
        st.plotly_chart(fig_profit, use_container_width=True)
    
    # ========================================================================
    # TAB 2: Firm Details
    # ========================================================================
    
    with tab2:
        st.header("🏭 Firmen-Detailansicht")
        
        # Quarter slider
        selected_quarter = st.slider(
            "Wähle Quartal",
            min_value=0,
            max_value=len(history['market']) - 1,
            value=len(history['market']) - 1,
            help="Scrolle durch die Quartale"
        )
        
        # Firm selector
        selected_firm = st.selectbox(
            "Firma auswählen",
            list(history['firms'].keys())
        )
        
        # Current quarter metrics
        st.subheader(f"📅 Quartal {selected_quarter}")
        
        firm_state = history['firms'][selected_firm][selected_quarter]
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Preis", f"{firm_state['price']:.2f}€")
        with col2:
            st.metric("Lohn", f"{firm_state['wage']:.2f}€")
        with col3:
            st.metric("Profit", f"{firm_state['profit']:.2f}€")
        with col4:
            st.metric("Kapital", f"{firm_state['capital']:.2f}€")
        with col5:
            st.metric("Lagerbestand", f"{firm_state['inventory']:.0f}")
        
        st.markdown("---")
        
        # Time series for selected firm
        st.subheader(f"{selected_firm} - Zeitreihenanalyse")
        
        firm_history = history['firms'][selected_firm]
        df_firm = pd.DataFrame(firm_history)
        
        # Multi-metric chart
        fig_firm = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Preis & Lohn', 'Profit', 'Kapital', 'Lagerbestand')
        )
        
        # Price & Wage
        fig_firm.add_trace(
            go.Scatter(x=df_firm['step'], y=df_firm['price'], name='Preis', line=dict(color='blue')),
            row=1, col=1
        )
        fig_firm.add_trace(
            go.Scatter(x=df_firm['step'], y=df_firm['wage'], name='Lohn', line=dict(color='green')),
            row=1, col=1
        )
        
        # Profit
        fig_firm.add_trace(
            go.Scatter(x=df_firm['step'], y=df_firm['profit'], name='Profit', line=dict(color='purple')),
            row=1, col=2
        )
        fig_firm.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
        
        # Capital
        fig_firm.add_trace(
            go.Scatter(x=df_firm['step'], y=df_firm['capital'], name='Kapital', line=dict(color='orange')),
            row=2, col=1
        )
        
        # Inventory
        fig_firm.add_trace(
            go.Scatter(x=df_firm['step'], y=df_firm['inventory'], name='Lager', line=dict(color='red')),
            row=2, col=2
        )
        
        fig_firm.update_xaxes(title_text="Quartal")
        fig_firm.update_layout(height=600, showlegend=False)
        
        st.plotly_chart(fig_firm, use_container_width=True)
        
        # Data table
        with st.expander("📊 Daten-Tabelle anzeigen"):
            st.dataframe(df_firm, use_container_width=True)
    
    # ========================================================================
    # TAB 3: Households
    # ========================================================================
    
    with tab3:
        st.header("🏠 Haushalts-Analyse")
        
        st.info("💡 Zeigt eine Stichprobe von 10 Haushalten")
        
        # Select quarter
        hh_quarter = st.slider(
            "Quartal wählen",
            min_value=0,
            max_value=len(history['households']) - 1,
            value=len(history['households']) - 1,
            key="hh_quarter"
        )
        
        hh_data = history['households'][hh_quarter]['sample']
        df_hh = pd.DataFrame(hh_data)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_money = df_hh['money'].mean()
            st.metric("Durchschnittliches Vermögen", f"{avg_money:.2f}€")
        
        with col2:
            avg_income = df_hh['income'].mean()
            st.metric("Durchschnittseinkommen", f"{avg_income:.2f}€")
        
        with col3:
            # Count employers
            employer_counts = df_hh['employer'].value_counts()
            st.metric("Beliebtester Arbeitgeber", employer_counts.index[0] if len(employer_counts) > 0 else "N/A")
        
        # Employer distribution
        st.subheader("📈 Arbeitgeber-Verteilung")
        employer_counts = df_hh['employer'].value_counts()
        
        fig_employers = go.Figure(data=[
            go.Bar(
                x=employer_counts.index,
                y=employer_counts.values,
                text=employer_counts.values,
                textposition='auto',
            )
        ])
        fig_employers.update_layout(
            xaxis_title='Firma',
            yaxis_title='Anzahl Mitarbeiter',
            height=400
        )
        st.plotly_chart(fig_employers, use_container_width=True)
        
        # Household table
        st.subheader("📋 Haushalts-Daten")
        st.dataframe(df_hh, use_container_width=True)
    
    # ========================================================================
    # TAB 4: Data Export
    # ========================================================================
    
    with tab4:
        st.header("💾 Daten Export")
        
        st.markdown("""
        Exportiere die kompletten Simulations-Daten für weitere Analysen.
        """)
        
        # JSON Export
        st.subheader("🗄️ JSON Export")
        
        json_str = json.dumps(history, indent=2)
        
        st.download_button(
            label="⬇️ Download JSON",
            data=json_str,
            file_name=f"simulation_{metadata.get('model', 'unknown')}.json",
            mime="application/json"
        )
        
        # CSV Exports
        st.subheader("📄 CSV Exports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Market data CSV
            df_market = pd.DataFrame(history['market'])
            csv_market = df_market.to_csv(index=False)
            
            st.download_button(
                label="⬇️ Market Data CSV",
                data=csv_market,
                file_name="market_data.csv",
                mime="text/csv"
            )
        
        with col2:
            # Firm data CSV (combined)
            firm_dfs = []
            for firm_name, firm_data in history['firms'].items():
                df = pd.DataFrame(firm_data)
                df['firm'] = firm_name
                firm_dfs.append(df)
            
            df_firms = pd.concat(firm_dfs, ignore_index=True)
            csv_firms = df_firms.to_csv(index=False)
            
            st.download_button(
                label="⬇️ Firm Data CSV",
                data=csv_firms,
                file_name="firm_data.csv",
                mime="text/csv"
            )
        
        # Show preview
        with st.expander("👁️ Datenvorschau"):
            st.caption("Market Data")
            st.dataframe(df_market.head(10))
            
            st.caption("Firm Data")
            st.dataframe(df_firms.head(10))

else:
    # Initial state - no simulation run yet
    st.info("👈 Konfiguriere die Simulation in der Sidebar und klicke auf 'Simulation starten'")
    
    st.markdown("""
    ## 👋 Willkommen zum VWL-Simulation Viewer!
    
    Dieses Dashboard visualisiert **Multi-Agent Reinforcement Learning** in einer Volkswirtschafts-Simulation.
    
    ### 🎯 Features:
    
    - 🧠 **Trainierte KI-Modelle**: Lade und teste trainierte RL-Policies
    - ⚙️ **Konfigurierbar**: Passe Firmen, Haushalte und Start-Parameter an
    - 📊 **Visualisierungen**: Interaktive Charts für BIP, Preise, Löhne, Profite
    - 🔍 **Detailanalyse**: Untersuche einzelne Firmen und Haushalte
    - 💾 **Export**: Exportiere Daten als JSON/CSV
    
    ### 🚀 Los geht's:
    
    1. Wähle ein trainiertes Modell (oder nutze 'random' Policy)
    2. Konfiguriere Parameter (Firmen, Haushalte, Quartale)
    3. Klicke auf '🚀 Simulation starten'
    4. Analysiere die Ergebnisse!
    """)
    
    # Connection status
    st.markdown("---")
    st.subheader("🔌 Backend Status")
    
    try:
        response = requests.get(f"{BACKEND_URL}/api/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            st.success(f"✅ Backend verbunden ({BACKEND_URL})")
            st.info(f"📁 Verfügbare Modelle: {data.get('models_available', 0)}")
        else:
            st.error("❌ Backend antwortet nicht korrekt")
    except:
        st.error("❌ Backend nicht erreichbar")
        st.warning(f"🚨 Stelle sicher, dass das Backend läuft: `cd backend && uvicorn app:app --reload`")
