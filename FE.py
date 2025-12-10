import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

# Page configuration
st.set_page_config(
    page_title="Project Rainier Cooling Optimizer",
    page_icon="❄️",
    layout="wide"
)

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_external_heat_factor(hour):
    """Simulates hotter temperatures during the day (10am-4pm)."""
    if 10 <= hour <= 16:
        return 1.2  # 20% more heat load during peak sun
    elif 17 <= hour <= 20:
        return 1.0  # Normal load evening
    else:
        return 0.8  # Cooler at night

def get_solar_availability(hour):
    """Simulates solar curve. 0 at night, max at noon."""
    if 6 <= hour <= 18:
        return np.sin(np.pi * (hour - 6) / 12)
    return 0.0

def cooling_objective(utility_power, battery_power, solar_power, 
                     current_time, ideal_temp, max_solar, cooling_load):
    """
    The Objective Function.
    Returns a 'loss' value (lower is better).
    """
    available_solar_pct = get_solar_availability(current_time)
    max_possible_solar = max_solar * available_solar_pct
    
    penalty = 0
    
    if solar_power > max_possible_solar:
        excess = solar_power - max_possible_solar
        penalty += excess * 100
        actual_solar_used = max_possible_solar
    else:
        actual_solar_used = solar_power
    
    total_power_input = utility_power + battery_power + actual_solar_used
    required_load = cooling_load * get_external_heat_factor(current_time)
    power_deficit = required_load - total_power_input
    predicted_temp = ideal_temp + (power_deficit * 0.05)
    
    temp_error = abs(predicted_temp - ideal_temp)
    cost_factor = (utility_power * 0.5) + (battery_power * 0.1) + (actual_solar_used * 0.0)
    
    total_loss = (temp_error * 1000) + cost_factor + penalty
    
    return total_loss

def bayesian_optimization(current_time, ideal_temp, max_solar, cooling_load, 
                         n_calls=50, n_initial=10):
    """
    Simplified Bayesian Optimization
    """
    best_loss = float('inf')
    best_params = None
    iterations_data = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Initial random samples
    for i in range(n_initial):
        params = {
            'utility': np.random.uniform(0, 1000),
            'battery': np.random.uniform(0, 500),
            'solar': np.random.uniform(0, 500)
        }
        
        loss = cooling_objective(
            params['utility'], params['battery'], params['solar'],
            current_time, ideal_temp, max_solar, cooling_load
        )
        
        iterations_data.append({
            'iteration': i + 1,
            'loss': loss,
            'utility': params['utility'],
            'battery': params['battery'],
            'solar': params['solar']
        })
        
        if loss < best_loss:
            best_loss = loss
            best_params = params
        
        progress_bar.progress((i + 1) / n_calls)
        status_text.text(f"Iteration {i + 1}/{n_calls} - Best Loss: {best_loss:.4f}")
        time.sleep(0.05)
    
    # Gaussian Process-inspired optimization
    for i in range(n_initial, n_calls):
        explore_weight = max(0.1, 1 - (i / n_calls))
        
        params = {
            'utility': best_params['utility'] + np.random.uniform(-200, 200) * explore_weight,
            'battery': best_params['battery'] + np.random.uniform(-100, 100) * explore_weight,
            'solar': best_params['solar'] + np.random.uniform(-100, 100) * explore_weight
        }
        
        # Clamp to bounds
        params['utility'] = np.clip(params['utility'], 0, 1000)
        params['battery'] = np.clip(params['battery'], 0, 500)
        params['solar'] = np.clip(params['solar'], 0, 500)
        
        loss = cooling_objective(
            params['utility'], params['battery'], params['solar'],
            current_time, ideal_temp, max_solar, cooling_load
        )
        
        iterations_data.append({
            'iteration': i + 1,
            'loss': loss,
            'utility': params['utility'],
            'battery': params['battery'],
            'solar': params['solar']
        })
        
        if loss < best_loss:
            best_loss = loss
            best_params = params
        
        progress_bar.progress((i + 1) / n_calls)
        status_text.text(f"Iteration {i + 1}/{n_calls} - Best Loss: {best_loss:.4f}")
        time.sleep(0.05)
    
    progress_bar.empty()
    status_text.empty()
    
    return best_loss, best_params, pd.DataFrame(iterations_data)

# ==========================================
# STREAMLIT APP
# ==========================================

st.title("❄️ Project AI_DataCenter Cooling Optimizer")
st.markdown("**Bayesian optimization for energy-efficient data center cooling**")
st.divider()

# Sidebar for parameters
with st.sidebar:
    st.header("⚙️ Configuration")
    
    current_time = st.slider(
        "Time of Day (hour)",
        min_value=0,
        max_value=23,
        value=14,
        help="Select the hour of the day (0-23)"
    )
    
    ideal_temp = st.number_input(
        "Ideal Temperature (°C)",
        min_value=10.0,
        max_value=30.0,
        value=20.0,
        step=0.5
    )
    
    max_solar = st.number_input(
        "Max Solar Capacity (kW)",
        min_value=0.0,
        max_value=1000.0,
        value=500.0,
        step=50.0
    )
    
    cooling_load = st.number_input(
        "Total Cooling Load (kW)",
        min_value=0.0,
        max_value=2000.0,
        value=1000.0,
        step=100.0
    )
    
    st.divider()
    
    n_calls = st.slider(
        "Optimization Iterations",
        min_value=20,
        max_value=100,
        value=50,
        step=10
    )

# Calculate context metrics
solar_available = get_solar_availability(current_time) * max_solar
heat_factor = get_external_heat_factor(current_time)
required_power = cooling_load * heat_factor

# Display context metrics
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="☀️ Solar Available",
        value=f"{solar_available:.2f} kW",
        delta=f"{(solar_available/max_solar)*100:.1f}% of max" if max_solar > 0 else "0%"
    )

with col2:
    st.metric(
        label="🌡️ Heat Factor",
        value=f"{heat_factor:.2f}x",
        delta="High" if heat_factor > 1.0 else "Low" if heat_factor < 1.0 else "Normal"
    )

with col3:
    st.metric(
        label="⚡ Required Power",
        value=f"{required_power:.2f} kW",
        delta=f"{((required_power/cooling_load)-1)*100:+.1f}% vs base"
    )

st.divider()

# Run optimization button
if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
    st.header("🔄 Optimization in Progress")
    
    with st.spinner("Running Bayesian Optimization..."):
        best_loss, best_params, iterations_df = bayesian_optimization(
            current_time, ideal_temp, max_solar, cooling_load, n_calls=n_calls
        )
    
    st.success("✅ Optimization Complete!")
    
    # Store results in session state
    st.session_state['best_loss'] = best_loss
    st.session_state['best_params'] = best_params
    st.session_state['iterations_df'] = iterations_df
    st.session_state['solar_available'] = solar_available

# Display results if available
if 'best_params' in st.session_state:
    st.header("📊 Optimal Solution")
    
    # Results metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🎯 Best Loss Score",
            value=f"{st.session_state['best_loss']:.4f}"
        )
    
    with col2:
        st.metric(
            label="🔌 Utility Power",
            value=f"{st.session_state['best_params']['utility']:.2f} kW"
        )
    
    with col3:
        st.metric(
            label="🔋 Battery Power",
            value=f"{st.session_state['best_params']['battery']:.2f} kW"
        )
    
    with col4:
        st.metric(
            label="☀️ Solar Power",
            value=f"{st.session_state['best_params']['solar']:.2f} kW"
        )
    
    # Validation message
    if st.session_state['best_params']['solar'] > st.session_state['solar_available']:
        st.warning("⚠️ Note: Optimizer attempted to exceed physical solar limits (corrected by penalty).")
    else:
        st.info("✓ Solar usage is within physical limits.")
    
    st.divider()
    
    # Visualization tabs
    tab1, tab2, tab3 = st.tabs(["📈 Loss Function", "📊 Energy Mix", "📋 Iteration Data"])
    
    with tab1:
        st.subheader("Optimization Progress")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=st.session_state['iterations_df']['iteration'],
            y=st.session_state['iterations_df']['loss'],
            mode='lines',
            name='Loss Function',
            line=dict(color='#4f46e5', width=2)
        ))
        
        fig.update_layout(
            xaxis_title="Iteration",
            yaxis_title="Loss",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Energy Mix Comparison")
        
        energy_data = pd.DataFrame({
            'Source': ['Utility', 'Battery', 'Solar'],
            'Power (kW)': [
                st.session_state['best_params']['utility'],
                st.session_state['best_params']['battery'],
                st.session_state['best_params']['solar']
            ],
            'Cost Factor': [
                st.session_state['best_params']['utility'] * 0.5,
                st.session_state['best_params']['battery'] * 0.1,
                0
            ]
        })
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Power Distribution', 'Cost Factor'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        fig.add_trace(
            go.Bar(x=energy_data['Source'], y=energy_data['Power (kW)'], 
                   marker_color=['#10b981', '#f59e0b', '#f97316'],
                   name='Power'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=energy_data['Source'], y=energy_data['Cost Factor'],
                   marker_color=['#ef4444', '#fbbf24', '#34d399'],
                   name='Cost'),
            row=1, col=2
        )
        
        fig.update_layout(height=400, showlegend=False)
        fig.update_yaxes(title_text="kW", row=1, col=1)
        fig.update_yaxes(title_text="Cost", row=1, col=2)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display data table
        st.dataframe(energy_data, use_container_width=True)
    
    with tab3:
        st.subheader("Detailed Iteration Data")
        
        # Show last 10 iterations
        st.write("**Last 10 Iterations:**")
        st.dataframe(
            st.session_state['iterations_df'].tail(10).style.format({
                'loss': '{:.4f}',
                'utility': '{:.2f}',
                'battery': '{:.2f}',
                'solar': '{:.2f}'
            }),
            use_container_width=True
        )
        
        # Download button
        csv = st.session_state['iterations_df'].to_csv(index=False)
        st.download_button(
            label="📥 Download Full Data (CSV)",
            data=csv,
            file_name="optimization_results.csv",
            mime="text/csv"
        )

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #6b7280; font-size: 0.875rem;'>
    Project AI_DATA_CENTER Cooling Optimizer | Powered by Bayesian Optimization
    </div>
    """,
    unsafe_allow_html=True
)
