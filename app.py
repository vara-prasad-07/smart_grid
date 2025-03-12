import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def generate_user_data():
    st.sidebar.header("User Input: Hourly Consumption Data")
    
    hours = list(range(24))
    user_data = []
    
    # Default values that will be replaced by user input
    default_values = [0.5, 0.7, 0.6, 0.5, 0.5, 0.8, 1.2, 2.5, 3.0, 2.8, 2.5, 2.3, 
                      2.7, 2.6, 2.4, 2.5, 2.9, 3.5, 4.0, 3.8, 3.0, 2.0, 1.5, 0.8]
    
    for hour in hours:
        consumption = st.sidebar.number_input(
            f"Hour {hour}:00 - {hour+1}:00 (kWh)", 
            min_value=0.0, 
            max_value=10.0, 
            value=default_values[hour] if hour < len(default_values) else 1.0,
            step=0.1
        )
        user_data.append(consumption)
    
    timestamps = [datetime.now().replace(hour=hour, minute=0, second=0, microsecond=0) for hour in hours]
    
    # Use fixed voltage values instead of random
    voltage = [230.0 for _ in hours]
    
    # Calculate current based on user input for power consumption
    current = [round(user_data[i] * 1000 / voltage[i], 2) for i in range(24)]
    
    return pd.DataFrame({
        'Timestamp': timestamps,
        'Power_Consumption_kWh': user_data,
        'Voltage_V': voltage,
        'Current_A': current
    })

st.title("Smart Grid: Energy Consumption Analysis")

df = generate_user_data()
df['Hour'] = df['Timestamp'].dt.hour

st.subheader("📊 Hourly Power Consumption Trend")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(df['Timestamp'], df['Power_Consumption_kWh'], marker='o', linestyle='-', color='b', label='Power Consumption')
ax.set_xlabel('Time')
ax.set_ylabel('Power Consumption (kWh)')
ax.set_title('Hourly Power Consumption')
ax.legend()
ax.grid()
st.pyplot(fig)

# Identifying Peak Load Hours
peak_hours = df.groupby('Hour')['Power_Consumption_kWh'].mean()
peak_hour = peak_hours.idxmax()
st.subheader(f'🔴 Peak Load Hour: {peak_hour}:00 - {peak_hour+1}:00 with average {peak_hours.max():.2f} kWh')

# Cost Estimation
cost_per_kwh = st.sidebar.slider("Cost per kWh ($)", min_value=0.05, max_value=0.50, value=0.15, step=0.01)
total_cost = df['Power_Consumption_kWh'].sum() * cost_per_kwh
st.subheader(f'💰 Estimated Total Energy Cost: ${total_cost:.2f}')

# Predictive Load Forecasting
st.subheader("📈 Predictive Load Forecasting")
window_size = st.sidebar.slider("Forecast Window Size", min_value=1, max_value=6, value=3)
df['Predicted_Consumption_kWh'] = df['Power_Consumption_kWh'].rolling(window=window_size, min_periods=1).mean()

# Forecast future hours
forecast_hours = st.sidebar.slider("Forecast Future Hours", min_value=0, max_value=24, value=6)
if forecast_hours > 0:
    last_timestamp = df['Timestamp'].iloc[-1]
    future_timestamps = [last_timestamp + timedelta(hours=i+1) for i in range(forecast_hours)]
    
    # Use the last few hours to predict future consumption
    last_n_values = df['Power_Consumption_kWh'].tail(window_size).values
    predicted_values = []
    
    for _ in range(forecast_hours):
        next_value = np.mean(last_n_values[-window_size:])
        predicted_values.append(next_value)
        last_n_values = np.append(last_n_values, next_value)
    
    future_df = pd.DataFrame({
        'Timestamp': future_timestamps,
        'Predicted_Consumption_kWh': predicted_values
    })
    
    # Plot with forecasted values
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['Timestamp'], df['Power_Consumption_kWh'], label='Actual Consumption', color='blue')
    ax.plot(df['Timestamp'], df['Predicted_Consumption_kWh'], label='Fitted Consumption', linestyle='dashed', color='red')
    ax.plot(future_df['Timestamp'], future_df['Predicted_Consumption_kWh'], label='Forecasted Consumption', linestyle='dashed', color='green')
    ax.set_xlabel('Time')
    ax.set_ylabel('Power Consumption (kWh)')
    ax.set_title('Actual vs Predicted Power Consumption')
    ax.legend()
    ax.grid()
else:
    # Original plot without forecasting
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df['Timestamp'], df['Power_Consumption_kWh'], label='Actual Consumption', color='blue')
    ax.plot(df['Timestamp'], df['Predicted_Consumption_kWh'], label='Fitted Consumption', linestyle='dashed', color='red')
    ax.set_xlabel('Time')
    ax.set_ylabel('Power Consumption (kWh)')
    ax.set_title('Actual vs Predicted Power Consumption')
    ax.legend()
    ax.grid()

st.pyplot(fig)

# Energy Consumption Summary
st.subheader("📊 Energy Consumption Summary")
total_consumption = df['Power_Consumption_kWh'].sum()
avg_consumption = df['Power_Consumption_kWh'].mean()
min_consumption = df['Power_Consumption_kWh'].min()
max_consumption = df['Power_Consumption_kWh'].max()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total (kWh)", f"{total_consumption:.2f}")
col2.metric("Average (kWh)", f"{avg_consumption:.2f}")
col3.metric("Minimum (kWh)", f"{min_consumption:.2f}")
col4.metric("Maximum (kWh)", f"{max_consumption:.2f}")

# Time of Day Analysis
st.subheader("⏰ Time of Day Analysis")
time_periods = {
    "Night (0-6)": (0, 6),
    "Morning (6-12)": (6, 12),
    "Afternoon (12-18)": (12, 18),
    "Evening (18-24)": (18, 24)
}

period_consumption = {}
for period, (start, end) in time_periods.items():
    period_data = df[(df['Hour'] >= start) & (df['Hour'] < end)]
    period_consumption[period] = period_data['Power_Consumption_kWh'].sum()

fig, ax = plt.subplots(figsize=(10, 6))
periods = list(period_consumption.keys())
values = list(period_consumption.values())
ax.bar(periods, values, color=['purple', 'orange', 'green', 'navy'])
ax.set_xlabel('Time Period')
ax.set_ylabel('Total Consumption (kWh)')
ax.set_title('Consumption by Time of Day')
for i, v in enumerate(values):
    ax.text(i, v + 0.1, f"{v:.2f}", ha='center')
st.pyplot(fig)

st.subheader("📄 Raw Data Table")
st.dataframe(df[['Timestamp', 'Hour', 'Power_Consumption_kWh', 'Voltage_V', 'Current_A', 'Predicted_Consumption_kWh']])