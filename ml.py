import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import seaborn as sns

# Set the Streamlit theme (optional)
st.set_page_config(
    page_title="Feedstuff Forecasting",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background-color: #dfe6e9;
        color: black;
    }
    .stButton>button {
        background-color: #0984e3;
        color: white;
    }
    .stButton>button:hover {
        background-color: #74b9ff;
    }
    .stRadio>div>div>label {
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the cleaned dataset
file_path = r'C:\Users\coreyhughes\Assignments\CA1\ML\cleaned_feeder.csv'
feeder_clean = pd.read_csv(file_path)

# Convert the 'TLIST(M1)' column to a datetime format
feeder_clean['Date'] = pd.to_datetime(feeder_clean['TLIST(M1)'], format='%Y%m')
feeder_clean.set_index('Date', inplace=True)

# Drop rows with NaN values
feeder_clean.dropna(inplace=True)

# Sidebar - Select options
st.sidebar.header('🌾 Forecasting Options')

# Sidebar - Select feedstuff type
selected_feedstuff = st.sidebar.selectbox('Select Feedstuff', feeder_clean['Type of Feedstuff'].unique())

# Sidebar - Select chart type
chart_type = st.sidebar.radio('Select Chart Type', ['Line Chart', 'Bar Chart'])

# Filter data for the selected feedstuff
selected_data = feeder_clean[feeder_clean['Type of Feedstuff'] == selected_feedstuff]

# Show historical prices
st.title(f'📈 Historical Prices and Forecasting for {selected_feedstuff}')
st.markdown("### Visualizing the historical prices data")

if chart_type == 'Line Chart':
    st.line_chart(selected_data['VALUE'])
else:
    st.bar_chart(selected_data['VALUE'])

# Option to view forecasts
show_forecasts = st.sidebar.checkbox('Show Forecasts')

if show_forecasts:
    if len(selected_data) < 12:
        st.warning("Insufficient data available for forecasting.")
    else:
        try:
            # Train the model on all data up to end of 2022
            model = ARIMA(selected_data['VALUE'], order=(5, 1, 0))
            model_fit = model.fit()
            
            # Forecast for 2023 and 2024
            forecast_steps = 24  # 24 months for 2 years forecast
            forecast = model_fit.forecast(steps=forecast_steps)

            # Create a date range for the forecast
            forecast_dates = pd.date_range(start=selected_data.index[-1], periods=forecast_steps + 1, freq='M')[1:]

            # Plot the forecast
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.lineplot(x=selected_data.index, y=selected_data['VALUE'], ax=ax, label='Actual', color='blue')
            sns.lineplot(x=forecast_dates, y=forecast, ax=ax, label='Forecasted', color='red')
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (Euro per Tonne)')
            ax.set_title(f'Actual vs Forecasted Prices for {selected_feedstuff} (2023-2024)')
            ax.legend()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error in forecasting: {e}")
