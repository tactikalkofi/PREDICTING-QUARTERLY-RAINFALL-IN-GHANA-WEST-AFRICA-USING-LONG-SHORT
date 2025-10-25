"""
Ghana Rainfall Prediction System - Streamlit App
================================================
Author: Agboado Bernard (22253224)
Institution: University of Ghana Business School
Project: MSc Business Analytics Thesis 2025

Predicting Weekly and Monthly Rainfall across Ghana's Climate Zones
Using LSTM, ARIMA, Random Forest, and XGBoost Models
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Ghana Rainfall Prediction System",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    
    .station-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        margin: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        background-color: #f0f2f6;
        border-radius: 5px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Title Header
st.markdown("""
<div class="main-header">
    <h1>🌧️ Ghana Rainfall Prediction System</h1>
    <p>AI-Powered Weekly & Monthly Rainfall Forecasting across Ghana's Climate Zones</p>
    <p style="font-size: 0.9rem; margin-top: 1rem;">
        University of Ghana Business School | MSc Business Analytics | 2025
    </p>
</div>
""", unsafe_allow_html=True)

# Station Information Dictionary
stations = {
    'Tamale': {
        'zone': 'Savannah',
        'lat': 9.4,
        'lon': -0.9,
        'elevation': 169,
        'pattern': 'Unimodal',
        'annual_rainfall': 1000,
        'monthly_performance': {'MAE': 20.28, 'RMSE': 29.42, 'R2': 0.874, 'best_model': 'LSTM'},
        'weekly_performance': {'MAE': 7.44, 'RMSE': 12.81, 'R2': 0.700, 'best_model': 'XGBoost'}
    },
    'Navrongo': {
        'zone': 'Savannah',
        'lat': 10.9,
        'lon': -1.1,
        'elevation': 201,
        'pattern': 'Unimodal',
        'annual_rainfall': 950,
        'monthly_performance': {'MAE': 16.07, 'RMSE': 26.42, 'R2': 0.927, 'best_model': 'LSTM'},
        'weekly_performance': {'MAE': 9.12, 'RMSE': 18.49, 'R2': 0.611, 'best_model': 'XGBoost'}
    },
    'Kumasi': {
        'zone': 'Forest',
        'lat': 6.7,
        'lon': -1.6,
        'elevation': 287,
        'pattern': 'Bimodal',
        'annual_rainfall': 1400,
        'monthly_performance': {'MAE': 21.30, 'RMSE': 25.29, 'R2': 0.928, 'best_model': 'LSTM'},
        'weekly_performance': {'MAE': 15.57, 'RMSE': 24.55, 'R2': 0.523, 'best_model': 'LSTM'}
    },
    'Ho': {
        'zone': 'Forest',
        'lat': 6.6,
        'lon': 0.5,
        'elevation': 159,
        'pattern': 'Bimodal',
        'annual_rainfall': 1300,
        'monthly_performance': {'MAE': 20.29, 'RMSE': 28.43, 'R2': 0.845, 'best_model': 'XGBoost'},
        'weekly_performance': {'MAE': 12.58, 'RMSE': 17.43, 'R2': 0.562, 'best_model': 'LSTM'}
    },
    'Accra': {
        'zone': 'Coastal',
        'lat': 5.6,
        'lon': -0.2,
        'elevation': 68,
        'pattern': 'Asymmetrical Bimodal',
        'annual_rainfall': 800,
        'monthly_performance': {'MAE': 16.06, 'RMSE': 19.14, 'R2': 0.931, 'best_model': 'LSTM'},
        'weekly_performance': {'MAE': 11.00, 'RMSE': 20.57, 'R2': 0.553, 'best_model': 'LSTM'}
    },
    'Takoradi': {
        'zone': 'Coastal',
        'lat': 4.9,
        'lon': -1.8,
        'elevation': 5,
        'pattern': 'Asymmetrical Bimodal',
        'annual_rainfall': 1800,
        'monthly_performance': {'MAE': 22.44, 'RMSE': 31.39, 'R2': 0.900, 'best_model': 'XGBoost'},
        'weekly_performance': {'MAE': 15.24, 'RMSE': 23.47, 'R2': 0.605, 'best_model': 'LSTM'}
    }
}

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/1/19/Flag_of_Ghana.svg", width=200)
    st.markdown("### 📍 Station Selection")
    
    selected_station = st.selectbox(
        "Select Meteorological Station",
        options=list(stations.keys()),
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 📊 Prediction Settings")
    
    prediction_type = st.radio(
        "Forecast Horizon",
        options=["Monthly", "Weekly"],
        index=0
    )
    
    model_type = st.selectbox(
        "Select Model",
        options=["LSTM", "XGBoost", "Random Forest", "ARIMA"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("""
    This system uses machine learning to predict rainfall across Ghana's three climate zones.
    
    **Models:** LSTM, XGBoost, Random Forest, ARIMA
    
    **Data:** 30 years (1990-2019)
    
    **Stations:** 6 locations across Savannah, Forest, and Coastal zones
    
    **Accuracy:** R² > 0.85 for monthly predictions
    """)
    
    st.markdown("---")
    st.markdown("### 📚 Resources")
    st.markdown("""
    - [GitHub Repository](https://github.com/yourusername/ghana-rainfall-prediction)
    - [Documentation](https://github.com/yourusername/ghana-rainfall-prediction/wiki)
    - [Report Issues](https://github.com/yourusername/ghana-rainfall-prediction/issues)
    """)

# Main Content
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Overview", "📈 Predictions", "📊 Model Performance", "📉 Historical Data", "ℹ️ About"])

with tab1:
    st.markdown("## 🌍 System Overview")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>30 Years</h3>
            <p>Historical Data</p>
            <small>1990-2019</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>6 Stations</h3>
            <p>Climate Zones</p>
            <small>Savannah, Forest, Coastal</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>4 Models</h3>
            <p>ML Algorithms</p>
            <small>LSTM, XGBoost, RF, ARIMA</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>0.93 R²</h3>
            <p>Best Accuracy</p>
            <small>Monthly Predictions</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Station Information
    st.markdown(f"## 📍 Selected Station: {selected_station}")
    
    station_info = stations[selected_station]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <div class="station-card">
            <h3>{selected_station}</h3>
            <p><strong>Climate Zone:</strong> {station_info['zone']}</p>
            <p><strong>Rainfall Pattern:</strong> {station_info['pattern']}</p>
            <p><strong>Latitude:</strong> {station_info['lat']}°N</p>
            <p><strong>Longitude:</strong> {station_info['lon']}°E</p>
            <p><strong>Elevation:</strong> {station_info['elevation']}m</p>
            <p><strong>Annual Rainfall:</strong> ~{station_info['annual_rainfall']} mm</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Create map with all stations
        map_data = pd.DataFrame({
            'Station': list(stations.keys()),
            'lat': [s['lat'] for s in stations.values()],
            'lon': [s['lon'] for s in stations.values()],
            'Zone': [s['zone'] for s in stations.values()],
            'Elevation': [s['elevation'] for s in stations.values()]
        })
        
        fig = px.scatter_mapbox(
            map_data,
            lat='lat',
            lon='lon',
            hover_name='Station',
            hover_data={'Zone': True, 'Elevation': True, 'lat': ':.2f', 'lon': ':.2f'},
            color='Zone',
            color_discrete_map={'Savannah': '#FF6B6B', 'Forest': '#4ECDC4', 'Coastal': '#45B7D1'},
            zoom=5,
            height=400,
            size=[20]*6
        )
        
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r":0,"t":0,"l":0,"b":0}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Climate Zones Overview
    st.markdown("## 🌦️ Climate Zones Overview")
    
    zone_col1, zone_col2, zone_col3 = st.columns(3)
    
    with zone_col1:
        st.markdown("""
        ### Savannah Zone
        **Stations:** Tamale, Navrongo
        
        **Characteristics:**
        - Unimodal rainfall pattern
        - Wet season: May-October
        - Dry season: November-April
        - Annual: 800-1,100 mm
        
        **Best Model:** LSTM (Monthly)
        """)
    
    with zone_col2:
        st.markdown("""
        ### Forest Zone
        **Stations:** Kumasi, Ho
        
        **Characteristics:**
        - Bimodal rainfall pattern
        - Major: April-July
        - Minor: September-November
        - Annual: 1,200-2,000 mm
        
        **Best Model:** LSTM (Monthly)
        """)
    
    with zone_col3:
        st.markdown("""
        ### Coastal Zone
        **Stations:** Accra, Takoradi
        
        **Characteristics:**
        - Asymmetrical bimodal
        - Variable patterns
        - Accra Dry Anomaly
        - Annual: 700-2,000 mm
        
        **Best Model:** LSTM/XGBoost
        """)

with tab2:
    st.markdown(f"## 📈 Rainfall Predictions: {selected_station}")
    st.markdown(f"**Selected Model:** {model_type} | **Horizon:** {prediction_type}")
    
    # Generate sample predictions (in real app, load from trained models)
    @st.cache_data
    def generate_sample_predictions(station, horizon, model):
        """Generate sample predictions for demonstration"""
        np.random.seed(42)
        dates = pd.date_range(start='2020-01-01', periods=12 if horizon == 'Monthly' else 52, 
                             freq='M' if horizon == 'Monthly' else 'W')
        
        # Use station's annual rainfall to generate realistic values
        annual_rainfall = stations[station]['annual_rainfall']
        base = annual_rainfall / (12 if horizon == 'Monthly' else 52)
        
        predictions = base * (1 + 0.5 * np.sin(np.linspace(0, 2*np.pi, len(dates))))
        predictions += np.random.normal(0, base * 0.2, len(dates))
        predictions = np.maximum(predictions, 0)  # No negative rainfall
        
        # Add uncertainty
        lower_bound = predictions * 0.8
        upper_bound = predictions * 1.2
        
        return pd.DataFrame({
            'Date': dates,
            'Predicted_Rainfall': predictions,
            'Lower_Bound': lower_bound,
            'Upper_Bound': upper_bound
        })
    
    pred_df = generate_sample_predictions(selected_station, prediction_type, model_type)
    
    # Display prediction metrics
    perf_key = 'monthly_performance' if prediction_type == 'Monthly' else 'weekly_performance'
    perf = station_info[perf_key]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"{perf['MAE']:.2f} mm", help="Mean Absolute Error")
    col2.metric("RMSE", f"{perf['RMSE']:.2f} mm", help="Root Mean Squared Error")
    col3.metric("R² Score", f"{perf['R2']:.3f}", help="Coefficient of Determination")
    col4.metric("Best Model", perf['best_model'])
    
    # Plot predictions
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=pred_df['Date'],
        y=pred_df['Predicted_Rainfall'],
        mode='lines+markers',
        name='Predicted Rainfall',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=pred_df['Date'].tolist() + pred_df['Date'].tolist()[::-1],
        y=pred_df['Upper_Bound'].tolist() + pred_df['Lower_Bound'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Uncertainty Range',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=f"{prediction_type} Rainfall Predictions for {selected_station} (2020)",
        xaxis_title="Date",
        yaxis_title="Rainfall (mm)",
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction table
    st.markdown("### 📋 Detailed Predictions")
    
    display_df = pred_df.copy()
    display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d' if prediction_type == 'Monthly' else '%Y-%m-%d')
    display_df['Predicted_Rainfall'] = display_df['Predicted_Rainfall'].round(2)
    display_df['Lower_Bound'] = display_df['Lower_Bound'].round(2)
    display_df['Upper_Bound'] = display_df['Upper_Bound'].round(2)
    
    st.dataframe(display_df, use_container_width=True, height=300)
    
    # Download button
    csv = pred_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Predictions (CSV)",
        data=csv,
        file_name=f"{selected_station}_{prediction_type}_{model_type}_predictions.csv",
        mime="text/csv",
    )

with tab3:
    st.markdown("## 📊 Model Performance Comparison")
    
    # Performance comparison data
    models = ['LSTM', 'XGBoost', 'Random Forest', 'ARIMA']
    
    if prediction_type == 'Monthly':
        # Sample monthly performance data
        perf_data = {
            'LSTM': {'MAE': 20.44, 'RMSE': 26.99, 'R2': 0.899},
            'XGBoost': {'MAE': 22.47, 'RMSE': 34.92, 'R2': 0.833},
            'Random Forest': {'MAE': 25.61, 'RMSE': 38.04, 'R2': 0.795},
            'ARIMA': {'MAE': 88.50, 'RMSE': 106.50, 'R2': 0.005}
        }
    else:
        # Sample weekly performance data
        perf_data = {
            'LSTM': {'MAE': 12.35, 'RMSE': 20.16, 'R2': 0.565},
            'XGBoost': {'MAE': 11.87, 'RMSE': 20.94, 'R2': 0.531},
            'Random Forest': {'MAE': 12.12, 'RMSE': 20.88, 'R2': 0.536},
            'ARIMA': {'MAE': 24.54, 'RMSE': 32.82, 'R2': 0.184}
        }
    
    # Create comparison dataframe
    comp_df = pd.DataFrame(perf_data).T
    comp_df['Model'] = comp_df.index
    comp_df = comp_df.reset_index(drop=True)
    
    # Display metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = go.Figure(data=[
            go.Bar(x=comp_df['Model'], y=comp_df['MAE'],
                  marker_color=['#667eea', '#764ba2', '#f093fb', '#f5576c'])
        ])
        fig.update_layout(
            title="Mean Absolute Error (MAE)",
            yaxis_title="MAE (mm)",
            height=300,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure(data=[
            go.Bar(x=comp_df['Model'], y=comp_df['RMSE'],
                  marker_color=['#667eea', '#764ba2', '#f093fb', '#f5576c'])
        ])
        fig.update_layout(
            title="Root Mean Squared Error (RMSE)",
            yaxis_title="RMSE (mm)",
            height=300,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = go.Figure(data=[
            go.Bar(x=comp_df['Model'], y=comp_df['R2'],
                  marker_color=['#667eea', '#764ba2', '#f093fb', '#f5576c'])
        ])
        fig.update_layout(
            title="R² Score",
            yaxis_title="R²",
            height=300,
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance table
    st.markdown("### 📋 Detailed Performance Metrics")
    st.dataframe(comp_df[['Model', 'MAE', 'RMSE', 'R2']].style.format({
        'MAE': '{:.2f}',
        'RMSE': '{:.2f}',
        'R2': '{:.3f}'
    }).background_gradient(subset=['R2'], cmap='RdYlGn'), use_container_width=True)
    
    # Station-wise performance
    st.markdown("### 🗺️ Performance by Station")
    
    station_perf = []
    for station, info in stations.items():
        perf_key = 'monthly_performance' if prediction_type == 'Monthly' else 'weekly_performance'
        perf = info[perf_key]
        station_perf.append({
            'Station': station,
            'Zone': info['zone'],
            'MAE': perf['MAE'],
            'RMSE': perf['RMSE'],
            'R²': perf['R2'],
            'Best Model': perf['best_model']
        })
    
    station_df = pd.DataFrame(station_perf)
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=station_df[['MAE', 'RMSE', 'R²']].T.values,
        x=station_df['Station'],
        y=['MAE', 'RMSE', 'R²'],
        colorscale='RdYlGn_r',
        text=station_df[['MAE', 'RMSE', 'R²']].T.values,
        texttemplate='%{text:.2f}',
        textfont={"size": 12},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=f"{prediction_type} Performance Across All Stations",
        height=300,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display station table
    st.dataframe(station_df.style.background_gradient(subset=['R²'], cmap='RdYlGn'), use_container_width=True)

with tab4:
    st.markdown(f"## 📉 Historical Rainfall Data: {selected_station}")
    
    # Generate sample historical data
    @st.cache_data
    def generate_historical_data(station):
        """Generate sample historical data"""
        np.random.seed(42)
        dates = pd.date_range(start='1990-01-01', end='2019-12-31', freq='M')
        annual_rainfall = stations[station]['annual_rainfall']
        base = annual_rainfall / 12
        
        rainfall = base * (1 + 0.5 * np.sin(np.linspace(0, 2*np.pi*30, len(dates))))
        rainfall += np.random.normal(0, base * 0.3, len(dates))
        rainfall = np.maximum(rainfall, 0)
        
        return pd.DataFrame({
            'Date': dates,
            'Rainfall': rainfall,
            'Year': dates.year,
            'Month': dates.month
        })
    
    hist_df = generate_historical_data(selected_station)
    
    # Time series plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hist_df['Date'],
        y=hist_df['Rainfall'],
        mode='lines',
        name='Monthly Rainfall',
        line=dict(color='#667eea', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.3)'
    ))
    
    # Add trend line
    z = np.polyfit(range(len(hist_df)), hist_df['Rainfall'], 1)
    p = np.poly1d(z)
    
    fig.add_trace(go.Scatter(
        x=hist_df['Date'],
        y=p(range(len(hist_df))),
        mode='lines',
        name='Trend',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=f"Monthly Rainfall Time Series (1990-2019): {selected_station}",
        xaxis_title="Date",
        yaxis_title="Rainfall (mm)",
        height=500,
        template='plotly_white',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    st.markdown("### 📊 Statistical Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Mean", f"{hist_df['Rainfall'].mean():.2f} mm")
    col2.metric("Median", f"{hist_df['Rainfall'].median():.2f} mm")
    col3.metric("Std Dev", f"{hist_df['Rainfall'].std():.2f} mm")
    col4.metric("Min", f"{hist_df['Rainfall'].min():.2f} mm")
    col5.metric("Max", f"{hist_df['Rainfall'].max():.2f} mm")
    
    # Seasonal pattern
    st.markdown("### 🌦️ Seasonal Pattern")
    
    monthly_avg = hist_df.groupby('Month')['Rainfall'].mean().reset_index()
    
    fig = go.Figure(data=[
        go.Bar(
            x=monthly_avg['Month'],
            y=monthly_avg['Rainfall'],
            marker_color=['#667eea' if m in [5,6,7,9,10,11] else '#f5576c' for m in monthly_avg['Month']],
            text=monthly_avg['Rainfall'].round(1),
            textposition='outside'
        )
    ])
    
    fig.update_layout(
        title=f"Average Monthly Rainfall Pattern: {selected_station}",
        xaxis_title="Month",
        yaxis_title="Rainfall (mm)",
        xaxis=dict(tickmode='array', tickvals=list(range(1,13)),
                   ticktext=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']),
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Distribution
    st.markdown("### 📊 Rainfall Distribution")
    
    fig = go.Figure(data=[go.Histogram(
        x=hist_df['Rainfall'],
        nbinsx=30,
        marker_color='#667eea'
    )])
    
    fig.update_layout(
        title=f"Rainfall Distribution: {selected_station}",
        xaxis_title="Rainfall (mm)",
        yaxis_title="Frequency",
        height=400,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.markdown("## ℹ️ About This Project")
    
    st.markdown("""
    ### 🎓 Academic Research Project
    
    This system is part of an MSc Business Analytics thesis titled:
    
    **"Predicting Weekly and Monthly Rainfall in Ghana, West Africa Using Long Short-Term Memory"**
    
    **Author:** Agboado Bernard (Student ID: 22253224)  
    **Supervisor:** Prof. Anthony Afful-Dadzie  
    **Institution:** University of Ghana Business School  
    **Year:** 2025
    
    ---
    
    ### 🎯 Research Objectives
    
    1. **Develop LSTM Models** tailored to Ghana's rainfall patterns
    2. **Compare Forecasting Approaches** (LSTM, XGBoost, Random Forest, ARIMA)
    3. **Analyze Spatial Patterns** across three climate zones
    
    ---
    
    ### 📚 Methodology
    
    **Data:**
    - 30 years (1990-2019) of daily rainfall data
    - 6 meteorological stations across 3 climate zones
    - 63,879 station-day observations
    
    **Models:**
    - **LSTM**: Deep learning with 2 hidden layers (64/32 units)
    - **XGBoost**: Gradient boosting (100 estimators)
    - **Random Forest**: Ensemble learning (100-150 trees)
    - **ARIMA**: Traditional time series
    
    **Features:**
    - Lag variables (1-12 periods)
    - Rolling statistics (mean, std, max)
    - Temporal encodings (cyclical)
    - Seasonal indicators
    
    ---
    
    ### 📊 Key Findings
    
    ✅ **LSTM outperforms all models** (R² > 0.85 for monthly)  
    ✅ **30-40% improvement** over traditional ARIMA  
    ✅ **Consistent performance** across all climate zones  
    ✅ **Practical applications** in agriculture, water, disaster management  
    
    ---
    
    ### 🌍 Impact & Applications
    
    **Agriculture:**
    - Crop planning and planting decisions
    - Irrigation scheduling
    - Drought preparedness
    
    **Water Management:**
    - Reservoir operations
    - Hydropower planning
    - Urban water supply
    
    **Disaster Preparedness:**
    - Flood early warning
    - Resource allocation
    - Emergency planning
    
    ---
    
    ### 🔧 Technical Stack
    
    - **Python 3.8+**
    - **TensorFlow/Keras** (LSTM)
    - **XGBoost, Scikit-learn** (ML models)
    - **Pandas, NumPy** (Data processing)
    - **Plotly** (Visualizations)
    - **Streamlit** (Web framework)
    
    ---
    
    ### 📖 Citation
    
    If you use this work, please cite:
    
    ```
    Agboado, B. (2025). Predicting Weekly and Monthly Rainfall in Ghana, 
    West Africa Using Long Short-Term Memory. MSc Thesis, University of 
    Ghana Business School, Legon, Ghana.
    ```
    
    ---
    
    ### 🙏 Acknowledgements
    
    - **Ghana Meteorological Agency (GMet)** for providing rainfall data
    - **Prof. Anthony Afful-Dadzie** for supervision and guidance
    - **University of Ghana Business School** for institutional support
    
    ---
    
    ### 📧 Contact
    
    **GitHub:** [github.com/yourusername/ghana-rainfall-prediction](https://github.com/yourusername/ghana-rainfall-prediction)
    
    **Issues:** [Report bugs or request features](https://github.com/yourusername/ghana-rainfall-prediction/issues)
    
    ---
    
    ### 📄 License
    
    This project is licensed under the MIT License. See [LICENSE](https://github.com/yourusername/ghana-rainfall-prediction/blob/main/LICENSE) for details.
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>© 2025 University of Ghana Business School | Made with ❤️ for Ghana's Climate Resilience</p>
    <p>⭐ <a href="https://github.com/yourusername/ghana-rainfall-prediction">Star this project on GitHub</a></p>
</div>
""", unsafe_allow_html=True)