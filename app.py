import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# Statistical Models
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import math

# Set page configuration
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
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

class GhanaRainfallPredictor:
    def __init__(self):
        self.stations = {
            'Accra': {'zone': 'Coastal', 'lat': 5.56, 'lon': -0.20, 'elevation': 68},
            'Takoradi': {'zone': 'Coastal', 'lat': 4.90, 'lon': -1.75, 'elevation': 4},
            'Kumasi': {'zone': 'Forest', 'lat': 6.68, 'lon': -1.62, 'elevation': 270},
            'Ho': {'zone': 'Forest', 'lat': 6.60, 'lon': 0.47, 'elevation': 158},
            'Tamale': {'zone': 'Savannah', 'lat': 9.40, 'lon': -0.84, 'elevation': 183},
            'Navrongo': {'zone': 'Savannah', 'lat': 10.90, 'lon': -1.09, 'elevation': 200}
        }
        
    def generate_sample_data(self):
        """Generate sample monthly rainfall data for demonstration"""
        years = list(range(1990, 2020))
        months = list(range(1, 13))
        data = []
        
        for station in self.stations.keys():
            zone = self.stations[station]['zone']
            zone_multiplier = {'Coastal': 1.2, 'Forest': 1.5, 'Savannah': 0.8}[zone]
            
            for year in years:
                for month in months:
                    # Seasonal patterns based on Ghana's climate
                    if zone == 'Savannah':
                        # Unimodal pattern - peak in July-August
                        if 6 <= month <= 8:
                            base_seasonal = 150 + (8 - abs(month - 7)) * 20
                        elif 4 <= month <= 10:
                            base_seasonal = 60
                        else:
                            base_seasonal = 10
                    else:
                        # Bimodal pattern - peaks in May-July and September-October
                        if 5 <= month <= 7:
                            base_seasonal = 120 + (6 - abs(month - 6)) * 30
                        elif 9 <= month <= 10:
                            base_seasonal = 80 + (9.5 - abs(month - 9.5)) * 20
                        elif 3 <= month <= 4:
                            base_seasonal = 40
                        else:
                            base_seasonal = 15
                    
                    year_trend = np.sin((year - 1990) * 0.1) * 15
                    noise = (np.random.random() - 0.5) * 30
                    rainfall = max(0, base_seasonal * zone_multiplier + year_trend + noise)
                    
                    data.append({
                        'Station': station,
                        'Year': year,
                        'Month': month,
                        'MonthlyRain_mm': round(rainfall, 1)
                    })
        
        return pd.DataFrame(data)
    
    def convert_to_quarterly(self, df):
        """Convert monthly data to quarterly"""
        df['Quarter'] = df['Month'].apply(lambda x: f"Q{(x-1)//3 + 1}")
        quarterly = df.groupby(['Station', 'Year', 'Quarter'])['MonthlyRain_mm'].sum().reset_index()
        quarterly.rename(columns={'MonthlyRain_mm': 'QuarterlyRain_mm'}, inplace=True)
        quarterly['Period'] = quarterly['Year'].astype(str) + '-' + quarterly['Quarter']
        return quarterly
    
    def prepare_lstm_data(self, data, lookback=4):
        """Prepare data for LSTM model"""
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data.reshape(-1, 1))
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y), scaler
    
    def build_lstm_model(self, input_shape):
        """Build LSTM model architecture"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        return model
    
    def train_lstm(self, station_data):
        """Train LSTM model"""
        rainfall_values = station_data['QuarterlyRain_mm'].values
        
        # Prepare data
        X, y, scaler = self.prepare_lstm_data(rainfall_values)
        
        # Split data
        train_size = int(0.7 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Reshape for LSTM
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # Build and train model
        model = self.build_lstm_model((X_train.shape[1], 1))
        
        with st.spinner("Training LSTM model..."):
            model.fit(X_train, y_train, epochs=1000, batch_size=64, verbose=0, validation_split=0.2)
        
        # Make predictions
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        return predictions.flatten(), y_test_actual.flatten()
    
    def train_arima(self, station_data):
        """Train ARIMA model"""
        rainfall_values = station_data['QuarterlyRain_mm'].values
        
        # Split data
        train_size = int(0.7 * len(rainfall_values))
        train_data = rainfall_values[:train_size]
        test_data = rainfall_values[train_size:]
        
        predictions = []
        
        with st.spinner("Training ARIMA model..."):
            for i in range(len(test_data)):
                # Use expanding window
                current_train = np.concatenate([train_data, test_data[:i]]) if i > 0 else train_data
                
                try:
                    model = ARIMA(current_train, order=(2, 1, 1))
                    fitted_model = model.fit()
                    forecast = fitted_model.forecast(steps=1)
                    predictions.append(forecast[0])
                except:
                    # Fallback to simple forecast
                    predictions.append(np.mean(current_train[-4:]))
        
        return np.array(predictions), test_data
    
    def train_mlr(self, station_data, station_name):
        """Train Multiple Linear Regression model"""
        # Create features
        data = station_data.copy()
        data['Year_norm'] = (data['Year'] - data['Year'].min()) / (data['Year'].max() - data['Year'].min())
        
        # One-hot encode quarters
        quarter_dummies = pd.get_dummies(data['Quarter'], prefix='Quarter')
        data = pd.concat([data, quarter_dummies], axis=1)
        
        # Zone-specific features
        zone = self.stations[station_name]['zone']
        data['Zone_Coastal'] = 1 if zone == 'Coastal' else 0
        data['Zone_Forest'] = 1 if zone == 'Forest' else 0
        data['Zone_Savannah'] = 1 if zone == 'Savannah' else 0
        
        # Select features
        feature_cols = ['Year_norm'] + [col for col in data.columns if col.startswith('Quarter_')] + \
                      [col for col in data.columns if col.startswith('Zone_')]
        
        X = data[feature_cols]
        y = data['QuarterlyRain_mm']
        
        # Split data
        train_size = int(0.7 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train model with polynomial features
        poly_model = Pipeline([
            ('poly', PolynomialFeatures(degree=2, include_bias=False)),
            ('linear', LinearRegression())
        ])
        
        with st.spinner("Training MLR model..."):
            poly_model.fit(X_train, y_train)
        
        predictions = poly_model.predict(X_test)
        
        return predictions, y_test.values
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate performance metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'R²': round(r2, 3)
        }

def main():
    st.markdown('<h1 class="main-header">🌧️ Ghana Quarterly Rainfall Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("**Comparative analysis using LSTM, ARIMA, and Multiple Linear Regression**")
    
    # Initialize predictor
    predictor = GhanaRainfallPredictor()
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Upload your rainfall dataset (CSV)", 
        type="csv",
        help="Expected format: Station, Year, Month, MonthlyRain_mm"
    )
    
    # Load data
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✅ Uploaded dataset with {len(df)} records")
            
            # Validate columns
            required_cols = ['Station', 'Year', 'Month', 'MonthlyRain_mm']
            if not all(col in df.columns for col in required_cols):
                st.sidebar.error(f"❌ Missing required columns: {required_cols}")
                st.stop()
                
        except Exception as e:
            st.sidebar.error(f"❌ Error loading file: {str(e)}")
            st.stop()
    else:
        df = predictor.generate_sample_data()
        st.sidebar.info("📊 Using sample dataset")
    
    # Station selection
    available_stations = df['Station'].unique()
    selected_station = st.sidebar.selectbox("Select Station", available_stations)
    
    # Convert to quarterly data
    quarterly_df = predictor.convert_to_quarterly(df)
    station_data = quarterly_df[quarterly_df['Station'] == selected_station].reset_index(drop=True)
    
    # Display dataset info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Stations", len(available_stations))
    with col3:
        st.metric("Years", f"{df['Year'].min()}-{df['Year'].max()}")
    with col4:
        st.metric("Selected Station", selected_station)
    
    # Station information
    if selected_station in predictor.stations:
        st.subheader("🗺️ Station Information")
        info = predictor.stations[selected_station]
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.info(f"**Zone:** {info['zone']}")
        with col2:
            st.info(f"**Latitude:** {info['lat']}°N")
        with col3:
            st.info(f"**Longitude:** {info['lon']}°E")
        with col4:
            st.info(f"**Elevation:** {info['elevation']}m")
    
    # Data visualization
    st.subheader("📈 Rainfall Patterns")
    
    # Time series plot
    fig_ts = px.line(station_data, x='Period', y='QuarterlyRain_mm', 
                     title=f'Quarterly Rainfall - {selected_station}',
                     labels={'QuarterlyRain_mm': 'Rainfall (mm)', 'Period': 'Year-Quarter'})
    fig_ts.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig_ts, use_container_width=True)
    
    # Seasonal patterns
    seasonal_data = station_data.groupby('Quarter')['QuarterlyRain_mm'].agg(['mean', 'std']).reset_index()
    fig_seasonal = px.bar(seasonal_data, x='Quarter', y='mean', error_y='std',
                         title=f'Seasonal Rainfall Pattern - {selected_station}',
                         labels={'mean': 'Average Rainfall (mm)'})
    st.plotly_chart(fig_seasonal, use_container_width=True)
    
    # Model training and comparison
    st.subheader("🤖 Model Training & Comparison")
    
    if st.button("🚀 Train All Models", type="primary"):
        if len(station_data) < 20:
            st.error("Insufficient data for training. Need at least 20 quarterly records.")
            return
        
        # Initialize results
        results = {}
        
        # Progress bar
        progress_bar = st.progress(0)
        
        try:
            # Train LSTM
            st.write("Training LSTM model...")
            progress_bar.progress(20)
            lstm_pred, lstm_actual = predictor.train_lstm(station_data)
            results['LSTM'] = {
                'predictions': lstm_pred,
                'actual': lstm_actual,
                'metrics': predictor.calculate_metrics(lstm_actual, lstm_pred)
            }
            progress_bar.progress(40)
            
            # Train ARIMA
            st.write("Training ARIMA model...")
            arima_pred, arima_actual = predictor.train_arima(station_data)
            results['ARIMA'] = {
                'predictions': arima_pred,
                'actual': arima_actual,
                'metrics': predictor.calculate_metrics(arima_actual, arima_pred)
            }
            progress_bar.progress(70)
            
            # Train MLR
            st.write("Training MLR model...")
            mlr_pred, mlr_actual = predictor.train_mlr(station_data, selected_station)
            results['MLR'] = {
                'predictions': mlr_pred,
                'actual': mlr_actual,
                'metrics': predictor.calculate_metrics(mlr_actual, mlr_pred)
            }
            progress_bar.progress(100)
            
            st.success("✅ All models trained successfully!")
            
            # Display metrics
            st.subheader("📊 Performance Metrics")
            
            metrics_df = pd.DataFrame({
                model: results[model]['metrics'] 
                for model in results.keys()
            }).T
            
            st.dataframe(metrics_df, use_container_width=True)
            
            # Metrics visualization
            fig_metrics = make_subplots(
                rows=1, cols=3,
                subplot_titles=['Mean Absolute Error (MAE)', 'Root Mean Square Error (RMSE)', 'R² Score']
            )
            
            models = list(results.keys())
            colors = ['#ff7f0e', '#2ca02c', '#d62728']
            
            for i, metric in enumerate(['MAE', 'RMSE', 'R²']):
                values = [results[model]['metrics'][metric] for model in models]
                fig_metrics.add_trace(
                    go.Bar(x=models, y=values, name=metric, marker_color=colors[i], showlegend=False),
                    row=1, col=i+1
                )
            
            fig_metrics.update_layout(height=400, title_text="Model Performance Comparison")
            st.plotly_chart(fig_metrics, use_container_width=True)
            
            # Predictions visualization
            st.subheader("🎯 Predictions vs Actual")
            
            # Create comparison plots for each model
            for model_name in results.keys():
                pred = results[model_name]['predictions']
                actual = results[model_name]['actual']
                
                # Create DataFrame for plotting
                plot_df = pd.DataFrame({
                    'Index': range(len(actual)),
                    'Actual': actual,
                    'Predicted': pred
                })
                
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(
                    x=plot_df['Index'], y=plot_df['Actual'],
                    mode='lines+markers', name='Actual',
                    line=dict(color='red', width=2)
                ))
                fig_pred.add_trace(go.Scatter(
                    x=plot_df['Index'], y=plot_df['Predicted'],
                    mode='lines+markers', name='Predicted',
                    line=dict(color='blue', width=2, dash='dash')
                ))
                
                fig_pred.update_layout(
                    title=f'{model_name} Model - Actual vs Predicted',
                    xaxis_title='Time Period',
                    yaxis_title='Rainfall (mm)',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_pred, use_container_width=True)
            
            # Best model recommendation
            best_model = min(results.keys(), key=lambda x: results[x]['metrics']['RMSE'])
            st.subheader("🏆 Model Recommendation")
            st.success(f"**Best performing model: {best_model}** (lowest RMSE: {results[best_model]['metrics']['RMSE']})")
            
            # Download results
            st.subheader("💾 Download Results")
            
            # Prepare results for download
            download_data = []
            for model in results.keys():
                for i, (actual, pred) in enumerate(zip(results[model]['actual'], results[model]['predictions'])):
                    download_data.append({
                        'Model': model,
                        'Period': i,
                        'Actual_Rainfall': actual,
                        'Predicted_Rainfall': pred,
                        'Error': abs(actual - pred)
                    })
            
            download_df = pd.DataFrame(download_data)
            csv = download_df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download Prediction Results",
                data=csv,
                file_name=f"rainfall_predictions_{selected_station}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"❌ Error during training: {str(e)}")
            st.info("This might be due to insufficient data or model complexity. Try with a larger dataset.")

if __name__ == "__main__":
    main()