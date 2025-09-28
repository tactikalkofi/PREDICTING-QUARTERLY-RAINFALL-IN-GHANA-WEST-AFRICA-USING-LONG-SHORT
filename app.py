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
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Statistical Models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression

# Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import math

# Set page configuration
st.set_page_config(
    page_title="Ghana Rainfall Prediction System",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

class AdvancedGhanaRainfallPredictor:
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
        """Generate realistic sample data with proper seasonality"""
        np.random.seed(42)  # For reproducibility
        years = list(range(1990, 2020))
        months = list(range(1, 13))
        data = []
        
        for station in self.stations.keys():
            zone = self.stations[station]['zone']
            zone_multiplier = {'Coastal': 1.2, 'Forest': 1.5, 'Savannah': 0.8}[zone]
            
            for year in years:
                for month in months:
                    # Enhanced seasonal patterns
                    if zone == 'Savannah':
                        # Unimodal pattern - peak in July-August
                        if month == 7:
                            base_seasonal = 180
                        elif month == 8:
                            base_seasonal = 170
                        elif month in [6, 9]:
                            base_seasonal = 120
                        elif month in [5, 10]:
                            base_seasonal = 70
                        elif month in [4, 11]:
                            base_seasonal = 30
                        else:
                            base_seasonal = 5
                    else:
                        # Bimodal pattern - peaks in May-July and September-October
                        if month == 6:
                            base_seasonal = 160
                        elif month in [5, 7]:
                            base_seasonal = 140
                        elif month == 9:
                            base_seasonal = 110
                        elif month == 10:
                            base_seasonal = 90
                        elif month in [4, 8]:
                            base_seasonal = 60
                        elif month in [3, 11]:
                            base_seasonal = 35
                        else:
                            base_seasonal = 15
                    
                    # Add inter-annual variability and trends
                    year_trend = np.sin((year - 1990) * 0.15) * 20
                    climate_cycle = np.sin((year - 1990) * 0.3) * 15  # ENSO-like cycle
                    noise = np.random.normal(0, 25)
                    
                    rainfall = max(0, (base_seasonal + year_trend + climate_cycle + noise) * zone_multiplier)
                    
                    data.append({
                        'Station': station,
                        'Year': year,
                        'Month': month,
                        'MonthlyRain_mm': round(rainfall, 1)
                    })
        
        return pd.DataFrame(data)
    
    def convert_to_quarterly(self, df):
        """Enhanced quarterly conversion with additional features"""
        df['Quarter'] = df['Month'].apply(lambda x: f"Q{(x-1)//3 + 1}")
        quarterly = df.groupby(['Station', 'Year', 'Quarter'])['MonthlyRain_mm'].sum().reset_index()
        quarterly.rename(columns={'MonthlyRain_mm': 'QuarterlyRain_mm'}, inplace=True)
        quarterly['Period'] = quarterly['Year'].astype(str) + '-' + quarterly['Quarter']
        
        # Add temporal features for enhanced modeling
        quarterly = quarterly.sort_values(['Station', 'Year', 'Quarter']).reset_index(drop=True)
        quarterly['Quarter_num'] = quarterly['Quarter'].str.extract('(\d)').astype(int)
        quarterly['Time_index'] = quarterly.groupby('Station').cumcount()
        
        return quarterly
    
    def create_enhanced_features(self, data, station_name):
        """Create sophisticated features for MLR"""
        enhanced_data = data.copy()
        
        # Temporal features
        enhanced_data['Year_norm'] = (enhanced_data['Year'] - enhanced_data['Year'].min()) / \
                                   (enhanced_data['Year'].max() - enhanced_data['Year'].min())
        
        # Lag features (previous quarters)
        for lag in [1, 2, 4]:
            enhanced_data[f'rainfall_lag_{lag}'] = enhanced_data['QuarterlyRain_mm'].shift(lag)
        
        # Moving averages
        for window in [2, 4, 8]:
            enhanced_data[f'rainfall_ma_{window}'] = enhanced_data['QuarterlyRain_mm'].rolling(
                window=window, min_periods=1).mean()
        
        # Seasonal features (trigonometric for cyclical nature)
        enhanced_data['sin_quarter'] = np.sin(2 * np.pi * enhanced_data['Quarter_num'] / 4)
        enhanced_data['cos_quarter'] = np.cos(2 * np.pi * enhanced_data['Quarter_num'] / 4)
        
        # Inter-annual cycles
        enhanced_data['sin_year'] = np.sin(2 * np.pi * enhanced_data['Year_norm'] * 3)
        enhanced_data['cos_year'] = np.cos(2 * np.pi * enhanced_data['Year_norm'] * 3)
        
        # Zone and location features
        zone = self.stations[station_name]['zone']
        enhanced_data['Zone_Coastal'] = 1 if zone == 'Coastal' else 0
        enhanced_data['Zone_Forest'] = 1 if zone == 'Forest' else 0
        enhanced_data['Zone_Savannah'] = 1 if zone == 'Savannah' else 0
        enhanced_data['Elevation'] = self.stations[station_name]['elevation']
        enhanced_data['Latitude'] = self.stations[station_name]['lat']
        
        # Quarter dummies
        quarter_dummies = pd.get_dummies(enhanced_data['Quarter'], prefix='Q')
        enhanced_data = pd.concat([enhanced_data, quarter_dummies], axis=1)
        
        return enhanced_data.dropna()
    
    def prepare_lstm_data(self, data, lookback=8, scaler_type='Standard'):
        """Enhanced data preparation without log transformation"""
        if scaler_type == 'Standard':
            scaler = StandardScaler()
        elif scaler_type == 'MinMax':
            scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            scaler = StandardScaler()  # Default to Standard for rainfall
        
        # Remove log transformation - use raw data for better seasonal pattern learning
        scaled_data = scaler.fit_transform(data.reshape(-1, 1))
        
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        return np.array(X), np.array(y), scaler
    
    def build_lstm_model(self, input_shape, config):
        """Build enhanced LSTM model with better capacity for seasonal patterns"""
        model = Sequential()
        
        # First LSTM layer - increased capacity
        model.add(LSTM(config['lstm1_units'], 
                      return_sequences=True, 
                      input_shape=input_shape,
                      dropout=config['dropout'],
                      recurrent_dropout=0.1))
        
        # Second LSTM layer
        model.add(LSTM(config['lstm2_units'], 
                      return_sequences=config.get('three_layers', False),
                      dropout=config['dropout'],
                      recurrent_dropout=0.1))
        
        # Optional third LSTM layer for complex patterns
        if config.get('three_layers', False):
            model.add(LSTM(config.get('lstm3_units', 32), 
                          return_sequences=False,
                          dropout=config['dropout'],
                          recurrent_dropout=0.1))
        
        # Dense layers with batch normalization
        model.add(Dense(config['dense_units'], activation='relu'))
        if config.get('use_batch_norm', True):
            from tensorflow.keras.layers import BatchNormalization
            model.add(BatchNormalization())
        model.add(Dropout(config['dropout']))
        
        # Optional second dense layer
        if config.get('two_dense_layers', False):
            model.add(Dense(config.get('dense2_units', 16), activation='relu'))
            model.add(Dropout(config['dropout'] * 0.5))
        
        model.add(Dense(1))
        
        # Adaptive learning rate
        optimizer = Adam(learning_rate=config['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def train_lstm(self, station_data, config):
        """Train enhanced LSTM model with improved parameters"""
        rainfall_values = station_data['QuarterlyRain_mm'].values
        
        # Prepare data with increased lookback
        X, y, scaler = self.prepare_lstm_data(rainfall_values, 
                                            lookback=config['lookback'],
                                            scaler_type=config['scaler'])
        
        if len(X) < 15:
            raise ValueError("Need at least 15 data points for LSTM training")
        
        # Enhanced train-validation-test split - more data for training
        train_size = int(config['train_split'] * len(X))
        val_size = int(config['val_split'] * len(X))
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size+val_size]
        y_val = y[train_size:train_size+val_size]
        X_test = X[train_size+val_size:]
        y_test = y[train_size+val_size:]
        
        # Reshape for LSTM
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        # Build model
        model = self.build_lstm_model((X_train.shape[1], 1), config)
        
        # Enhanced callbacks for better training
        from tensorflow.keras.callbacks import ReduceLROnPlateau
        
        early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=0)
        
        # Train model with more epochs
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if len(X_val) > 0 else None,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        # Make predictions
        predictions = model.predict(X_test, verbose=0)
        
        # Inverse transform (no need to reverse log transformation now)
        predictions = scaler.inverse_transform(predictions)
        y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
        
        return predictions.flatten(), y_test_actual.flatten(), history
    
    def train_arima(self, station_data, auto_arima=True):
        """Enhanced ARIMA with automatic parameter selection"""
        rainfall_values = station_data['QuarterlyRain_mm'].values
        
        # Split data
        train_size = int(0.8 * len(rainfall_values))
        train_data = rainfall_values[:train_size]
        test_data = rainfall_values[train_size:]
        
        if len(train_data) < 12:
            raise ValueError("Need at least 12 data points for ARIMA training")
        
        predictions = []
        best_order = (1, 1, 1)
        
        if auto_arima:
            # Find best parameters using AIC
            best_aic = np.inf
            for p in range(0, 3):
                for d in range(0, 2):
                    for q in range(0, 3):
                        try:
                            temp_model = ARIMA(train_data[:int(0.8*len(train_data))], order=(p,d,q))
                            temp_fitted = temp_model.fit()
                            if temp_fitted.aic < best_aic:
                                best_aic = temp_fitted.aic
                                best_order = (p,d,q)
                        except:
                            continue
        
        # Walk-forward validation
        for i in range(len(test_data)):
            current_train = np.concatenate([train_data, test_data[:i]]) if i > 0 else train_data
            
            try:
                # Try SARIMA first for seasonal patterns
                model = SARIMAX(current_train, 
                               order=best_order, 
                               seasonal_order=(1,1,1,4),
                               enforce_stationarity=False)
                fitted_model = model.fit(disp=False)
                forecast = fitted_model.forecast(steps=1)
                predictions.append(max(0, forecast.iloc[0]))
            except:
                try:
                    # Fallback to ARIMA
                    model = ARIMA(current_train, order=best_order)
                    fitted_model = model.fit()
                    forecast = fitted_model.forecast(steps=1)
                    predictions.append(max(0, forecast[0]))
                except:
                    # Ultimate fallback
                    predictions.append(np.mean(current_train[-4:]))
        
        return np.array(predictions), test_data, best_order
    
    def train_mlr(self, station_data, station_name, regularization='Ridge'):
        """Enhanced MLR with feature selection and regularization"""
        enhanced_data = self.create_enhanced_features(station_data, station_name)
        
        # Select features
        feature_cols = [col for col in enhanced_data.columns 
                       if col not in ['QuarterlyRain_mm', 'Station', 'Year', 'Quarter', 'Period', 'Time_index']]
        
        X = enhanced_data[feature_cols].fillna(0)
        y = enhanced_data['QuarterlyRain_mm']
        
        if len(X) < 15:
            raise ValueError("Need at least 15 data points for MLR training")
        
        # Split data
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Feature selection
        selector = SelectKBest(score_func=f_regression, k=min(10, len(feature_cols)))
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # Model selection based on regularization type
        if regularization == 'Ridge':
            model = Ridge(alpha=1.0)
        elif regularization == 'Lasso':
            model = Lasso(alpha=0.1)
        else:  # Polynomial
            model = Pipeline([
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('ridge', Ridge(alpha=1.0))
            ])
        
        model.fit(X_train_selected, y_train)
        predictions = model.predict(X_test_selected)
        predictions = np.maximum(0, predictions)
        
        return predictions, y_test.values, regularization, selector.get_feature_names_out()
    
    def calculate_enhanced_metrics(self, y_true, y_pred):
        """Calculate core performance metrics"""
        mae = mean_absolute_error(y_true, y_pred)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        return {
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'R²': round(r2, 3)
        }

def main():
    st.title("🌧️ Ghana Rainfall Prediction System")
    st.markdown("**Advanced comparative analysis using LSTM, ARIMA, and Multiple Linear Regression**")
    
    # Initialize predictor
    predictor = AdvancedGhanaRainfallPredictor()
    
    # Sidebar configuration
    st.sidebar.header("Model Configuration")
    
    # LSTM Configuration with improved defaults
    with st.sidebar.expander("LSTM Parameters", expanded=True):
        lstm_config = {
            'lstm1_units': st.slider("LSTM Layer 1 Units", 64, 256, 128),
            'lstm2_units': st.slider("LSTM Layer 2 Units", 32, 128, 64),
            'three_layers': st.checkbox("Use 3 LSTM Layers", value=False),
            'lstm3_units': st.slider("LSTM Layer 3 Units", 16, 64, 32) if st.session_state.get('three_layers', False) else 32,
            'dense_units': st.slider("Dense Layer Units", 16, 128, 64),
            'two_dense_layers': st.checkbox("Use 2 Dense Layers", value=False),
            'dense2_units': st.slider("Dense Layer 2 Units", 8, 64, 16) if st.session_state.get('two_dense_layers', False) else 16,
            'dropout': st.slider("Dropout Rate", 0.0, 0.3, 0.1),
            'learning_rate': st.selectbox("Learning Rate", [0.01, 0.001, 0.0001], index=0),
            'lookback': st.slider("Lookback Window (Quarters)", 6, 16, 10),
            'scaler': st.selectbox("Scaler Type", ["Standard", "MinMax"], index=0),
            'train_split': st.slider("Train Split", 0.7, 0.9, 0.85),
            'val_split': st.slider("Validation Split", 0.05, 0.2, 0.1),
            'epochs': st.slider("Max Epochs", 100, 300, 200),
            'batch_size': st.selectbox("Batch Size", [8, 16, 32], index=1),
            'use_batch_norm': st.checkbox("Use Batch Normalization", value=True)
        }
    
    # ARIMA Configuration
    with st.sidebar.expander("ARIMA Parameters", expanded=False):
        auto_arima = st.checkbox("Auto ARIMA Parameter Selection", value=True)
    
    # MLR Configuration
    with st.sidebar.expander("MLR Parameters", expanded=False):
        regularization = st.selectbox("Regularization", ["Ridge", "Lasso", "Polynomial"])
    
    # File upload with better error handling
    st.sidebar.subheader("Data Upload")
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV file", 
        type="csv",
        help="Columns: Station, Year, Month, MonthlyRain_mm"
    )
    
    # Load data with enhanced validation for actual dataset format
    if uploaded_file is not None:
        try:
            # Read the file
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✅ File loaded: {len(df)} records")
            
            # Show column information
            st.sidebar.write("**Columns found:**")
            st.sidebar.write(df.columns.tolist())
            
            # Check if this is the synop format (Station, Rainfall, Date)
            if 'Station' in df.columns and 'Rainfall' in df.columns and 'Date' in df.columns:
                st.sidebar.info("📊 Detected synop format data")
                
                # Convert Date column to datetime and extract Year and Month
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df['Year'] = df['Date'].dt.year
                df['Month'] = df['Date'].dt.month
                
                # Rename columns to match expected format
                df = df.rename(columns={'Rainfall': 'MonthlyRain_mm'})
                
                # Clean data
                df = df.dropna(subset=['Station', 'Year', 'Month', 'MonthlyRain_mm'])
                df['MonthlyRain_mm'] = pd.to_numeric(df['MonthlyRain_mm'], errors='coerce')
                df = df.dropna()
                
                # Filter valid months and non-negative rainfall
                df = df[(df['Month'] >= 1) & (df['Month'] <= 12)]
                df = df[df['MonthlyRain_mm'] >= 0]
                
                st.sidebar.success("✅ Data converted successfully")
                st.sidebar.write(f"**Date range:** {df['Year'].min()}-{df['Year'].max()}")
                
            else:
                # Check for standard format (Station, Year, Month, MonthlyRain_mm)
                required_cols = ['Station', 'Year', 'Month', 'MonthlyRain_mm']
                df_cols_lower = [col.lower() for col in df.columns]
                required_cols_lower = [col.lower() for col in required_cols]
                
                # Map columns if they exist with different cases
                column_mapping = {}
                for req_col in required_cols:
                    for df_col in df.columns:
                        if req_col.lower() == df_col.lower():
                            column_mapping[df_col] = req_col
                            break
                
                if len(column_mapping) == len(required_cols):
                    df = df.rename(columns=column_mapping)
                    st.sidebar.success("✅ Standard format detected")
                else:
                    missing_cols = [col for col in required_cols if col.lower() not in df_cols_lower]
                    st.sidebar.error(f"❌ Missing columns: {missing_cols}")
                    st.sidebar.write("Expected: Station, Year, Month, MonthlyRain_mm")
                    st.sidebar.write("OR: Station, Rainfall, Date")
                    st.stop()
                
                # Clean standard format data
                initial_length = len(df)
                df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
                df['Month'] = pd.to_numeric(df['Month'], errors='coerce')
                df['MonthlyRain_mm'] = pd.to_numeric(df['MonthlyRain_mm'], errors='coerce')
                
                # Remove rows with missing values
                df = df.dropna()
                
                # Validate ranges
                df = df[(df['Month'] >= 1) & (df['Month'] <= 12)]
                df = df[df['MonthlyRain_mm'] >= 0]
                
                if len(df) < initial_length:
                    st.sidebar.warning(f"⚠️ Cleaned data: {initial_length} → {len(df)} records")
            
            # Show final data preview
            st.sidebar.write("**Processed Data Preview:**")
            preview_df = df[['Station', 'Year', 'Month', 'MonthlyRain_mm']].head(3)
            st.sidebar.dataframe(preview_df, use_container_width=True)
            
            # Show available stations
            available_stations_preview = df['Station'].unique()[:5]
            st.sidebar.write(f"**Available Stations:** {', '.join(available_stations_preview)}")
            if len(df['Station'].unique()) > 5:
                st.sidebar.write(f"...and {len(df['Station'].unique()) - 5} more")
            
        except Exception as e:
            st.sidebar.error(f"❌ Error processing file: {str(e)}")
            st.sidebar.write("Expected formats:")
            st.sidebar.code("Format 1: Station,Year,Month,MonthlyRain_mm")
            st.sidebar.code("Format 2: Station,Rainfall,Date")
            st.stop()
    else:
        df = predictor.generate_sample_data()
        st.sidebar.info("📊 Using sample dataset")
    
    # Station selection
    available_stations = sorted(df['Station'].unique())
    selected_station = st.sidebar.selectbox("Select Station", available_stations)
    
    # Convert to quarterly
    quarterly_df = predictor.convert_to_quarterly(df)
    station_data = quarterly_df[quarterly_df['Station'] == selected_station].reset_index(drop=True)
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Stations", len(available_stations))
    with col3:
        st.metric("Years", f"{df['Year'].min()}-{df['Year'].max()}")
    with col4:
        st.metric("Quarterly Data", len(station_data))
    
    # Station info
    if selected_station in predictor.stations:
        st.subheader("Station Information")
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
    st.subheader("Data Analysis")
    
    # Time series with trend
    fig_ts = px.line(station_data, x='Period', y='QuarterlyRain_mm',
                     title=f'Quarterly Rainfall Time Series - {selected_station}')
    fig_ts.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig_ts, use_container_width=True)
    
    # Seasonal analysis
    seasonal_stats = station_data.groupby('Quarter')['QuarterlyRain_mm'].agg(['mean', 'std', 'count']).reset_index()
    fig_seasonal = px.bar(seasonal_stats, x='Quarter', y='mean', error_y='std',
                         title=f'Seasonal Pattern - {selected_station}')
    st.plotly_chart(fig_seasonal, use_container_width=True)
    
    # Model training
    st.subheader("Model Training & Comparison")
    
    if st.button("🚀 Train All Models", type="primary"):
        if len(station_data) < 20:
            st.error("Need at least 20 quarterly records for training")
            return
        
        results = {}
        progress_bar = st.progress(0)
        
        # Train LSTM
        try:
            with st.spinner("Training LSTM..."):
                lstm_pred, lstm_actual, lstm_history = predictor.train_lstm(station_data, lstm_config)
                results['LSTM'] = {
                    'predictions': lstm_pred,
                    'actual': lstm_actual,
                    'metrics': predictor.calculate_enhanced_metrics(lstm_actual, lstm_pred),
                    'history': lstm_history
                }
        except Exception as e:
            st.error(f"LSTM failed: {str(e)}")
        progress_bar.progress(33)
        
        # Train ARIMA
        try:
            with st.spinner("Training ARIMA..."):
                arima_pred, arima_actual, best_order = predictor.train_arima(station_data, auto_arima)
                results['ARIMA'] = {
                    'predictions': arima_pred,
                    'actual': arima_actual,
                    'metrics': predictor.calculate_enhanced_metrics(arima_actual, arima_pred),
                    'order': best_order
                }
        except Exception as e:
            st.error(f"ARIMA failed: {str(e)}")
        progress_bar.progress(66)
        
        # Train MLR
        try:
            with st.spinner("Training MLR..."):
                mlr_pred, mlr_actual, mlr_type, selected_features = predictor.train_mlr(
                    station_data, selected_station, regularization)
                results['MLR'] = {
                    'predictions': mlr_pred,
                    'actual': mlr_actual,
                    'metrics': predictor.calculate_enhanced_metrics(mlr_actual, mlr_pred),
                    'type': mlr_type,
                    'features': selected_features
                }
        except Exception as e:
            st.error(f"MLR failed: {str(e)}")
        progress_bar.progress(100)
        
        if not results:
            st.error("All models failed. Check your data and parameters.")
            return
        
        st.success(f"✅ Trained {len(results)} models successfully!")
        
        # Results section with tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance Metrics", "📈 Model Comparisons", "🎯 Individual Results", "📥 Download"])
        
        with tab1:
            st.subheader("Performance Comparison")
            metrics_df = pd.DataFrame({model: results[model]['metrics'] for model in results.keys()}).T
            st.dataframe(metrics_df, use_container_width=True)
            
            # Best model
            best_model = min(results.keys(), key=lambda x: results[x]['metrics']['RMSE'])
            st.success(f"**Best Model: {best_model}** (Lowest RMSE: {results[best_model]['metrics']['RMSE']})")
            
            # Metrics visualization
            fig_metrics = make_subplots(rows=1, cols=3, 
                                      subplot_titles=['Mean Absolute Error (MAE)', 'Root Mean Square Error (RMSE)', 'R² Score'])
            
            models = list(results.keys())
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            
            for i, metric in enumerate(['MAE', 'RMSE', 'R²']):
                values = [results[model]['metrics'][metric] for model in models]
                fig_metrics.add_trace(
                    go.Bar(x=models, y=values, name=metric, marker_color=colors[i], showlegend=False),
                    row=1, col=i+1
                )
            
            fig_metrics.update_layout(height=400, title_text="Model Performance Comparison")
            st.plotly_chart(fig_metrics, use_container_width=True)
        
        with tab2:
            st.subheader("Model Comparison Visualization")
            
            # Individual model line charts
            st.write("### Individual Model Performance")
            
            for model_name in results.keys():
                pred = results[model_name]['predictions']
                actual = results[model_name]['actual']
                
                # Individual line chart for each model
                fig_individual = go.Figure()
                
                # Add actual values
                fig_individual.add_trace(go.Scatter(
                    x=list(range(len(actual))),
                    y=actual,
                    mode='lines+markers',
                    name='Actual',
                    line=dict(color='red', width=3),
                    marker=dict(size=6)
                ))
                
                # Add predicted values
                fig_individual.add_trace(go.Scatter(
                    x=list(range(len(pred))),
                    y=pred,
                    mode='lines+markers',
                    name='Predicted',
                    line=dict(color='blue', width=3, dash='dash'),
                    marker=dict(size=6, symbol='diamond')
                ))
                
                fig_individual.update_layout(
                    title=f'{model_name} Model - Actual vs Predicted',
                    xaxis_title='Time Period (Quarters from Test Set)',
                    yaxis_title='Rainfall (mm)',
                    height=400,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_individual, use_container_width=True)
            
            st.write("### Combined Model Comparison")
            
            # Combined predictions plot
            fig_combined = go.Figure()
            
            # Add actual values
            first_model = list(results.keys())[0]
            actual_values = results[first_model]['actual']
            x_values = list(range(len(actual_values)))
            
            fig_combined.add_trace(go.Scatter(
                x=x_values, y=actual_values,
                mode='lines+markers', name='Actual',
                line=dict(color='black', width=3)
            ))
            
            # Add predictions for each model
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            for i, (model_name, color) in enumerate(zip(results.keys(), colors)):
                predictions = results[model_name]['predictions']
                fig_combined.add_trace(go.Scatter(
                    x=x_values, y=predictions,
                    mode='lines+markers', name=f'{model_name} Predicted',
                    line=dict(color=color, width=2, dash='dash')
                ))
            
            fig_combined.update_layout(
                title='All Models - Predictions vs Actual',
                xaxis_title='Time Period',
                yaxis_title='Rainfall (mm)',
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig_combined, use_container_width=True)
            
            # Residuals analysis
            st.write("### Residuals Analysis")
            fig_residuals = make_subplots(rows=1, cols=len(results), 
                                        subplot_titles=[f'{model} Residuals' for model in results.keys()])
            
            for i, model_name in enumerate(results.keys()):
                residuals = results[model_name]['actual'] - results[model_name]['predictions']
                fig_residuals.add_trace(
                    go.Scatter(x=list(range(len(residuals))), y=residuals,
                             mode='markers', name=f'{model_name}'),
                    row=1, col=i+1
                )
                fig_residuals.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=i+1)
            
            fig_residuals.update_layout(height=400, title_text="Residuals Analysis")
            st.plotly_chart(fig_residuals, use_container_width=True)
        
        with tab3:
            st.subheader("Individual Model Results")
            
            # Create separate tabs for each model
            model_tabs = st.tabs([f"{model} Analysis" for model in results.keys()])
            
            for i, model_name in enumerate(results.keys()):
                with model_tabs[i]:
                    st.write(f"## {model_name} Model Detailed Analysis")
                    
                    # Model-specific configuration details
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("### Model Configuration")
                        if model_name == 'LSTM' and 'history' in results[model_name]:
                            config_display = {
                                'LSTM Layer 1': f"{lstm_config['lstm1_units']} units",
                                'LSTM Layer 2': f"{lstm_config['lstm2_units']} units",
                                'Dense Layer': f"{lstm_config['dense_units']} units",
                                'Dropout': f"{lstm_config['dropout']:.1f}",
                                'Learning Rate': lstm_config['learning_rate'],
                                'Lookback Window': f"{lstm_config['lookback']} quarters",
                                'Scaler': lstm_config['scaler'],
                                'Epochs': lstm_config['epochs']
                            }
                            for key, value in config_display.items():
                                st.write(f"**{key}:** {value}")
                        
                        elif model_name == 'ARIMA' and 'order' in results[model_name]:
                            st.write(f"**ARIMA Order (p,d,q):** {results[model_name]['order']}")
                            st.write("**Type:** SARIMA with seasonal components")
                            st.write("**Seasonal Order:** (1,1,1,4) for quarterly data")
                        
                        elif model_name == 'MLR' and 'type' in results[model_name]:
                            st.write(f"**Regularization Type:** {results[model_name]['type']}")
                            st.write("**Feature Selection:** SelectKBest with f_regression")
                            if 'features' in results[model_name]:
                                st.write("**Top Selected Features:**")
                                features_list = results[model_name]['features'][:8]  # Show top 8
                                for feature in features_list:
                                    st.write(f"• {feature}")
                    
                    with col2:
                        st.write("### Performance Metrics")
                        metrics = results[model_name]['metrics']
                        
                        # Display metrics in a nice format
                        metric_cols = st.columns(3)
                        with metric_cols[0]:
                            st.metric("MAE", f"{metrics['MAE']} mm")
                        with metric_cols[1]:
                            st.metric("RMSE", f"{metrics['RMSE']} mm")
                        with metric_cols[2]:
                            st.metric("R²", f"{metrics['R²']}")
                    
                    # Training history for LSTM
                    if model_name == 'LSTM' and 'history' in results[model_name]:
                        st.write("### Training History")
                        history = results[model_name]['history'].history
                        
                        fig_history = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=['Loss During Training', 'MAE During Training']
                        )
                        
                        # Loss plot
                        fig_history.add_trace(
                            go.Scatter(y=history['loss'], name='Training Loss', line=dict(color='blue')),
                            row=1, col=1
                        )
                        if 'val_loss' in history:
                            fig_history.add_trace(
                                go.Scatter(y=history['val_loss'], name='Validation Loss', line=dict(color='red')),
                                row=1, col=1
                            )
                        
                        # MAE plot
                        if 'mae' in history:
                            fig_history.add_trace(
                                go.Scatter(y=history['mae'], name='Training MAE', line=dict(color='green')),
                                row=1, col=2
                            )
                        if 'val_mae' in history:
                            fig_history.add_trace(
                                go.Scatter(y=history['val_mae'], name='Validation MAE', line=dict(color='orange')),
                                row=1, col=2
                            )
                        
                        fig_history.update_xaxes(title_text="Epochs")
                        fig_history.update_yaxes(title_text="Loss", row=1, col=1)
                        fig_history.update_yaxes(title_text="MAE", row=1, col=2)
                        fig_history.update_layout(height=400, showlegend=True)
                        st.plotly_chart(fig_history, use_container_width=True)
                    
                    # Main prediction line chart
                    st.write("### Time Series Prediction Results")
                    pred = results[model_name]['predictions']
                    actual = results[model_name]['actual']
                    
                    fig_line = go.Figure()
                    
                    # Add actual values
                    fig_line.add_trace(go.Scatter(
                        x=list(range(len(actual))),
                        y=actual,
                        mode='lines+markers',
                        name='Actual Rainfall',
                        line=dict(color='red', width=3),
                        marker=dict(size=8)
                    ))
                    
                    # Add predicted values
                    fig_line.add_trace(go.Scatter(
                        x=list(range(len(pred))),
                        y=pred,
                        mode='lines+markers',
                        name=f'{model_name} Predicted',
                        line=dict(color='blue', width=3, dash='dash'),
                        marker=dict(size=6, symbol='diamond')
                    ))
                    
                    # Add error bands (optional)
                    errors = np.abs(actual - pred)
                    upper_bound = pred + errors
                    lower_bound = pred - errors
                    
                    fig_line.add_trace(go.Scatter(
                        x=list(range(len(pred))),
                        y=upper_bound,
                        fill=None,
                        mode='lines',
                        line_color='rgba(0,100,80,0)',
                        showlegend=False
                    ))
                    
                    fig_line.add_trace(go.Scatter(
                        x=list(range(len(pred))),
                        y=lower_bound,
                        fill='tonexty',
                        mode='lines',
                        line_color='rgba(0,100,80,0)',
                        name='Error Band',
                        fillcolor='rgba(135,206,235,0.3)'
                    ))
                    
                    fig_line.update_layout(
                        title=f'{model_name} Model - Actual vs Predicted Rainfall',
                        xaxis_title='Time Period (Quarters from Test Set)',
                        yaxis_title='Rainfall (mm)',
                        hovermode='x unified',
                        height=500,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig_line, use_container_width=True)
                    
                    # Prediction accuracy scatter plot
                    st.write("### Prediction Accuracy Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Scatter plot
                        fig_scatter = px.scatter(
                            x=actual, y=pred,
                            title=f'{model_name} - Predicted vs Actual',
                            labels={'x': 'Actual Rainfall (mm)', 'y': 'Predicted Rainfall (mm)'},
                            opacity=0.7
                        )
                        
                        # Add perfect prediction line
                        min_val = min(actual.min(), pred.min())
                        max_val = max(actual.max(), pred.max())
                        fig_scatter.add_trace(go.Scatter(
                            x=[min_val, max_val], 
                            y=[min_val, max_val],
                            mode='lines', 
                            name='Perfect Prediction',
                            line=dict(dash='dash', color='red', width=2)
                        ))
                        
                        # Add R² annotation
                        r_squared = metrics['R²']
                        fig_scatter.add_annotation(
                            x=0.05, y=0.95,
                            text=f"R² = {r_squared}",
                            showarrow=False,
                            font=dict(size=14, color="black"),
                            bgcolor="white",
                            bordercolor="black",
                            borderwidth=1,
                            xref="paper", yref="paper"
                        )
                        
                        st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    with col2:
                        # Residuals plot
                        residuals = actual - pred
                        
                        fig_residuals = go.Figure()
                        fig_residuals.add_trace(go.Scatter(
                            x=list(range(len(residuals))),
                            y=residuals,
                            mode='markers',
                            name='Residuals',
                            marker=dict(size=8, opacity=0.7)
                        ))
                        fig_residuals.add_hline(y=0, line_dash="dash", line_color="red", 
                                               annotation_text="Zero Error Line")
                        
                        fig_residuals.update_layout(
                            title=f'{model_name} - Residuals Plot',
                            xaxis_title='Time Period',
                            yaxis_title='Residuals (Actual - Predicted)',
                            height=400
                        )
                        
                        st.plotly_chart(fig_residuals, use_container_width=True)
                    
                    # Error statistics
                    st.write("### Error Analysis")
                    error_stats_cols = st.columns(4)
                    
                    with error_stats_cols[0]:
                        st.metric("Mean Error", f"{np.mean(residuals):.2f} mm")
                    with error_stats_cols[1]:
                        st.metric("Std Error", f"{np.std(residuals):.2f} mm")
                    with error_stats_cols[2]:
                        st.metric("Max Over-prediction", f"{np.min(residuals):.2f} mm")
                    with error_stats_cols[3]:
                        st.metric("Max Under-prediction", f"{np.max(residuals):.2f} mm")
        
        with tab4:
            st.subheader("Download Results")
            
            # Comprehensive results dataframe
            download_data = []
            for model in results.keys():
                for i, (actual, pred) in enumerate(zip(results[model]['actual'], results[model]['predictions'])):
                    download_data.append({
                        'Model': model,
                        'Time_Period': i,
                        'Actual_Rainfall_mm': round(actual, 2),
                        'Predicted_Rainfall_mm': round(pred, 2),
                        'Absolute_Error': round(abs(actual - pred), 2),
                        'Squared_Error': round((actual - pred)**2, 2),
                        'Percentage_Error': round(abs((actual - pred) / actual * 100), 2) if actual != 0 else 0
                    })
            
            download_df = pd.DataFrame(download_data)
            
            # Add summary statistics
            summary_stats = []
            for model in results.keys():
                metrics = results[model]['metrics']
                summary_stats.append({
                    'Model': model,
                    'MAE': metrics['MAE'],
                    'RMSE': metrics['RMSE'],
                    'R_squared': metrics['R²']
                })
            
            summary_df = pd.DataFrame(summary_stats)
            
            # Display preview
            st.write("**Predictions Data Preview:**")
            st.dataframe(download_df.head(10))
            
            st.write("**Model Summary:**")
            st.dataframe(summary_df)
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                predictions_csv = download_df.to_csv(index=False)
                st.download_button(
                    label="Download Detailed Predictions",
                    data=predictions_csv,
                    file_name=f"detailed_predictions_{selected_station}.csv",
                    mime="text/csv"
                )
            
            with col2:
                summary_csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="Download Model Summary",
                    data=summary_csv,
                    file_name=f"model_summary_{selected_station}.csv",
                    mime="text/csv"
                )
            
            # Generate analysis report
            st.subheader("Analysis Report")
            
            report = f"""
# Ghana Rainfall Prediction Analysis Report

## Dataset Information
- **Station:** {selected_station}
- **Climate Zone:** {predictor.stations[selected_station]['zone']}
- **Location:** {predictor.stations[selected_station]['lat']}°N, {predictor.stations[selected_station]['lon']}°E
- **Elevation:** {predictor.stations[selected_station]['elevation']}m
- **Data Period:** {station_data['Year'].min()}-{station_data['Year'].max()}
- **Total Quarters:** {len(station_data)}

## Model Performance Summary

"""
            
            for model in results.keys():
                metrics = results[model]['metrics']
                report += f"""
### {model} Model
- **Mean Absolute Error (MAE):** {metrics['MAE']} mm
- **Root Mean Square Error (RMSE):** {metrics['RMSE']} mm
- **R-squared (R²):** {metrics['R²']}

"""
            
            report += f"""
## Key Findings

1. **Best Performing Model:** {best_model} (lowest RMSE: {results[best_model]['metrics']['RMSE']} mm)
2. **Climate Zone Characteristics:** {predictor.stations[selected_station]['zone']} zone shows {'bimodal' if predictor.stations[selected_station]['zone'] != 'Savannah' else 'unimodal'} rainfall pattern
3. **Model Reliability:** R² values range from {min(results[m]['metrics']['R²'] for m in results.keys()):.3f} to {max(results[m]['metrics']['R²'] for m in results.keys()):.3f}

## Recommendations

- The {best_model} model is recommended for operational rainfall forecasting at {selected_station}
- Consider ensemble methods combining multiple models for improved reliability
- Regular model retraining is recommended as new data becomes available
- Uncertainty quantification should be included in operational forecasts

---
*Report generated using Ghana Rainfall Prediction System*
*Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
            
            st.markdown(report)
            
            # Download report
            st.download_button(
                label="Download Analysis Report",
                data=report,
                file_name=f"rainfall_analysis_report_{selected_station}.md",
                mime="text/markdown"
            )

if __name__ == "__main__":
    main()