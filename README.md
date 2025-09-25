# Ghana Quarterly Rainfall Prediction System

A comprehensive machine learning application for predicting quarterly rainfall in Ghana using LSTM, ARIMA, and Multiple Linear Regression models. This project is part of research on "Predicting Quarterly Rainfall in Ghana, West Africa using Long Short-Term Memory."

## Features

- **Multiple Model Comparison**: LSTM, ARIMA, and MLR implementations
- **Interactive Web Interface**: Built with Streamlit for easy use
- **File Upload Support**: Upload your own CSV datasets
- **Six Meteorological Stations**: Coverage across Ghana's three climate zones
- **Comprehensive Analysis**: Performance metrics, visualizations, and downloadable results
- **Real-time Training**: Watch models train with progress indicators

## Climate Zones Coverage

### Coastal Zone
- **Accra** (5.56°N, -0.20°E, 68m elevation)
- **Takoradi** (4.90°N, -1.75°E, 4m elevation)

### Forest Zone  
- **Kumasi** (6.68°N, -1.62°E, 270m elevation)
- **Ho** (6.60°N, 0.47°E, 158m elevation)

### Savannah Zone
- **Tamale** (9.40°N, -0.84°E, 183m elevation)
- **Navrongo** (10.90°N, -1.09°E, 200m elevation)

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ghana-rainfall-prediction.git
cd ghana-rainfall-prediction
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Usage

1. **Run the Streamlit application**
```bash
streamlit run rainfall_app.py
```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Upload your dataset** (optional) or use the sample data
   - Expected CSV format: `Station, Year, Month, MonthlyRain_mm`

4. **Select a meteorological station** from the dropdown

5. **Click "Train All Models"** to run the comparative analysis

6. **View results** including:
   - Performance metrics (MAE, RMSE, R²)
   - Interactive visualizations
   - Model comparisons
   - Prediction vs actual rainfall plots

7. **Download results** as CSV for further analysis

## Dataset Format

Your CSV file should have the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| Station | Meteorological station name | Accra |
| Year | Year of observation | 2019 |
| Month | Month (1-12) | 6 |
| MonthlyRain_mm | Monthly rainfall in millimeters | 142.5 |

## Model Architecture

### LSTM (Long Short-Term Memory)
- **Architecture**: 2-layer LSTM (64 and 32 units)
- **Dropout**: 20% to prevent overfitting
- **Optimizer**: Adam with learning rate 0.001
- **Loss Function**: Mean Squared Error (MSE)
- **Lookback Window**: 4 quarters (1 year)

### ARIMA (Autoregressive Integrated Moving Average)
- **Configuration**: ARIMA(2,1,1)
- **Approach**: Expanding window forecasting
- **Seasonal Handling**: Quarterly pattern recognition

### MLR (Multiple Linear Regression)
- **Features**: Year trends, quarterly patterns, zone-specific variables
- **Enhancement**: Polynomial features (degree 2)
- **Encoding**: One-hot encoding for categorical variables

## Performance Metrics

The application evaluates models using three standard metrics:

- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual values
- **RMSE (Root Mean Square Error)**: Square root of average squared differences (penalizes larger errors more)
- **R² (Coefficient of Determination)**: Proportion of variance explained by the model

## Research Context

This application implements the methodology described in the research thesis:

> "Predicting Quarterly Rainfall in Ghana, West Africa using Long Short-Term Memory"

The research addresses the critical need for accurate rainfall forecasting in Ghana for:
- Agricultural planning and food security
- Water resource management
- Disaster preparedness
- Hydroelectric power planning

## Sample Data

The application includes synthetic rainfall data based on Ghana's actual climate patterns:

- **Savannah Zone**: Unimodal rainfall pattern with peak in July-August
- **Forest/Coastal Zones**: Bimodal rainfall pattern with peaks in May-July and September-October
- **Time Period**: 30 years (1990-2019)
- **Frequency**: Monthly data aggregated to quarterly

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Research Citation

If you use this code in your research, please cite:

```
@thesis{agboado2025rainfall,
  title={Predicting Quarterly Rainfall in Ghana, West Africa using Long Short-Term Memory},
  author={Agboado, Bernard},
  year={2025},
  school={University of Ghana Business School}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Ghana Meteorological Agency (GMet) for data sources
- University of Ghana Business School
- Research supervisors and committee members

## Contact

**Author**: Agboado Bernard  
**Institution**: University of Ghana Business School  
**Email**: [your.email@example.com]

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed correctly
```bash
pip install --upgrade -r requirements.txt
```

2. **Memory Issues**: If training fails with large datasets, try:
   - Reducing the dataset size
   - Using a smaller LSTM architecture
   - Increasing virtual memory

3. **File Upload Issues**: Ensure your CSV file:
   - Has the correct column names
   - Contains numeric data in the rainfall column
   - Has no missing station names

### System Requirements

- **Python**: 3.8 or higher
- **RAM**: Minimum 4GB, recommended 8GB+
- **Storage**: At least 2GB free space
- **Internet**: Required for initial package installation

## Future Enhancements

- [ ] Integration with real-time weather APIs
- [ ] Support for additional African countries
- [ ] Advanced ensemble methods
- [ ] Mobile-responsive design
- [ ] Multi-language support
- [ ] Automated model retraining