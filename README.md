# Rainfall Forecasting: LSTM vs SARIMA vs Regression
Streamlit app that loads rainfall from a URL (GitHub raw or any HTTP), runs EDA, and compares LSTM, SARIMA, Regression, and Seasonal-Naïve.

## Deploy on Streamlit Cloud
1. Fork/clone this repo.
2. Add your data:
   - Option A (public small file): put CSV at `data/synop_rr_1990_2019.csv` and set `DEFAULT_DATA_URL` in `streamlit_app.py` to your raw URL.
   - Option B (recommended): keep data in another repo or storage and set `DATA_URL` in Streamlit **Secrets**.
3. In Streamlit Cloud:
   - New app → pick this repo → `streamlit_app.py`.
   - Set Python version via `runtime.txt` (3.11).
   - In **App settings → Secrets**, add:
     ```
     DATA_URL = https://raw.githubusercontent.com/<user>/<repo>/main/data/synop_rr_1990_2019.csv
     ```
4. Deploy.
