# Rainfall Forecasting & Model Comparison (Upload CSV)
# LSTM (optional), SARIMA, Seasonal-Dummy Regression, Seasonal-Naïve
# Python 3.11 compatible

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ------------------------
# Page
# ------------------------
st.set_page_config(page_title="Rainfall: LSTM / SARIMA / Regression", page_icon="🌧️", layout="wide")

# Defaults
DEFAULT_LOOKBACK = 8          # periods (Q or M)
DEFAULT_TEST_PERIODS = 8      # last N periods for test
DEFAULT_FH_Q = 4              # forecast horizon quarters
DEFAULT_FH_M = 12             # forecast horizon months

RAIN_COL_CANDIDATES = ["rainfall_mm","rain_mm","rainfall","rain","rr","precip","precip_mm","prcp","prcp_mm"]

# ------------------------
# TensorFlow (optional, lazy)
# ------------------------
def tf_available():
    try:
        import tensorflow as tf  # noqa
        return True
    except Exception:
        return False

def build_lstm_model(units1, units2, dropout):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    model = Sequential([
        LSTM(units1, return_sequences=True, input_shape=(None, 1)),
        Dropout(dropout),
        LSTM(units2),
        Dropout(dropout),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def get_early_stopping(patience):
    from tensorflow.keras.callbacks import EarlyStopping
    return EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)

# ------------------------
# Sidebar
# ------------------------
with st.sidebar:
    st.title("🌧️ Rainfall Forecasting")
    st.caption("Upload a CSV, explore EDA, and compare models.")

    st.markdown("---")
    st.subheader("1) Upload CSV")
    st.write("Accepted layouts:")
    st.code("Long: date, station, rainfall_mm|rain|rr|precip\nWide: date|year,month[,day], <Station1>, <Station2>, ...")
    uploaded = st.file_uploader("Upload rainfall CSV", type=["csv"])

    st.markdown("---")
    st.subheader("2) Frequency & Aggregation")
    freq_mode = st.radio("Model frequency", ["Quarterly","Monthly"], index=0, horizontal=True)
    rule = "Q" if freq_mode == "Quarterly" else "M"
    seasonality = 4 if rule == "Q" else 12
    agg_method = st.selectbox("Aggregate to model frequency", ["sum","mean"], index=0)

    st.markdown("---")
    st.subheader("3) Split & Horizon")
    lookback = st.number_input("LSTM lookback (periods)", 4, 60, DEFAULT_LOOKBACK, step=1)
    test_periods = st.number_input("Test size (last N periods)", 4, 60, DEFAULT_TEST_PERIODS, step=1)
    default_h = DEFAULT_FH_Q if rule == "Q" else DEFAULT_FH_M
    forecast_h = st.number_input("Forecast horizon (periods)", 1, 24, default_h, step=1)

    st.markdown("---")
    st.subheader("4) Models")
    use_lstm = st.checkbox("LSTM (deep learning)", value=True)
    if use_lstm and not tf_available():
        st.warning('TensorFlow not found. App will still run. To enable LSTM locally/Cloud: pip install "tensorflow==2.18.1"')

    use_sarima = st.checkbox("SARIMA (statistical)", value=True)
    use_reg = st.checkbox("Seasonal-Dummy Regression", value=True)
    use_baseline = st.checkbox(f"Seasonal-Naïve (t-{seasonality}) baseline", value=True)

    if use_lstm:
        st.markdown("**LSTM Hyperparameters**")
        lstm_units_1 = st.slider("LSTM Layer 1 units", 16, 256, 64, step=16)
        lstm_units_2 = st.slider("LSTM Layer 2 units", 8, 128, 32, step=8)
        dropout_rate = st.slider("Dropout", 0.0, 0.6, 0.2, step=0.05)
        batch_size = st.select_slider("Batch size", options=[16,32,64,128], value=32)
        epochs = st.select_slider("Epochs (max)", options=[20,30,40,50,75,100], value=50)
        patience = st.slider("Early stopping patience", 3, 15, 8, step=1)

    if use_sarima:
        st.markdown("**SARIMA orders**")
        p = st.number_input("p", 0, 3, 1)
        d = st.number_input("d", 0, 2, 0)
        q = st.number_input("q", 0, 3, 1)
        P = st.number_input("P (seasonal)", 0, 3, 1)
        D = st.number_input("D (seasonal)", 0, 2, 0)
        Q = st.number_input("Q (seasonal)", 0, 3, 1)

# ------------------------
# Helpers
# ------------------------
def _end_of_month_date(year, month, day=None):
    if day is None or pd.isna(day):
        return pd.Period(f"{int(year)}-{int(month):02d}", freq="M").to_timestamp(how="end")
    return pd.Timestamp(year=int(year), month=int(month), day=int(day))

def coerce_tidy(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Build/parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    else:
        has_y, has_m, has_d = ("year" in df.columns), ("month" in df.columns), ("day" in df.columns)
        if has_y and has_m:
            y = pd.to_numeric(df["year"], errors="coerce")
            m = pd.to_numeric(df["month"], errors="coerce")
            if has_d:
                d = pd.to_numeric(df["day"], errors="coerce")
                df["date"] = pd.to_datetime(dict(year=y, month=m, day=d), errors="coerce")
            else:
                df["date"] = pd.to_datetime([_end_of_month_date(yy, mm) for yy, mm in zip(y, m)], errors="coerce")
        else:
            alt = [c for c in df.columns if c in ("obs_date","dt","time","timestamp")]
            df["date"] = pd.to_datetime(df[alt[0]], errors="coerce") if alt else pd.NaT

    has_station = "station" in df.columns
    rain_col = next((c for c in RAIN_COL_CANDIDATES if c in df.columns), None)

    # Long format
    if has_station and (rain_col is not None):
        if rain_col != "rainfall_mm":
            df = df.rename(columns={rain_col: "rainfall_mm"})
        df["rainfall_mm"] = pd.to_numeric(df["rainfall_mm"], errors="coerce")
        out = df.dropna(subset=["date","station","rainfall_mm"]).copy()
        out["station"] = out["station"].astype(str).str.strip()
        return out.sort_values(["station","date"]).reset_index(drop=True)

    # Wide → Long
    meta = {"date","year","month","day","obs_date","dt","time","timestamp","station"}
    cand = [c for c in df.columns if c not in meta]
    if cand:
        if len(cand) == 1 and (rain_col is not None or not has_station):
            c = cand[0]
            if rain_col is None: rain_col = c
            tmp = df[["date", c]].rename(columns={c:"rainfall_mm"})
            tmp["rainfall_mm"] = pd.to_numeric(tmp["rainfall_mm"], errors="coerce")
            tmp["station"] = "Unknown"
            return tmp.dropna(subset=["date","rainfall_mm"]).sort_values(["station","date"]).reset_index(drop=True)

        station_cols = []
        for c in cand:
            if c in RAIN_COL_CANDIDATES or c == "station":
                continue
            if not df[c].isna().all():
                station_cols.append(c)
        if station_cols:
            keep = ["date"] if "date" in df.columns else []
            long_df = df[keep + station_cols].melt(id_vars=keep, var_name="station", value_name="rainfall_mm")
            long_df["rainfall_mm"] = pd.to_numeric(long_df["rainfall_mm"], errors="coerce")
            long_df = long_df.dropna(subset=["date","rainfall_mm"])
            long_df["station"] = long_df["station"].astype(str).str.strip()
            return long_df.sort_values(["station","date"]).reset_index(drop=True)

    # Long but unknown rainfall column → first numeric
    if has_station and rain_col is None:
        numeric_cols = [c for c in df.columns if c not in ("date","station") and pd.api.types.is_numeric_dtype(df[c])]
        if not numeric_cols:
            for c in df.columns:
                if c not in ("date","station"):
                    df[c] = pd.to_numeric(df[c], errors="ignore")
            numeric_cols = [c for c in df.columns if c not in ("date","station") and pd.api.types.is_numeric_dtype(df[c])]
        if numeric_cols:
            picked = numeric_cols[0]
            tmp = df[["date","station",picked]].rename(columns={picked:"rainfall_mm"})
            tmp["rainfall_mm"] = pd.to_numeric(tmp["rainfall_mm"], errors="coerce")
            tmp = tmp.dropna(subset=["date","station","rainfall_mm"])
            tmp["station"] = tmp["station"].astype(str).str.strip()
            return tmp.sort_values(["station","date"]).reset_index(drop=True)

    raise ValueError("Could not infer columns. Provide Long or Wide format as described.")

def to_freq(df: pd.DataFrame, station: str, agg: str, rule: str) -> pd.Series:
    s = df[df["station"] == station].set_index("date")["rainfall_mm"].sort_index()
    out = s.resample(rule).sum() if agg == "sum" else s.resample(rule).mean()
    return out.asfreq(rule).dropna()

def make_sequences(series_1d: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(series_1d)):
        X.append(series_1d[i - lookback:i])
        y.append(series_1d[i])
    X = np.array(X); y = np.array(y)
    return X.reshape((X.shape[0], X.shape[1], 1)), y

def seasonal_naive(y_true: np.ndarray, season: int):
    if len(y_true) <= season:
        return np.array([])
    return y_true[:-season]

def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    denom = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - np.sum((y_true - y_pred)**2) / denom if denom > 0 else 0.0
    return mae, rmse, r2

def recursive_forecast(model, last_window, horizon, scaler):
    preds = []
    window = last_window.copy()
    for _ in range(horizon):
        x = window.reshape(1, window.shape[0], 1)
        y_hat = model.predict(x, verbose=0)[0][0]
        preds.append(y_hat)
        window = np.roll(window, -1)
        window[-1] = y_hat
    preds = np.array(preds).reshape(-1, 1)
    return scaler.inverse_transform(preds).flatten()

# ------------------------
# Main
# ------------------------
st.title("🌧️ Rainfall — LSTM / SARIMA / Regression (Upload CSV)")
st.caption("Upload data, explore EDA, train multiple models on the same split, compare metrics, and forecast ahead.")

# Sample data generator (optional)
with st.expander("No data yet? Generate a tiny sample CSV"):
    if st.button("Create & download sample CSV"):
        dates = pd.date_range("2010-01-01", "2019-12-31", freq="D")
        def synth(station):
            rng = np.random.default_rng(7 if station=="Accra" else 9)
            base = 4*np.sin(2*np.pi*dates.dayofyear/365.25) + (2 if station=="Accra" else 2.5)
            rain = np.maximum(0, (base + rng.normal(0, 1, len(dates))) * 10)
            return pd.DataFrame({"date": dates, "station": station, "rainfall_mm": rain})
        demo = pd.concat([synth("Accra"), synth("Tamale")]).reset_index(drop=True)
        st.download_button("Download sample.csv", demo.to_csv(index=False), file_name="sample_rainfall.csv", mime="text/csv")

if uploaded is None:
    st.info("Upload a CSV to begin.")
    st.stop()

# Load & coerce
try:
    raw_df = pd.read_csv(uploaded)
    df = coerce_tidy(raw_df)
except Exception as e:
    st.error(f"Could not parse your CSV: {e}")
    st.stop()

# Station pick
stations = sorted(df["station"].astype(str).unique().tolist())
station = st.selectbox("Select station", stations, index=0)

# ------------------------
# EDA
# ------------------------
st.subheader("Exploratory Data Analysis (EDA)")
with st.expander("Dataset overview"):
    cov = (df.groupby("station")
           .agg(first_date=("date","min"), last_date=("date","max"),
                n_rows=("rainfall_mm","size"), total_mm=("rainfall_mm","sum"),
                mean_mm=("rainfall_mm","mean"))
           .sort_values("total_mm", ascending=False))
    st.dataframe(cov, use_container_width=True)

s_df = df[df["station"] == station].copy().sort_values("date")
if s_df.empty:
    st.warning("No rows for this station.")
    st.stop()

eda_agg = st.radio("EDA aggregation (for plots only)", ["Monthly","Quarterly"], index=0, horizontal=True)
eda_rule = "M" if eda_agg == "Monthly" else "Q"
agg_func = "sum" if agg_method == "sum" else "mean"
ts = (s_df.set_index("date")["rainfall_mm"].resample(eda_rule).sum().dropna()
      if agg_func=="sum"
      else s_df.set_index("date")["rainfall_mm"].resample(eda_rule).mean().dropna())
ts.name = "rainfall_mm"

tab1, tab2, tab3, tab4 = st.tabs(["📈 Time series","📊 Seasonality","🗺️ Year–Month Heatmap","📦 Distribution"])
with tab1:
    fig_ts = px.line(ts, title=f"{eda_agg} rainfall — {station} ({agg_func})",
                     labels={"value":"rainfall (mm)","index":"date"})
    fig_ts.update_layout(height=320, margin=dict(l=10,r=10,t=40,b=0))
    st.plotly_chart(fig_ts, use_container_width=True)
with tab2:
    if eda_rule == "M":
        clim = ts.groupby(ts.index.month).agg(["mean","median","std"]).rename_axis("month")
        fig_seas = px.bar(clim.reset_index(), x="month", y="mean", error_y="std",
                          title=f"{station} — mean ± std by month")
    else:
        qnum = ts.index.to_period("Q").quarter
        clim = ts.groupby(qnum).agg(["mean","median","std"]).rename_axis("quarter")
        fig_seas = px.bar(clim.reset_index(), x="quarter", y="mean", error_y="std",
                          title=f"{station} — mean ± std by quarter")
    st.dataframe(clim.round(2), use_container_width=True)
    fig_seas.update_layout(height=320, margin=dict(l=10,r=10,t=40,b=0))
    st.plotly_chart(fig_seas, use_container_width=True)
with tab3:
    m = s_df.set_index("date")["rainfall_mm"].resample("M").sum().dropna()
    hm = (m.to_frame("mm")
          .assign(Year=lambda x: x.index.year, Month=lambda x: x.index.month)
          .pivot_table(index="Year", columns="Month", values="mm", aggfunc="sum")
          .sort_index())
    fig_hm = px.imshow(hm, aspect="auto", labels=dict(color="mm"),
                       title=f"{station} — Year × Month rainfall (sum)")
    fig_hm.update_layout(height=360, margin=dict(l=10,r=10,t=40,b=0))
    st.plotly_chart(fig_hm, use_container_width=True)
with tab4:
    dist_df = ts.reset_index().rename(columns={"index":"date", ts.name:"rainfall_mm"})
    dist_df["period"] = dist_df["date"].dt.to_period("M" if eda_rule=="M" else "Q").astype(str)
    fig_box = px.box(dist_df, y="rainfall_mm", points="outliers", title="Boxplot of aggregated rainfall")
    fig_box.update_layout(height=320, margin=dict(l=10,r=10,t=40,b=0))
    st.plotly_chart(fig_box, use_container_width=True)

st.markdown("---")

# ------------------------
# Modeling series (Monthly/Quarterly)
# ------------------------
series = to_freq(df, station, agg=agg_method, rule=rule)
min_required = lookback + test_periods + max(seasonality, 4)
if len(series) < min_required:
    st.warning(
        f"Not enough periods after resampling for current settings. "
        f"Have {len(series)}, need ≥ {min_required}. Reduce lookback/test or use longer history."
    )
    st.stop()

st.subheader(f"{freq_mode} series for modeling")
st.write(f"**Station:** {station} | **Periods:** {len(series)} | **From** {series.index.min().date()} → **To** {series.index.max().date()}")
st.plotly_chart(px.line(series, title=f"{freq_mode} Rainfall — {station}"), use_container_width=True)

values = series.values.reshape(-1, 1)
train_vals = values[:-test_periods]
test_vals = values[-test_periods:]
test_index = series.index[-test_periods:]

rows = []
pred_test = {}
pred_future = {}

# Seasonal-Naïve
if use_baseline:
    true_series = values.flatten()
    base_all = seasonal_naive(true_series, season=seasonality)
    baseline_test = base_all[-test_periods:] if base_all.size > 0 else None
    if baseline_test is not None and len(baseline_test) == len(test_vals):
        y_true = test_vals.flatten()
        mae, rmse, r2 = metrics(y_true, baseline_test)
        rows.append([f"Seasonal-Naïve (t-{seasonality})", round(mae,2), round(rmse,2), round(r2,3)])
        pred_test["Seasonal-Naïve"] = baseline_test
        template = true_series[-seasonality:]
        reps = int(np.ceil((forecast_h) / seasonality))
        pred_future["Seasonal-Naïve"] = np.tile(template, reps)[:forecast_h]

# LSTM
if use_lstm and tf_available():
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_vals)
    full_scaled = scaler.transform(values)

    X_full, y_full = make_sequences(full_scaled.flatten(), lookback)
    test_start_idx = len(values) - test_periods - 1
    y_full_start = lookback
    test_idx_in_yfull = test_start_idx - y_full_start + 1
    X_train, y_train = X_full[:test_idx_in_yfull], y_full[:test_idx_in_yfull]
    X_test, y_test = X_full[test_idx_in_yfull:], y_full[test_idx_in_yfull:]

    with st.spinner("Training LSTM..."):
        model = build_lstm_model(lstm_units_1, lstm_units_2, dropout_rate)
        es = get_early_stopping(patience)
        model.fit(X_train, y_train, validation_data=(X_test, y_test),
                  epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])

    yhat_test_scaled = model.predict(X_test, verbose=0).reshape(-1, 1)
    yhat_test = scaler.inverse_transform(yhat_test_scaled).flatten()
    y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

    mae, rmse, r2 = metrics(y_test_unscaled, yhat_test)
    rows.append(["LSTM", round(mae,2), round(rmse,2), round(r2,3)])
    pred_test["LSTM"] = yhat_test

    last_window_scaled = full_scaled[-lookback:]
    pred_future["LSTM"] = recursive_forecast(model, last_window_scaled, forecast_h, scaler)
elif use_lstm:
    st.info("Skipping LSTM because TensorFlow isn’t installed on this environment.")

# SARIMA
if use_sarima:
    order = (p, d, q)
    seasonal_order = (P, D, Q, seasonality)
    with st.spinner("Fitting SARIMA..."):
        sar = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    sar_pred = sar.get_prediction(start=test_index[0], end=test_index[-1], dynamic=False)
    yhat_test = sar_pred.predicted_mean.values
    y_true = test_vals.flatten()
    mae, rmse, r2 = metrics(y_true, yhat_test)
    rows.append([f"SARIMA{order}x{seasonal_order}", round(mae,2), round(rmse,2), round(r2,3)])
    pred_test["SARIMA"] = yhat_test
    pred_future["SARIMA"] = sar.get_forecast(steps=forecast_h).predicted_mean.values

# Regression (trend + seasonal dummies)
if use_reg:
    df_y = series.to_frame("y").copy()
    df_y["t"] = np.arange(len(df_y))
    if rule == "M":
        df_y["season"] = df_y.index.month
        dummies = pd.get_dummies(df_y["season"].astype(int), prefix="M", drop_first=True)
    else:
        df_y["season"] = df_y.index.to_period("Q").quarter
        dummies = pd.get_dummies(df_y["season"].astype(int), prefix="Q", drop_first=True)
    X = pd.concat([pd.Series(1.0, index=df_y.index, name="const"), df_y["t"], dummies], axis=1)
    y = df_y["y"].values
    reg = sm.OLS(y, X).fit()
    X_test_reg = X.iloc[-test_periods:, :]
    y_true = y[-test_periods:]
    yhat_test = reg.predict(X_test_reg).values
    mae, rmse, r2 = metrics(y_true, yhat_test)
    rows.append(["Regression (trend + seasonal dummies)", round(mae,2), round(rmse,2), round(r2,3)])
    pred_test["Regression"] = yhat_test

    last_idx = series.index[-1]
    last_per = last_idx.to_period(rule)
    fut_periods = pd.period_range(last_per + 1, periods=forecast_h, freq=rule)
    fut_index = fut_periods.to_timestamp(how="end")
    fut_df = pd.DataFrame(index=fut_index)
    fut_df["t"] = np.arange(len(df_y), len(df_y)+forecast_h)
    if rule == "M":
        fut_df["season"] = fut_df.index.month
        d_f = pd.get_dummies(fut_df["season"].astype(int), prefix="M", drop_first=True).reindex(columns=dummies.columns, fill_value=0)
    else:
        fut_df["season"] = fut_df.index.to_period("Q").quarter
        d_f = pd.get_dummies(fut_df["season"].astype(int), prefix="Q", drop_first=True).reindex(columns=dummies.columns, fill_value=0)
    X_fut = pd.concat([pd.Series(1.0, index=fut_df.index, name="const"), fut_df["t"], d_f], axis=1)
    pred_future["Regression"] = reg.predict(X_fut).values

# Results table
st.subheader(f"Evaluation — last {test_periods} {freq_mode.lower()} periods")
res_df = pd.DataFrame(rows, columns=["Model","MAE (mm)","RMSE (mm)","R²"])
st.dataframe(res_df, use_container_width=True)

# Test plot
fig_test = go.Figure()
fig_test.add_trace(go.Scatter(x=test_index, y=test_vals.flatten(), mode="lines+markers", name="Actual"))
for name, pred in pred_test.items():
    fig_test.add_trace(go.Scatter(x=test_index, y=pred, mode="lines+markers", name=name))
fig_test.update_layout(title=f"Test Set — Actual vs Predicted ({station}, {freq_mode})", height=380, margin=dict(l=10,r=10,t=40,b=0))
st.plotly_chart(fig_test, use_container_width=True)

# Forecasts
last_idx = series.index[-1]
last_per = last_idx.to_period(rule)
future_periods = pd.period_range(last_per + 1, periods=forecast_h, freq=rule)
future_idx = future_periods.to_timestamp(how="end")
fcast_table = pd.DataFrame({"date": future_idx})
for name, arr in pred_future.items():
    fcast_table[name] = np.round(arr, 2)

st.subheader(f"Forecast next {forecast_h} {freq_mode.lower()} periods")
st.dataframe(fcast_table, use_container_width=True)

fig_f = go.Figure()
fig_f.add_trace(go.Scatter(x=series.index, y=series.values.flatten(), mode="lines", name="History"))
for name, arr in pred_future.items():
    fig_f.add_trace(go.Scatter(x=future_idx, y=arr, mode="lines+markers", name=name))
fig_f.update_layout(title=f"Historical & {forecast_h}-Period Forecast — {station} ({freq_mode})",
                    height=380, margin=dict(l=10,r=10,t=40,b=0))
st.plotly_chart(fig_f, use_container_width=True)

st.download_button("⬇️ Download forecasts (CSV)",
                   data=fcast_table.to_csv(index=False),
                   file_name=f"{station}_{freq_mode.lower()}_multi_model_forecasts_{forecast_h}.csv",
                   mime="text/csv")

st.caption("Notes: LSTM is optional; if TensorFlow isn't present, the app still runs SARIMA/Regression/Naïve.")
