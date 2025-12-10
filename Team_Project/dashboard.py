"""
Streamlit dashboard for Café Order Forecaster
"""

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV




ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Data" / "sales_with_weather_tx.xlsx"

BASE_FEATURES = [
    "temperature_C",
    "rain_mm",
    "cloud_cover_pct",
    "wind_speed_kmh",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "is_morning",
    "is_afternoon",
    "is_evening",
    "is_night",
    "qty_lag_1h",
    "qty_lag_24h",
    "qty_mean_24h",
    "qty_std_24h",
    "qty_mean_7d",
]


@st.cache_data(show_spinner=False)
def load_transactions():
    df = pd.read_excel(DATA_PATH)
    df["datetime"] = pd.to_datetime(
        df["transaction_date"].astype(str)
        + " "
        + df["transaction_time"].astype(str)
    )
    df["hour"] = df["datetime"].dt.floor("h")
    df["store_id"] = df["store_id"].fillna(-1)
    df["product_category"] = df["product_category"].fillna("Unknown")
    return df


def aggregate_total(df_tx: pd.DataFrame) -> pd.DataFrame:
    hourly = (
        df_tx.groupby("hour", as_index=False)
        .agg(
            qty=("transaction_qty", "sum"),
            temperature_C=("temperature_C", "mean"),
            rain_mm=("rain_mm", "mean"),
            cloud_cover_pct=("cloud_cover_pct", "mean"),
            wind_speed_kmh=("wind_speed_kmh", "mean"),
            store_id=(
                "store_id",
                lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0],
            ),
        )
        .sort_values("hour")
    )
    hourly["cafe_id"] = hourly["store_id"].astype(str)
    hourly = pd.get_dummies(hourly, columns=["cafe_id"], prefix="cafe")
    hourly = hourly.drop(columns=["store_id"])
    return hourly


def aggregate_by_category(df_tx: pd.DataFrame) -> pd.DataFrame:
    hourly = (
        df_tx.groupby(["hour", "product_category"], as_index=False)
        .agg(
            qty=("transaction_qty", "sum"),
            temperature_C=("temperature_C", "mean"),
            rain_mm=("rain_mm", "mean"),
            cloud_cover_pct=("cloud_cover_pct", "mean"),
            wind_speed_kmh=("wind_speed_kmh", "mean"),
            store_id=(
                "store_id",
                lambda x: x.mode().iloc[0] if len(x.mode()) else x.iloc[0],
            ),
        )
        .sort_values(["product_category", "hour"])
    )
    hourly["cafe_id"] = hourly["store_id"].astype(str)
    hourly = pd.get_dummies(hourly, columns=["cafe_id"], prefix="cafe")
    hourly = hourly.drop(columns=["store_id"])
    return hourly


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_of_day"] = df["hour"].dt.hour
    df["day_of_week"] = df["hour"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["is_morning"] = df["hour_of_day"].between(7, 11).astype(int)
    df["is_afternoon"] = df["hour_of_day"].between(12, 17).astype(int)
    df["is_evening"] = df["hour_of_day"].between(18, 22).astype(int)
    df["is_night"] = (~(df["is_morning"] | df["is_afternoon"] | df["is_evening"])).astype(
        int
    )
    return df


def add_lag_features(df: pd.DataFrame, group_cols=None) -> pd.DataFrame:
    df = df.copy()

    def _apply(group):
        group = group.sort_values("hour")
        group["qty_lag_1h"] = group["qty"].shift(1)
        group["qty_lag_24h"] = group["qty"].shift(24)
        group["qty_mean_24h"] = group["qty"].rolling(24).mean()
        group["qty_std_24h"] = group["qty"].rolling(24).std()
        group["qty_mean_7d"] = group["qty"].rolling(168).mean()
        return group

    if group_cols:
        df = (
            df.groupby(group_cols, group_keys=False)
            .apply(_apply)
            .reset_index(drop=True)
        )
    else:
        df = _apply(df)

    df = df.dropna(subset=["qty_lag_1h", "qty_mean_24h"])
    return df


def feature_columns(df: pd.DataFrame):
    cafe_cols = [col for col in df.columns if col.startswith("cafe_")]
    return BASE_FEATURES + cafe_cols

def train_models(X_train, y_train, model_type="rf"):
    if model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return model

    elif model_type == "lr":
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("regressor", LinearRegression()),
            ]
        )
        model.fit(X_train, y_train)
        return model

    elif model_type == "mlp":
        # Base MLP inside a preprocessing pipeline (no target transform)
        base_mlp = MLPRegressor(
            hidden_layer_sizes=(64, 32),     # smaller network = less overfitting
            activation="relu",
            solver="adam",
            learning_rate="adaptive",
            learning_rate_init=1e-3,
            max_iter=600,
            random_state=42,
            alpha=1e-3,                      # stronger L2 regularisation
            batch_size=64,
            early_stopping=True,
            n_iter_no_change=10,
            validation_fraction=0.15,
            verbose=True,
        )

        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("mlp", base_mlp),
            ]
        )

        # Hyper-parameters for Randomised Search
        param_dist = {
            "mlp__hidden_layer_sizes": [
                (64, 32),
                (64, 64),
                (128, 64),
            ],
            "mlp__alpha": np.logspace(-4, -2, 3),          # [1e-4, 1e-3, 1e-2]
            "mlp__learning_rate_init": [1e-3, 5e-4],
            "mlp__batch_size": [32, 64],
        }

        tscv = TimeSeriesSplit(n_splits=3)

        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_dist,
            n_iter=10,
            cv=tscv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            random_state=42,
            verbose=1,
        )

        search.fit(X_train, y_train)
        print("Best MLP params:", search.best_params_)
        return search.best_estimator_


    else:
        raise ValueError(f"Unknown model_type: {model_type}")



def evaluate(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    return {
        "train": {
            "mae": mean_absolute_error(y_train, y_train_pred),
            "rmse": math.sqrt(mean_squared_error(y_train, y_train_pred)),
            "r2": r2_score(y_train, y_train_pred),
            "mape": mean_absolute_percentage_error(y_train, y_train_pred),
        },
        "test": {
            "mae": mean_absolute_error(y_test, y_test_pred),
            "rmse": math.sqrt(mean_squared_error(y_test, y_test_pred)),
            "r2": r2_score(y_test, y_test_pred),
            "mape": mean_absolute_percentage_error(y_test, y_test_pred),
        },
    }


@st.cache_resource(show_spinner=False)
def prepare_total_models():
    df_tx = load_transactions()
    df_hourly = aggregate_total(df_tx)
    df_hourly = add_time_features(df_hourly)
    df_hourly = add_lag_features(df_hourly)
    feats = feature_columns(df_hourly)

    X = df_hourly[feats]
    y = df_hourly["qty"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=0
    )

    
    rf_model  = train_models(X_train, y_train, "rf")
    lin_model = train_models(X_train, y_train, "lr")
    mlp_model = train_models(X_train, y_train, "mlp")   

    
    rf_metrics  = evaluate(rf_model,  X_train, y_train, X_test, y_test)
    lr_metrics  = evaluate(lin_model, X_train, y_train, X_test, y_test)
    mlp_metrics = evaluate(mlp_model, X_train, y_train, X_test, y_test)

    history_cols = ["hour", "qty"] + [c for c in feats if c.startswith("cafe_")]
    history = df_hourly[history_cols].rename(columns={"qty": "orders"})

    return (
        {
            "Random Forest":       {"model": rf_model,  "metrics": rf_metrics},
            "Linear Regression":   {"model": lin_model, "metrics": lr_metrics},
            "Neural Network (MLP)": {"model": mlp_model, "metrics": mlp_metrics}, 
        },
        feats,
        history,
    )



@st.cache_resource(show_spinner=False)
def prepare_category_models():
    df_tx = load_transactions()
    df_cat = aggregate_by_category(df_tx)
    df_cat = add_time_features(df_cat)
    df_cat = add_lag_features(df_cat, group_cols=["product_category"])
    cafe_cols = [col for col in df_cat.columns if col.startswith("cafe_")]
    feats = BASE_FEATURES + cafe_cols

    models = {}
    histories = {}
    metrics = {}

    for category, group in df_cat.groupby("product_category"):
        if len(group) < 200:
            continue
        cat_df = group.copy()
        for col in cafe_cols:
            if col not in cat_df.columns:
                cat_df[col] = 0
        X = cat_df[feats]
        y = cat_df["qty"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False, random_state=42
        )
        model = train_models(X_train, y_train, "rf")
        cat_metrics = evaluate(model, X_train, y_train, X_test, y_test)
        metrics[category] = cat_metrics
        history_cols = ["hour", "qty"] + cafe_cols
        histories[category] = cat_df[history_cols].rename(columns={"qty": "orders"})
        models[category] = {"model": model, "metrics": cat_metrics}

    return models, histories, metrics, feats


@st.cache_data(ttl=3600, show_spinner=False)
def get_weather_forecast(hours=168, lat=40.7128, lon=-74.0060, timezone="America/New_York"):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(
            [
                "temperature_2m",
                "precipitation",
                "cloud_cover",
                "wind_speed_10m",
            ]
        ),
        "forecast_hours": hours,
        "temperature_unit": "celsius",
        "windspeed_unit": "kmh",
        "precipitation_unit": "mm",
        "timezone": timezone,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]
    df_forecast = pd.DataFrame(
        {
            "hour": pd.to_datetime(hourly["time"]),
            "temperature_C": hourly["temperature_2m"],
            "rain_mm": hourly["precipitation"],
            "cloud_cover_pct": hourly["cloud_cover"],
            "wind_speed_kmh": hourly["wind_speed_10m"],
        }
    )
    df_forecast = add_time_features(df_forecast)
    return df_forecast


def ensure_feature_columns(df, feature_cols):
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feature_cols]


def autoregressive_forecast(
    model, feature_cols, history_df, weather_df, horizon_hours
):
    hist = history_df.sort_values("hour").copy()
    cafe_cols = [c for c in feature_cols if c.startswith("cafe_")]
    preds = []

    for i in range(horizon_hours):
        weather_row = weather_df.iloc[i]
        last_row = hist.iloc[-1]
        recent_window = hist.tail(24)
        qty_mean_24h = recent_window["orders"].mean()
        qty_std_24h = recent_window["orders"].std()
        qty_mean_7d = hist.tail(168)["orders"].mean() if len(hist) >= 168 else hist["orders"].mean()

        feature_row = {
            "temperature_C": weather_row["temperature_C"],
            "rain_mm": weather_row["rain_mm"],
            "cloud_cover_pct": weather_row["cloud_cover_pct"],
            "wind_speed_kmh": weather_row["wind_speed_kmh"],
            "hour_of_day": weather_row["hour_of_day"],
            "day_of_week": weather_row["day_of_week"],
            "is_weekend": weather_row["is_weekend"],
            "hour_sin": weather_row["hour_sin"],
            "hour_cos": weather_row["hour_cos"],
            "day_sin": weather_row["day_sin"],
            "day_cos": weather_row["day_cos"],
            "is_morning": weather_row["is_morning"],
            "is_afternoon": weather_row["is_afternoon"],
            "is_evening": weather_row["is_evening"],
            "is_night": weather_row["is_night"],
            "qty_lag_1h": last_row["orders"],
            "qty_lag_24h": hist.iloc[-24:]["orders"].mean() if len(hist) >= 24 else last_row["orders"],
            "qty_mean_24h": qty_mean_24h,
            "qty_std_24h": qty_std_24h if not math.isnan(qty_std_24h) else 0,
            "qty_mean_7d": qty_mean_7d if not math.isnan(qty_mean_7d) else qty_mean_24h,
        }

        for col in cafe_cols:
            feature_row[col] = last_row.get(col, 0)

        row_df = pd.DataFrame([feature_row])
        row_df = ensure_feature_columns(row_df, feature_cols)
        y_hat = float(model.predict(row_df)[0])
        y_hat = max(0.0, y_hat)
        y_hat = min(y_hat, 1e4)
        preds.append({"hour": weather_row["hour"], "pred_qty": y_hat})
        new_hist = {"hour": weather_row["hour"], "orders": y_hat}
        for col in cafe_cols:
            new_hist[col] = feature_row[col]
        hist = pd.concat([hist, pd.DataFrame([new_hist])], ignore_index=True)

    return pd.DataFrame(preds)


def build_manual_feature(
    temperature, rain, cloud_cover, wind_speed, hour_of_day, day_of_week, is_weekend, cafe_cols
):
    base = {
        "temperature_C": temperature,
        "rain_mm": rain,
        "cloud_cover_pct": cloud_cover,
        "wind_speed_kmh": wind_speed,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "is_weekend": int(is_weekend),
        "hour_sin": math.sin(2 * math.pi * hour_of_day / 24),
        "hour_cos": math.cos(2 * math.pi * hour_of_day / 24),
        "day_sin": math.sin(2 * math.pi * day_of_week / 7),
        "day_cos": math.cos(2 * math.pi * day_of_week / 7),
        "is_morning": int(7 <= hour_of_day <= 11),
        "is_afternoon": int(12 <= hour_of_day <= 17),
        "is_evening": int(18 <= hour_of_day <= 22),
        "is_night": int(hour_of_day < 7 or hour_of_day > 22),
    }
    for col in cafe_cols:
        base[col] = 0
    return base


def manual_prediction(
    model, feature_cols, latest_hist_row, inputs
):
    cafe_cols = [c for c in feature_cols if c.startswith("cafe_")]
    feature_row = build_manual_feature(
        inputs["temperature"],
        inputs["rain"],
        inputs["cloud_cover"],
        inputs["wind_speed"],
        inputs["hour_of_day"],
        inputs["day_of_week"],
        inputs["is_weekend"],
        cafe_cols,
    )
    feature_row["qty_lag_1h"] = latest_hist_row["orders"]
    feature_row["qty_lag_24h"] = latest_hist_row["orders"]
    feature_row["qty_mean_24h"] = inputs.get("qty_mean_24h", latest_hist_row["orders"])
    feature_row["qty_std_24h"] = inputs.get("qty_std_24h", 0)
    feature_row["qty_mean_7d"] = inputs.get("qty_mean_7d", latest_hist_row["orders"])
    selected_cafe = inputs.get("cafe")
    if cafe_cols and selected_cafe:
        target_col = f"cafe_{selected_cafe}"
        if target_col in feature_row:
            feature_row[target_col] = 1
    row_df = pd.DataFrame([feature_row])
    row_df = ensure_feature_columns(row_df, feature_cols)
    return max(0, float(model.predict(row_df)[0]))


def kpi_cards(col, title, value, subtitle=None, delta=None):
    with col:
        st.markdown("### " + title)
        st.markdown(f"## {value}")
        if subtitle:
            st.caption(subtitle)
        if delta is not None:
            color = "green" if delta >= 0 else "red"
            st.markdown(f"<span style='color:{color};font-weight:bold'>{delta:+.1f}%</span>", unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="Café Order Forecaster",
        page_icon="📈",
        layout="wide",
    )

    st.title("Café Order Forecaster")
    st.caption("ML-powered demand prediction with weather-aware Random Forest, MLP & Linear Regression models.")

    models, feature_cols, total_history = prepare_total_models()
    category_models, category_histories, category_metrics, category_features = prepare_category_models()
    weather_df = get_weather_forecast(hours=168)

    categories = ["All"] + sorted(category_models.keys())
    horizon_options = {
        "24 hours": 24,
        "48 hours": 48,
        "72 hours": 72,
        "96 hours": 96,
        "120 hours": 120,
        "144 hours": 144,
        "7 days (168h)": 168,
    }

    with st.sidebar:
        st.header("Forecast Controls")
        category_choice = st.selectbox("Product Category", categories)
        horizon_label = st.select_slider(
            "Forecast Horizon",
            options=list(horizon_options.keys()),
            value="48 hours",
        )
        horizon = horizon_options[horizon_label]
        model_choice = st.selectbox(
            "Model",
            list(models.keys()),
            index=0,
            disabled=category_choice != "All",
            help="Category forecasts currently use Random Forest.",
        )

    if category_choice == "All":
        selected_model = models[model_choice]
        feature_set = feature_cols
        history = total_history
    else:
        selected_model = category_models[category_choice]
        feature_set = category_features
        history = category_histories[category_choice]

    forecast = autoregressive_forecast(
        selected_model["model"],
        feature_set,
        history,
        weather_df,
        horizon,
    )

    forecast["pred_orders"] = forecast["pred_qty"]
    peak_hour = forecast.loc[forecast["pred_orders"].idxmax()]
    period_forecast = forecast["pred_orders"].sum()
    avg_per_hour = forecast["pred_orders"].mean()
    
    accuracy = selected_model.get("metrics", {}).get("test", {}).get("r2", 0) * 100

    col1, col2, col3, col4 = st.columns(4)
    kpi_cards(col1, "Period's Forecast", f"{period_forecast:.0f}", f"Total for {horizon}h")
    kpi_cards(col2, "Peak Hour", peak_hour["hour"].strftime("%a %H:%M"), f"{peak_hour['pred_orders']:.0f} orders expected")
    kpi_cards(col3, "Avg per Hour", f"{avg_per_hour:.1f}", "Average orders/hour")
    kpi_cards(col4, "Model Accuracy", f"{accuracy:.1f}%", model_choice)

    st.subheader("Order Forecast")
    chart = px.line(
        forecast,
        x="hour",
        y="pred_orders",
        markers=True,
        labels={"hour": "Date", "pred_orders": "Predicted Quantity"},
        title=f"{category_choice} forecast for next {horizon}h using {model_choice}",
    )
    st.plotly_chart(chart, use_container_width=True)

    st.subheader("Weather Forecast")
    weather_forecast_df = weather_df.head(horizon).copy()
    
    fig_weather = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Temperature (°C)", "Rain (mm)", "Cloud Cover (%)", "Wind Speed (km/h)"),
        vertical_spacing=0.2,
        horizontal_spacing=0.1
    )
    
    fig_weather.add_trace(
        go.Scatter(x=weather_forecast_df["hour"], y=weather_forecast_df["temperature_C"], 
                  name="Temperature", line=dict(color="red")),
        row=1, col=1
    )
    fig_weather.add_trace(
        go.Scatter(x=weather_forecast_df["hour"], y=weather_forecast_df["rain_mm"], 
                  name="Rain", line=dict(color="blue")),
        row=1, col=2
    )
    fig_weather.add_trace(
        go.Scatter(x=weather_forecast_df["hour"], y=weather_forecast_df["cloud_cover_pct"], 
                  name="Cloud Cover", line=dict(color="gray")),
        row=2, col=1
    )
    fig_weather.add_trace(
        go.Scatter(x=weather_forecast_df["hour"], y=weather_forecast_df["wind_speed_kmh"], 
                  name="Wind Speed", line=dict(color="green")),
        row=2, col=2
    )
    
    fig_weather.update_xaxes(title_text="Date", row=2, col=1)
    fig_weather.update_xaxes(title_text="Date", row=2, col=2)
    fig_weather.update_yaxes(title_text="°C", row=1, col=1)
    fig_weather.update_yaxes(title_text="mm", row=1, col=2)
    fig_weather.update_yaxes(title_text="%", row=2, col=1)
    fig_weather.update_yaxes(title_text="km/h", row=2, col=2)
    
    fig_weather.update_layout(
        height=600,
        title_text=f"Weather forecast for next {horizon}h",
        showlegend=False
    )
    st.plotly_chart(fig_weather, use_container_width=True)

    st.subheader("Manual Prediction")
    cafes = [col.replace("cafe_", "") for col in feature_set if col.startswith("cafe_")]
    weather_now = weather_df.iloc[0]
    with st.form("manual_prediction_form"):
        c1, c2, c3 = st.columns(3)
        temperature = c1.number_input("Temperature (°C)", value=float(weather_now["temperature_C"]))
        rain = c2.number_input("Rain (mm)", value=float(weather_now["rain_mm"]), min_value=0.0)
        cloud = c3.number_input("Cloud Cover (%)", value=float(weather_now["cloud_cover_pct"]), min_value=0.0, max_value=100.0)

        c4, c5, c6 = st.columns(3)
        wind = c4.number_input("Wind Speed (km/h)", value=float(weather_now["wind_speed_kmh"]), min_value=0.0)
        hour_of_day = c5.number_input("Hour of Day (0-23)", min_value=0, max_value=23, value=int(weather_now["hour_of_day"]))
        day_of_week = c6.number_input("Day of Week (0=Mon)", min_value=0, max_value=6, value=int(weather_now["day_of_week"]))

        c7, c8 = st.columns(2)
        is_weekend = c7.toggle("Weekend", value=bool(weather_now["is_weekend"]))
        cafe_choice = c8.selectbox("Cafe", ["Auto"] + cafes) if cafes else "Auto"

        submitted = st.form_submit_button("Predict Orders")

    if submitted:
        latest_row = history.iloc[-1]
        manual_inputs = {
            "temperature": temperature,
            "rain": rain,
            "cloud_cover": cloud,
            "wind_speed": wind,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "cafe": None if cafe_choice == "Auto" else cafe_choice,
            "qty_mean_24h": history.tail(24)["orders"].mean(),
            "qty_std_24h": history.tail(24)["orders"].std(),
            "qty_mean_7d": history.tail(24 * 7)["orders"].mean(),
        }
        pred = manual_prediction(
            selected_model["model"],
            feature_set,
            latest_row,
            manual_inputs,
        )
        st.success(f"Predicted demand: **{pred:.1f} orders**")

    st.divider()
    st.markdown(
        "Built with Streamlit · Weather powered by Open-Meteo · Models trained on Random Forest, MLP and Linear Regression."
    )


if __name__ == "__main__":
    main()

