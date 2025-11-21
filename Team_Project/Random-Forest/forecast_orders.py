"""
Weather forecast-based order prediction using Random Forest

This script:
1. Fetches weather forecast via Open-Meteo API
2. Builds the same features as training data
3. Performs autoregressive multi-step prediction of future orders
"""

import pandas as pd
import numpy as np
import requests
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

FEATURES = [
    "temperature_C", "rain_mm", "cloud_cover_pct", "wind_speed_kmh",
    "hour_of_day", "day_of_week", "is_weekend",
    "orders_lag_1h", "orders_mean_24h"
]


def get_weather_forecast(lat, lon, hours_ahead=168, timezone="America/New_York"):
    """
    Fetches hourly weather forecast for the next hours_ahead hours.
    
    Parameters:
    -----------
    lat : float
        Latitude (e.g., 40.7128 for New York)
    lon : float
        Longitude (e.g., -74.0060 for New York)
    hours_ahead : int
        Number of hours ahead to fetch forecast (default: 168 = 7 days, max 240)
    timezone : str
        Timezone (default: "America/New_York")
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: hour, temperature_C, rain_mm, cloud_cover_pct, wind_speed_kmh
    """
    url = "https://api.open-meteo.com/v1/forecast"
    
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join([
            "temperature_2m",
            "precipitation",
            "cloud_cover",
            "wind_speed_10m"
        ]),
        "forecast_hours": hours_ahead,
        "timezone": timezone,
        "temperature_unit": "celsius",
        "windspeed_unit": "kmh",
        "precipitation_unit": "mm",
    }
    
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        
        hourly = data["hourly"]
        df_forecast = pd.DataFrame({
            "hour": pd.to_datetime(hourly["time"]),
            "temperature_C": hourly["temperature_2m"],
            "rain_mm": hourly["precipitation"],
            "cloud_cover_pct": hourly["cloud_cover"],
            "wind_speed_kmh": hourly["wind_speed_10m"],
        })
        
        print(f"✓ Fetched weather forecast for {len(df_forecast)} hours")
        return df_forecast
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Error fetching weather forecast: {e}")
        raise


def add_time_features(df):
    """
    Adds time features to a DataFrame with 'hour' column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'hour' column (datetime)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added columns: hour_of_day, day_of_week, is_weekend
    """
    df = df.copy()
    df["hour_of_day"] = df["hour"].dt.hour
    df["day_of_week"] = df["hour"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df


def forecast_orders(rf_model, df_history, df_forecast_weather, horizon_hours=24):
    """
    Performs autoregressive multi-step prediction of orders for upcoming hours.
    
    Parameters:
    -----------
    rf_model : RandomForestRegressor
        Trained Random Forest model
    df_history : pd.DataFrame
        Historical data with columns 'hour' (datetime) and 'orders', sorted by time
    df_forecast_weather : pd.DataFrame
        Weather forecast DataFrame from get_weather_forecast + add_time_features
    horizon_hours : int
        Number of hours ahead to predict
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: hour, pred_orders
    """
    hist = df_history.sort_values("hour").copy()
    df_forecast_weather = df_forecast_weather.sort_values("hour").reset_index(drop=True)
    
    horizon_hours = min(horizon_hours, len(df_forecast_weather))
    
    preds = []
    
    for i in range(horizon_hours):
        row_weather = df_forecast_weather.iloc[i]
        t = row_weather["hour"]
        
        if len(hist) > 0:
            last_orders = hist["orders"].iloc[-1]
            
            cutoff_time = t - pd.Timedelta(hours=24)
            last_24h = hist[hist["hour"] > cutoff_time]
            
            if len(last_24h) > 0:
                orders_mean_24h = last_24h["orders"].mean()
            else:
                orders_mean_24h = hist["orders"].tail(24).mean() if len(hist) >= 24 else hist["orders"].mean()
        else:
            last_orders = 0
            orders_mean_24h = 0
        
        feat = {
            "temperature_C": row_weather["temperature_C"],
            "rain_mm": row_weather["rain_mm"],
            "cloud_cover_pct": row_weather["cloud_cover_pct"],
            "wind_speed_kmh": row_weather["wind_speed_kmh"],
            "hour_of_day": row_weather["hour_of_day"],
            "day_of_week": row_weather["day_of_week"],
            "is_weekend": row_weather["is_weekend"],
            "orders_lag_1h": last_orders,
            "orders_mean_24h": orders_mean_24h,
        }
        
        X_new = pd.DataFrame([feat])[FEATURES]
        y_hat = rf_model.predict(X_new)[0]
        y_hat = max(0, y_hat)
        
        preds.append({"hour": t, "pred_orders": y_hat})
        
        hist = pd.concat([
            hist,
            pd.DataFrame([{"hour": t, "orders": y_hat}])
        ], ignore_index=True)
        
        if (i + 1) % 6 == 0:
            print(f"  Predicted {i + 1}/{horizon_hours} hours...")
    
    return pd.DataFrame(preds)


def main():
    print("=== LOADING MODEL AND HISTORY ===")
    
    data_path = Path("../Data/hourly_orders_with_weather.xlsx")
    if not data_path.exists():
        data_path = Path("Data/hourly_orders_with_weather.xlsx")
    if not data_path.exists():
        print(f"❌ Could not find {data_path}")
        print("   Run Random_Forest.py first to create training data")
        return
    
    df = pd.read_excel(data_path)
    df["hour"] = pd.to_datetime(df["hour"])
    df = df.sort_values("hour")
    
    df["orders_lag_1h"] = df["orders"].shift(1)
    df["orders_mean_24h"] = df["orders"].rolling(24).mean()
    df = df.dropna(subset=["orders_lag_1h", "orders_mean_24h"])
    
    df_history = df[["hour", "orders"]].copy()
    
    print(f"✓ Loaded history: {len(df_history)} hours")
    print(f"  From: {df_history['hour'].min()}")
    print(f"  To: {df_history['hour'].max()}")
    
    model_path = Path("rf_orders_model.pkl")
    if not model_path.exists():
        model_path = Path("../Random-Forest/rf_orders_model.pkl")
    if not model_path.exists():
        print(f"❌ Could not find model file")
        print("   Run Random_Forest.py first to train and save the model")
        return
    
    rf = joblib.load(model_path)
    print(f"✓ Loaded model from {model_path}")
    
    print("\n=== FETCHING WEATHER FORECAST ===")
    
    lat, lon = 40.7128, -74.0060
    hours_ahead = 168
    
    forecast_weather = get_weather_forecast(lat, lon, hours_ahead=hours_ahead)
    forecast_weather = add_time_features(forecast_weather)
    
    print(f"✓ Weather forecast prepared: {len(forecast_weather)} hours ({len(forecast_weather)/24:.1f} days)")
    print(f"  From: {forecast_weather['hour'].min()}")
    print(f"  To: {forecast_weather['hour'].max()}")
    
    print("\n=== SHORT-TERM ORDER FORECAST (48 hours) ===")
    horizon_short = 48
    
    pred_df_short = forecast_orders(
        rf_model=rf,
        df_history=df_history,
        df_forecast_weather=forecast_weather,
        horizon_hours=horizon_short
    )
    
    print(f"\n✓ Short-term prediction complete: {len(pred_df_short)} hours")
    print("\nFirst 10 predictions (short-term):")
    print(pred_df_short.head(10).to_string(index=False))
    
    print("\n=== LONG-TERM ORDER FORECAST (7 days) ===")
    horizon_long = 168
    
    pred_df_long = forecast_orders(
        rf_model=rf,
        df_history=df_history,
        df_forecast_weather=forecast_weather,
        horizon_hours=horizon_long
    )
    
    print(f"\n✓ Long-term prediction complete: {len(pred_df_long)} hours ({len(pred_df_long)/24:.1f} days)")
    print("\nFirst 10 predictions (long-term):")
    print(pred_df_long.head(10).to_string(index=False))
    
    print("\n=== CREATING VISUALIZATIONS ===")
    
    plt.figure(figsize=(16, 7))
    plt.plot(pred_df_short["hour"], pred_df_short["pred_orders"], 
             marker="o", linewidth=2.5, markersize=6, 
             label="Predicted orders", color="#2E86AB", markerfacecolor="#A23B72", markeredgewidth=1.5)
    plt.xticks(rotation=45, ha='right')
    plt.title("Hourly Order Forecast: Next 48 Hours", fontsize=16, fontweight="bold", pad=20)
    plt.ylabel("Number of orders", fontsize=13, fontweight="bold")
    plt.xlabel("Time (hour)", fontsize=13, fontweight="bold")
    plt.grid(True, alpha=0.4, linestyle='--')
    plt.legend(fontsize=12, loc='best')
    
    for i in range(len(pred_df_short)):
        hour = pred_df_short.iloc[i]["hour"]
        if hour.hour == 0:
            plt.axvline(x=hour, color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plot_path_short = Path("forecast_plot_48h.png")
    plt.savefig(plot_path_short, dpi=150, bbox_inches="tight")
    print(f"✓ Hourly 48-hour plot saved to {plot_path_short.absolute()}")
    plt.close()
    
    plt.figure(figsize=(18, 7))
    plt.plot(pred_df_long["hour"], pred_df_long["pred_orders"], 
             linewidth=2.5, label="Predicted orders (7 days)", color="#F18F01", alpha=0.9)
    plt.xticks(rotation=45, ha='right')
    plt.title("Order Forecast: Next 7 Days", fontsize=16, fontweight="bold", pad=20)
    plt.ylabel("Number of orders", fontsize=13, fontweight="bold")
    plt.xlabel("Date and time", fontsize=13, fontweight="bold")
    plt.grid(True, alpha=0.4, linestyle='--')
    plt.legend(fontsize=12, loc='best')
    
    for i in range(len(pred_df_long)):
        hour = pred_df_long.iloc[i]["hour"]
        if hour.hour == 0:
            plt.axvline(x=hour, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            plt.text(hour, plt.ylim()[1] * 0.98, hour.strftime('%m/%d'), 
                    rotation=90, ha='right', va='top', fontsize=9, alpha=0.7)
    
    plt.tight_layout()
    plot_path_long = Path("forecast_plot_7d.png")
    plt.savefig(plot_path_long, dpi=150, bbox_inches="tight")
    print(f"✓ 7-day plot saved to {plot_path_long.absolute()}")
    plt.show()
    
    output_path_short = Path("forecast_predictions_48h.csv")
    pred_df_short.to_csv(output_path_short, index=False)
    print(f"✓ Short-term predictions saved to {output_path_short.absolute()}")
    
    output_path_long = Path("forecast_predictions_7d.csv")
    pred_df_long.to_csv(output_path_long, index=False)
    print(f"✓ Long-term predictions saved to {output_path_long.absolute()}")
    
    print("\n=== COMPLETE ===")


if __name__ == "__main__":
    main()
