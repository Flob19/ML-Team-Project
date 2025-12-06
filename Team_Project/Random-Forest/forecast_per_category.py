"""
Forecast demand per product category using Random Forest

This script:
1. Reads sales_with_weather_tx.xlsx
2. Aggregates to hourly orders per category
3. Trains a separate RandomForest model for each category
4. Makes 7-day forecasts per category using weather API
5. Saves results to Excel and creates visualizations
"""

import pandas as pd
import numpy as np
import requests
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from openpyxl import load_workbook

data_path = Path("../Data/sales_with_weather_tx.xlsx")
if not data_path.exists():
    data_path = Path("Data/sales_with_weather_tx.xlsx")

print("=== LOADING DATA ===")
df = pd.read_excel(data_path)

if 'datetime' not in df.columns:
    df['datetime'] = pd.to_datetime(df['transaction_date'].astype(str) + ' ' + df['transaction_time'].astype(str))
if 'hour' not in df.columns:
    df['hour'] = pd.to_datetime(df['datetime']).dt.floor('h')

print(f"Loaded {len(df)} transactions")
print(f"Product categories: {df['product_category'].unique()}")

print("\n=== AGGREGATING HOURLY DATA PER CATEGORY ===")
df_hourly = (
    df.groupby(['hour', 'product_category'], as_index=False)
    .agg(
        orders=('transaction_id', 'count'),
        total_qty=('transaction_qty', 'sum'),
        temperature_C=('temperature_C', 'mean'),
        rain_mm=('rain_mm', 'mean'),
        cloud_cover_pct=('cloud_cover_pct', 'mean'),
        wind_speed_kmh=('wind_speed_kmh', 'mean'),
        store_id=('store_id', lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
    )
)

df_hourly['cafe_id'] = df_hourly['store_id'].astype(str)
df_hourly = pd.get_dummies(df_hourly, columns=['cafe_id'], prefix='cafe')
df_hourly = df_hourly.drop(columns=['store_id'])

print(f"Aggregated to {len(df_hourly)} hour-category combinations")

categories = sorted(df_hourly['product_category'].unique())
print(f"Categories: {categories}")

def get_weather_forecast(lat, lon, hours_ahead=168, timezone="America/New_York"):
    """Fetch hourly weather forecast from Open-Meteo API"""
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
    """Add time features to DataFrame with 'hour' column"""
    df = df.copy()
    df["hour_of_day"] = df["hour"].dt.hour
    df["day_of_week"] = df["hour"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    return df

def forecast_category_orders(rf_model, df_history, df_forecast_weather, horizon_hours=168):
    """Autoregressive multi-step forecast for a category"""
    hist = df_history.sort_values("hour").copy()
    df_forecast_weather = df_forecast_weather.sort_values("hour").reset_index(drop=True)
    
    horizon_hours = min(horizon_hours, len(df_forecast_weather))
    
    cafe_features = [col for col in hist.columns if col.startswith('cafe_')]
    base_features = [
        "temperature_C", "rain_mm", "cloud_cover_pct", "wind_speed_kmh",
        "hour_of_day", "day_of_week", "is_weekend",
        "qty_lag_1h", "qty_mean_24h"
    ]
    features = base_features + cafe_features
    
    preds = []
    
    for i in range(horizon_hours):
        row_weather = df_forecast_weather.iloc[i]
        t = row_weather["hour"]
        
        if len(hist) > 0:
            last_qty = hist["orders"].iloc[-1]
            
            cutoff_time = t - pd.Timedelta(hours=24)
            last_24h = hist[hist["hour"] > cutoff_time]
            
            if len(last_24h) > 0:
                qty_mean_24h = last_24h["orders"].mean()
            else:
                qty_mean_24h = hist["orders"].tail(24).mean() if len(hist) >= 24 else hist["orders"].mean()
            
            cafe_values = {col: hist[col].iloc[-1] for col in cafe_features} if cafe_features else {}
        else:
            last_qty = 0
            qty_mean_24h = 0
            cafe_values = {col: 0 for col in cafe_features} if cafe_features else {}
        
        feat = {
            "temperature_C": row_weather["temperature_C"],
            "rain_mm": row_weather["rain_mm"],
            "cloud_cover_pct": row_weather["cloud_cover_pct"],
            "wind_speed_kmh": row_weather["wind_speed_kmh"],
            "hour_of_day": row_weather["hour_of_day"],
            "day_of_week": row_weather["day_of_week"],
            "is_weekend": row_weather["is_weekend"],
            "qty_lag_1h": last_qty,
            "qty_mean_24h": qty_mean_24h,
            **cafe_values
        }
        
        X_new = pd.DataFrame([feat])[features]
        y_hat = rf_model.predict(X_new)[0]
        y_hat = max(0, y_hat)
        
        preds.append({"hour": t, "pred_qty": y_hat})
        
        new_row = {"hour": t, "orders": y_hat}
        new_row.update(cafe_values)
        hist = pd.concat([
            hist,
            pd.DataFrame([new_row])
        ], ignore_index=True)
        
        if (i + 1) % 24 == 0:
            print(f"    Predicted {i + 1}/{horizon_hours} hours...")
    
    return pd.DataFrame(preds)

print("\n=== TRAINING MODELS PER CATEGORY ===")
models = {}
results = {}

for category in categories:
    print(f"\n--- {category} ---")
    
    df_cat = df_hourly[df_hourly['product_category'] == category].copy()
    df_cat = df_cat.sort_values('hour')
    
    df_cat['hour_of_day'] = df_cat['hour'].dt.hour
    df_cat['day_of_week'] = df_cat['hour'].dt.dayofweek
    df_cat['is_weekend'] = df_cat['day_of_week'].isin([5, 6]).astype(int)
    
    df_cat['qty_lag_1h'] = df_cat['total_qty'].shift(1)
    df_cat['qty_mean_24h'] = df_cat['total_qty'].rolling(24).mean()
    df_cat = df_cat.dropna(subset=['qty_lag_1h', 'qty_mean_24h'])
    
    if len(df_cat) < 50:
        print(f"  ⚠️  Skipping {category}: insufficient data ({len(df_cat)} rows)")
        continue
    
    cafe_features = [col for col in df_cat.columns if col.startswith('cafe_')]
    features = [
        "temperature_C", "rain_mm", "cloud_cover_pct", "wind_speed_kmh",
        "hour_of_day", "day_of_week", "is_weekend",
        "qty_lag_1h", "qty_mean_24h"
    ] + cafe_features
    
    X = df_cat[features]
    y = df_cat['total_qty']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )
    
    print(f"  Running grid search for {category}...")
    tscv = TimeSeriesSplit(n_splits=3)
    
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [6, 8, 10],
        "min_samples_split": [5, 10, 15],
        "min_samples_leaf": [2, 4, 5]
    }
    
    base_rf = RandomForestRegressor(
        random_state=42,
        n_jobs=-1
    )
    
    grid = GridSearchCV(
        estimator=base_rf,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=0
    )
    
    grid.fit(X_train, y_train)
    rf = grid.best_estimator_
    
    print(f"  Best parameters: {grid.best_params_}")
    print(f"  Best CV score (MAE): {-grid.best_score_:.2f}")
    models[category] = rf
    
    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results[category] = {
        'train_size': len(X_train),
        'test_size': len(X_test),
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }
    
    print(f"  Train: {len(X_train)} rows, Test: {len(X_test)} rows")
    print(f"  MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
    
    history_cols = ['hour', 'total_qty'] + [col for col in df_cat.columns if col.startswith('cafe_')]
    df_cat_history = df_cat[history_cols].copy()
    df_cat_history = df_cat_history.rename(columns={'total_qty': 'orders'})
    results[category]['history'] = df_cat_history

print("\n=== FETCHING WEATHER FORECAST ===")
lat, lon = 40.7128, -74.0060
forecast_weather = get_weather_forecast(lat, lon, hours_ahead=168)
forecast_weather = add_time_features(forecast_weather)

print(f"✓ Weather forecast prepared: {len(forecast_weather)} hours ({len(forecast_weather)/24:.1f} days)")
print(f"  From: {forecast_weather['hour'].min()}")
print(f"  To: {forecast_weather['hour'].max()}")

print("\n=== GENERATING 7-DAY FORECASTS PER CATEGORY ===")
all_forecasts = []

for category in models.keys():
    print(f"\n--- {category} ---")
    df_history = results[category]['history']
    
    pred_df = forecast_category_orders(
        rf_model=models[category],
        df_history=df_history,
        df_forecast_weather=forecast_weather,
        horizon_hours=168
    )
    
    pred_df['product_category'] = category
    all_forecasts.append(pred_df)
    
    print(f"  ✓ Forecast complete: {len(pred_df)} hours")

forecast_df = pd.concat(all_forecasts, ignore_index=True)
forecast_df = forecast_df.rename(columns={'pred_qty': 'pred_orders'})
forecast_df = forecast_df[['hour', 'product_category', 'pred_orders']]

print(f"\n=== SAVING TO EXCEL ===")
excel_path = data_path
with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='replace') as writer:
    forecast_df.to_excel(writer, sheet_name='forecast_per_category', index=False)

print(f"✓ Saved forecast to {excel_path} (sheet: 'forecast_per_category')")

print("\n=== CREATING VISUALIZATIONS ===")

trained_categories = [cat for cat in categories if cat in models.keys()]
n_categories = len(trained_categories)

fig, axes = plt.subplots(n_categories, 1, figsize=(18, 4 * n_categories))
if n_categories == 1:
    axes = [axes]

colors = plt.cm.tab10(np.linspace(0, 1, n_categories))

for idx, category in enumerate(trained_categories):
    cat_forecast = forecast_df[forecast_df['product_category'] == category]
    
    axes[idx].plot(cat_forecast['hour'], cat_forecast['pred_orders'], 
                   linewidth=2.5, color=colors[idx], label=f'{category} forecast', alpha=0.9)
    axes[idx].set_title(f'7-Day Forecast: {category}', fontsize=14, fontweight='bold', pad=15)
    axes[idx].set_ylabel('Predicted Quantity', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Date and Time', fontsize=12, fontweight='bold')
    axes[idx].grid(True, alpha=0.3, linestyle='--')
    axes[idx].legend(fontsize=11)
    axes[idx].tick_params(axis='x', rotation=45)
    
    for i in range(len(cat_forecast)):
        hour = cat_forecast.iloc[i]["hour"]
        if hour.hour == 0:
            axes[idx].axvline(x=hour, color='gray', linestyle=':', alpha=0.5, linewidth=1)

plt.tight_layout()
plot_path = Path("forecast_per_category.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"✓ Saved individual category plots to {plot_path.absolute()}")
plt.close()

fig, ax = plt.subplots(figsize=(20, 8))
for idx, category in enumerate(trained_categories):
    cat_forecast = forecast_df[forecast_df['product_category'] == category]
    ax.plot(cat_forecast['hour'], cat_forecast['pred_orders'], 
            linewidth=2, label=category, alpha=0.8, marker='o', markersize=2)

ax.set_title('7-Day Forecast: All Product Categories', fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('Predicted Quantity', fontsize=13, fontweight='bold')
ax.set_xlabel('Date and Time', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='best', fontsize=11, ncol=2)
ax.tick_params(axis='x', rotation=45)

for i in range(len(forecast_df)):
    hour = forecast_df.iloc[i]["hour"]
    if hour.hour == 0:
        ax.axvline(x=hour, color='gray', linestyle=':', alpha=0.3, linewidth=1)

plt.tight_layout()
plot_path_combined = Path("forecast_all_categories.png")
plt.savefig(plot_path_combined, dpi=150, bbox_inches="tight")
print(f"✓ Saved combined plot to {plot_path_combined.absolute()}")
plt.close()

fig, ax = plt.subplots(figsize=(14, 8))
category_totals = forecast_df.groupby('product_category')['pred_orders'].sum().sort_values(ascending=False)
bars = ax.barh(category_totals.index, category_totals.values, color=plt.cm.viridis(np.linspace(0, 1, len(category_totals))))
ax.set_title('Total Predicted Quantity per Category (7 Days)', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('Total Quantity (7 days)', fontsize=13, fontweight='bold')
ax.set_ylabel('Product Category', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x', linestyle='--')

for i, (cat, val) in enumerate(category_totals.items()):
    ax.text(val + 50, i, f'{int(val)}', va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plot_path_totals = Path("forecast_category_totals.png")
plt.savefig(plot_path_totals, dpi=150, bbox_inches="tight")
print(f"✓ Saved totals bar chart to {plot_path_totals.absolute()}")
plt.close()

print("\n=== SUMMARY ===")
print("\nModel Performance:")
for category, res in results.items():
    if 'r2' in res:
        print(f"  {category}: R² = {res['r2']:.4f}, MAE = {res['mae']:.2f}")

print(f"\nForecast Summary (Total Quantity per Category):")
for category in categories:
    if category in forecast_df['product_category'].values:
        cat_forecast = forecast_df[forecast_df['product_category'] == category]
        total_7d = cat_forecast['pred_orders'].sum()
        print(f"  {category}: Mean/hour = {cat_forecast['pred_orders'].mean():.2f}, "
              f"Total 7 days = {total_7d:.0f}, "
              f"Min = {cat_forecast['pred_orders'].min():.2f}, "
              f"Max = {cat_forecast['pred_orders'].max():.2f}")

print("\n=== COMPLETE ===")

