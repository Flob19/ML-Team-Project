import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import joblib
from pathlib import Path

data_path = Path("../Data/sales_with_weather_tx.xlsx")
if not data_path.exists():
    data_path = Path("Data/sales_with_weather_tx.xlsx")

df_tx = pd.read_excel(data_path)
df_tx['datetime'] = pd.to_datetime(df_tx['transaction_date'].astype(str) + ' ' + df_tx['transaction_time'].astype(str))
df_tx['hour'] = df_tx['datetime'].dt.floor('h')

df = (
    df_tx
    .groupby('hour', as_index=False)
    .agg(
        total_qty=('transaction_qty', 'sum'),
        temperature_C=('temperature_C', 'mean'),
        rain_mm=('rain_mm', 'mean'),
        cloud_cover_pct=('cloud_cover_pct', 'mean'),
        wind_speed_kmh=('wind_speed_kmh', 'mean'),
        store_id=('store_id', lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
    )
)

df['cafe_id'] = df['store_id'].astype(str)
df = pd.get_dummies(df, columns=['cafe_id'], prefix='cafe')
df = df.drop(columns=['store_id'])

df = df.sort_values("hour")

df['hour_of_day'] = df['hour'].dt.hour
df['day_of_week'] = df['hour'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)

df['is_morning'] = df['hour_of_day'].isin([7,8,9,10,11]).astype(int)
df['is_afternoon'] = df['hour_of_day'].isin([12,13,14,15,16,17]).astype(int)
df['is_evening'] = df['hour_of_day'].isin([18,19,20,21,22]).astype(int)
df['is_night'] = df['hour_of_day'].isin([0,1,2,3,4,5,6]).astype(int)

df["qty_lag_1h"] = df["total_qty"].shift(1)
df["qty_lag_24h"] = df["total_qty"].shift(24)
df["qty_mean_24h"] = df["total_qty"].rolling(24).mean()
df["qty_std_24h"] = df["total_qty"].rolling(24).std()
df["qty_mean_7d"] = df["total_qty"].rolling(168).mean()

df = df.dropna(subset=["qty_lag_1h", "qty_mean_24h"])

cafe_features = [col for col in df.columns if col.startswith('cafe_')]
features = [
    "temperature_C", "rain_mm", "cloud_cover_pct", "wind_speed_kmh",
    "hour_of_day", "day_of_week", "is_weekend",
    "is_morning", "is_afternoon", "is_evening", "is_night",
    "qty_lag_1h", "qty_lag_24h", "qty_mean_24h", "qty_std_24h", "qty_mean_7d"
] + cafe_features
X = df[features]
y = df["total_qty"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=0)

print("=== DATA OVERVIEW ===")
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
print(f"Features: {len(features)}")
print("\nFirst 5 rows in training data:")
print(X_train.head(5))
print("\nFirst 5 rows in test data:")
print(X_test.head(5))

RUN_GRID_SEARCH = True

if RUN_GRID_SEARCH:
    print("\n=== HYPERPARAMETER TUNING ===")
    tscv = TimeSeriesSplit(n_splits=5)
    
    param_grid = {
        "n_estimators": [200, 300, 400],
        "max_depth": [8, 10, 12, 15],
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
        verbose=1
    )
    
    print("Running grid search with time-series cross-validation...")
    grid.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid.best_params_}")
    print(f"Best CV score (MAE): {-grid.best_score_:.2f}")
    
    rf = grid.best_estimator_
else:
    print("\n=== USING PREVIOUSLY OPTIMIZED PARAMETERS ===")
    best_params = {
        "n_estimators": 400,
        "max_depth": 12,
        "min_samples_split": 5,
        "min_samples_leaf": 4
    }
    print(f"Parameters: {best_params}")
    rf = RandomForestRegressor(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

model_path = Path("rf_orders_model.pkl")
joblib.dump(rf, model_path)
print(f"\nModel saved to {model_path.absolute()}")

y_train_pred = rf.predict(X_train)
y_pred = rf.predict(X_test)

train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_pred)

print("\n=== MODEL RESULTS ===")
print(f"\nTraining Set:")
print(f"  MAE:  {train_mae:.2f}")
print(f"  RMSE: {train_rmse:.2f}")
print(f"  R²:   {train_r2:.4f}")

print(f"\nTest Set:")
print(f"  MAE:  {test_mae:.2f}")
print(f"  RMSE: {test_rmse:.2f}")
print(f"  R²:   {test_r2:.4f}")

print(f"\nOverfitting Analysis:")
print(f"  R² gap: {train_r2 - test_r2:.4f}")
print(f"  Test MAE is {test_mae/train_mae:.2f}x higher than train")

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

print("\n=== TOP 10 FEATURES ===")
for i in range(min(10, len(features))):
    print(f"  {i+1}. {features[indices[i]]}: {importances[indices[i]]:.4f}")

plt.figure(figsize=(10,6))
plt.title("Feature Importance - Random Forest")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), np.array(features)[indices], rotation=45, ha='right')
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(14,6))
plt.plot(y_test.values[:200], label="Actual quantity", linewidth=2.5, color="#2E86AB", alpha=0.8)
plt.plot(y_pred[:200], label="Predicted quantity", linewidth=2.5, color="#A23B72", alpha=0.8)
plt.legend(fontsize=12, loc='best')
plt.title("Actual vs Predicted Hourly Quantity (Random Forest)", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Time index", fontsize=12, fontweight="bold")
plt.ylabel("Total Quantity", fontsize=12, fontweight="bold")
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig("predictions_comparison.png", dpi=150, bbox_inches="tight")
print(f"✓ Saved predictions comparison plot")
plt.close()
