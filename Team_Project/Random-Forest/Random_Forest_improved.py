import pandas as pd
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
import joblib
from pathlib import Path

data_path = Path("../Data/hourly_orders_with_weather.xlsx")
if not data_path.exists():
    data_path = Path("Data/hourly_orders_with_weather.xlsx")
df = pd.read_excel(data_path)

df['hour'] = pd.to_datetime(df['hour'])
df = df.sort_values("hour")

df['hour_of_day'] = df['hour'].dt.hour
df['day_of_week'] = df['hour'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)

df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

df['is_morning'] = df['hour_of_day'].isin([7,8,9,10,11]).astype(int)
df['is_afternoon'] = df['hour_of_day'].isin([12,13,14,15,16,17]).astype(int)
df['is_evening'] = df['hour_of_day'].isin([18,19,20,21,22]).astype(int)
df['is_night'] = df['hour_of_day'].isin([0,1,2,3,4,5,6]).astype(int)

df["orders_lag_1h"] = df["orders"].shift(1)
df["orders_lag_24h"] = df["orders"].shift(24)
df["orders_mean_24h"] = df["orders"].rolling(24).mean()
df["orders_std_24h"] = df["orders"].rolling(24).std()
df["orders_mean_7d"] = df["orders"].rolling(168).mean()

df = df.dropna(subset=["orders_lag_1h", "orders_mean_24h"])

features = [
    "temperature_C", "rain_mm", "cloud_cover_pct", "wind_speed_kmh",
    "hour_of_day", "day_of_week", "is_weekend",
    "hour_sin", "hour_cos", "day_sin", "day_cos",
    "is_morning", "is_afternoon", "is_evening", "is_night",
    "orders_lag_1h", "orders_lag_24h", "orders_mean_24h", "orders_std_24h", "orders_mean_7d"
]

X = df[features]
y = df["orders"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=0)

print("=== DATA OVERVIEW ===")
print("Training rows:", len(X_train))
print("Test rows:", len(X_test))
print(f"Features: {len(features)}")

print("\n=== HYPERPARAMETER TUNING ===")
tscv = TimeSeriesSplit(n_splits=3)
param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [8, 10, 12],
    'min_samples_split': [10, 15],
    'min_samples_leaf': [4, 5]
}

rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    rf_base, param_grid, cv=tscv, 
    scoring='neg_mean_absolute_error',
    n_jobs=-1, verbose=1
)

print("Running grid search with time-series cross-validation...")
grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score (MAE): {-grid_search.best_score_:.2f}")

rf = grid_search.best_estimator_

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
plt.title("Feature Importance - Random Forest (Improved)")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), np.array(features)[indices], rotation=45, ha='right')
plt.tight_layout()
plt.savefig("feature_importance_improved.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(12,5))
plt.plot(y_test.values[:200], label="Actual orders", linewidth=2)
plt.plot(y_pred[:200], label="Predicted orders", linewidth=2)
plt.legend()
plt.title("Actual vs Predicted Hourly Sales (Improved Model)")
plt.xlabel("Time index")
plt.ylabel("Number of orders")
plt.tight_layout()
plt.savefig("predictions_improved.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n=== COMPLETE ===")

