import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# === 1. Läs in datan ===
df = pd.read_excel("Data/hourly_orders_with_weather.xlsx")

# === 2. Skapa tidsfeatures ===
df['hour'] = pd.to_datetime(df['hour'])
df['hour_of_day'] = df['hour'].dt.hour
df['day_of_week'] = df['hour'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)

# === 3. (Valfritt) Lagg/rolling för historik ===
df = df.sort_values("hour")
df["orders_lag_1h"] = df["orders"].shift(1)
df["orders_mean_24h"] = df["orders"].rolling(24).mean()
df = df.dropna(subset=["orders_lag_1h","orders_mean_24h"])

# === 4. Definiera features och mål ===
features = [
    "temperature_C", "rain_mm", "cloud_cover_pct", "wind_speed_kmh",
    "hour_of_day", "day_of_week", "is_weekend", "orders_lag_1h", "orders_mean_24h"
]
X = df[features]
y = df["orders"]

# === 5. Dela upp data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# === 6. Visa datasetinfo (som i exemplet du visade) ===
print("=== DATA OVERVIEW ===")
print("Antal träningsrader:", len(X_train))
print("Antal testrader:", len(X_test))
print("\nFörsta 5 rader i träningsdatan:")
print(X_train.head(5))
print("\nFörsta 5 rader i testdatan:")
print(X_test.head(5))

# === 7. Träna Random Forest ===
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# === 8. Gör prognos och utvärdera ===
y_pred = rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n=== MODELLRESULTAT ===")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# === 9. Feature importance ===
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(8,4))
plt.title("Feature Importance - Random Forest")
plt.bar(range(len(importances)), importances[indices], align="center")
plt.xticks(range(len(importances)), np.array(features)[indices], rotation=45)
plt.tight_layout()
plt.show()

# === 10. Faktiska vs predikterade värden (visualisering) ===
plt.figure(figsize=(12,5))
plt.plot(y_test.values[:200], label="Faktiska ordrar", linewidth=2)
plt.plot(y_pred[:200], label="Predikterade ordrar", linewidth=2)
plt.legend()
plt.title("Faktisk vs Predikterad timförsäljning (Random Forest)")
plt.xlabel("Tidsindex")
plt.ylabel("Antal ordrar")
plt.tight_layout()
plt.show()
