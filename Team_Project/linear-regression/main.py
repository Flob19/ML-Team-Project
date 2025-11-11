import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# -------------------
# Paths and settings
# -------------------

initial_dataroot = "../Data/sales_with_weather_tx.xlsx"
training_dataroot = "../Data/sales_with_weather_tx_train.csv"
testing_dataroot = "../Data/sales_with_weather_tx_test.csv"

train_pred_path = "../Data/hourly_demand_predictions_train.csv"
test_pred_path = "../Data/hourly_demand_predictions_test.csv"
all_pred_path = "../Data/hourly_demand_predictions_all.csv"

time_col = "datetime"
category_col = "product_category"

weather_cols = [
    "temperature_C",
    "rain_mm",
    "snow_cm",
    "cloud_cover_pct",
    "wind_speed_kmh",
]

test_size = 0.2
random_state = 42

# -------------------
# Helper functions
# -------------------


def to_float(series):
    """Convert strings with commas as decimal separators to float."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return (
        series.astype(str)
        .str.replace(",", ".", regex=False)
        .str.replace(" ", "", regex=False)
        .replace("nan", np.nan)
        .astype(float)
    )


def ensure_weather_numeric(df, weather_columns):
    for col in weather_columns:
        if col not in df.columns:
            raise KeyError(f"Missing weather column: {col}")
        df[col] = to_float(df[col])
    return df


def aggregate_hourly_counts(df, time_column, category_column, weather_columns):
    """
    Returns:
      X_features: per-hour weather + time features.
      Y_counts: per-hour counts for each product_category.
      index: hour_floor (timestamp)
    """
    # Ensure datetime
    df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
    df = df.dropna(subset=[time_column])

    # Floor to hour
    df["hour_floor"] = df[time_column].dt.floor("h")

    # Time features
    df["hour"] = df[time_column].dt.hour
    df["dayofweek"] = df[time_column].dt.dayofweek
    df["month"] = df[time_column].dt.month

    # Weather to numeric
    df = ensure_weather_numeric(df, weather_columns)

    # Group by hour + category: get counts and mean weather
    grouped = (
        df.groupby(["hour_floor", category_column])
        .agg(
            count=("product_id", "count"),
            temperature_C=("temperature_C", "mean"),
            rain_mm=("rain_mm", "mean"),
            snow_cm=("snow_cm", "mean"),
            cloud_cover_pct=("cloud_cover_pct", "mean"),
            wind_speed_kmh=("wind_speed_kmh", "mean"),
            hour=("hour", "first"),
            dayofweek=("dayofweek", "first"),
            month=("month", "first"),
        )
        .reset_index()
    )

    # Pivot counts so each category is one target column
    pivot_counts = grouped.pivot(
        index="hour_floor", columns=category_column, values="count"
    ).fillna(0)

    # Features per hour (shared across categories)
    features = (
        grouped.groupby("hour_floor")
        .agg(
            temperature_C=("temperature_C", "mean"),
            rain_mm=("rain_mm", "mean"),
            snow_cm=("snow_cm", "mean"),
            cloud_cover_pct=("cloud_cover_pct", "mean"),
            wind_speed_kmh=("wind_speed_kmh", "mean"),
            hour=("hour", "first"),
            dayofweek=("dayofweek", "first"),
            month=("month", "first"),
        )
        .reindex(pivot_counts.index)
    )

    # Sort indexes to align
    features = features.sort_index()
    pivot_counts = pivot_counts.sort_index()

    return features, pivot_counts


# -------------------
# Load and split data
# -------------------

if not (os.path.exists(training_dataroot) and os.path.exists(testing_dataroot)):
    df_full = pd.read_excel(initial_dataroot, sheet_name=0)

    # Drop columns not needed for demand modeling
    drop_cols = [
        "transaction_id",
        "transaction_date",
        "transaction_time",
        "store_id",
        "store_location",
        "unit_price",
    ]
    df_full = df_full.drop(columns=drop_cols)

    train_df, test_df = train_test_split(
        df_full,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    train_df.to_csv(training_dataroot, index=False)
    test_df.to_csv(testing_dataroot, index=False)
else:
    train_df = pd.read_csv(training_dataroot)
    test_df = pd.read_csv(testing_dataroot)

# -------------------
# Build hourly demand dataset (train/test)
# -------------------

X_train_full, Y_train_full = aggregate_hourly_counts(
    train_df, time_col, category_col, weather_cols
)
X_test_full, Y_test_full = aggregate_hourly_counts(
    test_df, time_col, category_col, weather_cols
)

# Ensure same target columns (categories) in both sets
all_categories = sorted(
    set(Y_train_full.columns).union(set(Y_test_full.columns))
)
Y_train_full = Y_train_full.reindex(columns=all_categories, fill_value=0)
Y_test_full = Y_test_full.reindex(columns=all_categories, fill_value=0)

feature_cols = [
    "temperature_C",
    "rain_mm",
    "snow_cm",
    "cloud_cover_pct",
    "wind_speed_kmh",
    "hour",
    "dayofweek",
    "month",
]

X_train = X_train_full[feature_cols]
X_test = X_test_full[feature_cols]

# -------------------
# Define and train model (Linear Regression)
# -------------------

model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("regressor", MultiOutputRegressor(LinearRegression())),
    ]
)

model.fit(X_train, Y_train_full)

# -------------------
# Evaluation (printed)
# -------------------

Y_pred_test = model.predict(X_test)

print("Evaluation on hourly demand (per category) - TEST:")
for i, cat in enumerate(all_categories):
    y_true_cat = Y_test_full.iloc[:, i]
    y_pred_cat = Y_pred_test[:, i]

    mae = mean_absolute_error(y_true_cat, y_pred_cat)
    r2 = r2_score(y_true_cat, y_pred_cat)

    print(f"- {cat}: MAE={mae:.3f}, R2={r2:.3f}")

# -------------------
# Build prediction DataFrames and save to CSV
# -------------------

# Helper: clip negatives (linear regression can produce <0)
def clip_predictions(pred_array):
    return np.maximum(pred_array, 0)


# 1) Predictions for TRAIN hours
Y_pred_train = clip_predictions(model.predict(X_train))

train_pred_df = X_train_full.copy()
train_pred_df.index.name = "hour_floor"
for i, cat in enumerate(all_categories):
    train_pred_df[f"pred_{cat}"] = Y_pred_train[:, i]

# 2) Predictions for TEST hours
Y_pred_test = clip_predictions(Y_pred_test)

test_pred_df = X_test_full.copy()
test_pred_df.index.name = "hour_floor"
for i, cat in enumerate(all_categories):
    test_pred_df[f"pred_{cat}"] = Y_pred_test[:, i]

# 3) Combined (TRAIN + TEST) for full analysis
all_pred_df = (
    pd.concat(
        [
            train_pred_df.assign(split="train"),
            test_pred_df.assign(split="test"),
        ]
    )
    .sort_index()
)

# Save to CSV
os.makedirs(os.path.dirname(train_pred_path), exist_ok=True)
train_pred_df.to_csv(train_pred_path)
test_pred_df.to_csv(test_pred_path)
all_pred_df.to_csv(all_pred_path)

print(f"\nSaved train predictions to: {train_pred_path}")
print(f"Saved test predictions to:  {test_pred_path}")
print(f"Saved all predictions to:   {all_pred_path}")


# -------------------
# Optional: function for ad-hoc prediction
# -------------------


def predict_hourly_demand(
    temperature_C,
    rain_mm,
    snow_cm,
    cloud_cover_pct,
    wind_speed_kmh,
    dt,
):
    """
    Predict how many drinks (per category) will be sold in the given hour
    for specified weather and datetime.
    """
    dt = pd.to_datetime(dt)

    row = pd.DataFrame(
        [
            {
                "temperature_C": float(temperature_C),
                "rain_mm": float(rain_mm),
                "snow_cm": float(snow_cm),
                "cloud_cover_pct": float(cloud_cover_pct),
                "wind_speed_kmh": float(wind_speed_kmh),
                "hour": dt.hour,
                "dayofweek": dt.dayofweek,
                "month": dt.month,
            }
        ]
    )

    preds = clip_predictions(model.predict(row)[0])
    return dict(zip(all_categories, preds))


if __name__ == "__main__":
    example = predict_hourly_demand(
        temperature_C=25.0,
        rain_mm=20.0,
        snow_cm=0.0,
        cloud_cover_pct=0.0,
        wind_speed_kmh=0.0,
        dt="2028-01-05 09:00:00",
    )
    print("\nExample prediction for given weather:")
    for cat, val in example.items():
        print(f"{cat}: {val:.2f}")