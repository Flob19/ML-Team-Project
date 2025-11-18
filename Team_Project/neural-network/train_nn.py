# ==========================
# ☕ COFFEE PRICE PREDICTION
# Neural Network with Weather + Sales Features
# ==========================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from datetime import datetime

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
import joblib

# ==============================
# 1️⃣ --- Setup paths and folders
# ==============================
DATA = Path("Data")
RESULTS_DIR = Path("results") / datetime.now().strftime("%Y-%m-%d_%H-%M")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ==============================
# 2️⃣ --- Load Data
# ==============================
df = pd.read_excel(DATA / "sales_with_weather_tx.xlsx")
target = "unit_price"

X = df.drop(columns=[
    target,
    "transaction_id",
    "datetime",
    "hour",
    "transaction_date",
    "transaction_time",
    "product_detail"
])
y = df[target]

# ==============================
# 3️⃣ --- Identify feature types
# ==============================
categorical = X.select_dtypes(include=["object"]).columns.tolist()
numeric = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

print("📋 Categorical features:", categorical)
print("📊 Numeric features:", numeric)

# ==============================
# 4️⃣ --- Preprocessing
# ==============================
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

# Split before fitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

joblib.dump(preprocessor, RESULTS_DIR / "preprocessor.pkl")

# ==============================
# 5️⃣ --- Build Neural Network
# ==============================
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_processed.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# ==============================
# 6️⃣ --- Train Model
# ==============================
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train_processed, y_train,
    validation_data=(X_test_processed, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# ==============================
# 7️⃣ --- Evaluate
# ==============================
loss, mae = model.evaluate(X_test_processed, y_test)
print(f"\n✅ Test MAE: {mae:.3f}")

model.save(RESULTS_DIR / "coffee_price_nn.keras")
print(f"Model saved in {RESULTS_DIR}")

with open(RESULTS_DIR / "metrics.csv", "w") as f:
    f.write(f"Test_MAE,{mae:.4f}\n")


# ==========================
# ⚖️ MODEL COMPARISON — XGBoost Benchmark
# ==========================
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

print("\n🚀 Training XGBoost model for comparison...")

# Ensure dense arrays (XGBoost doesn’t accept sparse matrices)
X_train_dense = X_train_processed.toarray() if hasattr(X_train_processed, "toarray") else X_train_processed
X_test_dense = X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed

# Define XGBoost model
xgb = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train the model
xgb.fit(X_train_dense, y_train)

# Evaluate
y_pred_xgb = xgb.predict(X_test_dense)
xgb_mae = mean_absolute_error(y_test, y_pred_xgb)

print(f"✅ XGBoost Test MAE: {xgb_mae:.3f}")
print(f"📊 NeuralNet MAE: {mae:.3f} | XGBoost MAE: {xgb_mae:.3f}")

# Save predictions for further analysis
results_df = pd.DataFrame({
    "actual_price": y_test,
    "nn_pred": model.predict(X_test_processed).flatten(),
    "xgb_pred": y_pred_xgb
})
results_df.to_csv("results_model_comparison.csv", index=False)
print("🗂 Results saved as results_model_comparison.csv")

# Optional — quick visualization
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4))
plt.bar(["NeuralNet", "XGBoost"], [mae, xgb_mae], color=["skyblue", "salmon"])
plt.title("Model Comparison: Mean Absolute Error (MAE)")
plt.ylabel("MAE (lower is better)")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()


# ==============================
# 8️⃣ --- Visualizations
# ==============================
def savefig(name):
    plt.savefig(RESULTS_DIR / name, bbox_inches="tight")
    plt.close()

## 📈 Loss curves
plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Training Loss", linewidth=2)
plt.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
savefig("training_loss.png")

## 📊 MAE curves
plt.figure(figsize=(10, 5))
plt.plot(history.history["mae"], label="Training MAE", linewidth=2)
plt.plot(history.history["val_mae"], label="Validation MAE", linewidth=2)
plt.xlabel("Epochs")
plt.ylabel("Mean Absolute Error")
plt.title("Training vs Validation MAE")
plt.legend()
plt.grid(True)
savefig("training_mae.png")

# ==============================
# 9️⃣ --- Actual vs Predicted
# ==============================
y_pred = model.predict(X_test_processed).flatten()
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Coffee Prices")
plt.grid(True)
savefig("actual_vs_predicted.png")

# ==============================
# 🔟 --- Feature info
# ==============================
print("\n=== Feature Overview ===")
print("Categorical features:", categorical)
print("Numeric features:", numeric)
print(f"Total encoded feature count: {X_train_processed.shape[1]}")
print(f"\nPredicted label: {target}")
print(f"Target value range: {y.min():.2f} → {y.max():.2f}")

cat_encoder = preprocessor.named_transformers_["cat"]
num_features = numeric
cat_features = cat_encoder.get_feature_names_out(categorical)
all_features = np.concatenate([num_features, cat_features])
print(f"\n=== Final Feature Names ({len(all_features)} total) ===")
print(all_features)

# ==============================
# 1️⃣1️⃣ --- Correlation heatmap
# ==============================
corr_data = df[numeric + [target]].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_data, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation between Numeric Features and Target")
savefig("correlation_heatmap.png")

# ==============================
# 1️⃣2️⃣ --- Permutation Importance
# ==============================
X_dense = X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed

try:
    baseline = LinearRegression()
    baseline.fit(X_train_processed, y_train)
    result = permutation_importance(
        baseline,
        X_dense,
        y_test,
        n_repeats=10,
        random_state=42
    )
    importance_df = pd.DataFrame({
        "feature": all_features,
        "importance": result.importances_mean
    }).sort_values("importance", ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df.head(15), x="importance", y="feature", palette="viridis")
    plt.title("Top 15 Feature Importances (approximate)")
    savefig("feature_importance.png")

    importance_df.head(15).to_csv(RESULTS_DIR / "top_features.csv", index=False)
    print(f"Top features saved to {RESULTS_DIR}/top_features.csv")

except Exception as e:
    print("⚠️ Permutation importance skipped:", e)
# ==============================
# 1️⃣3️⃣ --- Explainability with SHAP (stable version)
# ==============================
import shap
import numpy as np

# Convert to dense array for SHAP
X_dense = X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed

# Select a small representative sample for explainability
X_sample = X_dense[:200]
print(f"Computing SHAP values for {X_sample.shape[0]} samples using KernelExplainer...")

# SHAP requires a prediction function
f = lambda x: model.predict(x).flatten()

# Use a small background sample to speed up
background = X_dense[np.random.choice(X_dense.shape[0], 100, replace=False)]

explainer = shap.KernelExplainer(f, background)
shap_values = explainer.shap_values(X_sample, nsamples=100)

# --- Global importance summary plot ---
shap.summary_plot(shap_values, X_sample, feature_names=all_features, show=False)
savefig("shap_summary_plot.png")

# --- Feature importance bar plot ---
shap.summary_plot(shap_values, X_sample, feature_names=all_features, plot_type="bar", show=False)
savefig("shap_feature_importance_bar.png")

# --- Save feature importance data ---
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_df = pd.DataFrame({
    "feature": all_features,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False)
shap_df.to_csv(RESULTS_DIR / "shap_top_features.csv", index=False)

print("✅ SHAP plots and top feature importances saved in", RESULTS_DIR)
