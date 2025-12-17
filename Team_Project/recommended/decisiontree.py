import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

# 1. Load data
print("Loading and preparing data...")
# training_data = pd.read_csv("../Data/sales_with_weather_tx_train.csv")
training_data = pd.read_excel("../Data/sales_with_weather_tx.xlsx")

# Drop ID columns that aren't features
drop_cols = ["product_id", "product_type", "transaction_id", "transaction_date", "transaction_time", "datetime"]
for col in drop_cols:
    if col in training_data.columns:
        training_data = training_data.drop(columns=[col])

# Convert hour to numeric (hour of day) if it's a timestamp
if "hour" in training_data.columns:
    if pd.api.types.is_datetime64_any_dtype(training_data["hour"]):
        training_data["hour"] = training_data["hour"].dt.hour
    else:
        # If it's string or something else, try to convert to datetime first then extract hour, or just leave it if it's already int
        try:
            training_data["hour"] = pd.to_datetime(training_data["hour"]).dt.hour
        except:
            pass # Assume it's already numeric or categorical

# 2. Prepare X and y
y_train = training_data["product_detail"]
X_train = training_data.drop(columns=["product_detail"])

print(f"Data shape: {X_train.shape}, Labels shape: {y_train.shape}")
print("Columns:", X_train.columns.tolist())

# 3. Define Preprocessing Pipeline
print("Setting up preprocessing pipeline...")
onehot_cols = X_train.select_dtypes(include=['object']).columns
numeric_cols = X_train.select_dtypes(exclude=['object']).columns

# Define transformers
numeric_transformer = "passthrough"
categorical_transformer = OneHotEncoder(sparse_output=True, handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, onehot_cols)
    ]
)

# 4. Split Data
print("Performing random train/test split...")
# Note: We split the original DataFrame, not the preprocessed matrix.
# The pipeline will handle preprocessing.
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=0)

print(f"Train rows: {X_tr.shape[0]} | Val rows: {X_val.shape[0]}")

# 5. Setup and Train Decision Tree Model
print("Setting up and training Decision Tree model...")

dt_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler(with_mean=False)),
    ("decisiontree", DecisionTreeClassifier(max_depth=20, min_samples_leaf=10))
])

# Train model
dt_pipeline.fit(X_tr, y_tr)
print("Model trained.")

# Save model
model_filename = "decision_tree_model.pkl"
joblib.dump(dt_pipeline, model_filename)
print(f"Model saved to {model_filename}")

# 6. Validate Model
print("Evaluating model...")
train_accuracy = dt_pipeline.score(X_tr, y_tr)
print(f"Training Accuracy: {train_accuracy:.4f}")

val_accuracy = dt_pipeline.score(X_val, y_val)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# 7. Random Sample Inspection
print("\n--- Random Sample Predictions from Validation Set ---")
sample_size = 10
n_val = X_val.shape[0]
if n_val < sample_size:
    sample_size = n_val

random_indices = np.random.choice(n_val, size=sample_size, replace=False)

# Get the samples
X_sample = X_val.iloc[random_indices]
y_sample_actual = y_val.iloc[random_indices].values

# Predict
y_sample_pred = dt_pipeline.predict(X_sample)

# Display results
results_df = pd.DataFrame({
    'Actual': y_sample_actual,
    'Predicted': y_sample_pred,
    'Match': y_sample_actual == y_sample_pred
})

print(results_df)

# 8. Feature Importance Graph
print("\nGenerating feature importance graph...")

# Get the model from the pipeline
model = dt_pipeline.named_steps['decisiontree']
importances = model.feature_importances_

# Get feature names from preprocessor
# The preprocessor is a ColumnTransformer. We need to get feature names from its transformers.
# 'num' transformer is 'passthrough', so names are numeric_cols
# 'cat' transformer is OneHotEncoder, so we get names from it.

preprocessor_step = dt_pipeline.named_steps['preprocessor']
cat_encoder = preprocessor_step.named_transformers_['cat']
cat_feature_names = cat_encoder.get_feature_names_out(onehot_cols)

all_feature_names = np.concatenate([numeric_cols, cat_feature_names])

# Create a DataFrame to organize them
feature_imp_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
})

# Sort by importance and select top 20
top_features = feature_imp_df.sort_values(by='Importance', ascending=False).head(20)

# Plot
plt.figure(figsize=(12, 8))
plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Top 20 Most Important Features in Decision Tree')
plt.gca().invert_yaxis()
plt.tight_layout()

output_file = 'feature_importance.png'
plt.savefig(output_file)
print(f"Feature importance graph saved to {output_file}")
